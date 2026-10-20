import argparse
import cv2
import json
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict
import numpy as np
import sys
import os

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

# Import ByteTrack
from Pipeline_STGCN.bytetrack_utils.byte_tracker import BYTETracker
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def run_pose_estimation_on_frame(frame, pose_model, tracker):
    """
    Run pose estimation on a single frame using the provided YOLOv11 pose model from ultralytics.
    Returns a list of joint points (keypoints) for each detected person, where each joint is [x, y, confidence].
    Also returns tracked IDs for each person.
    """
    results = pose_model.predict(frame)
    joint_points = []
    tracked_ids = []

    # Convert pose results to bounding boxes for tracking
    detections = []
    for result in results:
        # Prefer using YOLO's native detection boxes if available
        if hasattr(result, "boxes") and hasattr(result.boxes, "xyxy"):
            boxes = result.boxes.xyxy
            scores = result.boxes.conf if hasattr(result.boxes, "conf") else None
            if hasattr(boxes, "cpu"):
                boxes = boxes.cpu().numpy()
            if scores is not None and hasattr(scores, "cpu"):
                scores = scores.cpu().numpy()
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(float, box)
                score = float(scores[i]) if scores is not None else 1.0
                detections.append([x1, y1, x2, y2, score])
        # Fallback: compute from keypoints if boxes not available
        elif hasattr(result, "keypoints") and hasattr(result.keypoints, "xy"):
            keypoints_xy = result.keypoints.xy  # shape: (num_people, num_joints, 2)
            keypoints_conf = getattr(result.keypoints, "conf", None)  # shape: (num_people, num_joints)

            # Get bounding boxes from keypoints
            for person_xy in keypoints_xy:
                # Convert to numpy array if it's a tensor
                if hasattr(person_xy, 'cpu'):
                    person_xy = person_xy.cpu().numpy()

                # Skip if no keypoints
                if len(person_xy) == 0:
                    continue

                x_coords = person_xy[:, 0]
                y_coords = person_xy[:, 1]

                # Skip if coordinates are empty
                if len(x_coords) == 0 or len(y_coords) == 0:
                    continue

                x1, y1 = float(np.min(x_coords)), float(np.min(y_coords))
                x2, y2 = float(np.max(x_coords)), float(np.max(y_coords))
                detections.append([x1, y1, x2, y2, 1.0])  # [x1, y1, x2, y2, score]

    logger.debug(f"Generated {len(detections)} bounding boxes from keypoints")

    # Update tracker with detections
    if len(detections) > 0:
        detections = np.array(detections)
        online_targets = tracker.update(detections, [frame.shape[0], frame.shape[1]], frame.shape[:2])

        # Map tracked IDs to detections
        id_to_detection = {t.track_id: i for i, t in enumerate(online_targets)}
        logger.debug(f"Tracking {len(online_targets)} targets with IDs: {list(id_to_detection.keys())}")
    else:
        online_targets = []
        id_to_detection = {}
        logger.debug("No detections to track in this frame")

    # Process keypoints with tracked IDs
    for i, result in enumerate(results):
        if hasattr(result, "keypoints") and hasattr(result.keypoints, "xy"):
            keypoints_xy = result.keypoints.xy
            keypoints_conf = getattr(result.keypoints, "conf", None)

            if keypoints_conf is not None:
                for person_idx, (person_xy, person_conf) in enumerate(zip(keypoints_xy, keypoints_conf)):
                    # Skip if no keypoints
                    if len(person_xy) == 0:
                        continue

                    joints = []
                    # Convert to numpy if tensors
                    if hasattr(person_xy, 'cpu'):
                        person_xy = person_xy.cpu().numpy()
                    if hasattr(person_conf, 'cpu'):
                        person_conf = person_conf.cpu().numpy()

                    for (xy, conf) in zip(person_xy, person_conf):
                        joints.append([float(xy[0]), float(xy[1]), float(conf)])
                    joint_points.append(joints)

                    # Get tracked ID if available
                    if person_idx in id_to_detection.values():
                        track_id = [k for k, v in id_to_detection.items() if v == person_idx][0]
                        tracked_ids.append(track_id)
                        logger.debug(f"Assigned track ID {track_id} to person {person_idx}")
                    else:
                        tracked_ids.append(-1)
                        logger.debug(f"No track ID available for person {person_idx}")
            else:
                for person in keypoints_xy:
                    # Skip if no keypoints
                    if len(person) == 0:
                        continue

                    # Convert to numpy if tensor
                    if hasattr(person, 'cpu'):
                        person = person.cpu().numpy()
                    joints = [[float(xy[0]), float(xy[1]), 1.0] for xy in person]
                    joint_points.append(joints)
                    tracked_ids.append(-1)

    return joint_points, tracked_ids

def process_video_and_json(video_path, json_path, pose_model, output_json_path=None):
    # Reset global track ID counter for each new video
    from Pipeline_STGCN.bytetrack_utils.basetrack import BaseTrack
    BaseTrack._count = 0

    logger.info(f"Processing video: {video_path}")
    logger.info(f"Input JSON: {json_path}")
    if output_json_path:
        logger.info(f"Output JSON: {output_json_path}")

    # Load JSON data
    with open(json_path, "r") as f:
        frame_data = json.load(f)
    logger.info(f"Loaded {len(frame_data)} frames from JSON")

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Video dimensions: {width}x{height}")

    # Initialize tracker
    tracker = BYTETracker(track_thresh=0.5, track_buffer=30, match_thresh=0.8)
    logger.info("Initialized ByteTrack with parameters:")
    logger.info(f"- Track threshold: 0.5")
    logger.info(f"- Track buffer: 30")
    logger.info(f"- Match threshold: 0.8")

    # Track ID frequencies
    id_frequencies = defaultdict(int)
    frame_idx = 0
    updated_data = []

    for entry in frame_data:
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"Not enough frames in {video_path.name} for JSON entries.")
            break

        # Run pose estimation with tracking
        joints, tracked_ids = run_pose_estimation_on_frame(frame, pose_model, tracker)

        # Update ID frequencies
        for track_id in tracked_ids:
            if track_id != -1:
                id_frequencies[track_id] += 1

        # Create new entry without frame-level label
        new_entry = {
            "frame_number": entry.get("frame_number", frame_idx),
            "width": width,
            "height": height,
            "label": entry.get("label", "non-cleaning")
        }

        # Create new joints structure with IDs
        new_entry["joints"] = []
        for joint_set, track_id in zip(joints, tracked_ids):
            new_entry["joints"].append({
                "id": int(track_id),
                "keypoints": joint_set
            })

        updated_data.append(new_entry)
        frame_idx += 1
        if frame_idx % 10 == 0:
            logger.info(f"Processed {frame_idx}/{len(frame_data)} frames for {video_path.name}")
            logger.debug(f"Current ID frequencies: {dict(id_frequencies)}")

    cap.release()

    # Find most frequent ID (cleaner)
    if id_frequencies:
        cleaner_id = max(id_frequencies.items(), key=lambda x: x[1])[0]
        logger.info(f"Identified cleaner with ID {cleaner_id} (appeared in {id_frequencies[cleaner_id]} frames)")
    else:
        cleaner_id = -1
        logger.warning("No tracked IDs found in the video")

    # Update pose-level labels based on frame labels and tracked IDs
    cleaning_frames = 0
    non_cleaning_frames = 0
    for entry, original_entry in zip(updated_data, frame_data):
        frame_label = original_entry.get("label", "non-cleaning")
        if frame_label == "cleaning":
            cleaning_frames += 1
        else:
            non_cleaning_frames += 1

        ids_in_frame = [joint_data["id"] for joint_data in entry["joints"]]
        logger.info(f"cleaner id: {cleaner_id}")
        logger.info(f"other ids: {ids_in_frame}")
        logger.info(f"label: {frame_label}")

        for joint_data in entry["joints"]:
            track_id = joint_data["id"]
            # Ensure both are int for comparison
            try:
                track_id_int = int(track_id)
                cleaner_id_int = int(cleaner_id)
            except Exception:
                track_id_int = track_id
                cleaner_id_int = cleaner_id

            if frame_label.lower() == "cleaning" and track_id_int == cleaner_id_int:
                joint_data["pose_label"] = "cleaning"
                logger.info(f"assigning label: cleaning to id {track_id}")
            else:
                joint_data["pose_label"] = "non-cleaner"
                logger.info(f"assigning label: non-cleaner to id {track_id}")

    logger.info(f"Frame statistics:")
    logger.info(f"- Cleaning frames: {cleaning_frames}")
    logger.info(f"- Non-cleaning frames: {non_cleaning_frames}")

    # Save updated JSON
    if output_json_path is None:
        output_json_path = json_path
    with open(output_json_path, "w") as f:
        json.dump(updated_data, f, indent=2)
    logger.info(f"Saved updated JSON with joints to {output_json_path}")

def main():
    parser = argparse.ArgumentParser(description="Add pose joint points to video frame JSONs using YOLOv11 pose")
    parser.add_argument('--video-dir', type=str, required=True, help='Directory containing .mp4 videos and corresponding .json files')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory to save updated JSONs (default: overwrite originals)')
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None

    # Load YOLOv11 pose model (always use 'yolov11l-pose' from ultralytics)
    pose_model = YOLO('yolo11l-pose')

    # Process each video/json pair
    for video_file in video_dir.glob("*.mp4"):
        json_file = video_file.with_suffix(".json")
        if not json_file.exists():
            print(f"Skipping {video_file.name}: no matching JSON found.")
            continue
        print(f"Processing {video_file.name} and {json_file.name}...")
        output_json = (output_dir / json_file.name) if output_dir else None
        process_video_and_json(video_file, json_file, pose_model, output_json)

if __name__ == "__main__":
    main()
