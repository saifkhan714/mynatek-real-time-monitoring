# video_to_json_with_tracking.py
import argparse
import cv2
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict, deque
import math

# ByteTracker implementation (simplified but effective)
class STrack:
    def __init__(self, bbox, score, track_id):
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.score = score
        self.kalman_filter = self._init_kalman()
        self.state = 'tracked'
        self.frame_id = 0
        self.tracklet_len = 0
        self.is_activated = False
        self.time_since_update = 0
        
        # For pose consistency
        self.pose_history = deque(maxlen=5)  # Keep last 5 poses for smoothing
        self.last_pose = None
        
    def _init_kalman(self):
        """Initialize Kalman filter for bbox tracking"""
        # Simple constant velocity model
        from scipy.optimize import linear_sum_assignment
        # For simplicity, we'll use a basic tracking without full Kalman
        return None
    
    def update(self, bbox, score, pose=None):
        """Update track with new detection"""
        self.bbox = bbox
        self.score = score
        self.time_since_update = 0
        self.tracklet_len += 1
        
        if pose is not None:
            self.pose_history.append(pose)
            self.last_pose = pose
    
    def predict(self):
        """Predict next state (simplified)"""
        self.time_since_update += 1
        return self.bbox
    
    def get_smoothed_pose(self):
        """Get temporally smoothed pose"""
        if len(self.pose_history) == 0:
            return self.last_pose
        
        if len(self.pose_history) == 1:
            return self.pose_history[0]
        
        # Simple moving average smoothing
        poses = np.array(self.pose_history)
        weights = np.linspace(0.5, 1.0, len(poses))  # Recent poses have higher weight
        weights = weights / weights.sum()
        
        smoothed = np.average(poses, axis=0, weights=weights)
        return smoothed

class ByteTracker:
    def __init__(self, frame_rate=30, track_thresh=0.6, track_buffer=30, match_thresh=0.8):
        self.frame_rate = frame_rate
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []     # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        
        self.frame_id = 0
        self.track_id_count = 0
        
    def update(self, detections):
        """
        Update tracker with new detections
        detections: list of dict with keys: 'bbox', 'score', 'pose'
        """
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        
        # Separate high and low confidence detections
        remain_inds = [i for i, det in enumerate(detections) if det['score'] >= self.track_thresh]
        
        detections_high = [detections[i] for i in remain_inds]
        
        # Predict current locations of existing tracks
        for strack in self.tracked_stracks:
            strack.predict()
        
        # Association: match detections to existing tracks
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(
            detections_high, self.tracked_stracks
        )
        
        # Update matched tracks
        for m in matched:
            track = self.tracked_stracks[m[1]]
            det = detections_high[m[0]]
            track.update(det['bbox'], det['score'], det['pose'])
            activated_starcks.append(track)
        
        # Handle unmatched detections (new tracks)
        for i in unmatched_dets:
            det = detections_high[i]
            if det['score'] >= self.track_thresh:
                track = STrack(det['bbox'], det['score'], self._next_id())
                track.update(det['bbox'], det['score'], det['pose'])
                track.is_activated = True
                activated_starcks.append(track)
        
        # Handle unmatched tracks (lost tracks)
        for i in unmatched_trks:
            track = self.tracked_stracks[i]
            if track.time_since_update <= self.track_buffer:
                lost_stracks.append(track)
            else:
                removed_stracks.append(track)
        
        # Update track lists
        self.tracked_stracks = activated_starcks
        self.lost_stracks = lost_stracks
        self.removed_stracks.extend(removed_stracks)
        
        # Return active tracks
        return [track for track in self.tracked_stracks if track.is_activated]
    
    def _associate_detections_to_trackers(self, detections, trackers):
        """Associate detections to trackers using IoU and pose similarity"""
        if len(trackers) == 0:
            return [], list(range(len(detections))), []
        
        # Compute cost matrix
        cost_matrix = np.zeros((len(detections), len(trackers)))
        
        for d, det in enumerate(detections):
            for t, track in enumerate(trackers):
                # IoU cost
                iou = self._bbox_iou(det['bbox'], track.bbox)
                
                # Pose similarity cost (if both have poses)
                pose_sim = 0
                if det['pose'] is not None and track.last_pose is not None:
                    pose_sim = self._pose_similarity(det['pose'], track.last_pose)
                
                # Combined cost (lower is better)
                cost_matrix[d, t] = 1.0 - (0.6 * iou + 0.4 * pose_sim)
        
        # Hungarian algorithm for optimal assignment
        try:
            from scipy.optimize import linear_sum_assignment
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            # Filter matches based on threshold
            matches = []
            unmatched_dets = list(range(len(detections)))
            unmatched_trks = list(range(len(trackers)))
            
            for r, c in zip(row_indices, col_indices):
                if cost_matrix[r, c] < (1.0 - self.match_thresh):
                    matches.append([r, c])
                    unmatched_dets.remove(r) if r in unmatched_dets else None
                    unmatched_trks.remove(c) if c in unmatched_trks else None
            
            return matches, unmatched_dets, unmatched_trks
            
        except ImportError:
            print("Warning: scipy not available, using greedy matching")
            return self._greedy_assignment(cost_matrix, detections, trackers)
    
    def _greedy_assignment(self, cost_matrix, detections, trackers):
        """Fallback greedy assignment if scipy not available"""
        matches = []
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(range(len(trackers)))
        
        while len(unmatched_dets) > 0 and len(unmatched_trks) > 0:
            # Find minimum cost
            min_cost = float('inf')
            min_det, min_trk = -1, -1
            
            for d in unmatched_dets:
                for t in unmatched_trks:
                    if cost_matrix[d, t] < min_cost:
                        min_cost = cost_matrix[d, t]
                        min_det, min_trk = d, t
            
            if min_cost < (1.0 - self.match_thresh):
                matches.append([min_det, min_trk])
                unmatched_dets.remove(min_det)
                unmatched_trks.remove(min_trk)
            else:
                break
        
        return matches, unmatched_dets, unmatched_trks
    
    def _bbox_iou(self, bbox1, bbox2):
        """Calculate IoU between two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Calculate intersection
        x_min = max(x1_min, x2_min)
        y_min = max(y1_min, y2_min)
        x_max = min(x1_max, x2_max)
        y_max = min(y1_max, y2_max)
        
        if x_max <= x_min or y_max <= y_min:
            return 0.0
        
        intersection = (x_max - x_min) * (y_max - y_min)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _pose_similarity(self, pose1, pose2, conf_thresh=0.3):
        """Calculate pose similarity between two pose keypoints"""
        if pose1 is None or pose2 is None:
            return 0.0
        
        pose1 = np.array(pose1)
        pose2 = np.array(pose2)
        
        if pose1.shape != pose2.shape:
            return 0.0
        
        # Only consider keypoints with sufficient confidence
        valid_mask = (pose1[:, 2] > conf_thresh) & (pose2[:, 2] > conf_thresh)
        
        if not np.any(valid_mask):
            return 0.0
        
        # Calculate normalized distance
        valid_pose1 = pose1[valid_mask, :2]
        valid_pose2 = pose2[valid_mask, :2]
        
        # Euclidean distance between corresponding keypoints
        distances = np.linalg.norm(valid_pose1 - valid_pose2, axis=1)
        mean_distance = np.mean(distances)
        
        # Convert distance to similarity (0-1, higher is more similar)
        # Assuming max reasonable distance is around 100 pixels in normalized coords
        similarity = max(0, 1.0 - mean_distance / 0.2)
        
        return similarity
    
    def _next_id(self):
        """Get next track ID"""
        self.track_id_count += 1
        return self.track_id_count

def run_pose_estimation_with_tracking(frame, pose_model):
    """
    Run pose estimation on a single frame using YOLO
    Returns detections in format expected by ByteTracker
    """
    results = pose_model.predict(frame, verbose=False)
    detections = []
    
    for result in results:
        if hasattr(result, "boxes") and hasattr(result, "keypoints"):
            boxes = result.boxes
            keypoints = result.keypoints
            
            if boxes is not None and keypoints is not None:
                for box, kpts in zip(boxes, keypoints):
                    # Get bounding box
                    bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                    score = box.conf[0].cpu().numpy()
                    
                    # Get keypoints
                    if hasattr(kpts, 'xy') and hasattr(kpts, 'conf'):
                        kpts_xy = kpts.xy[0].cpu().numpy()  # [num_joints, 2]
                        kpts_conf = kpts.conf[0].cpu().numpy()  # [num_joints]
                        
                        # Combine into [x, y, conf] format
                        pose = np.concatenate([
                            kpts_xy, 
                            kpts_conf.reshape(-1, 1)
                        ], axis=1)
                    else:
                        pose = None
                    
                    detections.append({
                        'bbox': bbox,
                        'score': float(score),
                        'pose': pose
                    })
    
    return detections

def get_video_label(video_path, cleaning_keywords=None):
    """
    Determine if video is cleaning or non-cleaning based on filename or path.
    """
    if cleaning_keywords is None:
        cleaning_keywords = [
            'clean', 'wash', 'mop', 'sweep', 'vacuum', 'wipe', 'scrub', 
            'polish', 'dust', 'brush', 'rinse', 'soap', 'detergent',
            'dishes', 'floor', 'window', 'toilet', 'bathroom', 'kitchen'
        ]
    
    video_name = video_path.name.lower()
    video_parent = str(video_path.parent).lower()
    
    # Check if any cleaning keyword is in the filename or path
    for keyword in cleaning_keywords:
        if keyword in video_name or keyword in video_parent:
            return "Cleaning"
    
    return "Not-Cleaning"

def select_best_person_for_frame(active_tracks, frame_shape):
    """
    Select the best person from tracked persons for this frame
    Criteria: highest confidence, most central, most complete pose
    """
    if not active_tracks:
        return None
    
    best_track = None
    best_score = -1
    
    frame_center_x, frame_center_y = frame_shape[1] // 2, frame_shape[0] // 2
    
    for track in active_tracks:
        if track.last_pose is None:
            continue
        
        pose = track.get_smoothed_pose()  # Get temporally smoothed pose
        
        # Calculate score based on multiple criteria
        score = 0
        
        # 1. Detection confidence
        score += track.score * 0.3
        
        # 2. Pose completeness (how many keypoints are valid)
        valid_keypoints = np.sum(pose[:, 2] > 0.1)
        completeness = valid_keypoints / len(pose)
        score += completeness * 0.3
        
        # 3. Centrality (person closer to center gets higher score)
        bbox_center_x = (track.bbox[0] + track.bbox[2]) / 2
        bbox_center_y = (track.bbox[1] + track.bbox[3]) / 2
        distance_to_center = math.sqrt(
            (bbox_center_x - frame_center_x)**2 + (bbox_center_y - frame_center_y)**2
        )
        max_distance = math.sqrt(frame_center_x**2 + frame_center_y**2)
        centrality = 1.0 - (distance_to_center / max_distance)
        score += centrality * 0.2
        
        # 4. Track stability (longer tracks are preferred)
        stability = min(track.tracklet_len / 30.0, 1.0)  # Normalize to 30 frames
        score += stability * 0.2
        
        if score > best_score:
            best_score = score
            best_track = track
    
    return best_track

def process_video_with_tracking(video_path, pose_model, output_json_path=None, label=None):
    """
    Process a single video with ByteTracker for person consistency
    """
    # Initialize tracker
    tracker = ByteTracker(frame_rate=30, track_thresh=0.6, match_thresh=0.8)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Determine label if not provided
    if label is None:
        label = get_video_label(video_path)
    
    frame_idx = 0
    video_data = []
    
    print(f"Processing video: {video_path.name}")
    print(f"Total frames: {total_frames}, FPS: {fps}, Resolution: {width}x{height}")
    print(f"Label: {label}")
    
    # Track statistics
    track_stats = {
        'total_persons_detected': 0,
        'frames_with_person': 0,
        'frames_without_person': 0,
        'unique_track_ids': set()
    }
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run pose estimation
        detections = run_pose_estimation_with_tracking(frame, pose_model)
        
        # Update tracker
        active_tracks = tracker.update(detections)
        
        # Select best person for this frame
        best_track = select_best_person_for_frame(active_tracks, frame.shape)
        
        # Create frame entry
        frame_entry = {
            "frame_num": frame_idx,
            "clicked": False,
            "x": 0,
            "y": 0,
            "label": label,
            "width": width,
            "height": height,
            "joints": []
        }
        
        if best_track is not None:
            # Get smoothed pose for consistency
            smoothed_pose = best_track.get_smoothed_pose()
            
            # Add tracking information
            frame_entry["joints"] = [smoothed_pose.tolist()]
            frame_entry["track_id"] = best_track.track_id
            frame_entry["detection_score"] = float(best_track.score)
            frame_entry["track_length"] = best_track.tracklet_len
            
            # Update statistics
            track_stats['frames_with_person'] += 1
            track_stats['unique_track_ids'].add(best_track.track_id)
        else:
            frame_entry["track_id"] = -1  # No person detected
            track_stats['frames_without_person'] += 1
        
        track_stats['total_persons_detected'] += len(detections)
        video_data.append(frame_entry)
        frame_idx += 1
        
        # Progress update
        if frame_idx % 30 == 0:  # Every 30 frames
            print(f"Processed {frame_idx}/{total_frames} frames ({frame_idx/total_frames*100:.1f}%)")
    
    cap.release()
    
    # Set output path if not provided
    if output_json_path is None:
        output_json_path = video_path.with_suffix(".json")
    
    # Add metadata to JSON
    metadata = {
        "video_info": {
            "filename": video_path.name,
            "total_frames": frame_idx,
            "fps": fps,
            "width": width,
            "height": height,
            "label": label
        },
        "tracking_stats": {
            "frames_with_person": track_stats['frames_with_person'],
            "frames_without_person": track_stats['frames_without_person'],
            "person_detection_rate": track_stats['frames_with_person'] / frame_idx if frame_idx > 0 else 0,
            "unique_persons_tracked": len(track_stats['unique_track_ids']),
            "avg_persons_per_frame": track_stats['total_persons_detected'] / frame_idx if frame_idx > 0 else 0
        }
    }
    
    # Save JSON data
    output_data = {
        "metadata": metadata,
        "frames": video_data
    }
    
    with open(output_json_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Saved pose data to {output_json_path}")
    print(f"Tracking Statistics:")
    print(f"  - Total frames processed: {frame_idx}")
    print(f"  - Frames with person: {track_stats['frames_with_person']} ({track_stats['frames_with_person']/frame_idx*100:.1f}%)")
    print(f"  - Unique persons tracked: {len(track_stats['unique_track_ids'])}")
    print(f"  - Person detection rate: {track_stats['frames_with_person']/frame_idx*100:.1f}%")
    
    return output_json_path

def main():
    parser = argparse.ArgumentParser(description="Process videos with ByteTracker + YOLO pose estimation for temporal consistency")
    parser.add_argument('--video-dir', type=str, default=None, help='Directory containing .mp4 videos')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory to save JSON files (default: same as video directory)')
    parser.add_argument('--video-file', type=str, default=None, help='Process single video file instead of directory')
    parser.add_argument('--label', type=str, default=None, choices=['Cleaning', 'Not-Cleaning'], 
                       help='Force label for all videos (default: auto-detect from filename)')
    parser.add_argument('--cleaning-keywords', type=str, nargs='+', default=None,
                       help='Custom keywords to identify cleaning videos (space-separated)')
    parser.add_argument('--track-thresh', type=float, default=0.6, help='Tracking confidence threshold')
    parser.add_argument('--match-thresh', type=float, default=0.8, help='Matching threshold for tracking')
    parser.add_argument('--model-size', type=str, default='l', choices=['n', 's', 'm', 'l', 'x'], 
                       help='YOLO model size (n=nano, s=small, m=medium, l=large, x=extra-large)')
    
    args = parser.parse_args()

    # Load YOLO pose model
    model_name = f'yolo11{args.model_size}-pose.pt'
    print(f"Loading {model_name} pose model...")
    try:
        pose_model = YOLO(model_name)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have ultralytics installed: pip install ultralytics")
        return

    if args.video_file:
        # Process single video file
        video_path = Path(args.video_file)
        if not video_path.exists():
            print(f"Error: Video file {video_path} not found!")
            return
        
        output_dir = Path(args.output_dir) if args.output_dir else video_path.parent
        output_json = output_dir / (video_path.stem + "_tracked.json")
        
        process_video_with_tracking(video_path, pose_model, output_json, args.label)
    else:
        # Process all videos in directory
        if not args.video_dir:
            print("Error: Either --video-dir or --video-file must be specified!")
            return
            
        video_dir = Path(args.video_dir)
        output_dir = Path(args.output_dir) if args.output_dir else video_dir
        
        if not video_dir.exists():
            print(f"Error: Video directory {video_dir} not found!")
            return
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all video files
        video_files = (list(video_dir.glob("*.mp4")) + 
                      list(video_dir.glob("*.avi")) + 
                      list(video_dir.glob("*.mov")) + 
                      list(video_dir.glob("*.MP4")) + 
                      list(video_dir.glob("*.AVI")) + 
                      list(video_dir.glob("*.MOV")))
        
        if not video_files:
            print(f"No video files found in {video_dir}")
            return
        
        print(f"Found {len(video_files)} video files to process")
        
        # Process each video
        cleaning_count = 0
        not_cleaning_count = 0
        successful_videos = 0
        
        for i, video_file in enumerate(video_files, 1):
            print(f"\n=== Processing video {i}/{len(video_files)} ===")
            output_json = output_dir / (video_file.stem + "_tracked.json")
            
            try:
                # Determine label for this video
                video_label = args.label if args.label else get_video_label(video_file, args.cleaning_keywords)
                
                process_video_with_tracking(video_file, pose_model, output_json, video_label)
                print(f"✓ Successfully processed {video_file.name} - Label: {video_label}")
                
                # Count labels
                if video_label == "Cleaning":
                    cleaning_count += 1
                else:
                    not_cleaning_count += 1
                
                successful_videos += 1
                    
            except Exception as e:
                print(f"✗ Error processing {video_file.name}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\n=== Processing Complete! ===")
        print(f"Successfully processed: {successful_videos}/{len(video_files)} videos")
        print(f"Cleaning videos: {cleaning_count}")
        print(f"Not-Cleaning videos: {not_cleaning_count}")
        print(f"JSON files saved to: {output_dir}")
        print(f"\nNext steps:")
        print(f"1. Update your json_to_stgcn script to handle the new JSON format with 'frames' array")
        print(f"2. Use the 'track_id' field to ensure temporal consistency")
        print(f"3. The poses are already smoothed for better temporal stability")

if __name__ == "__main__":
    main()