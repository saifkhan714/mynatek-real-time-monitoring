import os
import cv2
import torch
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import mediapipe as mp

from stgcn.ActionsEstLoader import TSSTG
from stgcn.pose_utils import normalize_points_with_size, scale_pose

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def extract_frames(video_path, output_dir=None):
    """Extract frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return None

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for i in tqdm(range(frame_count), desc="Extracting frames"):
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame.copy())

        if output_dir:
            cv2.imwrite(os.path.join(output_dir, f"frame_{i:04d}.jpg"), frame)

    cap.release()
    return frames, (width, height), fps

def mediapipe_to_stgcn_format(pose_landmark, image_shape):
    """
    Convert MediaPipe pose landmarks to format expected by STGCN.
    Maps 33 MediaPipe joints to 18 joints used in our model.
    """
    h, w = image_shape[:2]
    # Initialize array for 18 joints (x, y, score)
    skeleton = np.zeros((18, 3))

    # MediaPipe provides 33 landmarks, we need to map to our 18 joints
    # Format is [x, y, visibility]

    # Basic mapping from MediaPipe to 18-joint format:
    # 0: nose (0)
    # 1: neck (between shoulders - 11,12)
    # 2: right shoulder (12)
    # 3: right elbow (14)
    # 4: right wrist (16)
    # 5: left shoulder (11)
    # 6: left elbow (13)
    # 7: left wrist (15)
    # 8: right hip (24)
    # 9: right knee (26)
    # 10: right ankle (28)
    # 11: left hip (23)
    # 12: left knee (25)
    # 13: left ankle (27)
    # 14: right eye (5)
    # 15: left eye (2)
    # 16: right ear (8)
    # 17: left ear (7)

    mapping = {
        0: 0,   # nose
        1: [11, 12],  # neck (average of shoulders)
        2: 12,  # right shoulder
        3: 14,  # right elbow
        4: 16,  # right wrist
        5: 11,  # left shoulder
        6: 13,  # left elbow
        7: 15,  # left wrist
        8: 24,  # right hip
        9: 26,  # right knee
        10: 28, # right ankle
        11: 23, # left hip
        12: 25, # left knee
        13: 27, # left ankle
        14: 5,  # right eye
        15: 2,  # left eye
        16: 8,  # right ear
        17: 7   # left ear
    }

    landmarks = pose_landmark.landmark

    for i in range(18):
        if i == 1:  # Neck is average of shoulders
            idx1, idx2 = mapping[i]
            x = (landmarks[idx1].x + landmarks[idx2].x) / 2
            y = (landmarks[idx1].y + landmarks[idx2].y) / 2
            # Average visibility
            v = (landmarks[idx1].visibility + landmarks[idx2].visibility) / 2
        else:
            idx = mapping[i]
            x = landmarks[idx].x
            y = landmarks[idx].y
            v = landmarks[idx].visibility

        # Store coordinates directly (MediaPipe provides normalized coordinates)
        skeleton[i, 0] = x
        skeleton[i, 1] = y
        skeleton[i, 2] = v  # Use visibility as confidence score

    return skeleton

def detect_poses(frames):
    """Detect poses in video frames using MediaPipe."""
    all_poses = []

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

        for i, frame in enumerate(tqdm(frames, desc="Detecting poses")):
            # Convert the BGR image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Process the image and get pose landmarks
            results = pose.process(image)

            if results.pose_landmarks:
                # Convert from MediaPipe format to our 18-joint format
                skeleton = mediapipe_to_stgcn_format(results.pose_landmarks, frame.shape)
                all_poses.append(skeleton)
            else:
                # If no pose detected, add zeros
                all_poses.append(np.zeros((18, 3)))

    return np.array(all_poses)

def sliding_window_predictions(model, poses, frame_size, window_size=30, stride=15):
    """Get predictions using sliding window approach."""
    num_frames = len(poses)
    predictions = []
    window_positions = []

    # Handle case when video is shorter than the window size
    if num_frames < window_size:
        # Pad with zeros to reach window_size
        padding = np.zeros((window_size - num_frames, poses.shape[1], poses.shape[2]))
        input_poses = np.concatenate([poses, padding], axis=0)
        pred = model.predict(input_poses, frame_size)
        predictions.append(pred[0])
        window_positions.append(0)
    else:
        # Slide window across the video
        for i in range(0, num_frames - window_size + 1, stride):
            input_poses = poses[i:i+window_size]
            pred = model.predict(input_poses, frame_size)
            predictions.append(pred[0])
            window_positions.append(i)

    return np.array(predictions), window_positions

def plot_predictions(predictions, class_names, window_positions, fps, save_path=None):
    """Plot action probabilities over time."""
    fig, ax = plt.subplots(figsize=(15, 8))

    # Convert frame indices to time in seconds
    time_positions = [pos / fps for pos in window_positions]

    for i, class_name in enumerate(class_names):
        # Extract probabilities for this class
        class_probs = predictions[:, i]
        ax.plot(time_positions, class_probs, label=class_name)

    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Probability')
    ax.set_title('Action Recognition Probabilities')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path)
        print(f"Saved prediction plot to {save_path}")

    plt.close()

def create_visualization_video(frames, poses, predictions, class_names, window_positions, output_path):
    """Create a video visualization of action recognition results."""
    height, width = frames[0].shape[:2]

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))

    # Find best prediction for each frame
    num_frames = len(frames)
    frame_predictions = np.zeros((num_frames, len(class_names)))

    # Assign window predictions to frames
    for i, win_pos in enumerate(window_positions):
        # The prediction applies to all frames in the window
        win_end = min(win_pos + 30, num_frames)
        for j in range(win_pos, win_end):
            frame_predictions[j] = predictions[i]

    # Create visualization
    for i, frame in enumerate(tqdm(frames, desc="Creating visualization")):
        # Draw skeleton on frame if available
        if i < len(poses) and np.any(poses[i]):
            frame_with_skeleton = frame.copy()
            for j, joint in enumerate(poses[i]):
                if joint[2] > 0.1:  # Only draw if confidence is high enough
                    x, y = int(joint[0]), int(joint[1])
                    cv2.circle(frame_with_skeleton, (x, y), 5, (0, 255, 0), -1)

                    # Draw connections between joints (simplified)
                    if j > 0:
                        # Simple connections - you can define your own based on skeleton structure
                        if j in [2, 3, 4] and j-1 in [1, 2, 3]:  # Right arm
                            prev_joint = poses[i][j-1]
                            cv2.line(frame_with_skeleton,
                                    (int(prev_joint[0]), int(prev_joint[1])),
                                    (x, y), (0, 255, 0), 2)
                        elif j in [5, 6, 7] and j-1 in [1, 5, 6]:  # Left arm
                            prev_joint = poses[i][j-1]
                            cv2.line(frame_with_skeleton,
                                    (int(prev_joint[0]), int(prev_joint[1])),
                                    (x, y), (0, 255, 0), 2)
                        elif j in [8, 9, 10] and j-1 in [1, 8, 9]:  # Right leg
                            prev_joint = poses[i][j-1]
                            cv2.line(frame_with_skeleton,
                                    (int(prev_joint[0]), int(prev_joint[1])),
                                    (x, y), (0, 255, 0), 2)
                        elif j in [11, 12, 13] and j-1 in [1, 11, 12]:  # Left leg
                            prev_joint = poses[i][j-1]
                            cv2.line(frame_with_skeleton,
                                    (int(prev_joint[0]), int(prev_joint[1])),
                                    (x, y), (0, 255, 0), 2)

            frame = frame_with_skeleton

        # Get most probable action for this frame
        if i < len(frame_predictions):
            action_idx = np.argmax(frame_predictions[i])
            action_prob = frame_predictions[i][action_idx]
            action_text = f"{class_names[action_idx]}: {action_prob:.2f}"

            # Draw info on frame
            cv2.putText(frame, action_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw probability bars
            bar_height = 20
            y_offset = 50
            for j, prob in enumerate(frame_predictions[i]):
                # Draw label
                cv2.putText(frame, class_names[j], (10, y_offset + j * (bar_height + 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Draw bar
                bar_width = int(prob * 200)
                cv2.rectangle(frame, (100, y_offset + j * (bar_height + 10) - bar_height),
                              (100 + bar_width, y_offset + j * (bar_height + 10)),
                              (0, 255, 0), -1)

                # Draw probability value
                cv2.putText(frame, f"{prob:.2f}", (310, y_offset + j * (bar_height + 10) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        out.write(frame)

    out.release()
    print(f"Video saved to {output_path}")

def draw_skeleton(frame, skeleton, color=(0, 255, 0)):
    """Draw skeleton on frame."""
    # Define connections between joints
    connections = [
        (0, 1),   # nose to neck
        (1, 2),   # neck to right shoulder
        (2, 3),   # right shoulder to right elbow
        (3, 4),   # right elbow to right wrist
        (1, 5),   # neck to left shoulder
        (5, 6),   # left shoulder to left elbow
        (6, 7),   # left elbow to left wrist
        (1, 8),   # neck to right hip
        (8, 9),   # right hip to right knee
        (9, 10),  # right knee to right ankle
        (1, 11),  # neck to left hip
        (11, 12), # left hip to left knee
        (12, 13), # left knee to left ankle
        (0, 14),  # nose to right eye
        (0, 15),  # nose to left eye
        (14, 16), # right eye to right ear
        (15, 17)  # left eye to left ear
    ]
    
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Draw joints
    for i in range(len(skeleton)):
        if skeleton[i, 2] > 0.1:  # Only draw if confidence is high enough
            # Convert normalized coordinates to pixel coordinates
            x = int(skeleton[i, 0] * width)
            y = int(skeleton[i, 1] * height)
            # Draw larger circles for better visibility
            cv2.circle(frame, (x, y), 6, color, -1)
            # Add white border to joints
            cv2.circle(frame, (x, y), 7, (255, 255, 255), 1)
    
    # Draw connections
    for connection in connections:
        if skeleton[connection[0], 2] > 0.1 and skeleton[connection[1], 2] > 0.1:
            # Convert normalized coordinates to pixel coordinates
            start_point = (
                int(skeleton[connection[0], 0] * width),
                int(skeleton[connection[0], 1] * height)
            )
            end_point = (
                int(skeleton[connection[1], 0] * width),
                int(skeleton[connection[1], 1] * height)
            )
            # Draw thicker lines for better visibility
            cv2.line(frame, start_point, end_point, color, 3)
            # Add white border to lines
            cv2.line(frame, start_point, end_point, (255, 255, 255), 1)

def main():
    parser = argparse.ArgumentParser(description='Test STGCN on video')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--model', type=str, default='output/models/best_model.pth',
                        help='Path to model weights')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run inference on')
    parser.add_argument('--output', type=str, default='output/processed_video.mp4',
                        help='Path to save processed video')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Load video
    print(f"Processing video: {args.video}")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize video writer with H.264 codec
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Use H.264 codec
    video_writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    if not video_writer.isOpened():
        print("Error: Could not create output video.")
        return

    # Initialize model
    action_model = TSSTG(weight_file=args.model, device=args.device)

    # Process frames
    frames = []
    print("Extracting frames...")
    for _ in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    # Detect poses
    print("Detecting poses...")
    poses = []
    for frame in tqdm(frames):
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect pose
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            # Convert from MediaPipe format to our 18-joint format
            skeleton = mediapipe_to_stgcn_format(results.pose_landmarks, frame.shape)
            poses.append(skeleton)
        else:
            # If no pose detected, add zeros
            poses.append(np.zeros((18, 3)))

    # Convert poses to numpy array
    poses = np.array(poses)
    
    # Store original poses for visualization
    vis_poses = poses.copy()
    
    # Normalize and scale poses for model input
    poses[:, :, :2] = normalize_points_with_size(poses[:, :, :2], width, height)
    poses[:, :, :2] = scale_pose(poses[:, :, :2])
    
    # Prepare input for model
    pts = torch.tensor(poses, dtype=torch.float32)
    pts = pts.permute(2, 0, 1)[None, :]  # (1, C, T, V)
    
    # Create motion input
    mot = pts[:, :2, 1:, :] - pts[:, :2, :-1, :]
    
    # Make prediction
    with torch.no_grad():
        model_output = action_model.predict(pts, mot)
    
    # Get prediction
    pred = model_output[0]
    action = "Cleaning" if pred[0] > 0.5 else "Not-Cleaning"
    confidence = pred[0] if pred[0] > 0.5 else 1 - pred[0]
    
    print(f"\nPrediction: {action}")
    print(f"Confidence: {confidence:.2f}")

    # Create visualization
    print("Creating visualization...")
    for i, frame in enumerate(tqdm(frames)):
        # Create a copy of the frame for drawing
        frame_with_overlay = frame.copy()
        
        # Draw skeleton using original coordinates
        if i < len(vis_poses):
            draw_skeleton(frame_with_overlay, vis_poses[i])
        
        # Add prediction text with background for better visibility
        text = f"{action}: {confidence:.2f}"
        # Get text size
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        # Draw background rectangle
        cv2.rectangle(frame_with_overlay, (5, 5), (text_width + 15, text_height + 15), (0, 0, 0), -1)
        # Draw text
        cv2.putText(frame_with_overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Ensure frame is in BGR format and write
        if frame_with_overlay.shape[2] == 3:  # Check if frame has 3 channels
            video_writer.write(frame_with_overlay)
        else:
            print(f"Warning: Frame {i} has incorrect format. Skipping.")

    # Clean up
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    print(f"\nProcessed video saved to: {args.output}")

if __name__ == '__main__':
    main()