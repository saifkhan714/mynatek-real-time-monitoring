from pathlib import Path
import yaml
import cv2
import numpy as np

def create_data_yaml(data_dir: str, class_names: list, output_path: str):
    """Create YAML configuration file for YOLO training"""
    # data_dir is the frames directory; set path to its parent
    frames_dir = Path(data_dir)
    parent_dir = str(frames_dir.parent)
    data = {
        'path': parent_dir,
        'train': 'frames/train',
        'val': 'frames/val',
        'names': {i: name for i, name in enumerate(class_names)}
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

def load_data_yaml(yaml_path):
    """Load data.yaml file containing class names and frame information"""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data

def extract_frames(video_path, output_dir, total_frames):
    """Extract frames from video and save them to output directory"""
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame interval to get desired number of frames
    if frame_count > total_frames:
        interval = frame_count / total_frames
    else:
        interval = 1
    
    frame_idx = 0
    saved_idx = 0
    
    while saved_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % interval < 1:
            frame_path = output_dir / f"frame_{saved_idx:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            saved_idx += 1
            
        frame_idx += 1
    
    cap.release()
    return saved_idx

def load_segmentation_annotation(annotation_path):
    """Load segmentation annotation from text file"""
    annotations = []
    with open(annotation_path, 'r') as f:
        for line in f:
            values = line.strip().split()
            if len(values) < 3:  # Need at least class_id and one point
                continue
                
            class_id = int(values[0])
            points = np.array([float(x) for x in values[1:]]).reshape(-1, 2)
            annotations.append({
                'class_id': class_id,
                'points': points
            })
    return annotations
