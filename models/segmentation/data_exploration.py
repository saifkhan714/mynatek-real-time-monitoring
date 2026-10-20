import os
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import load_data_yaml, load_segmentation_annotation

class DataExplorer:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.video_files = list(self.data_dir.glob('*.mp4'))
        self.video_dirs = [self.data_dir / video.stem for video in self.video_files]
        
        # Load data.yaml from first video directory to get class names
        if self.video_dirs:
            try:
                data_yaml = load_data_yaml(self.video_dirs[0] / 'data.yaml')
                
                # Validate required fields
                required_fields = ['names', 'frames', 'processed']
                missing_fields = [field for field in required_fields if field not in data_yaml]
                if missing_fields:
                    raise ValueError(f"data.yaml is missing required fields: {missing_fields}")
                
                # Handle class names
                if not isinstance(data_yaml['names'], dict) or not all(isinstance(k, int) for k in data_yaml['names'].keys()):
                    raise ValueError("'names' in data.yaml must be a dictionary with integer keys")

                
                # Sort class names by their numeric keys
                self.class_names = [data_yaml['names'][i] for i in range(len(data_yaml['names']))]
                
                # Get frame information
                self.total_frames = int(data_yaml['frames'])
                self.processed_frames = int(data_yaml['processed'])
                
                print(f"Loaded {len(self.class_names)} classes:")
                for i, name in enumerate(self.class_names):
                    print(f"  {i}: {name}")
                print(f"Total frames: {self.total_frames}")
                print(f"Processed frames: {self.processed_frames}")
                
            except Exception as e:
                print(f"Error loading data.yaml: {str(e)}")
                print(f"Please ensure data.yaml exists in {self.video_dirs[0]}")
                raise
        else:
            print(f"No video directories found in {self.data_dir}")
            self.class_names = []
            self.total_frames = 0
            self.processed_frames = 0
        
    def get_video_info(self):
        """Get information about videos and their annotations"""
        info = []
        for video_path, video_dir in zip(self.video_files, self.video_dirs):
            if not video_dir.exists():
                print(f"Warning: Directory not found for video {video_path.name}")
                continue
                
            try:
                data_yaml = load_data_yaml(video_dir / 'data.yaml')
                label_dir = video_dir / 'labels'
                
                if not label_dir.exists():
                    print(f"Warning: Labels directory not found for video {video_path.name}")
                    continue
                    
                label_files = list(label_dir.glob('frame_*.txt'))
                processed_frames = int(data_yaml.get('processed', 0))
                
                info.append({
                    'video': video_path.name,
                    'total_frames': int(data_yaml.get('frames', 0)),
                    'processed_frames': processed_frames,
                    'annotation_files': len(label_files)
                })
            except Exception as e:
                print(f"Error processing video {video_path.name}: {str(e)}")
                continue
            
        return pd.DataFrame(info)
    
    def plot_class_histogram(self):
        """Plot histogram of class distributions across all videos"""
        classes = []
        
        for video_dir in self.video_dirs:
            label_dir = video_dir / 'labels'
            if not label_dir.exists():
                continue
                
            for txt in label_dir.glob('frame_*.txt'):
                annotations = load_segmentation_annotation(txt)
                classes.extend([ann['class_id'] for ann in annotations])
        
        if classes:
            counts = pd.Series(classes).value_counts().sort_index()
            tick_labels = [
                self.class_names[i] if i < len(self.class_names) else f'cls_{i}'
                for i in counts.index
            ]
            ax = counts.plot.bar()
            ax.set_xticklabels(tick_labels, rotation=90)
            plt.title('Class histogram across all videos')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.show()
        else:
            print('⚠️  No labels found – check label files.')
            
    def analyze_mask_statistics(self):
        """Analyze mask counts and areas across all videos"""
        mask_counts, mask_areas = [], []
        
        for video_dir in self.video_dirs:
            label_dir = video_dir / 'labels'
            if not label_dir.exists():
                continue
                
            for txt in label_dir.glob('frame_*.txt'):
                annotations = load_segmentation_annotation(txt)
                mask_counts.append(len(annotations))
                
                for ann in annotations:
                    points = ann['points']
                    xs, ys = points[:,0], points[:,1]
                    mask_areas.append((xs.max()-xs.min())*(ys.max()-ys.min()))
        
        if mask_counts:
            # Plot mask counts
            pd.Series(mask_counts).value_counts().sort_index().plot.bar()
            plt.title('#objects per frame')
            plt.show()
            
            # Plot mask areas
            pd.Series(mask_areas).hist(bins=30)
            plt.title('Approximate mask area distribution')
            plt.show()
        else:
            print('⚠️  No masks found – check annotation files.') 