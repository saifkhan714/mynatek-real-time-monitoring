from pathlib import Path
import argparse
from data_exploration import DataExplorer
from visualization import SegmentationVisualizer
from training import YOLOSegmentationTrainer
from inference import SegmentationInference
from utils import extract_frames, load_data_yaml

#load class names from classes.txt
def load_class_names(classes_file):
    with open(classes_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def parse_args():
    parser = argparse.ArgumentParser(description='Segmentation training and inference pipeline')
    parser.add_argument('--data-dir', type=str, required=True,
                      help='Path to the data directory containing videos and their annotations')
    parser.add_argument('--output-dir', type=str, default='runs/segment',
                      help='Path to save training outputs (default: runs/segment)')
    parser.add_argument('--extract-frames', action='store_true',
                      help='Extract frames from videos before training')
    parser.add_argument('--resume', action='store_true',
                      help='Resume training from last checkpoint')
    return parser.parse_args()

def process_video(video_path, video_dir, output_dir, extract_frames_flag=False):
    """Process a single video and its annotations"""
    # Load data.yaml
    data_yaml = load_data_yaml(video_dir / 'data.yaml')
    total_frames = data_yaml['frames']
    processed_frames = data_yaml['processed']
    
    # Create output directories
    video_output_dir = output_dir / video_path.stem
    frames_dir = video_output_dir / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract frames if requested
    if extract_frames_flag:
        print(f"Extracting {total_frames} frames from {video_path.name}...")
        extract_frames(video_path, frames_dir, total_frames)

    # Create data.yaml in frames directory for YOLO training
    from utils import create_data_yaml
    class_names = [data_yaml['names'][i] for i in range(len(data_yaml['names']))]
    create_data_yaml(
        data_dir=str(frames_dir),
        class_names=class_names,
        output_path=str(frames_dir / 'data.yaml')
    )
    
    return {
        'video_name': video_path.name,
        'total_frames': total_frames,
        'processed_frames': processed_frames,
        'frames_dir': frames_dir,
        'labels_dir': video_dir / 'labels',
        'class_names': class_names
    }

def main():
    args = parse_args()
    DATA_DIR = Path(args.data_dir)
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize data explorer
    explorer = DataExplorer(DATA_DIR)
    
    # Get video information
    video_info = explorer.get_video_info()
    print("\nVideo Information:")
    print(video_info)
    
    # Process each video
    processed_videos = []
    for video_path in DATA_DIR.glob('*.mp4'):
        video_dir = DATA_DIR / video_path.stem
        if not video_dir.exists():
            print(f"Warning: No directory found for video {video_path.name}")
            continue
            
        video_data = process_video(
            video_path, 
            video_dir, 
            OUTPUT_DIR,
            args.extract_frames
        )
        processed_videos.append(video_data)
    
    if not processed_videos:
        print("No valid videos found to process")
        return
    
    # Data exploration
    print("\nAnalyzing data distribution...")
    explorer.plot_class_histogram()
    explorer.analyze_mask_statistics()
    
    # Sequential training: use previous best.pt as starting weights for next video
    prev_best_weights = None
    for idx, video_data in enumerate(processed_videos):
        print(f"\nTraining on {video_data['video_name']}...")
        # Use previous best.pt as weights if available
        if prev_best_weights is not None:
            trainer = YOLOSegmentationTrainer(weights=str(prev_best_weights))
        else:
            trainer = YOLOSegmentationTrainer()
        trainer.train(
            data_yaml=str(video_data['frames_dir'] / 'data.yaml'),
            project_dir=str(OUTPUT_DIR / video_data['video_name']),
            processed_frames=video_data['processed_frames'],
            labels_dir=str(video_data['labels_dir']),
            resume=args.resume
        )
        # Update prev_best_weights to point to the latest best.pt
        best_weights_path = OUTPUT_DIR / video_data['video_name'] / 'train/weights/best.pt'
        if best_weights_path.exists():
            prev_best_weights = best_weights_path

    # After last video, copy final best.pt and important files to summary directory
    if prev_best_weights is not None:
        summary_dir = OUTPUT_DIR / 'final_model'
        summary_dir.mkdir(parents=True, exist_ok=True)
        from shutil import copy2
        copy2(prev_best_weights, summary_dir / 'best.pt')
        # Optionally copy other files (e.g., data.yaml, results.csv)
        last_video = processed_videos[-1]
        results_csv = OUTPUT_DIR / last_video['video_name'] / 'train/results.csv'
        data_yaml = last_video['frames_dir'] / 'data.yaml'
        if results_csv.exists():
            copy2(results_csv, summary_dir / 'results.csv')
        if data_yaml.exists():
            copy2(data_yaml, summary_dir / 'data.yaml')

    # Inference setup
    if prev_best_weights is not None:
        inference = SegmentationInference(summary_dir / 'best.pt')
    else:
        inference = SegmentationInference(OUTPUT_DIR / 'weights/best.pt')

    # Example inference on a test frame
    if processed_videos:
        test_video = processed_videos[0]
        test_frame = test_video['frames_dir'] / 'frame_000000.jpg'
        if test_frame.exists():
            print("\nRunning inference on test frame...")
            inference.infer_image(str(test_frame))

if __name__ == '__main__':
    main()
