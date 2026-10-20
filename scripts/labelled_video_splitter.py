import cv2
import json
import argparse
from pathlib import Path
import re
from datetime import datetime

def parse_time(time_str):
    """Convert time string to seconds.
    Format: M.SS where M is minutes and SS is seconds
    Example: 1.26 means 1 minute 26 seconds = 86 seconds
    """
    try:
        minutes, seconds = map(float, time_str.split('.'))
        total_seconds = (minutes * 60) + seconds
        return total_seconds
    except ValueError:
        raise ValueError(f"Invalid time format: {time_str}. Expected format: M.SS (e.g., 1.26 for 1 minute 26 seconds)")

def parse_time_ranges(time_file, video_duration):
    """Parse time ranges from text file and validate against video duration."""
    ranges = []
    with open(time_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Split by comma and handle different separators
            times = re.split(r'[,;]', line)
            if len(times) != 2:
                raise ValueError(f"Invalid time range format at line {line_num}: {line}. Expected format: M.SS,M.SS")

            try:
                start = parse_time(times[0].strip())
                end = parse_time(times[1].strip())

                # Validate times
                if start < 0 or end < 0:
                    raise ValueError(f"Negative time values not allowed at line {line_num}: {line}")
                if start >= end:
                    raise ValueError(f"Start time must be less than end time at line {line_num}: {line}")
                if end > video_duration:
                    raise ValueError(f"End time {format_time(end)} exceeds video duration {format_time(video_duration)} at line {line_num}: {line}")

                ranges.append((start, end))
            except ValueError as e:
                raise ValueError(f"Error at line {line_num}: {str(e)}")

    return sorted(ranges)

def format_time(seconds):
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

def split_video_and_labels(video_path, time_file, json_path, output_dir=None):
    """Split video and its corresponding JSON labels into chunks while removing specified time segments."""
    # Open video first to get duration
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    # Load JSON labels
    with open(json_path, 'r') as f:
        labels = json.load(f)

    # Create a frame number to label mapping
    label_map = {entry['frame_num']: entry for entry in labels}

    # Parse and validate time ranges
    try:
        remove_ranges = parse_time_ranges(time_file, duration)
    except ValueError as e:
        cap.release()
        raise ValueError(f"Time range validation failed: {str(e)}")

    # Get other video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create output directory if not specified
    if output_dir is None:
        output_dir = Path(video_path).parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Generate keep ranges (inverse of remove ranges)
    keep_ranges = []
    current_time = 0

    for start, end in remove_ranges:
        if start > current_time:
            keep_ranges.append((current_time, start))
        current_time = end

    if current_time < duration:
        keep_ranges.append((current_time, duration))

    # Process each keep range
    for i, (start_time, end_time) in enumerate(keep_ranges):
        # Calculate frame numbers
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        # Create output filenames
        video_output_path = output_dir / f"{Path(video_path).stem}_{i}.mp4"
        json_output_path = output_dir / f"{Path(video_path).stem}_{i}_labels.json"

        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_output_path), fourcc, fps, (width, height))

        # Set video position to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Create new labels list for this chunk
        chunk_labels = []
        new_frame_num = 1

        # Write frames and collect labels
        for frame_num in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break

            out.write(frame)

            # Get label for this frame (default to "Not-Cleaning" if not found)
            label_entry = label_map.get(frame_num, {
                "frame_num": frame_num,
                "clicked": False,
                "x": 0,
                "y": 0,
                "label": "Not-Cleaning"
            })

            # Create new label entry with adjusted frame number
            new_label_entry = {
                "frame_num": new_frame_num,
                "clicked": label_entry["clicked"],
                "x": label_entry["x"],
                "y": label_entry["y"],
                "label": label_entry["label"]
            }
            chunk_labels.append(new_label_entry)
            new_frame_num += 1

        out.release()

        # Save labels for this chunk
        with open(json_output_path, 'w') as f:
            json.dump(chunk_labels, f, indent=4)

        # Print progress
        print(f"Created chunk {i}:")
        print(f"  Video: {video_output_path}")
        print(f"  Labels: {json_output_path}")
        print(f"  Duration: {format_time(start_time)} to {format_time(end_time)}")
        print(f"  Frames: {len(chunk_labels)}")

    cap.release()
    print("\nVideo and label splitting completed!")

def main():
    parser = argparse.ArgumentParser(description="Split video and its JSON labels into chunks while removing specified time segments")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("time_file", help="Path to text file containing time ranges to remove")
    parser.add_argument("json_path", help="Path to JSON file containing frame labels")
    parser.add_argument("--output-dir", help="Directory to save output videos and labels (default: same as input video)")

    args = parser.parse_args()

    try:
        split_video_and_labels(args.video_path, args.time_file, args.json_path, args.output_dir)
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
