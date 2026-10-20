import json
import argparse
from pathlib import Path

def fill_missing_labels(json_path, output_path=None, create_new=False):
    """Fill in missing frame labels using the last known label."""
    # Load JSON labels
    with open(json_path, 'r') as f:
        labels = json.load(f)

    # Sort labels by frame number
    labels.sort(key=lambda x: x['frame_num'])

    # Create a mapping of frame numbers to labels
    frame_to_label = {entry['frame_num']: entry for entry in labels}

    # Find the range of frames
    min_frame = min(entry['frame_num'] for entry in labels)
    max_frame = max(entry['frame_num'] for entry in labels)

    # Find missing frames and fill them
    last_known_label = None
    filled_labels = []

    for frame_num in range(min_frame, max_frame + 1):
        if frame_num in frame_to_label:
            # Frame exists, update last known label
            last_known_label = frame_to_label[frame_num]
            filled_labels.append(last_known_label)
        elif last_known_label is not None:
            # Frame is missing, use last known label
            new_entry = {
                "frame_num": frame_num,
                "clicked": last_known_label["clicked"],
                "x": last_known_label["x"],
                "y": last_known_label["y"],
                "label": last_known_label["label"]
            }
            filled_labels.append(new_entry)
            print(f"Filled missing frame {frame_num} with label from frame {last_known_label['frame_num']}")

    # Determine output path
    if create_new:
        if output_path is None:
            output_path = str(Path(json_path).with_name(f"{Path(json_path).stem}_filled.json"))
    else:
        output_path = json_path

    # Save the filled labels
    with open(output_path, 'w') as f:
        json.dump(filled_labels, f, indent=4)

    print(f"\nProcessed {len(filled_labels)} frames")
    print(f"Original labels: {len(labels)}")
    print(f"Filled labels: {len(filled_labels) - len(labels)}")
    if create_new:
        print(f"Output saved to: {output_path}")
    else:
        print(f"Updated file: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Fill in missing frame labels using the last known label")
    parser.add_argument("json_path", help="Path to input JSON file containing frame labels")
    parser.add_argument("--output", help="Path to output JSON file (only used with --create-new)")
    parser.add_argument("--create-new", action="store_true", help="Create a new file instead of updating the input file")

    args = parser.parse_args()

    try:
        fill_missing_labels(args.json_path, args.output, args.create_new)
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
