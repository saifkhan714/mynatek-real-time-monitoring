import cv2
import random
import argparse
from ultralytics import YOLO

def get_random_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        raise RuntimeError("Video has no frames.")
    idx = random.randint(0, frame_count - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read frame {idx} from video.")
    return frame, idx

def run_inference(model_path, data_yaml, frame):
    model = YOLO(model_path)
    results = model.predict(source=frame, data=data_yaml, save=False, show=False, conf=0.25)
    return results[0]

def draw_results(frame, results):
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = results.names[cls] if hasattr(results, "names") and cls in results.names else str(cls)
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        print(f"Detected: {label} | Confidence: {conf:.3f} | Box: ({x1}, {y1}), ({x2}, {y2})")
    return frame

def main():
    parser = argparse.ArgumentParser(description="Run YOLOv8 inference on a random frame from a video.")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained YOLOv8 weights (e.g., best.pt)")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--output", type=str, default=None, help="Path to save output image (optional)")
    args = parser.parse_args()

    frame, idx = get_random_frame(args.video)
    print(f"Selected frame {idx} from video.")

    results = run_inference(args.weights, args.data, frame)
    frame_with_boxes = draw_results(frame, results)

    cv2.imshow("YOLOv8 Inference", frame_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if args.output:
        cv2.imwrite(args.output, frame_with_boxes)
        print(f"Saved output to {args.output}")

if __name__ == "__main__":
    main()
