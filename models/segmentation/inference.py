import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from utils import load_data_yaml

class SegmentationInference:
    def __init__(self, model_path: str):
        """Initialize the segmentation inference model
        
        Args:
            model_path: Path to the trained model weights
        """
        self.model = YOLO(model_path)
        
    def infer_image(self, image_path: str, conf_threshold: float = 0.25):
        """Run inference on a single image
        
        Args:
            image_path: Path to the input image
            conf_threshold: Confidence threshold for detections
        """
        results = self.model(image_path, conf=conf_threshold)[0]
        
        # Get the original image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        # Draw segmentation masks
        if results.masks is not None:
            for mask, cls, conf in zip(results.masks.data, results.boxes.cls, results.boxes.conf):
                # Convert mask to binary
                binary_mask = mask.cpu().numpy().astype(np.uint8)

                # Resize mask to match image size
                mask_resized = cv2.resize(binary_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

                # Create colored mask
                color = np.random.randint(0, 255, 3).tolist()
                colored_mask = np.zeros_like(image)
                colored_mask[mask_resized > 0] = color

                # Blend mask with image
                alpha = 0.5
                image = cv2.addWeighted(image, 1, colored_mask, alpha, 0)
                
                # Draw class label and confidence
                label = f"{results.names[int(cls)]} {conf:.2f}"
                cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        return image
        
    def infer_video(self, video_path: str, output_path: str = None, conf_threshold: float = 0.25):
        """Run inference on a video
        
        Args:
            video_path: Path to the input video
            output_path: Path to save the output video (optional)
            conf_threshold: Confidence threshold for detections
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Initialize video writer if output path is provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run inference
            results = self.model(frame, conf=conf_threshold)[0]
            
            # Draw segmentation masks
            if results.masks is not None:
                for mask, cls, conf in zip(results.masks.data, results.boxes.cls, results.boxes.conf):
                    # Convert mask to binary
                    binary_mask = mask.cpu().numpy().astype(np.uint8)

                    # Resize mask to match frame size
                    mask_resized = cv2.resize(binary_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

                    # Create colored mask
                    color = np.random.randint(0, 255, 3).tolist()
                    colored_mask = np.zeros_like(frame)
                    colored_mask[mask_resized > 0] = color

                    # Blend mask with frame
                    alpha = 0.5
                    frame = cv2.addWeighted(frame, 1, colored_mask, alpha, 0)
                    
                    # Draw class label and confidence
                    label = f"{results.names[int(cls)]} {conf:.2f}"
                    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Write frame if output path is provided
            if writer:
                writer.write(frame)
                
        cap.release()
        if writer:
            writer.release()
