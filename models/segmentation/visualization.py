import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path

class SegmentationVisualizer:
    def __init__(self, class_names: list):
        self.class_names = class_names
        
    def render_sample(self, img_path: Path, lbl_path: Path, alpha: float = 0.40):
        """Render a single image with bounding boxes and segmentation masks"""
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        with open(lbl_path) as f:
            for line in f:
                parts = list(map(float, line.split()))
                cls, cx, cy, bw, bh, *poly = parts
                cls = int(cls)

                # Draw bounding box
                x, y = cx*w, cy*h; bw, bh = bw*w, bh*h
                p1 = (int(x-bw/2), int(y-bh/2)); p2 = (int(x+bw/2), int(y+bh/2))
                cv2.rectangle(img, p1, p2, (0,255,0), 2)

                label = self.class_names[cls] if cls < len(self.class_names) else f'cls_{cls}'
                cv2.putText(img, label, p1, cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255,255,255), 2)

                # Draw polygon mask
                if poly:
                    pts = np.array(poly).reshape(-1,2)
                    pts[:,0] *= w; pts[:,1] *= h
                    overlay = img.copy()
                    cv2.fillPoly(overlay, [pts.astype(int)], (0,128,255))
                    img = cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)

        return img

    def show_seg_collage(self, img_dir: Path, lbl_dir: Path, n: int = 9, title: str = 'Segmentation collage'):
        """Show a grid of segmentation examples"""
        paths = random.sample(list(img_dir.glob('*.jpg')), n)
        plt.figure(figsize=(10,10))
        
        for i, p in enumerate(paths, 1):
            img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
            h, w, _ = img.shape
            txt = lbl_dir / (p.stem + '.txt')
            colours = [(0,255,0),(255,0,0),(0,0,255),(255,255,0)]
            
            with open(txt) as f:
                for line in f:
                    cls, *poly = line.split()
                    pts = self._poly_to_pts(' '.join(poly), w, h)
                    img = self._draw_mask(img, pts, random.choice(colours))
                    
            plt.subplot(3,3,i)
            plt.imshow(img)
            plt.axis('off')
            
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def _poly_to_pts(poly: str, w: int, h: int) -> np.ndarray:
        """Convert normalized polygon coordinates to pixel coordinates"""
        coords = list(map(float, poly.split()))
        pts = np.array(coords).reshape(-1, 2)
        pts[:, 0] *= w
        pts[:, 1] *= h
        return pts.astype(int)
    
    @staticmethod
    def _draw_mask(img: np.ndarray, pts: np.ndarray, colour: tuple) -> np.ndarray:
        """Draw a mask overlay on the image"""
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], colour)
        return cv2.addWeighted(overlay, 0.4, img, 0.6, 0) 