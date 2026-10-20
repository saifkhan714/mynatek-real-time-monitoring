import os
import torch
import numpy as np

from stgcn.Models import TwoStreamSpatialTemporalGraph
from stgcn.pose_utils import normalize_points_with_size, scale_pose

class TSSTG(object):
    """Two-Stream Spatial Temporal Graph Model Loader."""
    def __init__(self,
                 weight_file='stgcn_custom_saved/tsstg-model.pth',
                 device=None):
        dev = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(dev)
        print(f"[ST-GCN] Using device: {self.device}")

        self.graph_args = {'layout': 'coco', 'strategy': 'spatial'}
        self.class_names = ['not_cleaning', 'cleaning']
        self.num_class = len(self.class_names)

        self.model = TwoStreamSpatialTemporalGraph(self.graph_args, self.num_class).to(self.device)
        try:
            state = torch.load(weight_file, map_location=self.device)
            self.model.load_state_dict(state)
        except RuntimeError as e:
            print(f"Warning: Could not load pretrained weights due to size mismatch. Model will use random weights. Details: {e}")
        self.model.eval()


    def predict(self, pts, image_size):
        """Predict actions from single person skeleton points and score in time sequence."""
        pts[:, :, :2] = normalize_points_with_size(pts[:, :, :2], image_size[0], image_size[1])
        pts[:, :, :2] = scale_pose(pts[:, :, :2])
        # For YOLOv11/COCO 17 keypoints, do NOT add extra keypoints
        # pts = np.concatenate((pts, np.expand_dims((pts[:, 1, :] + pts[:, 2, :]) / 2, 1)), axis=1)

        pts = torch.tensor(pts, dtype=torch.float32)
        pts = pts.permute(2, 0, 1)[None, :]

        mot = pts[:, :2, 1:, :] - pts[:, :2, :-1, :]
        mot = mot.to(self.device)
        pts = pts.to(self.device)

        out = self.model((pts, mot))
        return out.detach().cpu().numpy()
