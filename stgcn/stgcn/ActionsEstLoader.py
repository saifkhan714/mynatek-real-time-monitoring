import os
import torch
import torch.nn as nn
import numpy as np

from stgcn.Models import TwoStreamSpatialTemporalGraph
from stgcn.pose_utils import normalize_points_with_size, scale_pose


class TSSTG(nn.Module):
    """Two-Stream Spatial Temporal Graph Model Loader"""
    def __init__(self, weight_file=None, device='cpu'):
        super().__init__()
        self.device = device
        
        # Initialize model with basic architecture first
        graph_args = {'strategy': 'spatial'}
        self.model = TwoStreamSpatialTemporalGraph(graph_args, 2).to(device)
        
        if weight_file is not None:
            print(f'Loading action recognition model from {weight_file}')
            # Load state dict
            state_dict = torch.load(weight_file, map_location=torch.device(device))
            
            # Check if the loaded model has the old architecture
            if 'fcn.weight' in state_dict:
                # Create a new state dict for the enhanced architecture
                new_state_dict = {}
                
                # Map old parameters to new architecture
                for key, value in state_dict.items():
                    if key == 'fcn.weight':
                        new_state_dict['fusion.8.weight'] = value
                    elif key == 'fcn.bias':
                        new_state_dict['fusion.8.bias'] = value
                    else:
                        # For other parameters, try to map them to the new architecture
                        if 'pts_stream' in key or 'mot_stream' in key:
                            # Remove attention-related parameters from the key
                            if 'attention' not in key:
                                new_key = key
                                new_state_dict[new_key] = value
                
                # Load the mapped state dict
                self.model.load_state_dict(new_state_dict, strict=False)
                print("Loaded old model architecture and mapped to new architecture")
            else:
                # Load the new architecture directly
                self.model.load_state_dict(state_dict)
                print("Loaded new model architecture")
            
            self.model.eval()
    
    def predict(self, pts, mot):
        """Predict action from points and motion."""
        with torch.no_grad():
            pts = torch.tensor(pts, dtype=torch.float32, device=self.device)
            mot = torch.tensor(mot, dtype=torch.float32, device=self.device)
            
            # Add batch dimension if needed
            if pts.dim() == 3:
                pts = pts.unsqueeze(0)
            if mot.dim() == 3:
                mot = mot.unsqueeze(0)
            
            # Forward pass
            out = self.model((pts, mot))
            
            return out.cpu().numpy()