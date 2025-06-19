import os
import numpy as np
import torch
from torch.utils.data import Dataset

class SketchVoxelDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.samples = [f.split('_sketch')[0] for f in os.listdir(data_dir) 
                       if f.endswith('_sketch.npy')]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        model_id = self.samples[idx]
        
        # Load sketch
        sketch = np.load(os.path.join(self.data_dir, f"{model_id}_sketch.npy"))
        sketch = torch.tensor(sketch, dtype=torch.float32).unsqueeze(0)  # [1, 224, 224]
        
        # Load points and normalize
        points = np.load(os.path.join(self.data_dir, f"{model_id}_points.npy"))
        points = torch.tensor(points / 223.0, dtype=torch.float32)  # [256, 2]
        
        # Load voxel
        voxel = np.load(os.path.join(self.data_dir, f"{model_id}_voxel.npy"))
        voxel = torch.tensor(voxel, dtype=torch.float32).unsqueeze(0)  # [1, 32, 32, 32]
        
        return sketch, points, voxel