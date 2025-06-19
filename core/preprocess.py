import os
import numpy as np
from PIL import Image
import binvox_rw
import cv2
import torch
import torch.nn.functional as F

def farthest_point_sampling(points, n_samples):
    n = points.shape[0]
    sampled_indices = np.zeros(n_samples, dtype=int)
    distances = np.full(n, float('inf'))
    
    first_idx = np.random.randint(n)
    sampled_indices[0] = first_idx
    current_point = points[first_idx]
    
    dists = np.linalg.norm(points - current_point, axis=1)
    distances = np.minimum(distances, dists)

    for i in range(1, n_samples):
        idx = np.argmax(distances)
        sampled_indices[i] = idx
        current_point = points[idx]
        new_dists = np.linalg.norm(points - current_point, axis=1)
        distances = np.minimum(distances, new_dists)
    
    return points[sampled_indices]

def sketch_to_binary_and_pointcloud(sketch_path, num_points=256):
    img = Image.open(sketch_path).convert('L')
    img_np = np.array(img)
    
    # Binarize
    binary = (img_np < 50).astype(np.uint8)
    
    # Find bounding box
    non_zero = np.argwhere(binary)
    if non_zero.size == 0:
        cropped = binary
    else:
        rmin, rmax = non_zero[:,0].min(), non_zero[:,0].max()
        cmin, cmax = non_zero[:,1].min(), non_zero[:,1].max()
        cropped = binary[rmin:rmax+1, cmin:cmax+1]
    
    # Resize to 224x224
    resized = np.array(Image.fromarray(cropped).resize((224, 224), Image.NEAREST))
    resized_binary = (resized > 0).astype(np.uint8)
    
    # Get point coordinates
    coords = np.argwhere(resized_binary)
    if coords.size == 0:
        points = np.zeros((num_points, 2), dtype=np.float32)
    else:
        points = coords[:, [1, 0]]  # (x, y)
        if len(points) > num_points:
            points = farthest_point_sampling(points, num_points)
        elif len(points) < num_points:
            points = np.pad(points, ((0, num_points - len(points)), (0, 0)), 
                            mode='edge')
    
    return resized_binary, points.astype(np.float32)

def downsample_voxel(voxel, target_size=32):
    data = torch.tensor(voxel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    data = F.adaptive_max_pool3d(data, target_size)
    return data.squeeze().numpy().astype(np.uint8)

def preprocess_dataset(sketch_dir, voxel_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    sketch_files = [f for f in os.listdir(sketch_dir) if f.endswith('.png')]
    
    for sketch_file in sketch_files:
        model_id = os.path.splitext(sketch_file)[0]
        sketch_path = os.path.join(sketch_dir, sketch_file)
        binvox_path = os.path.join(voxel_dir, f"{model_id}.binvox")
        
        if not os.path.exists(binvox_path):
            print(f"Missing voxel for: {model_id}")
            continue
        
        # Process sketch
        binary_img, points = sketch_to_binary_and_pointcloud(sketch_path)
        
        # Process voxel
        with open(binvox_path, 'rb') as f:
            model = binvox_rw.read_as_3d_array(f)
        voxel = downsample_voxel(model.data, 32)
        
        # Save results
        np.save(os.path.join(output_dir, f"{model_id}_sketch.npy"), binary_img)
        np.save(os.path.join(output_dir, f"{model_id}_points.npy"), points)
        np.save(os.path.join(output_dir, f"{model_id}_voxel.npy"), voxel)
        
        print(f"Processed: {model_id}")