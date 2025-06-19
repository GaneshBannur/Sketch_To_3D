# MonoSketch3D: A Dual-Channel Framework for 3D Shape Generation from Hand-Drawn Single-View Sketches

This repository contains the official implementation of **MonoSketch3D**, a novel framework for 3D voxel generation from single-view sketches. Our BCwrite in markdown codemarkdown# MonoSketch3D: A Dual-Channel Framework for 3D Shape Generation from Hand-Drawn Single-View Sketches

This repository contains the official implementation of **MonoSketch3D**, a novel framework for 3D voxel generation from single-view sketches. Our approach introduces a dual-channel processing design that extracts both pixel-wise and point-wise features from sketches, achieving a 23% improvement over the baseline Pix2Vox architecture.

## Overview

MonoSketch3D addresses the challenging problem of sketch-based 3D reconstruction by:
- Processing sketches as both 2D raster images and unordered 2D point clouds
- Introducing complementary SktConv and SktPoint modules for feature extraction
- Employing a Cross-Feature Attention Module (CFAM) to fuse multi-modal features
- Achieving superior reconstruction quality across varying sketch styles and viewpoints

## Datasets

We use a custom **Shapenet-Sketch** dataset created from ShapeNet for our experiments. The dataset contains 12,138 valid samples across 9 object categories.

### Dataset Downloads
- **ShapeNet rendering images**: http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz
- **ShapeNet voxelized models**: http://cvgl.stanford.edu/data2/ShapeNetVox32.tgz
- **Our processed Shapenet-Sketch dataset**: Available at [Google Drive](https://drive.google.com/file/d/1Bo_m4b2inA3AuGibV9R8S_sIZTOiE12V/view?usp=sharing)

### Dataset Structure
The processed dataset includes:
- **Sketch images** (.png): Edge-based representations generated using Photo-Sketching algorithm
- **3D voxel grids** (.npy): 32×32×32 voxel representations
- **2D point clouds** (.npy): Depth-aware sampled points with (x,y,z) coordinates

### Object Categories
- Aeroplane, Bench, Cabinet, Chair, Display, Lamp, Speaker, Sofa, Table

## Pretrained Models
*To be released*

## Prerequisites

### Clone the Code Repository
```bash
git clone https://github.com/GaneshBannur/Sketch_To_3D.git
cd Sketch_To_3D
```

### Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Key Dependencies
- Python 3.8+
- PyTorch 1.12+
- CUDA 11.6+ (for GPU acceleration)
- NumPy
- OpenCV
- Matplotlib
- Weights & Biases (for experiment tracking)
- Accelerate (for mixed precision training)

### Hardware Requirements
- **Training**: NVIDIA A100 GPU (or equivalent with 24GB+ VRAM)
- **Inference**: NVIDIA GTX 1080 or better (8GB+ VRAM recommended)

### Installation

Install PyTorch (with CUDA support):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Install additional dependencies:
```bash
pip install opencv-python matplotlib wandb accelerate
pip install numpy scipy scikit-image
```

Install DepthAnythingV2 (for depth estimation):
```bash
pip install depth-anything-v2
```

## Configuration

### Update Settings in `config.py`

Update the file paths for the datasets:
```python
# Shapenet-Sketch Dataset Paths
__C.DATASETS.SHAPENET_SKETCH.SKETCH_PATH = '/path/to/Datasets/Shapenet-Sketch/sketches/%s/%s.png'
__C.DATASETS.SHAPENET_SKETCH.VOXEL_PATH = '/path/to/Datasets/Shapenet-Sketch/voxels/%s/%s_voxel.npy'
__C.DATASETS.SHAPENET_SKETCH.POINTS_PATH = '/path/to/Datasets/Shapenet-Sketch/points/%s/%s_points.npy'

# Original ShapeNet Paths (if using raw data)
__C.DATASETS.SHAPENET.RENDERING_PATH = '/path/to/Datasets/ShapeNet/ShapeNetRendering/%s/%s/rendering/%02d.png'
__C.DATASETS.SHAPENET.VOXEL_PATH = '/path/to/Datasets/ShapeNet/ShapeNetVox32/%s/%s/model.binvox'

# Model Configuration
__C.MODEL.DUAL_CHANNEL = True
__C.MODEL.USE_CFAM = True
__C.MODEL.VOXEL_SIZE = 32
```

### Training Configuration
```python
# Training Parameters
__C.TRAIN.BATCH_SIZE = 256
__C.TRAIN.LEARNING_RATE = 0.001
__C.TRAIN.NUM_EPOCHS = 100
__C.TRAIN.MIXED_PRECISION = True

# Optimizer Settings
__C.TRAIN.OPTIMIZER = 'Adam'
__C.TRAIN.BETA1 = 0.9
__C.TRAIN.BETA2 = 0.999
```

## Get Started

### Training MonoSketch3D
To train MonoSketch3D from scratch:
```bash
python3 runner.py --train --config configs/monosketch3d.yaml
```

To train with custom settings:
```bash
python3 runner.py --train --config configs/monosketch3d.yaml --batch_size 128 --lr 0.0005
```

To resume training from checkpoint:
```bash
python3 runner.py --train --resume --weights=/path/to/checkpoint.pth
```

### Testing MonoSketch3D
To test with pretrained model:
```bash
python3 runner.py --test --weights=/path/to/pretrained/monosketch3d.pth
```

To test on specific categories:
```bash
python3 runner.py --test --weights=/path/to/pretrained/monosketch3d.pth --categories chair,table,sofa
```

### Inference on Custom Sketches
To reconstruct 3D models from your own sketches:
```bash
python3 inference.py --input /path/to/sketch.png --output /path/to/output/ --weights /path/to/pretrained/monosketch3d.pth
```

### Batch Processing
To process multiple sketches:
```bash
python3 batch_inference.py --input_dir /path/to/sketches/ --output_dir /path/to/outputs/ --weights /path/to/pretrained/monosketch3d.pth
```
