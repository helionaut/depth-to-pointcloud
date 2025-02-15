# Depth Map to Point Cloud Conversion

This project provides tools for converting depth maps into 3D point clouds, with support for proper camera transformations and multi-view reconstruction.

## Project Structure

- `single_camera_ply.py`: Converts depth maps from individual cameras into a combined point cloud
- `dist2depth.py`: Converts radial distance maps to depth maps
- `stereo_voxel.py`: Processes stereo image pairs for 3D reconstruction with voxelization

## Input Data

The project expects input files in the `input` folder:
- Depth maps: `MetricDepth-VRayCam001.VRayZDepth.0021.exr`, `MetricDepth-VRayCam002.VRayZDepth.0021.exr`
- Color images: `MetricDepth-VRayCam0010021.png`, `MetricDepth-VRayCam0020021.png`

## Camera Parameters

The scripts use the following camera parameters:
- Focal length: 6.0mm
- Film gate (sensor width): 14.186mm
- Camera baseline: 1.549421m
- Tilt angle: 11.0 degrees

## Usage

### Environment Setup

Create and activate a Python virtual environment:

```bash
# Windows (Command Prompt)
python -m venv env
env\Scripts\activate

# Windows (PowerShell)
python -m venv env
.\env\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Converting Depth Maps to Point Cloud

```bash
# Process single camera depth maps and combine into one point cloud
python single_camera_ply.py

# Convert radial distance maps to depth maps
python dist2depth.py

# Process stereo pairs with voxelization
python stereo_voxel.py
```

## Output

- `single_camera_ply.py` generates `combined_scene.ply`
- `dist2depth.py` saves depth maps in the `output` folder
- `stereo_voxel.py` creates `fused_point_cloud.ply`

## Dependencies

Required Python packages (specified in `requirements.txt`):
- numpy
- opencv-python
- matplotlib 