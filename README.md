# Depth Map Conversion Environment

This project converts a radial distance map into a depth map using a set of camera parameters. The main script, `dist2depth.py`, performs the following steps:

1. Reads an input image (expected to contain a radial distance map) from `/mnt/data/0010021_exr.png`.
2. If the image has multiple channels, it extracts the first channel and scales it accordingly.
3. Converts the radial distance to depth by taking into account camera parameters such as focal length, sensor width (film gate), and image dimensions.
4. Normalizes and saves the resulting depth map as a PNG image (`/mnt/data/depth_image.png`).
5. Displays the depth map using matplotlib.

## Environment Setup

### Requirements

The required Python packages are listed in `requirements.txt`:

- numpy
- opencv-python
- matplotlib

### Creating a Virtual Environment

It is recommended to use a Python virtual environment. You can create and activate one using the following commands:

#### Using venv (Python built-in):

For Windows (Command Prompt):

```
python -m venv env
env\Scripts\activate
pip install -r requirements.txt
```

For Windows (PowerShell):

```
python -m venv env
.\env\Scripts\Activate.ps1
pip install -r requirements.txt
```

#### Using Conda:

```
conda create -n depth_env python=3.9
conda activate depth_env
pip install -r requirements.txt
```

## Running the Script

Once the environment is set up and the required packages are installed, you can run the script as follows:

```
python dist2depth.py
```

Ensure that the input image exists at the specified path (`/mnt/data/0010021_exr.png`) or update the script accordingly.

## Notes

- If the input image is not found, the script will print an error message.
- The script uses OpenCV's image reading and writing functions, so make sure the image paths are correctly set up for your system.

Happy coding! 