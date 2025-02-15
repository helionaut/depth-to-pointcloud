import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import numpy as np
import cv2
# Define IMWRITE_EXR_TYPE_FLOAT if it doesn't exist
if not hasattr(cv2, 'IMWRITE_EXR_TYPE_FLOAT'):
    cv2.IMWRITE_EXR_TYPE_FLOAT = 1
import matplotlib.pyplot as plt
import glob

# Function to convert radial distances to depth based on camera intrinsics
def convert_radial_to_depth(radial_distance_map: np.ndarray,
                              focal_length_mm: float,
                              film_gate_mm: float,  # Sensor width
                              image_width_px: int,
                              image_height_px: int) -> np.ndarray:
    """
    Converts radial distances to depth based on camera intrinsics.

    Parameters:
        radial_distance_map: np.ndarray - The radial distance map (distance from optical center).
        focal_length_mm: float - Focal length of the camera in millimeters.
        film_gate_mm: float - Sensor width (film gate) in millimeters.
        image_width_px: int - Width of the image in pixels.
        image_height_px: int - Height of the image in pixels.

    Returns:
        np.ndarray - The depth map (distance along the z-axis).
    """
    # Compute the sensor's width and height in mm
    sensor_width_mm = film_gate_mm
    aspect_ratio = image_width_px / image_height_px
    sensor_height_mm = sensor_width_mm / aspect_ratio

    # Create a meshgrid of pixel coordinates (u, v)
    v_coords, u_coords = np.indices((image_height_px, image_width_px))

    # Shift to principal point = (cx, cy)
    cx = image_width_px / 2.0
    cy = image_height_px / 2.0

    # Convert pixels to millimeters on the sensor plane
    x_mm = (u_coords - cx) * (sensor_width_mm / image_width_px)
    y_mm = (v_coords - cy) * (sensor_height_mm / image_height_px)

    # Direction vector magnitudes
    mag = np.sqrt(x_mm**2 + y_mm**2 + focal_length_mm**2)
    cos_theta = focal_length_mm / mag  # cos(theta)

    # Depth = radial_distance_map * cos(theta)
    depth_map = radial_distance_map * cos_theta

    return depth_map

# Define the input folder containing EXR files
input_dir = 'input'
# Glob all EXR files in the directory
exr_files = glob.glob(os.path.join(input_dir, '*.exr'))

if not exr_files:
    print('No EXR files found in the input folder.')
    exit(1)

# Create output folder if it doesn't exist
os.makedirs('output', exist_ok=True)

# Process each EXR file
for exr_path in exr_files:
    input_image = cv2.imread(exr_path, cv2.IMREAD_UNCHANGED)
    if input_image is not None:
        # For EXR files, assume the image is in full precision (float32) and does not require scaling.
        if len(input_image.shape) == 3:
            radial_distance_map = input_image[:, :, 0].astype(np.float32)
        else:
            radial_distance_map = input_image.astype(np.float32)
        
        # Camera parameters (can be adjusted as needed)
        focal_length_mm = 6.0
        film_gate_mm = 14.186
        image_width_px, image_height_px = input_image.shape[1], input_image.shape[0]
        
        # Convert radial distances to depth
        depth_map = convert_radial_to_depth(
            radial_distance_map=radial_distance_map,
            focal_length_mm=focal_length_mm,
            film_gate_mm=film_gate_mm,
            image_width_px=image_width_px,
            image_height_px=image_height_px
        )
        
        # Ensure depth_map is of type float32 before saving
        depth_map = np.float32(depth_map)
        
        # Save the depth map with full accuracy as an EXR file
        base_name = os.path.splitext(os.path.basename(exr_path))[0]
        output_path = os.path.join('output', base_name + '_depth.exr')
        cv2.imwrite(output_path, depth_map, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
        
        # Display the depth map
        plt.figure()
        plt.imshow(depth_map, cmap='gray')
        plt.title(f'Depth Map: {base_name}')
        plt.colorbar()
        plt.axis('off')
        plt.show()
        
        print(f'Depth image saved to: {output_path}')
    else:
        print(f'Failed to load {exr_path}')
