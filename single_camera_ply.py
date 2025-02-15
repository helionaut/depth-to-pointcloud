import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import numpy as np
import cv2
import glob


def convert_radial_to_depth(radial_distance_map, focal_length_px, W, H):
    """
    Convert radial distances to depth (Z coordinate) values.
    Uses the angle between the optical axis and the ray to each pixel.
    """
    cx = W / 2.0
    cy = H / 2.0
    
    # Create pixel coordinate grids
    u = np.arange(W)
    v = np.arange(H)
    u_grid, v_grid = np.meshgrid(u, v)
    
    # Convert pixel coordinates to normalized image coordinates
    x = (u_grid - cx) / focal_length_px
    y = (v_grid - cy) / focal_length_px
    
    # Compute cos(theta) = z/r where r is radial distance
    # At each pixel (x,y), the ray direction is (x,y,1) normalized
    # cos(theta) is the z component of this normalized vector
    cos_theta = 1.0 / np.sqrt(x*x + y*y + 1.0)
    
    # Convert radial distance to depth: Z = r * cos(theta)
    depth_map = radial_distance_map * cos_theta
    return depth_map


def unproject_using_pinhole(depth_map, focal_length_px, W, H):
    """
    Unproject depth map to 3D points using the pinhole camera model.
    depth_map contains Z coordinates (along optical axis).
    """
    cx = W / 2.0
    cy = H / 2.0
    
    # Create pixel coordinate grids
    u = np.arange(W)
    v = np.arange(H)
    u_grid, v_grid = np.meshgrid(u, v)
    
    # Compute X,Y using similar triangles
    X = (u_grid - cx) * depth_map / focal_length_px
    Y = (v_grid - cy) * depth_map / focal_length_px
    Z = depth_map
    
    points = np.stack([X, Y, Z], axis=-1)
    return points.reshape(-1, 3)


def save_ply(filename, points, colors):
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex {}\n".format(points.shape[0]))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for pt, col in zip(points, colors):
            f.write("{} {} {} {} {} {}\n".format(pt[0], pt[1], pt[2], int(col[0]), int(col[1]), int(col[2])))


def rotation_matrix_x(angle_deg):
    """Create rotation matrix around X axis"""
    angle_rad = np.deg2rad(angle_deg)
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])

def rotation_matrix_y(angle_deg):
    """Create rotation matrix around Y axis"""
    angle_rad = np.deg2rad(angle_deg)
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])

def transform_points(points, R, T):
    """Apply rigid transformation to points"""
    return points @ R.T + T


# Camera parameters
focal_length_mm = 6.0
film_gate_mm = 14.186  # sensor width in mm

# Load all EXR depth maps from the 'input' folder (distances already in meters)
exr_files = glob.glob(os.path.join('input', '*.exr'))
if len(exr_files) < 2:
    print("Need at least 2 EXR files in input folder.")
    exit(1)

# Sort the files to ensure consistent order
exr_files.sort()
print("Processing depth maps:", exr_files)

all_points = []
all_colors = []

# Camera extrinsics from 3ds Max (converting from centimeters to meters)
# VRayCam001 transform
R1 = np.array([
    [0.981627, 0, 0.190809],
    [0, 1, 0],
    [-0.190809, 0, 0.981627]
])
T1 = np.array([414.748, -8.19647, 1136.55]) / 100.0  # Convert camera position from cm to m

# VRayCam002 transform
R2 = np.array([
    [0.981627, 0, -0.190809],
    [0, 1, 0],
    [0.190809, 0, 0.981627]
])
T2 = np.array([569.473, -8.19647, 1136.55]) / 100.0  # Convert camera position from cm to m

# Process each depth map
for i, depth_file in enumerate(exr_files[:2]):  # Take first two EXR files
    # Load radial distance map
    radial_distance_map = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
    if radial_distance_map is None:
        print("Cannot load depth map:", depth_file)
        continue
    
    # Keep radial distances in original scale (centimeters)
    radial_distance_map = np.float32(radial_distance_map)
    
    # Load corresponding color image
    base_name = os.path.splitext(os.path.basename(depth_file))[0]
    if ".VRayZDepth." in base_name:
        alt_base = base_name.replace(".VRayZDepth.", "")
        color_file = os.path.join('input', alt_base + '.png')
    else:
        color_file = os.path.join('input', base_name + '.png')
    print("Using color file:", color_file)
    
    if os.path.exists(color_file):
        color_img = cv2.imread(color_file)
        if color_img is None:
            print("Cannot load color image:", color_file)
            continue
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
    else:
        print("Color file not found:", color_file)
        continue
    
    # Get image dimensions
    H, W = radial_distance_map.shape
    
    # Compute focal length in pixels
    focal_length_px = (focal_length_mm / film_gate_mm) * W
    
    # Convert radial distances to depth values
    depth_map = convert_radial_to_depth(radial_distance_map, focal_length_px, W, H)
    
    # Unproject to 3D points
    points = unproject_using_pinhole(depth_map, focal_length_px, W, H)
    
    # Get corresponding colors
    colors = color_img.reshape(-1, 3)
    
    # Apply exact camera transformations from 3ds Max
    if i == 0:  # First camera
        R = R1
        T = T1
    else:  # Second camera
        R = R2
        T = T2
    
    # Transform points to world coordinates
    points = transform_points(points, R, T)
    
    # Filter out points with zero or negative depth
    valid = (depth_map.ravel() > 0)
    points = points[valid]
    colors = colors[valid]
    
    # Add to collection
    all_points.append(points)
    all_colors.append(colors)

# Combine points and colors from both cameras
if len(all_points) > 0:
    combined_points = np.concatenate(all_points, axis=0)
    combined_colors = np.concatenate(all_colors, axis=0)
    
    # Save the combined point cloud
    output_ply = "combined_scene.ply"
    save_ply(output_ply, combined_points, combined_colors)
    print("Saved combined PLY:", output_ply)
else:
    print("No valid point clouds to save.") 