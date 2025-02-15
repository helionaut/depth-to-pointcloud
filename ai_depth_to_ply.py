import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import numpy as np
import cv2

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

# Camera parameters (matching those from single_camera_ply.py)
focal_length_mm = 6.0
film_gate_mm = 14.186  # sensor width in mm

# Camera extrinsics from 3ds Max (converting from centimeters to meters)
# VRayCam001 transform
R = np.array([
    [0.981627, 0, 0.190809],
    [0, 1, 0],
    [-0.190809, 0, 0.981627]
])
T = np.array([414.748, -8.19647, 1136.55]) / 100.0  # Convert camera position from cm to m

# Load the AI depth metric EXR
depth_file = "output/ai_depth_metric.exr"
depth_map = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
if depth_map is None:
    print(f"Cannot load depth map: {depth_file}")
    exit(1)

if len(depth_map.shape) > 2:
    depth_map = depth_map[:,:,0]

# Load the corresponding color image
color_file = "input/MetricDepth-VRayCam0010021.png"
color_img = cv2.imread(color_file)
if color_img is None:
    print(f"Cannot load color image: {color_file}")
    exit(1)
color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

# Get image dimensions
H, W = depth_map.shape

# Compute focal length in pixels
focal_length_px = (focal_length_mm / film_gate_mm) * W

# Unproject to 3D points
points = unproject_using_pinhole(depth_map, focal_length_px, W, H)

# Get corresponding colors
colors = color_img.reshape(-1, 3)

# Transform points to world coordinates
points = points @ R.T + T

# Filter out points with zero or negative depth
valid = (depth_map.ravel() > 0)
points = points[valid]
colors = colors[valid]

# Save the point cloud
output_ply = "ai_depth_pointcloud.ply"
save_ply(output_ply, points, colors)
print(f"Saved PLY file: {output_ply}")

# Print some statistics
print("\nPoint Cloud Statistics:")
print(f"Total points: {points.shape[0]}")
print(f"X range: {points[:,0].min():.3f} to {points[:,0].max():.3f}")
print(f"Y range: {points[:,1].min():.3f} to {points[:,1].max():.3f}")
print(f"Z range: {points[:,2].min():.3f} to {points[:,2].max():.3f}") 