import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import numpy as np
import cv2
import glob
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting


def unproject_depth_pixel(depth_map, focal_length_px):
    """
    Unprojects a depth map (in pixel coordinates) to a point cloud in the camera coordinate system.
    Assumes depth_map gives distance along the optical axis (in meters).
    """
    H, W = depth_map.shape
    cx = W / 2.0
    cy = H / 2.0
    u = np.arange(W)
    v = np.arange(H)
    u_grid, v_grid = np.meshgrid(u, v)
    X = (u_grid - cx) * depth_map / focal_length_px
    Y = (v_grid - cy) * depth_map / focal_length_px
    Z = depth_map
    points = np.stack([X, Y, Z], axis=-1)
    return points.reshape(-1, 3)


def transform_points(points, R, T):
    """
    Applies a rigid body transformation to points.
    points: (N, 3), R: (3,3), T: (3,)
    """
    return (R @ points.T).T + T


def rotation_matrix_y(angle_rad):
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c]
    ])


def rotation_matrix_x(angle_rad):
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])

# Camera intrinsic parameters
focal_length_mm = 6.0
film_gate_mm = 14.186  # sensor width in mm

# Stereo extrinsics parameters
baseline = 1.549421  # meters between the cameras
height = 3.0         # assumed camera height in world coordinates (meters)
angle_deg = 11.0
angle_rad = np.deg2rad(angle_deg)

# For visualization, apply a flip and rotation to the left camera's points
R_flip = rotation_matrix_x(np.pi)  # flip camera coordinate system
R_cam_left = rotation_matrix_y(angle_rad) @ R_flip
T_cam_left = np.array([-baseline/2.0, 0.0, height])

# Load stereo images from the 'input' folder
color_files = glob.glob(os.path.join('input', '*.png'))
color_files.sort()
if len(color_files) < 2:
    print("Need at least 2 color images in the 'input' folder for stereo computation.")
    exit(1)

left_img_path = color_files[0]
right_img_path = color_files[1]

left_img = cv2.imread(left_img_path)
right_img = cv2.imread(right_img_path)
if left_img is None or right_img is None:
    print("Error loading stereo pair.")
    exit(1)

# Optionally downsample images for speed
downsample_scale = 0.5
left_img = cv2.resize(left_img, None, fx=downsample_scale, fy=downsample_scale, interpolation=cv2.INTER_LINEAR)
right_img = cv2.resize(right_img, None, fx=downsample_scale, fy=downsample_scale, interpolation=cv2.INTER_LINEAR)

left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

H, W = left_gray.shape
# Compute focal length in pixels
focal_length_px = (focal_length_mm / film_gate_mm) * W

# Compute disparity map using StereoBM
numDisparities = 16 * 5  # must be divisible by 16
blockSize = 15
stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)

# Compute disparity (note: values are scaled by 16)
disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
# Avoid division by zero
disparity[disparity < 1.0] = 1.0

# Compute depth map using triangulation: depth = (focal_length_px * baseline) / disparity
depth_map = (focal_length_px * baseline) / disparity

# Unproject the left image's depth map to 3D points
points_cam = unproject_depth_pixel(depth_map, focal_length_px)

# Use left image for color
left_img_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
colors = left_img_rgb.reshape(-1, 3)

# Transform points to world coordinates using left camera extrinsics
points_world = transform_points(points_cam, R_cam_left, T_cam_left)

# Voxelization
voxel_size = 0.05  # voxel size in meters
min_bound = points_world.min(axis=0)
max_bound = points_world.max(axis=0)
dims = np.ceil((max_bound - min_bound) / voxel_size).astype(int)
print("Voxel grid dimensions:", dims)

voxel_dict = {}
for pt, col in zip(points_world, colors):
    idx = tuple(((pt - min_bound) / voxel_size).astype(int))
    if idx in voxel_dict:
        voxel_dict[idx]['sum_color'] += col
        voxel_dict[idx]['count'] += 1
        voxel_dict[idx]['sum_pt'] += pt
    else:
        voxel_dict[idx] = {'sum_color': col.astype(np.float32),
                           'count': 1,
                           'sum_pt': pt.copy()}

voxel_centers = []
voxel_colors = []
for idx, data in voxel_dict.items():
    center = data['sum_pt'] / data['count']
    avg_color = data['sum_color'] / data['count']
    voxel_centers.append(center)
    voxel_colors.append(avg_color)

voxel_centers = np.array(voxel_centers)
voxel_colors = np.clip(np.array(voxel_colors).astype(np.uint8), 0, 255)

# For faster visualization, subsample the point cloud
sample_rate = 4
points_sampled = points_world[::sample_rate]
colors_sampled = colors[::sample_rate]

# Visualize the 3D reconstruction as a scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points_sampled[:,0], points_sampled[:,1], points_sampled[:,2], c=colors_sampled/255.0, s=1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title("3D Reconstruction (Displacement/Disparity Approach)")
plt.show()


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
            f.write("{} {} {} {} {} {}\n".format(pt[0], pt[1], pt[2], col[0], col[1], col[2]))

save_ply("fused_point_cloud.ply", voxel_centers, voxel_colors)
print("Saved fused voxelized point cloud to fused_point_cloud.ply") 