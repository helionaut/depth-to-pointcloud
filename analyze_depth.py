import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import interp1d
from scipy.optimize import minimize

def print_depth_stats(name, depth_data, valid_mask=None):
    if valid_mask is None:
        valid_mask = depth_data > 0
    valid_data = depth_data[valid_mask]
    
    # Calculate percentiles
    percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    percentile_values = np.percentile(valid_data, percentiles)
    
    print(f"\n{name} Statistics:")
    print(f"Shape: {depth_data.shape}")
    print(f"Data type: {depth_data.dtype}")
    print(f"Range: {np.min(valid_data):.3f} to {np.max(valid_data):.3f}")
    print(f"Mean: {np.mean(valid_data):.3f}")
    print(f"Median: {np.median(valid_data):.3f}")
    print(f"Std dev: {np.std(valid_data):.3f}")
    print(f"Integral: {np.sum(valid_data):.3f}")
    print(f"Valid pixels: {np.sum(valid_mask)}")
    print("\nPercentiles:")
    for p, v in zip(percentiles, percentile_values):
        print(f"{p}th percentile: {v:.3f}")
    return valid_data

def optimize_gamma_scale(ai_depth, metric_depth, valid_mask):
    """Optimize gamma and scale to match depth distributions with focus on close distances."""
    ai_valid = ai_depth[valid_mask]
    metric_valid = metric_depth[valid_mask]
    
    # Get metric depth minimum and maximum for valid pixels
    metric_min = np.min(metric_valid)
    metric_max = np.max(metric_valid)
    
    # Create close range mask for integral matching
    close_range_mask = metric_valid < 8.0
    metric_close = metric_valid[close_range_mask]
    ai_close = ai_valid[close_range_mask]
    
    def objective(params):
        gamma, scale = params
        # Apply transformation
        transformed = metric_min + scale * (1.0 - ai_valid) ** gamma
        
        # Calculate close range integral match
        transformed_close = transformed[close_range_mask]
        close_integral_diff = np.abs(np.sum(transformed_close) - np.sum(metric_close))
        
        # Calculate errors with distance-based weights (more weight to closer distances)
        errors = transformed - metric_valid
        weights = np.exp(-0.5 * (metric_valid - metric_min) / 8.0)
        
        # Weighted error terms
        weighted_errors = errors * weights
        mse = np.mean(weighted_errors ** 2)
        
        # Penalize minimum and maximum differences
        min_penalty = np.abs(np.min(transformed) - metric_min) * 1000
        max_penalty = np.abs(np.max(transformed) - metric_max) * 1000
        
        # Final objective: prioritize close range integral match and accuracy
        return close_integral_diff + mse + min_penalty + max_penalty
    
    # Initial guess based on close range statistics
    initial_gamma = 1.5
    initial_scale = (np.max(metric_close) - metric_min) / (1.0 ** initial_gamma)
    
    result = minimize(objective, [initial_gamma, initial_scale], method='Nelder-Mead')
    return result.x, metric_min

# Load the reference metric depth (converted EXR)
metric_depth_path = "output/MetricDepth-VRayCam001.VRayZDepth.0021_depth.exr"
metric_depth = cv2.imread(metric_depth_path, cv2.IMREAD_UNCHANGED)
if len(metric_depth.shape) > 2:
    metric_depth = metric_depth[:,:,0]

print("\n=== Raw Metric Depth Analysis ===")
print(f"Metric depth type: {metric_depth.dtype}")
print(f"Metric depth range before any masking: {np.min(metric_depth):.3f} to {np.max(metric_depth):.3f}")

# Load the AI depth
ai_depth_path = "input/0010021_dadepth.png"
ai_depth = cv2.imread(ai_depth_path, cv2.IMREAD_UNCHANGED)
if len(ai_depth.shape) > 2:
    ai_depth = ai_depth[:,:,0]

print("\n=== Raw AI Depth Analysis ===")
print(f"AI depth type: {ai_depth.dtype}")
print(f"AI depth range before normalization: {np.min(ai_depth)} to {np.max(ai_depth)}")

# Convert AI depth to float32 and normalize to [0,1]
ai_depth = ai_depth.astype(np.float32) / 255.0

# Resize AI depth to match metric depth if needed
if ai_depth.shape != metric_depth.shape:
    print(f"\nResizing AI depth from {ai_depth.shape} to {metric_depth.shape}")
    ai_depth = cv2.resize(ai_depth, (metric_depth.shape[1], metric_depth.shape[0]))

# Get valid depth regions (non-zero)
valid_metric = metric_depth > 0
valid_ai = ai_depth > 0
valid_both = valid_metric & valid_ai

print("\n=== Input Data Analysis ===")
metric_valid = print_depth_stats("Reference Metric Depth (Converted EXR)", metric_depth, valid_both)
ai_valid = print_depth_stats("Input AI Depth (Normalized PNG)", ai_depth, valid_both)

# Optimize gamma and scale parameters
(gamma, scale), metric_min = optimize_gamma_scale(ai_depth, metric_depth, valid_both)
print(f"\nOptimized parameters:")
print(f"Gamma: {gamma:.3f}")
print(f"Scale: {scale:.3f}")
print(f"Minimum depth: {metric_min:.3f}")

# Apply non-linear transformation
ai_depth_metric = np.zeros_like(ai_depth)
ai_depth_metric[valid_both] = metric_min + scale * (1.0 - ai_depth[valid_both]) ** gamma

print("\n=== New Output Analysis ===")
new_output_valid = print_depth_stats("New Output Depth", ai_depth_metric, valid_both)

# Compute error metrics against reference metric depth
diff = np.abs(metric_depth - ai_depth_metric)
error_percent = (diff[valid_both] / metric_valid) * 100

# Calculate separate error metrics for close and far regions
close_mask = (metric_depth < 8.0) & valid_both
far_mask = (metric_depth >= 8.0) & valid_both

# Calculate close range integrals for comparison
metric_close_integral = np.sum(metric_depth[close_mask])
ai_close_integral = np.sum(ai_depth_metric[close_mask])
close_integral_diff = np.abs(metric_close_integral - ai_close_integral)
close_integral_diff_percent = (close_integral_diff / metric_close_integral) * 100

print(f"\n=== Close Range Integral Analysis ===")
print(f"Metric Close Range Integral: {metric_close_integral:.3f}")
print(f"AI Close Range Integral: {ai_close_integral:.3f}")
print(f"Close Range Integral Difference: {close_integral_diff:.3f}")
print(f"Close Range Integral Difference (%): {close_integral_diff_percent:.3f}%")

print(f"\n=== Error Analysis ===")
print(f"Overall Metrics:")
print(f"Mean Absolute Error: {np.mean(diff[valid_both]):.3f}")
print(f"Median Absolute Error: {np.median(diff[valid_both]):.3f}")
print(f"Mean Relative Error: {np.mean(error_percent):.3f}%")
print(f"Median Relative Error: {np.median(error_percent):.3f}%")
print(f"Max Absolute Error: {np.max(diff[valid_both]):.3f}")
print(f"90th percentile Error: {np.percentile(diff[valid_both], 90):.3f}")

print(f"\nClose Range (<8m) Metrics:")
print(f"Mean Absolute Error: {np.mean(diff[close_mask]):.3f}")
print(f"Median Absolute Error: {np.median(diff[close_mask]):.3f}")
print(f"Mean Relative Error: {np.mean((diff[close_mask] / metric_depth[close_mask]) * 100):.3f}%")

print(f"\nFar Range (≥8m) Metrics:")
print(f"Mean Absolute Error: {np.mean(diff[far_mask]):.3f}")
print(f"Median Absolute Error: {np.median(diff[far_mask]):.3f}")
print(f"Mean Relative Error: {np.mean((diff[far_mask] / metric_depth[far_mask]) * 100):.3f}%")

# Plot comparison with consistent color ranges
plt.figure(figsize=(20, 5))

vmin = min(np.min(metric_depth[valid_metric]), np.min(ai_depth_metric[valid_both]))
vmax = max(np.max(metric_depth[valid_metric]), np.max(ai_depth_metric[valid_both]))

plt.subplot(141)
plt.imshow(metric_depth, cmap='viridis', vmin=vmin, vmax=vmax)
plt.colorbar()
plt.title('Reference Metric Depth')

plt.subplot(142)
plt.imshow(ai_depth, cmap='viridis')
plt.colorbar()
plt.title('Input AI Depth (Normalized)')

plt.subplot(143)
plt.imshow(ai_depth_metric, cmap='viridis', vmin=vmin, vmax=vmax)
plt.colorbar()
plt.title('Converted AI Depth')

plt.subplot(144)
plt.imshow(diff, cmap='viridis')
plt.colorbar()
plt.title('Absolute Difference')

plt.tight_layout()
plt.savefig('depth_comparison.png')
plt.close()

# Plot detailed analysis
plt.figure(figsize=(20, 10))

plt.subplot(231)
plt.hist(metric_valid[metric_valid < 8], bins=50, alpha=0.5, label='Reference Metric (<8m)', density=True)
plt.hist(new_output_valid[new_output_valid < 8], bins=50, alpha=0.5, label='Converted AI (<8m)', density=True)
plt.axvline(x=8, color='r', linestyle='--', label='8m threshold')
plt.legend()
plt.title('Close Range Depth Distribution')

plt.subplot(232)
plt.scatter(metric_valid[::100], new_output_valid[::100], alpha=0.1, s=1, c=metric_valid[::100], cmap='viridis')
plt.plot([vmin, vmax], [vmin, vmax], 'r--', label='Ideal')
plt.axvline(x=8, color='r', linestyle='--', label='8m threshold')
plt.xlabel('Reference Metric Depth')
plt.ylabel('Converted AI Depth')
plt.legend()
plt.title('Depth Correlation')

plt.subplot(233)
close_errors = error_percent[metric_valid < 8]
far_errors = error_percent[metric_valid >= 8]
plt.hist(close_errors, bins=50, alpha=0.5, label='<8m', density=True)
plt.hist(far_errors, bins=50, alpha=0.5, label='≥8m', density=True)
plt.legend()
plt.title('Relative Error Distribution (%)')
plt.xlabel('Error %')

# Plot the non-linear transformation curve
plt.subplot(234)
x = np.linspace(0, 1, 1000)
y = metric_min + scale * (1.0 - x) ** gamma
plt.plot(x, y)
plt.axhline(y=8, color='r', linestyle='--', label='8m threshold')
plt.title(f'Non-linear Transform (γ={gamma:.3f}, scale={scale:.3f})')
plt.xlabel('Input AI Depth')
plt.ylabel('Output Metric Depth')
plt.legend()
plt.grid(True)

plt.subplot(235)
row_middle = metric_depth.shape[0] // 2
plt.plot(metric_depth[row_middle, :], label='Reference Metric')
plt.plot(ai_depth_metric[row_middle, :], label='Converted AI')
plt.axhline(y=8, color='r', linestyle='--', label='8m threshold')
plt.legend()
plt.title('Middle Row Profile')

plt.subplot(236)
col_middle = metric_depth.shape[1] // 2
plt.plot(metric_depth[:, col_middle], label='Reference Metric')
plt.plot(ai_depth_metric[:, col_middle], label='Converted AI')
plt.axhline(y=8, color='r', linestyle='--', label='8m threshold')
plt.legend()
plt.title('Middle Column Profile')

plt.tight_layout()
plt.savefig('depth_analysis.png')
plt.close()

# Save the converted AI depth
cv2.imwrite('output/ai_depth_metric.exr', ai_depth_metric.astype(np.float32), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT]) 