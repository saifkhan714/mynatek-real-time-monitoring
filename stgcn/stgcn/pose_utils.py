import numpy as np

def normalize_points_with_size(xy, width, height, flip=False):
    """Normalize scale points in image with size of image to (0-1)."""
    # Ensure proper dimensions
    if xy.ndim == 2:
        xy = np.expand_dims(xy, 0)  # Add batch dimension

    # Create a copy to avoid modifying the original
    xy_normalized = xy.copy()

    # Normalize x and y coordinates by width and height
    xy_normalized[:, :, 0] = xy_normalized[:, :, 0] / width
    xy_normalized[:, :, 1] = xy_normalized[:, :, 1] / height

    if flip:
        xy_normalized[:, :, 0] = 1 - xy_normalized[:, :, 0]

    return xy_normalized

def scale_pose(xy):
    """Normalize pose points by scale with max/min value of each pose."""
    # Ensure proper dimensions
    if xy.ndim == 2:
        xy = np.expand_dims(xy, 0)  # Add batch dimension

    # Create a copy to avoid modifying the original
    xy_scaled = xy.copy()

    for i in range(xy.shape[0]):
        # Try to find valid min and max, avoiding NaN
        xy_min = np.nanmin(xy[i], axis=0)
        xy_max = np.nanmax(xy[i], axis=0)

        # Check for invalid min/max (all NaN)
        if np.any(np.isnan(xy_min)) or np.any(np.isnan(xy_max)):
            continue

        # Check for division by zero
        range_xy = xy_max - xy_min
        if np.any(np.isclose(range_xy, 0)):
            continue

        # Apply scaling
        xy_scaled[i] = 2 * (xy[i] - xy_min) / range_xy - 1

    if xy.ndim == 3:
        return xy_scaled
    else:
        return xy_scaled.squeeze()