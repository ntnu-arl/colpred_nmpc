import numpy as np
import cv2


## define morphology kernels
FULL_KERNEL_5 = np.ones((5, 5), np.uint8)
FULL_KERNEL_7 = np.ones((7, 7), np.uint8)
FULL_KERNEL_100 = np.ones((100, 100), np.uint8)
DIAMOND_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)


def fill_in_fast(depth_map, max_depth=1.1, extrapolate=False, blur_type='bilateral'):
    """Fast depth completion.
    Args:
        depth_map: projected depths
        max_depth: max depth value for inversion
        extrapolate: whether to extrapolate by extending depths to top of
            the frame, and applying a 31x31 full kernel dilation
        blur_type:
            'bilateral' - preserves local structure (recommended)
            'gaussian' - provides lower RMSE
    Returns:
        depth_map: dense depth map
    """
    dmin = 0.01

    ## invert
    valid_pixels = depth_map > dmin
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    ## dilate
    depth_map = cv2.dilate(depth_map, DIAMOND_KERNEL_7)

    ## hole closing
    depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, FULL_KERNEL_5)

    # Fill empty spaces with dilated values
    empty_pixels = depth_map < dmin
    dilated = cv2.dilate(depth_map, FULL_KERNEL_7)
    depth_map[empty_pixels] = dilated[empty_pixels]

    # Extend pixels to top and left of image
    if extrapolate:
        # ## extend top
        # top_row_pixels = np.argmax(depth_map > dmin, axis=0)
        # top_pixel_values = depth_map[top_row_pixels, range(depth_map.shape[1])]
        # for pixel_col_idx in range(depth_map.shape[1]):
        #     depth_map[0:top_row_pixels[pixel_col_idx], pixel_col_idx] = top_pixel_values[pixel_col_idx]

        ## extend left
        left_col_pixels = np.argmax(depth_map > dmin, axis=1)
        left_pixel_values = depth_map[range(depth_map.shape[0]), left_col_pixels]
        for pixel_row_idx in range(depth_map.shape[0]):
            depth_map[pixel_row_idx, 0:left_col_pixels[pixel_row_idx]] = left_pixel_values[pixel_row_idx]

        # Large Fill
        empty_pixels = depth_map < dmin
        dilated = cv2.dilate(depth_map, FULL_KERNEL_100)
        depth_map[empty_pixels] = dilated[empty_pixels]

    ## blurs
    ## median blur
    depth_map = cv2.medianBlur(depth_map, 5)

    if blur_type == 'bilateral':
        ## bilateral blur
        depth_map = cv2.bilateralFilter(depth_map, 5, 1.5, 2.0)

    elif blur_type == 'gaussian':
        ## gaussian blur
        valid_pixels = depth_map > dmin
        blurred = cv2.GaussianBlur(depth_map, (5, 5), 0)
        depth_map[valid_pixels] = blurred[valid_pixels]

    ## invert
    valid_pixels = depth_map > dmin
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    return depth_map
