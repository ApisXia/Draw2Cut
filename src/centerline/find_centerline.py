import cv2
import numpy as np

from skimage.morphology import skeletonize, thin
from scipy.ndimage import gaussian_filter1d


def find_centerline(binary_image):
    binary_image = thin(binary_image // 255, max_num_iter=20).astype(np.uint8) * 255

    # Skeletonize the binary image to get the centerline
    skeleton = skeletonize(binary_image // 255, method="lee").astype(np.uint8) * 255

    # Find contours of the skeleton (centerlines)
    centerline_contours, _ = cv2.findContours(
        skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # remove center line less than 2 points
    centerline_contours = [
        contour for contour in centerline_contours if len(contour) > 2
    ]

    return centerline_contours


def find_centerline_groups(binary_image, connectivity=8):
    # Find connected components (masks)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_image, connectivity, cv2.CV_32S
    )

    all_centerlines = []
    all_masks = []

    # Iterate through each connected component (excluding the background)
    for label in range(1, num_labels):
        mask = np.zeros_like(binary_image, dtype=np.uint8)
        mask[labels == label] = 255

        # Find centerline for the current mask
        centerline_contours = find_centerline(mask)
        all_centerlines.extend(centerline_contours)
        all_masks.append(mask)

    return all_centerlines, all_masks


# add filter to smooth the centerline
def filter_centerlines(centerline_contours, filter_size=5):
    smoothed_centerline_contours = []
    for centerline_contour in centerline_contours:
        sigma = filter_size
        centerline_contour[:, 0, 0] = gaussian_filter1d(
            centerline_contour[:, 0, 0], sigma=sigma
        )
        centerline_contour[:, 0, 1] = gaussian_filter1d(
            centerline_contour[:, 0, 1], sigma=sigma
        )
        smoothed_centerline_contours.append(centerline_contour)
    return smoothed_centerline_contours


# Now just temporary solution, no need currently
def centerline_downsample(centerline_contour, downsample_factor: int = 1):
    downsampled_contour = centerline_contour[::downsample_factor]
    # add the first point to the end to make it a closed loop
    downsampled_contour = np.concatenate(
        [downsampled_contour, downsampled_contour[0:1, ...]], axis=0
    )
    downsampled_contour = downsampled_contour.squeeze(axis=1)
    return downsampled_contour
