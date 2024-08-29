import os
import cv2
import numpy as np

from copy import deepcopy
from sklearn.cluster import KMeans

from src.mask.extract_mask import extract_marks_with_colors, draw_extracted_marks

if __name__ == "__main__":
    # visualize the extracted marks folder
    vis_folder = "src/mask/color_cluster_vis"
    os.makedirs(vis_folder, exist_ok=True)

    # clear all image start with color_ and cluster_
    for file in os.listdir(vis_folder):
        if file.startswith("color_") or file.startswith("cluster_"):
            os.remove(os.path.join(vis_folder, file))

    # load image
    image_path = "src/mask/image_color_collection.png"
    if not os.path.exists(image_path):
        raise ValueError("Image not found")
    img = cv2.imread(image_path)

    color_masks_dict = extract_marks_with_colors(img)

    # draw mask for each color
    for i, (color, mask) in enumerate(color_masks_dict.items()):
        result_image = draw_extracted_marks({color: mask}, img.shape)
        cv2.imwrite(os.path.join(vis_folder, f"color_{i}.png"), result_image)
        print(f"{color} saved")

    # using kmeans cluster the colors (key in the dict, bgr) into n clusters
    # split into n dict accordingly, draw separately
    cluster_num = 9  # best is 5 and 7
    colors = list(color_masks_dict.keys())
    # all color transformed to HSV
    colors_hsv = cv2.cvtColor(
        np.uint8(colors).reshape(-1, 1, 3), cv2.COLOR_BGR2HSV
    ).reshape(-1, 3)

    # map hue to x-y plane
    kmeans_trainings = np.zeros((len(colors_hsv), 4))
    kmeans_trainings[:, 0] = np.sin(colors_hsv[:, 0] / 90 * np.pi)
    kmeans_trainings[:, 1] = np.cos(colors_hsv[:, 0] / 90 * np.pi)
    kmeans_trainings[:, 2] = colors_hsv[:, 1] / 255
    kmeans_trainings[:, 3] = colors_hsv[:, 2] / 255

    kmeans = KMeans(n_clusters=cluster_num)
    kmeans.fit(kmeans_trainings)
    labels = kmeans.labels_
    print("labels:", labels)

    # draw a distance matrix between each cluster center
    cluster_centers = kmeans.cluster_centers_
    distance_matrix = np.zeros((cluster_num, cluster_num))
    for i in range(cluster_num):
        for j in range(cluster_num):
            distance_matrix[i, j] = np.linalg.norm(
                cluster_centers[i] - cluster_centers[j]
            )
    # print largest distance
    print("largest distance:", np.max(distance_matrix))
    # print smallest distance expect 0
    print("smallest distance:", np.min(distance_matrix[distance_matrix > 0]))
    # print cluster center one by one
    for i, center in enumerate(cluster_centers):
        # transform back to hsv
        center_hsv = np.zeros(3)
        center_hsv[0] = np.arctan2(center[0], center[1]) / np.pi * 90
        if center_hsv[0] < 0:
            center_hsv[0] += 180
        center_hsv[1] = center[2] * 255
        center_hsv[2] = center[3] * 255

        center_bgr = cv2.cvtColor(np.uint8([[center_hsv]]), cv2.COLOR_HSV2BGR).reshape(
            3
        )
        # print(f"cluster_{i} center in BGR:", center_bgr)
        print(f"cluster_{i} center in HSV:", center_hsv)

    for i in np.unique(labels):
        filtered_color_masks_dict = {
            color: mask
            for j, (color, mask) in enumerate(color_masks_dict.items())
            if labels[j] == i
        }
        result_image = draw_extracted_marks(filtered_color_masks_dict, img.shape)
        cv2.imwrite(os.path.join(vis_folder, f"cluster_{i}.png"), result_image)
        print(f"cluster_{i} saved")
