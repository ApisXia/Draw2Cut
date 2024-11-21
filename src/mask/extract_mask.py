import os
import cv2
import json
import numpy as np

from sklearn.cluster import KMeans


def convert_rgb_to_hsv(rgb):
    color_rgb = np.uint8([[list(rgb)]])
    color_hsv = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2HSV)
    return color_hsv[0][0]


def create_mask_HSV(image_path, lower_hsv, upper_hsv):
    img = cv2.imread(image_path)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(img_hsv, lower_hsv, upper_hsv)
    return mask


def create_mask_RGB(image_path, lower_rgb, upper_rgb):
    img = cv2.imread(image_path)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask = cv2.inRange(img_hsv, lower_rgb, upper_rgb)
    return mask


def get_mark_mask(mark_type: dict, image_path: str) -> np.ndarray:
    rgb = mark_type["color"]
    mode = mark_type["mode"]
    color_toleration = mark_type["toleration"]
    if mode == "rgb":
        color = rgb
    elif mode == "hsv":
        color = convert_rgb_to_hsv(rgb)
    else:
        raise ValueError("Invalid mode")

    # define ranges (for extremes, we use 0-255)
    lower_bound = np.array(
        [
            color[0] - color_toleration[0],
            color[1] - color_toleration[1],
            color[2] - color_toleration[2],
        ]
    )
    upper_bound = np.array(
        [
            color[0] + color_toleration[0],
            color[1] + color_toleration[1],
            color[2] + color_toleration[2],
        ]
    )

    # get the mask
    if mode == "rgb":
        img_binary = create_mask_RGB(image_path, lower_bound, upper_bound)
    elif mode == "hsv":
        img_binary = create_mask_HSV(image_path, lower_bound, upper_bound)
    else:
        raise ValueError("Invalid mode")

    # do dilation and erosion
    dilate_erode_iterations = 4
    kernel = np.ones((5, 5), np.uint8)
    img_binary = cv2.dilate(img_binary, kernel, iterations=dilate_erode_iterations)
    img_binary = cv2.erode(img_binary, kernel, iterations=dilate_erode_iterations)

    return img_binary


""" New color extraction method """

PREDEFINED_COLOR_TYPE_VALUES = None
with open("src/mask/color_type_values.json", "r") as f:
    PREDEFINED_COLOR_TYPE_VALUES = json.load(f)

SEMANTIC_COLOR_MATCHING_MIN_DISTANCE = 100


def extract_marks_with_colors(
    image: np.ndarray, min_connect_ratio: float = 0.0001
) -> dict:
    # 转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # # 使用直方图均衡化来增强对比度
    # gray_image = cv2.equalizeHist(gray_image)

    # 使用高斯模糊去噪
    # blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 2)

    # use OTSU thresholding to get binary image with multiple channels
    b_channel, g_channel, r_channel = cv2.split(image)
    _, binary_b = cv2.threshold(
        b_channel, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    _, binary_g = cv2.threshold(
        g_channel, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    _, binary_r = cv2.threshold(
        r_channel, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    binary_image = cv2.bitwise_or(binary_b, binary_g)
    binary_image = cv2.bitwise_or(binary_image, binary_r)

    # binary_image = cv2.adaptiveThreshold(
    #     blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    # )

    # cv2.imwrite("test_binary_image.png", binary_image)
    # assert False

    # 根据连通域分析分割各个圆
    num_labels, labels_im = cv2.connectedComponents(binary_image)

    # 去掉过小的连通域， 占总image的比例小于min_connect_ratio的连通域
    connect_ratio = np.zeros(num_labels)
    for label in range(1, num_labels):
        connect_ratio[label] = np.sum(labels_im == label) / labels_im.size

    # 创建结果字典
    color_masks_dict = {}

    # 为每个mark创建掩码并提取颜色
    for label in range(1, num_labels):
        # 如果连通域的比例小于min_connect_ratio，则跳过
        if connect_ratio[label] < min_connect_ratio:
            continue

        # 创建单个mark的掩码
        single_mark_mask = np.zeros_like(gray_image)
        single_mark_mask[labels_im == label] = 255

        # 提取该圆的颜色部分
        mark_color = cv2.bitwise_and(image, image, mask=single_mark_mask)

        # 将颜色转换为HSV
        hsv_color = cv2.cvtColor(mark_color, cv2.COLOR_BGR2HSV)

        # 仅提取有颜色的部分
        non_zero_mask = single_mark_mask > 0
        hsv_pixels = hsv_color[non_zero_mask]

        # 如果颜色像素数过少，则跳过
        if len(hsv_pixels) < 10:
            continue

        # map hue to x-y plane
        kmeans_trainings = np.zeros((len(hsv_pixels), 4))
        kmeans_trainings[:, 0] = np.sin(hsv_pixels[:, 0] / 90 * np.pi)
        kmeans_trainings[:, 1] = np.cos(hsv_pixels[:, 0] / 90 * np.pi)
        kmeans_trainings[:, 2] = hsv_pixels[:, 1] / 255
        kmeans_trainings[:, 3] = hsv_pixels[:, 2] / 255

        # KMeans聚类
        kmeans = KMeans(n_clusters=min(3, len(kmeans_trainings) // 10))
        kmeans.fit(kmeans_trainings)
        dominant_color = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]

        # transform back to hsv
        dominant_color_hsv = np.zeros(3)
        dominant_color_hsv[0] = (
            np.arctan2(dominant_color[0], dominant_color[1]) / np.pi * 90
        )
        if dominant_color_hsv[0] < 0:
            dominant_color_hsv[0] += 180
        dominant_color_hsv[1] = dominant_color[2] * 255
        dominant_color_hsv[2] = dominant_color[3] * 255

        # 将颜色转换为整数
        dominant_color_hsv = tuple(map(int, dominant_color_hsv))

        # 将颜色转换为BGR
        dominant_bgr_color = cv2.cvtColor(
            np.uint8(dominant_color_hsv).reshape(1, 1, 3), cv2.COLOR_HSV2BGR
        )[0][0]

        # 存储结果
        color_masks_dict[tuple(dominant_bgr_color)] = single_mark_mask

    return color_masks_dict


def draw_extracted_marks(color_masks_dict: dict, image_shape: tuple) -> np.ndarray:
    # 创建一个空白图像
    result_image = np.zeros(image_shape, dtype=np.uint8)

    # 为每个mark上色
    for color, mask in color_masks_dict.items():
        result_image[mask == 255] = color

    return result_image


def find_in_predefined_colors(color_mask_dict: dict) -> dict:
    semantic_color_mask_dict = {}
    for color in color_mask_dict.keys():
        temp_option_dict = {}

        # transform color to hsv
        color_hsv = cv2.cvtColor(np.uint8([[list(color)]]), cv2.COLOR_BGR2HSV).reshape(
            3
        )

        color_hsv_normalized = np.zeros(4)
        color_hsv_normalized[0] = np.sin(color_hsv[0] / 90 * np.pi)
        color_hsv_normalized[1] = np.cos(color_hsv[0] / 90 * np.pi)
        color_hsv_normalized[2] = color_hsv[1] / 255
        color_hsv_normalized[3] = color_hsv[2] / 255

        for predefined_color in PREDEFINED_COLOR_TYPE_VALUES:
            target_color_normalized = np.zeros(4)
            target_color_normalized[0] = np.sin(
                predefined_color["HSV_value"][0] / 90 * np.pi
            )
            target_color_normalized[1] = np.cos(
                predefined_color["HSV_value"][0] / 90 * np.pi
            )
            target_color_normalized[2] = predefined_color["HSV_value"][1] / 255
            target_color_normalized[3] = predefined_color["HSV_value"][2] / 255

            color_distance = np.linalg.norm(
                color_hsv_normalized - target_color_normalized
            )
            if color_distance < predefined_color["tolerance"]:
                temp_option_dict[predefined_color["type"]] = color_distance
        # find closest predefined color index
        if len(temp_option_dict) == 0:
            continue
        closest_color_type = min(temp_option_dict, key=temp_option_dict.get)
        if closest_color_type not in semantic_color_mask_dict.keys():
            semantic_color_mask_dict[closest_color_type] = color_mask_dict[color]
        else:  # merge two masks
            semantic_color_mask_dict[closest_color_type] = cv2.bitwise_or(
                semantic_color_mask_dict[closest_color_type], color_mask_dict[color]
            )
    return semantic_color_mask_dict


if __name__ == "__main__":
    # load image
    image_path = "src/mask/image_color_collection.png"
    if not os.path.exists(image_path):
        raise ValueError("Image not found")
    img = cv2.imread(image_path)

    # # target color
    # for mark_type_name in MARK_TYPES.keys():
    #     img_binary = get_mark_mask(MARK_TYPES[mark_type_name], image_path)

    #     # save the mask
    #     cv2.imwrite(
    #         os.path.join(
    #             images_folder,
    #             MARK_SAVING_TEMPLATE.format(mark_type_name=mark_type_name),
    #         ),
    #         img_binary,
    #     )

    color_masks_dict = extract_marks_with_colors(img)
    colored_masks_img = draw_extracted_marks(color_masks_dict, img.shape)
    cv2.imwrite("colored_masks_img.png", colored_masks_img)

    # find semantic color mask
    semantic_color_mask_dict = find_in_predefined_colors(color_masks_dict)
    semantic_saving_folder = "src/mask/semantic_color_mask_vis"
    os.makedirs(semantic_saving_folder, exist_ok=True)
    for i, (color_type, mask) in enumerate(semantic_color_mask_dict.items()):
        cv2.imwrite(
            os.path.join(semantic_saving_folder, f"semantic_{color_type}.png"), mask
        )
        print(f"{color_type} saved")
