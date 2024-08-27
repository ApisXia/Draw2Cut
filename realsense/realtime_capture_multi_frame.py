import os
import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d

from copy import deepcopy

from visualization.QRcode_localization import localize_qr_codes


def create_custom_point_cloud(depth_image, color_image, intrinsic):
    # Get the intrinsic parameters
    fx = intrinsic.intrinsic_matrix[0, 0]
    fy = intrinsic.intrinsic_matrix[1, 1]
    cx = intrinsic.intrinsic_matrix[0, 2]
    cy = intrinsic.intrinsic_matrix[1, 2]

    height, width = depth_image.shape
    points = []
    colors = []

    for v in range(height):
        for u in range(width):
            d = depth_image[v, u]
            # if d > 0:  # You can change this condition to include all depth values
            z = d / 0.001  # Convert depth from meters to millimeters
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            points.append([x, y, z])
            colors.append(color_image[v, u])  # Normalize color values

    points = np.array(points)
    colors = np.array(colors)

    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def fix_depth_image(depth_image, surrounding_size=7):
    # if depth_image is smaller than 0, then need to fix it
    # based on surounding 7x7 pixels, if over 30% of the total pixels are valid (larger than 0), use the dominant value of surrounding 3x3 pixels
    # if not, use the median value of the surrounding 7x7 pixels
    valid_mask = depth_image > 0
    depth_image_copy = deepcopy(depth_image)

    for h in range(depth_image.shape[0]):
        for w in range(depth_image.shape[1]):
            if not valid_mask[h, w]:
                # get the surrounding 7x7 pixels
                surrounding_pixels = []
                surrounding_counter = 0
                for i in range(-surrounding_size // 2, surrounding_size // 2 + 1):
                    for j in range(-surrounding_size // 2, surrounding_size // 2 + 1):
                        if (
                            h + i >= 0
                            and h + i < depth_image.shape[0]
                            and w + j >= 0
                            and w + j < depth_image.shape[1]
                        ):
                            surrounding_counter += 1
                            if valid_mask[h + i, w + j]:
                                surrounding_pixels.append(depth_image[h + i, w + j])
                surrounding_pixels = np.array(surrounding_pixels)
                if len(surrounding_pixels) / surrounding_counter > 0.3:
                    hist, bin_edges = np.histogram(surrounding_pixels, bins="auto")
                    largest_bin_index = np.argmax(hist)
                    largest_bin_mask = (
                        surrounding_pixels >= bin_edges[largest_bin_index]
                    ) & (surrounding_pixels < bin_edges[largest_bin_index + 1])
                    largest_bin_pixels = surrounding_pixels[largest_bin_mask]
                    depth_image_copy[h, w] = np.mean(largest_bin_pixels)
                else:
                    depth_image_copy[h, w] = np.median(surrounding_pixels)
    return depth_image_copy


def show_image(img, name="image"):
    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
    try:
        img = img.reshape((800, 1280, 3))
    except:
        img = img.reshape((800, 1280))
    cv2.imshow(name, (img * 255).astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


QUEUE_SIZE = 50
DEPTH_QUEUE = np.zeros((QUEUE_SIZE, 800, 1280))
DEPTH_VALID_QUEUE = np.zeros((QUEUE_SIZE, 800, 1280))


def depth_queue(
    depth_image,
):
    global DEPTH_QUEUE, DEPTH_VALID_QUEUE  # Declare global variables

    DEPTH_QUEUE = np.roll(DEPTH_QUEUE, 1, axis=0)
    DEPTH_VALID_QUEUE = np.roll(DEPTH_VALID_QUEUE, 1, axis=0)

    DEPTH_QUEUE[0] = depth_image
    DEPTH_VALID_QUEUE[0] = depth_image > 1

    # Calculate the sum of DEPTH_VALID_QUEUE along axis 0
    valid_sum = np.sum(DEPTH_VALID_QUEUE, axis=0)

    # Ensure the minimum value of the sum is 1
    valid_sum = np.maximum(valid_sum, 1)

    # return average depth image using valid mask
    average_depth_image = np.sum(DEPTH_QUEUE * DEPTH_VALID_QUEUE, axis=0) / valid_sum
    return average_depth_image


if __name__ == "__main__":
    case_name = "test0827_5"
    samping_number = 20
    saving_opt = True

    if saving_opt:
        saving_path = os.path.join("data", case_name)
        if os.path.exists(saving_path):
            raise ValueError("The folder already exists")
        os.makedirs(saving_path)

    align = rs.align(rs.stream.color)

    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 800, rs.format.rgb8, 30)
    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    sensor = profile.get_device().query_sensors()[1]
    sensor.set_option(rs.option.exposure, 80)

    # get camera intrinsics
    intr = (
        profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    )
    print(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy
    )
    camera_parameters = np.asarray(pinhole_camera_intrinsic.intrinsic_matrix)

    # define a queue to store the RGBD images
    image_list = []
    sampling_counter = 0
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        if len(image_list) < samping_number:
            image_list.append(color_image)
        elif len(image_list) == samping_number - 1:
            print("Color Image collection is done.")

        # draw the QR code to the color image
        color_image_with_qr = localize_qr_codes(deepcopy(color_image), resize_factor=2)

        depth_frame = aligned_frames.get_depth_frame()
        depth_image = np.asarray(depth_frame.get_data())

        depth_image = depth_queue(depth_image)

        # show both modalities
        depth_image_clipped = np.clip(depth_image, 0, 1800)
        depth_image_display = cv2.normalize(
            depth_image_clipped, None, 0, 255, cv2.NORM_MINMAX
        )
        depth_image_display = np.uint8(depth_image_display)

        combined_image = np.hstack(
            (
                cv2.cvtColor(color_image_with_qr, cv2.COLOR_RGB2BGR),
                cv2.applyColorMap(depth_image_display, cv2.COLORMAP_PLASMA),
            )
        )

        cv2.namedWindow("color and depth image", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("color and depth image", combined_image)
        sampling_counter += 1
        if sampling_counter == QUEUE_SIZE:
            print("!!! Depth queue is full.")

        if cv2.waitKey(1) != -1:
            print("finish")
            break

    if saving_opt:
        print(f"*** Saving point cloud... ***")
        # Convert the scaled NumPy array back to an Open3D image
        depth = o3d.geometry.Image(depth_image.astype(np.float32))
        color = o3d.geometry.Image(color_image)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, convert_rgb_to_intensity=False
        )
        # pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        #     rgbd, pinhole_camera_intrinsic
        # )
        # Assuming rgbd_image is an Open3D RGBDImage object
        depth_image = np.asarray(rgbd.depth)

        # fix the depth image
        fix_times = 3
        for _ in range(fix_times):
            depth_image = fix_depth_image(depth_image)

        color_image = np.asarray(rgbd.color)
        pcd = create_custom_point_cloud(
            depth_image, color_image, pinhole_camera_intrinsic
        )

        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        # pipeline.stop()

        # o3d.visualization.draw_geometries([pcd])

        # save the point cloud to npz file
        points_pos = np.asarray(pcd.points)
        transformed_color = np.asarray(pcd.colors)
        depth = np.asarray(depth)
        # show_image(points_pos, "points_pos")
        # show_image(transformed_color, "transformed_color")
        # show_image(depth, "depth")
        # color_image = cv2.resize(color_image, (1280, 800))
        np.savez(
            os.path.join(saving_path, "point_cloud.npz"),
            points_pos=points_pos,
            transformed_color=transformed_color,
            depth=depth,
            color_image=image_list,
            camera_parameters=camera_parameters,
        )
        print(f"*** Saving point cloud is Done. ***")

    pipeline.stop()
    exit()
