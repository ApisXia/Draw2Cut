import os
import cv2
import threading
import numpy as np
import pandas as pd
import open3d as o3d

from PyQt5 import QtCore
from copy import deepcopy

from visualization.QRcode_localization import localize_qr_codes


def depth_queue(
    depth_image,
    depth_queue,
    depth_valid_queue,
):
    # roll the depth queue and depth valid queue
    depth_queue = np.roll(depth_queue, 1, axis=0)
    depth_valid_queue = np.roll(depth_valid_queue, 1, axis=0)

    # update the depth queue and depth valid queue
    depth_queue[0] = depth_image
    depth_valid_queue[0] = depth_image > 1

    # calculate the valid sum
    valid_sum = np.sum(depth_valid_queue, axis=0)

    # ensure the valid sum is larger than 1
    valid_sum = np.maximum(valid_sum, 1)

    # calculate the average depth image
    average_depth_image = np.sum(depth_queue * depth_valid_queue, axis=0) / valid_sum
    average_depth_image = average_depth_image.astype(np.uint16)
    return average_depth_image, depth_queue, depth_valid_queue


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


class CaptureThread(QtCore.QThread):
    # define sending signal to update image
    image_updated = QtCore.pyqtSignal(np.ndarray)
    depth_updated = QtCore.pyqtSignal(np.ndarray)
    message_signal = QtCore.pyqtSignal(str, str)

    def __init__(self):
        super(CaptureThread, self).__init__()
        self._stop_event = threading.Event()

        # all these will be updated by the GUI
        self.saving_opt = False
        self.sampling_number = 20
        self.case_name = "default_name"
        self.exposure_level = 140
        self.depth_queue_size = 50
        self.camera_index = 0

        # define a queue to store the depth images
        self.depth_queue = np.zeros((self.depth_queue_size, 800, 1280), dtype=np.uint16)
        self.depth_valid_queue = np.zeros(
            (self.depth_queue_size, 800, 1280), dtype=bool
        )

        # debug setting
        self.use_ordinary_camera = False

    def run(self):
        self.message_signal.emit("Start capturing images...", "info")

        # define the saving path if saving option is on
        if self.saving_opt:
            saving_path = os.path.join("data", self.case_name)

        if self.use_ordinary_camera:
            # 使用 OpenCV 从普通摄像头获取图像
            cap = cv2.VideoCapture(self.camera_index)
            if not cap.isOpened():
                self.message_signal.emit("Cannot open camera.", "error")
                return

            image_list = []

            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    self.message_signal.emit("Cannot receive frame.", "error")
                    break

                # color_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if len(image_list) < self.sampling_number:
                    image_list.append(frame)
                elif len(image_list) == self.sampling_number - 1:
                    print("Color Image collection is done.", "info")

                # add qr code localization
                color_image_with_qr = localize_qr_codes(frame, resize_factor=2)
                fake_depth_image = cv2.cvtColor(color_image_with_qr, cv2.COLOR_BGR2GRAY)

                # send signal to update the image and the depth (fake)
                self.image_updated.emit(color_image_with_qr)
                self.depth_updated.emit(fake_depth_image)

                cv2.waitKey(1)

            cap.release()
            return

        import pyrealsense2 as rs

        # realsense data stream setup
        align = rs.align(rs.stream.color)
        config = rs.config()
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 800, rs.format.rgb8, 30)
        pipeline = rs.pipeline()
        profile = pipeline.start(config)

        sensor = profile.get_device().query_sensors()[1]
        sensor.set_option(rs.option.exposure, self.exposure_level)

        # get camera intrinsics
        intr = (
            profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )

        # [ ] better print format
        print(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)

        # get camera intrinsic parameters for open3d
        o3d_intr = o3d.camera.PinholeCameraIntrinsic(
            intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy
        )

        image_list = []
        sampling_counter = 0

        # stop when the stop event is set
        while not self._stop_event.is_set():
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            if len(image_list) < self.sampling_number:
                image_list.append(color_image)
                # print(f"Image {len(image_list)} is collected.")
            elif len(image_list) == self.sampling_number:
                # [ ] set that to message center later
                print("Color Image collection is done.")
                self.message_signal.emit("Color Image collection is done.", "info")

            # 在彩色图像上绘制 QR 码
            color_image_with_qr = localize_qr_codes(
                deepcopy(color_image), resize_factor=2
            )

            depth_frame = aligned_frames.get_depth_frame()
            depth_image = np.asarray(depth_frame.get_data())

            # get the average depth image
            average_depth_image, self.depth_queue, self.depth_valid_queue = depth_queue(
                depth_image,
                self.depth_queue,
                self.depth_valid_queue,
            )

            # show both modalities
            depth_image_clipped = np.clip(average_depth_image, 0, 1800)
            depth_image_display = cv2.normalize(
                depth_image_clipped, None, 0, 255, cv2.NORM_MINMAX
            )
            depth_image_display = np.uint8(depth_image_display)

            depth_colormap = cv2.applyColorMap(depth_image_display, cv2.COLORMAP_PLASMA)

            self.image_updated.emit(
                cv2.cvtColor(color_image_with_qr, cv2.COLOR_BGR2RGB)
            )
            self.depth_updated.emit(depth_colormap)

            if self.saving_opt and self._stop_event.is_set():
                # 执行保存操作
                print("Saving point cloud...")
                self.message_signal.emit("Saving point cloud...", "info")

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
                pcd = create_custom_point_cloud(depth_image, color_image, o3d_intr)

                pcd.transform(
                    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
                )

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
                )
                print(f"*** Saving point cloud is Done. ***")
                self.message_signal.emit("Saving point cloud is Done.", "info")
                break

        pipeline.stop()

    def stop(self):
        self._stop_event.set()
        self.wait()
