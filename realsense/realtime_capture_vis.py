import os
import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import queue

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
            z = d / 0.001  # Convert depth from mm to meters
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


def show_image(img, name="image"):
    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
    try:
        img = img.reshape((800, 1280, 3))
    except:
        img = img.reshape((800, 1280))
    cv2.imshow(name, (img * 255).astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    case_name = "test0823"
    samping_number = 10
    saving_opt = False

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

    # get camera intrinsics
    intr = (
        profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    )
    print(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy
    )
    camera_parameters = np.asarray(pinhole_camera_intrinsic.intrinsic_matrix)

    # Initialize queues with max size 10
    queue_size = 10
    color_images_queue = queue.Queue(maxsize=queue_size)
    depth_infos_queue = queue.Queue(maxsize=queue_size)

    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        # draw the QR code to the color image
        color_image = localize_qr_codes(color_image, resize_factor=2)

        depth_info = np.asarray(aligned_frames.get_depth_frame().get_data())

        # TODO (not working) add the frames to the queue
        # if not color_images_queue.full() and not depth_infos_queue.full():
        #     color_images_queue.put(color_image)
        #     depth_infos_queue.put(depth_info)
        # else:
        #     # remove the oldest frames
        #     color_images_queue.get()
        #     depth_infos_queue.get()

        #     # add the new frames
        #     color_images_queue.put(color_image)
        #     depth_infos_queue.put(depth_info)

        cv2.namedWindow("color image", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("color image", cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(1) != -1:
            print("finish")
            break

    if not saving_opt:
        pipeline.stop()
        exit()

    case_idx = 0
    while case_idx < samping_number:
        depth_frame = aligned_frames.get_depth_frame()
        depth_data = np.asarray(depth_frame.get_data())

        # Perform the multiplication
        depth_data_scaled = depth_data

        # Convert the scaled NumPy array back to an Open3D image
        depth = o3d.geometry.Image(depth_data_scaled.astype(np.float32))
        color = o3d.geometry.Image(color_image)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, convert_rgb_to_intensity=False
        )
        # pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        #     rgbd, pinhole_camera_intrinsic
        # )
        # Assuming rgbd_image is an Open3D RGBDImage object
        depth_image = np.asarray(rgbd.depth)
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
            os.path.join(saving_path, "point_cloud_" + str(case_idx) + ".npz"),
            points_pos=points_pos,
            transformed_color=transformed_color,
            depth=depth,
            color_image=color_image,
            camera_parameters=camera_parameters,
        )
        print(f"*** Saving point cloud {case_idx} is Done. ***")
        case_idx += 1

    pipeline.stop()
