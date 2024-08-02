import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d

if __name__ == "__main__":
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

    frame_count = 0
    max_frame = 100

    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        depth_info = np.asarray(aligned_frames.get_depth_frame().get_data())

        cv2.namedWindow("color image", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("color image", cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))

        # wait for 100ms
        cv2.waitKey(100)

        depth_frame = aligned_frames.get_depth_frame()
        depth = o3d.geometry.Image(np.asanyarray(depth_frame.get_data()))
        color = o3d.geometry.Image(color_image)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, convert_rgb_to_intensity=False
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, pinhole_camera_intrinsic
        )

        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        # save the point cloud to npz file
        points_pos = np.asarray(pcd.points)
        transformed_color = np.asarray(pcd.colors)
        depth = np.asarray(depth)
        np.savez(
            f"data/test/point_cloud_{frame_count}.npz",
            points_pos=points_pos,
            transformed_color=transformed_color,
            depth=depth,
            color_image=color_image,
        )

        frame_count += 1
        print(f"frame count: {frame_count}")
        if frame_count == max_frame:
            break

    pipeline.stop()
