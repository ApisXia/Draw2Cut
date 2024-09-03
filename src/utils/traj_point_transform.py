import numpy as np

from configs.load_config import CONFIG


def down_sampling_to_real_scale(
    trajectories: list,
):
    sampled_trajectories = [
        [
            (
                point[0] / CONFIG["surface_upscale"],
                point[1] / CONFIG["surface_upscale"],
                point[2],
            )
            for point in trajectory
        ]
        for trajectory in trajectories
    ]
    return sampled_trajectories


def add_x_y_offset(
    trajectories: list,
    x_offset: int,
    y_offset: int,
):
    offseted_trajectories = [
        [(point[0] + x_offset, point[1] + y_offset, point[2]) for point in trajectory]
        for trajectory in trajectories
    ]
    return offseted_trajectories


def vis_points_ransformation(
    trajectories: list,
    x_offset: int,
    y_offset: int,
    z_offset: int,
):
    # add the offset to the trajectories
    vis_trajectories = [
        [
            (
                -(point[0] + x_offset),
                point[1] + y_offset,
                z_offset + CONFIG["offset_z_level"] + point[2] * CONFIG["z_expension"],
            )
            for point in trajectory
        ]
        for trajectory in trajectories
    ]
    # combine list of list to list
    vis_trajectories = [
        point for trajectory in vis_trajectories for point in trajectory
    ]
    vis_trajectories = np.array(vis_trajectories)

    return vis_trajectories
