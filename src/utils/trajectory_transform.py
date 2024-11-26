import numpy as np

from configs.load_config import CONFIG


def down_scaling_to_real(
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


def vis_points_transformation(
    trajectories: list,
    x_offset: int,
    y_offset: int,
    z_offset: int,
):
    # add the offset to the trajectories
    vis_trajectory = [
        [
            (
                point[0] + x_offset,
                point[1] + y_offset,
                z_offset + point[2] * CONFIG["z_expension"],
            )
            for point in trajectory
        ]
        for trajectory in trajectories
    ]
    # combine list of list to list
    vis_trajectory_combined = [
        point for trajectory in vis_trajectory for point in trajectory
    ]
    vis_trajectory_combined = np.array(vis_trajectory_combined)

    return vis_trajectory_combined, vis_trajectory
