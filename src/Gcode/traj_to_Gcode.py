from configs.load_config import CONFIG


def generate_gcode(
    coarse_trajectories,
    fine_trajectories,
    ultra_fine_trajectories,
    z_surface_level,
    stop_at_end=False,
):
    lefting_distance = 15  # it is millimeters

    gcode = ""

    # Set initial conditions
    gcode += "G20 ; Set units to inches\n"  # Use G20 for inches, G21 for millimeters
    gcode += "G90 ; Set to absolute positioning\n"
    gcode += f"G0 X0.0000 Y0.0000 Z{(z_surface_level+lefting_distance)*0.03937:.4f} ; Move spindle to start position\n"
    # gcode += f"S{spindle_speed} ; Set spindle speed\n"

    # Set feed rate
    coarse_feed_rate = CONFIG["feed_rate"]["coarse"]
    fine_feed_rate = CONFIG["feed_rate"]["fine"]

    # Iterate and generate G-code for coarse cutting
    gcode += traj2codes(
        coarse_trajectories, z_surface_level, lefting_distance, coarse_feed_rate
    )
    # Iterate and generate G-code for fine cutting
    gcode += traj2codes(
        fine_trajectories, z_surface_level, lefting_distance, fine_feed_rate
    )
    # Iterate and generate G-code for ultra fine cutting
    gcode += traj2codes(
        ultra_fine_trajectories, z_surface_level, lefting_distance, fine_feed_rate
    )

    if not stop_at_end:
        # Move spindle to start position
        gcode += f"G0 X0.0000 Y0.0000 Z{(z_surface_level+lefting_distance)*0.03937:.4f} ; Move spindle to start position\n"
    gcode += "M30 ; End of program\n"
    return gcode


def traj2codes(trajs, z_surface_level, lefting_distance, feed_rate):
    gcode = ""
    # Iterate and generate G-code for fine cutting
    for trajectory in trajs:
        # Move spindle to start point of trajectory
        start_point = trajectory[0]
        y, x, z = start_point
        gcode += f"G0 X{x*0.03937:.4f} Y{y*0.03937:.4f} Z{(z_surface_level+lefting_distance)*0.03937:.4f} ; Move spindle to start point\n"

        # Move spindle down to the z_surface_level - defined depth
        gcode += f"G1 Z{(z_surface_level + z)*0.03937:.4f} F{feed_rate} ; Move spindle down\n"

        for point in trajectory:
            y, x, z = point
            gcode += f"G1 X{x*0.03937:.4f} Y{y*0.03937:.4f} Z{(z_surface_level + z)*0.03937:.4f} F{feed_rate} ; Move to next point\n"

        # Move spindle up to the z_surface_level + z_surface_level
        gcode += f"G1 Z{(z_surface_level+lefting_distance)*0.03937:.4f} F{feed_rate} ; Move spindle up\n"

    return gcode
