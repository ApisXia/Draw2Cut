# All process + Visualization

- Two pages
- Layout needed: Vis Panel + Control

## Page 2

Referring to "separate_wood_surface.py". Name it as "Step2: Separate Surface"

- Vis Panel: show separated surface
- Control: 
  - Select case
  - origin_label = CONFIG["origin_label"]
  - x_axis_label = CONFIG["x_axis_label"]
  - y_axis_label = CONFIG["y_axis_label"]

## Page 3

Refering to "get_trajectory_Gcode.py". Name it as "Step3: Gen trajectory and Gcode"

- Vis Panel:
  - The same as "Test: Replay"
  - Within the surface region, connect empty space with a surface.
- Control:
  - Select case
  - smooth_size (CONFIG)
  - spindle_radius
  - offset_z_level
  - line_cutting_depth
  - bulk_carving_depth
  - z_expension
