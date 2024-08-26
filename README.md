# WoodCut Project

| Version: 0.1.1 (24.08.26)

- It is the installation guidance of this project.

## 1. Installation

- The environment of this project is managed by `poetry`, install it using the following command after creating environment using `anaconda` or `virenv`.

``` bash
pip install poetry
```

## 2. Scan the wood

- Run the following code to start interface.

``` bash
python realsense/realtime_capture_multi_frame.py
```

- Press `Q` when the the depth map is stable.

## 3. Code running

- Run the following codes in order.

### 3.1 Separate Wood surface

``` bash
python get_trajectory_Gcode.py
```

### 3.2 Get Gcode

``` bash
python get_trajectory_Gcode.py
```

### 3.3 Preview Cutting Trajectory

``` bash
python cut_preview.py
```
