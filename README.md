
# [CHI2025] Draw2Cut: Direct On-Material Annotations for CNC Milling

**Official Website:** [https://apisxia.github.io/Draw2Cut/](https://apisxia.github.io/Draw2Cut/)

---

## 1. Installation

This project uses `poetry` for environment management. Follow these steps to set up the environment:

1. Create a virtual environment using `Anaconda` or `venv`.

2. Install `poetry` and project dependencies:

   ```bash
   pip install poetry
   poetry install
   ```

### External Dependencies

The project requires the external library `zbar`. Install it using the commands below:

- **For Ubuntu**:
  ```bash
  sudo apt-get install libzbar0
  ```

- **For Mac**:
  ```bash
  brew install zbar
  ```

### Depth Camera Support

The current `pyproject.toml` does not include `pyrealsense2`. If you need depth camera support, install it manually:

- **Note**: This library is not compatible with macOS.
  
  ```bash
  pip install pyrealsense2
  ```

If depth camera functionality is not required, you can skip this step.

### Download Demo Data

- **Download Test Data**
   - Access the demo data from [Google Drive](https://drive.google.com/drive/folders/1EA1aSkaY4Srou1xx2owAVgIeyRJQJpu4?usp=sharing)
   - Download the `point_cloud.npz` file

-  **Setup Data Directory**
   - Create a `data` folder in the project root if it doesn't exist
   - Create a `teatable` subfolder inside `data`
   - Place the downloaded `point_cloud.npz` file at:
     ```
     data/
     └── teatable/
         └── point_cloud.npz
     ```

---

## 2. Interface

Run the following command to launch the interface:

```bash
python main.py
```

### Notes for Ubuntu Users:
- **Library Errors**: Additional libraries may need to be installed if you encounter errors.

- **Display Scaling**: Layout issues may arise if the display scaling is not set to 100%.

---

## 3. Steps for Workflow

### Step 1: Capture Point Cloud

1. Click **Start Capture** to simultaneously capture color and depth images.

2. Switch visualization modes using:
   - **Show Depth**
   - **Show Color Image**

---

### Step 2: Separate Cutting Surface

1. Select the relevant case.

2. Click **Show Collected Point Cloud** to view data from Step 1.

3. Adjust parameters and click **Start Separation** to segment the cutting surface. The result will appear in the left window.

---

### Step 3: Mask & Centerline Extraction

1. Select the relevant case.

2. Click **Start Mask Extraction** to generate color masks. Results will appear in the left window.

3. Review the mask table:
   - Click 💡 to view masks for specific colors.
   - ❌ indicates undetected masks for that color type.

4. Edit parameters in the table:
   - **Type**, **HSV Value**, and **Action** settings will affect future mask extractions.

5. Save the current configuration by clicking **Save Current**.

6. Click **Start Centerline Extraction** to detect centerlines:
   - Use the table to inspect results for each type of centerline.

7. If results are unsatisfactory, adjust the color table and repeat the process.

8. Remember to click **Save to File**.

---

### Step 4: Trajectory, Visualization, & G-code Generation

1. Select the relevant case.
   
2. Click **Start Trajectory Planning** to generate cutting trajectories:
   - View the 2D visualization of coarse trajectories at different cutting levels.

3. Use the visualization options:
   - **Original Mesh**: Display the pre-cut mesh.
   - **Target Mesh**: Display the target mesh.

4. Click **Start Animation** to play the cutting animation:
   - Use **Stop** to halt the animation.

5. Generate G-code:
   - Click **Generate Gcode** to create G-code files for CNC execution.

---

### Additional Notes

Ensure all parameters are fine-tuned during each step for optimal results. Repeat steps as necessary to refine outputs.

---

### Citation

```
@inproceedings{gui2025draw2cut,
  title={Draw2Cut: Direct On-Material Annotations for CNC Milling},
  author={Gui, Xinyue and Xia, Ding and Gao, Wang and Doga Dogan, Mustafa and Larsson, Maria and Igarashi, Takeo},
  booktitle={Proceedings of The ACM conference on Human Factors in Computing Systems (CHI)},
  year={2025},
}
```
