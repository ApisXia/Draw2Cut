# color is RGB format, while toleration is HSV range
MARK_TYPES = {
    "contour": {"color": [108, 92, 104], "toleration": [15, 15, 20], "mode": "hsv"},
    "behaviour": {"color": [1, 1, 1], "toleration": [50, 50, 50], "mode": "rgb"},
}

MARK_SAVING_TEMPLATE = "mask_{mark_type_name}_image.png"

WARPPING_RESOLUTION = 1  # mm

FOLDER_PATH = "data/colortest/*.npz"

SURFACE_UPSCALE = 10
ROW_INTERVAL = 5

# temp
GT_X_LENGTH = 259
GT_Y_LENGTH = 258
