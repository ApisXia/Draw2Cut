# color is RGB format, while toleration is HSV range
MARK_TYPES = {
    "contour": {"color": [99, 93, 162], "toleration": [50, 50, 50], "mode": "hsv"},
    "behaviour": {"color": [210, 97, 110], "toleration": [50, 50, 50], "mode": "rgb"},
}

MARK_SAVING_TEMPLATE = "mask_{mark_type_name}_image.png"

WARPPING_RESOLUTION = 1  # mm

FOLDER_PATH = "data/0330_02/*.npz"

SURFACE_UPSCALE = 10
ROW_INTERVAL = 5

# temp
GT_X_LENGTH = 259
GT_Y_LENGTH = 258
