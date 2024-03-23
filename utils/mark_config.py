# color is RGB format, while toleration is HSV range
MARK_TYPES = {
    "contour": {"color": [105, 94, 176], "toleration": [40, 90, 90], "mode": "hsv"},
    "behaviour": {"color": [229, 124, 134], "toleration": [10, 10, 30], "mode": "rgb"},
}

MARK_SAVING_TEMPLATE = "mask_{mark_type_name}_image.png"

WARPPING_RESOLUTION = 1  # mm

FOLDER_PATH = "data/0323_black/*.npz"

SURFACE_UPSCALE = 10
