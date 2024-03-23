# color is RGB format, while toleration is HSV range
MARK_TYPES = {
    "contour": {"color": [101, 90, 169], "toleration": [40, 50, 50], "mode": "hsv"},
    "behaviour": {"color": [71, 153, 146], "toleration": [10, 10, 30], "mode": "rgb"},
}

MARK_SAVING_TEMPLATE = "mask_{mark_type_name}_image.png"

WARPPING_RESOLUTION = 1  # mm

FOLDER_PATH = "data/0323/*.npz"
