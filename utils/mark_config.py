# color is RGB format, while toleration is HSV range
MARK_TYPES = {
    "contour": {"color": [41, 27, 125], "toleration": [25, 70, 70]},
    "annotation": {"color": [184, 54, 78], "toleration": [25, 70, 70]},
}

MARK_SAVING_TEMPLATE = "mask_{mark_type_name}_image.png"
