import cv2
from glob import glob
import numpy as np

file_path = "data/0329_02/*.npz"
data_list = glob(file_path)

for data_path in data_list:
    # load data
    data = np.load(data_path)
    cv2.imshow("Depth", data["depth_colorized"])
    cv2.imshow("IR", data["ir_colorized"])
    cv2.imshow("Color", data["color"])
    cv2.imshow("Transformed Depth", data["transformed_depth_colorized"])
    cv2.imshow("Transformed Color", data["transformed_color"])
    cv2.imshow("Transformed IR", data["transformed_ir_colorized"])
    cv2.waitKey(10)
cv2.destroyAllWindows()
