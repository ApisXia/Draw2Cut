import cv2
import numpy as np
from pyzbar import pyzbar


def get_qr_codes_position(image, resize_factor=2):
    # Resize the image to a larger size
    resized_image = image.astype(np.uint8)
    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(resized_image, None, fx=resize_factor, fy=resize_factor)

    # binarize the image
    binarized_image = cv2.threshold(
        resized_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )[1]

    # Find QR codes and their information
    qr_codes = pyzbar.decode(binarized_image, symbols=[pyzbar.ZBarSymbol.QRCODE])

    qr_code_dict = {}

    # Iterate over the QR codes
    for qr_code in qr_codes:
        # Extract the bounding box coordinates
        (x, y, w, h) = qr_code.rect

        # Scale the coordinates back to the original image size
        x = x // resize_factor
        y = y // resize_factor
        w = w // resize_factor
        h = h // resize_factor

        # Find the center of the QR code
        center_x = int(x + (w // 2))
        center_y = int(y + (h // 2))

        # x,y need to be int
        x = int(x)
        y = int(y)

        # Get the information from the QR code
        qr_code_data = qr_code.data.decode("utf-8")

        # only use number in this sentence
        qr_code_data = str(qr_code_data.split(" ")[-1])

        qr_code_dict[qr_code_data] = (center_x, center_y)

    return qr_code_dict
