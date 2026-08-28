
#Based ono roboflow blogspot:
#   https://blog.roboflow.com/computer-vision-camera-focus-guide/

import cv2
import numpy as np


def variance_of_laplacian(img) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()

    return variance


def brenner(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert image to 16-bit integer format
    converted_image = gray.astype(np.int16)

    # Get the dimensions of the image
    height, width = converted_image.shape

    # Initialize two matrices for horizontal and vertical focus measures
    horizontal_diff = np.zeros((height, width))
    vertical_diff = np.zeros((height, width))

    # Calculate horizontal and vertical focus measures
    horizontal_diff[:, : width - 2] = np.clip(
        converted_image[:, 2:] - converted_image[:, :-2], 0, None
    )
    vertical_diff[: height - 2, :] = np.clip(
        converted_image[2:, :] - converted_image[:-2, :], 0, None
    )

    # Calculate final focus measure
    focus_measure = np.max((horizontal_diff, vertical_diff), axis=0) ** 2

    return focus_measure.mean()


def tenengrad(img) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
    magnitude = cv2.magnitude(sx, sy)

    return magnitude.mean()
