from PIL import Image
import numpy as np
import cv2


def cv2_to_pil(img: np.ndarray) -> Image.Image:
    cv_frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_frame = Image.fromarray(cv_frame_rgb)
    return pil_frame


def pil_to_cv2(img: Image.Image) -> np.ndarray:
    pil_data = img.convert('RGB')
    return np.array(pil_data)[:, :, ::-1]
