import cv2
import numpy as np

from config import (
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    CROP_TOP,
    V_MIN,
    V_MAX,
    W_SCALE
)


def preprocess_image(image_rgb):

    if image_rgb is None:
        raise ValueError("Imagen invalida")

    if image_rgb.shape != (480, 640, 3):
        raise ValueError(
            f"Resolucion inesperada: {image_rgb.shape}"
        )

    image = image_rgb[CROP_TOP:, :, :]

    image = cv2.resize(
        image,
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        interpolation=cv2.INTER_AREA
    )

    image = image.astype(np.float32) / 255.0

    image = np.transpose(image, (2, 0, 1))

    return np.ascontiguousarray(image)


def normalize_labels(v, w):

    return np.array([
        (v - V_MIN) / (V_MAX - V_MIN),
        w / W_SCALE
    ], dtype=np.float32)


def denormalize_labels(v_norm, w_norm):

    v = v_norm * (V_MAX - V_MIN) + V_MIN
    w = w_norm * W_SCALE

    return float(v), float(w)
