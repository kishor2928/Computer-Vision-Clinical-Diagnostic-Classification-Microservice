"""
Image preprocessing utilities for the diagnostic pipeline.
Kept deliberately simple and deterministic so training and inference
use the exact same transforms (helps avoid the classic train/serve skew).
"""

from typing import Tuple, Optional
import cv2
import numpy as np
from PIL import Image
import io


# Target size used by both the CNN and the classical pipeline
TARGET_SIZE = (224, 224)


def load_image_from_bytes(data: bytes) -> np.ndarray:
    """Decode raw bytes into a BGR OpenCV image."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image – unsupported or corrupt format")
    return img


def load_image_from_path(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image at {path}")
    return img


def resize_and_pad(img: np.ndarray, size: Tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    """
    Resize while preserving aspect ratio, then pad to exact size.
    Better than naked cv2.resize for medical images where distortion matters.
    """
    h, w = img.shape[:2]
    target_w, target_h = size

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return canvas


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization.
    Often helps with low-contrast clinical images (X-rays, derm, etc.).
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def normalize_for_cnn(img: np.ndarray) -> np.ndarray:
    """
    Convert BGR uint8 → RGB float32 in [0, 1] and channel-first for PyTorch.
    Shape ends up as (3, H, W).
    """
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # simple ImageNet-ish normalization (optional, kept light)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = (rgb - mean) / std
    return np.transpose(rgb, (2, 0, 1))  # CHW


def extract_hog_features(img: np.ndarray) -> np.ndarray:
    """
    Classic HOG features for the SVM baseline.
    Uses OpenCV's HOG descriptor – fast and still surprisingly competitive
    on small medical datasets when you don't have thousands of samples.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))  # HOG likes fixed size
    hog = cv2.HOGDescriptor(
        _winSize=(128, 128),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )
    return hog.compute(gray).flatten()


def full_preprocess_for_cnn(data: bytes) -> np.ndarray:
    """End-to-end path used by the Flask inference endpoint for the CNN."""
    img = load_image_from_bytes(data)
    img = apply_clahe(img)
    img = resize_and_pad(img)
    return normalize_for_cnn(img)


def full_preprocess_for_svm(data: bytes) -> np.ndarray:
    """Same idea for the classical pipeline."""
    img = load_image_from_bytes(data)
    img = apply_clahe(img)
    img = resize_and_pad(img, size=(128, 128))
    return extract_hog_features(img)
