import numpy as np
import cv2
from app.preprocessing.image_utils import (
    resize_and_pad,
    apply_clahe,
    normalize_for_cnn,
    extract_hog_features,
    TARGET_SIZE,
)


def _dummy_bgr(h=300, w=400):
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def test_resize_and_pad_shape():
    img = _dummy_bgr()
    out = resize_and_pad(img)
    assert out.shape == (TARGET_SIZE[1], TARGET_SIZE[0], 3)


def test_clahe_preserves_shape():
    img = _dummy_bgr()
    out = apply_clahe(img)
    assert out.shape == img.shape


def test_normalize_cnn_shape_and_range():
    img = _dummy_bgr(224, 224)
    out = normalize_for_cnn(img)
    assert out.shape == (3, 224, 224)
    # roughly centered after ImageNet norm
    assert out.mean() < 2.0


def test_hog_feature_length():
    img = _dummy_bgr(128, 128)
    feats = extract_hog_features(img)
    assert feats.ndim == 1
    assert len(feats) > 100  # HOG produces a reasonably long vector
