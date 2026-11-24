"""
Classical baseline: HOG features + Linear SVM.
Useful when you have very little data and want something that
doesn't overfit as easily as a randomly initialized CNN.

Training uses stratified splits and a proper pipeline so that
StandardScaler never sees the test fold (the classic leakage trap).
"""

from pathlib import Path
from typing import Optional, Dict, Any
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import logging

logger = logging.getLogger(__name__)


class SVMDiagnosticPipeline:
    def __init__(self, model_path: Optional[str] = None):
        self.pipeline: Optional[Pipeline] = None
        self.class_names = ["normal", "abnormal"]
        self._loaded = False

        if model_path and Path(model_path).exists():
            self.load(model_path)
        else:
            logger.warning("No SVM model found – predictions will fail until trained/loaded.")

    def load(self, path: str):
        self.pipeline = joblib.load(path)
        self._loaded = True
        logger.info(f"Loaded SVM pipeline from {path}")

    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        if self.pipeline is None:
            raise RuntimeError("SVM model not loaded")

        # features shape (n_features,) → (1, n_features)
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # CalibratedClassifierCV gives predict_proba
        if hasattr(self.pipeline, "predict_proba"):
            probs = self.pipeline.predict_proba(features)[0]
            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])
            prob_dict = {
                self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))
            }
        else:
            pred_idx = int(self.pipeline.predict(features)[0])
            confidence = None
            prob_dict = None

        return {
            "label": self.class_names[pred_idx],
            "confidence": confidence,
            "probabilities": prob_dict,
            "model": "svm_hog",
            "weights_loaded": self._loaded,
        }


def build_training_pipeline() -> Pipeline:
    """
    Factory used by the training script.
    Note the order: scaler lives *inside* the pipeline so that
    cross-validation / stratified splits never leak statistics.
    """
    base_svm = LinearSVC(C=1.0, dual="auto", max_iter=5000, class_weight="balanced")
    # calibration so we get probabilities instead of just hard labels
    calibrated = CalibratedClassifierCV(base_svm, cv=3, method="sigmoid")

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", calibrated),
        ]
    )
    return pipe
