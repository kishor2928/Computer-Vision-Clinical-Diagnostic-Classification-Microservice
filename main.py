"""
Flask inference microservice for clinical image classification.
Supports both the CNN and the classical HOG+SVM path.
"""

import os
import io
import logging
from pathlib import Path

from flask import Flask, request, jsonify, g
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge
import torch
import numpy as np

from app.preprocessing.image_utils import (
    full_preprocess_for_cnn,
    full_preprocess_for_svm,
)
from app.models.cnn_model import CNNInferenceWrapper
from app.models.svm_pipeline import SVMDiagnosticPipeline
from app.middleware.logging_middleware import (
    setup_logging,
    request_id_middleware,
    log_request_end,
    handle_exceptions,
)

setup_logging(os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("diagnostic_service")

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB upload limit
    CORS(app)

    # load models once at startup
    weights_dir = Path(os.getenv("MODEL_DIR", "models/weights"))
    cnn_path = weights_dir / "cnn_diagnostic.pt"
    svm_path = weights_dir / "svm_hog.joblib"

    app.cnn = CNNInferenceWrapper(str(cnn_path) if cnn_path.exists() else None)
    app.svm = SVMDiagnosticPipeline(str(svm_path) if svm_path.exists() else None)

    # middleware hooks
    @app.before_request
    def _before():
        request_id_middleware()

    @app.after_request
    def _after(response):
        return log_request_end(response)

    handle_exceptions(app)

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "cnn_loaded": app.cnn._loaded,
                "svm_loaded": app.svm._loaded,
            }
        )

    @app.post("/predict")
    def predict():
        """
        Multipart form:
          - image: the file
          - model: optional, "cnn" | "svm" (default cnn)
        """
        if "image" not in request.files:
            return jsonify({"error": "missing_image", "message": "No image file provided"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "empty_filename"}), 400

        model_choice = request.form.get("model", "cnn").lower().strip()
        if model_choice not in ("cnn", "svm"):
            return jsonify({"error": "invalid_model", "message": "model must be 'cnn' or 'svm'"}), 400

        raw = file.read()
        if not raw:
            return jsonify({"error": "empty_file"}), 400

        try:
            if model_choice == "cnn":
                arr = full_preprocess_for_cnn(raw)
                tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, 3, H, W)
                result = app.cnn.predict(tensor)
            else:
                feats = full_preprocess_for_svm(raw)
                result = app.svm.predict(feats)
        except ValueError as e:
            logger.warning("preprocess_failed", extra={"request_id": g.request_id, "err": str(e)})
            return jsonify({"error": "invalid_image", "message": str(e)}), 400
        except RuntimeError as e:
            return jsonify({"error": "model_not_ready", "message": str(e)}), 503
        except Exception as e:
            logger.exception("predict_failed")
            return jsonify({"error": "inference_error", "message": str(e)}), 500

        result["request_id"] = g.request_id
        return jsonify(result)

    @app.get("/")
    def root():
        return jsonify(
            {
                "service": "Clinical Diagnostic Classification Microservice",
                "endpoints": {
                    "health": "GET /health",
                    "predict": "POST /predict  (multipart: image + optional model=cnn|svm)",
                },
            }
        )

    return app


# for `flask run` / gunicorn
app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
