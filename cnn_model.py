"""
Lightweight CNN for binary clinical image classification.
Intentionally small so it can run on CPU without drama.
In a real project this would be a fine-tuned ResNet / EfficientNet;
here we keep it educational and self-contained.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class SimpleDiagnosticCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class CNNInferenceWrapper:
    """
    Thin wrapper that loads weights once and exposes a predict() method
    the Flask layer can call safely.
    """

    def __init__(self, weights_path: Optional[str] = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = SimpleDiagnosticCNN(num_classes=2)
        self.model.to(self.device)
        self.model.eval()
        self.class_names = ["normal", "abnormal"]  # placeholder labels
        self._loaded = False

        if weights_path and Path(weights_path).exists():
            self.load(weights_path)
        else:
            logger.warning(
                "No CNN weights found – model will run with random init. "
                "Fine for smoke tests, useless for real predictions."
            )

    def load(self, path: str):
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()
        self._loaded = True
        logger.info(f"Loaded CNN weights from {path}")

    @torch.no_grad()
    def predict(self, tensor: torch.Tensor) -> dict:
        """
        tensor expected shape (1, 3, H, W) already normalized.
        Returns class probabilities + predicted label.
        """
        tensor = tensor.to(self.device)
        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(probs.argmax())
        return {
            "label": self.class_names[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": {
                self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))
            },
            "model": "cnn",
            "weights_loaded": self._loaded,
        }
