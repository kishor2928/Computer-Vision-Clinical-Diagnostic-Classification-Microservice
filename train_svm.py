"""
Train the HOG + LinearSVM baseline with proper stratified splitting.

This script is deliberately explicit about the train/test split
so that anyone reading the code can see we never fit the scaler
on the test fold.
"""

import sys
from pathlib import Path
import argparse
import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# make the package importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models.svm_pipeline import build_training_pipeline
from app.preprocessing.image_utils import load_image_from_path, apply_clahe, resize_and_pad, extract_hog_features


def load_dataset(root: Path):
    """
    Expects a simple folder layout:
        root/
          normal/
            *.jpg|png
          abnormal/
            *.jpg|png
    """
    X, y = [], []
    for label_idx, class_name in enumerate(["normal", "abnormal"]):
        folder = root / class_name
        if not folder.exists():
            print(f"WARNING: missing folder {folder}")
            continue
        for img_path in folder.glob("*"):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            try:
                img = load_image_from_path(str(img_path))
                img = apply_clahe(img)
                img = resize_and_pad(img, size=(128, 128))
                feats = extract_hog_features(img)
                X.append(feats)
                y.append(label_idx)
            except Exception as e:
                print(f"  skipped {img_path.name}: {e}")
    return np.array(X), np.array(y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="Path to normal/ + abnormal/ folders")
    parser.add_argument("--out", type=str, default="models/weights/svm_hog.joblib")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    data_root = Path(args.data_dir)
    print(f"Loading images from {data_root} …")
    X, y = load_dataset(data_root)
    print(f"Loaded {len(X)} samples  (class balance: {np.bincount(y)})")

    if len(X) < 20:
        print("Not enough samples for a meaningful stratified split. Aborting.")
        return

    # ------------------------------------------------------------------
    # Hold-out set FIRST – this is the key anti-leakage step
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        stratify=y,
        random_state=42,
    )
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

    pipe = build_training_pipeline()

    # quick CV on the *training* fold only
    cv = StratifiedKFold(n_splits=min(5, len(X_train) // 4), shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro")
    print(f"Stratified CV F1 (train folds only): {scores.mean():.3f} ± {scores.std():.3f}")

    # fit on full train, evaluate on held-out test
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print("\nHeld-out test set:")
    print(classification_report(y_test, y_pred, target_names=["normal", "abnormal"]))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_path)
    print(f"\nSaved pipeline → {out_path}")


if __name__ == "__main__":
    main()
