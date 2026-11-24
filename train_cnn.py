"""
Minimal training loop for the SimpleDiagnosticCNN.

Again, stratified split first so validation metrics are honest.
This is *not* a production training script – no mixed precision,
no heavy augmentation, no distributed stuff. Just enough to
produce a .pt file you can drop into models/weights/.
"""

import sys
from pathlib import Path
import argparse
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models.cnn_model import SimpleDiagnosticCNN
from app.preprocessing.image_utils import load_image_from_path, apply_clahe, resize_and_pad, normalize_for_cnn


class FolderDataset(Dataset):
    def __init__(self, root: Path):
        self.samples = []
        for label, name in enumerate(["normal", "abnormal"]):
            folder = root / name
            if not folder.exists():
                continue
            for p in folder.glob("*"):
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    self.samples.append((p, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = load_image_from_path(str(path))
        img = apply_clahe(img)
        img = resize_and_pad(img)
        tensor = torch.from_numpy(normalize_for_cnn(img))
        return tensor, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="models/weights/cnn_diagnostic.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ds = FolderDataset(Path(args.data_dir))
    print(f"Total images: {len(ds)}")
    if len(ds) < 20:
        print("Need more samples. Exiting.")
        return

    labels = [lbl for _, lbl in ds.samples]
    indices = list(range(len(ds)))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=42
    )

    train_loader = DataLoader(Subset(ds, train_idx), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=args.batch_size)

    model = SimpleDiagnosticCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
        train_loss = running / len(train_idx)

        # validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        val_acc = correct / total if total else 0.0
        print(f"Epoch {epoch:02d}  train_loss={train_loss:.4f}  val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out)
            print(f"  → saved best model ({val_acc:.3f})")

    print(f"Finished. Best val accuracy: {best_val_acc:.3f}")


if __name__ == "__main__":
    main()
