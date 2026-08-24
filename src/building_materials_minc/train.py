"""Fine-tune a material patch classifier on MINC-2500 (facade subset).

Downloads MINC-2500 from the HuggingFace Hub on first run, keeps the 8 facade
target classes (see labels.py), and fine-tunes a timm backbone. Saves a
self-describing checkpoint (weights + arch + class names + preprocessing) to
``runs/best.pt`` for inference.

    python src/building_materials_minc/train.py                 # full run
    python src/building_materials_minc/train.py --limit 400 --epochs 1   # smoke
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from building_materials_minc.data import MincFacadeDataset
from building_materials_minc.labels import NUM_CLASSES, TARGET_CLASSES
from building_materials_minc.model import build_classifier, build_transforms


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="mcimpoi/minc-2500_split_1", help="HF dataset id.")
    p.add_argument("--arch", default="convnext_tiny", help="timm backbone.")
    p.add_argument("--epochs", type=int, default=18)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None, help="Cap samples/split (smoke).")
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", default="src/building_materials_minc/runs")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, num_classes) -> tuple[float, np.ndarray]:
    """Return (overall accuracy, per-class accuracy)."""
    model.eval()
    correct = np.zeros(num_classes, dtype=np.int64)
    total = np.zeros(num_classes, dtype=np.int64)
    for images, targets in tqdm(loader, desc="val", leave=False):
        images = images.to(device, non_blocking=True)
        preds = model(images).argmax(1).cpu().numpy()
        t = targets.numpy()
        for c in range(num_classes):
            m = t == c
            total[c] += int(m.sum())
            correct[c] += int((preds[m] == c).sum())
    per_class = np.where(total > 0, correct / np.maximum(total, 1), np.nan)
    overall = correct.sum() / max(1, total.sum())
    return float(overall), per_class


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    from datasets import load_dataset
    print(f"[minc] loading {args.dataset} (downloads on first run)...", flush=True)
    ds = load_dataset(args.dataset)
    train_split, val_split = ds["train"], ds["validation"]
    if args.limit:
        train_split = train_split.select(range(min(args.limit, len(train_split))))
        val_split = val_split.select(range(min(args.limit, len(val_split))))

    model = build_classifier(NUM_CLASSES, arch=args.arch, pretrained=True).to(device)
    train_tf = build_transforms(model, train=True)
    val_tf = build_transforms(model, train=False)
    data_config = timm.data.resolve_model_data_config(model)

    train_ds = MincFacadeDataset(train_split, train_tf)
    val_ds = MincFacadeDataset(val_split, val_tf)
    print(f"[minc] facade patches: train={len(train_ds)} val={len(val_ds)} "
          f"| classes={TARGET_CLASSES}", flush=True)

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    best_acc, best_path = -1.0, out / "best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for images, targets in tqdm(train_loader, desc=f"train {epoch}/{args.epochs}", leave=False):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    loss = criterion(model(images), targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = criterion(model(images), targets)
                loss.backward()
                optimizer.step()
            running += loss.item() * images.size(0)
        scheduler.step()
        acc, per_class = evaluate(model, val_loader, device, NUM_CLASSES)
        pc = " ".join(f"{c}={per_class[i]:.2f}" for i, c in enumerate(TARGET_CLASSES))
        print(f"[minc] epoch {epoch:2d}/{args.epochs} loss={running/max(1,len(train_ds)):.4f} "
              f"val_acc={acc:.4f} | {pc}", flush=True)
        if acc > best_acc:
            best_acc = acc
            torch.save({
                "model_state": model.state_dict(),
                "arch": args.arch,
                "class_names": TARGET_CLASSES,
                "num_classes": NUM_CLASSES,
                "data_config": data_config,
                "val_acc": acc,
                "epoch": epoch,
            }, best_path)
            print(f"  ↳ new best val_acc={acc:.4f} -> {best_path}", flush=True)

    print(f"[minc] done. best val_acc={best_acc:.4f} | checkpoint: {best_path}", flush=True)


if __name__ == "__main__":
    main()
