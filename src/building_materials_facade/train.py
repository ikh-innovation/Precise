"""Train the DINOv3 Facade-8 material classifier on the pooled manifest.

Reads data/material_datasets/facade8_manifest.csv (see build_dataset.py),
trains a frozen-DINOv3 + MLP head (or fine-tunes the last blocks with
--unfreeze-blocks), and saves a self-describing checkpoint to runs/best.pt —
the same checkpoint shape the MINC backend uses, so the inference backend can
rebuild the model and the eval transform without drift.

    python src/building_materials_facade/train.py                      # frozen probe
    python src/building_materials_facade/train.py --unfreeze-blocks 2   # + fine-tune tail
    python src/building_materials_facade/train.py --epochs 1 --limit 200  # smoke
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from building_materials_facade.data import ManifestDataset  # noqa: E402
from building_materials_facade.labels import NUM_CLASSES, TARGET_CLASSES, TARGET_TO_IDX  # noqa: E402
from building_materials_facade.model import DEFAULT_ARCH, build_classifier  # noqa: E402

MANIFEST = _SRC.parent / "data" / "material_datasets" / "facade8_manifest.csv"


def load_rows(split: str, limit: int | None) -> list[dict]:
    rows = []
    with MANIFEST.open() as fh:
        for r in csv.DictReader(fh):
            if r["split"] == split:
                rows.append(r)
    if limit is not None:
        # keep a class-balanced slice for smoke tests
        by_t: dict[str, list[dict]] = {}
        for r in rows:
            by_t.setdefault(r["target"], []).append(r)
        rows = [r for items in by_t.values() for r in items[: max(1, limit // NUM_CLASSES)]]
    return rows


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Return (overall_acc, macro_f1, per_class_acc, confusion_matrix)."""
    model.eval()
    ys, ps = [], []
    for x, y in tqdm(loader, desc="val", leave=False):
        x = x.to(device, non_blocking=True)
        pred = model(x).argmax(1).cpu().numpy()
        ps.append(pred)
        ys.append(y.numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    overall = float((y == p).mean())
    macro_f1 = float(f1_score(y, p, average="macro", labels=list(range(NUM_CLASSES)), zero_division=0))
    cm = confusion_matrix(y, p, labels=list(range(NUM_CLASSES)))
    per_class = np.where(cm.sum(1) > 0, cm.diagonal() / np.maximum(cm.sum(1), 1), np.nan)
    return overall, macro_f1, per_class, cm


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default=DEFAULT_ARCH)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3, help="Head LR.")
    ap.add_argument("--backbone-lr", type=float, default=1e-5, help="LR for unfrozen blocks.")
    ap.add_argument("--weight-decay", type=float, default=5e-2)
    ap.add_argument("--unfreeze-blocks", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="Class-balanced cap/split (smoke).")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", default="src/building_materials_facade/runs")
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if not MANIFEST.exists():
        sys.exit(f"manifest not found: {MANIFEST} — run build_dataset.py first")

    model = build_classifier(NUM_CLASSES, arch=args.arch, pretrained=True,
                             unfreeze_blocks=args.unfreeze_blocks,
                             hidden=args.hidden, dropout=args.dropout).to(device)
    data_config = timm.data.resolve_model_data_config(model.backbone)
    mean, std = data_config["mean"], data_config["std"]
    input_size = data_config["input_size"][-1]

    train_rows = load_rows("train", args.limit)
    val_rows = load_rows("val", args.limit)
    test_rows = load_rows("test", args.limit)
    print(f"[facade8] train={len(train_rows)} val={len(val_rows)} test={len(test_rows)} "
          f"| arch={args.arch} unfreeze={args.unfreeze_blocks} input={input_size}", flush=True)

    train_ds = ManifestDataset(train_rows, mean, std, input_size, train=True)
    val_ds = ManifestDataset(val_rows, mean, std, input_size, train=False)
    test_ds = ManifestDataset(test_rows, mean, std, input_size, train=False)

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=pin)

    # Class-weighted CE to counter pooled-source imbalance.
    counts = Counter(int(r["target_idx"]) for r in train_rows)
    freq = np.array([counts.get(i, 0) for i in range(NUM_CLASSES)], dtype=np.float64)
    weights = np.where(freq > 0, freq.sum() / (NUM_CLASSES * np.maximum(freq, 1)), 0.0)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device),
                                    label_smoothing=0.05)
    print(f"[facade8] train class counts: "
          f"{ {TARGET_CLASSES[i]: int(freq[i]) for i in range(NUM_CLASSES)} }", flush=True)

    # Param groups: head (full LR) + optionally unfrozen backbone blocks (low LR).
    head_params = list(model.head.parameters())
    bb_params = [p for n, p in model.backbone.named_parameters() if p.requires_grad]
    groups = [{"params": head_params, "lr": args.lr}]
    if bb_params:
        groups.append({"params": bb_params, "lr": args.backbone_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    best_f1, best_path = -1.0, out / "best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in tqdm(train_loader, desc=f"train {epoch}/{args.epochs}", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    loss = criterion(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
            running += loss.item() * x.size(0)
        scheduler.step()

        acc, macro_f1, per_class, _ = evaluate(model, val_loader, device)
        pc = " ".join(f"{c}={per_class[i]:.2f}" for i, c in enumerate(TARGET_CLASSES))
        print(f"[facade8] epoch {epoch:2d}/{args.epochs} loss={running/max(1,len(train_ds)):.4f} "
              f"val_acc={acc:.4f} val_macroF1={macro_f1:.4f} | {pc}", flush=True)

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save({
                "model_state": model.state_dict(),
                "arch": args.arch,
                "class_names": TARGET_CLASSES,
                "num_classes": NUM_CLASSES,
                "data_config": data_config,
                "unfreeze_blocks": args.unfreeze_blocks,
                "hidden": args.hidden,
                "dropout": args.dropout,
                "val_acc": acc,
                "val_macro_f1": macro_f1,
                "epoch": epoch,
            }, best_path)
            print(f"  ↳ new best val_macroF1={macro_f1:.4f} -> {best_path}", flush=True)

    # Final test report on the best checkpoint.
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    acc, macro_f1, per_class, cm = evaluate(model, test_loader, device)
    print(f"\n[facade8] TEST  acc={acc:.4f} macroF1={macro_f1:.4f}", flush=True)
    for i, c in enumerate(TARGET_CLASSES):
        print(f"  {c:9s} acc={per_class[i]:.3f}", flush=True)
    print("[facade8] confusion matrix (rows=true, cols=pred):", flush=True)
    print("           " + " ".join(f"{c[:5]:>6s}" for c in TARGET_CLASSES), flush=True)
    for i, c in enumerate(TARGET_CLASSES):
        print(f"  {c:9s} " + " ".join(f"{int(v):6d}" for v in cm[i]), flush=True)
    print(f"[facade8] done. best val_macroF1={best_f1:.4f} | checkpoint: {best_path}", flush=True)


if __name__ == "__main__":
    main()
