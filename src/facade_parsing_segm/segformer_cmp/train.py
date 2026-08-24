"""Fine-tune SegFormer on the full 12-class CMP Facade Database.

Tuned-for-accuracy recipe (overridable via CLI):

    backbone   nvidia/mit-b4    (SegFormer-B4; b0..b5 selectable, bigger = better)
    optimizer  AdamW            (Trainer default)
    lr         6e-5  + cosine schedule, 10% warmup
    epochs     80    (early stopping, patience 15, best by mean IoU)
    batch      2 x grad-accum 4 = effective 8
    precision  bf16 on GPU
    aug        hflip + brightness/contrast jitter (train split)
    metric     mean IoU (best-model selection)

Run on the GPU env. The backbone is downloaded from the HuggingFace Hub on
first run. Examples::

    python src/facade_parsing_segm/segformer_cmp/train.py            # b4, full recipe
    python src/facade_parsing_segm/segformer_cmp/train.py --backbone nvidia/mit-b5
    # quick functional check (few images, 1 epoch):
    python src/facade_parsing_segm/segformer_cmp/train.py --limit 8 --epochs 1 --patience 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from facade_parsing_segm.segformer_cmp.data import (
    CmpSegformerDataset,
    discover_pairs,
    split_pairs,
)
from facade_parsing_segm.segformer_cmp.labels import (
    CLASS_NAMES,
    ID2LABEL,
    LABEL2ID,
    NUM_CLASSES,
)


def make_compute_metrics(num_classes: int):
    """Return a Trainer ``compute_metrics`` computing dataset-level mean IoU.

    SegFormer logits come out at H/4; we upsample to the label resolution,
    argmax, and accumulate a confusion matrix (no `evaluate`/`datasets` dep).
    """

    def compute_metrics(eval_pred) -> dict:
        logits, labels = eval_pred
        logits_t = torch.as_tensor(logits, dtype=torch.float32)
        labels = np.asarray(labels)
        upsampled = F.interpolate(
            logits_t, size=labels.shape[-2:], mode="bilinear", align_corners=False
        )
        preds = upsampled.argmax(dim=1).numpy()
        valid = (labels >= 0) & (labels < num_classes)
        cm = np.bincount(
            num_classes * labels[valid].astype(np.int64) + preds[valid].astype(np.int64),
            minlength=num_classes ** 2,
        ).reshape(num_classes, num_classes).astype(np.float64)
        tp = np.diag(cm)
        union = cm.sum(0) + cm.sum(1) - tp
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = np.where(union > 0, tp / np.maximum(union, 1e-9), np.nan)
        total = cm.sum()
        out = {
            "mean_iou": float(np.nanmean(iou)) if np.isfinite(iou).any() else 0.0,
            "pixel_acc": float(tp.sum() / total) if total > 0 else 0.0,
        }
        # Per-class IoU (NaN where the class is absent in this eval set) so the
        # per-epoch report can show which classes are learning.
        for i, name in enumerate(CLASS_NAMES):
            out[f"iou_{name}"] = float(iou[i])
        return out

    return compute_metrics


class EpochReport(TrainerCallback):
    """Shout out mean IoU, pixel acc, and per-class IoU after every eval."""

    def __init__(self, class_names: list[str]) -> None:
        self.class_names = class_names

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        ep = state.epoch or 0.0
        mi = metrics.get("eval_mean_iou", float("nan"))
        pa = metrics.get("eval_pixel_acc", float("nan"))
        loss = metrics.get("eval_loss", float("nan"))
        print(
            f"\n[segformer-cmp] ===== epoch {ep:5.1f} =====  "
            f"mean_IoU={mi:.4f}  pixel_acc={pa:.4f}  eval_loss={loss:.4f}",
            flush=True,
        )
        per = "  ".join(
            f"{n}={metrics.get(f'eval_iou_{n}', float('nan')):.3f}"
            for n in self.class_names
        )
        print(f"  per-class IoU: {per}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="base")
    p.add_argument("--backbone", default="nvidia/mit-b4",
                   help="HF SegFormer encoder (nvidia/mit-b0 .. mit-b5; bigger = better).")
    p.add_argument("--lr", type=float, default=6e-5)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4,
                   help="Gradient accumulation (effective batch = batch_size * grad_accum).")
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--scheduler", default="cosine", help="LR scheduler type.")
    p.add_argument("--patience", type=int, default=15, help="Early-stop patience (epochs); 0 disables.")
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None, help="Cap #samples (debug).")
    p.add_argument("--out-dir", default="src/facade_parsing_segm/segformer_cmp/runs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    processor = SegformerImageProcessor(do_reduce_labels=False)

    pairs = discover_pairs(args.data_root)
    if args.limit:
        pairs = pairs[: args.limit]
    train_pairs, val_pairs = split_pairs(pairs, args.val_split, args.seed)
    print(f"[segformer-cmp] {len(pairs)} pairs -> train={len(train_pairs)} val={len(val_pairs)}"
          f" | {NUM_CLASSES} classes", flush=True)
    if not train_pairs or not val_pairs:
        raise RuntimeError("Not enough data after split; check --data-root/--limit.")

    train_ds = CmpSegformerDataset(train_pairs, processor, train=True)
    val_ds = CmpSegformerDataset(val_pairs, processor, train=False)

    model = SegformerForSemanticSegmentation.from_pretrained(
        args.backbone,
        num_labels=NUM_CLASSES,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    use_cuda = torch.cuda.is_available()
    ta_kwargs = dict(
        output_dir=args.out_dir,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup_ratio,
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="mean_iou",
        greater_is_better=True,
        remove_unused_columns=False,
        dataloader_num_workers=args.num_workers,
        bf16=use_cuda,  # Blackwell supports bf16; more stable than fp16
        seed=args.seed,
        report_to="none",
    )
    # transformers renamed evaluation_strategy -> eval_strategy (>=4.46/5.x).
    import inspect
    params = inspect.signature(TrainingArguments.__init__).parameters
    ta_kwargs["eval_strategy" if "eval_strategy" in params else "evaluation_strategy"] = "epoch"
    training_args = TrainingArguments(**ta_kwargs)

    callbacks = [EpochReport(CLASS_NAMES)]
    if args.patience and args.patience > 0:
        from transformers import EarlyStoppingCallback
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.patience))

    print(f"[segformer-cmp] backbone={args.backbone} epochs={args.epochs} "
          f"eff_batch={args.batch_size * args.grad_accum} lr={args.lr} "
          f"sched={args.scheduler} bf16={use_cuda}", flush=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=make_compute_metrics(NUM_CLASSES),
        callbacks=callbacks,
    )
    trainer.train()

    best_dir = str(Path(args.out_dir) / "best")
    trainer.save_model(best_dir)
    processor.save_pretrained(best_dir)
    metrics = trainer.evaluate()
    print(f"[segformer-cmp] best model saved to {best_dir}", flush=True)
    print(f"[segformer-cmp] final eval: mean_iou={metrics.get('eval_mean_iou'):.4f} "
          f"pixel_acc={metrics.get('eval_pixel_acc'):.4f}", flush=True)


if __name__ == "__main__":
    main()
