"""Module 1 — Facade parsing via SEEM semantic segmentation."""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from config import FacadeParsingConfig
from .schemas import (
    BBox,
    ClassMask,
    FacadeParsingResult,
    ImageInfo,
    Metadata,
    Polygon,
    ViewType,
)


PRECISE_ROOT = Path(__file__).resolve().parents[2]
SEEM_DIR = PRECISE_ROOT / "third_party" / "SEEM"
WEIGHTS_DIR = PRECISE_ROOT / "third_party" / "weights"

BACKBONE_REGISTRY: dict[str, dict[str, str]] = {
    "focal-l": {
        "config": "focall_unicl_lang_demo.yaml",
        "weight": "seem_focall_v1.pt",
        "url": "https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focall_v1.pt",
    },
    "samvit-l": {
        "config": "samvitl_unicl_lang_v1.yaml",
        "weight": "seem_samvitl_v1.pt",
        "url": "https://huggingface.co/xdecoder/SEEM/resolve/main/seem_samvitl_v1.pt",
    },
}


def _patch_cuda_for_cpu() -> None:
    """No-op `.cuda()` / `torch.cuda.*` so SEEM imports cleanly on CPU-only hosts."""
    if torch.cuda.is_available():
        return
    torch.Tensor.cuda = lambda self, *a, **kw: self
    torch.nn.Module.cuda = lambda self, *a, **kw: self
    torch.cuda.current_device = lambda: torch.device("cpu")
    torch.cuda.empty_cache = lambda: None
    torch.cuda.synchronize = lambda *a, **kw: None


def _ensure_seem_on_path() -> None:
    """Insert the vendored SEEM repo on `sys.path`."""
    if not SEEM_DIR.exists():
        raise FileNotFoundError(
            f"SEEM repo not found at {SEEM_DIR}. "
            f"Run `bash src/facade_parsing/setup_seem.sh` first."
        )
    seem_str = str(SEEM_DIR)
    if seem_str not in sys.path:
        sys.path.insert(0, seem_str)


def _stub_ms_deformable_attention() -> None:
    """Inject a stub `MultiScaleDeformableAttention` so SEEM imports on CPU."""
    import types

    if "MultiScaleDeformableAttention" in sys.modules:
        return

    def _unavailable(*_a, **_kw):
        raise RuntimeError("MSDA CUDA op unavailable on CPU; using pytorch fallback")

    stub = types.ModuleType("MultiScaleDeformableAttention")
    stub.ms_deform_attn_forward = _unavailable
    stub.ms_deform_attn_backward = _unavailable
    sys.modules["MultiScaleDeformableAttention"] = stub


def _download_weight(url: str, dest: Path) -> None:
    """Download a SEEM checkpoint to `dest` with curl (resumable)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    print(f"[facade-parser] downloading {dest.name} from {url}", flush=True)
    subprocess.run(
        ["curl", "-L", "--fail", "-C", "-", "-o", str(tmp), url], check=True
    )
    tmp.rename(dest)


def _resolve_backbone(backbone: str) -> tuple[Path, Path]:
    """Return `(config_path, weight_path)` for `backbone`, downloading weights if absent."""
    if backbone not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone {backbone!r}. Available: {sorted(BACKBONE_REGISTRY)}"
        )
    entry = BACKBONE_REGISTRY[backbone]
    config_path = SEEM_DIR / "configs" / "seem" / entry["config"]
    weight_path = WEIGHTS_DIR / entry["weight"]
    if not config_path.exists():
        raise FileNotFoundError(
            f"SEEM config not found at {config_path}. "
            f"Run `bash src/facade_parsing/setup_seem.sh` first."
        )
    if not weight_path.exists():
        _download_weight(entry["url"], weight_path)
    return config_path, weight_path


def ensure_backbone_assets(backbone: str) -> tuple[Path, Path]:
    """Public wrapper around `_resolve_backbone` so weight downloads run before any UI spinner."""
    return _resolve_backbone(backbone)


class FacadeParser:
    """Text-vocabulary semantic segmentation of a facade image with SEEM."""

    def __init__(
        self,
        image_path: str | Path,
        cfg: FacadeParsingConfig,
        view_type: ViewType = "facade",
    ) -> None:
        """Construct the parser, load SEEM, and bind the prompt vocabulary.

        Args:
            image_path: Path to the input facade image.
            cfg: Module 1 configuration section.
            view_type: Pipeline view type, propagated into the result schema.
        """
        self.image_path = Path(image_path)
        self.cfg = cfg
        self.view_type: ViewType = view_type
        self._prompt_groups: dict[str, list[str]] = {
            p: list(cfg.prompt_synonyms.get(p, [p])) for p in cfg.prompts
        }
        self._flat_vocab: list[str] = [
            term for group in self._prompt_groups.values() for term in group
        ]
        self.config_path, self.weight_path = _resolve_backbone(cfg.backbone)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.last_visualization_image: np.ndarray | None = None
        self.last_mask_dir: Path | None = None
        self.last_class_pixel_counts: dict[str, int] = {p: 0 for p in cfg.prompts}

        self._model = None
        self._transform = transforms.Compose(
            [transforms.Resize(cfg.input_resize_short_side, interpolation=Image.BICUBIC)]
        )
        self._load_seem()

    def _load_seem(self) -> None:
        """Load SEEM with the configured backbone and bind synonym vocab."""
        _patch_cuda_for_cpu()
        _stub_ms_deformable_attention()
        _ensure_seem_on_path()

        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")

        from modeling.BaseModel import BaseModel
        from modeling import build_model
        from utils.arguments import load_opt_from_config_files

        opt = load_opt_from_config_files([str(self.config_path)])
        if isinstance(opt, dict):
            opt["device"] = str(self.device)
            backbone_cfg = opt.get("MODEL", {}).get("BACKBONE", {})
            if backbone_cfg.get("NAME") == "vit":
                backbone_cfg["LOAD_PRETRAINED"] = False

        model = (
            BaseModel(opt, build_model(opt))
            .from_pretrained(str(self.weight_path))
            .eval()
            .to(self.device)
        )

        vocab = list(self._flat_vocab) + ["background"]
        with torch.no_grad():
            model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
                vocab, is_eval=True
            )

        model.model.semantic_on = True
        model.model.panoptic_on = False
        model.model.instance_on = False
        model.model.task_switch["spatial"] = False
        model.model.task_switch["visual"] = False
        model.model.task_switch["grounding"] = False
        model.model.task_switch["audio"] = False

        self._model = model

    @torch.no_grad()
    def _run_semantic(
        self, image_tensor: torch.Tensor, t_h: int, t_w: int
    ) -> torch.Tensor:
        """Run SEEM's semantic head and return a `(V, H, W)` probability map."""
        assert self._model is not None
        data = {"image": image_tensor, "height": t_h, "width": t_w}
        results = self._model.model.evaluate([data])
        return results[0]["sem_seg"].cpu()

    @staticmethod
    def _crop_foreground(
        image_rgb: np.ndarray,
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """Isolate the central building with GrabCut.

        Returns:
            `(cropped_rgb, (x, y, w, h))` of the crop in original-image coords.
            Falls back to the full image if GrabCut yields no foreground.
        """
        h, w = image_rgb.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        rect = (int(w * 0.1), int(h * 0.1), int(w * 0.8), int(h * 0.8))
        cv2.grabCut(image_rgb, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
        binary = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
        coords = cv2.findNonZero(binary)
        if coords is None:
            return image_rgb, (0, 0, w, h)
        x, y, bw, bh = cv2.boundingRect(coords)
        return image_rgb[y:y + bh, x:x + bw], (x, y, bw, bh)

    def _extract_polygons(self, mask: np.ndarray) -> list[list[list[float]]]:
        """Extract simplified outer-contour polygons from a binary mask."""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        polys: list[list[list[float]]] = []
        for c in contours:
            if cv2.contourArea(c) < self.cfg.min_polygon_area_px:
                continue
            epsilon = self.cfg.poly_simplify_eps_ratio * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            polys.append(approx.reshape(-1, 2).astype(float).tolist())
        return polys

    @staticmethod
    def _bbox_from_mask(mask: np.ndarray) -> tuple[float, float, float, float] | None:
        """Axis-aligned xyxy bbox of `mask`, or `None` if empty."""
        ys, xs = mask.nonzero()
        if xs.size == 0:
            return None
        return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())

    @staticmethod
    def _normalize_bbox(box: tuple[float, ...], w: int, h: int) -> list[float]:
        """Normalize an xyxy bbox to `[0, 1]`."""
        x1, y1, x2, y2 = box
        return [x1 / w, y1 / h, x2 / w, y2 / h]

    @staticmethod
    def _normalize_polygon(
        pts: list[list[float]], w: int, h: int
    ) -> list[list[float]]:
        """Normalize a list of polygon points to `[0, 1]`."""
        return [[x / w, y / h] for x, y in pts]

    def _draw_overlay(
        self,
        image_bgr: np.ndarray,
        masks: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Composite per-class colored overlays on `image_bgr` and return the BGR uint8 array."""
        colors = {k: tuple(v) for k, v in self.cfg.label_colors_bgr.items()}
        alpha = self.cfg.overlay_alpha
        base = image_bgr.astype(np.float32)
        composite = base.copy()
        paint_priority = {"house": 0, "door": 1, "window": 2}
        order = sorted(masks.keys(), key=lambda k: paint_priority.get(k, 99))
        for label in order:
            mask = masks[label]
            if mask.sum() == 0:
                continue
            color = colors.get(label, (200, 200, 200))
            color_layer = np.zeros_like(base, dtype=np.float32)
            color_layer[:] = color
            bool_mask = mask.astype(bool)
            composite[bool_mask] = (
                composite[bool_mask] * (1 - alpha) + color_layer[bool_mask] * alpha
            )
        for label in order:
            mask = masks[label]
            if mask.sum() == 0:
                continue
            color = colors.get(label, (200, 200, 200))
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(composite, contours, -1, color, 2)
        return composite.clip(0, 255).astype(np.uint8)

    def parse(self) -> FacadeParsingResult:
        """Run SEEM and return a structured per-class result.

        Returns:
            `FacadeParsingResult` with per-class bbox, polygons, confidence, and an
            optional overlay path (when `cfg.save_visualization` is true).
        """
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image not found: {self.image_path}")

        image_pil = Image.open(self.image_path).convert("RGB")
        orig_w, orig_h = image_pil.size
        image_rgb_full = np.asarray(image_pil)
        image_bgr = cv2.cvtColor(image_rgb_full, cv2.COLOR_RGB2BGR)

        if self.cfg.crop_foreground:
            crop_rgb, (cx, cy, cw, ch) = self._crop_foreground(image_rgb_full)
            inference_pil = Image.fromarray(crop_rgb)
        else:
            inference_pil = image_pil
            cx, cy, cw, ch = 0, 0, orig_w, orig_h

        image_t = self._transform(inference_pil)
        t_w, t_h = image_t.size
        image_np = np.asarray(image_t)
        image_tensor = (
            torch.from_numpy(image_np.copy()).permute(2, 0, 1).to(self.device)
        )

        sem_seg = self._run_semantic(image_tensor, t_h, t_w)
        sem_seg_crop = F.interpolate(
            sem_seg.unsqueeze(0).float(),
            size=(ch, cw),
            mode="bilinear",
            align_corners=False,
        )[0].numpy()
        sem_seg_full = np.zeros(
            (sem_seg_crop.shape[0], orig_h, orig_w), dtype=sem_seg_crop.dtype
        )
        sem_seg_full[:, cy:cy + ch, cx:cx + cw] = sem_seg_crop

        merged: dict[str, np.ndarray] = {}
        cursor = 0
        for prompt, terms in self._prompt_groups.items():
            n = len(terms)
            block = sem_seg_full[cursor:cursor + n]
            merged[prompt] = block.sum(axis=0) if n > 1 else block[0]
            cursor += n

        mask_dir: Path | None = None
        if self.cfg.save_masks:
            mask_dir = self.image_path.with_name(f"{self.image_path.stem}_masks")
            mask_dir.mkdir(exist_ok=True)
            self.last_mask_dir = mask_dir

        masks: dict[str, np.ndarray] = {}
        classes: list[ClassMask] = []
        for prompt in self.cfg.prompts:
            prob = merged[prompt]
            mask = (prob > self.cfg.threshold).astype(np.uint8)
            masks[prompt] = mask
            pixel_area = int(mask.sum())
            self.last_class_pixel_counts[prompt] = pixel_area

            if pixel_area == 0:
                classes.append(
                    ClassMask(
                        label=prompt,
                        confidence=0.0,
                        pixel_area=0,
                        bbox=None,
                        polygons=[],
                        mask_path=None,
                    )
                )
                continue

            confidence = float(prob[mask.astype(bool)].mean())
            box = self._bbox_from_mask(mask)
            bbox = None
            if box is not None:
                bbox = BBox(
                    pixel=[round(v, 2) for v in box],
                    normalized=[
                        round(v, 6) for v in self._normalize_bbox(box, orig_w, orig_h)
                    ],
                )
            polys_pixel = self._extract_polygons(mask)
            polygons = [
                Polygon(
                    pixel=[[round(x, 2), round(y, 2)] for x, y in pts],
                    normalized=[
                        [round(x, 6), round(y, 6)]
                        for x, y in self._normalize_polygon(pts, orig_w, orig_h)
                    ],
                )
                for pts in polys_pixel
            ]
            mask_path: str | None = None
            if mask_dir is not None:
                p = mask_dir / f"{prompt}.png"
                cv2.imwrite(str(p), (mask * 255).astype(np.uint8))
                mask_path = str(p)
            classes.append(
                ClassMask(
                    label=prompt,
                    confidence=round(min(max(confidence, 0.0), 1.0), 4),
                    pixel_area=pixel_area,
                    bbox=bbox,
                    polygons=polygons,
                    mask_path=mask_path,
                )
            )

        self.last_visualization_image = self._draw_overlay(image_bgr, masks)

        return FacadeParsingResult(
            image=ImageInfo(id=self.image_path.stem, width=orig_w, height=orig_h),
            view_type=self.view_type,
            classes=classes,
            metadata=Metadata(
                model_version=self.cfg.model_version,
                backbone=f"seem-{self.cfg.backbone}",
                prompts=self._flat_vocab,
                threshold=self.cfg.threshold,
                timestamp=datetime.now(timezone.utc),
            ),
        )
