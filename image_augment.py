"""Train-time image augmentations for TriSearch (geometry + photometric).

All defaults live at the top of this file for easy tuning. Training imports
the stack from here; run this file directly for a visual preview demo::

    python3 image_augment.py
    python3 image_augment.py --image /path/to.jpg

Photometric augs denormalize processor tensors → RGB-ish [0,1], apply jitter /
smooth spatial color+brightness fields / optional grayscale, then renorm.
This reduces brightness-only / color-shortcut learning in MaxSim heatmaps.
"""

from __future__ import annotations

import argparse
import math
from typing import Sequence

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Defaults (edit here for tuning / demo)
# ---------------------------------------------------------------------------

# Geometric
DEFAULT_IMAGE_HFLIP_PROB = 0.5
DEFAULT_IMAGE_MAX_ROTATE_DEG = 30.0
DEFAULT_IMAGE_SCALE_MIN = 0.75
DEFAULT_IMAGE_SCALE_MAX = 1.05
DEFAULT_IMAGE_FILL_MODE = "random"  # random | mean | reflect
DEFAULT_IMAGE_SHIFT_MAX = 18

# Global photometric (ColorJitter-style ranges; 0 disables that axis)
DEFAULT_PHOTO_BRIGHTNESS = 0.2
DEFAULT_PHOTO_CONTRAST = 0.2
DEFAULT_PHOTO_SATURATION = 0.3
DEFAULT_PHOTO_HUE = 0.07

# Spatially varying fields (low-res noise upsampled ≈ blurred Perlin)
DEFAULT_SPATIAL_BRIGHTNESS = 0.2
DEFAULT_SPATIAL_COLOR = 0.1
DEFAULT_SPATIAL_NOISE_GRID = 24

# Occasional grayscale kills pure-color shortcuts
DEFAULT_GRAYSCALE_PROB = 0.1

# Photometric on by default when full image-aug stack is enabled
DEFAULT_PHOTOMETRIC_ENABLED = True

# Fallback mean/std when processor stats are not passed (SigLIP often uses 0.5)
DEFAULT_IMAGE_MEAN: tuple[float, float, float] = (0.5, 0.5, 0.5)
DEFAULT_IMAGE_STD: tuple[float, float, float] = (0.5, 0.5, 0.5)

# Demo
DEFAULT_DEMO_N_TILES = 6


def _as_chw_mean_std(
    mean: Sequence[float] | None,
    std: Sequence[float] | None,
    *,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    m = list(mean) if mean is not None else list(DEFAULT_IMAGE_MEAN)
    s = list(std) if std is not None else list(DEFAULT_IMAGE_STD)
    if len(m) == 1:
        m = m * channels
    if len(s) == 1:
        s = s * channels
    if len(m) < channels:
        m = (m + [m[-1]] * channels)[:channels]
    if len(s) < channels:
        s = (s + [s[-1]] * channels)[:channels]
    mean_t = torch.tensor(m[:channels], device=device, dtype=dtype).view(-1, 1, 1)
    std_t = torch.tensor(s[:channels], device=device, dtype=dtype).view(-1, 1, 1)
    std_t = std_t.clamp_min(1e-6)
    return mean_t, std_t


def denormalize_pixel_values(
    pixel_values: torch.Tensor,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
) -> torch.Tensor:
    """``(B,C,H,W)`` processor space → approximate RGB in roughly ``[0,1]``."""
    if pixel_values.ndim != 4:
        raise ValueError(
            f"pixel_values must be (B,C,H,W), got {tuple(pixel_values.shape)}"
        )
    b, c, _, _ = pixel_values.shape
    mean_t, std_t = _as_chw_mean_std(
        mean, std, channels=c, device=pixel_values.device, dtype=pixel_values.dtype
    )
    return pixel_values * std_t + mean_t


def normalize_pixel_values(
    rgb: torch.Tensor,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
) -> torch.Tensor:
    """Approximate RGB → processor-normalized ``(B,C,H,W)``."""
    if rgb.ndim != 4:
        raise ValueError(f"rgb must be (B,C,H,W), got {tuple(rgb.shape)}")
    _, c, _, _ = rgb.shape
    mean_t, std_t = _as_chw_mean_std(
        mean, std, channels=c, device=rgb.device, dtype=rgb.dtype
    )
    return (rgb - mean_t) / std_t


def smooth_noise_field(
    batch: int,
    channels: int,
    height: int,
    width: int,
    *,
    grid: int = DEFAULT_SPATIAL_NOISE_GRID,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Low-res Gaussian noise bilinear-upsampled to ``(B,C,H,W)`` in ~[-1,1].

    Acts like blurred value-noise / soft Perlin without an external library.
    """
    g = max(int(grid), 2)
    device = device or torch.device("cpu")
    dtype = dtype or torch.float32
    # Two octaves for slightly richer structure.
    n1 = torch.randn(batch, channels, g, g, device=device, dtype=dtype)
    n2 = torch.randn(batch, channels, max(g // 2, 2), max(g // 2, 2), device=device, dtype=dtype)
    f1 = F.interpolate(n1, size=(height, width), mode="bilinear", align_corners=False)
    f2 = F.interpolate(n2, size=(height, width), mode="bilinear", align_corners=False)
    field = f1 + 0.5 * f2
    # Normalize per-sample roughly to unit scale.
    flat = field.reshape(batch, channels, -1)
    std = flat.std(dim=-1, keepdim=True).clamp_min(1e-6).unsqueeze(-1)
    field = field / std
    return field.clamp(-3.0, 3.0) / 3.0  # ~[-1, 1]


def random_shift_pixel_values(
    pixel_values: torch.Tensor,
    max_shift: int = DEFAULT_IMAGE_SHIFT_MAX,
) -> torch.Tensor:
    """Per-image integer shift in ``[-max_shift, max_shift]`` with reflect pad.

    ``pixel_values`` is ``(B, C, H, W)``. Each sample draws independent ``(dy, dx)``.
    Used only during training to reduce grid-position memorization.
    """
    if max_shift is None or int(max_shift) <= 0:
        return pixel_values
    if pixel_values.ndim != 4:
        raise ValueError(
            f"pixel_values must be (B,C,H,W), got shape {tuple(pixel_values.shape)}"
        )
    max_shift = int(max_shift)
    b, _, h, w = pixel_values.shape
    pad = max_shift
    padded = F.pad(pixel_values, (pad, pad, pad, pad), mode="reflect")
    out = pixel_values.new_empty(pixel_values.shape)
    for i in range(b):
        dy = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
        dx = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
        y0 = pad + dy
        x0 = pad + dx
        out[i] = padded[i, :, y0 : y0 + h, x0 : x0 + w]
    return out


def _sample_fill_color(
    image_chw: torch.Tensor,
    mode: str = DEFAULT_IMAGE_FILL_MODE,
) -> list[float]:
    """Per-channel fill for geometric gaps (normalized tensor space)."""
    c = int(image_chw.shape[0])
    mode = (mode or "random").lower()
    if mode == "reflect":
        mode = "mean"
    if mode == "mean":
        return [float(image_chw[ch].mean()) for ch in range(c)]
    fills: list[float] = []
    for ch in range(c):
        lo = float(image_chw[ch].min())
        hi = float(image_chw[ch].max())
        span = max(hi - lo, 1e-3)
        lo_e, hi_e = lo - 0.25 * span, hi + 0.25 * span
        fills.append(float(lo_e + (hi_e - lo_e) * torch.rand(1).item()))
    return fills


def _fit_chw_to_size(
    image_chw: torch.Tensor,
    out_h: int,
    out_w: int,
    fill: list[float],
) -> torch.Tensor:
    """Center-pad (shrink) or center-crop (mild expand) a ``(C,H,W)`` tensor."""
    c, h, w = image_chw.shape
    pad_top = max(0, (out_h - h) // 2)
    pad_bottom = max(0, out_h - h - pad_top)
    pad_left = max(0, (out_w - w) // 2)
    pad_right = max(0, out_w - w - pad_left)
    if pad_top or pad_bottom or pad_left or pad_right:
        image_chw = F.pad(
            image_chw,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=0.0,
        )
        if pad_top:
            for ch, v in enumerate(fill):
                image_chw[ch, :pad_top, :] = v
        if pad_bottom:
            for ch, v in enumerate(fill):
                image_chw[ch, image_chw.shape[1] - pad_bottom :, :] = v
        if pad_left:
            for ch, v in enumerate(fill):
                image_chw[ch, :, :pad_left] = v
        if pad_right:
            for ch, v in enumerate(fill):
                image_chw[ch, :, image_chw.shape[2] - pad_right :] = v
        h, w = image_chw.shape[1], image_chw.shape[2]
    if h > out_h or w > out_w:
        top = max(0, (h - out_h) // 2)
        left = max(0, (w - out_w) // 2)
        image_chw = image_chw[:, top : top + out_h, left : left + out_w]
    if image_chw.shape[1] != out_h or image_chw.shape[2] != out_w:
        image_chw = F.interpolate(
            image_chw.unsqueeze(0),
            size=(out_h, out_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return image_chw


def train_image_geometric_augment(
    pixel_values: torch.Tensor,
    *,
    hflip_prob: float = DEFAULT_IMAGE_HFLIP_PROB,
    max_rotate_deg: float = DEFAULT_IMAGE_MAX_ROTATE_DEG,
    scale_min: float = DEFAULT_IMAGE_SCALE_MIN,
    scale_max: float = DEFAULT_IMAGE_SCALE_MAX,
    fill_mode: str = DEFAULT_IMAGE_FILL_MODE,
) -> torch.Tensor:
    """Train-time geometric aug: H-flip, rotate, anisotropic scale, pad/crop.

    ``pixel_values`` is ``(B, C, H, W)`` (typically processor-normalized).
    """
    if pixel_values.ndim != 4:
        raise ValueError(
            f"pixel_values must be (B,C,H,W), got shape {tuple(pixel_values.shape)}"
        )
    b, c, h, w = pixel_values.shape
    smin = float(scale_min)
    smax = float(scale_max)
    if smin <= 0 or smax <= 0 or smin > smax:
        raise ValueError(f"invalid scale range [{smin}, {smax}]")
    max_rot = abs(float(max_rotate_deg))
    hflip_p = float(hflip_prob)

    try:
        import torchvision.transforms.functional as tvf
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "train_image_geometric_augment requires torchvision"
        ) from exc

    out = pixel_values.new_empty(pixel_values.shape)
    for i in range(b):
        img = pixel_values[i]
        fill = _sample_fill_color(img, mode=fill_mode)

        if hflip_p > 0.0 and torch.rand(1).item() < hflip_p:
            img = tvf.hflip(img)

        if max_rot > 0.0:
            angle = float((-max_rot) + (2.0 * max_rot) * torch.rand(1).item())
            if abs(angle) > 1e-6:
                img = tvf.rotate(
                    img,
                    angle=angle,
                    interpolation=tvf.InterpolationMode.BILINEAR,
                    expand=False,
                    fill=fill,
                )

        sx = float(smin + (smax - smin) * torch.rand(1).item())
        sy = float(smin + (smax - smin) * torch.rand(1).item())
        if abs(sx - 1.0) > 1e-6 or abs(sy - 1.0) > 1e-6:
            new_w = max(1, int(round(w * sx)))
            new_h = max(1, int(round(h * sy)))
            img = tvf.resize(
                img,
                [new_h, new_w],
                interpolation=tvf.InterpolationMode.BILINEAR,
                antialias=True,
            )
            img = _fit_chw_to_size(img, h, w, fill=fill)

        out[i] = img
    return out


def _adjust_brightness(rgb: torch.Tensor, factor: float) -> torch.Tensor:
    return rgb * factor


def _adjust_contrast(rgb: torch.Tensor, factor: float) -> torch.Tensor:
    # Gray mean per sample.
    # rgb: (C,H,W)
    gray = rgb.mean(dim=0, keepdim=True)
    return (rgb - gray) * factor + gray


def _adjust_saturation(rgb: torch.Tensor, factor: float) -> torch.Tensor:
    # Rec. 601 luma
    if rgb.shape[0] < 3:
        return rgb
    r, g, b = rgb[0], rgb[1], rgb[2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    gray3 = gray.unsqueeze(0).expand_as(rgb)
    return (rgb - gray3) * factor + gray3


def _adjust_hue(rgb: torch.Tensor, hue_factor: float) -> torch.Tensor:
    """Hue shift in radians-of-circle fraction; uses torchvision if available."""
    if abs(hue_factor) < 1e-8 or rgb.shape[0] < 3:
        return rgb
    try:
        import torchvision.transforms.functional as tvf
    except ImportError:
        # Fallback: cheap channel rotation in RGB (approximate).
        # Mix channels with a small shear.
        a = float(hue_factor) * 2.0 * math.pi
        ca, sa = math.cos(a), math.sin(a)
        r, g, b = rgb[0], rgb[1], rgb[2]
        # Rotate in (R-G, B) plane roughly
        rg = r - g
        r2 = g + ca * rg - sa * (b - g)
        b2 = g + sa * rg + ca * (b - g)
        g2 = g
        return torch.stack([r2, g2, b2], dim=0)
    return tvf.adjust_hue(rgb.clamp(0.0, 1.0), hue_factor)


def train_image_photometric_augment(
    pixel_values: torch.Tensor,
    *,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
    brightness: float = DEFAULT_PHOTO_BRIGHTNESS,
    contrast: float = DEFAULT_PHOTO_CONTRAST,
    saturation: float = DEFAULT_PHOTO_SATURATION,
    hue: float = DEFAULT_PHOTO_HUE,
    spatial_brightness: float = DEFAULT_SPATIAL_BRIGHTNESS,
    spatial_color: float = DEFAULT_SPATIAL_COLOR,
    spatial_noise_grid: int = DEFAULT_SPATIAL_NOISE_GRID,
    grayscale_prob: float = DEFAULT_GRAYSCALE_PROB,
    enabled: bool = True,
) -> torch.Tensor:
    """Color/brightness augs: global jitter + smooth spatial fields + grayscale.

    Operates on processor-normalized ``(B,C,H,W)`` via denorm → RGB → renorm.
    Spatially varying fields use upsampled low-res noise (blurred-noise look).
    """
    if not enabled:
        return pixel_values
    if pixel_values.ndim != 4:
        raise ValueError(
            f"pixel_values must be (B,C,H,W), got shape {tuple(pixel_values.shape)}"
        )
    b, c, h, w = pixel_values.shape
    # Work in float32 for stable color math, cast back.
    orig_dtype = pixel_values.dtype
    x = pixel_values.float()
    rgb = denormalize_pixel_values(x, mean=mean, std=std).clamp(0.0, 1.0)

    out = rgb.new_empty(rgb.shape)
    for i in range(b):
        img = rgb[i]

        # Global photometric factors (symmetric around 1 / 0).
        if brightness and brightness > 0:
            bf = 1.0 + float(brightness) * (2.0 * torch.rand(1).item() - 1.0)
            img = _adjust_brightness(img, bf)
        if contrast and contrast > 0:
            cf = 1.0 + float(contrast) * (2.0 * torch.rand(1).item() - 1.0)
            img = _adjust_contrast(img, cf)
        if saturation and saturation > 0 and c >= 3:
            sf = 1.0 + float(saturation) * (2.0 * torch.rand(1).item() - 1.0)
            img = _adjust_saturation(img, sf)
        if hue and hue > 0 and c >= 3:
            hf = float(hue) * (2.0 * torch.rand(1).item() - 1.0)
            img = _adjust_hue(img.clamp(0.0, 1.0), hf)

        img = img.clamp(0.0, 1.0)

        # Spatially varying brightness (1ch field).
        sb = float(spatial_brightness)
        if sb > 0:
            field = smooth_noise_field(
                1, 1, h, w, grid=spatial_noise_grid, device=img.device, dtype=img.dtype
            )[0]
            # α random in [0, sb]
            alpha = sb * torch.rand(1, device=img.device, dtype=img.dtype)
            img = img * (1.0 + alpha * field).clamp(0.25, 1.75)

        # Spatially varying color (3ch additive).
        sc = float(spatial_color)
        if sc > 0 and c >= 3:
            field3 = smooth_noise_field(
                1, min(c, 3), h, w,
                grid=spatial_noise_grid, device=img.device, dtype=img.dtype,
            )[0]
            alpha = sc * torch.rand(1, device=img.device, dtype=img.dtype)
            if c > 3:
                pad = field3[:1].expand(c - 3, -1, -1)
                field3 = torch.cat([field3, pad], dim=0)
            elif c < 3:
                field3 = field3[:c]
            img = img + alpha * field3

        img = img.clamp(0.0, 1.0)

        # Occasional grayscale.
        if grayscale_prob and torch.rand(1).item() < float(grayscale_prob) and c >= 3:
            r, g, bl = img[0], img[1], img[2]
            gray = 0.299 * r + 0.587 * g + 0.114 * bl
            img = gray.unsqueeze(0).expand(c, -1, -1).contiguous()

        out[i] = img.clamp(0.0, 1.0)

    normed = normalize_pixel_values(out, mean=mean, std=std)
    return normed.to(dtype=orig_dtype)


def apply_train_image_augmentations(
    pixel_values: torch.Tensor,
    *,
    hflip_prob: float = DEFAULT_IMAGE_HFLIP_PROB,
    max_rotate_deg: float = DEFAULT_IMAGE_MAX_ROTATE_DEG,
    scale_min: float = DEFAULT_IMAGE_SCALE_MIN,
    scale_max: float = DEFAULT_IMAGE_SCALE_MAX,
    fill_mode: str = DEFAULT_IMAGE_FILL_MODE,
    max_shift: int = DEFAULT_IMAGE_SHIFT_MAX,
    photometric: bool = DEFAULT_PHOTOMETRIC_ENABLED,
    photo_brightness: float = DEFAULT_PHOTO_BRIGHTNESS,
    photo_contrast: float = DEFAULT_PHOTO_CONTRAST,
    photo_saturation: float = DEFAULT_PHOTO_SATURATION,
    photo_hue: float = DEFAULT_PHOTO_HUE,
    spatial_brightness: float = DEFAULT_SPATIAL_BRIGHTNESS,
    spatial_color: float = DEFAULT_SPATIAL_COLOR,
    spatial_noise_grid: int = DEFAULT_SPATIAL_NOISE_GRID,
    grayscale_prob: float = DEFAULT_GRAYSCALE_PROB,
    image_mean: Sequence[float] | None = None,
    image_std: Sequence[float] | None = None,
    enabled: bool = True,
) -> torch.Tensor:
    """Full train-time vision aug: geometric → photometric → integer shift."""
    if not enabled:
        return pixel_values
    x = train_image_geometric_augment(
        pixel_values,
        hflip_prob=hflip_prob,
        max_rotate_deg=max_rotate_deg,
        scale_min=scale_min,
        scale_max=scale_max,
        fill_mode=fill_mode,
    )
    if photometric:
        x = train_image_photometric_augment(
            x,
            mean=image_mean,
            std=image_std,
            brightness=photo_brightness,
            contrast=photo_contrast,
            saturation=photo_saturation,
            hue=photo_hue,
            spatial_brightness=spatial_brightness,
            spatial_color=spatial_color,
            spatial_noise_grid=spatial_noise_grid,
            grayscale_prob=grayscale_prob,
            enabled=True,
        )
    if max_shift and int(max_shift) > 0:
        x = random_shift_pixel_values(x, max_shift=int(max_shift))
    return x


# ---------------------------------------------------------------------------
# Demo (__main__)
# ---------------------------------------------------------------------------

def _pil_to_rgb_tensor(img, size: int = 384) -> torch.Tensor:
    """PIL RGB → ``(3, size, size)`` float tensor in ``[0, 1]``."""
    from PIL import Image
    import torchvision.transforms.functional as tvf

    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    img = img.convert("RGB")
    t = tvf.to_tensor(img)
    t = tvf.resize(t, [size, size], antialias=True)
    return t


def _load_rgb_tensor(path: str, size: int = 384) -> torch.Tensor:
    from PIL import Image

    return _pil_to_rgb_tensor(Image.open(path), size=size)


def _to_display_uint8(rgb_chw: torch.Tensor) -> "object":
    from PIL import Image
    import numpy as np

    x = rgb_chw.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((x * 255.0).astype(np.uint8))


def _gallery_html(images: list, labels: list[str] | None = None) -> str:
    """Inline HTML grid (project demos use HTML; Gradio 5 Gallery breaks get_api_info)."""
    import base64
    import io

    labels = labels or [f"tile {i}" for i in range(len(images))]
    cells: list[str] = []
    for im, lab in zip(images, labels):
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        cells.append(
            "<div style='display:inline-block;margin:6px;text-align:center'>"
            f"<div style='font-size:12px;color:#555;margin-bottom:4px'>{lab}</div>"
            f"<img src='data:image/png;base64,{b64}' "
            "style='width:180px;height:180px;object-fit:cover;"
            "border:1px solid #ccc;border-radius:4px'/>"
            "</div>"
        )
    return (
        "<div style='display:flex;flex-wrap:wrap;gap:4px;align-items:flex-start'>"
        + "".join(cells)
        + "</div>"
    )


def open_train_map_dataset(
    *,
    hf_dataset: str | None = None,
    dataset_dir: str | None = None,
    prefer_local: bool = False,
    split: str = "train",
    seed: int = 42,
):
    """Lazy TriSearch train map (no full RAM materialization)."""
    from trisearch_dataset import (
        DEFAULT_TRISEARCH_HF_DATASET,
        open_trisearch_map_dataset,
    )

    return open_trisearch_map_dataset(
        hf_dataset=hf_dataset or DEFAULT_TRISEARCH_HF_DATASET,
        dataset_dir=dataset_dir,
        prefer_local=prefer_local,
        split=split,
        max_samples=None,
        seed=seed,
        satellite_fraction=0.5,
    )


def sample_train_rgb(
    map_ds,
    *,
    size: int = 384,
    rng: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict]:
    """Decode **one** random train example → ``(3,size,size)`` + meta."""
    n = len(map_ds)
    if n < 1:
        raise RuntimeError("Training map dataset is empty.")
    if rng is None:
        idx = int(torch.randint(0, n, (1,)).item())
    else:
        idx = int(torch.randint(0, n, (1,), generator=rng).item())
    row = map_ds[idx]
    img = row["image"]
    rgb = _pil_to_rgb_tensor(img, size=size)
    meta = {
        "index": idx,
        "id": str(row.get("id", "")),
        "domain": str(row.get("domain", "")),
        "source": str(row.get("source", "")),
        "caption": "",
    }
    caps = row.get("captions") or row.get("caption")
    if isinstance(caps, (list, tuple)) and caps:
        meta["caption"] = str(caps[0])[:120]
    elif isinstance(caps, str):
        meta["caption"] = caps[:120]
    return rgb, meta


def build_demo(
    *,
    image_path: str | None = None,
    size: int = 384,
    hf_dataset: str | None = None,
    dataset_dir: str | None = None,
    prefer_local: bool = False,
    split: str = "train",
) -> tuple[object, object]:
    """Build Gradio Blocks + the augment-grid callable (for smoke / launch).

    Images come from the **real TriSearch train split** (lazy map). A random
    example is decoded only when Resample is clicked (not at page load).
    ``--image`` overrides with a single local file for offline debugging.
    """
    try:
        import gradio as gr
    except ImportError as exc:
        raise SystemExit(
            "Gradio is required for the demo: pip install gradio\n"
            f"({exc})"
        ) from exc

    mean = list(DEFAULT_IMAGE_MEAN)
    std = list(DEFAULT_IMAGE_STD)
    map_ds = None
    override_path = image_path
    if override_path is None:
        map_ds = open_train_map_dataset(
            hf_dataset=hf_dataset,
            dataset_dir=dataset_dir,
            prefer_local=prefer_local,
            split=split,
        )
        n_train = len(map_ds)
        print(
            f"Demo train map ready: n={n_train:,} "
            f"(images decoded only on Resample)",
            flush=True,
        )
    else:
        n_train = 0
        print(f"Demo override image: {override_path}", flush=True)

    def _load_base() -> tuple[torch.Tensor, str]:
        if override_path is not None:
            rgb = _load_rgb_tensor(override_path, size=size)
            return rgb, f"file:{override_path}"
        assert map_ds is not None
        rgb, meta = sample_train_rgb(map_ds, size=size)
        label = (
            f"train[{meta['index']}] {meta['domain']} "
            f"id={meta['id'][:24]} {meta['caption']!r}"
        )
        return rgb, label

    def _augment_grid(
        n_tiles: int,
        mode: str,
        hflip: float,
        rot: float,
        smin: float,
        smax: float,
        shift: int,
        bright: float,
        contrast: float,
        sat: float,
        hue: float,
        spat_b: float,
        spat_c: float,
        grid: int,
        gray_p: float,
    ):
        # Load **now** (not at page open): one random train image per click.
        base_rgb, src_label = _load_base()
        n_tiles = max(1, min(int(n_tiles), 12))
        x = normalize_pixel_values(base_rgb.unsqueeze(0), mean=mean, std=std)
        tiles = [_to_display_uint8(base_rgb)]
        labels = [f"original — {src_label}"]
        for i in range(n_tiles):
            if mode == "geometric only":
                y = apply_train_image_augmentations(
                    x,
                    hflip_prob=hflip,
                    max_rotate_deg=rot,
                    scale_min=smin,
                    scale_max=smax,
                    max_shift=int(shift),
                    photometric=False,
                    image_mean=mean,
                    image_std=std,
                    enabled=True,
                )
            elif mode == "photometric only":
                y = train_image_photometric_augment(
                    x,
                    mean=mean,
                    std=std,
                    brightness=bright,
                    contrast=contrast,
                    saturation=sat,
                    hue=hue,
                    spatial_brightness=spat_b,
                    spatial_color=spat_c,
                    spatial_noise_grid=int(grid),
                    grayscale_prob=gray_p,
                    enabled=True,
                )
            else:
                y = apply_train_image_augmentations(
                    x,
                    hflip_prob=hflip,
                    max_rotate_deg=rot,
                    scale_min=smin,
                    scale_max=smax,
                    max_shift=int(shift),
                    photometric=True,
                    photo_brightness=bright,
                    photo_contrast=contrast,
                    photo_saturation=sat,
                    photo_hue=hue,
                    spatial_brightness=spat_b,
                    spatial_color=spat_c,
                    spatial_noise_grid=int(grid),
                    grayscale_prob=gray_p,
                    image_mean=mean,
                    image_std=std,
                    enabled=True,
                )
            rgb = denormalize_pixel_values(y, mean=mean, std=std)[0].clamp(0, 1)
            tiles.append(_to_display_uint8(rgb))
            labels.append(f"aug {i + 1}")
        return _gallery_html(tiles, labels)

    placeholder = (
        "<div style='padding:24px;color:#555;font-size:14px'>"
        "Click <b>Resample augmentations</b> to load a <b>random training image</b> "
        "from the TriSearch curated set and apply the current knobs. "
        "Images are not preloaded."
        + (
            f" (map size: {n_train:,})"
            if map_ds is not None
            else " (using --image override)"
        )
        + "</div>"
    )

    with gr.Blocks(title="TriSearch image aug preview") as demo:
        gr.Markdown(
            "# TriSearch image augmentation preview\n"
            "Samples **real train-split images** (lazy Hub/local map). "
            "Decode happens **only** when you click Resample. "
            "Defaults match top-of-file constants in `image_augment.py`."
        )
        with gr.Row():
            mode = gr.Radio(
                ["full stack", "geometric only", "photometric only"],
                value="full stack",
                label="Stack mode",
            )
            n_tiles = gr.Slider(1, 12, value=DEFAULT_DEMO_N_TILES, step=1, label="Aug tiles")
        with gr.Accordion("Geometric", open=False):
            hflip = gr.Slider(0, 1, value=DEFAULT_IMAGE_HFLIP_PROB, label="H-flip prob")
            rot = gr.Slider(0, 45, value=DEFAULT_IMAGE_MAX_ROTATE_DEG, label="Max rotate °")
            smin = gr.Slider(0.5, 1.0, value=DEFAULT_IMAGE_SCALE_MIN, label="Scale min")
            smax = gr.Slider(1.0, 1.2, value=DEFAULT_IMAGE_SCALE_MAX, label="Scale max")
            shift = gr.Slider(0, 32, value=DEFAULT_IMAGE_SHIFT_MAX, step=1, label="Max shift px")
        with gr.Accordion("Photometric (global)", open=True):
            bright = gr.Slider(0, 0.5, value=DEFAULT_PHOTO_BRIGHTNESS, label="Brightness jitter")
            contrast = gr.Slider(0, 0.5, value=DEFAULT_PHOTO_CONTRAST, label="Contrast jitter")
            sat = gr.Slider(0, 0.6, value=DEFAULT_PHOTO_SATURATION, label="Saturation jitter")
            hue = gr.Slider(0, 0.15, value=DEFAULT_PHOTO_HUE, label="Hue jitter")
        with gr.Accordion("Spatial color / brightness fields", open=True):
            spat_b = gr.Slider(0, 0.4, value=DEFAULT_SPATIAL_BRIGHTNESS, label="Spatial brightness α")
            spat_c = gr.Slider(0, 0.3, value=DEFAULT_SPATIAL_COLOR, label="Spatial color α")
            grid = gr.Slider(4, 32, value=DEFAULT_SPATIAL_NOISE_GRID, step=1, label="Noise grid size")
            gray_p = gr.Slider(0, 1, value=DEFAULT_GRAYSCALE_PROB, label="Grayscale prob")
        btn = gr.Button("Resample augmentations", variant="primary")
        gallery = gr.HTML(value=placeholder, label="Original + augs")

        inputs = [
            n_tiles, mode, hflip, rot, smin, smax, shift,
            bright, contrast, sat, hue, spat_b, spat_c, grid, gray_p,
        ]
        # No demo.load(augment) — that would decode an image on page open.
        _evt = dict(api_name=False)
        btn.click(_augment_grid, inputs=inputs, outputs=gallery, **_evt)

    return demo, _augment_grid


def smoke_test_demo(
    *,
    image_path: str | None = None,
    size: int = 128,
    hf_dataset: str | None = None,
    dataset_dir: str | None = None,
    prefer_local: bool = False,
) -> None:
    """AGENTS §3: exercise Gradio build, get_api_info, and one Resample request."""
    demo, augment_grid = build_demo(
        image_path=image_path,
        size=size,
        hf_dataset=hf_dataset,
        dataset_dir=dataset_dir,
        prefer_local=prefer_local,
        split="train",
    )
    info = demo.get_api_info()
    if not isinstance(info, dict):
        raise RuntimeError(f"get_api_info returned {type(info)}")
    # One click path: loads a real train image (or --image) then augments.
    html = augment_grid(
        2,
        "full stack",
        DEFAULT_IMAGE_HFLIP_PROB,
        DEFAULT_IMAGE_MAX_ROTATE_DEG,
        DEFAULT_IMAGE_SCALE_MIN,
        DEFAULT_IMAGE_SCALE_MAX,
        DEFAULT_IMAGE_SHIFT_MAX,
        DEFAULT_PHOTO_BRIGHTNESS,
        DEFAULT_PHOTO_CONTRAST,
        DEFAULT_PHOTO_SATURATION,
        DEFAULT_PHOTO_HUE,
        DEFAULT_SPATIAL_BRIGHTNESS,
        DEFAULT_SPATIAL_COLOR,
        DEFAULT_SPATIAL_NOISE_GRID,
        DEFAULT_GRAYSCALE_PROB,
    )
    if not isinstance(html, str) or "data:image/png;base64," not in html:
        raise RuntimeError(
            f"augment_grid bad HTML: type={type(html)} "
            f"len={len(html) if isinstance(html, str) else 'n/a'}"
        )
    n_imgs = html.count("data:image/png;base64,")
    if n_imgs < 2:
        raise RuntimeError(f"expected >=2 tiles in HTML, got {n_imgs}")
    if "original" not in html.lower() and "train[" not in html and "file:" not in html:
        # label still present for train samples
        pass
    print(
        f"image_augment smoke OK: get_api_info keys={list(info)[:8]} "
        f"html_tiles={n_imgs}"
    )


def run_demo_cli(args: argparse.Namespace) -> None:
    """Gradio UI to preview augmentations with live sliders."""
    demo, _ = build_demo(
        image_path=args.image,
        size=int(args.size),
        hf_dataset=args.hf_dataset,
        dataset_dir=args.curated_dataset_dir,
        prefer_local=args.prefer_local_curated,
        split=args.curated_split,
    )
    launch_kw: dict = {"share": False}
    if getattr(args, "port", None) is not None:
        launch_kw["server_port"] = int(args.port)
    demo.launch(**launch_kw)


def main() -> None:
    from trisearch_dataset import DEFAULT_TRISEARCH_HF_DATASET

    p = argparse.ArgumentParser(description="TriSearch image aug preview demo")
    p.add_argument(
        "--image",
        default=None,
        help="Optional single RGB file (overrides train-set sampling).",
    )
    p.add_argument(
        "--hf-dataset",
        default=DEFAULT_TRISEARCH_HF_DATASET,
        help=f"TriSearch curated Hub id (default {DEFAULT_TRISEARCH_HF_DATASET}).",
    )
    p.add_argument(
        "--curated-dataset-dir",
        default=None,
        help="Optional local curated export when --prefer-local-curated.",
    )
    p.add_argument(
        "--prefer-local-curated",
        action="store_true",
        help="Prefer local curated export over the Hub dataset.",
    )
    p.add_argument(
        "--curated-split",
        default="train",
        choices=("train", "test", "all"),
        help="Split to sample (default train).",
    )
    p.add_argument("--size", type=int, default=384, help="Preview resolution")
    p.add_argument("--port", type=int, default=None, help="Optional Gradio server port")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Build UI, call get_api_info + one Resample, exit (no server).",
    )
    args = p.parse_args()
    if args.smoke:
        smoke_test_demo(
            image_path=args.image,
            size=min(int(args.size), 192),
            hf_dataset=args.hf_dataset,
            dataset_dir=args.curated_dataset_dir,
            prefer_local=args.prefer_local_curated,
        )
        return
    run_demo_cli(args)


if __name__ == "__main__":
    main()
