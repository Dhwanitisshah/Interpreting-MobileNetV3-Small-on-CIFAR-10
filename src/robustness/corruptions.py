"""Image corruptions for explanation-robustness evaluation (Phase 7).

Corruptions are applied to the 224x224 UN-NORMALIZED uint8 image (HWC, RGB,
[0, 255]) -- i.e. before `torchvision.transforms.Normalize` -- then the result
is converted back to a normalized model-input tensor. This ordering matters:
corrupting in normalized (mean/std-shifted) float space would apply noise,
blur, and compression artifacts on the wrong intensity scale (ImageNet
normalization is not [0, 1], so e.g. jpeg_compression's quantization and
brightness's clipping would behave differently, and the corruption would no
longer resemble a real-world camera/sensor artifact on raw pixels).

Ordering: uint8 [0,255] image -> apply_corruption -> uint8 [0,255] image ->
ToTensor (-> [0,1] float) -> Normalize(IMAGENET_MEAN, IMAGENET_STD) -> model.

Primary implementation is `imagecorruptions` (Hendrycks & Dietterich, 2019 /
Michaelis et al., 2019), matching the standard ImageNet-C benchmark. If it is
not importable (e.g. a Windows environment where its opencv dependency fails
to install), a manual numpy/PIL fallback covering the same six corruption
names is used instead -- visually similar in spirit but not bit-identical to
the published ImageNet-C corruptions.
"""

import io
from typing import Tuple

import numpy as np
from PIL import Image, ImageFilter

CORRUPTIONS = (
    "gaussian_noise",
    "motion_blur",
    "jpeg_compression",
    "brightness",
    "contrast",
    "defocus_blur",
)

try:
    from imagecorruptions import corrupt as _ic_corrupt

    _HAVE_IMAGECORRUPTIONS = True
except ImportError:
    _HAVE_IMAGECORRUPTIONS = False


def _check_severity(severity: int) -> None:
    if severity not in (1, 2, 3, 4, 5):
        raise ValueError(f"severity must be an int in 1..5, got {severity!r}")


# ---------------------------------------------------------------------------
# Manual numpy/PIL fallback (used only if `imagecorruptions` is unavailable).
# ---------------------------------------------------------------------------

def _manual_gaussian_noise(img: np.ndarray, severity: int) -> np.ndarray:
    sigma = [0.04, 0.08, 0.12, 0.18, 0.26][severity - 1] * 255.0
    rng = np.random.default_rng(severity)
    noisy = img.astype(np.float64) + rng.normal(0, sigma, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _manual_motion_blur(img: np.ndarray, severity: int) -> np.ndarray:
    ksize = [5, 9, 13, 17, 21][severity - 1]
    kernel = np.zeros((ksize, ksize), dtype=np.float64)
    kernel[ksize // 2, :] = 1.0
    kernel /= kernel.sum()

    from scipy.ndimage import convolve

    out = np.stack(
        [convolve(img[..., c].astype(np.float64), kernel, mode="nearest") for c in range(img.shape[-1])],
        axis=-1,
    )
    return np.clip(out, 0, 255).astype(np.uint8)


def _manual_jpeg_compression(img: np.ndarray, severity: int) -> np.ndarray:
    quality = [80, 60, 40, 20, 10][severity - 1]
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def _manual_brightness(img: np.ndarray, severity: int) -> np.ndarray:
    # Low-light: darken (subtract, in HSV V-channel) rather than brighten.
    factor = [0.9, 0.75, 0.6, 0.45, 0.3][severity - 1]
    hsv = np.array(Image.fromarray(img).convert("HSV"), dtype=np.float64)
    hsv[..., 2] = np.clip(hsv[..., 2] * factor, 0, 255)
    return np.array(Image.fromarray(hsv.astype(np.uint8), mode="HSV").convert("RGB"))


def _manual_contrast(img: np.ndarray, severity: int) -> np.ndarray:
    factor = [0.75, 0.6, 0.45, 0.3, 0.15][severity - 1]
    mean = img.astype(np.float64).mean(axis=(0, 1), keepdims=True)
    out = (img.astype(np.float64) - mean) * factor + mean
    return np.clip(out, 0, 255).astype(np.uint8)


def _manual_defocus_blur(img: np.ndarray, severity: int) -> np.ndarray:
    radius = [1.5, 2.5, 3.5, 5.0, 6.5][severity - 1]
    blurred = Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=radius))
    return np.array(blurred)


_MANUAL_FNS = {
    "gaussian_noise": _manual_gaussian_noise,
    "motion_blur": _manual_motion_blur,
    "jpeg_compression": _manual_jpeg_compression,
    "brightness": _manual_brightness,
    "contrast": _manual_contrast,
    "defocus_blur": _manual_defocus_blur,
}


def apply_corruption(image_uint8_hwc: np.ndarray, name: str, severity: int) -> np.ndarray:
    """Apply one named corruption at the given severity (1-5) to an HWC uint8
    RGB image. Returns a new uint8 array of the same shape."""
    if name not in CORRUPTIONS:
        raise ValueError(f"Unknown corruption '{name}'. Expected one of {CORRUPTIONS}.")
    _check_severity(severity)
    if image_uint8_hwc.dtype != np.uint8:
        raise ValueError(f"apply_corruption expects a uint8 image, got dtype={image_uint8_hwc.dtype}")

    if _HAVE_IMAGECORRUPTIONS:
        out = _ic_corrupt(image_uint8_hwc, corruption_name=name, severity=severity)
        return np.asarray(out, dtype=np.uint8)

    return _MANUAL_FNS[name](image_uint8_hwc, severity)
