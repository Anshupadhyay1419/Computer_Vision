from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def ensure_grayscale_image(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = arr.mean(axis=2)
    return arr / 255.0 if arr.max() > 1.0 else arr


def to_pil(arr: np.ndarray, path: str | Path) -> None:
    img = np.clip(np.asarray(arr, dtype=np.float32), 0.0, 1.0)
    img = (img * 255.0).astype(np.uint8)
    Image.fromarray(img, mode="L").save(path)


def gaussian_blur_image(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 2:
        arr = arr.squeeze()
    image_pil = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    blurred = image_pil.filter(ImageFilter.GaussianBlur(radius=sigma))
    return np.asarray(blurred, dtype=np.float32) / 255.0


def salt_pepper_noise(image: np.ndarray, amount: float = 0.10, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = np.asarray(image, dtype=np.float32).copy()
    flat = arr.ravel()
    n = flat.size
    num_noisy = max(1, int(round(amount * n)))
    idx = rng.choice(n, size=num_noisy, replace=False)
    vals = rng.random(num_noisy)
    for i, pos in enumerate(idx):
        flat[pos] = 0.0 if vals[i] < 0.5 else 1.0
    return flat.reshape(arr.shape)


def speckle_noise(image: np.ndarray, variance: float = 0.10, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = np.asarray(image, dtype=np.float32).copy()
    noise = rng.normal(0.0, math.sqrt(variance), size=arr.shape)
    noisy = arr + arr * noise
    return np.clip(noisy, 0.0, 1.0)
