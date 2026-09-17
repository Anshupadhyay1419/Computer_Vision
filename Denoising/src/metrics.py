import math
from typing import Iterable, Sequence

import numpy as np
from skimage.metrics import structural_similarity


def mse(pred: Sequence[float] | np.ndarray, target: Sequence[float] | np.ndarray) -> float:
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    return float(np.mean((p - t) ** 2))


def psnr(pred: Sequence[float] | np.ndarray, target: Sequence[float] | np.ndarray, data_range: float = 1.0) -> float:
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    err = np.mean((p - t) ** 2)
    eps = 1e-12
    return float(10 * math.log10((data_range ** 2) / max(err, eps)))


def ssim(pred: Sequence[float] | np.ndarray, target: Sequence[float] | np.ndarray, data_range: float = 1.0) -> float:
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if p.ndim == 2:
        p = p[np.newaxis, :, :]
        t = t[np.newaxis, :, :]
    return float(structural_similarity(p[0], t[0], data_range=data_range))


def summarize_metrics(values: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))
