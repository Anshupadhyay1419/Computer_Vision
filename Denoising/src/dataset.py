from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms

from .config import (
    CLEAN_DIR,
    GAUSSIAN_DIR,
    NUM_IMAGES,
    PROJECT_ROOT,
    SALT_PEPPER_AMOUNT,
    SALT_PEPPER_DIR,
    SEED,
    SPECKLE_DIR,
    SPECKLE_VARIANCE,
    TRAIN_RATIO,
    GAUSSIAN_KERNEL_SIZE,
    GAUSSIAN_SIGMA,
)
from .noise import gaussian_blur_image, salt_pepper_noise, speckle_noise, to_pil


def set_seed(seed: int = SEED) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_mnist_subset() -> tuple[list[int], list[tuple[int, int]]]:
    dataset = datasets.MNIST(root=str(PROJECT_ROOT / "data" / "raw"), train=True, download=True)
    idxs = list(range(len(dataset)))
    rng = np.random.default_rng(SEED)
    selected = rng.choice(idxs, size=NUM_IMAGES, replace=False)
    selected = sorted(selected.tolist())
    items = []
    for idx in selected:
        image, label = dataset[idx]
        items.append((idx, int(label)))
    return selected, items


def save_clean_dataset(selected_items: list[tuple[int, int]]) -> list[Path]:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    dataset = datasets.MNIST(root=str(PROJECT_ROOT / "data" / "raw"), train=True, download=True)
    for i, (mnist_index, _) in enumerate(selected_items):
        image, _ = dataset[mnist_index]
        arr = np.asarray(image, dtype=np.float32) / 255.0
        to_pil(arr, CLEAN_DIR / f"{i:03d}.png")
    return sorted(CLEAN_DIR.glob("*.png"))


def create_corrupted_datasets(selected_items: list[tuple[int, int]]) -> None:
    GAUSSIAN_DIR.mkdir(parents=True, exist_ok=True)
    SALT_PEPPER_DIR.mkdir(parents=True, exist_ok=True)
    SPECKLE_DIR.mkdir(parents=True, exist_ok=True)

    dataset = datasets.MNIST(root=str(PROJECT_ROOT / "data" / "raw"), train=True, download=True)
    for i, (mnist_index, label) in enumerate(selected_items):
        image, _ = dataset[mnist_index]
        arr = np.asarray(image, dtype=np.float32) / 255.0

        blurred = gaussian_blur_image(arr, kernel_size=GAUSSIAN_KERNEL_SIZE, sigma=GAUSSIAN_SIGMA)
        to_pil(blurred, GAUSSIAN_DIR / f"{i:03d}.png")

        noisy_sp = salt_pepper_noise(arr, amount=SALT_PEPPER_AMOUNT, seed=SEED + i)
        to_pil(noisy_sp, SALT_PEPPER_DIR / f"{i:03d}.png")

        noisy_spec = speckle_noise(arr, variance=SPECKLE_VARIANCE, seed=SEED + i + 1000)
        to_pil(noisy_spec, SPECKLE_DIR / f"{i:03d}.png")


def build_metadata(selected_items: list[tuple[int, int]]) -> pd.DataFrame:
    rows = []
    for i, (mnist_index, label) in enumerate(selected_items):
        rows.append(
            {
                "index": i,
                "mnist_index": mnist_index,
                "mnist_label": label,
                "clean_path": str(CLEAN_DIR / f"{i:03d}.png"),
                "gaussian_blur_path": str(GAUSSIAN_DIR / f"{i:03d}.png"),
                "salt_pepper_path": str(SALT_PEPPER_DIR / f"{i:03d}.png"),
                "speckle_path": str(SPECKLE_DIR / f"{i:03d}.png"),
            }
        )
    return pd.DataFrame(rows)


def create_split(metadata: pd.DataFrame) -> tuple[list[int], list[int]]:
    n_train = int(len(metadata) * TRAIN_RATIO)
    train_idx = metadata["index"].tolist()[:n_train]
    test_idx = metadata["index"].tolist()[n_train:]
    pd.DataFrame({"index": train_idx}).to_csv(PROJECT_ROOT / "train_indices.csv", index=False)
    pd.DataFrame({"index": test_idx}).to_csv(PROJECT_ROOT / "test_indices.csv", index=False)
    return train_idx, test_idx


def ensure_dataset_counts() -> None:
    for folder in [CLEAN_DIR, GAUSSIAN_DIR, SALT_PEPPER_DIR, SPECKLE_DIR]:
        files = sorted(folder.glob("*.png"))
        if len(files) != NUM_IMAGES:
            raise ValueError(f"{folder} has {len(files)} files, expected {NUM_IMAGES}.")


def load_image_array(path: str | Path) -> np.ndarray:
    img = np.asarray(__import__("PIL").Image.open(path), dtype=np.float32) / 255.0
    if img.ndim == 3:
        img = img[..., 0]
    return img.astype(np.float32)
