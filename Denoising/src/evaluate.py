from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from skimage.metrics import structural_similarity

from .config import PROJECT_ROOT
from .metrics import mse, psnr, ssim, summarize_metrics


def evaluate_dataset(model: torch.nn.Module, paths: list[str], clean_paths: list[str], device: torch.device) -> tuple[list[float], list[float], list[float], list[float], list[float], list[float]]:
    model.eval()
    baseline_mse, baseline_psnr, baseline_ssim = [], [], []
    model_mse, model_psnr, model_ssim = [], [], []

    for idx, img_path in enumerate(paths):
        clean = np.asarray(__import__("PIL").Image.open(clean_paths[idx]), dtype=np.float32) / 255.0
        corrupted = np.asarray(__import__("PIL").Image.open(img_path), dtype=np.float32) / 255.0
        x = torch.from_numpy(corrupted.reshape(1, 1, 28, 28)).float().to(device)
        with torch.no_grad():
            out = model(x).cpu().numpy()[0, 0]

        baseline_mse.append(mse(corrupted, clean))
        baseline_psnr.append(psnr(corrupted, clean))
        baseline_ssim.append(ssim(corrupted, clean))

        model_mse.append(mse(out, clean))
        model_psnr.append(psnr(out, clean))
        model_ssim.append(ssim(out, clean))

    return baseline_mse, baseline_psnr, baseline_ssim, model_mse, model_psnr, model_ssim


def compute_summary_rows(dataset_name: str, baseline: tuple[list[float], list[float], list[float]], model: tuple[list[float], list[float], list[float]]) -> list[dict]:
    rows = []
    metric_names = ["MSE", "PSNR", "SSIM"]
    for stage, values in [("Baseline", baseline), ("Model", model)]:
        metrics = [summarize_metrics(vals) for vals in values]
        rows.append({
            "Dataset": dataset_name,
            "Stage": stage,
            "MSE": metrics[0][0],
            "PSNR": metrics[1][0],
            "SSIM": metrics[2][0],
        })
    return rows


def save_result_tables(summary_rows: list[dict], per_image_rows: list[dict]) -> None:
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(PROJECT_ROOT / "outputs" / "results" / "metrics.csv", index=False)
    per_image_df = pd.DataFrame(per_image_rows)
    per_image_df.to_csv(PROJECT_ROOT / "outputs" / "results" / "per_image_metrics.csv", index=False)
