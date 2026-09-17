from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import PLOTS_DIR, PROJECT_ROOT


def plot_training_loss(history: list[float]) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(history) + 1), history, marker="o")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "training_loss.png", dpi=200)
    plt.close()


def plot_metric_comparison(summary_df: pd.DataFrame, metric: str, output_name: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    filtered = summary_df[summary_df["Metric"] == metric]
    labels = filtered["Dataset"].tolist()
    baseline = filtered["Baseline"].tolist()
    model = filtered["Model"].tolist()
    x = np.arange(len(labels))

    plt.figure(figsize=(9, 5.5))
    widths = 0.35
    plt.bar(x - widths / 2, baseline, width=widths, label="Baseline")
    plt.bar(x + widths / 2, model, width=widths, label="Model")
    plt.xticks(x, labels)
    plt.title(f"{metric} Comparison")
    plt.ylabel(metric)
    plt.legend()
    for xi, b, m in zip(x, baseline, model):
        plt.text(xi - widths / 2, b + 0.02, f"{b:.3f}", ha="center", va="bottom", fontsize=8)
        plt.text(xi + widths / 2, m + 0.02, f"{m:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / output_name, dpi=200)
    plt.close()


def plot_improvement(summary_df: pd.DataFrame) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = ["MSE", "PSNR", "SSIM"]
    improvement = []
    for dataset in ["Clean", "Gaussian Blur", "Salt & Pepper", "Speckle"]:
        row_baseline = summary_df[(summary_df["Dataset"] == dataset) & (summary_df["Stage"] == "Baseline")].iloc[0]
        row_model = summary_df[(summary_df["Dataset"] == dataset) & (summary_df["Stage"] == "Model")].iloc[0]
        mse_imp = row_baseline["MSE"] - row_model["MSE"]
        psnr_imp = row_model["PSNR"] - row_baseline["PSNR"]
        ssim_imp = row_model["SSIM"] - row_baseline["SSIM"]
        improvement.append({"Dataset": dataset, "MSE Improvement": mse_imp, "PSNR Improvement": psnr_imp, "SSIM Improvement": ssim_imp})

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, metric in enumerate(["MSE Improvement", "PSNR Improvement", "SSIM Improvement"]):
        values = [row[metric] for row in improvement]
        labels = [row["Dataset"] for row in improvement]
        axes[i].bar(labels, values, color=["#4c72b0", "#dd8452", "#55a868", "#c44e52"])
        axes[i].set_title(metric)
        axes[i].axhline(0, color="black", linewidth=0.8)
        for bar, val in zip(axes[i].patches, values):
            axes[i].text(bar.get_x() + bar.get_width()/2, val + (0.01 if val >= 0 else -0.02), f"{val:.3f}", ha="center", va="bottom" if val >= 0 else "top")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "metric_improvement.png", dpi=200)
    plt.close()


def make_dataset_visualization() -> None:
    data_dir = PROJECT_ROOT / "data"
    fig, axes = plt.subplots(4, 5, figsize=(15, 8))
    dataset_names = ["Clean", "Gaussian Blur", "Salt & Pepper", "Speckle"]
    dataset_dirs = [data_dir / "clean", data_dir / "gaussian_blur", data_dir / "salt_pepper", data_dir / "speckle"]
    for i, dir_path in enumerate(dataset_dirs):
        file_paths = sorted(dir_path.glob("*.png"))[:5]
        for j, path in enumerate(file_paths):
            img = plt.imread(path)
            axes[i, j].imshow(img, cmap="gray")
            axes[i, j].axis("off")
        axes[i, 0].set_ylabel(dataset_names[i], rotation=90, size="large")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "dataset_comparison.png", dpi=200)
    plt.close()


def qualitative_comparison(sample_rows: list[dict]) -> None:
    fig, axes = plt.subplots(len(sample_rows), 3, figsize=(18, 5 * len(sample_rows)))
    if len(sample_rows) == 1:
        axes = np.array([axes])

    for i, row in enumerate(sample_rows):
        input_img = row["input_img"]
        output_img = row["output_img"]
        label = row["dataset"]

        axes[i, 0].imshow(input_img, cmap="gray")
        axes[i, 0].set_title(f"{label} Input")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(output_img, cmap="gray")
        axes[i, 1].set_title(f"{label} Model Output")
        axes[i, 1].axis("off")

        axes[i, 2].axis("off")
        metrics_text = (
            f"MSE: {row['mse']:.4f}\n"
            f"PSNR: {row['psnr']:.2f}\n"
            f"SSIM: {row['ssim']:.4f}"
        )
        axes[i, 2].text(0.05, 0.75, f"{label}\n\n{metrics_text}",
                        fontsize=12, va="top", ha="left",
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "qualitative_comparison.png", dpi=200)
    plt.close()


def plot_summary_charts(metrics_df: pd.DataFrame) -> None:
    baseline = metrics_df[metrics_df["Stage"] == "Baseline"].copy()
    model = metrics_df[metrics_df["Stage"] == "Model"].copy()

    for metric in ["MSE", "PSNR", "SSIM"]:
        labels = baseline["Dataset"].tolist()
        b = baseline[metric].tolist()
        m = model[metric].tolist()
        x = np.arange(len(labels))
        plt.figure(figsize=(9, 5))
        plt.bar(x - 0.2, b, width=0.35, label="Baseline")
        plt.bar(x + 0.2, m, width=0.35, label="Model")
        plt.xticks(x, labels)
        plt.ylabel(metric)
        plt.title(f"{metric} Comparison")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"{metric.lower()}_comparison.png", dpi=200)
        plt.close()
