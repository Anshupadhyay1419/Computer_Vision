from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import (
    CLEAN_DIR,
    DATA_DIR,
    GAUSSIAN_DIR,
    GAUSSIAN_KERNEL_SIZE,
    GAUSSIAN_SIGMA,
    MODEL_DIR,
    NUM_IMAGES,
    OUTPUTS_DIR,
    PLOTS_DIR,
    PROJECT_ROOT,
    RESULTS_DIR,
    SALT_PEPPER_AMOUNT,
    SALT_PEPPER_DIR,
    SEED,
    SPECKLE_DIR,
    SPECKLE_VARIANCE,
    TRAIN_RATIO,
    DEVICE,
)
from src.dataset import (
    build_metadata,
    create_corrupted_datasets,
    create_split,
    ensure_dataset_counts,
    save_clean_dataset,
    select_mnist_subset,
    set_seed,
)
from src.evaluate import compute_summary_rows, evaluate_dataset, save_result_tables
from src.model import Autoencoder, build_model
from src.train import load_model, train_autoencoder
from src.visualize import make_dataset_visualization, plot_improvement, plot_summary_charts, plot_training_loss, qualitative_comparison


def ensure_directories() -> None:
    for path in [DATA_DIR, CLEAN_DIR, GAUSSIAN_DIR, SALT_PEPPER_DIR, SPECKLE_DIR, MODEL_DIR, PLOTS_DIR, RESULTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    for folder in [CLEAN_DIR, GAUSSIAN_DIR, SALT_PEPPER_DIR, SPECKLE_DIR]:
        for file in folder.glob("*.png"):
            file.unlink()


def build_main_pipeline() -> None:
    ensure_directories()
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    mnist_subset, selected_items = select_mnist_subset()
    save_clean_dataset(selected_items)
    create_corrupted_datasets(selected_items)
    ensure_dataset_counts()
    metadata = build_metadata(selected_items)
    metadata.to_csv(PROJECT_ROOT / "metadata.csv", index=False)

    train_indices, test_indices = create_split(metadata)
    train_paths = [str(CLEAN_DIR / f"{idx:03d}.png") for idx in train_indices]
    test_paths = [str(CLEAN_DIR / f"{idx:03d}.png") for idx in test_indices]

    gaussian_test = [str(GAUSSIAN_DIR / f"{idx:03d}.png") for idx in test_indices]
    salt_pepper_test = [str(SALT_PEPPER_DIR / f"{idx:03d}.png") for idx in test_indices]
    speckle_test = [str(SPECKLE_DIR / f"{idx:03d}.png") for idx in test_indices]

    model, history = train_autoencoder(train_paths, device)
    plot_training_loss(history)
    model_path = MODEL_DIR / "best_model.pth"
    torch.save(model.state_dict(), model_path)

    results = []
    per_image_rows = []
    dataset_specs = {
        "Clean": (test_paths, test_paths),
        "Gaussian Blur": (gaussian_test, test_paths),
        "Salt & Pepper": (salt_pepper_test, test_paths),
        "Speckle": (speckle_test, test_paths),
    }

    for dataset_name, (corrupt_paths, clean_paths) in dataset_specs.items():
        baseline_mse, baseline_psnr, baseline_ssim, model_mse, model_psnr, model_ssim = evaluate_dataset(model, corrupt_paths, clean_paths, device)
        result_row = {
            "Dataset": dataset_name,
            "Stage": "Baseline",
            "MSE": float(np.mean(baseline_mse)),
            "PSNR": float(np.mean(baseline_psnr)),
            "SSIM": float(np.mean(baseline_ssim)),
        }
        results.append(result_row)
        results.append({
            "Dataset": dataset_name,
            "Stage": "Model",
            "MSE": float(np.mean(model_mse)),
            "PSNR": float(np.mean(model_psnr)),
            "SSIM": float(np.mean(model_ssim)),
        })

        for image_index, (base_mse, base_psnr, base_ssim, m_mse, m_psnr, m_ssim) in enumerate(zip(baseline_mse, baseline_psnr, baseline_ssim, model_mse, model_psnr, model_ssim)):
            label = int(metadata.loc[metadata['index'] == test_indices[image_index], 'mnist_label'].iloc[0])
            per_image_rows.append({
                "image_index": test_indices[image_index],
                "label": label,
                "dataset": dataset_name,
                "baseline_mse": float(base_mse),
                "baseline_psnr": float(base_psnr),
                "baseline_ssim": float(base_ssim),
                "model_mse": float(m_mse),
                "model_psnr": float(m_psnr),
                "model_ssim": float(m_ssim),
            })

    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    per_image_df = pd.DataFrame(per_image_rows)
    per_image_df.to_csv(RESULTS_DIR / "per_image_metrics.csv", index=False)

    plot_summary_charts(metrics_df)
    plot_improvement(metrics_df)
    make_dataset_visualization()

    sample_index = test_indices[0]
    clean_img = np.asarray(__import__("PIL").Image.open(str(CLEAN_DIR / f"{sample_index:03d}.png")), dtype=np.float32) / 255.0
    sample_rows = []
    for dataset_name, input_path in [
        ("Gaussian Blur", str(GAUSSIAN_DIR / f"{sample_index:03d}.png")),
        ("Salt & Pepper", str(SALT_PEPPER_DIR / f"{sample_index:03d}.png")),
        ("Speckle", str(SPECKLE_DIR / f"{sample_index:03d}.png")),
    ]:
        input_img = np.asarray(__import__("PIL").Image.open(input_path), dtype=np.float32) / 255.0
        x = torch.from_numpy(input_img.reshape(1, 1, 28, 28)).float().to(device)
        with torch.no_grad():
            output_img = model(x).cpu().numpy()[0, 0]
        mse_val = float(np.mean((output_img - clean_img) ** 2))
        psnr_val = float(10 * np.log10((1.0 ** 2) / max(mse_val, 1e-12)))
        ssim_val = float(__import__("skimage.metrics").metrics.structural_similarity(output_img, clean_img, data_range=1.0))
        sample_rows.append({
            "dataset": dataset_name,
            "input_img": input_img,
            "output_img": output_img,
            "mse": mse_val,
            "psnr": psnr_val,
            "ssim": ssim_val,
        })

    qualitative_comparison(sample_rows)

    print("====================================================")
    print("MNIST NOISE EXPERIMENT RESULTS")
    print("====================================================")
    print(f"{'Dataset':<16} {'Stage':<12} {'MSE':>10} {'PSNR':>10} {'SSIM':>10}")
    print("--------------------------------------------------------")
    for row in results:
        print(f"{row['Dataset']:<16} {row['Stage']:<12} {row['MSE']:>10.4f} {row['PSNR']:>10.4f} {row['SSIM']:>10.4f}")
    print("--------------------------------------------------------")
    print("Results saved to:")
    print(f"{RESULTS_DIR}")
    print("Plots saved to:")
    print(f"{PLOTS_DIR}")
    print("Model saved to:")
    print(f"{model_path}")
    print("====================================================")


if __name__ == "__main__":
    build_main_pipeline()
