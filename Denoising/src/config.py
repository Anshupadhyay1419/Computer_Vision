import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CLEAN_DIR = DATA_DIR / "clean"
GAUSSIAN_DIR = DATA_DIR / "gaussian_blur"
SALT_PEPPER_DIR = DATA_DIR / "salt_pepper"
SPECKLE_DIR = DATA_DIR / "speckle"
RAW_DIR = DATA_DIR / "raw"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUTS_DIR / "model"
PLOTS_DIR = OUTPUTS_DIR / "plots"
RESULTS_DIR = OUTPUTS_DIR / "results"

SEED = 42
NUM_IMAGES = 500
TRAIN_RATIO = 0.8
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.001
GAUSSIAN_KERNEL_SIZE = 5
GAUSSIAN_SIGMA = 1.0
SALT_PEPPER_AMOUNT = 0.10
SPECKLE_VARIANCE = 0.10
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "") else "cpu"
