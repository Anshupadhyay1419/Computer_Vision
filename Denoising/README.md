# MNIST Noise Experiment

This project studies how common image degradations affect a simple deep-learning denoising/reconstruction model trained on clean MNIST images.

## Objective
The main experiment is:

- Train on clean MNIST digits
- Test on clean digits, Gaussian blur, salt-and-pepper noise, and speckle noise
- Compare the raw corrupted image against the model reconstruction using MSE, PSNR, and SSIM

The clean images are used as the target output for the autoencoder.

## Dataset
The project uses the standard MNIST dataset from torchvision. To keep the experiment manageable and reproducible, exactly 500 images are selected using a fixed random seed.

Why 500 images?
- It reduces computational cost while keeping a meaningful dataset for model training and evaluation.
- It makes the experiment easy to reproduce and inspect.
- It matches the requirement for a small controlled study of degradation effects.

## Four dataset types
The project generates four datasets from the same original 500 images:

1. Clean: original MNIST digits in normalized [0,1] form
2. Gaussian Blur: applies Gaussian smoothing with configurable kernel size and sigma
3. Salt & Pepper: randomly flips a configurable proportion of pixels to 0 or 1
4. Speckle: multiplicative Gaussian noise applied to the image and clipped back to [0,1]

## Train/test split
The 500 images are split reproducibly using a fixed random seed:

- 80% training: 400 images
- 20% testing: 100 images

The split is shared across all four datasets so the same original samples are compared consistently.

## Model architecture
A simple autoencoder is used:

- Encoder: Conv2d -> ReLU -> MaxPool -> Conv2d -> ReLU -> MaxPool
- Decoder: ConvTranspose2d -> ReLU -> ConvTranspose2d -> ReLU -> Conv2d -> Sigmoid

The model reconstructs a 1 x 28 x 28 grayscale image and is trained to output the clean image.

## Training procedure
- Input: clean image
- Target: clean image
- Loss: MSELoss
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 32
- Epochs: configurable (default 25)

The model is trained only on the clean training images. The noisy versions are only used for testing robustness.

## Baseline evaluation
Before applying the model, the corrupted images are compared directly to the clean ground truth using:

- MSE
- PSNR
- SSIM

This is the baseline degradation quality.

## Model evaluation
After training, the same model is applied to the test clean, Gaussian blur, salt-and-pepper, and speckle images. The output is compared against the original clean ground truth.

## Metrics
### MSE
Mean squared error between model output and target image.
- Lower is better.

### PSNR
Peak signal-to-noise ratio.
- Higher is better.

### SSIM
Structural similarity index.
- Higher is better.

## How to run
From the project root:

```bash
python main.py
```

This will:
1. Create the dataset directories
2. Load and select 500 MNIST images
3. Generate corrupted datasets
4. Save metadata and train/test split files
5. Train the autoencoder
6. Evaluate all datasets
7. Save metrics and plots
8. Print the final summary

## Expected output files
- outputs/model/best_model.pth
- outputs/results/metrics.csv
- outputs/results/per_image_metrics.csv
- outputs/plots/training_loss.png
- outputs/plots/mse_comparison.png
- outputs/plots/psnr_comparison.png
- outputs/plots/ssim_comparison.png
- outputs/plots/metric_improvement.png
- outputs/plots/dataset_comparison.png
- outputs/plots/qualitative_comparison.png

## Interpreting the graphs
- MSE comparison shows how much error decreases after reconstruction
- PSNR comparison shows signal quality improvement
- SSIM comparison shows structural similarity improvement
- The training loss plot indicates convergence
- The qualitative figure helps visually inspect reconstruction quality

This experiment shows whether a model trained only on clean data can recover degraded images reasonably well.
