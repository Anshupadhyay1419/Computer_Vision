from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import BATCH_SIZE, EPOCHS, LEARNING_RATE, SEED, PROJECT_ROOT
from .dataset import set_seed
from .model import Autoencoder


class MnistImageDataset(Dataset):
    def __init__(self, image_paths: list[str], transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = np.asarray(__import__("PIL").Image.open(path), dtype=np.float32) / 255.0
        image = image.reshape(1, 28, 28)
        image = torch.from_numpy(image).float()
        return image, path


def train_autoencoder(train_paths: list[str], device: torch.device) -> tuple[Autoencoder, list[float]]:
    set_seed(SEED)
    model = Autoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader = DataLoader(MnistImageDataset(train_paths), batch_size=BATCH_SIZE, shuffle=True)
    history = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for batch_images, _ in train_loader:
            batch_images = batch_images.to(device)
            optimizer.zero_grad()
            outputs = model(batch_images)
            loss = criterion(outputs, batch_images)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_images.size(0)
        epoch_loss /= len(train_loader.dataset)
        history.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{EPOCHS} - loss: {epoch_loss:.6f}")

    torch.save(model.state_dict(), PROJECT_ROOT / "outputs" / "model" / "best_model.pth")
    return model, history


def load_model(model_path: str | Path, device: torch.device) -> Autoencoder:
    model = Autoencoder().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model
