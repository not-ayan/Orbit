"""
Model training engine with early stopping, validation tracking, and checkpointing.
"""

from typing import Dict, Optional, Tuple, Union
from pathlib import Path
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across random, numpy, and torch for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


class ModelTrainer:
    """
    Standardized trainer for ORBIT reconstruction models.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        patience: int = 5,
        device: Optional[str] = None,
        checkpoint_dir: Union[str, Path] = "results/checkpoints",
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2
        )
        self.criterion = nn.MSELoss()
        self.early_stopping = EarlyStopping(patience=patience)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.history: Dict[str, list] = {"train_loss": [], "val_loss": [], "lr": []}
        self.best_val_loss = float("inf")
        self.best_model_path = self.checkpoint_dir / "best_model.pt"

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = len(dataloader)

        for batch_x, _ in dataloader:
            batch_x = batch_x.to(self.device)
            self.optimizer.zero_grad()
            recon_x = self.model(batch_x)
            loss = self.criterion(recon_x, batch_x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / max(1, num_batches)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = len(dataloader)

        for batch_x, _ in dataloader:
            batch_x = batch_x.to(self.device)
            recon_x = self.model(batch_x)
            loss = self.criterion(recon_x, batch_x)
            total_loss += loss.item()

        return total_loss / max(1, num_batches)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 20,
    ) -> Dict[str, list]:
        """
        Full training loop with optional validation tracking and early stopping.
        """
        print(f"Training on device: {self.device} | Total Epochs: {epochs}")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["train_loss"].append(train_loss)
            self.history["lr"].append(current_lr)

            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                self.history["val_loss"].append(val_loss)
                self.scheduler.step(val_loss)
                target_eval_loss = val_loss
            else:
                val_loss = None
                self.scheduler.step(train_loss)
                target_eval_loss = train_loss

            is_best = target_eval_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = target_eval_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": target_eval_loss,
                    },
                    self.best_model_path,
                )
                marker = "*"
            else:
                marker = " "

            val_str = f"Val MSE: {val_loss:.6f}" if val_loss is not None else "No Val"
            print(
                f"Epoch {epoch:02d}/{epochs:02d} {marker} | Train MSE: {train_loss:.6f} | {val_str} | LR: {current_lr:.2e}"
            )

            if val_loader is not None and self.early_stopping(val_loss):
                print(f"Early stopping triggered at epoch {epoch}")
                break

        # Save training history
        with open(self.checkpoint_dir / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        # Load best model weights
        if self.best_model_path.exists():
            checkpoint = torch.load(self.best_model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded best model from epoch {checkpoint['epoch']} with Loss: {checkpoint['val_loss']:.6f}")

        return self.history


__all__ = [
    "set_seed",
    "EarlyStopping",
    "ModelTrainer",
]

