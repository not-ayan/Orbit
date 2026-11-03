"""
Self-supervised masked pretraining trainer for ORBIT.
Optimizes masked reconstruction loss and saves domain-transferable backbone checkpoints.
"""

from pathlib import Path
from typing import Dict, Optional, Union
import json
import torch
from torch.utils.data import DataLoader

from orbit.models.orbit import ORBITPretrainer
from orbit.training.trainer import EarlyStopping


class MaskedPretrainer:
    """
    Dedicated trainer for self-supervised masked reconstruction pretraining.
    """

    def __init__(
        self,
        model: ORBITPretrainer,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 5,
        device: Optional[str] = None,
        checkpoint_dir: Union[str, Path] = "results/pretraining",
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2
        )
        self.early_stopping = EarlyStopping(patience=patience)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.history: Dict[str, list] = {"masked_train_loss": [], "masked_val_loss": [], "lr": []}
        self.best_val_loss = float("inf")
        self.best_model_path = self.checkpoint_dir / "best_pretrain_model.pt"
        self.backbone_path = self.checkpoint_dir / "pretrained_backbone.pt"

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = len(dataloader)

        for batch_x, _ in dataloader:
            batch_x = batch_x.to(self.device)
            self.optimizer.zero_grad()
            loss, _, _ = self.model.forward_pretrain(batch_x)
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
            loss, _, _ = self.model.forward_pretrain(batch_x)
            total_loss += loss.item()

        return total_loss / max(1, num_batches)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 15,
    ) -> Dict[str, list]:
        """
        Executes pretraining loop.
        """
        print(f"Pretraining on device: {self.device} | Total Epochs: {epochs}")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader)
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.scheduler.step(val_loss)
            self.history["masked_train_loss"].append(train_loss)
            self.history["masked_val_loss"].append(val_loss)
            self.history["lr"].append(current_lr)

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                # Save full pretraining model
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "masked_val_loss": val_loss,
                        "mask_ratio": self.model.mask_ratio,
                    },
                    self.best_model_path,
                )
                # Save portable backbone weights for Step 5 cross-domain transfer
                torch.save(self.model.get_backbone_state(), self.backbone_path)
                marker = "*"
            else:
                marker = " "

            print(
                f"Epoch {epoch:02d}/{epochs:02d} {marker} | Masked Train MSE: {train_loss:.6f} | Masked Val MSE: {val_loss:.6f} | LR: {current_lr:.2e}"
            )

            if self.early_stopping(val_loss):
                print(f"Early stopping triggered at epoch {epoch}")
                break

        # Save pretraining history
        with open(self.checkpoint_dir / "pretrain_history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        print(f"Pretraining complete. Pretrained backbone saved to {self.backbone_path}")
        return self.history


__all__ = [
    "MaskedPretrainer",
]

