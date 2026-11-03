"""
Parameter-Efficient Adapters and Target Architecture for ORBIT.

Supports:
1. TemporalAdapterLayer: Residual bottleneck adapter (d_model -> d_adapter -> d_model).
2. AdaptedTransformerModel: Integrates pretrained backbone with target-specific projections and adapters.
3. Adaptation Modes:
   - 'zero_shot': 0% target data. Requires identical sensor schema or explicit sensor correspondence mapping.
   - 'target_calibration': Unsupervised fitting of projection layers only on target normal calibration data (backbone frozen).
   - 'adapter': Freeze backbone; train projection layers and residual bottleneck adapters.
   - 'full_finetune': Train all model parameters end-to-end.
"""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import math
import torch
import torch.nn as nn

from orbit.models.temporal_encoder import (
    PositionalEncoding,
    DomainInputProjection,
    DomainReconstructionHead,
)
from orbit.models.checkpoint import (
    CheckpointValidationError,
    save_orbit_checkpoint,
    load_orbit_checkpoint,
    validate_inference_completeness,
)


class AdaptationMode(str, Enum):
    ZERO_SHOT = "zero_shot"                      # 0% target data, requires shared schema or explicit sensor map
    TARGET_CALIBRATION = "target_calibration"    # Fit target projection & recon head on target calibration data; backbone frozen
    LINEAR_PROBE = "linear_probe"                # Alias for TARGET_CALIBRATION
    ADAPTER = "adapter"                          # Freeze backbone; train target projection, bottleneck adapters, and head
    FULL_FINETUNE = "full_finetune"              # Train all parameters end-to-end


class TemporalAdapterLayer(nn.Module):
    """
    Parameter-Efficient Bottleneck Residual Adapter.
    Projects d_model down to a small bottleneck d_adapter, applies non-linearity,
    and projects back up with a residual skip connection and scaling factor.
    """

    def __init__(
        self,
        d_model: int = 64,
        d_adapter: int = 16,
        dropout: float = 0.1,
        scale: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_adapter = d_adapter
        self.scale = scale

        self.down_proj = nn.Linear(d_model, d_adapter)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up_proj = nn.Linear(d_adapter, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

        # Initialize up_proj to near-zero so adapter starts close to exact identity
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, L, d_model)
        Returns:
            Residual adapted tensor of shape (B, L, d_model)
        """
        h = self.layer_norm(x)
        h = self.down_proj(h)
        h = self.act(h)
        h = self.dropout(h)
        adapted = self.up_proj(h) * self.scale
        return x + adapted


class AdaptedTransformerModel(nn.Module):
    """
    ORBIT Target-Domain Adapted Model.
    Wraps a pretrained temporal Transformer backbone and integrates
    domain-specific projections and parameter-efficient adapters.
    """

    def __init__(
        self,
        target_dim: int,
        pretrained_backbone_path: Optional[Union[str, Path]] = None,
        source_dim: Optional[int] = None,
        sensor_mapping: Optional[Union[Dict[int, int], torch.Tensor]] = None,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        d_adapter: int = 16,
        dropout: float = 0.1,
        mode: Union[str, AdaptationMode] = AdaptationMode.ADAPTER,
        max_len: int = 512,
    ):
        super().__init__()
        self.target_dim = target_dim
        self.source_dim = source_dim
        self.d_model = d_model
        self.num_layers = num_layers
        self.d_adapter = d_adapter
        self.mode = AdaptationMode(mode) if isinstance(mode, str) else mode
        self.sensor_mapping = sensor_mapping
        self.is_calibrated = False

        # 1. Target-domain input projection (F_target -> d_model)
        self.target_projection = DomainInputProjection(
            input_dim=target_dim, d_model=d_model, dropout=dropout
        )

        # 2. Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model=d_model, max_len=max_len, dropout=dropout
        )

        # 3. Backbone Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )

        # 4. Bottleneck adapter layers (inserted per encoder layer)
        self.adapters = nn.ModuleList([
            TemporalAdapterLayer(d_model=d_model, d_adapter=d_adapter, dropout=dropout)
            for _ in range(num_layers)
        ])

        # 5. Target-domain reconstruction head (d_model -> F_target)
        self.target_recon_head = DomainReconstructionHead(
            d_model=d_model, output_dim=target_dim, dropout=dropout
        )

        # 6. Source-domain components (optional, used if full checkpoint is loaded for zero-shot)
        self.source_projection: Optional[DomainInputProjection] = None
        self.source_recon_head: Optional[DomainReconstructionHead] = None

        # Load pretrained weights if provided
        if pretrained_backbone_path is not None:
            self.load_pretrained_checkpoint(pretrained_backbone_path)

        # Configure trainable parameters according to mode
        self.configure_trainable_parameters()

    def load_pretrained_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """
        Loads pretrained weights. Validates that required components exist
        and checks compatibility for zero-shot evaluation.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Backbone checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # 1. Handle complete ORBIT checkpoint payload
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict):
            state = checkpoint
        else:
            raise CheckpointValidationError(f"Invalid checkpoint format in {checkpoint_path}")

        # Load backbone
        backbone_loaded = False
        if "transformer_encoder" in state:
            self.transformer_encoder.load_state_dict(state["transformer_encoder"])
            backbone_loaded = True
        elif any(k.startswith("transformer_encoder.") for k in state.keys()):
            encoder_dict = {
                k.replace("transformer_encoder.", ""): v
                for k, v in state.items()
                if k.startswith("transformer_encoder.")
            }
            self.transformer_encoder.load_state_dict(encoder_dict)
            backbone_loaded = True

        if not backbone_loaded:
            raise CheckpointValidationError(
                f"Missing required 'transformer_encoder' weights in {checkpoint_path}"
            )

        # Load positional encoding
        if "pos_encoder" in state:
            self.pos_encoder.load_state_dict(state["pos_encoder"])
        elif any(k.startswith("pos_encoder.") for k in state.keys()):
            pe_dict = {
                k.replace("pos_encoder.", ""): v
                for k, v in state.items()
                if k.startswith("pos_encoder.")
            }
            self.pos_encoder.load_state_dict(pe_dict)

        # Load source projection and head if present using explicit None checks
        source_in_w = state.get("input_projection.proj.0.weight")
        if source_in_w is None:
            source_in_w = state.get("input_projection.weight")

        source_out_w = state.get("recon_head.head.weight")
        if source_out_w is None:
            source_out_w = state.get("recon_head.weight")

        if source_in_w is not None and source_out_w is not None:
            self.source_dim = source_in_w.shape[1]
            # Initialize source projection & recon head
            self.source_projection = DomainInputProjection(
                input_dim=self.source_dim, d_model=self.d_model, dropout=0.0
            )
            self.source_recon_head = DomainReconstructionHead(
                d_model=self.d_model, output_dim=self.source_dim, dropout=0.0
            )
            in_proj_dict = {
                k.replace("input_projection.", ""): v
                for k, v in state.items()
                if k.startswith("input_projection.")
            }
            out_head_dict = {
                k.replace("recon_head.", ""): v
                for k, v in state.items()
                if k.startswith("recon_head.")
            }
            self.source_projection.load_state_dict(in_proj_dict, strict=False)
            self.source_recon_head.load_state_dict(out_head_dict, strict=False)

            # If target_dim matches source_dim, directly load into target projection & head
            if self.target_dim == self.source_dim:
                self.target_projection.load_state_dict(in_proj_dict, strict=False)
                self.target_recon_head.load_state_dict(out_head_dict, strict=False)
                self.is_calibrated = True

        print(f"Successfully loaded pretrained backbone from {checkpoint_path}")

    def configure_trainable_parameters(self) -> None:
        """
        Enforces parameter freezing based on the selected adaptation mode.
        """
        for param in self.parameters():
            param.requires_grad = True

        if self.mode == AdaptationMode.ZERO_SHOT:
            # Completely freeze all parameters
            for param in self.parameters():
                param.requires_grad = False

        elif self.mode in (AdaptationMode.TARGET_CALIBRATION, AdaptationMode.LINEAR_PROBE):
            # Freeze backbone and adapters; train ONLY target projection and reconstruction head
            for param in self.transformer_encoder.parameters():
                param.requires_grad = False
            for param in self.pos_encoder.parameters():
                param.requires_grad = False
            for param in self.adapters.parameters():
                param.requires_grad = False
            for param in self.target_projection.parameters():
                param.requires_grad = True
            for param in self.target_recon_head.parameters():
                param.requires_grad = True

        elif self.mode == AdaptationMode.ADAPTER:
            # Freeze backbone; train target projection, adapters, and head
            for param in self.transformer_encoder.parameters():
                param.requires_grad = False
            for param in self.pos_encoder.parameters():
                param.requires_grad = False
            for param in self.adapters.parameters():
                param.requires_grad = True
            for param in self.target_projection.parameters():
                param.requires_grad = True
            for param in self.target_recon_head.parameters():
                param.requires_grad = True

        elif self.mode == AdaptationMode.FULL_FINETUNE:
            # All parameters trainable
            pass

    def mark_calibrated(self) -> None:
        """Explicitly flags the target projection and head as calibrated on target data."""
        self.is_calibrated = True

    def validate_for_inference(self) -> None:
        """
        Validates that all inference components are trained/calibrated and deterministic.
        Rejects uncalibrated random projections in ZERO_SHOT mode.
        """
        validate_inference_completeness(self, model_name=f"AdaptedTransformerModel({self.mode.value})")

        if self.mode == AdaptationMode.ZERO_SHOT:
            if self.sensor_mapping is not None:
                if self.source_projection is None or self.source_recon_head is None:
                    raise CheckpointValidationError(
                        "Sensor-mapped zero-shot inference requires trained source_projection and "
                        "source_recon_head in checkpoint, but they were not found."
                    )
            elif not self.is_calibrated:
                raise CheckpointValidationError(
                    f"Cannot perform zero-shot inference: target domain has {self.target_dim} features "
                    f"while source domain has {self.source_dim} features, and no explicit sensor mapping or "
                    f"calibrated projection is provided. Uninitialized random projections are mathematically "
                    f"invalid for zero-shot evaluation. Use AdaptationMode.TARGET_CALIBRATION with a budgeted "
                    f"calibration split, or specify sensor_mapping."
                )

    def get_parameter_summary(self) -> Dict[str, Any]:
        """Returns parameter counts (total, trainable, frozen)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": frozen,
            "trainable_percent": round(trainable / max(1, total) * 100, 2),
            "adaptation_mode": self.mode.value,
            "is_calibrated": self.is_calibrated,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Target telemetry of shape (B, L, F_target)
        Returns:
            x_hat: Reconstructed telemetry of shape (B, L, F_target)
        """
        # Always enforce inference safety in zero-shot mode
        if self.mode == AdaptationMode.ZERO_SHOT:
            self.validate_for_inference()

        # Handle explicit sensor mapping zero-shot path
        if self.mode == AdaptationMode.ZERO_SHOT and self.sensor_mapping is not None:
            B, L, _ = x.shape
            src_dim = self.source_dim or max(self.sensor_mapping.values()) + 1 if isinstance(self.sensor_mapping, dict) else self.source_dim
            x_source = torch.zeros(B, L, src_dim, device=x.device, dtype=x.dtype)

            if isinstance(self.sensor_mapping, dict):
                for tgt_idx, src_idx in self.sensor_mapping.items():
                    x_source[:, :, int(src_idx)] = x[:, :, int(tgt_idx)]
            elif isinstance(self.sensor_mapping, torch.Tensor):
                if self.sensor_mapping.ndim == 1:
                    x_source[:, :, self.sensor_mapping] = x
                elif self.sensor_mapping.ndim == 2:
                    x_source = torch.matmul(x, self.sensor_mapping.to(x.device))

            # Forward through source-domain pipeline
            h = self.source_projection(x_source)
            h = self.pos_encoder(h)
            h = self.transformer_encoder(h)
            x_hat_source = self.source_recon_head(h)

            # Map reconstruction back to target features
            x_hat = torch.zeros_like(x)
            if isinstance(self.sensor_mapping, dict):
                for tgt_idx, src_idx in self.sensor_mapping.items():
                    x_hat[:, :, int(tgt_idx)] = x_hat_source[:, :, int(src_idx)]
            elif isinstance(self.sensor_mapping, torch.Tensor):
                if self.sensor_mapping.ndim == 1:
                    x_hat = x_hat_source[:, :, self.sensor_mapping]
                elif self.sensor_mapping.ndim == 2:
                    x_hat = torch.matmul(x_hat_source, self.sensor_mapping.t().to(x.device))

            return x_hat

        # 1. Standard path: Project target sensors F_target -> d_model
        h = self.target_projection(x)
        # 2. Inject positional information
        h = self.pos_encoder(h)

        # 3. Forward through Transformer layers interleaved with adapters
        if self.mode == AdaptationMode.ADAPTER:
            for layer_idx, enc_layer in enumerate(self.transformer_encoder.layers):
                h = enc_layer(h)
                h = self.adapters[layer_idx](h)
            if self.transformer_encoder.norm is not None:
                h = self.transformer_encoder.norm(h)
        else:
            h = self.transformer_encoder(h)

        # 4. Reconstruct back to target sensors
        x_hat = self.target_recon_head(h)
        return x_hat

    def compute_reconstruction_error(
        self, x: torch.Tensor, reduction: str = "none"
    ) -> torch.Tensor:
        self.eval()
        self.validate_for_inference()
        with torch.no_grad():
            x_hat = self.forward(x)
            sq_err = (x - x_hat) ** 2
            if reduction == "feature_mean":
                return sq_err.mean(dim=-1)
            elif reduction == "window_mean":
                return sq_err.mean(dim=[-2, -1])
            return sq_err

    def save_checkpoint(
        self,
        filepath: Union[str, Path],
        schema_info: Optional[Dict[str, Any]] = None,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Saves complete versioned model checkpoint."""
        architecture = {
            "target_dim": self.target_dim,
            "source_dim": self.source_dim,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "d_adapter": self.d_adapter,
            "mode": self.mode.value,
        }
        schema = schema_info or {"feature_dim": self.target_dim}
        return save_orbit_checkpoint(
            model=self,
            filepath=filepath,
            architecture=architecture,
            schema_info=schema,
            adaptation_mode=self.mode.value,
            provenance=provenance,
            extra_state={"is_calibrated": self.is_calibrated},
        )


__all__ = [
    "AdaptationMode",
    "TemporalAdapterLayer",
    "AdaptedTransformerModel",
]
