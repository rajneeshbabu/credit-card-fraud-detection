"""
model_arch.py
─────────────
Shared PyTorch model definition used by both train.py and app.py.

Device priority:  Apple Silicon MPS  >  NVIDIA CUDA  >  CPU
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────
#  Device selection
# ─────────────────────────────────────────────
def get_device() -> tuple[torch.device, str]:
    """
    Returns (device, description_string).
    Prioritises Apple Silicon MPS → NVIDIA CUDA → CPU.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps"), "Apple Silicon GPU  (MPS) ⚡"
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return torch.device("cuda"), f"NVIDIA GPU — {name} ⚡"
    return torch.device("cpu"), "CPU"


# ─────────────────────────────────────────────
#  Autoencoder
# ─────────────────────────────────────────────
class FraudAutoencoder(nn.Module):
    """
    Encoder–Decoder trained on LEGITIMATE transactions only.
    Fraud detection via reconstruction error:
        high MSE  →  transaction is out-of-distribution  →  likely fraud.

    Architecture
    ────────────
    Encoder:  input_dim → 64 → 32 → 16  (bottleneck)
    Decoder:  16 → 32 → 64 → input_dim
    """

    def __init__(self, input_dim: int):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),

            nn.Linear(32, 16),
            nn.ReLU(),                       # bottleneck
        )

        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.1),

            nn.Linear(32, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),

            nn.Linear(64, input_dim),        # linear output — MSE loss
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE reconstruction error (no grad)."""
        with torch.no_grad():
            recon = self.forward(x)
            return torch.mean((x - recon) ** 2, dim=1)
