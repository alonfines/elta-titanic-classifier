"""PyTorch model definition shared between `train.py` and `ds_app.py`.

Kept deliberately small: one hidden layer, width 16. At 16 engineered features and 712
training rows (80% of 891), a bigger network buys memorization risk rather than accuracy —
Titanic survival is a shallow problem already largely explained by Sex/Pclass/Age/Title/family
size, all of which are hand-engineered inputs to this model already (see `preprocessing.py` and
`eda.ipynb`). See the README design-choices section for the full reasoning.
"""
from __future__ import annotations

import torch
from torch import nn

HIDDEN_DIM = 16
DROPOUT = 0.3


class TitanicNet(nn.Module):
    """Single-hidden-layer MLP for Titanic survival classification.

    Outputs raw logits (no final sigmoid) — pair with `nn.BCEWithLogitsLoss` for training
    (more numerically stable than a manual sigmoid + `BCELoss`) and `torch.sigmoid` to recover
    probabilities at inference time.
    """

    def __init__(self, n_features: int, hidden_dim: int = HIDDEN_DIM, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
