"""
Neural network models for F-TDM.

TrustAttentionTDM  – attention-based trusted worker mechanism (primary model)
TruthDiscoveryMLP  – original baseline MLP (for comparison)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionModule(nn.Module):
    """
    Learns trust weights over 4 worker observations.

    Input : [d1, d2, d3, d4]   (batch, 4)
    Output: attention weights  (batch, 4)  — sum to 1 via softmax
    """

    def __init__(self, n_workers: int = 4, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(n_workers, hidden)
        self.fc2 = nn.Linear(hidden, n_workers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        logits = self.fc2(h)
        return F.softmax(logits, dim=-1)


class TrustAttentionTDM(nn.Module):
    """
    Attention-based Truth Discovery Model.

    Architecture
    ────────────
    Input [d1,d2,d3,d4]
         ↓
    AttentionModule → weights [a1,a2,a3,a4]
         ↓
    Weighted sum d_agg = Σ aᵢ·dᵢ   (scalar per sample)
         ↓
    MLP: 1 → 1024 → 1   (predicted true value)
    """

    def __init__(self, n_workers: int = 4, attn_hidden: int = 64, mlp_hidden: int = 1024):
        super().__init__()
        self.attention = AttentionModule(n_workers, attn_hidden)
        self.mlp = nn.Sequential(
            nn.Linear(1, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (batch, 4) — worker observations

        Returns
        -------
        pred    : (batch, 1) — predicted true value
        weights : (batch, 4) — attention weights (trust proxy)
        """
        weights = self.attention(x)              # (batch, 4)
        d_agg = (weights * x).sum(dim=-1, keepdim=True)  # (batch, 1)
        pred = self.mlp(d_agg)                   # (batch, 1)
        return pred, weights


class TruthDiscoveryMLP(nn.Module):
    """
    Original baseline MLP (F-TDM without trust mechanism).

    Input : [d1, d2, d3, d4]   (batch, 4)
    Output: predicted true value (batch, 1)
    """

    def __init__(self, input_dim: int = 4, hidden: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)
