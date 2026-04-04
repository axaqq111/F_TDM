"""
model.py
--------
F-TDM neural network definition.

Paper: "A Coverage-Aware High-Quality Sensing Data Collection Method
        in Mobile Crowd Sensing"
        IEEE Transactions on Mobile Computing, Vol. 24, No. 4, April 2025

Architecture (Table II):
  Input  : 4  (D_W — one group of human participant readings)
  Hidden : 1024 neurons, ReLU activation
  Output : 1  (predicted true value)
"""

import torch
import torch.nn as nn


class TruthDiscoveryMLP(nn.Module):
    """
    MLP regression network used inside F-TDM.

    The network maps a vector of 4 human participant readings (D_W for a
    single group) to a single predicted true value.

    Parameters
    ----------
    input_size  : int  – number of human participants per group (default 4)
    hidden_size : int  – number of hidden units (default 1024, per Table II)
    output_size : int  – prediction dimension (default 1)
    """

    def __init__(self,
                 input_size:  int = 4,
                 hidden_size: int = 1024,
                 output_size: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size,  hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, input_size)
            Human participant readings D_W.

        Returns
        -------
        torch.Tensor, shape (batch, output_size)
            Predicted true values D_P.
        """
        return self.net(x)


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = TruthDiscoveryMLP()
    print(model)

    # Dummy forward pass: batch of 8 groups, each with 4 human readings
    x = torch.randn(8, 4)
    y = model(x)
    print(f"\nInput shape : {x.shape}")
    print(f"Output shape: {y.shape}")  # expected (8, 1)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
