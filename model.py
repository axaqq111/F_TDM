"""
model.py
--------
F-TDM neural network definitions.

Paper: "A Coverage-Aware High-Quality Sensing Data Collection Method
        in Mobile Crowd Sensing"
        IEEE Transactions on Mobile Computing, Vol. 24, No. 4, April 2025

Architecture (Table II):
  Input  : 4  (D_W — one group of human participant readings)
  Hidden : 1024 neurons, ReLU activation
  Output : 1  (predicted true value)

An extended class TrustAwareTruthDiscoveryMLP adds dynamic worker trust
weights to improve prediction quality.  Two fusion modes are supported:

  concat   (default / recommended)
    Concatenate D_W (4-dim) and trust_weights (4-dim) → 8-dim input.
    Gives the network full flexibility to learn how to use trust info.

  weighted
    Element-wise multiply D_W * trust_weights before feeding the original
    4-dim input.  Simpler but less expressive.

Both modes share the same hidden_size=1024 and output_size=1.
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


class TrustAwareTruthDiscoveryMLP(nn.Module):
    """
    Trust-aware extension of TruthDiscoveryMLP.

    Incorporates per-worker dynamic trust weights into the forward pass.
    Two fusion modes are available (see module docstring).

    Parameters
    ----------
    input_size  : int  – number of workers per group (default 4)
    hidden_size : int  – hidden units (default 1024, per Table II)
    output_size : int  – prediction dimension (default 1)
    fusion_mode : str  – 'concat' (default) or 'weighted'
    """

    def __init__(self,
                 input_size:  int = 4,
                 hidden_size: int = 1024,
                 output_size: int = 1,
                 fusion_mode: str = "concat"):
        super().__init__()
        if fusion_mode not in ("concat", "weighted"):
            raise ValueError(
                f"fusion_mode must be 'concat' or 'weighted', got '{fusion_mode}'"
            )
        self.fusion_mode = fusion_mode

        # concat doubles the effective input size
        actual_input = input_size * 2 if fusion_mode == "concat" else input_size

        self.net = nn.Sequential(
            nn.Linear(actual_input, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,  output_size),
        )

    def forward(self,
                x: torch.Tensor,
                trust_weights: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass with optional trust weights.

        Parameters
        ----------
        x             : shape (batch, input_size)  — worker readings D_W
        trust_weights : shape (batch, input_size)  — normalised trust weights
                        (requires_grad should be False; managed by
                        DynamicTrustManager).  If None, falls back to
                        standard behaviour (no trust weighting).

        Returns
        -------
        torch.Tensor, shape (batch, output_size) — predicted true values D_P
        """
        if self.fusion_mode == "weighted":
            if trust_weights is not None:
                x = x * trust_weights
            # else: no weighting — pass raw D_W unchanged
        else:  # concat
            if trust_weights is None:
                # Fallback: use uniform weights so input dimension matches
                trust_weights = torch.full_like(x, 1.0 / x.shape[-1])
            x = torch.cat([x, trust_weights], dim=-1)
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

    print()
    for mode in ("concat", "weighted"):
        ta_model = TrustAwareTruthDiscoveryMLP(fusion_mode=mode)
        trust = torch.ones(8, 4) / 4
        y_ta  = ta_model(x, trust_weights=trust)
        n     = sum(p.numel() for p in ta_model.parameters())
        print(f"TrustAwareTruthDiscoveryMLP (mode={mode}): "
              f"output {y_ta.shape}, params {n:,}")
