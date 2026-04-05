"""
MAML training for F-TDM with Trusted Worker Mechanism.

Algorithm 1 (Model-Agnostic Meta-Learning):
  For each meta-iteration:
    1. Sample a task (support set, query set)
    2. Compute adapted parameters via inner-loop gradient step on support set
    3. Evaluate adapted parameters on query set
    4. Update meta-parameters via outer-loop gradient step

Supports two models:
  --baseline  : TruthDiscoveryMLP (original F-TDM)
  (default)   : TrustAttentionTDM (attention-based trust mechanism)
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_preprocessing import load_and_preprocess
from model import TrustAttentionTDM, TruthDiscoveryMLP

ALPHA = 0.0005   # inner-loop learning rate
BETA = 0.0005    # outer-loop learning rate
EPOCHS = 5000
HIDDEN = 1024
SEED = 42


# ─── Adapted forward passes ───────────────────────────────────────────────────

def adapted_forward_trust(
    x: torch.Tensor,
    params: list[torch.Tensor],
) -> torch.Tensor:
    """
    Manual forward pass for TrustAttentionTDM with custom parameters.

    params order:
      [attn_w0, attn_b0, attn_w1, attn_b1, mlp_w0, mlp_b0, mlp_w1, mlp_b1]
    """
    attn_w0, attn_b0, attn_w1, attn_b1, mlp_w0, mlp_b0, mlp_w1, mlp_b1 = params

    # Attention module
    h = F.relu(F.linear(x, attn_w0, attn_b0))
    logits = F.linear(h, attn_w1, attn_b1)
    weights = F.softmax(logits, dim=-1)

    # Weighted aggregation
    d_agg = (weights * x).sum(dim=-1, keepdim=True)

    # MLP
    h2 = F.relu(F.linear(d_agg, mlp_w0, mlp_b0))
    pred = F.linear(h2, mlp_w1, mlp_b1)
    return pred


def adapted_forward_baseline(
    x: torch.Tensor,
    params: list[torch.Tensor],
) -> torch.Tensor:
    """
    Manual forward pass for TruthDiscoveryMLP with custom parameters.

    params order: [w0, b0, w1, b1]
    """
    w0, b0, w1, b1 = params
    h = F.relu(F.linear(x, w0, b0))
    return F.linear(h, w1, b1)


# ─── Inner loop ───────────────────────────────────────────────────────────────

def inner_update(
    model: nn.Module,
    sup_x: torch.Tensor,
    sup_y: torch.Tensor,
    alpha: float,
    use_baseline: bool,
) -> list[torch.Tensor]:
    """One gradient step on the support set; returns adapted parameters."""
    params = list(model.parameters())

    if use_baseline:
        pred = adapted_forward_baseline(sup_x, params)
    else:
        pred = adapted_forward_trust(sup_x, params)

    loss = F.mse_loss(pred, sup_y)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    adapted = [p - alpha * g for p, g in zip(params, grads)]
    return adapted


# ─── Training ─────────────────────────────────────────────────────────────────

def train(
    mcs_path: str = "AirQuality_MCS.csv",
    epochs: int = EPOCHS,
    alpha: float = ALPHA,
    beta: float = BETA,
    use_baseline: bool = False,
    save_path: str | None = None,
    support_size: int = 10,
    seed: int = SEED,
) -> nn.Module:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("[train] Loading data …")
    train_tasks, _, _, _ = load_and_preprocess(
        mcs_path=mcs_path,
        support_size=support_size,
        seed=seed,
    )
    print(f"[train] {len(train_tasks)} meta-tasks loaded")

    # ── Model ─────────────────────────────────────────────────────────────────
    if use_baseline:
        model = TruthDiscoveryMLP(input_dim=4, hidden=HIDDEN).to(device)
        model_name = "TruthDiscoveryMLP (baseline)"
    else:
        model = TrustAttentionTDM(n_workers=4, mlp_hidden=HIDDEN).to(device)
        model_name = "TrustAttentionTDM"
    print(f"[train] Model: {model_name}")

    optimizer = optim.SGD(model.parameters(), lr=beta)

    # ── MAML training loop ────────────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        # Sample a random task
        task_idx = int(rng.integers(0, len(train_tasks)))
        sup_x, sup_y, qry_x, qry_y = train_tasks[task_idx]

        sup_x = torch.tensor(sup_x, dtype=torch.float32).to(device)
        sup_y = torch.tensor(sup_y, dtype=torch.float32).to(device)
        qry_x = torch.tensor(qry_x, dtype=torch.float32).to(device)
        qry_y = torch.tensor(qry_y, dtype=torch.float32).to(device)

        # Inner update
        adapted_params = inner_update(model, sup_x, sup_y, alpha, use_baseline)

        # Outer loss on query set with adapted params
        if use_baseline:
            qry_pred = adapted_forward_baseline(qry_x, adapted_params)
        else:
            qry_pred = adapted_forward_trust(qry_x, adapted_params)

        outer_loss = F.mse_loss(qry_pred, qry_y)

        optimizer.zero_grad()
        outer_loss.backward()
        optimizer.step()

        if epoch % 500 == 0 or epoch == 1:
            print(f"[train] Epoch {epoch:5d}/{epochs}  outer_loss={outer_loss.item():.6f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    if save_path is None:
        save_path = (
            "ftdm_baseline_model.pth" if use_baseline else "ftdm_trust_model.pth"
        )
    torch.save(model.state_dict(), save_path)
    print(f"[train] Model saved to {save_path}")
    return model


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train F-TDM model")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Train with TruthDiscoveryMLP (baseline) instead of TrustAttentionTDM",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--beta", type=float, default=BETA)
    parser.add_argument("--support-size", type=int, default=10)
    parser.add_argument("--mcs-path", type=str, default="AirQuality_MCS.csv")
    parser.add_argument("--save-path", type=str, default=None)
    args = parser.parse_args()

    train(
        mcs_path=args.mcs_path,
        epochs=args.epochs,
        alpha=args.alpha,
        beta=args.beta,
        use_baseline=args.baseline,
        save_path=args.save_path,
        support_size=args.support_size,
    )
