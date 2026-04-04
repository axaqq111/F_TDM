"""
train.py
--------
MAML-based meta-learning training for F-TDM (Algorithm 1).

Paper: "A Coverage-Aware High-Quality Sensing Data Collection Method
        in Mobile Crowd Sensing"
        IEEE Transactions on Mobile Computing, Vol. 24, No. 4, April 2025

Algorithm 1 (Truth Discovery Model Training Procedure)
-------------------------------------------------------
Input : M types of tasks T_m, regression network Δ, step sizes α and β
Output: trained truth discovery model

1:  Initialize parameters θ of Δ randomly
2:  while not reach epochs do
3:    for i = 1, …, M do
4:      Obtain support dataset D_support = {D_W, D_U} from T_i
5:      Use current model to predict: D_P = f_θ(D_W)
6:      Evaluate L_support_i = (1/K) ‖D_P − D_U‖²   [Equation 1, MSE]
7:      Evaluate ∇_θ L_support_i
8:      Compute adapted parameters: θ'_i = θ − α ∇_θ L_support_i
9:      Obtain query dataset D_query = {D_W, D_U} from T_i
10:     Evaluate L_i with weights θ'_i on D_query
11:   end for
12:   Evaluate L_sum = Σ L_i(f_{θ'_i})
13:   Update θ ← θ − β ∇_θ L_sum
14: end while

Note on gradients
-----------------
Standard MAML requires second-order gradients (∇_θ of the query loss
computed through the inner-loop update step).  This implementation uses
the **first-order approximation** (FOMAML): the inner-loop gradient step
is performed with `torch.autograd.grad` while keeping the computational
graph attached so that the outer gradient flows through the adapted
parameters.  This is memory-efficient and matches the single-step inner
loop described in the paper.
"""

import copy
import os
import random

import numpy as np
import torch
import torch.nn as nn

from data_preprocessing import preprocess
from model import TruthDiscoveryMLP

# ── Hyper-parameters (Table II) ───────────────────────────────────────────────
ALPHA        = 0.0005   # inner-loop learning rate
BETA         = 0.0005   # outer-loop learning rate
EPOCHS       = 5000     # number of training epochs
HIDDEN_SIZE  = 1024     # MLP hidden units
INPUT_SIZE   = 4        # D_W dimension (4 human participants)
PRINT_EVERY  = 100      # print loss every N epochs
MODEL_SAVE   = "ftdm_model.pth"


# ── Helpers ───────────────────────────────────────────────────────────────────

def to_tensor(arr: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(arr, dtype=dtype)


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Equation 1 from the paper:
        L = (1/K) * ||D_P - D_U||^2
    This is the standard MSE loss.
    """
    return nn.functional.mse_loss(pred.squeeze(-1), target)


# ── Inner-loop adapted forward pass ──────────────────────────────────────────

def adapted_forward(model: nn.Module,
                    x: torch.Tensor,
                    params: list) -> torch.Tensor:
    """
    Run a forward pass of the 3-layer MLP using the supplied *params* list
    instead of model.parameters().  This allows the outer loop to
    differentiate through the inner-loop update.

    The model architecture is:
        Linear(4 → 1024) → ReLU → Linear(1024 → 1)

    params order: [w0, b0, w1, b1]
    """
    w0, b0, w1, b1 = params
    x = torch.relu(x @ w0.T + b0)
    x = x @ w1.T + b1
    return x


# ── Single inner-loop update (Line 7-8 of Algorithm 1) ───────────────────────

def inner_update(model: nn.Module,
                 support_W: torch.Tensor,
                 support_U: torch.Tensor,
                 alpha: float) -> list:
    """
    Perform a single gradient-descent step on the support set and return
    the adapted parameters θ'_i.

    Parameters
    ----------
    model      : the current meta-model with parameters θ
    support_W  : D_W for the support set, shape (K_s, 4)
    support_U  : D_U for the support set, shape (K_s,)
    alpha      : inner-loop learning rate

    Returns
    -------
    adapted_params : list of updated parameter tensors (still in graph)
    """
    # Line 5: D_P = f_θ(D_W)
    params = list(model.parameters())

    # Forward pass on support set using current parameters
    pred = adapted_forward(model, support_W, params)

    # Line 6: L_support_i  (Equation 1)
    loss = mse_loss(pred, support_U)

    # Line 7: ∇_θ L_support_i
    # create_graph=True keeps the second-order graph for outer gradient
    grads = torch.autograd.grad(loss, params, create_graph=True)

    # Line 8: θ'_i = θ - α * ∇_θ L_support_i
    adapted_params = [p - alpha * g for p, g in zip(params, grads)]
    return adapted_params


# ── Main training function ────────────────────────────────────────────────────

def train(csv_path: str = "AirQualityUCI.csv",
          alpha: float = ALPHA,
          beta:  float = BETA,
          epochs: int  = EPOCHS,
          model_save: str = MODEL_SAVE):
    """
    Execute Algorithm 1: MAML-based meta-learning training.

    Returns the trained TruthDiscoveryMLP model.
    """
    # ── Preprocessing ──────────────────────────────────────────────────────
    print("Loading and preprocessing data …")
    train_tasks, nox_data, scalers = preprocess(csv_path)

    col_names  = list(train_tasks.keys())   # M training data types
    M          = len(col_names)
    print(f"Training on {M} data types: {col_names}")

    # Flatten all tasks per type for random sampling during training
    all_tasks = {col: train_tasks[col] for col in col_names}
    total_tasks = {col: len(tasks) for col, tasks in all_tasks.items()}
    print(f"Tasks per type: {total_tasks}")

    # ── Model initialisation (Line 1) ───────────────────────────────────────
    model = TruthDiscoveryMLP(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE)
    outer_optimizer = torch.optim.Adam(model.parameters(), lr=beta)

    print(f"\nStarting training for {epochs} epochs …")
    print(f"  α (inner lr) = {alpha},  β (outer lr) = {beta}")
    print("-" * 60)

    # ── Outer loop (Line 2: while not reach epochs) ─────────────────────────
    for epoch in range(1, epochs + 1):
        outer_optimizer.zero_grad()

        query_losses = []

        # ── Inner loop (Line 3: for i = 1, …, M) ──────────────────────────
        for col in col_names:
            tasks = all_tasks[col]
            # Sample one task from this data type
            task = random.choice(tasks)

            sup_W, sup_U = task["support"]
            qry_W, qry_U = task["query"]

            # Convert to tensors
            sup_W_t = to_tensor(sup_W)
            sup_U_t = to_tensor(sup_U)
            qry_W_t = to_tensor(qry_W)
            qry_U_t = to_tensor(qry_U)

            # Line 4-8: inner update → θ'_i
            adapted_params = inner_update(model, sup_W_t, sup_U_t, alpha)

            # Line 9-10: evaluate on query set with θ'_i
            qry_pred = adapted_forward(model, qry_W_t, adapted_params)
            L_i = mse_loss(qry_pred, qry_U_t)
            query_losses.append(L_i)

        # Line 12: L_sum = Σ L_i
        L_sum = sum(query_losses)

        # Line 13: θ ← θ - β * ∇_θ L_sum  (via Adam)
        L_sum.backward()
        outer_optimizer.step()

        if epoch % PRINT_EVERY == 0 or epoch == 1:
            print(f"Epoch {epoch:5d}/{epochs} | "
                  f"Meta-loss (L_sum): {L_sum.item():.6f}")

    print("-" * 60)
    print("Training complete.")

    # Save the trained model
    torch.save(model.state_dict(), model_save)
    print(f"Model saved to '{model_save}'.")

    return model


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train()
