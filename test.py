"""
test.py
-------
Few-shot fine-tuning on NOx(GT) and RMSE evaluation for F-TDM.

Paper: "A Coverage-Aware High-Quality Sensing Data Collection Method
        in Mobile Crowd Sensing"
        IEEE Transactions on Mobile Computing, Vol. 24, No. 4, April 2025

Few-shot testing protocol (Section V of the paper):
  - After training on 6 data types, fine-tune on NOx(GT) with only a
    few samples: 6, 7, 8, 9, and 10 PoIs (learning PoIs).
  - Each learning PoI consists of 4 human participant values + 1 UAV
    ground truth (one group).
  - Test on samples up to 10 × the number of learning PoIs.
  - Evaluation metric: RMSE = sqrt( Σ(y_i - ŷ_i)² / n )
"""

import copy
import os

import numpy as np
import torch
import torch.nn as nn

from data_preprocessing import preprocess
from model import TruthDiscoveryMLP

# ── Hyper-parameters ──────────────────────────────────────────────────────────
ALPHA        = 0.0005   # fine-tuning learning rate (same as inner-loop α)
HIDDEN_SIZE  = 1024
INPUT_SIZE   = 4
FEW_SHOT_K   = [6, 7, 8, 9, 10]   # number of learning PoIs to evaluate
TEST_MULTIPLIER = 10               # test on up to 10× the learning PoIs
MODEL_LOAD   = "ftdm_model.pth"


# ── Helpers ───────────────────────────────────────────────────────────────────

def to_tensor(arr: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(arr, dtype=dtype)


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """RMSE = sqrt( Σ(y - ŷ)² / n )"""
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def fine_tune(model: nn.Module,
              support_W: torch.Tensor,
              support_U: torch.Tensor,
              n_steps: int = 10,
              lr: float = ALPHA) -> nn.Module:
    """
    Fine-tune a *copy* of the model on the few-shot support set.

    Parameters
    ----------
    model     : pre-trained TruthDiscoveryMLP
    support_W : shape (k, 4)  – human participant readings
    support_U : shape (k,)    – UAV ground truth
    n_steps   : number of gradient steps during fine-tuning
    lr        : fine-tuning learning rate

    Returns
    -------
    fine-tuned model copy
    """
    ft_model = copy.deepcopy(model)
    optimizer = torch.optim.SGD(ft_model.parameters(), lr=lr)

    ft_model.train()
    for _ in range(n_steps):
        optimizer.zero_grad()
        pred = ft_model(support_W).squeeze(-1)
        loss = nn.functional.mse_loss(pred, support_U)
        loss.backward()
        optimizer.step()

    return ft_model


# ── Main evaluation function ──────────────────────────────────────────────────

def evaluate(csv_path: str  = "AirQualityUCI.csv",
             model_path: str = MODEL_LOAD,
             few_shot_k: list = FEW_SHOT_K,
             test_multiplier: int = TEST_MULTIPLIER,
             fine_tune_steps: int = 10):
    """
    Load the trained model, fine-tune on NOx few-shot samples, and report
    RMSE for each few-shot setting.

    Parameters
    ----------
    csv_path        : path to AirQualityUCI.csv
    model_path      : path to the saved model weights
    few_shot_k      : list of PoI counts to test (e.g. [6,7,8,9,10])
    test_multiplier : test set size = test_multiplier × k
    fine_tune_steps : gradient steps during few-shot fine-tuning
    """
    # ── Load pre-trained model ─────────────────────────────────────────────
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            "Run train.py first."
        )

    model = TruthDiscoveryMLP(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    print(f"Loaded model from '{model_path}'.")

    # ── Preprocessing – NOx data ───────────────────────────────────────────
    _, nox_data, _ = preprocess(csv_path)
    nox_W, nox_U, nox_scaler = nox_data   # normalised arrays

    n_total = len(nox_W)
    print(f"Total NOx groups available: {n_total}\n")

    # ── Few-shot evaluation loop ───────────────────────────────────────────
    print("=" * 60)
    print(" Few-shot Fine-tuning & Evaluation on NOx(GT)")
    print("=" * 60)
    print(f"{'PoIs (k)':>10} | {'Fine-tune samples':>18} | "
          f"{'Test samples':>13} | {'RMSE':>10}")
    print("-" * 60)

    results = {}

    for k in few_shot_k:
        # We need k support groups + up to 10k test groups
        n_needed = k + test_multiplier * k
        if n_needed > n_total:
            # Use whatever is available, keeping at least k for fine-tuning
            n_test = max(1, n_total - k)
        else:
            n_test = test_multiplier * k

        # Support set: first k groups
        sup_W = to_tensor(nox_W[:k])
        sup_U = to_tensor(nox_U[:k])

        # Test set: the next n_test groups (no UAV ground truth used here)
        test_W = to_tensor(nox_W[k: k + n_test])
        test_U = nox_U[k: k + n_test]  # kept as numpy for RMSE calculation

        if len(test_W) == 0:
            print(f"{k:>10} | {k:>18} | {'N/A':>13} | {'N/A':>10}")
            continue

        # Fine-tune on the k support samples
        ft_model = fine_tune(model, sup_W, sup_U,
                             n_steps=fine_tune_steps, lr=ALPHA)

        # Predict on test set (human participant data only)
        ft_model.eval()
        with torch.no_grad():
            pred_norm = ft_model(test_W).squeeze(-1).numpy()

        # Compute RMSE in normalised space
        error = rmse(pred_norm, test_U)

        results[k] = error
        print(f"{k:>10} | {k:>18} | {len(test_W):>13} | {error:>10.6f}")

    print("=" * 60)
    print("\nSummary: RMSE per few-shot setting")
    for k, r in results.items():
        print(f"  k = {k:2d} PoIs  →  RMSE = {r:.6f}")

    return results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    evaluate()
