"""
test.py
-------
Few-shot fine-tuning on NOx(GT) and RMSE evaluation for F-TDM.

Paper: "A Coverage-Aware High-Quality Sensing Data Collection Method
        in Mobile Crowd Sensing"
        IEEE Transactions on Mobile Computing, Vol. 24, No. 4, April 2025

Few-shot testing protocol (Section V / Fig. 4 of the paper):
  - After training on 6 data types, fine-tune on NOx(GT) with only a
    few samples: 6, 7, 8, 9, and 10 PoIs (learning PoIs).
  - Fig 4(a)-(e): For each k, test on multiple numbers of predicted PoIs
    starting at 10×k and incrementing by 10.
  - Fig 4(f): Fixed 100 test PoIs, k varies from 5 to 10.
  - Evaluation metric: RMSE = sqrt( sum(y_i - yhat_i)^2 / n )
    where y_i and yhat_i are on the **original NOx scale** (inverse-transformed).
"""

import copy
import json
import os

import numpy as np
import torch
import torch.nn as nn

from data_preprocessing import preprocess
from model import TruthDiscoveryMLP, TrustAwareTruthDiscoveryMLP
from trust_manager import DynamicTrustManager

# ── Hyper-parameters ──────────────────────────────────────────────────────────
ALPHA        = 0.0005   # fine-tuning learning rate (same as inner-loop α)
HIDDEN_SIZE  = 1024
INPUT_SIZE   = 4
FEW_SHOT_K   = [6, 7, 8, 9, 10]   # number of learning PoIs (for backward compat)
TEST_MULTIPLIER = 10               # kept for backward compatibility with main.py
MODEL_LOAD   = "ftdm_model.pth"

# ── Fig 4(a)-(e): test multiple predicted PoIs counts per k ──────────────────
# "The number of test PoIs started at a minimum of ten times the number of
#  learning PoIs, subsequently increased incrementally by 10."
TEST_CONFIGS = {
    6:  [60, 70, 80, 90, 100, 110],
    7:  [70, 80, 90, 100, 110, 120],
    8:  [80, 90, 100, 110, 120, 130],
    9:  [90, 100, 110, 120, 130, 140],
    10: [100, 110, 120, 130, 140, 150],
}

# ── Fig 4(f): fixed 100 test PoIs, k varies from 5 to 10 ─────────────────────
FIG4F_K_VALUES  = [5, 6, 7, 8, 9, 10]
FIG4F_TEST_POIS = 100


# ── Helpers ───────────────────────────────────────────────────────────────────

def to_tensor(arr: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(arr, dtype=dtype)


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """RMSE = sqrt( sum(y - yhat)^2 / n )"""
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def _load_model_and_config(model_path: str):
    """
    Load the model and its accompanying config JSON.

    Returns (model, use_trust, fusion_mode, decay_factor).
    Falls back gracefully when no config file exists (legacy baseline model).
    """
    # Read config if present
    config_path = model_path.replace(".pth", "_config.json")
    if os.path.isfile(config_path):
        with open(config_path) as f:
            config = json.load(f)
        use_trust    = config.get("use_trust", False)
        fusion_mode  = config.get("fusion_mode", "concat")
        decay_factor = config.get("decay_factor", 0.95)
        hidden_size  = config.get("hidden_size", HIDDEN_SIZE)
        input_size   = config.get("input_size",  INPUT_SIZE)
    else:
        use_trust    = False
        fusion_mode  = "concat"
        decay_factor = 0.95
        hidden_size  = HIDDEN_SIZE
        input_size   = INPUT_SIZE

    # Build the appropriate model class
    if use_trust:
        model = TrustAwareTruthDiscoveryMLP(
            input_size=input_size,
            hidden_size=hidden_size,
            fusion_mode=fusion_mode,
        )
    else:
        model = TruthDiscoveryMLP(input_size=input_size,
                                  hidden_size=hidden_size)

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model, use_trust, fusion_mode, decay_factor


def fine_tune(model: nn.Module,
              support_W: torch.Tensor,
              support_U: torch.Tensor,
              n_steps: int = 10,
              lr: float = ALPHA,
              trust_manager: DynamicTrustManager = None) -> nn.Module:
    """
    Fine-tune a *copy* of the model on the few-shot support set.

    Parameters
    ----------
    model         : pre-trained model (TruthDiscoveryMLP or
                    TrustAwareTruthDiscoveryMLP)
    support_W     : shape (k, 4)  – human participant readings (normalised)
    support_U     : shape (k,)    – UAV ground truth (normalised)
    n_steps       : number of gradient steps during fine-tuning
    lr            : fine-tuning learning rate
    trust_manager : optional DynamicTrustManager; if provided, trust weights
                    are computed from the support set and passed to the model

    Returns
    -------
    fine-tuned model copy
    """
    ft_model = copy.deepcopy(model)
    optimizer = torch.optim.SGD(ft_model.parameters(), lr=lr)

    # Compute trust weights if a manager is provided
    if trust_manager is not None:
        trust_manager.batch_update(
            support_W.numpy(), support_U.numpy()
        )
        weights = trust_manager.get_weights()
        trust_t = torch.tensor(
            np.tile(weights, (len(support_W), 1)),
            dtype=torch.float32,
            requires_grad=False,
        )
    else:
        trust_t = None

    ft_model.train()
    for _ in range(n_steps):
        optimizer.zero_grad()
        if trust_t is not None:
            pred = ft_model(support_W, trust_weights=trust_t).squeeze(-1)
        else:
            pred = ft_model(support_W).squeeze(-1)
        loss = nn.functional.mse_loss(pred, support_U)
        loss.backward()
        optimizer.step()

    return ft_model


# ── Main evaluation function (Fig 4(a)-(e)) ──────────────────────────────────

def evaluate(csv_path: str  = "AirQualityUCI.csv",
             model_path: str = MODEL_LOAD,
             few_shot_k: list = FEW_SHOT_K,
             test_multiplier: int = TEST_MULTIPLIER,
             fine_tune_steps: int = 10,
             use_trust: bool = None,
             fusion_mode: str = None,
             decay_factor: float = None):
    """
    Load the trained model, fine-tune on NOx few-shot samples, and report
    RMSE on the **original NOx scale** for each (k, test_pois) combination,
    reproducing Fig. 4(a)-(e) of the paper.

    Parameters
    ----------
    csv_path        : path to AirQualityUCI.csv
    model_path      : path to the saved model weights
    few_shot_k      : list of learning PoI counts (default: [6,7,8,9,10])
    test_multiplier : kept for backward compatibility; TEST_CONFIGS is used
                      internally for k values defined there
    fine_tune_steps : gradient steps during few-shot fine-tuning
    use_trust       : override trust flag (default: read from config file)
    fusion_mode     : override fusion mode (default: read from config file)
    decay_factor    : override decay factor (default: read from config file)
    """
    # ── Load pre-trained model ─────────────────────────────────────────────
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            "Run train.py first."
        )

    model, cfg_use_trust, cfg_fusion, cfg_decay = _load_model_and_config(
        model_path
    )

    # CLI overrides take precedence over config file values
    if use_trust    is not None:  cfg_use_trust = use_trust
    if fusion_mode  is not None:  cfg_fusion    = fusion_mode
    if decay_factor is not None:  cfg_decay     = decay_factor

    print(f"Loaded model from '{model_path}'.")
    if cfg_use_trust:
        print(f"  Trust mode: ON  (fusion={cfg_fusion}, λ={cfg_decay})")
    else:
        print("  Trust mode: OFF (baseline)")

    # ── Preprocessing – NOx data ───────────────────────────────────────────
    _, nox_data, _ = preprocess(csv_path)
    nox_W, nox_U, nox_scaler = nox_data   # normalised arrays + scaler

    n_total = len(nox_W)
    print(f"Total NOx groups available: {n_total}\n")

    # ── Fig 4(a)-(e) evaluation ────────────────────────────────────────────
    print("=" * 64)
    print(" Fig 4(a)-(e): F-TDM Few-shot Evaluation on NOx(GT)")
    print(" (RMSE on original NOx scale)")
    print("=" * 64)

    results = {}

    for k in few_shot_k:
        # Determine test PoIs list: use TEST_CONFIGS if defined for this k,
        # else fall back to the single test_multiplier×k value
        if k in TEST_CONFIGS:
            test_pois_list = TEST_CONFIGS[k]
        else:
            test_pois_list = [test_multiplier * k]

        print(f"\n--- k={k} (learning PoIs: {k}) ---")
        print(f"  {'Test PoIs':>9} | {'RMSE':>10}")

        results[k] = {}

        # Support set: first k groups (normalised) — shared across all test_pois
        sup_W = to_tensor(nox_W[:k])
        sup_U = to_tensor(nox_U[:k])

        # Create a fresh trust manager for NOx fine-tuning (if trust is on)
        tm = (DynamicTrustManager(n_workers=INPUT_SIZE, decay_factor=cfg_decay)
              if cfg_use_trust else None)

        # Fine-tune once per k on the support set
        ft_model = fine_tune(model, sup_W, sup_U,
                             n_steps=fine_tune_steps, lr=ALPHA,
                             trust_manager=tm)
        ft_model.eval()

        # Build trust weights tensor for inference (if trust is on)
        if tm is not None:
            weights_np = tm.get_weights()
        else:
            weights_np = None

        for n_test in test_pois_list:
            # Test set: next n_test groups after the support set
            end_idx = k + n_test
            if end_idx > n_total:
                n_test = max(1, n_total - k)
                end_idx = n_total

            test_W      = to_tensor(nox_W[k:end_idx])
            test_U_norm = nox_U[k:end_idx]

            if len(test_W) == 0:
                print(f"  {n_test:>9} | {'N/A':>10}")
                continue

            # Predict on test set (normalised)
            with torch.no_grad():
                if weights_np is not None:
                    tw = torch.tensor(
                        np.tile(weights_np, (len(test_W), 1)),
                        dtype=torch.float32,
                    )
                    pred_norm = ft_model(test_W,
                                        trust_weights=tw).squeeze(-1).numpy()
                else:
                    pred_norm = ft_model(test_W).squeeze(-1).numpy()

            # Inverse-transform to original scale for RMSE
            pred_orig   = nox_scaler.inverse_transform(
                pred_norm.reshape(-1, 1)
            ).flatten()
            test_U_orig = nox_scaler.inverse_transform(
                test_U_norm.reshape(-1, 1)
            ).flatten()

            error = rmse(pred_orig, test_U_orig)
            results[k][n_test] = error
            print(f"  {n_test:>9} | {error:>10.4f}")

    print("\n" + "=" * 64)

    # ── Also run Fig 4(f) ─────────────────────────────────────────────────
    evaluate_fig4f(csv_path=csv_path, model_path=model_path,
                   model=model, nox_W=nox_W, nox_U=nox_U,
                   nox_scaler=nox_scaler, fine_tune_steps=fine_tune_steps,
                   use_trust=cfg_use_trust, decay_factor=cfg_decay)

    return results


# ── Fig 4(f) evaluation function ─────────────────────────────────────────────

def evaluate_fig4f(csv_path: str   = "AirQualityUCI.csv",
                   model_path: str = MODEL_LOAD,
                   model=None,
                   nox_W=None,
                   nox_U=None,
                   nox_scaler=None,
                   fine_tune_steps: int = 10,
                   use_trust: bool = False,
                   decay_factor: float = 0.95):
    """
    Reproduce Fig. 4(f): fixed 100 test PoIs, k varies from 5 to 10.

    "The experiment was set 100 identical PoIs, and all the methods learned
     sensed PoIs from 5 to 10."

    Parameters
    ----------
    csv_path        : path to AirQualityUCI.csv (used only if model/data not provided)
    model_path      : path to the saved model weights (used only if model not provided)
    model           : pre-loaded model (optional, avoids redundant loading)
    nox_W           : normalised human-participant NOx array (optional)
    nox_U           : normalised UAV ground-truth NOx array (optional)
    nox_scaler      : fitted MinMaxScaler for NOx (optional)
    fine_tune_steps : gradient steps during few-shot fine-tuning
    use_trust       : whether to use dynamic trust during fine-tuning
    decay_factor    : EMA decay factor for DynamicTrustManager
    """
    # Load model and data only if not already provided
    if model is None:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Trained model not found at '{model_path}'. "
                "Run train.py first."
            )
        model, use_trust, _, decay_factor = _load_model_and_config(model_path)
        model.eval()
        print(f"Loaded model from '{model_path}'.")

    if nox_W is None or nox_U is None or nox_scaler is None:
        _, nox_data, _ = preprocess(csv_path)
        nox_W, nox_U, nox_scaler = nox_data

    n_total = len(nox_W)

    print("\n" + "=" * 64)
    print(" Fig 4(f): Fixed 100 Test PoIs, Varying Learning Samples")
    print("=" * 64)
    print(f"  {'k (samples)':>11} | {'Test PoIs':>10} | {'RMSE':>10}")

    results_f = {}

    for k in FIG4F_K_VALUES:
        n_test  = FIG4F_TEST_POIS
        end_idx = k + n_test
        if end_idx > n_total:
            n_test  = max(1, n_total - k)
            end_idx = n_total

        sup_W = to_tensor(nox_W[:k])
        sup_U = to_tensor(nox_U[:k])

        test_W      = to_tensor(nox_W[k:end_idx])
        test_U_norm = nox_U[k:end_idx]

        if len(test_W) == 0:
            print(f"  {k:>11} | {n_test:>10} | {'N/A':>10}")
            continue

        # Create a fresh trust manager for each k (if trust is on)
        tm = (DynamicTrustManager(n_workers=INPUT_SIZE, decay_factor=decay_factor)
              if use_trust else None)

        ft_model = fine_tune(model, sup_W, sup_U,
                             n_steps=fine_tune_steps, lr=ALPHA,
                             trust_manager=tm)
        ft_model.eval()

        with torch.no_grad():
            if tm is not None:
                weights_np = tm.get_weights()
                tw = torch.tensor(
                    np.tile(weights_np, (len(test_W), 1)),
                    dtype=torch.float32,
                )
                pred_norm = ft_model(test_W,
                                     trust_weights=tw).squeeze(-1).numpy()
            else:
                pred_norm = ft_model(test_W).squeeze(-1).numpy()

        pred_orig   = nox_scaler.inverse_transform(
            pred_norm.reshape(-1, 1)
        ).flatten()
        test_U_orig = nox_scaler.inverse_transform(
            test_U_norm.reshape(-1, 1)
        ).flatten()

        error = rmse(pred_orig, test_U_orig)
        results_f[k] = error
        print(f"  {k:>11} | {n_test:>10} | {error:>10.4f}")

    print("=" * 64)

    return results_f


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    evaluate()
