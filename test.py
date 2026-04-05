"""
Few-shot testing for F-TDM with Trusted Worker Mechanism.

Evaluation protocol (matching paper Fig. 4):
  k = 6, 7, 8, 9, 10  learning PoIs (support size)
  For each k, test on varying numbers of query PoIs.

Trust evaluation:
  - Extract attention weights from TrustAttentionTDM for each test sample
  - Load ground_truth_trust.csv and compute:
      * Pearson correlation between attention weights and ground-truth trust scores
      * Classification accuracy: weight > median → "trusted"
"""

from __future__ import annotations

import argparse
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr

from data_preprocessing import load_and_preprocess
from model import TrustAttentionTDM, TruthDiscoveryMLP
from train import (
    ALPHA,
    HIDDEN,
)

TRUST_CSV = "ground_truth_trust.csv"
TEST_TYPE = "NOx(GT)"

# Fig 4(a)-(e): varying test PoI counts for each k
TEST_POI_COUNTS = {
    6:  [60,  70,  80,  90, 100, 110],
    7:  [70,  80,  90, 100, 110, 120],
    8:  [80,  90, 100, 110, 120, 130],
    9:  [90, 100, 110, 120, 130, 140],
    10: [100, 110, 120, 130, 140, 150],
}

# Fig 4(f): fixed 100 test PoIs, k from 5 to 10
FIG4F_TEST_N = 100
FIG4F_K_RANGE = list(range(5, 11))


def _rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def fine_tune_model(
    model,
    sup_x: torch.Tensor,
    sup_y: torch.Tensor,
    n_steps: int = 50,
    lr: float = ALPHA,
) -> torch.nn.Module:
    """Fine-tune a copy of the model on the support set with multiple gradient steps."""
    ft_model = copy.deepcopy(model)
    optimizer = torch.optim.SGD(ft_model.parameters(), lr=lr * 2)  # warmup lr
    ft_model.train()

    for step in range(n_steps):
        if step == 10:
            # Switch to normal lr after warmup
            for pg in optimizer.param_groups:
                pg["lr"] = lr
        optimizer.zero_grad()
        if isinstance(ft_model, TrustAttentionTDM):
            pred, _ = ft_model(sup_x)
        else:
            pred = ft_model(sup_x)
        loss = F.mse_loss(pred, sup_y)
        loss.backward()
        optimizer.step()

    return ft_model


def _load_ground_truth_trust(nox_rows_idx: np.ndarray) -> np.ndarray | None:
    """Load ground-truth trust scores for specified row indices (NOx test rows)."""
    try:
        df = pd.read_csv(TRUST_CSV)
        nox_df = df[df["data_type"] == TEST_TYPE].reset_index(drop=True)
        if len(nox_rows_idx) > len(nox_df):
            return None
        sub = nox_df.iloc[nox_rows_idx]
        scores = sub[
            ["trust_score_1", "trust_score_2", "trust_score_3", "trust_score_4"]
        ].values.astype(np.float32)
        return scores
    except FileNotFoundError:
        return None


def evaluate_trust(
    attn_weights: np.ndarray,
    gt_trust: np.ndarray,
    worker_pool_path: str = "worker_pool.csv",
    worker_ids: np.ndarray | None = None,
) -> dict:
    """
    Compute trust evaluation metrics.

    Parameters
    ----------
    attn_weights    : (N, 4) attention weights from model
    gt_trust        : (N, 4) ground-truth trust scores
    worker_pool_path: path to worker_pool.csv
    worker_ids      : (N, 4) worker IDs for classification accuracy

    Returns
    -------
    dict with keys: pearson_r, pearson_p, classification_accuracy
    """
    # Flatten for correlation
    aw_flat = attn_weights.ravel()
    gt_flat = gt_trust.ravel()

    r, p = pearsonr(aw_flat, gt_flat)

    result = {"pearson_r": float(r), "pearson_p": float(p)}

    # Classification accuracy: weight > median → "trusted"
    if worker_ids is not None:
        try:
            wp = pd.read_csv(worker_pool_path).set_index("worker_id")
        except FileNotFoundError:
            return result

        try:
            medians = np.median(attn_weights, axis=1, keepdims=True)  # (N,1)
            pred_trusted = attn_weights > medians  # (N,4) bool

            n_correct = 0
            n_total = 0
            for i in range(len(worker_ids)):
                for j in range(4):
                    wid = int(worker_ids[i, j])
                    if wid not in wp.index:
                        continue
                    actual_level = wp.at[wid, "trust_level"]
                    model_says_trusted = bool(pred_trusted[i, j])
                    actual_trusted = actual_level == "trusted"
                    n_correct += int(model_says_trusted == actual_trusted)
                    n_total += 1

            if n_total > 0:
                result["classification_accuracy"] = n_correct / n_total
        except KeyError as e:
            print(f"  [evaluate_trust] Skipping classification accuracy: missing column {e}")

    return result


def test(
    model_path: str = "ftdm_trust_model.pth",
    mcs_path: str = "AirQuality_MCS.csv",
    use_baseline: bool = False,
    alpha: float = ALPHA,
    seed: int = 42,
    worker_pool_path: str = "worker_pool.csv",
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[test] Device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    _, _, test_data, _, nox_scaler = load_and_preprocess(
        mcs_path=mcs_path,
        seed=seed,
    )
    X_all = test_data["X"]          # (N, 4)
    y_all = test_data["y"]          # (N, 1)
    wids_all = test_data["worker_ids"]  # (N, 4)

    n_total = len(X_all)
    print(f"[test] NOx test set size: {n_total}")

    # ── Load model ────────────────────────────────────────────────────────────
    if use_baseline:
        model = TruthDiscoveryMLP(input_dim=4, hidden=HIDDEN)
    else:
        model = TrustAttentionTDM(n_workers=4, mlp_hidden=HIDDEN)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"[test] Loaded model from {model_path}")

    # ── Fig 4(a)-(e): vary test PoI counts per k ──────────────────────────────
    print("\n=== Fig 4(a)-(e): RMSE vs Number of Test PoIs ===")
    print(f"{'k':>4}  {'test_n':>8}  {'RMSE (norm)':>12}  {'RMSE (orig)':>12}")

    for k, test_counts in TEST_POI_COUNTS.items():
        for test_n in test_counts:
            needed = k + test_n
            if needed > n_total:
                print(f"  k={k}, test_n={test_n}: not enough data, skipping")
                continue

            # Sequential split: first k as support, next test_n as query
            sup_idx = np.arange(k)
            qry_idx = np.arange(k, k + test_n)

            sup_x = torch.tensor(X_all[sup_idx], dtype=torch.float32).to(device)
            sup_y = torch.tensor(y_all[sup_idx], dtype=torch.float32).to(device)
            qry_x = torch.tensor(X_all[qry_idx], dtype=torch.float32).to(device)
            qry_y = y_all[qry_idx]

            ft_model = fine_tune_model(model, sup_x, sup_y, n_steps=50, lr=alpha)
            ft_model.eval()
            with torch.no_grad():
                if isinstance(ft_model, TrustAttentionTDM):
                    pred_norm, _ = ft_model(qry_x)
                else:
                    pred_norm = ft_model(qry_x)
            pred_norm = pred_norm.cpu().numpy()

            rmse_norm = _rmse(pred_norm, qry_y)
            # Inverse-transform to original scale
            pred_orig = nox_scaler.inverse_transform(pred_norm)
            true_orig = nox_scaler.inverse_transform(qry_y)
            rmse_orig = _rmse(pred_orig, true_orig)

            print(f"  k={k:2d}  test_n={test_n:5d}  {rmse_norm:12.6f}  {rmse_orig:12.4f}")

    # ── Fig 4(f): fixed 100 test PoIs, varying k ─────────────────────────────
    print("\n=== Fig 4(f): RMSE vs k (100 test PoIs) ===")
    print(f"{'k':>4}  {'RMSE (norm)':>12}  {'RMSE (orig)':>12}")

    for k in FIG4F_K_RANGE:
        needed = k + FIG4F_TEST_N
        if needed > n_total:
            print(f"  k={k}: not enough data, skipping")
            continue

        # Sequential split
        sup_idx = np.arange(k)
        qry_idx = np.arange(k, k + FIG4F_TEST_N)

        sup_x = torch.tensor(X_all[sup_idx], dtype=torch.float32).to(device)
        sup_y = torch.tensor(y_all[sup_idx], dtype=torch.float32).to(device)
        qry_x = torch.tensor(X_all[qry_idx], dtype=torch.float32).to(device)
        qry_y_np = y_all[qry_idx]

        ft_model = fine_tune_model(model, sup_x, sup_y, n_steps=50, lr=alpha)
        ft_model.eval()
        with torch.no_grad():
            if isinstance(ft_model, TrustAttentionTDM):
                pred_norm, _ = ft_model(qry_x)
            else:
                pred_norm = ft_model(qry_x)
        pred_norm_np = pred_norm.cpu().numpy()

        rmse_norm = _rmse(pred_norm_np, qry_y_np)
        pred_orig = nox_scaler.inverse_transform(pred_norm_np)
        true_orig = nox_scaler.inverse_transform(qry_y_np)
        rmse_orig = _rmse(pred_orig, true_orig)

        print(f"  k={k:2d}  {rmse_norm:12.6f}  {rmse_orig:12.4f}")

    # ── Trust evaluation (TrustAttentionTDM only) ─────────────────────────────
    if not use_baseline:
        print("\n=== Trust Mechanism Evaluation ===")
        # Use up to 2000 samples for more reliable statistics
        eval_n = min(2000, n_total - 10)
        sup_idx = np.arange(10)
        qry_idx = np.arange(10, 10 + eval_n)

        sup_x = torch.tensor(X_all[sup_idx], dtype=torch.float32).to(device)
        sup_y = torch.tensor(y_all[sup_idx], dtype=torch.float32).to(device)
        qry_x = torch.tensor(X_all[qry_idx], dtype=torch.float32).to(device)

        ft_model = fine_tune_model(model, sup_x, sup_y, n_steps=50, lr=alpha)
        ft_model.eval()
        with torch.no_grad():
            _, attn_weights_t = ft_model(qry_x)
        attn_np = attn_weights_t.cpu().numpy()  # (eval_n, 4)

        gt_trust = _load_ground_truth_trust(qry_idx)
        if gt_trust is not None:
            metrics = evaluate_trust(
                attn_np,
                gt_trust,
                worker_pool_path=worker_pool_path,
                worker_ids=wids_all[qry_idx],
            )
            print(
                f"  Pearson r = {metrics['pearson_r']:.4f}  "
                f"(p = {metrics['pearson_p']:.4e})"
            )
            if "classification_accuracy" in metrics:
                print(
                    f"  Worker trust classification accuracy = "
                    f"{metrics['classification_accuracy']:.4f}"
                )
        else:
            print("  ground_truth_trust.csv not found; skipping trust evaluation")

        # Attention weight statistics for debugging
        print(f"\n=== Attention Weight Statistics ===")
        print(f"  Mean attention weight: {attn_np.mean():.4f}")
        print(f"  Std attention weight: {attn_np.std():.4f}")
        # Per-worker-type breakdown using worker_pool
        try:
            wp = pd.read_csv(worker_pool_path).set_index("worker_id")
            wids_eval = wids_all[qry_idx]  # (eval_n, 4)
            for level in ["trusted", "normal", "malicious"]:
                level_weights = []
                for i in range(len(wids_eval)):
                    for j in range(4):
                        wid = int(wids_eval[i, j])
                        if wid in wp.index and wp.at[wid, "trust_level"] == level:
                            level_weights.append(attn_np[i, j])
                if level_weights:
                    print(
                        f"  {level}: mean_weight={np.mean(level_weights):.4f}  "
                        f"(n={len(level_weights)})"
                    )
        except (FileNotFoundError, KeyError):
            pass


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test F-TDM model")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to saved model (auto-detected if omitted)",
    )
    parser.add_argument("--mcs-path", type=str, default="AirQuality_MCS.csv")
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--worker-pool-path", type=str, default="worker_pool.csv")
    args = parser.parse_args()

    if args.model_path is None:
        args.model_path = (
            "ftdm_baseline_model.pth" if args.baseline else "ftdm_trust_model.pth"
        )

    test(
        model_path=args.model_path,
        mcs_path=args.mcs_path,
        use_baseline=args.baseline,
        alpha=args.alpha,
        worker_pool_path=args.worker_pool_path,
    )
