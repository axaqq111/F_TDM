"""
Main entry point for F-TDM experiment with Trusted Worker Mechanism.

Usage:
    python generate_dataset.py                    # Generate new dataset
    python main.py --mode trust --epochs 5000     # Train + test with trust mechanism
    python main.py --mode baseline --epochs 5000  # Train + test original F-TDM as baseline
    python main.py --generate-data --mode trust   # Generate data, then train + test
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(
        description="F-TDM Trusted Worker Mechanism Experiment"
    )
    parser.add_argument(
        "--mode",
        choices=["trust", "baseline"],
        default="trust",
        help="Model mode: 'trust' for TrustAttentionTDM, 'baseline' for TruthDiscoveryMLP",
    )
    parser.add_argument(
        "--generate-data",
        action="store_true",
        help="Generate new MCS dataset before training",
    )
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--alpha", type=float, default=0.0005)
    parser.add_argument("--beta", type=float, default=0.0005)
    parser.add_argument("--support-size", type=int, default=10)
    parser.add_argument("--mcs-path", type=str, default="AirQuality_MCS.csv")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--worker-pool-path", type=str, default="worker_pool.csv")
    args = parser.parse_args()

    use_baseline = args.mode == "baseline"

    # ── Step 1: (Optional) Generate dataset ───────────────────────────────────
    if args.generate_data or not os.path.exists(args.mcs_path):
        print("=" * 60)
        print("Step 1: Generating MCS dataset …")
        print("=" * 60)
        from generate_dataset import generate
        generate()
    else:
        print(f"[main] Using existing dataset: {args.mcs_path}")

    # ── Step 2: Train model ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Step 2: Training ({args.mode} mode, {args.epochs} epochs) …")
    print("=" * 60)

    from train import train

    model_path = args.model_path
    if model_path is None:
        model_path = (
            "ftdm_baseline_model.pth" if use_baseline else "ftdm_trust_model.pth"
        )

    train(
        mcs_path=args.mcs_path,
        epochs=args.epochs,
        alpha=args.alpha,
        beta=args.beta,
        use_baseline=use_baseline,
        save_path=model_path,
        support_size=args.support_size,
    )

    # ── Step 3: Evaluate on NOx ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 3: Evaluating on NOx(GT) test set …")
    print("=" * 60)

    from test import test

    test(
        model_path=model_path,
        mcs_path=args.mcs_path,
        use_baseline=use_baseline,
        alpha=args.alpha,
        worker_pool_path=args.worker_pool_path,
    )

    print("\n" + "=" * 60)
    print("Experiment complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
