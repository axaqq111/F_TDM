"""
main.py
-------
Main entry point for the F-TDM experiment pipeline.

Paper: "A Coverage-Aware High-Quality Sensing Data Collection Method
        in Mobile Crowd Sensing"
        IEEE Transactions on Mobile Computing, Vol. 24, No. 4, April 2025

Usage
-----
    python main.py [--csv CSV_PATH] [--epochs N] [--alpha A] [--beta B]
                   [--skip-train] [--model MODEL_PATH]
"""

import argparse
import sys

from train import train, ALPHA, BETA, EPOCHS, MODEL_SAVE
from test  import evaluate, FEW_SHOT_K, TEST_MULTIPLIER


def parse_args():
    parser = argparse.ArgumentParser(
        description="F-TDM: Few-shot samples based Truth Discovery Method"
    )
    parser.add_argument("--csv",         default="AirQualityUCI.csv",
                        help="Path to AirQualityUCI.csv  (default: %(default)s)")
    parser.add_argument("--epochs",      type=int,   default=EPOCHS,
                        help=f"Training epochs  (default: {EPOCHS})")
    parser.add_argument("--alpha",       type=float, default=ALPHA,
                        help=f"Inner-loop learning rate α  (default: {ALPHA})")
    parser.add_argument("--beta",        type=float, default=BETA,
                        help=f"Outer-loop learning rate β  (default: {BETA})")
    parser.add_argument("--model",       default=MODEL_SAVE,
                        help=f"Path to save/load the model  (default: {MODEL_SAVE})")
    parser.add_argument("--skip-train",  action="store_true",
                        help="Skip training and load an existing model for evaluation")
    parser.add_argument("--fine-tune-steps", type=int, default=10,
                        help="Gradient steps during few-shot fine-tuning  (default: 10)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  F-TDM: Few-shot samples based Truth Discovery Method    ║")
    print("║  IEEE TMC, Vol. 24, No. 4, April 2025                    ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    # ── Step 1: Training ───────────────────────────────────────────────────
    if not args.skip_train:
        print("▶ Phase 1 – Meta-learning training on 6 data types")
        print("-" * 60)
        train(
            csv_path   = args.csv,
            alpha      = args.alpha,
            beta       = args.beta,
            epochs     = args.epochs,
            model_save = args.model,
        )
        print()
    else:
        print("▶ Phase 1 – Skipping training (--skip-train flag set)")
        print()

    # ── Step 2: Few-shot evaluation on NOx ────────────────────────────────
    print("▶ Phase 2 – Few-shot fine-tuning and evaluation on NOx(GT)")
    print("-" * 60)
    results = evaluate(
        csv_path        = args.csv,
        model_path      = args.model,
        few_shot_k      = FEW_SHOT_K,
        test_multiplier = TEST_MULTIPLIER,
        fine_tune_steps = args.fine_tune_steps,
    )

    print("\n✔ Experiment complete.")
    return results


if __name__ == "__main__":
    main()
