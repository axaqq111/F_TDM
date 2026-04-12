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
                   [--no-trust] [--fusion-mode {concat,weighted}]
                   [--decay-factor LAMBDA]
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

    # ── Trust-related arguments ────────────────────────────────────────────
    trust_group = parser.add_mutually_exclusive_group()
    trust_group.add_argument(
        "--use-trust",  dest="use_trust", action="store_true", default=True,
        help="Enable dynamic worker trust evaluation (default: enabled)",
    )
    trust_group.add_argument(
        "--no-trust",   dest="use_trust", action="store_false",
        help="Disable trust evaluation for baseline comparison",
    )
    parser.add_argument(
        "--fusion-mode", choices=["concat", "weighted"], default="concat",
        help="Trust fusion mode: 'concat' (default) or 'weighted'",
    )
    parser.add_argument(
        "--decay-factor", type=float, default=0.95,
        help="EMA time-decay factor λ for DynamicTrustManager (default: 0.95)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  F-TDM: Few-shot samples based Truth Discovery Method    ║")
    print("║  IEEE TMC, Vol. 24, No. 4, April 2025                    ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    if args.use_trust:
        print(f"  Dynamic trust: ON  "
              f"(fusion={args.fusion_mode}, λ={args.decay_factor})")
    else:
        print("  Dynamic trust: OFF (baseline mode)")
    print()

    # ── Step 1: Training ───────────────────────────────────────────────────
    if not args.skip_train:
        print("▶ Phase 1 – Meta-learning training on 6 data types")
        print("-" * 60)
        train(
            csv_path     = args.csv,
            alpha        = args.alpha,
            beta         = args.beta,
            epochs       = args.epochs,
            model_save   = args.model,
            use_trust    = args.use_trust,
            fusion_mode  = args.fusion_mode,
            decay_factor = args.decay_factor,
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
        use_trust       = args.use_trust    if not args.skip_train else None,
        fusion_mode     = args.fusion_mode  if not args.skip_train else None,
        decay_factor    = args.decay_factor if not args.skip_train else None,
    )

    print("\n✔ Experiment complete.")
    return results


if __name__ == "__main__":
    main()
