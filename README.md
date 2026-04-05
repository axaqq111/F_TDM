# F-TDM: Federated Truth Discovery with Trusted Worker Mechanism

This repository implements the **F-TDM** (Few-shot Truth Discovery Model) extended with an **Attention-Based Trusted Worker Mechanism**. The model uses MAML (Model-Agnostic Meta-Learning) to learn a truth-discovery function from multiple data types and generalises to unseen types with only a few labelled PoIs.

---

## Repository Structure

```
F_TDM/
├── AirQualityUCI.csv          # Original UCI dataset (do NOT modify)
├── generate_dataset.py        # Step 1 – Generate simulated MCS dataset
├── data_preprocessing.py      # Step 2 – Load & preprocess MCS data for MAML
├── model.py                   # Step 3 – TrustAttentionTDM + TruthDiscoveryMLP
├── train.py                   # Step 4 – MAML training
├── test.py                    # Step 5 – Few-shot evaluation + trust metrics
├── main.py                    # Step 6 – End-to-end pipeline entry point
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

Generated data files (created by `generate_dataset.py`):
```
worker_pool.csv          – 200 worker profiles (sigma_ratio, outlier_prob, trust_level)
AirQuality_MCS.csv       – Simulated MCS sensing data (4 workers per PoI)
ground_truth_trust.csv   – Per-worker ground-truth trust scores
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate the MCS dataset

```bash
python generate_dataset.py
```

### 3. Train and evaluate

```bash
# Trust mechanism model (default)
python main.py --mode trust --epochs 5000

# Baseline model (original F-TDM)
python main.py --mode baseline --epochs 5000

# Generate data first, then train
python main.py --generate-data --mode trust --epochs 5000
```

---

## Dataset Generation (`generate_dataset.py`)

The original `AirQualityUCI.csv` contains hourly air-quality measurements from an Italian city. Seven columns are used as data types:

| Column          | Role        |
|-----------------|-------------|
| `CO(GT)`        | Train        |
| `PT08.S1(CO)`   | Train        |
| `C6H6(GT)`      | Train        |
| `PT08.S2(NMHC)` | Train        |
| `NO2(GT)`       | Train        |
| `T`             | Train        |
| `NOx(GT)`       | **Test**     |

### Worker Pool (200 workers)

| Group       | IDs      | Count | sigma_ratio       | outlier_prob      |
|-------------|----------|-------|-------------------|-------------------|
| Trusted     | 0 – 39   | 40    | Uniform(0.02, 0.06) | Uniform(0.00, 0.02) |
| Normal      | 40 – 119 | 80    | Uniform(0.10, 0.25) | Uniform(0.05, 0.15) |
| Malicious   | 120–199  | 80    | Uniform(0.30, 0.60) | Uniform(0.20, 0.50) |

Each worker's **actual sigma** for a given data type = `sigma_ratio × data_range_of_that_type`.

Workers are **static** — their sigma_ratio and outlier_prob do not change over time, but at each PoI a fresh set of 4 workers is sampled from the pool (with replacement).

### Sensing Process (per PoI)

For each timestamp and data type:
1. True value taken from the original CSV column (-200 rows skipped).
2. 4 workers sampled **with replacement** from the pool.
3. Each worker generates a value:
   - With probability `(1 - outlier_prob)`: `value = true + N(0, sigma²)`
   - With probability `outlier_prob`: anomalous value = `true × (1 ± Uniform(0.5, 2.0))`
4. Trust score = `1 / (1 + |generated - true| / data_range)` (saved to `ground_truth_trust.csv`)

All random operations use **seed = 42** for reproducibility.

---

## Model Architecture

### TrustAttentionTDM (primary model)

```
Input: [d1, d2, d3, d4]  (4 worker observations)
          ↓
   ┌─────────────────────┐
   │  Attention Module   │
   │  Linear(4 → 64)     │
   │  ReLU               │
   │  Linear(64 → 4)     │
   │  Softmax            │
   │  → [a1, a2, a3, a4] │  ← learned trust weights
   └────────┬────────────┘
            ↓
   Concatenate: d_concat = [d1, d2, d3, d4, a1, a2, a3, a4]  (dim=8)
            ↓
   ┌─────────────────────┐
   │  MLP Regression     │
   │  Linear(8 → 1024)   │
   │  ReLU               │
   │  Linear(1024 → 1)   │
   │  → predicted value  │
   └─────────────────────┘
```

The attention weights serve as a **proxy for worker trustworthiness**. The model learns to assign higher weights to more reliable workers without ever seeing worker IDs — it can only observe the 4 numerical values.

### TruthDiscoveryMLP (baseline)

```
Input: [d1, d2, d3, d4]
          ↓
   Linear(4 → 1024) → ReLU → Linear(1024 → 1)
```

---

## Training (`train.py`)

MAML training framework:
- **Inner loop** (support set): one gradient step with learning rate α = 0.0005
- **Outer loop** (query set): meta-update with SGD, learning rate β = 0.0005
- **Epochs**: 5000
- **Hidden size**: 1024

Saved checkpoints:
- `ftdm_trust_model.pth` — TrustAttentionTDM
- `ftdm_baseline_model.pth` — TruthDiscoveryMLP

---

## Evaluation (`test.py`)

### RMSE (Fig. 4 format)

Few-shot testing on **NOx(GT)**:

| k  | Test PoI counts               |
|----|-------------------------------|
| 6  | 60, 70, 80, 90, 100, 110      |
| 7  | 70, 80, 90, 100, 110, 120     |
| 8  | 80, 90, 100, 110, 120, 130    |
| 9  | 90, 100, 110, 120, 130, 140   |
| 10 | 100, 110, 120, 130, 140, 150  |

Fig. 4(f): fixed 100 test PoIs, k from 5 to 10.

RMSE is reported in both **normalised** and **original** scale.

### Trust Evaluation (TrustAttentionTDM only)

After adaptation:
1. **Pearson correlation** between model attention weights and ground-truth trust scores from `ground_truth_trust.csv`.
2. **Classification accuracy**: attention weight > median → "trusted", compared with actual `trust_level` from `worker_pool.csv`.

---

## Expected Output Format

```
=== Fig 4(a)-(e): RMSE vs Number of Test PoIs ===
   k    test_n   RMSE (norm)   RMSE (orig)
  k= 6  test_n=   60      0.051234      75.6987
  ...

=== Fig 4(f): RMSE vs k (100 test PoIs) ===
   k   RMSE (norm)   RMSE (orig)
  k= 5      0.063450      93.7235
  ...

=== Trust Mechanism Evaluation ===
  Pearson r = 0.6421  (p = 1.23e-45)
  Worker trust classification accuracy = 0.7234
```

---

## Notes

- `AirQualityUCI.csv` is **never modified**.
- The model receives only the 4 numerical values — **worker IDs are invisible to the model**.
- Worker IDs are used only in post-hoc trust evaluation.
- All experiments use random seed 42 for reproducibility.
