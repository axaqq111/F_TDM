# F-TDM: Few-shot Samples Based Truth Discovery Method

Reproduction of the **F-TDM** experiment from:

> Ye Wang, Hui Gao, Edith C. H. Ngai, Kun Niu, Tan Yang, Bo Zhang, Wendong Wang.  
> **"A Coverage-Aware High-Quality Sensing Data Collection Method in Mobile Crowd Sensing."**  
> *IEEE Transactions on Mobile Computing*, Vol. 24, No. 4, April 2025.  
> DOI: [10.1109/TMC.2024.3502567](https://ieeexplore.ieee.org/document/10758242)

---

## Overview

F-TDM applies **MAML** (Model-Agnostic Meta-Learning) to the *truth discovery* problem in Mobile Crowd Sensing (MCS).  UAVs provide a small number of high-quality ground-truth readings, which are used as few-shot samples to train a model that can calibrate noisy data contributed by human participants.

This implementation adds an optional **Dynamic Worker Trust Evaluation** mechanism on top of the baseline F-TDM, which learns each worker's reliability over time and uses it to improve prediction quality.

### Key ideas

| Concept | Description |
|---|---|
| **Data sources** | Human participant data `D_W` (noisy) + UAV ground truth `D_U` (trusted) |
| **Group structure** | Every 5 consecutive readings: rows 0–3 → `D_W`, row 4 → `D_U` |
| **Meta-learning** | MAML with 1 inner-loop update per task, 6 training data types |
| **Few-shot test** | Fine-tune on NOx(GT) with 6–10 PoIs, evaluate on up to 10× test samples |
| **Dynamic trust** | Per-worker trust scores updated via EMA; used to weight/augment `D_W` |

---

## Repository Structure

```
F_TDM/
├── AirQualityUCI.csv         # UCI Air Quality dataset (9,357 rows)
├── data_preprocessing.py     # Data loading, normalisation, task creation
├── model.py                  # MLP regression models (baseline + trust-aware)
├── trust_manager.py          # Dynamic worker trust evaluation module (NEW)
├── train.py                  # MAML training — Algorithm 1 from the paper
├── test.py                   # Few-shot fine-tuning and RMSE evaluation
├── main.py                   # End-to-end pipeline entry point
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## Dynamic Worker Trust Evaluation

### Motivation

In real Mobile Crowd Sensing scenarios, workers have varying reliability — some consistently provide accurate readings while others are noisy or unreliable.  The dynamic trust mechanism learns each worker's reliability over time, so the model can use this information to produce better predictions.

### Trust Update Rule

After each task/group, the trust score for worker `i` is updated using an **exponential moving average (EMA)** with a time-decay factor `λ`:

```
trust_i = λ * trust_i + (1 - λ) * (1 - error_i / max_error)
```

- `error_i = |d_w[i] - d_u|` — absolute deviation from the UAV ground truth
- `λ` (default 0.95) — higher values give more weight to historical performance
- Scores are clipped to `[0.01, 1.0]` to prevent zero weights
- Workers with lower errors receive higher trust increments

Trust weights are then normalised to sum to 1 and passed to the neural network.

### Fusion Modes

Two ways to incorporate trust weights into the MLP are supported:

| Mode | Description | Input size | Recommended? |
|---|---|---|---|
| **`concat`** (default) | Concatenate `D_W` (4-dim) and `trust_weights` (4-dim) → 8-dim input | 8 | ✅ Yes |
| **`weighted`** | Element-wise multiply `D_W * trust_weights` → 4-dim input | 4 | For ablation |

`concat` is the recommended default because it gives the neural network full flexibility to learn how trust information should be combined with worker readings.  The `weighted` mode forces a multiplicative relationship which may be too restrictive.

### Implementation

The `DynamicTrustManager` class in `trust_manager.py`:
- Maintains trust scores (initialised to 0.5) for all workers
- Updates scores via `update_trust(d_w, d_u)` or `batch_update(D_W, D_U)`
- Returns normalised weights via `get_weights()`
- Stores the full history in `trust_history` for analysis
- Can be serialised/deserialised with `state_dict()` / `load_state_dict()`
- Supports `reset()` to reinitialise

The `TrustAwareTruthDiscoveryMLP` class in `model.py` wraps the standard MLP and accepts an optional `trust_weights` tensor in its `forward()` method.  When `trust_weights=None`, it falls back to standard F-TDM behaviour.

Trust weights have `requires_grad=False` — they are computed by the trust manager, not learned by backpropagation.

---

## Dataset

The file `AirQualityUCI.csv` is already present in the repository root.  
It contains **9,357 hourly air quality measurements** with the following columns:

```
Date, Time, CO(GT), PT08.S1(CO), NMHC(GT), C6H6(GT), PT08.S2(NMHC),
NOx(GT), PT08.S3(NOx), NO2(GT), PT08.S4(NO2), PT08.S5(O3), T, RH, AH
```

### Selected data types

| # | Column | Role |
|---|---|---|
| 1 | `CO(GT)` | Training |
| 2 | `PT08.S1(CO)` | Training |
| 3 | `C6H6(GT)` | Training |
| 4 | `PT08.S2(NMHC)` | Training |
| 5 | `NO2(GT)` | Training |
| 6 | `T` (Temperature) | Training |
| 7 | `NOx(GT)` | **Test only** (few-shot) |

Outlier values (`-200`) are handled as follows:
- **Human participant data (D_W):** `-200` entries are kept to simulate noisy/unreliable contributions. The min-max scaler is fitted on clean values only (excluding `-200`), and `-200` entries in D_W are replaced with `0.0` after normalization as a placeholder.
- **UAV ground truth (D_U):** Any group whose UAV value is `-200` is skipped entirely, since reliable ground truth is required for training and evaluation.

---

## Installation

```bash
pip install -r requirements.txt
```

Python **3.7+** is required.

---

## Running the Experiment

### Full pipeline (train + evaluate) — with dynamic trust (default)

```bash
python main.py
```

### Baseline F-TDM (no trust) — reproduces original paper results

```bash
python main.py --no-trust
```

### Options

```
--csv             Path to AirQualityUCI.csv   (default: AirQualityUCI.csv)
--epochs          Training epochs             (default: 5000)
--alpha           Inner-loop learning rate α  (default: 0.0005)
--beta            Outer-loop learning rate β  (default: 0.0005)
--model           Path to save/load model     (default: ftdm_model.pth)
--skip-train      Skip training, load model for evaluation only
--fine-tune-steps Gradient steps in fine-tuning (default: 10)

Trust-related options:
--use-trust       Enable dynamic worker trust evaluation (default)
--no-trust        Disable trust for baseline comparison
--fusion-mode     Trust fusion mode: 'concat' (default) or 'weighted'
--decay-factor    EMA time-decay factor λ (default: 0.95)
```

### Examples

```bash
# Quick smoke test with fewer epochs
python main.py --epochs 100

# Evaluate a previously trained model (auto-detects trust config)
python main.py --skip-train --model ftdm_model.pth

# Baseline F-TDM (original paper, no trust)
python main.py --no-trust

# Trust-aware with weighted fusion mode (ablation experiment)
python main.py --fusion-mode weighted

# Tune the EMA decay factor
python main.py --decay-factor 0.9

# Run individual modules
python data_preprocessing.py   # preprocessing summary
python model.py                # model architecture summary
python trust_manager.py        # trust manager demo
python train.py                # training only
python test.py                 # evaluation only (model must exist)
```

---

## Experimental Parameters (Table II)

| Parameter | Value |
|---|---|
| Inner-loop updates per task | 1 |
| Hidden size | 1024 |
| Inner-loop learning rate α | 0.0005 |
| Outer-loop learning rate β | 0.0005 |
| Training epochs | 5000 |
| Trust EMA decay factor λ | 0.95 (default) |

---

## Algorithm 1 — Training

```
Input : M types of tasks T_m, network Δ, step sizes α, β
Output: trained truth discovery model

1:  Initialize parameters θ of Δ randomly
2:  while not reach epochs do
3:    for i = 1, …, M do
4:      Obtain D_support = {D_W, D_U} from T_i
5:      [Trust ON] get trust_weights from DynamicTrustManager[i]
6:      D_P = f_θ(D_W, trust_weights)
7:      L_support_i = (1/K) ‖D_P − D_U‖²      [Equation 1, MSE]
8:      Compute ∇_θ L_support_i
9:      θ'_i = θ − α * ∇_θ L_support_i         [one inner update]
10:     Obtain D_query = {D_W, D_U} from T_i
11:     L_i = loss(f_{θ'_i}, D_query, trust_weights)
12:     [Trust ON] update DynamicTrustManager[i] using D_support
13:   end for
14:   L_sum = Σ L_i(f_{θ'_i})
15:   θ ← θ − β * ∇_θ L_sum
16: end while
```

> **Note:** The implementation uses the **first-order MAML approximation** (FOMAML).  `create_graph=True` is passed to `torch.autograd.grad` so that the outer gradient still flows through the inner-loop update step.

---

## Citation

```bibtex
@article{wang2025coverage,
  title   = {A Coverage-Aware High-Quality Sensing Data Collection Method
             in Mobile Crowd Sensing},
  author  = {Wang, Ye and Gao, Hui and Ngai, Edith C. H. and Niu, Kun and
             Yang, Tan and Zhang, Bo and Wang, Wendong},
  journal = {IEEE Transactions on Mobile Computing},
  volume  = {24},
  number  = {4},
  year    = {2025},
  doi     = {10.1109/TMC.2024.3502567}
}
```
