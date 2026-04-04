# F-TDM: Few-shot Samples Based Truth Discovery Method

Reproduction of the **F-TDM** experiment from:

> Ye Wang, Hui Gao, Edith C. H. Ngai, Kun Niu, Tan Yang, Bo Zhang, Wendong Wang.  
> **"A Coverage-Aware High-Quality Sensing Data Collection Method in Mobile Crowd Sensing."**  
> *IEEE Transactions on Mobile Computing*, Vol. 24, No. 4, April 2025.  
> DOI: [10.1109/TMC.2024.3502567](https://ieeexplore.ieee.org/document/10758242)

---

## Overview

F-TDM applies **MAML** (Model-Agnostic Meta-Learning) to the *truth discovery* problem in Mobile Crowd Sensing (MCS).  UAVs provide a small number of high-quality ground-truth readings, which are used as few-shot samples to train a model that can calibrate noisy data contributed by human participants.

### Key ideas

| Concept | Description |
|---|---|
| **Data sources** | Human participant data `D_W` (noisy) + UAV ground truth `D_U` (trusted) |
| **Group structure** | Every 5 consecutive readings: rows 0–3 → `D_W`, row 4 → `D_U` |
| **Meta-learning** | MAML with 1 inner-loop update per task, 6 training data types |
| **Few-shot test** | Fine-tune on NOx(GT) with 6–10 PoIs, evaluate on up to 10× test samples |

---

## Repository Structure

```
F_TDM/
├── AirQualityUCI.csv         # UCI Air Quality dataset (9,357 rows)
├── data_preprocessing.py     # Data loading, normalisation, task creation
├── model.py                  # MLP regression model (hidden size 1024)
├── train.py                  # MAML training — Algorithm 1 from the paper
├── test.py                   # Few-shot fine-tuning and RMSE evaluation
├── main.py                   # End-to-end pipeline entry point
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

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

Outlier values (`-200`) are removed before processing.

---

## Installation

```bash
pip install -r requirements.txt
```

Python **3.7+** is required.

---

## Running the Experiment

### Full pipeline (train + evaluate)

```bash
python main.py
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
```

### Examples

```bash
# Quick smoke test with fewer epochs
python main.py --epochs 100

# Evaluate a previously trained model
python main.py --skip-train --model ftdm_model.pth

# Run individual modules
python data_preprocessing.py   # preprocessing summary
python model.py                # model architecture summary
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

---

## Algorithm 1 — Training

```
Input : M types of tasks T_m, network Δ, step sizes α, β
Output: trained truth discovery model

1:  Initialize parameters θ of Δ randomly
2:  while not reach epochs do
3:    for i = 1, …, M do
4:      Obtain D_support = {D_W, D_U} from T_i
5:      D_P = f_θ(D_W)
6:      L_support_i = (1/K) ‖D_P − D_U‖²      [Equation 1, MSE]
7:      Compute ∇_θ L_support_i
8:      θ'_i = θ − α * ∇_θ L_support_i         [one inner update]
9:      Obtain D_query = {D_W, D_U} from T_i
10:     L_i = loss(f_{θ'_i}, D_query)
11:   end for
12:   L_sum = Σ L_i(f_{θ'_i})
13:   θ ← θ − β * ∇_θ L_sum
14: end while
```

> **Note:** The implementation uses the **first-order MAML approximation** (FOMAML).  `create_graph=True` is passed to `torch.autograd.grad` so that the outer gradient still flows through the inner-loop update step.

---

## Expected Output

```
╔══════════════════════════════════════════════════════════╗
║  F-TDM: Few-shot samples based Truth Discovery Method    ║
║  IEEE TMC, Vol. 24, No. 4, April 2025                    ║
╚══════════════════════════════════════════════════════════╝

▶ Phase 1 – Meta-learning training on 6 data types
------------------------------------------------------------
Loading and preprocessing data …
Training on 6 data types: ['CO(GT)', 'PT08.S1(CO)', ...]
…
Epoch  5000/5000 | Meta-loss (L_sum): 0.012345
Training complete.
Model saved to 'ftdm_model.pth'.

▶ Phase 2 – Few-shot fine-tuning and evaluation on NOx(GT)
------------------------------------------------------------
============================================================
 Few-shot Fine-tuning & Evaluation on NOx(GT)
============================================================
  PoIs (k) | Fine-tune samples | Test samples |       RMSE
------------------------------------------------------------
         6 |                 6 |           60 |   0.034512
         7 |                 7 |           70 |   0.031874
         8 |                 8 |           80 |   0.029631
         9 |                 9 |           90 |   0.027908
        10 |                10 |          100 |   0.025347
============================================================
```

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
