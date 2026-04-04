"""
data_preprocessing.py
---------------------
Data loading and preprocessing for the F-TDM experiment.

Paper: "A Coverage-Aware High-Quality Sensing Data Collection Method
        in Mobile Crowd Sensing"
        IEEE Transactions on Mobile Computing, Vol. 24, No. 4, April 2025

Steps:
  1. Load AirQualityUCI.csv and extract the 7 selected data types.
  2. Keep outlier rows (-200) for human participant data (D_W) to simulate
     noisy/unreliable contributions; only skip groups where the UAV value
     (5th element, D_U) is -200.
  3. Fit min-max scalers on clean values only (excluding -200 outliers).
  4. Normalize data; replace -200 values in D_W with 0.0 (placeholder).
  5. Group every 5 consecutive readings into one group:
       - rows 0-3 → human participant data (D_W, shape [4])
       - row  4   → UAV ground truth      (D_U, scalar)
  6. Skip groups whose UAV value was originally -200.
  7. Split data types into 6 training types and 1 test type (NOx(GT)).
  8. For each training data type build tasks, where each task exposes
     a support set and a query set (both are (D_W, D_U) pairs).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ── Column configuration ──────────────────────────────────────────────────────
# 7 data types selected from the dataset (Table II / problem statement)
SELECTED_COLS = [
    "CO(GT)",
    "PT08.S1(CO)",
    "C6H6(GT)",
    "PT08.S2(NMHC)",
    "NO2(GT)",
    "T",
    "NOx(GT)",   # reserved for few-shot testing
]

# Training types (6) and test type (1)
TRAIN_COLS = [c for c in SELECTED_COLS if c != "NOx(GT)"]
TEST_COL   = "NOx(GT)"

OUTLIER_VALUE = -200      # sentinel value used in the dataset
GROUP_SIZE    = 5         # 4 human participants + 1 UAV per group
N_HUMAN       = 4         # number of human participant readings per group

# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(csv_path: str = "AirQualityUCI.csv") -> pd.DataFrame:
    """
    Load the CSV file and return a DataFrame containing only the 7 selected
    columns.  Outlier rows (-200) are kept so that realistic noise in human
    participant data (D_W) is preserved.
    """
    df = pd.read_csv(csv_path)

    # Keep only the columns we need
    df = df[SELECTED_COLS].copy()
    df.reset_index(drop=True, inplace=True)

    return df


# ── Normalisation ─────────────────────────────────────────────────────────────

def normalize_data(df: pd.DataFrame):
    """
    Apply min-max normalisation column-wise, fitting only on non-outlier values.
    Outlier entries (-200) are replaced with 0.0 after fitting so they act as a
    known placeholder; the scaler range is determined by real sensor readings.

    Returns
    -------
    df_norm  : pd.DataFrame  – normalised values in [0, 1] (outliers → 0.0)
    scalers  : dict          – {col: fitted MinMaxScaler} for inverse transform
    """
    scalers = {}
    df_norm = df.copy().astype(float)
    for col in df.columns:
        raw = df[col].values.astype(float)
        clean = raw[raw != OUTLIER_VALUE].reshape(-1, 1)
        scaler = MinMaxScaler()
        scaler.fit(clean)
        # Transform all values, then replace outlier-marked positions with 0.0
        transformed = scaler.transform(raw.reshape(-1, 1)).flatten()
        transformed[raw == OUTLIER_VALUE] = 0.0
        df_norm[col] = transformed
        scalers[col] = scaler
    return df_norm, scalers


# ── Grouping ──────────────────────────────────────────────────────────────────

def group_into_sets(series_orig: np.ndarray,
                    series_norm: np.ndarray) -> tuple:
    """
    Group 1-D arrays of sensor readings into (D_W, D_U) pairs.

    For every GROUP_SIZE (= 5) consecutive readings:
      D_W[k] = series[k*5 : k*5+4]   (4 human participant values)
      D_U[k] = series[k*5 + 4]        (UAV ground truth)

    Groups whose UAV value was originally -200 are skipped.
    Outlier (-200) values in the human participant slots of D_W have already
    been replaced with 0.0 in `normalize_data`.

    Parameters
    ----------
    series_orig : original (non-normalised) values, used to detect -200
    series_norm : normalised values (outliers already replaced with 0.0)

    Returns
    -------
    D_W : np.ndarray, shape (n_valid_groups, 4)
    D_U : np.ndarray, shape (n_valid_groups,)
    """
    n = len(series_orig)
    n_groups = n // GROUP_SIZE
    orig = series_orig[: n_groups * GROUP_SIZE].reshape(n_groups, GROUP_SIZE)
    norm = series_norm[: n_groups * GROUP_SIZE].reshape(n_groups, GROUP_SIZE)

    # Keep groups where the UAV (5th) reading was NOT an outlier
    uav_orig = orig[:, N_HUMAN]
    valid_mask = uav_orig != OUTLIER_VALUE

    D_W = norm[valid_mask, :N_HUMAN]
    D_U = norm[valid_mask, N_HUMAN]
    return D_W, D_U


# ── Task construction ─────────────────────────────────────────────────────────

def build_tasks(D_W: np.ndarray, D_U: np.ndarray,
                support_ratio: float = 0.5,
                task_size: int = 20) -> list:
    """
    Split (D_W, D_U) into a list of tasks.  Each task is a dict with
    'support' and 'query' keys, each holding (D_W_subset, D_U_subset).

    Parameters
    ----------
    D_W          : shape (n_groups, 4)
    D_U          : shape (n_groups,)
    support_ratio: fraction of each task used as the support set
    task_size    : number of groups per task (the rest form the query set
                   inside each task)
    """
    n_groups = len(D_W)
    tasks = []

    idx = 0
    while idx + task_size <= n_groups:
        chunk_W = D_W[idx: idx + task_size]
        chunk_U = D_U[idx: idx + task_size]

        split = int(task_size * support_ratio)
        tasks.append({
            "support": (chunk_W[:split],  chunk_U[:split]),
            "query":   (chunk_W[split:],  chunk_U[split:]),
        })
        idx += task_size

    return tasks


# ── Main preprocessing pipeline ───────────────────────────────────────────────

def preprocess(csv_path: str = "AirQualityUCI.csv",
               support_ratio: float = 0.5,
               task_size: int = 20):
    """
    Full preprocessing pipeline.

    Returns
    -------
    train_tasks : dict  {col_name: list_of_tasks}  – 6 training data types
    nox_data    : (D_W, D_U, scaler)               – NOx groups for testing
    scalers     : dict  {col_name: MinMaxScaler}
    """
    df = load_data(csv_path)
    df_norm, scalers = normalize_data(df)

    # Build tasks for each training data type
    train_tasks = {}
    for col in TRAIN_COLS:
        series_orig = df[col].values.astype(float)
        series_norm = df_norm[col].values
        D_W, D_U = group_into_sets(series_orig, series_norm)
        train_tasks[col] = build_tasks(D_W, D_U,
                                       support_ratio=support_ratio,
                                       task_size=task_size)

    # NOx data kept separate for few-shot fine-tuning / testing
    nox_orig = df[TEST_COL].values.astype(float)
    nox_norm = df_norm[TEST_COL].values
    nox_D_W, nox_D_U = group_into_sets(nox_orig, nox_norm)
    nox_data = (nox_D_W, nox_D_U, scalers[TEST_COL])

    return train_tasks, nox_data, scalers


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_tasks, nox_data, scalers = preprocess()

    print("=== Preprocessing Summary ===")
    print(f"Training data types: {list(train_tasks.keys())}")
    for col, tasks in train_tasks.items():
        s_size = len(tasks[0]["support"][0])
        q_size = len(tasks[0]["query"][0])
        print(f"  {col}: {len(tasks)} tasks, "
              f"support={s_size} groups, query={q_size} groups")

    nox_W, nox_U, _ = nox_data
    print(f"\nNOx(GT) groups for testing: {len(nox_W)}")
    print(f"D_W shape per group: {nox_W.shape}")
    print("Preprocessing complete.")
