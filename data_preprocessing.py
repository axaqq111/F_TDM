"""
data_preprocessing.py
---------------------
Data loading and preprocessing for the F-TDM experiment.

Paper: "A Coverage-Aware High-Quality Sensing Data Collection Method
        in Mobile Crowd Sensing"
        IEEE Transactions on Mobile Computing, Vol. 24, No. 4, April 2025

Steps:
  1. Load AirQualityUCI.csv and extract the 7 selected data types.
  2. Replace outlier marker (-200) with NaN; drop rows that contain any NaN.
  3. Normalize each column with min-max scaling.
  4. Group every 5 consecutive readings into one group:
       - rows 0-3 → human participant data (D_W, shape [4])
       - row  4   → UAV ground truth      (D_U, scalar)
  5. Split data types into 6 training types and 1 test type (NOx(GT)).
  6. For each training data type build tasks, where each task exposes
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
    Load the CSV file and return a cleaned DataFrame containing only the
    7 selected columns.

    Outlier rows (any value == -200) are removed so that the remaining data
    is clean and suitable for normalisation and grouping.
    """
    df = pd.read_csv(csv_path)

    # Keep only the columns we need
    df = df[SELECTED_COLS].copy()

    # Replace outlier sentinel with NaN, then drop rows with any NaN
    df.replace(OUTLIER_VALUE, np.nan, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


# ── Normalisation ─────────────────────────────────────────────────────────────

def normalize_data(df: pd.DataFrame):
    """
    Apply min-max normalisation column-wise.

    Returns
    -------
    df_norm  : pd.DataFrame  – normalised values in [0, 1]
    scalers  : dict          – {col: fitted MinMaxScaler} for inverse transform
    """
    scalers = {}
    df_norm = df.copy()
    for col in df.columns:
        scaler = MinMaxScaler()
        df_norm[col] = scaler.fit_transform(df[[col]])
        scalers[col] = scaler
    return df_norm, scalers


# ── Grouping ──────────────────────────────────────────────────────────────────

def group_into_sets(series: np.ndarray) -> tuple:
    """
    Group a 1-D array of sensor readings into (D_W, D_U) pairs.

    For every GROUP_SIZE (= 5) consecutive readings:
      D_W[k] = series[k*5 : k*5+4]   (4 human participant values)
      D_U[k] = series[k*5 + 4]        (UAV ground truth)

    Returns
    -------
    D_W : np.ndarray, shape (n_groups, 4)
    D_U : np.ndarray, shape (n_groups,)
    """
    n = len(series)
    n_groups = n // GROUP_SIZE
    series = series[: n_groups * GROUP_SIZE]  # truncate to full groups

    reshaped = series.reshape(n_groups, GROUP_SIZE)
    D_W = reshaped[:, :N_HUMAN]   # first 4 columns → human participants
    D_U = reshaped[:, N_HUMAN]    # 5th column      → UAV ground truth
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
    nox_data    : (D_W, D_U, scaler)               – raw NOx groups for testing
    scalers     : dict  {col_name: MinMaxScaler}
    """
    df = load_data(csv_path)
    df_norm, scalers = normalize_data(df)

    # Build tasks for each training data type
    train_tasks = {}
    for col in TRAIN_COLS:
        series = df_norm[col].values
        D_W, D_U = group_into_sets(series)
        train_tasks[col] = build_tasks(D_W, D_U,
                                       support_ratio=support_ratio,
                                       task_size=task_size)

    # NOx data kept separate for few-shot fine-tuning / testing
    nox_series = df_norm[TEST_COL].values
    nox_D_W, nox_D_U = group_into_sets(nox_series)
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
