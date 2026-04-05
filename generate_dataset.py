"""
Generate a simulated MCS dataset with a worker pool from AirQualityUCI.csv.

Worker pool: 200 workers
  - 40 trusted  (ID  0~ 39, 20%)
  - 80 normal   (ID 40~119, 40%)
  - 80 malicious(ID120~199, 40%)

Output files:
  worker_pool.csv        - worker profiles
  AirQuality_MCS.csv     - sensing data (4 workers sampled per PoI)
  ground_truth_trust.csv - per-worker trust scores based on deviation
"""

import os
import numpy as np
import pandas as pd

SEED = 42
INPUT_CSV = "AirQualityUCI.csv"
WORKER_POOL_CSV = "worker_pool.csv"
MCS_CSV = "AirQuality_MCS.csv"
TRUST_CSV = "ground_truth_trust.csv"

DATA_TYPES = [
    "CO(GT)",
    "PT08.S1(CO)",
    "C6H6(GT)",
    "PT08.S2(NMHC)",
    "NO2(GT)",
    "T",
    "NOx(GT)",
]

N_WORKERS_PER_POI = 4


def _load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Combine Date + Time into a single timestamp column
    df["timestamp"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        dayfirst=True,
        errors="coerce",
    )
    return df


def _compute_ranges(df: pd.DataFrame) -> dict:
    """Compute valid data ranges (excluding -200) for each data type."""
    ranges = {}
    for col in DATA_TYPES:
        valid = df[col][df[col] != -200]
        ranges[col] = float(valid.max() - valid.min())
    return ranges


def _build_worker_pool(rng: np.random.Generator) -> pd.DataFrame:
    """Create worker pool with static sigma_ratio and outlier_prob."""
    records = []

    # 40 trusted workers (ID 0-39)
    for wid in range(40):
        records.append(
            {
                "worker_id": wid,
                "trust_level": "trusted",
                "sigma_ratio": float(rng.uniform(0.02, 0.06)),
                "outlier_prob": float(rng.uniform(0.00, 0.02)),
            }
        )

    # 80 normal workers (ID 40-119)
    for wid in range(40, 120):
        records.append(
            {
                "worker_id": wid,
                "trust_level": "normal",
                "sigma_ratio": float(rng.uniform(0.10, 0.25)),
                "outlier_prob": float(rng.uniform(0.05, 0.15)),
            }
        )

    # 80 malicious workers (ID 120-199)
    for wid in range(120, 200):
        records.append(
            {
                "worker_id": wid,
                "trust_level": "malicious",
                "sigma_ratio": float(rng.uniform(0.30, 0.60)),
                "outlier_prob": float(rng.uniform(0.20, 0.50)),
            }
        )

    return pd.DataFrame(records)


def _generate_worker_value(
    true_value: float,
    sigma: float,
    outlier_prob: float,
    data_range: float,
    rng: np.random.Generator,
) -> float:
    """Generate one worker's reported value."""
    if rng.random() < outlier_prob:
        # Anomalous value: large deviation, do NOT use -200
        multiplier = rng.uniform(0.5, 2.0)
        sign = rng.choice([-1, 1])
        return float(true_value * (1 + sign * multiplier))
    else:
        noise = rng.normal(0, sigma)
        return float(true_value + noise)


def generate(input_csv: str = INPUT_CSV) -> None:
    rng = np.random.default_rng(SEED)

    # ── 1. Load raw data ──────────────────────────────────────────────────────
    df = _load_raw(input_csv)
    data_ranges = _compute_ranges(df)

    # ── 2. Build worker pool ──────────────────────────────────────────────────
    worker_pool_df = _build_worker_pool(rng)
    worker_pool_df.to_csv(WORKER_POOL_CSV, index=False)
    print(f"[generate_dataset] Saved {WORKER_POOL_CSV} ({len(worker_pool_df)} workers)")

    # ── 3. Generate sensing data ──────────────────────────────────────────────
    worker_array = worker_pool_df.to_dict("records")  # list of dicts for speed

    mcs_rows = []
    trust_rows = []

    for _, row in df.iterrows():
        ts = row["timestamp"]

        for dtype in DATA_TYPES:
            true_val = row[dtype]

            # Skip rows with invalid true value
            if true_val == -200 or pd.isna(true_val):
                continue

            data_range = data_ranges[dtype]

            # Sample 4 workers with replacement
            sampled_ids = rng.choice(200, size=N_WORKERS_PER_POI, replace=True)

            worker_values = []
            trust_scores = []

            for wid in sampled_ids:
                w = worker_array[int(wid)]
                sigma = w["sigma_ratio"] * data_range
                val = _generate_worker_value(
                    true_val, sigma, w["outlier_prob"], data_range, rng
                )
                worker_values.append(val)
                # Trust score = 1 / (1 + |generated - true| / data_range)
                ts_score = 1.0 / (
                    1.0 + abs(val - true_val) / (data_range + 1e-9)
                )
                trust_scores.append(ts_score)

            mcs_rows.append(
                {
                    "timestamp": ts,
                    "data_type": dtype,
                    "true_value": true_val,
                    "worker_id_1": int(sampled_ids[0]),
                    "worker_id_2": int(sampled_ids[1]),
                    "worker_id_3": int(sampled_ids[2]),
                    "worker_id_4": int(sampled_ids[3]),
                    "value_1": worker_values[0],
                    "value_2": worker_values[1],
                    "value_3": worker_values[2],
                    "value_4": worker_values[3],
                }
            )

            trust_rows.append(
                {
                    "timestamp": ts,
                    "data_type": dtype,
                    "worker_id_1": int(sampled_ids[0]),
                    "trust_score_1": trust_scores[0],
                    "worker_id_2": int(sampled_ids[1]),
                    "trust_score_2": trust_scores[1],
                    "worker_id_3": int(sampled_ids[2]),
                    "trust_score_3": trust_scores[2],
                    "worker_id_4": int(sampled_ids[3]),
                    "trust_score_4": trust_scores[3],
                }
            )

    mcs_df = pd.DataFrame(mcs_rows)
    trust_df = pd.DataFrame(trust_rows)

    mcs_df.to_csv(MCS_CSV, index=False)
    trust_df.to_csv(TRUST_CSV, index=False)

    print(f"[generate_dataset] Saved {MCS_CSV} ({len(mcs_df)} rows)")
    print(f"[generate_dataset] Saved {TRUST_CSV} ({len(trust_df)} rows)")


if __name__ == "__main__":
    generate()
