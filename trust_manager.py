"""
trust_manager.py
----------------
Dynamic worker trust evaluation for F-TDM.

Workers in Mobile Crowd Sensing vary in reliability.  This module maintains
a per-worker trust score that is updated via exponential moving average (EMA)
after each task/group, so that recent performance influences the score more
than older readings.

Update rule (EMA with time decay):
    trust_i = λ * trust_i + (1 - λ) * (1 - error_i / max_error)

where λ (decay_factor, default 0.95) is the time decay factor and error_i
is the absolute deviation of worker i's reading from the UAV ground truth.

Scores are clipped to [0.01, 1.0] to prevent zero weights.
"""

import numpy as np


class DynamicTrustManager:
    """
    Dynamic trust evaluation for a fixed number of workers.

    Parameters
    ----------
    n_workers    : number of human participants per group (default 4)
    init_trust   : initial trust score for all workers (default 0.5)
    decay_factor : EMA time-decay factor λ ∈ (0, 1) (default 0.95).
                   Higher values give more weight to historical performance.
    """

    def __init__(self,
                 n_workers: int = 4,
                 init_trust: float = 0.5,
                 decay_factor: float = 0.95):
        self.n_workers    = n_workers
        self.decay_factor = decay_factor

        self.trust_scores: np.ndarray = np.full(n_workers, init_trust,
                                                dtype=np.float64)
        # Full history of trust scores (each entry is a snapshot)
        self.trust_history: list = [self.trust_scores.copy()]

    # ── Single-group update ───────────────────────────────────────────────────

    def update_trust(self, d_w: np.ndarray, d_u: float) -> None:
        """
        Update trust scores using one group's readings and UAV ground truth.

        Parameters
        ----------
        d_w : shape (n_workers,) — normalised worker readings for this group
        d_u : scalar             — normalised UAV ground truth for this group
        """
        errors = np.abs(d_w - d_u)
        max_error = np.max(errors) + 1e-8   # guard against division by zero

        # Workers with smaller errors get a higher incremental trust
        error_based_trust = 1.0 - errors / max_error

        # EMA update: recent performance has more weight
        lam = self.decay_factor
        self.trust_scores = (lam * self.trust_scores
                             + (1.0 - lam) * error_based_trust)

        # Clip to [0.01, 1.0] to avoid degenerate zero weights
        self.trust_scores = np.clip(self.trust_scores, 0.01, 1.0)
        self.trust_history.append(self.trust_scores.copy())

    # ── Batch update ─────────────────────────────────────────────────────────

    def batch_update(self, D_W: np.ndarray, D_U: np.ndarray) -> None:
        """
        Update trust scores over a batch of groups (e.g. a support set).

        Parameters
        ----------
        D_W : shape (n_groups, n_workers) — worker readings
        D_U : shape (n_groups,)           — UAV ground truths
        """
        for i in range(len(D_W)):
            self.update_trust(D_W[i], float(D_U[i]))

    # ── Query ─────────────────────────────────────────────────────────────────

    def get_weights(self) -> np.ndarray:
        """
        Return trust weights normalised to sum to 1.

        Returns
        -------
        weights : shape (n_workers,), dtype float64, sums to 1.0
        """
        total = self.trust_scores.sum()
        return self.trust_scores / total

    # ── Persistence helpers ───────────────────────────────────────────────────

    def state_dict(self) -> dict:
        """Return a serialisable snapshot of the current state."""
        return {
            "n_workers":    self.n_workers,
            "decay_factor": self.decay_factor,
            "trust_scores": self.trust_scores.tolist(),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore state from a previously saved snapshot."""
        self.n_workers    = state["n_workers"]
        self.decay_factor = state["decay_factor"]
        self.trust_scores = np.array(state["trust_scores"], dtype=np.float64)

    # ── Utility ───────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset all trust scores to the initial value of 0.5."""
        self.trust_scores = np.full(self.n_workers, 0.5, dtype=np.float64)
        self.trust_history = [self.trust_scores.copy()]

    def __repr__(self) -> str:  # pragma: no cover
        weights = self.get_weights()
        scores  = ", ".join(f"{s:.4f}" for s in self.trust_scores)
        ws      = ", ".join(f"{w:.4f}" for w in weights)
        return (f"DynamicTrustManager(n_workers={self.n_workers}, "
                f"λ={self.decay_factor}, "
                f"scores=[{scores}], weights=[{ws}])")


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    tm  = DynamicTrustManager(n_workers=4, decay_factor=0.95)

    print("Initial state:", tm)

    # Simulate 20 groups: worker 0 is very accurate, worker 3 is noisy
    for _ in range(20):
        true_val = rng.uniform(0.3, 0.7)
        d_w = np.array([
            true_val + rng.normal(0, 0.01),   # accurate
            true_val + rng.normal(0, 0.05),   # moderate
            true_val + rng.normal(0, 0.10),   # noisy
            true_val + rng.normal(0, 0.20),   # very noisy
        ])
        tm.update_trust(d_w, true_val)

    print("After 20 updates:", tm)
    print(f"Weights: {tm.get_weights()}")
