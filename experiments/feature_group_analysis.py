"""Reproduce the feature-group diagnosis behind Section 4.1.

For the four prompt-only anchor baselines this prints real/simulated residual-feature
means and C2ST ablations.  The K=35 row establishes the two prose claims: simulated
within-patient standard deviation is about one quarter of reality, and removing the
action-response group lowers separability more than removing another single group.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import REALISM_DIR
from src.realism import c2st_cv, extract_patient_series, residual_features


FEATURES = [
    "std", "autocorr", "volatility", "up_rate",
    "action_response_0", "action_response_1", "action_response_2",
]
GROUPS = {
    "full": range(7),
    "no_std": [1, 2, 3, 4, 5, 6],
    "no_autocorr": [0, 2, 3, 4, 5, 6],
    "no_vol/up": [0, 1, 4, 5, 6],
    "no_action": [0, 1, 2, 3],
    "temporal_only": [1, 2, 3],
    "action_only": [4, 5, 6],
    "dispersion_only": [0],
}


def select(rows, indices):
    return [[row[index] for index in indices] for row in rows]


def main():
    for anchor in (0, 14, 35, 70):
        path = REALISM_DIR / f"heartsteps_baseline_K{anchor}.json"
        with path.open(encoding="utf-8") as handle:
            patients = json.load(handle)["patients"]
        records = [extract_patient_series(patient) for patient in patients]
        real, sim = residual_features(records)
        real_mean = [statistics.mean(row[j] for row in real) for j in range(7)]
        sim_mean = [statistics.mean(row[j] for row in sim) for j in range(7)]
        print(f"\nK={anchor} (n={len(real)})")
        for name, rmean, smean in zip(FEATURES, real_mean, sim_mean):
            print(f"  {name:<18} real={rmean:+.4f}  sim={smean:+.4f}")
        print("  C2ST probability-gap ablations:")
        for name, indices in GROUPS.items():
            result = c2st_cv(select(real, indices), select(sim, indices))
            print(f"    {name:<17} gap={result['gap']:.3f}  accuracy={result['pooled']:.3f}")


if __name__ == "__main__":
    main()
