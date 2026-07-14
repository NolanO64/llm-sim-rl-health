"""Cross-dataset generalisation on HPTN 067.

Trajectory-level C2ST is recomputed from the compact, de-identified artifacts.  The
runner summary is retained separately to make the 96 -> 87 -> 77 sample accounting
and the headline post-anchor means explicit.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.artifacts import load_result
from src.paths import REALISM_DIR
from src.realism import c2st_cv, extract_patient_series, residual_features


RUNS = [
    ("hptn067_llm", "llm", "LLM (ours)"),
    ("hptn067_baseline", "anchor_mean", "anchor-mean control"),
]


def main():
    saved = load_result("hptn_summary.json")
    for stem, key, label in RUNS:
        with (REALISM_DIR / f"{stem}.json").open(encoding="utf-8") as handle:
            patients = json.load(handle)["patients"]
        records = [extract_patient_series(patient) for patient in patients]
        post_anchor = sum(bool(record[0]) for record in records)
        real, sim = residual_features([record for record in records if record[0]])
        c2st = c2st_cv(real, sim)
        row = saved[key]
        assert len(patients) == row["n_total_patients"]
        assert post_anchor == row["n_post_anchor_patients"]
        assert len(real) == row["n_c2st_patients"]
        print(
            f"{label:<20} exp={row['experiment']}  "
            f"C2ST={c2st['pooled']:.3f}+/-{c2st['sd']:.3f}  "
            f"Spearman={row['spearman_patient']:.3f}  "
            f"heterogeneity={row['heterogeneity_ratio']:.2f}  "
            f"mean sim/real={row['mean_simulated']:.3f}/{row['mean_real']:.3f}  "
            f"patients={len(patients)}->{post_anchor}->{len(real)}"
        )


if __name__ == "__main__":
    main()
