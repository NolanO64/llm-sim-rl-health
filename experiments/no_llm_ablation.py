"""Appendix Table A7: LLM latent versus anchor-mean latent.

C2ST, Spearman, and heterogeneity are computed from the saved trajectories. The
mean column is read from each run summary because it uses post-anchor accounting.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import REALISM_DIR
from src.realism import (
    _patient_descriptives,
    _spearman,
    _std,
    c2st_cv,
    extract_patient_series,
    residual_features,
)


RUNS = {
    2: ("heartsteps_emission_llm_K2", "heartsteps_nollm_K2"),
    5: ("heartsteps_emission_llm_K5", "heartsteps_nollm_K5"),
    14: ("heartsteps_emission_llm_K14", "heartsteps_nollm_K14"),
    35: ("heartsteps_emission_two_part", "heartsteps_nollm_K35"),
}


def evaluate(stem):
    with (REALISM_DIR / f"{stem}.json").open(encoding="utf-8") as handle:
        data = json.load(handle)
    records = [extract_patient_series(patient) for patient in data["patients"]]
    records = [record for record in records if record[0]]
    c2st = c2st_cv(*residual_features(records))
    real_means, sim_means, _ = _patient_descriptives(records)
    point = {
        "spearman": _spearman(real_means, sim_means),
        "heterogeneity": _std(sim_means) / _std(real_means),
    }
    summary = data.get("summary") or {}
    point["mean"] = summary.get("mean_simulated")
    return c2st, point, data.get("exp")


def main():
    print("K   LLM exp / no-LLM exp   C2ST LLM / no-LLM      Spearman       heterogeneity       mean")
    for anchor, stems in RUNS.items():
        llm = evaluate(stems[0])
        base = evaluate(stems[1])
        print(
            f"{anchor:<3d} {llm[2]:>3} / {base[2]:<3}             "
            f"{llm[0]['pooled']:.3f}+/-{llm[0]['sd']:.3f} / "
            f"{base[0]['pooled']:.3f}+/-{base[0]['sd']:.3f}   "
            f"{llm[1]['spearman']:.3f} / {base[1]['spearman']:.3f}   "
            f"{llm[1]['heterogeneity']:.2f} / {base[1]['heterogeneity']:.2f}        "
            f"{llm[1]['mean']:.3f} / {base[1]['mean']:.3f}"
        )
if __name__ == "__main__":
    main()
