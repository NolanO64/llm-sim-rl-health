"""Emission-parameter sensitivity on HeartSteps (Table: sweep).

The two free parameters of the emission layer -- dispersion (sigma) and anchor
weight (w) -- are fixed a priori rather than tuned. To confirm the headline is not
an artefact of that choice, the neighbourhood of the default was run through the
full pipeline: one simulation run per (sigma, w) cell, same patients, same latent
predictions, decoder parameters overridden via validate_dataset's --decoder-sigma /
--decoder-anchor-weight. This script recomputes each cell's residual-feature C2ST
gap (0 = indistinguishable, lower is better), between-patient Spearman and
heterogeneity from the committed trajectory files. The final thesis reports
classifier accuracy (0.5 = indistinguishable), not the older probability-gap
diagnostic retained inside the C2ST result object.

The gap responds almost entirely to w, not sigma, and the default is deliberately
not the gap-minimising corner. CPU only, no LLM calls.
"""
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

CELLS = [
    ("heartsteps_sweep_s07_w035", 0.7, 0.35),
    ("heartsteps_sweep_s11_w035", 1.1, 0.35),
    ("heartsteps_emission_two_part", 0.9, 0.50),  # the fixed default
    ("heartsteps_sweep_s07_w065", 0.7, 0.65),
    ("heartsteps_sweep_s11_w065", 1.1, 0.65),
]


def main():
    print("emission-parameter sensitivity (one committed run per cell)\n")
    print("%-11s %-14s %-17s %-11s %s" %
          ("dispersion", "anchor weight", "C2ST accuracy", "Spearman", "heterogeneity"))
    for stem, sigma, weight in CELLS:
        with open(REALISM_DIR / (stem + ".json")) as f:
            patients = json.load(f)["patients"]
        records = [extract_patient_series(p) for p in patients]
        records = [r for r in records if r[0]]
        c2st = c2st_cv(*residual_features(records))
        real_means, sim_means, _ = _patient_descriptives(records)
        spearman = _spearman(real_means, sim_means)
        heterogeneity = _std(sim_means) / _std(real_means)
        tag = "  (default)" if (sigma, weight) == (0.9, 0.50) else ""
        print("%-11.1f %-14.2f %.3f +/- %-7.3f %-11.3f %.2f%s"
              % (sigma, weight, c2st["pooled"], c2st["sd"], spearman, heterogeneity, tag))


if __name__ == "__main__":
    main()
