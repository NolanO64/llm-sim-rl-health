"""Reanalyse saved trajectory features with/without mean alignment; no API calls."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paths import REPO_ROOT, RESULTS_DIR
from src.realism import action_vocab, c2st_cv, extract_patient_series, residual_features, trajectory_features


def classifier_summary(real, simulated):
    result = c2st_cv(real, simulated)
    if result is None:
        raise ValueError("Insufficient patient pairs")
    # Fold predictions within paired patients are not independent Bernoulli trials.
    return {key: value for key, value in result.items() if key != "p_binom"}


def analyze():
    rows, hashes = [], {}
    for name in ("heartsteps_baseline_K35.json", "heartsteps_emission_two_part.json",
                 "heartsteps_emission_stateful_hint.json"):
        path = REPO_ROOT / "data/realism" / name
        raw = path.read_bytes()
        hashes[path.relative_to(REPO_ROOT).as_posix()] = hashlib.sha256(raw).hexdigest()
        records = [extract_patient_series(p) for p in json.loads(raw)["patients"]]
        vocab = action_vocab(records)
        shifted_real, shifted_sim = residual_features(records)
        raw_real, raw_sim, means_real, means_sim = [], [], [], []
        for real, sim, _, actions in records:
            r = trajectory_features(real, actions, vocab, include_mean=False)
            s = trajectory_features(sim, actions, vocab, include_mean=False)
            if r is not None and s is not None:
                raw_real.append(r)
                raw_sim.append(s)
                means_real.append(trajectory_features(real, actions, vocab, include_mean=True))
                means_sim.append(trajectory_features(sim, actions, vocab, include_mean=True))
        rows.append({"artifact": name, "n_patient_pairs": len(raw_real),
                     "mean_aligned": classifier_summary(shifted_real, shifted_sim),
                     "unaligned_same_features": classifier_summary(raw_real, raw_sim),
                     "unaligned_including_mean": classifier_summary(means_real, means_sim)})
    return {"scope": "Reanalysis of existing trajectories; no new simulator calls or policy training.",
            "interpretation": "Lower classifier accuracy means harder to distinguish, not full realism.",
            "folds": "patient-paired/disjoint as in the original classifier",
            "p_value_note": "Naive binomial p-values omitted; paired observations are not independent trials.",
            "source_sha256": hashes, "variants": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "trajectory_alignment_audit.json")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output exists; use a new --out path")
    result = analyze()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for row in result["variants"]:
        print(row["artifact"], [row[k]["pooled"] for k in
                                ("mean_aligned", "unaligned_same_features", "unaligned_including_mean")])


if __name__ == "__main__":
    main()
