"""Paired-person effect diagnostics on existing saved predictions; no API calls."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paths import REPO_ROOT, RESULTS_DIR


def contrast(patients, action, field, balanced=False):
    pairs = []
    active, none = [], []
    for rows in patients:
        a = [r[field] for r in rows if r[0] == action and r[field] is not None]
        b = [r[field] for r in rows if r[0] == "0" and r[field] is not None]
        active.extend(a)
        none.extend(b)
        if a and b:
            pairs.append(np.mean(a) - np.mean(b))
    if balanced:
        return float(np.mean(pairs)) if pairs else float("nan")
    return float(np.mean(active) - np.mean(none)) if active and none else float("nan")

def audit_heartsteps(data, n_boot=2000):
    patients = []
    anchors = 0
    missing = 0
    for patient in data["patients"]:
        rows = []
        for step in patient["steps"]:
            if step.get("gt_override"):
                anchors += 1
                continue
            if step.get("gt_adherence") is None:
                missing += 1
                continue
            action = str(step["action_flat"])
            if action not in {"0", "1", "2"}:
                raise ValueError("Unexpected HeartSteps action")
            real = float(step["gt_adherence"])
            sim = step.get("adherence")
            sim = float(sim) if sim is not None else None
            if not 0 <= real <= 1 or (sim is not None and not 0 <= sim <= 1):
                raise ValueError("Outcome outside bounded scale")
            rows.append((action, real, sim))
        if rows:
            patients.append(rows)
    if not patients:
        raise ValueError("No retained patients")
    flat = [r for p in patients for r in p]
    result = {"patients": len(patients), "post_anchor_rows": len(flat),
              "excluded_anchor_rows": anchors, "missing_real_post_anchor": missing,
              "missing_sim_post_anchor": sum(r[2] is None for r in flat),
              "actions": {a: sum(r[0] == a for r in flat) for a in ("0", "1", "2")},
              "real_boundary_counts": {str(v): sum(r[1] == v for r in flat) for v in (0, 1)},
              "contrasts": {}}
    rng = np.random.default_rng(20261025)
    samples = rng.integers(len(patients), size=(n_boot, len(patients)))
    for action in ("1", "2"):
        point_real = contrast(patients, action, 1)
        point_sim = contrast(patients, action, 2)
        errors = []
        for indices in samples:
            selected = [patients[i] for i in indices]
            errors.append(contrast(selected, action, 2) - contrast(selected, action, 1))
        if not np.isfinite(errors).all():
            raise ValueError("Bootstrap sample has no contrast")
        result["contrasts"][action] = {
            "real_pooled": point_real, "sim_pooled": point_sim,
            "sim_minus_real": point_sim - point_real,
            "paired_patient_bootstrap_bias_ci95": np.quantile(errors, [.025, .975]).tolist(),
            "real_patient_balanced": contrast(patients, action, 1, True),
            "sim_patient_balanced": contrast(patients, action, 2, True),
            "patients_with_both_arms": sum(any(r[0] == action for r in p) and
                                            any(r[0] == "0" for r in p) for p in patients),
        }
    result["bootstrap"] = {"resamples": n_boot, "seed": 20261025,
                           "unit": "whole patient, paired real and simulated observations"}
    return result


def analyze():
    rows, hashes = {}, {}
    for tag in ("two_part", "stateful_hint"):
        path = REPO_ROOT / f"data/realism/heartsteps_emission_{tag}.json"
        raw = path.read_bytes()
        hashes[path.relative_to(REPO_ROOT).as_posix()] = hashlib.sha256(raw).hexdigest()
        rows[tag] = audit_heartsteps(json.loads(raw))
    return {"scope": "Aggregate saved-prediction audit; no new LLM calls or participant-level output.",
            "heartsteps": rows, "source_sha256": hashes,
            "limit": "Selected post-anchor bounded-scale contrasts, not the full trial causal estimate."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "heartsteps_effect_audit.json")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output exists; use a new --out")
    result = analyze()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for tag,row in result["heartsteps"].items():
        print(tag, row["contrasts"]["1"]["sim_minus_real"],
              row["contrasts"]["1"]["paired_patient_bootstrap_bias_ci95"])


if __name__ == "__main__":
    main()
