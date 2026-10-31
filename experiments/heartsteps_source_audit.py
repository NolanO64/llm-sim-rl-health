"""Local aggregate audit of public HeartSteps CSVs and saved compact trajectories.

No API client, prompt generation, participant-level export or model fitting.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np

POST_SOURCES = ("gfsteps30", "jbsteps30", "gfsteps40", "jbsteps40", "gfsteps60", "jbsteps60")
EXPECTED_SOURCE_SHA256 = {
    "suggestions.csv": "60ee7896183d7084a084f9a6e8ef2d0afea2e7fecdb41d6b15382ce1745c5ddb",
    "users.csv": "b07d93573b014e0ae1da822c616e9e811551ad77dc8390fb52a03202f7df0ed1",
    "LICENSE": "f5b745ef98087f531e719ee8ca6a96809444573ecc7173c6fa68eaad39b3cc3f",
}


def number(value):
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def truth(value):
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def outcome(row, sources=POST_SOURCES):
    for source in sources:
        v = number(row.get(source))
        if v is not None:
            return v, source
    return None, None


def action(row):
    if not truth(row.get("send")):
        return 0
    if truth(row.get("send.active")):
        return 1
    if truth(row.get("send.sedentary")):
        return 2
    return 1


def date(row):
    for key in ("sugg.decision.utime", "sugg.select.utime"):
        try:
            return datetime.fromisoformat(row.get(key, ""))
        except ValueError:
            pass
    return None


def clip(v):
    return min(1., max(0., v / 1500.))


def counts(rows):
    return {"rows": len(rows), "actions": dict(sorted(Counter(action(r) for r in rows).items())),
            "outcome_sources": dict(sorted(Counter(outcome(r)[1] for r in rows).items())),
            "randomization_draw_flags": dict(Counter(r.get("is.randomized", "") for r in rows)),
            "strict_30m_missing": sum(outcome(r, POST_SOURCES[:2])[0] is None for r in rows)}


def contrasts(patients, mode, n_boot=2000):
    """Observation-weighted arm differences with patient-cluster intervals."""
    sums, ns = np.zeros((len(patients), 3)), np.zeros((len(patients), 3))
    for i, rows in enumerate(patients):
        for row in rows:
            v, _ = outcome(row, POST_SOURCES[:2] if mode != "mixed_bounded" else POST_SOURCES)
            if v is None:
                continue
            a = action(row)
            sums[i, a] += v if mode == "strict_30m_raw_steps" else clip(v)
            ns[i, a] += 1
    rng = np.random.default_rng(20261025)
    idx = rng.integers(len(patients), size=(n_boot, len(patients)))
    boot_n = ns[idx].sum(axis=1)
    boot_sum = sums[idx].sum(axis=1)
    if (boot_n == 0).any() or (ns.sum(axis=0) == 0).any():
        raise ValueError("Missing treatment arm in estimate/bootstrap")
    means = sums.sum(axis=0) / ns.sum(axis=0)
    bmean = boot_sum / boot_n
    return {"mode": mode, "patients": len(patients), "arm_counts": ns.sum(axis=0).astype(int).tolist(),
            "arm_means": means.tolist(),
            "active_minus_none": {"point": float(means[1] - means[0]),
                "patient_cluster_ci95": np.quantile(bmean[:, 1] - bmean[:, 0], [.025, .975]).tolist()},
            "sedentary_minus_none": {"point": float(means[2] - means[0]),
                "patient_cluster_ci95": np.quantile(bmean[:, 2] - bmean[:, 0], [.025, .975]).tolist()},
            "interpretation": "Unadjusted retained-observation diagnostic, not a causal MRT estimator."}


def match_compact(patients, compact):
    if len(patients) != len(compact["patients"]):
        return {"exact_match": False, "reason": "patient count differs"}
    mismatched_patients, mismatched_rows, compared = 0, 0, 0
    for rows, p in zip(patients, compact["patients"]):
        saved = [s for s in p["steps"] if not s.get("gt_override") and s.get("gt_adherence") is not None]
        if len(rows) != len(saved):
            mismatched_patients += 1
            continue
        for row, step in zip(rows, saved):
            compared += 1
            if action(row) != int(step["action_flat"]) or not math.isclose(
                    clip(outcome(row)[0]), float(step["gt_adherence"]), abs_tol=1e-12, rel_tol=0):
                mismatched_rows += 1
    return {"exact_match": mismatched_patients == mismatched_rows == 0,
            "length_mismatched_patients": mismatched_patients, "value_or_action_mismatches": mismatched_rows,
            "compared_rows": compared, "matching_rule": "natural patient order, full action/outcome sequence"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source_hashes = {name: hashlib.sha256((args.dataset / name).read_bytes()).hexdigest()
                     for name in EXPECTED_SOURCE_SHA256}
    if source_hashes != EXPECTED_SOURCE_SHA256:
        raise ValueError("Dataset differs from the pinned public snapshot; verify provenance before auditing")
    with (args.dataset / "suggestions.csv").open(newline="", encoding="utf-8-sig") as f:
        raw = list(csv.DictReader(f))
    with (args.dataset / "users.csv").open(newline="", encoding="utf-8-sig") as f:
        profiles = list(csv.DictReader(f))
    compact = json.loads(args.compact.read_text(encoding="utf-8"))
    grouped = defaultdict(list)
    for row in raw:
        grouped[row["user.index"]].append(row)
    ids = sorted(grouped, key=lambda s: int(s))
    all_patients = [sorted(grouped[uid], key=lambda r: number(r.get("decision.index")) or 0) for uid in ids]
    available = [r for r in raw if truth(r.get("avail", "true"))]
    eligible_patients = [[r for r in p if truth(r.get("avail", "true")) and outcome(r)[0] is not None]
                         for p in all_patients]
    eligible = [r for p in eligible_patients for r in p]
    duplicate_keys = len(raw) - len({(r["user.index"], r["decision.index"]) for r in raw})
    result = {"scope": "Aggregate local source-data audit; no API calls, no causal or LLM fitting.",
              "source_url": "https://github.com/klasnja/HeartStepsV1",
              "source_commit": "3016391de426116bdef41880d72bc8cd4b9b2477",
              "source_sha256": source_hashes,
              "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "compact_sha256": hashlib.sha256(args.compact.read_bytes()).hexdigest(),
              "raw_rows": len(raw), "patients": len(ids), "profile_rows": len(profiles),
              "duplicate_patient_decision_keys": duplicate_keys,
              "unavailable_excluded": len(raw) - len(available),
              "available_missing_post_outcome_excluded": len(available) - len(eligible),
              "available_by_action": dict(sorted(Counter(action(r) for r in available).items())),
              "available_missing_post_by_action": dict(sorted(Counter(action(r) for r in available
                                                                       if outcome(r)[0] is None).items())),
              "eligible_patient_length_range": [min(map(len, eligible_patients)), max(map(len, eligible_patients))],
              "excluded_first_35_per_patient_total": sum(min(35, len(p)) for p in eligible_patients),
              "eligible_before_cap_and_anchor": counts(eligible), "cap_checks": {},
              "source_documentation": "https://github.com/klasnja/HeartStepsV1/wiki/Documentation-for-suggestions.csv",
              "randomization_note": "is.randomized is the realized random send/no-send draw, not an eligibility flag. Do not filter to True when comparing against no message.",
              "documented_action_probabilities": {"none": .4, "active": .3, "sedentary": .3},
              "probability_note": "Probabilities come from source documentation, not explicit per-row propensity columns; verify trial-analysis assumptions before causal fitting."}
    for cap in (210, 250, None):
        post = [(p if cap is None else p[:cap])[35:] for p in eligible_patients]
        flat = [r for p in post for r in p]
        key = "uncapped" if cap is None else str(cap)
        result["cap_checks"][key] = {"retained": counts(flat), "compact_reproduction": match_compact(post, compact),
            "mixed_bounded_contrast": contrasts(post, "mixed_bounded"),
            "strict_30m_bounded_contrast": contrasts(post, "strict_30m_bounded"),
            "strict_30m_raw_contrast": contrasts(post, "strict_30m_raw_steps")}
    result["full_available_sample_contrasts"] = [contrasts(eligible_patients, mode) for mode in
                                               ("mixed_bounded", "strict_30m_bounded", "strict_30m_raw_steps")]
    lag = []
    missing_current, missing_next, next_unavailable = 0, 0, 0
    for patient in all_patients:
        for current, later in zip(patient, patient[1:]):
            if not truth(current.get("avail")):
                continue
            now, then = date(current), date(later)
            if now is None or then is None:
                missing_current += 1
                continue
            lag.append((then - now).total_seconds() / 60)
            missing_next += outcome(later, POST_SOURCES[:2])[0] is None
            next_unavailable += not truth(later.get("avail"))
    result["next_raw_decision_feasibility"] = {"pairs_with_valid_times": len(lag),
        "pairs_missing_times": missing_current, "next_30m_outcome_missing": missing_next,
        "next_decision_unavailable": next_unavailable,
        "elapsed_minutes_quantiles": np.quantile(lag, [0, .25, .5, .75, 1]).tolist(),
        "negative_elapsed_intervals": sum(v < 0 for v in lag),
        "zero_elapsed_intervals": sum(v == 0 for v in lag),
        "note": "Pairs follow original decision index, not the next retained available row. Non-monotonic naive timestamps require time-zone/source reconciliation before defining lag windows. Do not condition naively on future availability."}
    result["subgroup_feasibility"] = {"profile_linkage": "Raw participant IDs available locally; no IDs exported here.",
        "age_nonmissing": sum(number(r.get("age")) is not None for r in profiles),
        "age_range": [min(number(r["age"]) for r in profiles if number(r.get("age")) is not None),
                      max(number(r["age"]) for r in profiles if number(r.get("age")) is not None)],
        "gender_category_counts": dict(Counter(r.get("gender", "") for r in profiles)),
        "validation_status": "Feasibility only, not independent subgroup or delayed-effect validation."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("raw_rows", "patients", "unavailable_excluded",
                                            "available_missing_post_outcome_excluded", "next_raw_decision_feasibility")}, indent=2))
    for cap, row in result["cap_checks"].items():
        print(cap, "n=", row["retained"]["rows"], "match=", row["compact_reproduction"],
              "sources=", row["retained"]["outcome_sources"], flush=True)


if __name__ == "__main__":
    main()
