"""Saved-score batch and policy-family sensitivity; no simulation or API calls."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics import spearman
from src.paths import REPO_ROOT, RESULTS_DIR


def family(name):
    if name in ("none", "random", "random-msg"):
        return "baseline"
    if name.startswith(("always", "gen-mix", "mA-mix", "mB-mix")):
        return "fixed_message"
    if name.startswith(("heuristic", "heur-mix")):
        return "heuristic"
    if name.startswith("online-tabQ"):
        return "online_rl"
    if name.startswith("online-eps"):
        return "online_rl_blends"
    if name.startswith("tabQ"):
        return "offline_rl"
    raise ValueError("Unrecognized policy family")


def ranking(rows, scores, indices=None):
    indices = np.arange(len(rows)) if indices is None else np.asarray(indices)
    if not len(indices):
        raise ValueError("Empty candidate set")
    reference = np.asarray([r["real_new"] for r in rows], dtype=float)
    real, score = reference[indices], np.asarray(scores)[indices]
    order_r, order_s = np.argsort(-real, kind="stable"), np.argsort(-score, kind="stable")
    selected = int(indices[order_s[0]])
    loss = float(real[order_r[0]] - reference[selected])
    k = min(5, len(indices))
    return {"n": len(indices), "rho": spearman(real, score),
            "selected": rows[selected]["policy"], "selected_reference_return": float(reference[selected]),
            "best_reference_return": float(real.max()), "regret": loss,
            "relative_regret_percent": 100 * loss / float(real.max()),
            "top_k": k, "top_k_overlap_count": len(set(order_r[:k]) & set(order_s[:k])),
            "reference_best_simulator_rank": int(np.where(order_s == order_r[0])[0][0]) + 1}


def analyze():
    benchmark = REPO_ROOT / "data/results/benchmark_robust.json"
    noise = REPO_ROOT / "data/results/benchmark_sigma_sweep.json"
    rows = [r for r in json.loads(benchmark.read_bytes())["rows"] if r["policy"] != "observable-ceiling"]
    scores = np.asarray([r["llm_mean"] for r in rows])
    batches = np.asarray([r["batch_means"] for r in rows])
    if len(rows) != 49 or batches.shape != (49, 3):
        raise ValueError("Expected the original 49 candidates and three scoring batches")
    groups = {name: [i for i,r in enumerate(rows) if family(r["policy"]) == name]
              for name in sorted({family(r["policy"]) for r in rows})}
    resamples = [ranking(rows, batches[:, sample].mean(axis=1))
                 for sample in itertools.product(range(3), repeat=3)]
    return {
        "scope": "Saved rounded reference returns and simulator batch means; no new rollouts or policy fitting.",
        "source_sha256": {p.relative_to(REPO_ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (benchmark, noise)},
        "pooled": ranking(rows, scores),
        "per_batch": [ranking(rows, batches[:, b]) for b in range(3)],
        "within_family": {name: ranking(rows, scores, idx) for name,idx in groups.items()},
        "leave_one_family_out": {name: ranking(rows, scores, [i for i in range(len(rows)) if i not in idx])
                                 for name,idx in groups.items()},
        "exploratory_batch_resampling": {
            "ordered_resamples": len(resamples),
            "regret_range": [min(r["regret"] for r in resamples), max(r["regret"] for r in resamples)],
            "top5_overlap_range": [min(r["top_k_overlap_count"] for r in resamples),
                                   max(r["top_k_overlap_count"] for r in resamples)],
            "limit": "Three batch means only; these ranges are not reliable 95% intervals."},
        "saved_context_noise_summary": [
            {k: row[k] for k in ("sigma", "rho", "best_llm_rank", "top_third")}
            for row in json.loads(noise.read_bytes())["sweep"]],
        "limits": "Related families are not independent algorithms. Reference values held fixed; no joint LLM/reference uncertainty."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "ranking_robustness_audit.json")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output exists; use a new --out")
    result = analyze()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("Per-batch loss:", [round(r["relative_regret_percent"], 1) for r in result["per_batch"]])
    print("Offline-family loss:", result["within_family"]["offline_rl"]["relative_regret_percent"])


if __name__ == "__main__":
    main()
