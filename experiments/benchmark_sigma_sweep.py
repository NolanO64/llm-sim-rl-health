"""Inference-noise gradient for benchmark validity (Figure 8).

The default path prints the saved sweep. ``--recompute`` re-evaluates policies
in StepCountJITAI and writes a new result file. No language-model calls are made
in either mode.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.artifacts import load_result
from src.paths import RESULTS_DIR


def print_saved(data):
    fixed = data["fixed"]
    print(f"{data['n']} candidate policies | sigma_s={fixed['sigma_s']}, a={fixed['chosen_a']} fixed\n")
    print("sigma  inf-err%  rho [95% CI]     best->LLM rank  top     middle  bottom")
    for row in data["sweep"]:
        print(
            f"{row['sigma']:<6.1f} {row['infer_err_pct']:<9.1f} "
            f"{row['rho']:.3f} [{row['ci'][0]:.2f},{row['ci'][1]:.2f}]     "
            f"{row['best_llm_rank']:>2}/{data['n']:<12} "
            f"{row['top_third']:+.3f}  {row['mid_third']:+.3f}  {row['bot_third']:+.3f}"
        )


def recompute():
    import numpy as np

    from src.artifacts import load_corpora, load_online_qtables
    from src.metrics import bootstrap_spearman, ranks, spearman, tertile_spearman
    from src.policies import build_policy_zoo
    from src.stepcount import CONFIGS, evaluate_real, inference_error

    sigmas = [0.4, 0.8, 1.2, 1.6, 2.0]
    fixed = {"sigma_s": CONFIGS["paper"]["sigma_s"], "chosen_a": CONFIGS["paper"]["chosen_a"]}
    zoo = build_policy_zoo(load_corpora(), load_online_qtables(), ceiling_policy=None)
    saved = {row["policy"]: row for row in load_result("benchmark_robust.json")["rows"]}
    labels = [name for name in zoo if name in saved]
    llm = [saved[name]["llm_mean"] for name in labels]
    batches = [saved[name]["batch_means"] for name in labels]
    sweep = []
    for sigma in sigmas:
        config = {"sigma": sigma, **fixed}
        real = [round(evaluate_real(zoo[name], config)) for name in labels]
        rho = spearman(llm, real)
        _, lo, hi = bootstrap_spearman(llm, real)
        top, middle, bottom = tertile_spearman(real, llm)
        best_rank = ranks(llm)[int(np.argmax(real))]
        per_batch = [spearman([values[b] for values in batches], real)
                     for b in range(len(batches[0]))]
        sweep.append({
            "sigma": sigma,
            "infer_err_pct": round(inference_error(config), 1),
            "rho": round(rho, 3),
            "ci": [round(lo, 2), round(hi, 2)],
            "best_llm_rank": best_rank,
            "top_third": round(top, 3),
            "mid_third": round(middle, 3),
            "bot_third": round(bottom, 3),
            "per_batch_rho": [round(value, 3) for value in per_batch],
        })
    result = {"fixed": fixed, "n": len(labels), "sweep": sweep}
    output = RESULTS_DIR / "benchmark_sigma_sweep_new.json"
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print_saved(result)
    print("\nwrote", output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    recompute() if args.recompute else print_saved(load_result("benchmark_sigma_sweep.json"))


if __name__ == "__main__":
    main()
