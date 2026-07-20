"""Decision consequences of selecting policies by LLM-world score.

This supplements the rank-correlation benchmark with selector-facing metrics:

* simple regret of choosing the simulator's top-ranked policy;
* whether the true best policy appears in the simulator's top-k shortlist;
* top-k overlap between simulator and reference rankings.

It uses the saved benchmark artifact and makes no LLM calls.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.artifacts import load_result
from src.paths import RESULTS_DIR


def main():
    data = load_result("benchmark_robust.json")
    rows = [row for row in data["rows"] if row["policy"] != "observable-ceiling"]
    real = np.array([float(row["real_new"]) for row in rows])
    llm = np.array([float(row["llm_mean"]) for row in rows])
    policies = [row["policy"] for row in rows]

    order_real = np.argsort(-real)
    order_llm = np.argsort(-llm)
    best_real = int(order_real[0])
    best_llm = int(order_llm[0])

    topk = {}
    for k in (1, 3, 5, 10):
        top_real = set(map(int, order_real[:k]))
        top_llm = set(map(int, order_llm[:k]))
        best_in_llm_topk = float(max(real[list(top_llm)]))
        topk[str(k)] = {
            "overlap": len(top_real & top_llm),
            "contains_true_best": best_real in top_llm,
            "best_reference_return_in_llm_topk": round(best_in_llm_topk, 3),
            "topk_regret": round(float(real[best_real] - best_in_llm_topk), 3),
        }

    payload = {
        "n_policies": len(rows),
        "true_best": {
            "policy": policies[best_real],
            "reference_return": round(float(real[best_real]), 3),
            "llm_score": round(float(llm[best_real]), 3),
            "llm_rank": int(np.where(order_llm == best_real)[0][0] + 1),
        },
        "simulator_top": {
            "policy": policies[best_llm],
            "llm_score": round(float(llm[best_llm]), 3),
            "reference_return": round(float(real[best_llm]), 3),
            "reference_rank": int(np.where(order_real == best_llm)[0][0] + 1),
        },
        "simple_regret": round(float(real[best_real] - real[best_llm]), 3),
        "relative_regret_percent": round(
            float(100 * (real[best_real] - real[best_llm]) / real[best_real]), 1
        ),
        "topk": topk,
    }

    out = RESULTS_DIR / "policy_selection_regret.json"
    with out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
