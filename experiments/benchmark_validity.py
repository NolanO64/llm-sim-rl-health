"""Benchmark validity: does the LLM world rank policies like reality? (headline result)

Reads the saved benchmark -- 49 candidate policies, each scored by the LLM world
as several independent batches of patients and by its true mean real return -- and
reports:

  * the pooled Spearman correlation between LLM score and real return, with a
    bootstrap confidence interval over policies;
  * the per-batch Spearman (run-to-run stability of the induced ranking);
  * the resolution within the top / middle / bottom third by real return;
  * whether the LLM picks the best real policy, and the top-5 overlap.

This is pure analysis of committed artifacts; the LLM scores themselves are
produced by experiments/score_policies_llm.py. The saved benchmark uses the
paper configuration (sigma=2).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.artifacts import load_result
from src.metrics import bootstrap_spearman, ranks, spearman, tertile_spearman


def top_k(values, k):
    return set(sorted(range(len(values)), key=lambda i: -values[i])[:k])


def main():
    data = load_result("benchmark_robust.json")
    rows = [r for r in data["rows"] if r["policy"] != "observable-ceiling"]
    llm = [r["llm_mean"] for r in rows]
    real = [r["real_new"] for r in rows]
    n = len(rows)
    n_batches = len(rows[0]["batch_means"])

    rho = spearman(llm, real)
    _, lo, hi = bootstrap_spearman(llm, real)
    per_batch = [spearman([r["batch_means"][b] for r in rows], real) for b in range(n_batches)]
    top, mid, bot = tertile_spearman(real, llm)

    llm_rank = ranks(llm)
    best = int(np.argmax(real))

    print("benchmark validity | n=%d candidate policies, %d batches of patients" % (n, n_batches))
    print("Spearman (pooled)    = %.3f   [95%% CI %.2f, %.2f]" % (rho, lo, hi))
    print("Spearman (per batch) = %s  ->  %.3f +/- %.3f"
          % ([round(x, 2) for x in per_batch], float(np.mean(per_batch)), float(np.std(per_batch))))
    print("tertile resolution   : top %.2f | middle %.2f | bottom %.2f" % (top, mid, bot))
    print("best real policy     : %-16s -> LLM rank %d/%d"
          % (rows[best]["policy"], llm_rank[best], n))
    print("top-5 overlap (real vs LLM) : %d/5" % len(top_k(real, 5) & top_k(llm, 5)))


if __name__ == "__main__":
    main()
