"""Where does the LLM ranking agree or disagree with reality? (pattern breakdown)

A finer look at the saved benchmark than experiments/benchmark_validity.py: it
identifies which policy the LLM thinks is best, which families of policy it
systematically over- or under-rates, and how the agreement varies across the
quality range. This is the analysis behind the "coarse filter, not a selector"
reading in the discussion. Pure analysis of committed artifacts.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.artifacts import load_result
from src.metrics import ranks, spearman


def family(policy):
    if policy in ("none", "random", "random-msg"):
        return "baseline"
    if policy.startswith("always") or policy.startswith(("gen-mix", "mA-mix", "mB-mix")):
        return "fixed / fixed-mix"
    if policy.startswith("heuristic") or policy.startswith("heur-mix"):
        return "heuristic family"
    if policy.startswith("online-tabQ"):
        return "online RL"
    if policy.startswith("online-eps"):
        return "online RL (weakened)"
    if policy.startswith("tabQ"):
        return "offline RL"
    return "other"


def main():
    rows = [r for r in load_result("benchmark_robust.json")["rows"] if r["policy"] != "observable-ceiling"]
    real = [r["real_new"] for r in rows]
    llm = [r["llm_mean"] for r in rows]
    n = len(rows)
    real_rank = ranks(real)
    llm_rank = ranks(llm)

    best_real = int(np.argmax(real))
    best_llm = int(np.argmax(llm))
    print("=== n=%d candidate policies ===" % n)
    print("best by real : %-16s (real %d, LLM %.2f, LLM rank %d/%d)"
          % (rows[best_real]["policy"], real[best_real], llm[best_real], llm_rank[best_real], n))
    print("best by LLM  : %-16s (LLM %.2f, real %d, real rank %d/%d)"
          % (rows[best_llm]["policy"], llm[best_llm], real[best_llm], real_rank[best_llm], n))

    spread = [(rows[i]["policy"], real_rank[i], llm_rank[i], real_rank[i] - llm_rank[i],
               real[i], llm[i]) for i in range(n)]
    print("\n--- LLM over-rates (good LLM rank, worse real rank) ---")
    for policy, rr, lr, gap, re, lm in sorted(spread, key=lambda x: -x[3])[:6]:
        print("  %-16s real #%2d  LLM #%2d  (%+d)   real=%4d  LLM=%.2f" % (policy, rr, lr, gap, re, lm))
    print("--- LLM under-rates (worse LLM rank, good real rank) ---")
    for policy, rr, lr, gap, re, lm in sorted(spread, key=lambda x: x[3])[:6]:
        print("  %-16s real #%2d  LLM #%2d  (%+d)   real=%4d  LLM=%.2f" % (policy, rr, lr, gap, re, lm))

    families = {}
    for i in range(n):
        families.setdefault(family(rows[i]["policy"]), []).append(i)
    print("\n--- by family: mean real rank vs mean LLM rank (gap<0 => LLM over-rates) ---")
    for name, idx in sorted(families.items(), key=lambda kv: np.mean([real_rank[i] for i in kv[1]])):
        mean_real = np.mean([real_rank[i] for i in idx])
        mean_llm = np.mean([llm_rank[i] for i in idx])
        print("  %-22s n=%2d  real rank %4.1f  LLM rank %4.1f  gap %+4.1f"
              % (name, len(idx), mean_real, mean_llm, mean_llm - mean_real))

    order = sorted(range(n), key=lambda i: -real[i])
    print("\n--- resolution by tertile of real return ---")
    for name, sub in [("top third (best real)", order[: n // 3]),
                      ("middle third", order[n // 3 : 2 * n // 3]),
                      ("bottom third (worst real)", order[2 * n // 3 :])]:
        print("  within %-26s Spearman = %.2f" % (name, spearman([real[i] for i in sub], [llm[i] for i in sub])))


if __name__ == "__main__":
    main()
