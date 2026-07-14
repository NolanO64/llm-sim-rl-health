"""Rank-correlation helpers for comparing LLM-world scores against real returns."""
import random

import numpy as np


def spearman(x, y):
    """Spearman rank correlation, computed as the Pearson correlation of ranks."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rank_x = np.argsort(np.argsort(x))
    rank_y = np.argsort(np.argsort(y))
    return float(np.corrcoef(rank_x, rank_y)[0, 1])


def bootstrap_spearman(x, y, n_boot=2000, seed=0):
    """Bootstrap the Spearman correlation by resampling the paired observations.

    Returns ``(mean, lo, hi)`` where ``lo``/``hi`` are the 2.5th and 97.5th
    percentiles of the bootstrap distribution.
    """
    rng = random.Random(seed)
    n = len(x)
    values = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        values.append(spearman([x[i] for i in idx], [y[i] for i in idx]))
    values.sort()
    mean = sum(values) / len(values)
    lo = values[int(0.025 * n_boot)]
    hi = values[int(0.975 * n_boot)]
    return mean, lo, hi


def ranks(values, descending=True):
    """1-based ranks of ``values`` (rank 1 is the best when ``descending``)."""
    order = sorted(range(len(values)), key=lambda i: -values[i] if descending else values[i])
    out = [0] * len(values)
    for position, i in enumerate(order):
        out[i] = position + 1
    return out


def tertile_spearman(real, llm):
    """Spearman within the top, middle and bottom third by real return.

    This exposes *where* in the quality range the LLM ranking agrees with reality.
    """
    n = len(real)
    order = sorted(range(n), key=lambda i: -real[i])
    thirds = (order[: n // 3], order[n // 3 : 2 * n // 3], order[2 * n // 3 :])
    return tuple(
        spearman([real[i] for i in sub], [llm[i] for i in sub]) for sub in thirds
    )
