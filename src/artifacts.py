"""Loaders for the saved artifacts (LLM corpora, trained Q-tables, result files).

These let every analysis reproduce its numbers from data committed to the
repository, without re-querying the language model.
"""
import json

import numpy as np

from .paths import CORPORA_DIR, ONLINE_Q_DIR, RESULTS_DIR


def load_corpora(ids=(0, 1, 2)):
    """The LLM-world training corpora used by the offline learners."""
    corpora = {}
    for corpus_id in ids:
        path = CORPORA_DIR / ("llmworld_corpus_c%d.json" % corpus_id)
        with open(path) as f:
            corpora[corpus_id] = json.load(f)
    return corpora


def load_online_qtables():
    """The Q-tables of the online learners trained inside the LLM world.

    Returns a dict {seed: Q-table}. These policies never saw the real environment
    during training (that is the point of the transfer study); the tables were
    saved by experiments/train_online_llm.py and are reused so the benchmark does
    not have to repeat the model calls.
    """
    with open(ONLINE_Q_DIR / "online_qtables.json") as f:
        data = json.load(f)
    return {entry["seed"]: np.array(entry["final_Q"]) for entry in data["per_seed"]}


def load_result(name):
    """Load a saved result JSON from data/results by file name."""
    with open(RESULTS_DIR / name) as f:
        return json.load(f)
