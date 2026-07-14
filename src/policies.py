"""The library of deployable policies that the benchmark ranks.

Every policy is a callable ``policy(obs, streak, rng) -> action`` with the action
in {0: none, 1: generic, 2: match_A, 3: match_B}. Stochastic policies draw from
the ``rng`` supplied by the evaluator, so that all policies are exposed to the
same randomness when scored on a shared set of seeds.

The set spans a deliberately wide range of quality: do-nothing and random
baselines, hand-coded heuristics and their stochastic mixtures, and the offline
and online learners trained in the LLM world.
"""
from collections import OrderedDict

from .tabular_q import (
    corpus_to_transitions,
    epsilon_blend_policy,
    greedy_policy,
    train_offline_q,
)


# --- elementary policies -------------------------------------------------

def no_action(obs, streak, rng):
    return 0


def random_action(obs, streak, rng):
    return int(rng.integers(4))


def random_message(obs, streak, rng):
    """Never sends a generic message; picks among {none, match_A, match_B}."""
    return int(rng.choice([0, 2, 3]))


def matched_heuristic(obs, streak, rng):
    """Always send the message matched to the inferred context."""
    return 2 + int(obs["c_infer"])


def always(action):
    def policy(obs, streak, rng):
        return action

    return policy


def heuristic_mix(p):
    """Send the matched message with probability ``p``, otherwise nothing."""
    def policy(obs, streak, rng):
        return (2 + int(obs["c_infer"])) if rng.random() < p else 0

    return policy


def action_mix(action, p):
    """Take ``action`` with probability ``p``, otherwise nothing."""
    def policy(obs, streak, rng):
        return action if rng.random() < p else 0

    return policy


# --- the full benchmark set ---------------------------------------------

def build_policy_zoo(corpora, online_qs, ceiling_policy=None):
    """Assemble every benchmarked policy in the canonical order.

    ``corpora``        -- dict {corpus_id: corpus} of LLM-world training corpora.
    ``online_qs``      -- dict {seed: Q-table} of the online LLM-world learners.
    ``ceiling_policy`` -- the trained observable-ceiling policy, or ``None`` to omit it.
    """
    zoo = OrderedDict()
    zoo["none"] = no_action
    zoo["random"] = random_action
    zoo["random-msg"] = random_message
    zoo["heuristic"] = matched_heuristic
    zoo["always-generic"] = always(1)
    zoo["always-matchA"] = always(2)
    zoo["always-matchB"] = always(3)

    for p in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
        zoo["heur-mix%.2f" % p] = heuristic_mix(p)

    for action, name in [(1, "gen"), (2, "mA"), (3, "mB")]:
        for p in (0.3, 0.6):
            zoo["%s-mix%.1f" % (name, p)] = action_mix(action, p)

    for corpus_id in (0, 1, 2):
        transitions = corpus_to_transitions(corpora[corpus_id])
        for seed in (0, 1, 2, 3):
            zoo["tabQ c%d s%d" % (corpus_id, seed)] = greedy_policy(
                train_offline_q(transitions, seed)
            )

    for seed in (0, 1, 2):
        zoo["online-tabQ s%d" % seed] = greedy_policy(online_qs[seed])

    for seed in (0, 1, 2):
        for eps in (0.2, 0.4, 0.6, 0.8):
            zoo["online-eps%.1f s%d" % (eps, seed)] = epsilon_blend_policy(online_qs[seed], eps)

    if ceiling_policy is not None:
        zoo["observable-ceiling"] = ceiling_policy

    return zoo
