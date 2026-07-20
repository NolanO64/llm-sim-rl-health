"""Anchor-only synthetic environment for the RQ3 control benchmark.

This module intentionally removes the LLM and all action semantics. Each
synthetic patient has a sampled baseline activity tendency. A short synthetic
warm-start anchor is generated from that baseline, and every later step uses the
anchor mean as the constant latent passed through the same two-part emission
layer used by the LLM world. Context is observed but reward does not depend on
context or action.

The point is diagnostic: if policies trained in this world transfer well to
StepCountJITAI, the headline result may be carried by the emission/baseline
machinery rather than by LLM decision-relevant structure.
"""
import random
import sys

import numpy as np

from .paths import REPO_ROOT

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from simulator.adherence_simulator.outcome_models import (  # noqa: E402
    OutcomeModelState,
    TwoPartLognormalOutcomeModel,
)


ACTIONS = ["none", "generic", "match_A", "match_B"]
EPISODE_LENGTH = 40
ANCHOR_LENGTH = 7


def make_emitter():
    return TwoPartLognormalOutcomeModel(sigma=0.90, anchor_weight=0.50)


def emit_outcome(emitter, latent, action_name, history, rng):
    state = OutcomeModelState(
        dataset_id="anchorworld",
        patient_id="synthetic",
        anchor_values=tuple(history),
        population_anchor_values=tuple(history),
    )
    result = emitter.decode(
        latent=latent,
        state=state,
        action=action_name,
        context={},
        simulated_history=history,
        rng=rng,
        deterministic=False,
        certainty=0.5,
    )
    return float(result.value)


def sample_baseline(rng):
    """Sample a bounded patient baseline tendency.

    The mixture gives patient heterogeneity without encoding any action effect.
    Values are on the same 0--1 scale as the LLM latent activity tendency.
    """
    level = rng.choice([0.04, 0.08, 0.13, 0.20, 0.30], p=[0.18, 0.25, 0.27, 0.20, 0.10])
    return float(np.clip(rng.normal(level, 0.025), 0.01, 0.45))


def make_anchor(baseline, seed, anchor_length=ANCHOR_LENGTH):
    """Generate a patient-local warm-start anchor from the baseline tendency."""
    emitter = make_emitter()
    rng = random.Random(seed)
    history = []
    for _ in range(anchor_length):
        outcome = emit_outcome(emitter, baseline, ACTIONS[0], history, rng)
        history.append(outcome)
    return history


def anchor_latent(anchor, fallback):
    """Use the warm-start mean as the constant latent for post-anchor steps."""
    if not anchor:
        return float(fallback)
    return float(np.clip(np.mean(anchor), 0.0, 1.0))


def generate_anchor_trajectory(actions, seed, episode_length=EPISODE_LENGTH):
    """Generate one anchor-only trajectory under a supplied action sequence."""
    rng = np.random.default_rng(seed)
    emit_rng = random.Random(seed + 17)
    emitter = make_emitter()
    baseline = sample_baseline(rng)
    history = make_anchor(baseline, seed + 101)
    latent = anchor_latent(history, baseline)
    context = "A" if rng.random() < 0.5 else "B"
    rows = []
    for t in range(min(len(actions), episode_length)):
        action = int(actions[t])
        if rng.random() < 0.20:
            context = "B" if context == "A" else "A"
        outcome = emit_outcome(emitter, latent, ACTIONS[action], history, emit_rng)
        history.append(outcome)
        rows.append({
            "context": context,
            "action": action,
            "latent": round(latent, 3),
            "outcome": round(outcome, 3),
            "quit": False,
        })
    return rows


def collect_anchor_episode(Q, eps, seed, episode_length=EPISODE_LENGTH):
    """Collect online Q-learning transitions in the anchor-only world."""
    rng = np.random.default_rng(seed)
    emit_rng = random.Random(seed + 17)
    emitter = make_emitter()
    baseline = sample_baseline(rng)
    history = make_anchor(baseline, seed + 101)
    latent = anchor_latent(history, baseline)
    context = 0 if rng.random() < 0.5 else 1
    streak = 0
    seq = []
    for _ in range(episode_length):
        bucket = min(streak, 3)
        if rng.random() < eps:
            action = int(rng.integers(4))
        else:
            action = int(Q[context, bucket].argmax())
        outcome = emit_outcome(emitter, latent, ACTIONS[action], history, emit_rng)
        history.append(outcome)
        seq.append((context, bucket, action, outcome))
        streak = 0 if action == 0 else streak + 1
        if rng.random() < 0.20:
            context = 1 - context

    transitions = []
    for i, (c, b, a, r) in enumerate(seq):
        if i + 1 < len(seq):
            transitions.append((c, b, a, r, seq[i + 1][0], seq[i + 1][1], False))
        else:
            transitions.append((c, b, a, r, 0, 0, True))
    return transitions
