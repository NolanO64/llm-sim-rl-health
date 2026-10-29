"""Action-conditioned statistical world with declared qualitative behavioral priors.

This module cannot query an LLM or StepCountJITAI. Numeric assumptions are explicit
in StatisticalConfig and are not fitted to reference returns or LLM trajectories.
"""
from dataclasses import asdict, dataclass, replace
import math
import random

import numpy as np

from simulator.adherence_simulator.outcome_models import (
    OutcomeModelState, TwoPartLognormalOutcomeModel,
)
from .synthetic_patient import sample_patient


ACTIONS = ("none", "generic", "match_A", "match_B")


@dataclass(frozen=True)
class StatisticalConfig:
    version: str = "action_conditioned_prior_v1"
    episode_length: int = 40
    baseline_levels: tuple = (0.04, 0.08, 0.13, 0.20, 0.30)
    response_levels: tuple = (0.02, 0.06, 0.12, 0.24)
    fatigue_increments: tuple = (0.04, 0.08, 0.16, 0.24)
    response_multiplier: float = 1.0
    generic_fraction: float = 0.5
    fatigue_recovery: float = 0.7
    fatigue_baseline_penalty: float = 0.5
    burden_memory: float = 0.8
    burden_increment: float = 0.15
    quit_intercept: float = 0.001
    quit_burden_scale: float = 0.05
    context_switch_probability: float = 0.5
    emission_sigma: float = 0.90
    emission_anchor_weight: float = 0.50

    def __post_init__(self):
        if self.episode_length < 1 or self.response_multiplier < 0:
            raise ValueError("episode length must be positive and response multiplier nonnegative")
        if len(self.baseline_levels) != 5 or len(self.response_levels) != 4 or len(self.fatigue_increments) != 4:
            raise ValueError("trait arrays must match the shared persona categories")
        values = (*self.baseline_levels, *self.response_levels, *self.fatigue_increments,
                  self.generic_fraction, self.fatigue_recovery, self.fatigue_baseline_penalty,
                  self.burden_memory, self.burden_increment, self.quit_intercept,
                  self.quit_burden_scale, self.context_switch_probability, self.emission_anchor_weight)
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
            raise ValueError("probabilities, trait levels and weights must be finite and in [0,1]")
        if not math.isfinite(self.response_multiplier) or not math.isfinite(self.emission_sigma) or self.emission_sigma < 0:
            raise ValueError("response multiplier and emission dispersion must be finite")

    def to_dict(self):
        return asdict(self)


DEFAULT_CONFIG = StatisticalConfig()
PRESETS = {
    "weak": replace(DEFAULT_CONFIG, response_multiplier=0.5),
    "default": DEFAULT_CONFIG,
    "strong": replace(DEFAULT_CONFIG, response_multiplier=2.0),
}


class StatisticalWorld:
    def __init__(self, seed, config=DEFAULT_CONFIG):
        self.config = config
        self.action_rng = np.random.default_rng(seed)
        self.person = sample_patient(self.action_rng)
        self.context = int(self.action_rng.random() >= 0.5)
        self.dynamics_rng = np.random.default_rng(seed + 17)
        self.emit_rng = random.Random(seed + 7)
        self.emitter = TwoPartLognormalOutcomeModel(
            sigma=config.emission_sigma, anchor_weight=config.emission_anchor_weight,
        )
        self.history = []
        self.streak = 0
        self.fatigue = 0.0
        self.burden = 0.0
        self.done = False

    def observation(self):
        return {"c_infer": self.context}

    def tendency(self, action):
        if action not in range(4):
            raise ValueError("action must be 0, 1, 2 or 3")
        baseline = self.config.baseline_levels[self.person.baseline]
        response = self.config.response_levels[self.person.reactivity]
        relevance = (self.config.generic_fraction if action == 1
                     else float(action == 2 + self.context) if action >= 2 else 0.0)
        latent = (baseline * (1 - self.config.fatigue_baseline_penalty * self.fatigue)
                  + self.config.response_multiplier * response * relevance * math.exp(-self.fatigue))
        return float(np.clip(latent, 0.0, 1.0))

    def step(self, action):
        if self.done:
            raise RuntimeError("episode is finished")
        latent = self.tendency(action)
        previous = [r["outcome"] for r in self.history]
        result = self.emitter.decode(
            latent=latent,
            state=OutcomeModelState(
                dataset_id="statisticalworld", patient_id="synthetic",
                anchor_values=tuple(previous), population_anchor_values=tuple(previous),
            ),
            action=ACTIONS[action], context={}, simulated_history=previous,
            rng=self.emit_rng, deterministic=False, certainty=0.5,
        )
        if action == 0:
            self.fatigue *= self.config.fatigue_recovery
        else:
            self.fatigue = min(1.0, self.fatigue + self.config.fatigue_increments[self.person.habituation])
        mismatch = float(action >= 2 and action != 2 + self.context)
        self.burden = min(1.0, self.config.burden_memory * self.burden
                          + self.config.burden_increment * (self.fatigue + mismatch))
        quit_probability = min(1.0, self.config.quit_intercept + self.config.quit_burden_scale * self.burden)
        quit_ = bool(self.dynamics_rng.random() < quit_probability)
        next_context = self.context ^ int(self.dynamics_rng.random() < self.config.context_switch_probability)
        row = {
            "context": "A" if self.context == 0 else "B", "action": int(action),
            "latent": latent, "outcome": round(float(result.value), 3), "quit": quit_,
            "next_context": "A" if next_context == 0 else "B",
        }
        self.history.append(row)
        self.streak = 0 if action == 0 else self.streak + 1
        self.context = next_context
        self.done = quit_ or len(self.history) >= self.config.episode_length
        return float(result.value), self.done


def collect_episode(Q, eps, seed, config=DEFAULT_CONFIG):
    world = StatisticalWorld(seed, config)
    transitions = []
    while not world.done:
        c, b = world.context, min(world.streak, 3)
        if world.action_rng.random() < eps:
            action = int(world.action_rng.integers(4))
        else:
            action = int(Q[c, b].argmax())
        reward, done = world.step(action)
        next_c, next_b = (0, 0) if done else (world.context, min(world.streak, 3))
        transitions.append((c, b, action, reward, next_c, next_b, done))
    return transitions


def rollout_policy(policy, seed, config=DEFAULT_CONFIG):
    world = StatisticalWorld(seed, config)
    policy_rng = np.random.default_rng(seed + 13)
    total = 0.0
    while not world.done:
        action = int(policy(world.observation(), world.streak, policy_rng))
        reward, _ = world.step(action)
        total += reward
    return total
