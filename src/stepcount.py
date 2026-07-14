"""Wrapper around the StepCountJITAI environment (Karine & Marlin, 2024).

This module defines the two environment configurations contrasted in the thesis,
a constructor, the canonical evaluation protocol used to score a policy by its
mean real return, and a helper that measures the context-inference error rate.

StepCountJITAI is an external dependency and must be available on ``PYTHONPATH``.
"""
import numpy as np

from StepCountJITAI.envs import StepCountJITAI

# StepCountJITAI exposes the context-noise standard deviation under a Greek key.
_SIGMA_S_KEY = "σs"

# Observations requested from the environment: true context (C), probability (P),
# inferred-context likelihood (L), habituation (H) and disengagement (D).
OBSERVATIONS = ["C", "P", "L", "H", "D"]

MAX_EPISODE_LENGTH = 50

# The two configurations used throughout the thesis. They differ only in the
# amount of context-inference noise:
#   clean -- low noise, about 10% of steps have a misidentified context;
#   paper -- the harder setting of Karine & Marlin (2024), about 41% error.
CONFIGS = {
    "clean": {"sigma": 0.4, "sigma_s": 10.0, "chosen_a": 0.01},
    "paper": {"sigma": 2.0, "sigma_s": 20.0, "chosen_a": 0.3},
}


def make_env(seed, config="paper"):
    """Build a StepCountJITAI instance for one of the named configurations."""
    cfg = CONFIGS[config] if isinstance(config, str) else config
    return StepCountJITAI(
        sigma=cfg["sigma"],
        chosen_obs_names=OBSERVATIONS,
        n_version=1,
        b_using_uniform=True,
        chosen_a=cfg["chosen_a"],
        seed=seed,
        max_episode_length=MAX_EPISODE_LENGTH,
        b_display=False,
        **{_SIGMA_S_KEY: cfg["sigma_s"]},
    )


def message_bucket(streak):
    """Discretise the run of consecutive messages into the state feature {0,1,2,3}."""
    return min(streak, 3)


def evaluate_real(policy, config="paper", seeds=range(8), n_episodes=80, base_seed=7000):
    """Mean undiscounted return of a policy in the real environment.

    A policy is a callable ``policy(obs, streak, rng) -> action`` where ``obs`` is
    the unpacked observation dict and ``streak`` is the number of consecutive
    messages sent so far. Every policy is evaluated against the same seed schedule,
    so the returns are paired and differences are not driven by environment noise.
    """
    returns = []
    for seed in seeds:
        env = make_env(seed, config)
        rng = np.random.default_rng(base_seed + seed)
        for _ in range(n_episodes):
            env.reset()
            streak = 0
            total = 0.0
            for _ in range(MAX_EPISODE_LENGTH):
                obs = env.unpack_obs_array(env.current_state)
                action = int(policy(obs, streak, rng))
                streak = 0 if action == 0 else streak + 1
                _, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                if terminated or truncated:
                    break
            returns.append(total)
    return float(np.mean(returns))


def inference_error(config="paper", n_steps=4000, seed=123):
    """Percentage of steps on which the inferred context differs from the true one."""
    env = make_env(seed, config)
    rng = np.random.default_rng(0)
    env.reset()
    misses = 0
    for _ in range(n_steps):
        obs = env.unpack_obs_array(env.current_state)
        if int(obs["c_true"]) != int(obs["c_infer"]):
            misses += 1
        _, _, terminated, truncated, _ = env.step(int(rng.integers(4)))
        if terminated or truncated:
            env.reset()
    return 100.0 * misses / n_steps
