"""Generic outcome-emission models for simulator rollouts.

These models sit between a latent LLM prediction and the realized environment
outcome. They are intentionally dataset-agnostic: datasets provide the action,
context, history, and warm-start observations; the outcome model decides how to
turn the latent tendency into a simulated observation.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class OutcomeModelResult:
    """Realized outcome plus diagnostics for one simulator step."""

    value: float
    expected_value: float
    p_positive: float | None = None
    positive_mean: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OutcomeModelState:
    """State available to a dataset-agnostic outcome model."""

    dataset_id: str
    patient_id: str
    anchor_values: tuple[float, ...] = ()
    population_anchor_values: tuple[float, ...] = ()


class OutcomeModel:
    """Base contract for online environment outcome models."""

    name = "none"

    def decode(
        self,
        *,
        latent: float,
        state: OutcomeModelState,
        action: str,
        context: dict[str, Any],
        simulated_history: list[float],
        rng: random.Random,
        deterministic: bool,
        certainty: float = 0.5,
    ) -> OutcomeModelResult:
        value = _clip(latent)
        return OutcomeModelResult(value=value, expected_value=value)


class TwoPartLognormalOutcomeModel(OutcomeModel):
    """Semi-continuous emission model for zero-heavy bounded outcomes.

    The LLM prediction is treated as a latent expected tendency in [0, 1].
    The realized observation is generated with:

    * a Bernoulli gate for any positive response;
    * a log-normal positive magnitude model when the gate is open.

    Warm-start anchor observations provide only a patient-level baseline; no
    post-anchor ground truth is read by this model.
    """

    name = "two_part_lognormal"

    def __init__(
        self,
        *,
        sigma: float = 0.90,
        anchor_weight: float = 0.50,
    ) -> None:
        self.sigma = float(sigma)
        self.anchor_weight = float(anchor_weight)

    def decode(
        self,
        *,
        latent: float,
        state: OutcomeModelState,
        action: str,
        context: dict[str, Any],
        simulated_history: list[float],
        rng: random.Random,
        deterministic: bool,
        certainty: float = 0.5,
    ) -> OutcomeModelResult:
        del action, context, certainty
        latent = _clip(latent)
        eff_sigma = self.sigma
        anchors = [value for value in state.anchor_values if 0.0 <= value <= 1.0]
        population = [
            value for value in state.population_anchor_values if 0.0 <= value <= 1.0
        ]
        baseline_values = anchors or population
        baseline_mean = _mean(baseline_values)
        baseline_positive_rate = _positive_rate(baseline_values)

        if baseline_mean is None:
            baseline_mean = latent
        if baseline_positive_rate is None:
            baseline_positive_rate = _clip(latent, 0.05, 0.95)

        latent_positive_rate = _clip(latent, 0.02, 0.98)
        p_positive = _sigmoid(
            (1.0 - self.anchor_weight) * _logit(latent_positive_rate)
            + self.anchor_weight * _logit(_clip(baseline_positive_rate, 0.02, 0.98))
        )

        if simulated_history:
            last = simulated_history[-1]
            if last <= 0.0:
                p_positive *= 0.90
            elif last >= 0.25:
                p_positive = 1.0 - ((1.0 - p_positive) * 0.90)
            p_positive = _clip(p_positive, 0.01, 0.99)

        target_mean = _clip(
            (1.0 - self.anchor_weight) * latent
            + self.anchor_weight * baseline_mean,
            0.0,
            1.0,
        )
        positive_mean = _clip(target_mean / max(p_positive, 1e-6), 0.005, 1.0)
        expected_value = _clip(p_positive * positive_mean)

        if deterministic:
            value = expected_value
        elif rng.random() > p_positive:
            value = 0.0
        else:
            mu = math.log(max(positive_mean, 1e-6)) - 0.5 * (eff_sigma ** 2)
            value = _clip(rng.lognormvariate(mu, eff_sigma))

        return OutcomeModelResult(
            value=value,
            expected_value=expected_value,
            p_positive=p_positive,
            positive_mean=positive_mean,
            metadata={
                "baseline_mean": baseline_mean,
                "baseline_positive_rate": baseline_positive_rate,
                "sigma": eff_sigma,
                "anchor_weight": self.anchor_weight,
            },
        )


ONLINE_OUTCOME_MODELS = {
    "two_part_lognormal": lambda: TwoPartLognormalOutcomeModel(),
}


def is_online_outcome_model(name: str | None) -> bool:
    return str(name or "none") in ONLINE_OUTCOME_MODELS


def build_outcome_model(name: str | None, *, sigma=None, anchor_weight=None) -> OutcomeModel | None:
    n = str(name or "none")
    if n not in ONLINE_OUTCOME_MODELS:
        return None
    kw = {}
    if sigma is not None:
        kw["sigma"] = float(sigma)
    if anchor_weight is not None:
        kw["anchor_weight"] = float(anchor_weight)
    if n == "two_part_lognormal":
        return TwoPartLognormalOutcomeModel(**kw)
    return ONLINE_OUTCOME_MODELS[n]()


def _positive_rate(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(1 for value in values if value > 0.0) / len(values)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _logit(value: float) -> float:
    value = _clip(value, 1e-6, 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)
