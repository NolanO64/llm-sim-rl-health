"""Shared synthetic persona distribution; no API or reference environment imports."""
from dataclasses import dataclass


BASELINES = ("very sedentary", "low-activity", "moderately active", "quite active", "very active")
REACTIVITIES = ("barely reacts to app prompts", "somewhat reacts to prompts",
                "reacts to prompts", "reacts strongly to prompts")
HABITUATIONS = ("tunes out repeated messages very slowly", "tunes out slowly",
                "average tolerance", "tunes out quickly")


@dataclass(frozen=True)
class PatientProfile:
    age: int
    baseline: int
    reactivity: int
    habituation: int

    @property
    def description(self):
        return "age %d; baseline %s; %s; %s" % (
            self.age, BASELINES[self.baseline], REACTIVITIES[self.reactivity],
            HABITUATIONS[self.habituation],
        )


def sample_patient(rng):
    # Preserve the original LLM persona sampler's draw order and distribution.
    return PatientProfile(
        int(rng.integers(20, 80)), int(rng.integers(len(BASELINES))),
        int(rng.integers(len(REACTIVITIES))), int(rng.integers(len(HABITUATIONS))),
    )
