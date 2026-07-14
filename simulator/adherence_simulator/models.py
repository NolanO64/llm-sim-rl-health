"""Typed response returned by an LLM backend."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class StepResponse:
    adherence: float
    adherence_certainty: float = 0.5
    reason_code: str = "no_change"
    state_update: dict[str, Any] = field(default_factory=dict)
    activity_bin: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "StepResponse":
        return cls(
            adherence=float(raw.get("adherence", 0.0)),
            adherence_certainty=float(raw.get("adherence_certainty", 0.5)),
            reason_code=str(raw.get("reason_code", "no_change")),
            state_update=dict(raw.get("state_update", {})),
            activity_bin=raw.get("activity_bin"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "adherence": self.adherence,
            "adherence_certainty": self.adherence_certainty,
            "reason_code": self.reason_code,
            "state_update": dict(self.state_update),
            "activity_bin": self.activity_bin,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)
