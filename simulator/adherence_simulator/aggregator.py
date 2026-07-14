"""Aggregation strategies for repeated LLM samples."""

import statistics
from collections import Counter
from typing import Any


def aggregate_adherence(
    responses: list[dict[str, Any]],
    method: str = "mean",
) -> tuple[float, float]:
    values = [response["adherence"] for response in responses]
    if len(values) == 1:
        return values[0], 0.0

    uncertainty = statistics.stdev(values)
    if method == "mean":
        return statistics.mean(values), uncertainty
    if method == "median":
        return statistics.median(values), uncertainty
    if method == "majority_vote":
        votes = [1 if value >= 0.5 else 0 for value in values]
        return sum(votes) / len(votes), uncertainty
    raise ValueError(f"Unknown aggregation method: {method}")


def aggregate_certainty(responses: list[dict[str, Any]]) -> float:
    return statistics.mean(response.get("adherence_certainty", 0.5) for response in responses)


def aggregate_reason_codes(responses: list[dict[str, Any]]) -> str:
    codes = [response.get("reason_code", "no_change") for response in responses]
    return Counter(codes).most_common(1)[0][0]
