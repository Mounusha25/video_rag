"""Deterministic retrieval metrics for timestamped video search."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean
from typing import Any, Iterable, Sequence
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse


@dataclass(frozen=True)
class RelevantMoment:
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    query: str
    source_url: str
    relevant_moments: tuple[RelevantMoment, ...]
    query_type: str = "mixed"
    tolerance_seconds: float = 5.0


def normalize_source_url(url: str) -> str:
    """Remove timestamp/tracking fields while keeping the video identity."""
    parsed = urlparse(url.strip())
    host = parsed.netloc.lower().removeprefix("www.")
    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in {"t", "start", "time_continue", "utm_source", "utm_medium", "utm_campaign"}
    ]
    return urlunparse((parsed.scheme.lower(), host, parsed.path.rstrip("/"), "", urlencode(query), ""))


def result_matches_moment(
    case: BenchmarkCase,
    result: dict[str, Any],
    moment: RelevantMoment,
) -> bool:
    if normalize_source_url(str(result.get("source_url", ""))) != normalize_source_url(case.source_url):
        return False
    try:
        timestamp = float(result["timestamp_seconds"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        moment.start_seconds - case.tolerance_seconds
        <= timestamp
        <= moment.end_seconds + case.tolerance_seconds
    )


def score_case(
    case: BenchmarkCase,
    results: Sequence[dict[str, Any]],
    ks: Iterable[int] = (1, 5, 10),
) -> dict[str, Any]:
    """Score one ranked result list against all labeled relevant moments."""
    if not case.relevant_moments:
        raise ValueError(f"Case {case.case_id!r} has no relevant moments")

    first_relevant_rank: int | None = None
    matched_at_rank: list[int | None] = [None] * len(case.relevant_moments)
    for rank, result in enumerate(results, start=1):
        for index, moment in enumerate(case.relevant_moments):
            if matched_at_rank[index] is None and result_matches_moment(case, result, moment):
                matched_at_rank[index] = rank
                if first_relevant_rank is None:
                    first_relevant_rank = rank
                break

    recalls = {
        f"recall_at_{k}": sum(rank is not None and rank <= k for rank in matched_at_rank)
        / len(matched_at_rank)
        for k in ks
    }
    return {
        "case_id": case.case_id,
        "query_type": case.query_type,
        "first_relevant_rank": first_relevant_rank,
        "reciprocal_rank": 0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank,
        "matched_moments": sum(rank is not None for rank in matched_at_rank),
        "total_relevant_moments": len(matched_at_rank),
        **recalls,
    }


def percentile(values: Sequence[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def aggregate_scores(
    case_scores: Sequence[dict[str, Any]],
    latencies_ms: Sequence[float],
    errors: int = 0,
) -> dict[str, Any]:
    completed = len(case_scores)
    recall_keys = sorted(
        {key for score in case_scores for key in score if key.startswith("recall_at_")},
        key=lambda item: int(item.rsplit("_", 1)[1]),
    )
    summary: dict[str, Any] = {
        "cases": completed + errors,
        "completed": completed,
        "errors": errors,
        "mrr": round(mean(score["reciprocal_rank"] for score in case_scores), 4) if case_scores else 0.0,
        "latency_ms": {
            "p50": round(percentile(latencies_ms, 0.50), 2),
            "p95": round(percentile(latencies_ms, 0.95), 2),
        },
    }
    for key in recall_keys:
        summary[key] = round(mean(score[key] for score in case_scores), 4)
    return summary
