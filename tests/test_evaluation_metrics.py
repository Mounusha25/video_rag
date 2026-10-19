import json

import pytest

from evals.metrics import (
    BenchmarkCase,
    RelevantMoment,
    aggregate_scores,
    normalize_source_url,
    score_case,
)
from evals.run import load_dataset


def benchmark_case() -> BenchmarkCase:
    return BenchmarkCase(
        case_id="visual-1",
        query="red car",
        source_url="https://www.youtube.com/watch?v=abc123",
        relevant_moments=(
            RelevantMoment(10, 15),
            RelevantMoment(40, 45),
        ),
        query_type="visual",
        tolerance_seconds=2,
    )


def test_score_case_computes_recall_and_reciprocal_rank():
    results = [
        {
            "source_url": "https://www.youtube.com/watch?v=wrong&t=12s",
            "timestamp_seconds": 12,
        },
        {
            "source_url": "https://www.youtube.com/watch?v=abc123&t=12s",
            "timestamp_seconds": 12,
        },
        {
            "source_url": "https://www.youtube.com/watch?v=abc123&t=44s",
            "timestamp_seconds": 44,
        },
    ]

    score = score_case(benchmark_case(), results, ks=(1, 2, 5))

    assert score["recall_at_1"] == 0
    assert score["recall_at_2"] == 0.5
    assert score["recall_at_5"] == 1
    assert score["reciprocal_rank"] == 0.5


def test_url_normalization_removes_timestamp_but_keeps_video_id():
    assert normalize_source_url(
        "https://www.youtube.com/watch?v=abc123&t=65s&utm_source=test"
    ) == normalize_source_url("https://youtube.com/watch?v=abc123")


def test_aggregate_scores_uses_interpolated_latency_percentiles():
    summary = aggregate_scores(
        [
            {"reciprocal_rank": 1.0, "recall_at_1": 1.0, "recall_at_5": 1.0},
            {"reciprocal_rank": 0.0, "recall_at_1": 0.0, "recall_at_5": 0.5},
        ],
        [100, 300],
    )
    assert summary["mrr"] == 0.5
    assert summary["recall_at_5"] == 0.75
    assert summary["latency_ms"] == {"p50": 200.0, "p95": 290.0}


def test_load_dataset_rejects_duplicate_ids(tmp_path):
    row = {
        "id": "duplicate",
        "query": "demo",
        "source_url": "https://youtu.be/abc",
        "relevant_moments": [{"start_seconds": 1, "end_seconds": 2}],
    }
    path = tmp_path / "benchmark.jsonl"
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")

    with pytest.raises(ValueError, match="duplicate id"):
        load_dataset(path)
