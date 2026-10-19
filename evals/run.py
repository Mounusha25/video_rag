"""Run the ShotSpot benchmark against a live API.

Example:
    python -m evals.run --dataset evals/data/benchmark.jsonl --modes fused
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx

from evals.metrics import (
    BenchmarkCase,
    RelevantMoment,
    aggregate_scores,
    score_case,
)

VALID_MODES = ("fused", "visual", "transcript")


def load_dataset(path: Path) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    seen_ids: set[str] = set()
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            item = json.loads(raw_line)
            case_id = str(item["id"])
            if case_id in seen_ids:
                raise ValueError(f"duplicate id {case_id!r}")
            moments = tuple(
                RelevantMoment(float(moment["start_seconds"]), float(moment["end_seconds"]))
                for moment in item["relevant_moments"]
            )
            if not moments:
                raise ValueError("relevant_moments must not be empty")
            if any(moment.start_seconds > moment.end_seconds for moment in moments):
                raise ValueError("moment start_seconds must be <= end_seconds")
            case = BenchmarkCase(
                case_id=case_id,
                query=str(item["query"]),
                source_url=str(item["source_url"]),
                relevant_moments=moments,
                query_type=str(item.get("query_type", "mixed")),
                tolerance_seconds=float(item.get("tolerance_seconds", 5.0)),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid benchmark row {line_number}: {exc}") from exc
        if not case.query.strip() or not case.source_url.strip():
            raise ValueError(f"Invalid benchmark row {line_number}: query and source_url are required")
        seen_ids.add(case.case_id)
        cases.append(case)
    if not cases:
        raise ValueError("Benchmark dataset is empty")
    return cases


def evaluate_mode(
    client: httpx.Client,
    cases: list[BenchmarkCase],
    mode: str,
    top_k: int,
    restrict_to_dataset: bool,
    warmup: bool,
) -> dict[str, Any]:
    scores: list[dict[str, Any]] = []
    latencies: list[float] = []
    failures: list[dict[str, str]] = []
    allowed_sources = sorted({case.source_url for case in cases})

    if warmup:
        payload: dict[str, Any] = {
            "query": cases[0].query,
            "top_k": top_k,
            "mode": mode,
            "min_score": -1.0,
        }
        if restrict_to_dataset:
            payload["allowed_sources"] = allowed_sources
        response = client.post("/search", json=payload)
        response.raise_for_status()
        body = response.json()
        if not body.get("ok"):
            raise RuntimeError(body.get("error", "warmup search returned ok=false"))
        print(f"{mode:<10} warmup complete (excluded from metrics)")

    for case in cases:
        payload: dict[str, Any] = {
            "query": case.query,
            "top_k": top_k,
            "mode": mode,
            "min_score": -1.0,
        }
        if restrict_to_dataset:
            payload["allowed_sources"] = allowed_sources

        started = time.perf_counter()
        try:
            response = client.post("/search", json=payload)
            latency_ms = (time.perf_counter() - started) * 1000
            response.raise_for_status()
            body = response.json()
            if not body.get("ok"):
                raise RuntimeError(body.get("error", "search returned ok=false"))
            result = score_case(case, body.get("results", []), ks=(1, 5, top_k))
            result["latency_ms"] = round(latency_ms, 2)
            scores.append(result)
            latencies.append(latency_ms)
            print(
                f"{mode:<10} {case.case_id:<28} "
                f"R@5={result['recall_at_5']:.3f} "
                f"RR={result['reciprocal_rank']:.3f} {latency_ms:8.1f} ms"
            )
        except Exception as exc:  # retain all failed cases in the report
            failures.append({"case_id": case.case_id, "error": str(exc)})
            print(f"{mode:<10} {case.case_id:<28} ERROR {exc}", file=sys.stderr)

    summary = aggregate_scores(scores, latencies, errors=len(failures))
    slices: dict[str, Any] = {}
    for query_type in sorted({case.query_type for case in cases}):
        ids = {case.case_id for case in cases if case.query_type == query_type}
        slice_scores = [score for score in scores if score["case_id"] in ids]
        slice_latencies = [score["latency_ms"] for score in slice_scores]
        slice_errors = sum(failure["case_id"] in ids for failure in failures)
        slices[query_type] = aggregate_scores(slice_scores, slice_latencies, slice_errors)
    return {"summary": summary, "slices": slices, "cases": scores, "failures": failures}


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ShotSpot retrieval evaluation",
        "",
        f"Dataset: `{report['dataset']}`  ",
        f"Generated: `{report['generated_at']}`",
        "",
        "| Mode | Cases | Errors | Recall@1 | Recall@5 | MRR | p50 ms | p95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, result in report["modes"].items():
        summary = result["summary"]
        lines.append(
            f"| {mode} | {summary['cases']} | {summary['errors']} | "
            f"{summary.get('recall_at_1', 0):.3f} | {summary.get('recall_at_5', 0):.3f} | "
            f"{summary['mrr']:.3f} | {summary['latency_ms']['p50']:.1f} | "
            f"{summary['latency_ms']['p95']:.1f} |"
        )
    lines.extend([
        "",
        "> Metrics describe only this versioned labeled dataset and the recorded run.",
        "",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ShotSpot timestamp retrieval")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--modes", default="fused", help="Comma-separated: fused,visual,transcript")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--restrict-to-dataset", action="store_true")
    parser.add_argument(
        "--warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Warm each retrieval mode before measuring latency (default: enabled)",
    )
    parser.add_argument("--fail-under-recall-5", type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.top_k < 5 or args.top_k > 100:
        raise SystemExit("--top-k must be between 5 and 100")
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    invalid_modes = sorted(set(modes) - set(VALID_MODES))
    if not modes or invalid_modes:
        raise SystemExit(f"Invalid modes: {invalid_modes or modes}. Choose from {VALID_MODES}")

    cases = load_dataset(args.dataset)
    report: dict[str, Any] = {
        "dataset": str(args.dataset),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "configuration": {
            "base_url": args.base_url,
            "top_k": args.top_k,
            "restrict_to_dataset": args.restrict_to_dataset,
            "warmup": args.warmup,
            "modes": modes,
        },
        "dataset_cases": [asdict(case) for case in cases],
        "modes": {},
    }
    with httpx.Client(base_url=args.base_url.rstrip("/"), timeout=args.timeout) as client:
        for mode in modes:
            report["modes"][mode] = evaluate_mode(
                client, cases, mode, args.top_k, args.restrict_to_dataset, args.warmup
            )

    output_text = json.dumps(report, indent=2)
    print(output_text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text + "\n")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(report))

    failed = any(result["summary"]["errors"] for result in report["modes"].values())
    if args.fail_under_recall_5 is not None:
        failed = failed or any(
            result["summary"].get("recall_at_5", 0) < args.fail_under_recall_5
            for result in report["modes"].values()
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
