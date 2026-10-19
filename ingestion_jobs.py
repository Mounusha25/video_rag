"""Durable MongoDB-backed ingestion job state.

The API and Modal workers share this schema so progress survives API restarts
and concurrent segment completions are recorded atomically.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pymongo import ASCENDING, DESCENDING, ReturnDocument

from db import get_db

COLLECTION_NAME = "ingestion_jobs"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_source_url(source_url: str) -> str:
    return source_url.strip().rstrip("/")


def get_jobs_collection(db=None):
    if db is None:
        db = get_db()
    return db[COLLECTION_NAME]


def ensure_job_indexes(collection=None) -> None:
    if collection is None:
        collection = get_jobs_collection()
    collection.create_index([("source_url", ASCENDING)], unique=True)
    collection.create_index([("status", ASCENDING), ("updated_at", DESCENDING)])


def queue_job(
    source_url: str,
    *,
    prompt: str = "",
    worker_target: int = 0,
    collection=None,
    now: datetime | None = None,
) -> dict[str, Any]:
    if collection is None:
        collection = get_jobs_collection()
    ensure_job_indexes(collection)
    now = now or utc_now()
    normalized = normalize_source_url(source_url)
    return collection.find_one_and_update(
        {"source_url": normalized},
        {
            "$set": {
                "source_url": normalized,
                "prompt": prompt,
                "worker_target": worker_target,
                "status": "queued",
                "total_segments": 0,
                "processed_segments": 0,
                "processed_frames": 0,
                "failed_segments": 0,
                "completed_segment_ids": [],
                "created_at": now,
                "updated_at": now,
                "started_at": None,
                "completed_at": None,
                "error": None,
                "remote_job_id": None,
            }
        },
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )


def attach_remote_job(
    source_url: str,
    remote_job_id: str,
    *,
    collection=None,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    if collection is None:
        collection = get_jobs_collection()
    return collection.find_one_and_update(
        {"source_url": normalize_source_url(source_url)},
        {"$set": {"remote_job_id": remote_job_id, "updated_at": now or utc_now()}},
        return_document=ReturnDocument.AFTER,
    )


def initialize_job(
    source_url: str,
    total_segments: int,
    *,
    collection=None,
    now: datetime | None = None,
) -> dict[str, Any]:
    if total_segments < 1:
        raise ValueError("total_segments must be positive")
    if collection is None:
        collection = get_jobs_collection()
    ensure_job_indexes(collection)
    now = now or utc_now()
    normalized = normalize_source_url(source_url)
    return collection.find_one_and_update(
        {"source_url": normalized},
        {
            "$set": {
                "source_url": normalized,
                "status": "processing",
                "total_segments": total_segments,
                "processed_segments": 0,
                "processed_frames": 0,
                "failed_segments": 0,
                "completed_segment_ids": [],
                "started_at": now,
                "updated_at": now,
                "completed_at": None,
                "error": None,
            },
            "$setOnInsert": {"created_at": now},
        },
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )


def record_segment(
    source_url: str,
    frame_count: int,
    *,
    segment_id: str | None = None,
    collection=None,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    """Atomically record one completed segment; segment_id makes retries idempotent."""
    if frame_count < 0:
        raise ValueError("frame_count must not be negative")
    if collection is None:
        collection = get_jobs_collection()
    now = now or utc_now()
    query: dict[str, Any] = {"source_url": normalize_source_url(source_url)}
    update: dict[str, Any] = {
        "$inc": {"processed_segments": 1, "processed_frames": frame_count},
        "$set": {"updated_at": now},
    }
    if segment_id:
        query["completed_segment_ids"] = {"$ne": segment_id}
        update["$addToSet"] = {"completed_segment_ids": segment_id}

    job = collection.find_one_and_update(
        query,
        update,
        return_document=ReturnDocument.AFTER,
    )
    if job is None:
        return collection.find_one({"source_url": normalize_source_url(source_url)})

    total = int(job.get("total_segments") or 0)
    processed = int(job.get("processed_segments") or 0)
    if total and processed >= total:
        job = collection.find_one_and_update(
            {"_id": job["_id"]},
            {
                "$set": {
                    "status": "completed",
                    "completed_at": now,
                    "updated_at": now,
                }
            },
            return_document=ReturnDocument.AFTER,
        )
    return job


def finish_job(
    source_url: str,
    *,
    failed_segments: int = 0,
    collection=None,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    if collection is None:
        collection = get_jobs_collection()
    now = now or utc_now()
    status = "completed" if failed_segments == 0 else "completed_with_errors"
    return collection.find_one_and_update(
        {"source_url": normalize_source_url(source_url)},
        {
            "$set": {
                "status": status,
                "failed_segments": failed_segments,
                "completed_at": now,
                "updated_at": now,
            }
        },
        return_document=ReturnDocument.AFTER,
    )


def fail_job(
    source_url: str,
    error: str,
    *,
    collection=None,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    if collection is None:
        collection = get_jobs_collection()
    now = now or utc_now()
    return collection.find_one_and_update(
        {"source_url": normalize_source_url(source_url)},
        {
            "$set": {
                "status": "failed",
                "error": error[:1000],
                "completed_at": now,
                "updated_at": now,
            }
        },
        return_document=ReturnDocument.AFTER,
    )


def get_job(source_url: str, *, collection=None) -> dict[str, Any] | None:
    if collection is None:
        collection = get_jobs_collection()
    return collection.find_one({"source_url": normalize_source_url(source_url)})


def public_job_status(job: dict[str, Any] | None) -> dict[str, Any]:
    if not job:
        return {"status": "not_found", "progress": 0}
    total = int(job.get("total_segments") or 0)
    processed = int(job.get("processed_segments") or 0)
    progress = min(1.0, processed / total) if total else 0.0
    return {
        "status": job.get("status", "unknown"),
        "progress": progress,
        "total_segments": total,
        "processed_segments": processed,
        "frames": int(job.get("processed_frames") or 0),
        "failed_segments": int(job.get("failed_segments") or 0),
        "job_id": job.get("remote_job_id"),
        "error": job.get("error"),
    }
