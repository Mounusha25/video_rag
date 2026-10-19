from datetime import datetime, timezone

import mongomock

from ingestion_jobs import (
    attach_remote_job,
    fail_job,
    finish_job,
    get_job,
    initialize_job,
    public_job_status,
    queue_job,
    record_segment,
)


def jobs_collection():
    return mongomock.MongoClient()["videorag"]["ingestion_jobs"]


def test_job_lifecycle_is_persisted_in_mongodb():
    collection = jobs_collection()
    source = "https://www.youtube.com/watch?v=demo/"

    queued = queue_job(
        source,
        prompt="red car",
        worker_target=4,
        collection=collection,
        now=datetime(2026, 10, 18, tzinfo=timezone.utc),
    )
    attach_remote_job(source, "modal-call-1", collection=collection)
    running = initialize_job(source, 2, collection=collection)

    assert queued["status"] == "queued"
    assert running["status"] == "processing"
    assert get_job(source, collection=collection)["remote_job_id"] == "modal-call-1"

    record_segment(source, 3, segment_id="0.000:16.000", collection=collection)
    completed = record_segment(
        source,
        2,
        segment_id="15.000:30.000",
        collection=collection,
    )

    assert public_job_status(completed) == {
        "status": "completed",
        "progress": 1.0,
        "total_segments": 2,
        "processed_segments": 2,
        "frames": 5,
        "failed_segments": 0,
        "job_id": "modal-call-1",
        "error": None,
    }


def test_segment_retry_does_not_double_count_progress():
    collection = jobs_collection()
    source = "https://youtu.be/retry-demo"
    initialize_job(source, 2, collection=collection)

    first = record_segment(
        source,
        4,
        segment_id="segment-1",
        collection=collection,
    )
    retried = record_segment(
        source,
        4,
        segment_id="segment-1",
        collection=collection,
    )

    assert first["processed_segments"] == 1
    assert retried["processed_segments"] == 1
    assert retried["processed_frames"] == 4
    assert retried["completed_segment_ids"] == ["segment-1"]


def test_finish_and_failure_states_are_durable():
    collection = jobs_collection()
    source = "https://youtu.be/partial-demo"
    initialize_job(source, 3, collection=collection)
    partial = finish_job(source, failed_segments=2, collection=collection)

    assert partial["status"] == "completed_with_errors"
    assert partial["failed_segments"] == 2

    failed = fail_job(source, "metadata unavailable", collection=collection)
    persisted = get_job(source, collection=collection)
    assert failed["status"] == "failed"
    assert persisted["error"] == "metadata unavailable"
