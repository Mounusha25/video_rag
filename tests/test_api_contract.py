from fastapi.testclient import TestClient

import app.backend.api as api
import db


client = TestClient(api.app)


def test_scrape_accepts_video_urls_and_rejects_non_video_pages():
    response = client.post(
        "/scrape",
        json={
            "urls": [
                "https://www.youtube.com/watch?v=abc123",
                "https://www.youtube.com/channel/not-a-video",
                "https://example.com/video",
            ]
        },
    )
    assert response.status_code == 200
    assert response.json() == {
        "valid": ["https://www.youtube.com/watch?v=abc123"],
        "invalid": [
            "https://www.youtube.com/channel/not-a-video",
            "https://example.com/video",
        ],
        "count": 1,
    }


def test_search_returns_deterministic_source_thumbnail(monkeypatch):
    monkeypatch.setattr(api, "embed_text", lambda _query: [1.0, 0.0])
    monkeypatch.setattr(
        api,
        "search",
        lambda *_args, **_kwargs: [
            {
                "_id": "frame-1",
                "score": 0.91,
                "time": 65,
                "source": "https://www.youtube.com/watch?v=abc123",
                "title": "Demo",
            }
        ],
    )

    response = client.post("/search", json={"query": "red car", "top_k": 5})

    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["timestamp"] == "1m 5s"
    assert result["timestamp_seconds"] == 65
    assert result["url"] == "https://img.youtube.com/vi/abc123/hqdefault.jpg"
    assert "unsplash" not in result["url"]


def test_bulk_write_requires_key_when_configured(monkeypatch):
    monkeypatch.setenv("API_KEY", "expected-secret")

    response = client.post(
        "/frames/bulk",
        json={"frames": [{"embedding": [1.0, 0.0], "source": "demo"}]},
    )

    assert response.status_code == 401


def test_search_request_bounds_are_validated():
    response = client.post("/search", json={"query": "demo", "top_k": 1000})
    assert response.status_code == 422


def test_search_mode_selects_corresponding_vector_field(monkeypatch):
    captured = {}
    monkeypatch.setattr(api, "embed_text", lambda _query: [1.0, 0.0])

    def fake_search(*_args, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(api, "search", fake_search)
    response = client.post(
        "/search",
        json={"query": "spoken explanation", "mode": "transcript", "min_score": -1},
    )

    assert response.status_code == 200
    assert response.json()["mode"] == "transcript"
    assert captured["vector_path"] == "transcript_embedding"


def test_ocr_search_uses_ocr_vector_and_returns_detected_text(monkeypatch):
    captured = {}
    monkeypatch.setattr(api, "embed_text", lambda _query: [1.0, 0.0])

    def fake_search(*_args, **kwargs):
        captured.update(kwargs)
        return [
            {
                "_id": "frame-ocr",
                "score": 0.88,
                "time": 30,
                "source": "https://www.youtube.com/watch?v=abc123",
                "title": "Demo",
                "ocr_text": "67% of buyers",
                "text_context": "Most buyers prefer self service",
            }
        ]

    monkeypatch.setattr(api, "search", fake_search)
    response = client.post(
        "/search",
        json={"query": "67 percent", "mode": "ocr", "min_score": -1},
    )

    assert response.status_code == 200
    result = response.json()["results"][0]
    assert captured["vector_path"] == "ocr_embedding"
    assert result["ocr_text"] == "67% of buyers"
    assert result["transcript"] == "Most buyers prefer self service"


def test_ingestion_status_reads_durable_job(monkeypatch):
    monkeypatch.setattr(
        api,
        "get_job",
        lambda _url: {
            "status": "processing",
            "total_segments": 4,
            "processed_segments": 2,
            "processed_frames": 7,
            "failed_segments": 0,
            "remote_job_id": "modal-123",
            "error": None,
        },
    )

    response = client.get(
        "/ingestion/status",
        params={"url": "https://www.youtube.com/watch?v=abc123"},
    )

    assert response.status_code == 200
    assert response.json()["progress"] == 0.5
    assert response.json()["frames"] == 7
    assert response.json()["job_id"] == "modal-123"


def test_ingestion_init_persists_processing_state(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    captured = {}

    def fake_initialize(url, total_segments):
        captured.update(url=url, total_segments=total_segments)
        return {
            "status": "processing",
            "total_segments": total_segments,
            "processed_segments": 0,
            "processed_frames": 0,
            "failed_segments": 0,
            "remote_job_id": None,
            "error": None,
        }

    monkeypatch.setattr(api, "initialize_job", fake_initialize)
    response = client.post(
        "/ingestion/init",
        json={"url": "https://youtu.be/demo", "total_segments": 3},
    )

    assert response.status_code == 200
    assert captured == {"url": "https://youtu.be/demo", "total_segments": 3}
    assert response.json()["status"] == "processing"


def test_bulk_write_records_idempotent_segment_after_insert(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setattr(db, "insert_frames", lambda documents: ["frame-1"] * len(documents))
    captured = {}

    def fake_record(source_url, frame_count, *, segment_id=None):
        captured.update(
            source_url=source_url,
            frame_count=frame_count,
            segment_id=segment_id,
        )
        return {"processed_segments": 1, "total_segments": 2}

    monkeypatch.setattr(api, "record_segment", fake_record)
    response = client.post(
        "/frames/bulk",
        json={
            "segment_id": "0.000:16.000",
            "frames": [
                {
                    "embedding": [1.0, 0.0],
                    "source": "https://youtu.be/demo",
                    "timestamp": 0,
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["count"] == 1
    assert captured == {
        "source_url": "https://youtu.be/demo",
        "frame_count": 1,
        "segment_id": "0.000:16.000",
    }
