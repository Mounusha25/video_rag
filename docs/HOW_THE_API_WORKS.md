# API contract

Interactive OpenAPI documentation is available at `/docs` when the FastAPI
service is running.

## Search

`POST /search`

```json
{
  "query": "a red car entering the frame",
  "top_k": 10,
  "source_url": "https://www.youtube.com/watch?v=example"
}
```

The API embeds the text with the same CLIP model used during ingestion, runs
MongoDB Atlas vector search, filters low-score noise, and returns ranked source
timestamps. `top_k` must be between 1 and 100.

## Ingestion

`POST /ingest/start?source_url=...&prompt=...` starts one Modal job.

`POST /dataset/create` accepts a prompt and optional URLs. Without URLs it uses
YouTube search and DuckDuckGo discovery, then submits valid YouTube or Twitch
video URLs to Modal.

```json
{
  "prompt": "people dancing in a red room",
  "urls": ["https://www.youtube.com/watch?v=example"],
  "scale": 10
}
```

Progress is exposed through `GET /ingestion/status?url=...`. Jobs are stored in
MongoDB's `ingestion_jobs` collection, so queued, processing, completed, and
failed states survive API restarts. Segment IDs make progress updates
idempotent.

## Modal callbacks

`POST /ingestion/init` initializes durable progress and `POST /frames/bulk`
stores frame vectors and records a completed segment. The primary Modal path
writes both frames and progress directly to MongoDB. These routes remain for
external workers; when `API_KEY` is set, both require:

```text
X-API-Key: <configured key>
```

External workers should send a stable `segment_id` alongside `frames` in the
`POST /frames/bulk` body. Repeating the same ID will not increment progress
twice.

## Health

`GET /health` checks both the API and MongoDB. It returns HTTP 503 when the
database is unavailable so deployment platforms do not mark an unusable API as
healthy.
