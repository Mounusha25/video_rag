# ShotSpot — Multimodal Video Search

![ShotSpot interface](example.png)

ShotSpot indexes moments from YouTube and Twitch videos and retrieves relevant
timestamps from a natural-language query. The current ingestion path combines
CLIP frame embeddings with Whisper transcript context, stores normalized vectors
in MongoDB Atlas, and returns timestamped results through FastAPI and Next.js.

## Current status

The repository contains a working ingestion and search architecture, but it is
still a portfolio prototype rather than a production service. Its current,
verified boundaries are explicit:

- The main Modal ingestor samples one frame every five seconds.
- It uses `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` for image and query vectors.
- It transcribes each segment with `openai/whisper-tiny` and fuses that context
  with the frame vector through normalized vector addition.
- YouTube and Twitch video URLs are accepted by the ingestion API.
- MongoDB Atlas `$vectorSearch` performs approximate nearest-neighbor retrieval.
- OCR exists as an experimental standalone Modal service; it is **not yet part
  of the production ingestion fusion path**.
- A versioned evaluation harness measures Recall@K, MRR, query-type slices,
  error counts, and p50/p95 latency against a labeled JSONL dataset.
- The first source-grounded benchmark contains 30 labeled cases across three
  real videos. Fused retrieval achieved 0.717 Recall@5 and 0.632 MRR with zero
  request errors; see [the benchmark report](docs/BENCHMARK_RESULTS.md) for the
  exact scope, ablations, and limitations.

## Architecture

```text
YouTube/Twitch URL
        │
        ▼
Modal ingestion orchestrator
        │ split into overlapping 15-second segments
        ├── Whisper-tiny transcript
        └── frames sampled every 5 seconds ──► CLIP image encoder
                                                │
                         CLIP transcript vector ─┤ normalized late fusion
                                                ▼
                                  MongoDB Atlas Vector Search
                                                │
Natural-language query ──► CLIP text encoder ───┘
                                                │
                                                ▼
                         ranked timestamps + source video links
```

The API uses the deployed Modal text embedder by default. Local CPU inference
is optional so the Fly/Docker image does not need to contain PyTorch and the
605 MB CLIP checkpoint.

Short benchmark videos can also be ingested locally with
`scripts/ingest_local.py`, which avoids requiring paid cloud GPU access.

## Repository layout

```text
app/backend/api.py       FastAPI search, ingestion, export, and status API
app/frontend/            Next.js dataset explorer
modal_infra/ingestor.py  GPU frame + transcript ingestion pipeline
modal_infra/embedder.py  GPU text-query embedding endpoint
modal_infra/ocr.py       Experimental OCR service (not fused yet)
db.py                    MongoDB inserts and Atlas vector search
evals/                   Labeled retrieval benchmark and metric runner
tests/                   deterministic API and timestamp unit tests
docs/                    deployment and Atlas setup guides
```

## API surface

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/` | Service metadata |
| `GET` | `/health` | API and MongoDB readiness |
| `POST` | `/scrape` | Validate supported video URLs |
| `POST` | `/dataset/create` | Discover or submit sources and start ingestion |
| `POST` | `/ingest/start` | Start one Modal ingestion job |
| `GET` | `/ingestion/status` | Read in-memory progress for a source URL |
| `POST` | `/search` | Retrieve fused, visual-only, or transcript-only moments |
| `POST` | `/frames/bulk` | Modal callback for storing frame vectors |
| `POST` | `/ingestion/init` | Modal callback for initializing progress |
| `GET` | `/dataset/export` | Export result metadata as a ZIP archive |

If `API_KEY` is configured, the two Modal callback routes require it through
the `X-API-Key` header. Search remains public. Configure `ALLOWED_ORIGINS` as a
comma-separated list for browser clients.

## Local development

### Backend using the deployed Modal embedder

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.backend.api:app --reload --port 8000
```

Required environment variables are documented in `.env.example`. At minimum,
search needs MongoDB and Modal credentials:

```text
MONGODB_URI=...
MONGODB_DB=videorag
MODAL_TOKEN_ID=...
MODAL_TOKEN_SECRET=...
```

### Optional local query embedding

```bash
pip install -r requirements-local.txt
export EMBEDDING_BACKEND=local
```

The first local search downloads and caches the CLIP checkpoint. GPU ingestion
still runs through Modal.

### Frontend

```bash
cd app/frontend
npm ci
npm run dev
```

The development frontend proxies `/api/*` to `http://localhost:8000`.

## Validation

```bash
pip install -r requirements-dev.txt
pytest -q

cd app/frontend
npm ci
npm run build
```

GitHub Actions runs backend tests, a production frontend build, and a Docker
build on pushes and pull requests. Fly deployment runs for code changes when
`FLY_API_TOKEN` is configured and skips cleanly otherwise; it can also be
started manually.

For retrieval evaluation, label real video moments and run the fused/visual/
transcript comparison documented in [docs/EVALUATION.md](docs/EVALUATION.md).
The committed [benchmark dataset](evals/data/benchmark.jsonl) and
[baseline report](docs/BENCHMARK_RESULTS.md) make the current result
reproducible without presenting it as universal model accuracy.

## Deployment

The lightweight API image uses Modal for query embedding:

```bash
docker build -t shotspot-api .
docker run --env-file .env -p 8000:8000 shotspot-api
```

For Fly.io, configure `MONGODB_URI`, `MONGODB_DB`, Modal credentials,
`ALLOWED_ORIGINS`, and optionally `API_KEY`. Then run the manual **Fly Deploy**
workflow or `fly deploy`. See [docs/DEPLOY.md](docs/DEPLOY.md).

## Roadmap

1. Expand the benchmark beyond three videos and add a held-out source split.
2. Integrate OCR into the primary ingestion path and measure its incremental gain.
3. Persist ingestion jobs instead of tracking them in process memory.
4. Tune modality weights against development cases, then evaluate once on held-out cases.

## Security note

Never commit browser cookie exports, `.env` files, MongoDB credentials, Modal
tokens, or API keys. `cookies.txt` is ignored and removed from current tracking.
If a cookie export was ever pushed to a public remote, revoke the corresponding
website sessions because deleting the current file does not erase Git history.

## License

This project is provided for educational and portfolio use.
