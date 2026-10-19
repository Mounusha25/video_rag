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
- A labeled retrieval benchmark is the next milestone. No Recall@K, MRR, or
  time-savings claims are made before that benchmark exists.

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

## Repository layout

```text
app/backend/api.py       FastAPI search, ingestion, export, and status API
app/frontend/            Next.js dataset explorer
modal_infra/ingestor.py  GPU frame + transcript ingestion pipeline
modal_infra/embedder.py  GPU text-query embedding endpoint
modal_infra/ocr.py       Experimental OCR service (not fused yet)
db.py                    MongoDB inserts and Atlas vector search
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
| `POST` | `/search` | Embed text and retrieve timestamped moments |
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
build on pushes and pull requests. Fly deployment is manual so a missing or
expired deployment token cannot make every source commit appear broken.

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

1. Add a versioned labeled query-to-timestamp benchmark.
2. Report Recall@1, Recall@5, MRR, timestamp tolerance, and p50/p95 latency.
3. Compare visual-only, transcript-only, and fused retrieval.
4. Integrate OCR into the primary ingestion path and measure its incremental gain.
5. Persist ingestion jobs instead of tracking them in process memory.

## Security note

Never commit browser cookie exports, `.env` files, MongoDB credentials, Modal
tokens, or API keys. `cookies.txt` is ignored and removed from current tracking.
If a cookie export was ever pushed to a public remote, revoke the corresponding
website sessions because deleting the current file does not erase Git history.

## License

This project is provided for educational and portfolio use.
