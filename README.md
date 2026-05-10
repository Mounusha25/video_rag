# ShotSpot — Semantic Video Search

![ShotSpot Demo](example.png)

---

## The problem

Finding a specific moment in a long video is painful. You scrub through manually, guess timestamps, and repeat. There is no equivalent of Ctrl+F for video.

ShotSpot fixes that. You describe what you're looking for in plain English and it returns the exact timestamps where that thing appears — across any YouTube video — with a clickable embedded preview at the right second.

---

## What it does

ShotSpot lets you search inside YouTube videos using plain English. Type what you're looking for — `"close-up of someone's face"`, `"explosion in the background"`, `"text on a whiteboard"` — and it returns the exact timestamps where that appears, with a clickable player that jumps straight to that moment.

No manual scrubbing. No timestamps guessing. No frame-by-frame review.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          USER QUERY                             │
│                   "dancing in a red room"                       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                    CLIP Text Encoder
                    (512-dim vector)
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              MongoDB Atlas — $vectorSearch                       │
│         cosine similarity across all stored frame vectors       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
               Top-K results ranked by score
               Grouped into timestamp segments
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATASET EXPLORER (UI)                        │
│       Frame cards with timestamp + embedded YouTube player      │
└─────────────────────────────────────────────────────────────────┘
```

---

## How a video gets indexed

Each video is processed once through an ingestion pipeline before it becomes searchable:

```
YouTube URL
    │
    ├─ yt-dlp download
    │
    ├─ Frame sampling (1 frame / 5 seconds)
    │       │
    │       └─ CLIP visual encoder → 512-dim visual vector
    │
    ├─ Whisper transcription (audio → text, chunked by timestamp)
    │       │
    │       └─ CLIP text encoder → 512-dim text vector
    │
    ├─ EasyOCR (on-screen text extraction)
    │       │
    │       └─ CLIP text encoder → 512-dim text vector
    │
    └─ Fusion: visual + text vectors averaged + renormalized
               → stored in MongoDB as one document per frame
                 with: embedding, timestamp, source_url, title
```

The fusion step combines what the frame *looks like* with what is *being said* and *shown as text* at that moment. This makes search work across visual, spoken, and written content simultaneously.

Ingestion runs on Modal GPU workers (A10G). Search runs locally on CPU — no GPU needed at query time.

---

## Tech stack

| Layer | Technology | Role |
|---|---|---|
| Embedding model | `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` | Visual + text encoding (512-dim) |
| Transcription | OpenAI Whisper large-v2 | Audio → timestamped text |
| OCR | EasyOCR | On-screen text extraction |
| Vector database | MongoDB Atlas Vector Search | Cosine similarity search at scale |
| GPU workers | Modal (A10G) | Serverless ingestion pipeline |
| Backend | FastAPI + uvicorn | REST API: search, ingest, stats |
| Frontend | Next.js 14, Tailwind CSS | Dataset Explorer UI |
| Deployment | Vercel | Frontend + serverless API |

---

## Workflow

### 1. Ingest a video

Submit a YouTube URL via the **ANALYZE DATA** tab. The backend triggers the Modal ingestion pipeline, which samples frames, runs CLIP + Whisper + OCR, fuses the embeddings, and writes one document per frame to MongoDB.

This runs once per video. Subsequent searches against the same video are instant.

### 2. Search

Enter a text query in the **ANALYZE DATA** tab and click **INITIALIZE INGEST**. If the video is already indexed, results load immediately from the existing vectors — no re-ingestion.

Each result shows:
- Matched timestamp (e.g. `2m 15s`)
- Similarity score
- Inline YouTube embed starting at that exact second

### 3. Export

Click **EXPORT DATASET (JSON)** to download all matched frames as a structured dataset with timestamps, scores, and source URLs — ready for use in model training or annotation pipelines.

---

## Running locally

### Prerequisites
- Python 3.11+
- Node.js 18+
- MongoDB Atlas cluster (free M0 tier works)
- `.env` file — copy `.env.example` and fill in values

### Backend

```bash
pip install -r requirements.txt
uvicorn app.backend.api:app --reload --port 8000
```

CLIP (~605 MB) is downloaded once on first startup and cached at `~/.cache/huggingface/`. No GPU needed for search — runs on CPU.

### Frontend

```bash
cd app/frontend
npm install
npm run dev
# → http://localhost:3000
```

The frontend proxies `/api/*` → `http://localhost:8000` via `next.config.js`.

### Create the MongoDB vector index (one-time)

```python
from pymongo import MongoClient
import certifi, os
from dotenv import load_dotenv
load_dotenv()

c = MongoClient(os.environ["MONGODB_URI"], tlsCAFile=certifi.where())
c["videorag"].command({
    "createSearchIndexes": "frames",
    "indexes": [{
        "name": "frame_vectors",
        "type": "vectorSearch",
        "definition": {
            "fields": [{
                "type": "vector",
                "path": "embedding",
                "numDimensions": 512,
                "similarity": "cosine"
            }]
        }
    }]
})
```

---

## Project structure

```
app/
  backend/api.py          # FastAPI — /search, /ingest/start, /stats, /frames/bulk
  frontend/               # Next.js 14 UI (Dataset Explorer, ingest controls)
modal_infra/
  ingestor.py             # Main GPU pipeline: download → sample → embed → write
  embedder.py             # Standalone CLIP text embedder (Modal function)
  ocr.py                  # EasyOCR worker
  transcription.py        # Whisper transcription worker
db.py                     # MongoDB client + $vectorSearch helpers
api/index.py              # Vercel serverless entry point
docs/                     # Atlas index setup, deployment guide
```

---

## Environment variables

| Variable | Description |
|---|---|
| `MONGODB_URI` | MongoDB Atlas connection string |
| `MONGODB_DB` | Database name (default: `videorag`) |
| `MODAL_TOKEN_ID` | Modal API token (required for ingestion) |
| `MODAL_TOKEN_SECRET` | Modal API secret (required for ingestion) |
| `YOUTUBE_API_KEY` | YouTube Data API key (optional, used for video metadata) |
  - Shared dataset libraries with role-based access control
  - API keys for CI/CD integration
  - Usage analytics and cost tracking dashboards
  - Webhook notifications for completed jobs
- **Impact**: Enable ML teams to standardize their data collection workflow, reducing time-to-model by 60%

## Impact & Vision

ShotSpot addresses a **$2.7B market opportunity** in computer vision data labeling. By automating video data collection:

- **Time savings**: 95% reduction in manual dataset curation time
- **Cost savings**: $50-100/hour human labeling → $5-10 automated processing
- **Accessibility**: Democratizes AI development for researchers without data engineering teams

Our vision: **Make video data as searchable and accessible as text**, enabling the next generation of computer vision applications.
