# Deployment

ShotSpot separates the lightweight FastAPI service from GPU inference. The API
image calls Modal for query embeddings and ingestion, so it does not install
PyTorch or model weights.

## Required configuration

Configure these as platform secrets, never in source control:

- `MONGODB_URI`
- `MONGODB_DB` (normally `videorag`)
- `MODAL_TOKEN_ID`
- `MODAL_TOKEN_SECRET`
- `ALLOWED_ORIGINS` (comma-separated frontend origins)
- `API_KEY` (recommended for Modal callback routes)

The Modal `mongo` secret must contain MongoDB access. Its ingestion environment
should also define `VECTOR_API_URL` and `VECTOR_API_KEY` so progress callbacks
can reach the API.

## Deploy Modal functions

```bash
modal deploy modal_infra/ingestor.py
modal deploy modal_infra/embedder.py
```

The backend currently looks up:

- app: `treehacks-video-ingestor-v2`, function: `ingest_video_orchestrator`
- app: `clip-embedder`, function: `embed_text`

## Validate the container

```bash
docker build -t shotspot-api .
docker run --env-file .env -p 8000:8000 shotspot-api
curl http://127.0.0.1:8000/health
```

`/health` returns HTTP 503 until MongoDB is reachable.

## Fly.io

After configuring secrets with `fly secrets set`, deploy with:

```bash
fly deploy
fly status
```

The GitHub **Fly Deploy** workflow is intentionally manual and requires the
`FLY_API_TOKEN` repository secret plus approval for the `production`
environment. Normal pushes run CI without attempting a deployment.

## Frontend

Deploy `app/frontend` separately and set `NEXT_PUBLIC_API_URL` to the public API
base URL. Add the frontend origin to `ALLOWED_ORIGINS` on the API.
