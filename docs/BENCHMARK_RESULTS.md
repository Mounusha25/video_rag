# ShotSpot retrieval baseline

This report records the first source-grounded multimodal retrieval baseline for
ShotSpot. It is evidence for this dataset and configuration only, not a claim
about arbitrary videos.

## Dataset and protocol

- Run date: 2026-10-19 UTC
- Dataset: `evals/data/benchmark.jsonl`
- Sources: three real YouTube videos already indexed in MongoDB Atlas
- Cases: 40 total — 10 visual, 10 transcript, 10 OCR, and 10 mixed
- Sampling: one source-resolution frame every five seconds; transcripts grouped
  into 15-second segments
- OCR: EasyOCR English recognition with a 0.35 confidence threshold; detected
  text embedded with the same CLIP text encoder used for queries
- Retrieval: MongoDB Atlas Vector Search, top 10, restricted to benchmark sources
- Query embedding: local CLIP inference on Apple arm64
- Timing: one unmeasured warm-up request per mode; measured latency covers the API,
  query embedding, Atlas retrieval, and response handling
- Errors: 0 across 160 measured searches

Labels were created from direct inspection of the videos and stored transcripts,
not from the search results. That keeps the benchmark independent of the current
ranking output.

## Overall results

| Mode | Recall@1 | Recall@5 | Recall@10 | MRR | p50 | p95 |
|---|---:|---:|---:|---:|---:|---:|
| Fused | 0.525 | **0.788** | **0.863** | **0.647** | 1157.5 ms | 1186.1 ms |
| Visual only | **0.538** | 0.713 | 0.838 | 0.636 | 1163.2 ms | 1195.0 ms |
| Transcript only | 0.425 | 0.575 | 0.650 | 0.482 | 1159.6 ms | 1178.6 ms |
| OCR only | 0.525 | 0.675 | 0.675 | 0.581 | 1159.6 ms | 1179.2 ms |

Fused retrieval produces the strongest overall MRR, Recall@5, and Recall@10.
Visual-only has slightly higher Recall@1, so fusion is not uniformly better at
every cutoff.

## Query-type ablation

| Mode | Visual R@5 | Transcript R@5 | OCR R@5 | Mixed R@5 |
|---|---:|---:|---:|---:|
| Fused | 0.450 | **0.800** | **1.000** | **0.900** |
| Visual only | **0.550** | 0.500 | 0.900 | **0.900** |
| Transcript only | 0.000 | 0.700 | 0.700 | **0.900** |
| OCR only | 0.200 | 0.700 | **1.000** | 0.800 |

The ablation behaves coherently: visual-only leads on visual questions, OCR-only
retrieves all visible-text cases at rank one, and transcript-only retrieves none
of the purely visual moments in the first five results. Equal available-modality
fusion is the strongest general baseline, but the weaker visual slice shows that
query-adaptive modality weighting remains a clear optimization target.

## Reproduce

After ingesting the three dataset sources and starting the API:

```bash
python -m evals.run \
  --dataset evals/data/benchmark.jsonl \
  --modes fused,visual,transcript,ocr \
  --top-k 10 \
  --restrict-to-dataset \
  --timeout 120 \
  --output evals/results/latest.json \
  --markdown evals/results/latest.md
```

The source URLs and relevant time ranges are versioned in the JSONL dataset.
Generated per-case reports remain ignored because they depend on the live Atlas
corpus and execution environment.

## Limitations

- Three videos are enough for an end-to-end baseline, not broad generalization.
- Cases are source-grounded but have not received a second independent annotator.
- Sources are searched together, but there is no held-out-video test split yet.
- The benchmark does not currently measure ingestion throughput or cold starts.
