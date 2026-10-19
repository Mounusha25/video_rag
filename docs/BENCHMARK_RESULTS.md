# ShotSpot retrieval baseline

This report records the first source-grounded multimodal retrieval baseline for
ShotSpot. It is evidence for this dataset and configuration only, not a claim
about arbitrary videos.

## Dataset and protocol

- Run date: 2026-10-19 UTC
- Dataset: `evals/data/benchmark.jsonl`
- Sources: three real YouTube videos already indexed in MongoDB Atlas
- Cases: 30 total — 10 visual, 10 transcript, and 10 mixed
- Sampling: one frame every five seconds; transcripts grouped into 15-second segments
- Retrieval: MongoDB Atlas Vector Search, top 10, restricted to benchmark sources
- Query embedding: local CLIP inference on Apple arm64
- Timing: one unmeasured warm-up request per mode; measured latency covers the API,
  query embedding, Atlas retrieval, and response handling
- Errors: 0 across 90 measured searches

Labels were created from direct inspection of the videos and stored transcripts,
not from the search results. That keeps the benchmark independent of the current
ranking output.

## Overall results

| Mode | Recall@1 | Recall@5 | Recall@10 | MRR | p50 | p95 |
|---|---:|---:|---:|---:|---:|---:|
| Fused | 0.483 | **0.717** | 0.850 | **0.632** | 1088.7 ms | 1121.3 ms |
| Visual only | 0.400 | 0.683 | **0.867** | 0.546 | 1081.9 ms | 1126.6 ms |
| Transcript only | 0.433 | 0.533 | 0.633 | 0.486 | 1078.1 ms | 1141.3 ms |

Fused retrieval produces the strongest overall rank quality and Recall@5.
Visual-only has slightly higher Recall@10, so fusion is not uniformly better at
every cutoff.

## Query-type ablation

| Mode | Visual R@5 | Transcript R@5 | Mixed R@5 |
|---|---:|---:|---:|
| Fused | 0.650 | 0.600 | 0.900 |
| Visual only | **0.750** | 0.600 | 0.700 |
| Transcript only | 0.000 | **0.700** | **0.900** |

The ablation behaves coherently: visual-only leads on visual questions,
transcript-only leads on spoken-content questions, and transcript-only retrieves
none of the purely visual moments in the first five results. The current equal
fusion is therefore a useful general baseline, but modality weighting is a clear
next optimization target.

## Reproduce

After ingesting the three dataset sources and starting the API:

```bash
python -m evals.run \
  --dataset evals/data/benchmark.jsonl \
  --modes fused,visual,transcript \
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
- The benchmark does not currently measure OCR, ingestion throughput, or cold starts.
