# Retrieval evaluation

ShotSpot's evaluation suite measures whether a text query retrieves the correct
video moment. It reports Recall@1, Recall@5, Recall@K, mean reciprocal rank
(MRR), errors, and p50/p95 end-to-end request latency. Results can be sliced by
`visual`, `transcript`, and `mixed` query types and compared across the fused,
visual-only, and transcript-only indexes.

## 1. Create a labeled dataset

Copy the template and replace every placeholder with a video that has already
been ingested:

```bash
cp evals/data/benchmark.template.jsonl evals/data/benchmark.jsonl
```

Each JSONL row contains one query and one or more correct time ranges:

```json
{"id":"video-01-visual","query":"a red car passes the camera","source_url":"https://www.youtube.com/watch?v=abc123","query_type":"visual","tolerance_seconds":5,"relevant_moments":[{"start_seconds":30,"end_seconds":38}]}
```

Label moments by watching the source video, not by inspecting search results.
That separation prevents the benchmark from being tuned to the current system.
Use at least 30 cases for a portfolio result: 10 visual, 10 transcript, and 10
mixed queries across multiple videos. Keep the JSONL file versioned once its
labels are stable.

## 2. Prepare modality vectors

New ingestion writes three 512-dimensional vectors per sampled frame:

- `visual_embedding`: CLIP image vector;
- `transcript_embedding`: CLIP vector of the segment transcript;
- `embedding`: normalized sum of the available image and transcript vectors.

Update the Atlas `frame_vectors` index using
[`ATLAS_VECTOR_INDEX.md`](ATLAS_VECTOR_INDEX.md), deploy the Modal ingestor, and
re-ingest the benchmark videos. Old rows support fused search only.

## 3. Run the benchmark

Start the API, then run:

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

`--restrict-to-dataset` searches only the sources named by the benchmark. Omit
it to measure retrieval against the entire database. Add
`--fail-under-recall-5 0.70` only after establishing and reviewing a baseline.

Generated reports are ignored by Git because live metrics depend on the indexed
corpus and deployed model. Commit a specific report only when it is accompanied
by the exact benchmark dataset, configuration, and commit SHA.

## Interpreting results

- Recall@5 answers: “What fraction of labeled moments appeared in the first
  five results?”
- MRR rewards putting the first correct result near rank one.
- Visual/transcript slices show which modality benefits from fusion.
- Latency includes query embedding, Atlas retrieval, and API overhead.
- A small hand-labeled dataset is evidence for that dataset only; do not present
  it as universal accuracy.
