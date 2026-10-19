# Set up Atlas Vector Search (required)

This project is **vector-based**: search uses only Atlas Vector Search. Create the **vector index** once.

---

## Create the vector index in Atlas

1. Open [Atlas](https://cloud.mongodb.com) → your project → your **cluster**.
2. Go to the **Search** tab (or **Atlas Search** in the left menu).
3. Click **Create Search Index**.
4. Choose **JSON Editor** (or **Visual Editor** if available).
5. **Database:** your DB (e.g. `TreeHacks2026`). **Collection:** `frames`.
6. **Index name:** `frame_vectors` (must match `index_name` in `search()`, or pass a different name in code).
7. Use this index definition:

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 512,
      "similarity": "cosine"
    },
    {
      "type": "vector",
      "path": "visual_embedding",
      "numDimensions": 512,
      "similarity": "cosine"
    },
    {
      "type": "vector",
      "path": "transcript_embedding",
      "numDimensions": 512,
      "similarity": "cosine"
    },
    {
      "type": "vector",
      "path": "ocr_embedding",
      "numDimensions": 512,
      "similarity": "cosine"
    },
    {
      "type": "filter",
      "path": "source"
    }
  ]
}
```

**Important:** 
1. `numDimensions` MUST be **512** for CLIP ViT-B/32.
2. The `source` filter field is required for dataset-scoped evaluation.
3. Existing documents created before multimodal evaluation contain only
   `embedding`. Re-ingest benchmark videos to populate `visual_embedding` and
   `transcript_embedding` and `ocr_embedding` before comparing modes.

# Troubleshooting
- **Seeing old video clips?** You likely forgot to add the `source` filter block to your Atlas Search Index definition. Without it, the `filter` in the query is silently ignored.

8. Create the index. It may take a few minutes to build.

If `frame_vectors` already exists, update it from the project root instead of
creating a duplicate:

```bash
python - <<'PY'
from dotenv import load_dotenv
load_dotenv('.env')
from db import update_vector_index
print(update_vector_index())
PY
```

Atlas keeps the old index queryable while the new definition builds. Confirm
that its status returns to `READY` before running OCR-mode evaluation.

---

## 2. Use vector search in code

```python
from db import insert_frame, search

# Insert
insert_frame([0.1, 0.2, 0.3, 0.4, 0.5], frame_id="f1")

# Vector search (uses the index)
results = search([0.1, 0.2, 0.3, 0.4, 0.5], top_k=10)
for doc in results:
    print(doc["score"], doc.get("frame_id"))
```

Use `index_name=` if your index has a different name.
