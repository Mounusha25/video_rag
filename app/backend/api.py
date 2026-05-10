# app/backend/api.py
import os
import sys
from typing import Any, List, Optional
import random
import time

# Load env vars from the root .env file
from dotenv import load_dotenv
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(project_root, ".env"))

# Ensure project root is on sys.path for top-level modules (db, time_stamp_grouping)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict
import modal
import requests
import zipfile
import io
import json

# Import the Modal function directly
try:
    from modal_infra.ingestor import ingest_video_orchestrator
except ImportError as e:
    print(f"Warning: Could not import ingestor: {e}")
    ingest_video_orchestrator = None

# --- MongoDB Integration ---
from db import get_collection, search

try:
    from time_stamp_grouping import get_timestamps as get_timestamp_segments
except ImportError:
    get_timestamp_segments = None

# --- CLIP Model (Text Only) ---
# We load a small CLIP model to embed text queries for search
try:
    from transformers import CLIPProcessor, CLIPModel
    import torch
    print("Loading CLIP (Text Encoder - LAION-2B)...")
    model_id = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)
    print("CLIP Loaded.")
except ImportError:
    print("Warning: transformers/torch not installed. Functionality will use remote Modal fallback.")
    processor = None
    model = None

def embed_text(text: str) -> List[float]:
    if model:
        # Local inference
        inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
            # normalize
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features[0].tolist()
    else:
        # Remote inference via Modal
        try:
            from modal_infra.embedder import embed_text as remote_embed
            print(f"Invoking Modal for text embedding: '{text[:20]}...'")
            # remote_embed.remote() calls the function on Modal
            result = remote_embed.remote(text)
            # The result from modal might come back as list or numpy array depending on definition
            # Our definition returns a list
            return result
        except ImportError:
             print("Error: Could not import 'embedder'. Ensure modal_infra is accessible.")
             raise HTTPException(status_code=500, detail="Search unavailable: Backend misconfigured (missing embedder).")
        except Exception as e:
            print(f"Error invoking Modal embedder: {e}")
            raise HTTPException(status_code=500, detail=f"Search failed: Remote embedding error ({str(e)})")

app = FastAPI(
    title="ShotSpot API",
    description="Just-in-Time Dataset Factory API",
    docs_url="/docs",
    openapi_url="/openapi.json",
    servers=[{"url": "/api"}],
    root_path="/api"
)

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request/Response models 

class ScrapeRequest(BaseModel):
    urls: List[str]
    output_format: str = "json"

class CreateDatasetRequest(BaseModel):
    prompt: str
    urls: List[str]
    scale: int = 10

class FrameUpload(BaseModel):
    model_config = ConfigDict(extra="allow")
    embedding: Optional[List[float]] = None
    vector: Optional[List[float]] = None
    source: Optional[str] = None
    timestamp: Optional[float] = None
    title: Optional[str] = None

class FrameUploadBulk(BaseModel):
    frames: List[FrameUpload]

class SearchRequest(BaseModel):
    query: str
    top_k: int = 24
    source_url: Optional[str] = None
    allowed_sources: Optional[List[str]] = None

# --- Stats Tracking ---
# Simple in-memory stats for the demo
class GlobalStats:
    total_frames_processed = 0
    total_bytes_processed = 0
    last_check_time = time.time()
    last_frames_count = 0
    last_bytes_count = 0
    current_workers_target = 0
    last_frame_arrival_time = 0

stats_tracker = GlobalStats()


# --- Ingestion Job Tracking ---
# Simple in-memory: { "url": { "total": int, "processed": int, "status": "processing" } }
ingestion_jobs = {}

class IngestionInitRequest(BaseModel):
    url: str
    total_segments: int

# --- Video Discovery ---
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", "")


def is_valid_video_url(url: str) -> bool:
    """
    Validates if a URL is a specific video/clip from Twitch OR YouTube.
    Excludes generic channel pages, directories, or search results.
    """
    if "twitch.tv" in url:
        bad_fragments = ["/clips?", "clips?filter", "/videos?", "videos?filter",
                         "/directory", "/p/", "/login", "/signup", "/downloads"]
        if any(bad in url for bad in bad_fragments):
            return False
        return any(good in url for good in ["/clip/", "/videos/", "/v/"])

    if "youtube.com" in url or "youtu.be" in url:
        if "google.com" in url:
            return False
        bad_fragments = ["/channel/", "/user/", "/c/", "/results",
                         "/feed/", "googleads", "/playlist"]
        if any(bad in url for bad in bad_fragments):
            return False
        return any(good in url for good in ["/watch", "youtu.be/", "/shorts/"])

    return False


def _discover_youtube(query: str, max_results: int = 10) -> List[str]:
    """Search YouTube Data API v3 for videos matching query. Returns watch URLs."""
    if not YOUTUBE_API_KEY:
        print("No YOUTUBE_API_KEY set, skipping YouTube API search.")
        return []

    try:
        resp = requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={
                "part": "snippet",
                "q": query,
                "type": "video",
                "maxResults": max_results,
                "key": YOUTUBE_API_KEY,
            },
            timeout=15,
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])
        urls = [
            f"https://www.youtube.com/watch?v={item['id']['videoId']}"
            for item in items
            if item.get("id", {}).get("videoId")
        ]
        print(f"[YouTube API] Found {len(urls)} videos for '{query}'")
        return urls
    except Exception as e:
        print(f"[YouTube API] Error: {e}")
        return []


def _discover_duckduckgo(query: str, max_results: int = 10) -> List[str]:
    """Use DuckDuckGo (free, no API key) to discover YouTube/Twitch video URLs."""
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        print("duckduckgo-search not installed. Run: pip install duckduckgo-search")
        return []

    discovered: List[str] = []
    search_queries = [
        f'site:twitch.tv/*/clip {query}',
        f'site:twitch.tv/videos {query}',
        f'site:youtube.com/watch {query}',
    ]
    try:
        with DDGS() as ddgs:
            for q in search_queries:
                if len(discovered) >= max_results:
                    break
                results = ddgs.text(q, max_results=5)
                for r in results:
                    url = r.get("href", "")
                    if url and is_valid_video_url(url) and url not in discovered:
                        discovered.append(url)
        print(f"[DuckDuckGo] Found {len(discovered)} URLs for '{query}'")
    except Exception as e:
        print(f"[DuckDuckGo] Error: {e}")
    return discovered


def discover_videos(query: str) -> List[str]:
    """
    Discover relevant YouTube and Twitch video URLs for a query.
    Primary: YouTube Data API v3 (requires YOUTUBE_API_KEY in .env).
    Fallback: DuckDuckGo search (free, no key needed) for Twitch + extra YouTube.
    """
    discovered: List[str] = []

    # 1. YouTube API (structured, reliable)
    yt_urls = _discover_youtube(query, max_results=10)
    discovered.extend(yt_urls)

    # 2. DuckDuckGo fallback — fills in Twitch results and supplements YouTube
    #    if the API key is missing or returned few results
    if len(discovered) < 10:
        ddg_urls = _discover_duckduckgo(query, max_results=10)
        for url in ddg_urls:
            if url not in discovered:
                discovered.append(url)

    discovered = list(dict.fromkeys(discovered))  # deduplicate, preserve order
    print(f"Discovery complete: {len(discovered)} URLs found for '{query}'")
    return discovered[:20]


@app.post("/scrape")
def scrape_endpoint(request: ScrapeRequest):
    """Validate a list of URLs and return which ones are recognised video links."""
    valid = [u for u in request.urls if is_valid_video_url(u)]
    invalid = [u for u in request.urls if not is_valid_video_url(u)]
    return {"valid": valid, "invalid": invalid, "count": len(valid)}

@app.post("/dataset/create")
async def create_dataset_endpoint(request: CreateDatasetRequest):
    print(f"Dataset Creation Request: '{request.prompt}'")
    stats_tracker.current_workers_target = request.scale
    stats_tracker.last_frame_arrival_time = time.time()

    # 1. Determine Source URLs
    urls = request.urls
    discovery_mode = False

    if not urls:
        # User provided NO URLs -> auto-discover via YouTube API + DuckDuckGo
        discovery_mode = True
        print(f"Auto-Discovery Mode: searching for '{request.prompt}'...")
        try:
            urls = discover_videos(request.prompt)
            print(f"Discovery Complete. Found {len(urls)} candidate sources.")
        except Exception as e:
            return {"ok": False, "error": f"Discovery failed: {str(e)}"}

    if not urls:
        return {"ok": False, "error": "No sources found. Try providing URLs manually or check your YOUTUBE_API_KEY."}

    # 2. Trigger Ingestion Pipeline (same logic as Analyze Data)
    results = []
    
    try:
        # Connect to Modal Function
        f = modal.Function.from_name("treehacks-video-ingestor-v2", "ingest_video_orchestrator")
        
        for url in urls:
            url = url.strip()
            if not url: 
                continue
            
            # Constraints: Only run ingestion (CLIP/Modal) on Twitch links
            # YouTube or other links are returned as "discovered" but not processed
            if "twitch.tv" not in url:
                results.append({
                    "url": url,
                    "status": "discovered_only", 
                    "message": "Ingestion skipped (Non-Twitch link)",
                    "job_id": None
                })
                continue
                
            try:
                # Spawn async - this runs the "Analyze Data" pipeline on this URL
                # (Download -> Split -> Frame -> Embed -> Vector DB)
                call = await f.spawn.aio(url)
                
                results.append({
                    "url": url,
                    "status": "started",
                    "job_id": call.object_id
                })
            except Exception as e:
                results.append({
                    "url": url,
                    "status": "error",
                    "error": str(e)
                })
                
    except modal.exception.NotFoundError:
        return {
            "ok": False,
            "error": "Modal App Not Found",
            "message": "Please run `modal deploy modal/ingestor.py`."
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

    return {
        "ok": True, 
        "message": f"Pipeline triggered for {len(results)} sources ({'Auto-Discovered' if discovery_mode else 'Manual'}).",
        "jobs": results,
        "mode": "discovery" if discovery_mode else "manual"
    }

@app.get("/dataset/export")
def export_dataset(query: Optional[str] = None):
    """
    Export dataset as a ZIP file containing JSON metadata.
    """
    try:
        from db import search 
        
        # 1. Fetch data
        if query:
            # Vector search if query provided
            vec = embed_text(query)
            results = search(vec, top_k=100) # Fetch up to 100 relevant
        else:
            # Dump latest 100 if no query
            coll = get_collection()
            cursor = coll.find().sort("_id", -1).limit(100)
            results = list(cursor)

        # 2. Create ZIP in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            
            # Add manifest
            manifest = {
                "export_date": time.time(),
                "query": query,
                "count": len(results),
                "source": "Ingest.ai"
            }
            zip_file.writestr("manifest.json", json.dumps(manifest, indent=2))
            
            # Add data items
            for i, doc in enumerate(results):
                # Clean up ObjectId
                doc["_id"] = str(doc.get("_id"))
                
                # In a real app, we would include the actual image file here
                # zip_file.writestr(f"data/{doc['_id']}.jpg", image_bytes)
                
                # For now, just the metadata
                zip_file.writestr(f"data/{doc['_id']}.json", json.dumps(doc, indent=2))
                
        zip_buffer.seek(0)
        
        filename = f"dataset_{int(time.time())}.zip"
        return StreamingResponse(
            zip_buffer, 
            media_type="application/zip", 
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        raise HTTPException(500, f"Export failed: {str(e)}")

# --- Endpoints ---

@app.get("/")
@app.get("/api")
def root():
    return {"status": "ok", "message": "Ingest.ai API Operational"}

@app.get("/stats")
def stats():
    """Real stats based on ingestion activity"""
    now = time.time()
    delta = now - stats_tracker.last_check_time
    
    # Avoid div/0 if polled too fast
    if delta < 1:
        delta = 1
        
    # Calculate rates
    new_frames = stats_tracker.total_frames_processed - stats_tracker.last_frames_count
    new_bytes = stats_tracker.total_bytes_processed - stats_tracker.last_bytes_count
    
    fps = new_frames / delta
    bandwidth = (new_bytes / 1024 / 1024) / delta # MB/s
    
    # Update "last" values for next poll
    stats_tracker.last_check_time = now
    stats_tracker.last_frames_count = stats_tracker.total_frames_processed
    stats_tracker.last_bytes_count = stats_tracker.total_bytes_processed
    
    # Determine active workers: if we received frames recently (10s), show target. Else 0.
    is_active = (now - stats_tracker.last_frame_arrival_time) < 10
    active = stats_tracker.current_workers_target if is_active else 0
    
    # Get total count from DB
    try:
        coll = get_collection()
        total_count = coll.count_documents({})
    except:
        total_count = 0
    
    return {
        "active_workers": active, 
        "fps_processed": round(fps, 1),
        "bandwidth_mbps": round(bandwidth, 1),
        "total_frames": total_count, 
    }

@app.post("/frames/bulk")
def upload_frames_bulk(body: FrameUploadBulk):
    ids = []
    print(f"Received {len(body.frames)} frames")
    
    # Update Stats
    stats_tracker.total_frames_processed += len(body.frames)
    # Estimate size in bytes (rough estimate: json body size)
    # Each frame has a 512-float vector (512 * 4 bytes = 2KB) plus metadata. Say 3KB per frame.
    stats_tracker.total_bytes_processed += len(body.frames) * 3000 
    stats_tracker.last_frame_arrival_time = time.time()
    
    # Update Per-Job Stats
    if body.frames:
        src = body.frames[0].source
        if src and src in ingestion_jobs:
            job = ingestion_jobs[src]
            job["processed_segments"] += 1
            job["processed_frames"] += len(body.frames)
            job["last_update"] = time.time()
            if job["processed_segments"] >= job["total_segments"]:
                job["status"] = "completed"
            print(f"📊 Job Update [{src}]: {job['processed_segments']}/{job['total_segments']} segments")

    # Need to import insert_frames from db (bulk version)
    try:
        from db import insert_frames
    except ImportError:
        raise HTTPException(status_code=500, detail="Backend misconfigured: db module not found.")
    
    docs_to_insert = []
    for frame in body.frames:
        # Get vector
        vec = frame.vector if frame.vector else frame.embedding
        if not vec:
            continue
            
        # Extract metadata
        meta = frame.model_dump(exclude={"vector", "embedding"})
        meta["embedding"] = vec
        docs_to_insert.append(meta)
        
    # Insert in batch
    try:
        inserted_ids = insert_frames(docs_to_insert)
        ids = [str(x) for x in inserted_ids]
    except Exception as e:
        print(f"Bulk insert failed: {e}")
            
    return {"ok": True, "count": len(ids), "ids": ids}

@app.post("/search")
def search_frames(req: SearchRequest):
    """
    Text-to-Image Search.
    1. Embed query (text) -> vector
    2. Search MongoDB
    """
    print(f"Searching for: {req.query}")
    
    # 1. Embed
    try:
        if not req.query:
             # Return random/latest if empty
             pass 
        vector = embed_text(req.query)
    except Exception as e:
        print(f"Embedding failed: {e}")
        return {"ok": False, "error": str(e)}

    # 2. Search DB
    try:
        # Build Filter Query
        filter_q = {}
        if req.source_url:
            print(f"Filtering by source: {req.source_url}")
            filter_q["source"] = req.source_url
        
        if req.allowed_sources:
            print(f"Filtering by {len(req.allowed_sources)} allowed sources")
            # MongoDB Atlas Search syntax for "IN" is slightly different depending on mapping
            # Standard MQL match uses $in
            filter_q["source"] = {"$in": req.allowed_sources}

        results = search(vector, top_k=req.top_k, filter_query=filter_q)
    except Exception as e:
        error_msg = str(e)
        # Handle missing index configuration gracefully
        if "needs to be indexed as filter" in error_msg:
            print("⚠️ CRITICAL WARNING: MongoDB Atlas Index is missing the 'filter' definition.")
            print("   Falling back to UNFILTERED search so the app works.")
            print("   ACTION REQUIRED: Add {'type': 'filter', 'path': 'source'} to your Atlas Search Index.")
            
            # Fallback: Retry without the filter
            results = search(vector, top_k=req.top_k, filter_query=None)
        else:
            print(f"DB Search failed: {e}")
            return {"ok": False, "error": str(e)}
        
    # 3. Format for Frontend
    formatted = []
    for doc in results:
        # Filter out low relevance scores (Noise reduction)
        # Cosine similarity for CLIP usually effectively ranges 0.2-0.3 for good matches
        score = doc.get("score", 0)
        if score < 0.22: 
            continue
            
        # doc has: embedding, time, title, source, score (from search)
        
        # Calculate pretty timestamp
        ts = doc.get("time", 0)
        minutes = int(ts // 60)
        seconds = int(ts % 60)
        time_str = f"{minutes}m {seconds}s"
        
        # Generate direct link to timestamp
        source = doc.get("source", "")
        link = source
        if "twitch.tv" in source:
             link = f"{source}?t={int(ts)}s"
        elif "youtube.com" in source or "youtu.be" in source:
             link = f"{source}&t={int(ts)}s"
             
        # Use a placeholder or frame extraction service for the actual image
        formatted.append({
            "id": str(doc.get("_id")),
            "url": f"https://source.unsplash.com/random/300x200?sig={random.randint(0,1000)}",
            "timestamp": time_str,
            "timestamp_seconds": int(ts),
            "score": doc.get("score", 0),
            "source_url": link,
            "title": doc.get("title", "Unknown")
        })

    return {"ok": True, "results": formatted}


class GroupTimestampsRequest(BaseModel):
    times: List[float]
    video_length: int


@app.post("/timestamps/group")
def group_timestamps(body: GroupTimestampsRequest):
    """Merge nearby inference timestamps into segments; returns segment start times (for sidebar candidates)."""
    if get_timestamp_segments is None:
        return {"ok": False, "error": "time_stamp_grouping not available", "starts": []}
    try:
        # Clamp times to valid range to avoid index errors
        times = [max(0, min(int(t), body.video_length - 1)) for t in body.times]
        if not times:
            return {"ok": True, "segments": [], "starts": []}
        segments = get_timestamp_segments(times, body.video_length)
        starts = [s for s, _ in segments]
        return {"ok": True, "segments": segments, "starts": starts}
    except Exception as e:
        return {"ok": False, "error": str(e), "starts": []}

@app.post("/ingest/start")
async def start_ingest(source_url: str, prompt: str, scale: int = 10, stealth: bool = False):
    """Trigger the ingestion process via the imported Modal function"""
    
    # Update Stats Target
    stats_tracker.current_workers_target = scale
    stats_tracker.last_frame_arrival_time = time.time() # Reset activity timer

    try:
        # 1. Lookup the DEPLOYED function. 
        # This requires you to run `modal deploy modal/ingestor.py` in your terminal first.
        # The first argument is the App Name (from ingestor.py), the second is the function name.
        f = modal.Function.from_name("treehacks-video-ingestor-v2", "ingest_video_orchestrator")
        
        # 2. Spawn the function execution asynchronously on the cloud
        call = await f.spawn.aio(source_url)
        
        return {
            "ok": True, 
            "job_id": call.object_id,
            "message": f"Started ingestion for '{prompt}' from {source_url} (ID: {call.object_id})"
        }
    except modal.exception.NotFoundError:
        return {
            "ok": False,
            "error": "Modal App Not Found",
            "message": "❌ Please run `modal deploy modal/ingestor.py` in your terminal to deploy the function first."
        }
    except Exception as e:
        env_vars = list(os.environ.keys())
        token_id_ok = "MODAL_TOKEN_ID" in os.environ
        token_secret_ok = "MODAL_TOKEN_SECRET" in os.environ
        print(f"Modal invocation failed: {e}")
        return {
            "ok": False,
            "error": str(e),
            "message": f"Failed to trigger Modal function: {str(e)} \nDEBUG INFO: MODAL_TOKEN_ID set? {token_id_ok}, MODAL_TOKEN_SECRET set? {token_secret_ok}"
        }

@app.post("/ingestion/init")
def init_ingestion(req: IngestionInitRequest):
    print(f"🏁 Ingestion Init: {req.total_segments} segments for {req.url}")
    ingestion_jobs[req.url] = {
        "total_segments": req.total_segments,
        "processed_segments": 0,
        "processed_frames": 0,
        "start_time": time.time(),
        "last_update": time.time(),
        "status": "processing"
    }
    return {"ok": True}

@app.get("/ingestion/status")
def get_ingestion_status(url: str):
    # Try exact match or match ignoring http/https/www variations?
    # For now, simplistic
    job = ingestion_jobs.get(url)
    if not job:
        # Check if we have any job that contains this URL substring (e.g. yt-dlp canonicalization)
        for job_url, j in ingestion_jobs.items():
            if url in job_url or job_url in url:
                job = j
                break
    
    if not job:
        return {"status": "not_found", "progress": 0}
    
    total = job["total_segments"]
    current = job["processed_segments"]
    # Cap progress at 99% until fully complete? Or just 1.0
    progress = min(1.0, current / total) if total > 0 else 0
    
    return {
        "status": job["status"],
        "progress": progress,
        "total_segments": total,
        "processed_segments": current,
        "frames": job["processed_frames"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
