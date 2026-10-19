"""Ingest a short benchmark video locally without paid GPU infrastructure."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from db import get_collection

MODEL_ID = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
WHISPER_ID = "openai/whisper-tiny"
OCR_LANGUAGES = ["en"]


def extract_ocr_text(reader, image, min_confidence: float = 0.35) -> str:
    """Return confidence-filtered OCR text in reading order for one frame."""
    if reader is None:
        return ""
    results = reader.readtext(np.asarray(image), detail=1, paragraph=False)
    parts = [
        str(text).strip()
        for _box, text, confidence in results
        if str(text).strip() and float(confidence) >= min_confidence
    ]
    return " ".join(parts)


def download_video(url: str, directory: Path) -> tuple[Path, str, float]:
    import yt_dlp

    output = directory / "source.%(ext)s"
    options = {
        "format": "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
        "outtmpl": str(output),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(options) as downloader:
        info = downloader.extract_info(url, download=True)
    files = [path for path in directory.iterdir() if path.is_file()]
    if not files:
        raise RuntimeError("yt-dlp completed without creating a media file")
    media_path = max(files, key=lambda path: path.stat().st_size)
    return media_path, str(info.get("title", "Unknown")), float(info.get("duration") or 0)


def extract_frames(media_path: Path, interval: int, duration: float):
    from PIL import Image

    # Preserve source resolution for OCR. CLIPProcessor performs its own resize
    # later; resizing here would make small interface text unreadable.
    with tempfile.TemporaryDirectory(prefix="shotspot-frames-") as frame_dir:
        output_pattern = str(Path(frame_dir) / "frame-%06d.jpg")
        command = [
            "ffmpeg",
            "-i",
            str(media_path),
            "-vf",
            f"fps=1/{interval}",
            "-q:v",
            "2",
            "-loglevel",
            "error",
            output_pattern,
        ]
        result = subprocess.run(command, capture_output=True, check=False)
        if result.returncode:
            raise RuntimeError(
                f"ffmpeg frame extraction failed: {result.stderr.decode()[-500:]}"
            )
        frames = []
        for index, frame_path in enumerate(sorted(Path(frame_dir).glob("frame-*.jpg"))):
            timestamp = min(index * interval, max(duration - 0.001, 0))
            with Image.open(frame_path) as image:
                frames.append((timestamp, image.convert("RGB").copy()))
        return frames


def extract_audio(media_path: Path, start: float, length: float) -> np.ndarray:
    command = [
        "ffmpeg",
        "-ss",
        str(start),
        "-t",
        str(length),
        "-i",
        str(media_path),
        "-f",
        "s16le",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        "-loglevel",
        "error",
        "-",
    ]
    result = subprocess.run(command, capture_output=True, check=True)
    return np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def ingest(
    url: str,
    interval: int,
    segment_duration: int,
    max_duration: int,
    keep_existing: bool,
    enable_ocr: bool = True,
    ocr_min_confidence: float = 0.35,
) -> int:
    import torch
    from transformers import CLIPModel, CLIPProcessor, pipeline

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading CLIP on {device} and Whisper on CPU...")
    clip_model = CLIPModel.from_pretrained(MODEL_ID).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(MODEL_ID)
    transcriber = pipeline("automatic-speech-recognition", model=WHISPER_ID, device=-1)
    ocr_reader = None
    if enable_ocr:
        try:
            import certifi
            import easyocr
        except ImportError as exc:
            raise RuntimeError(
                "OCR ingestion requires easyocr; install requirements-local.txt"
            ) from exc
        print("Loading EasyOCR on CPU...")
        # Python framework installs on macOS may not inherit the system keychain.
        # Point urllib (used by EasyOCR's first-run model download) at certifi.
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        ocr_reader = easyocr.Reader(OCR_LANGUAGES, gpu=False, verbose=False)

    with tempfile.TemporaryDirectory(prefix="shotspot-") as temporary:
        media_path, title, duration = download_video(url, Path(temporary))
        if duration <= 0:
            raise RuntimeError("The source did not report a usable duration")
        if duration > max_duration:
            raise ValueError(
                f"Video is {duration:.0f}s; choose a video under {max_duration}s or increase --max-duration"
            )
        print(f"Downloaded: {title} ({duration:.1f}s)")
        frames = extract_frames(media_path, interval, duration)
        if not frames:
            raise RuntimeError("No frames were extracted")

        documents = []
        for segment_start in range(0, int(duration) + 1, segment_duration):
            segment_end = min(segment_start + segment_duration, duration)
            segment_frames = [
                (timestamp, image)
                for timestamp, image in frames
                if segment_start <= timestamp < segment_end
            ]
            if not segment_frames:
                continue

            audio = extract_audio(media_path, segment_start, segment_end - segment_start)
            transcript = ""
            if audio.size:
                transcript = str(
                    transcriber({"raw": audio, "sampling_rate": 16000}).get("text", "")
                ).strip()

            image_inputs = clip_processor(
                images=[image for _, image in segment_frames], return_tensors="pt", padding=True
            ).to(device)
            with torch.no_grad():
                visual = clip_model.get_image_features(**image_inputs)
                visual = visual / visual.norm(p=2, dim=-1, keepdim=True)

            ocr_texts = [
                extract_ocr_text(ocr_reader, image, ocr_min_confidence)
                for _, image in segment_frames
            ]
            ocr_vectors: list[object | None] = [None] * len(ocr_texts)
            nonempty_ocr = [(index, text) for index, text in enumerate(ocr_texts) if text]
            if nonempty_ocr:
                ocr_inputs = clip_processor(
                    text=[text for _, text in nonempty_ocr],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(device)
                with torch.no_grad():
                    encoded_ocr = clip_model.get_text_features(**ocr_inputs)
                    encoded_ocr = encoded_ocr / encoded_ocr.norm(p=2, dim=-1, keepdim=True)
                for (index, _), vector in zip(nonempty_ocr, encoded_ocr):
                    ocr_vectors[index] = vector

            transcript_vector = None
            if transcript:
                text_inputs = clip_processor(
                    text=[transcript], return_tensors="pt", padding=True, truncation=True
                ).to(device)
                with torch.no_grad():
                    transcript_vector = clip_model.get_text_features(**text_inputs)[0]
                    transcript_vector = transcript_vector / transcript_vector.norm(p=2)

            for frame_index, ((timestamp, _), visual_vector) in enumerate(
                zip(segment_frames, visual)
            ):
                available_vectors = [visual_vector]
                if transcript_vector is not None:
                    available_vectors.append(transcript_vector)
                ocr_vector = ocr_vectors[frame_index]
                if ocr_vector is not None:
                    available_vectors.append(ocr_vector)
                fused = torch.stack(available_vectors).sum(dim=0)
                fused = fused / fused.norm(p=2)
                documents.append(
                    {
                        "source": url,
                        "title": title,
                        "time": timestamp,
                        "text_context": transcript,
                        "embedding": fused.detach().cpu().tolist(),
                        "visual_embedding": visual_vector.detach().cpu().tolist(),
                        "transcript_embedding": (
                            transcript_vector.detach().cpu().tolist()
                            if transcript_vector is not None
                            else None
                        ),
                        "ocr_text": ocr_texts[frame_index],
                        "ocr_embedding": (
                            ocr_vector.detach().cpu().tolist()
                            if ocr_vector is not None
                            else None
                        ),
                        "embedding_model": MODEL_ID,
                        "transcription_model": WHISPER_ID,
                        "ocr_model": "easyocr-en" if enable_ocr else None,
                        "fusion_strategy": "normalized_sum_available_modalities",
                        "ingestion_backend": "local",
                    }
                )
            print(
                f"Processed {segment_start:.0f}-{segment_end:.0f}s: "
                f"{len(segment_frames)} frames, {len(transcript)} transcript characters, "
                f"{sum(bool(text) for text in ocr_texts)} OCR-positive frames"
            )

    collection = get_collection()
    if not keep_existing:
        collection.delete_many({"source": url})
    result = collection.insert_many(documents)
    print(f"Inserted {len(result.inserted_ids)} frames into {collection.database.name}.{collection.name}")
    return len(result.inserted_ids)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--segment-duration", type=int, default=15)
    parser.add_argument("--max-duration", type=int, default=600)
    parser.add_argument("--keep-existing", action="store_true")
    parser.add_argument(
        "--ocr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Extract and embed visible frame text (default: enabled)",
    )
    parser.add_argument("--ocr-min-confidence", type=float, default=0.35)
    args = parser.parse_args()
    if args.interval < 1 or args.segment_duration < 1 or args.max_duration < 1:
        parser.error("duration values must be positive")
    if not 0 <= args.ocr_min_confidence <= 1:
        parser.error("--ocr-min-confidence must be between 0 and 1")

    load_dotenv(PROJECT_ROOT / ".env")
    if not os.environ.get("MONGODB_URI"):
        parser.error("MONGODB_URI is not configured in .env")
    ingest(
        args.url,
        args.interval,
        args.segment_duration,
        args.max_duration,
        args.keep_existing,
        enable_ocr=args.ocr,
        ocr_min_confidence=args.ocr_min_confidence,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
