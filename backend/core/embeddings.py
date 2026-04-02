"""
Embedding Engine – generates vector embeddings for all modalities.

Modalities:
- text  → Sentence Transformers (all-MiniLM-L6-v2)
- image → CLIP (openai/clip-vit-base-patch32)
- audio → Whisper transcription → text embedding
- video → frame sampling → CLIP embeddings averaged
- pdf   → pdfplumber text extraction → text embedding
"""
from __future__ import annotations

import io
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# ── Constants ──
MAX_PREVIEW_LENGTH = 2000       # max chars of extracted text to store as preview
MAX_EMBEDDING_TEXT_LENGTH = 512  # max chars fed to the embedding model (token budget)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy singletons – models are loaded on first use to avoid startup delays
# ---------------------------------------------------------------------------
_text_model = None
_clip_model = None
_clip_processor = None
_whisper_model = None


def _get_text_model():
    global _text_model
    if _text_model is None:
        from sentence_transformers import SentenceTransformer
        from config import settings
        logger.info("Loading text embedding model: %s", settings.TEXT_MODEL)
        _text_model = SentenceTransformer(settings.TEXT_MODEL)
    return _text_model


def _get_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        from transformers import CLIPModel, CLIPProcessor
        from config import settings
        logger.info("Loading CLIP model: %s", settings.CLIP_MODEL)
        _clip_processor = CLIPProcessor.from_pretrained(settings.CLIP_MODEL)
        _clip_model = CLIPModel.from_pretrained(settings.CLIP_MODEL)
    return _clip_model, _clip_processor


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        from config import settings
        logger.info("Loading Whisper model: %s", settings.WHISPER_MODEL)
        _whisper_model = whisper.load_model(settings.WHISPER_MODEL)
    return _whisper_model


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def embed_text(text: str) -> np.ndarray:
    """Return a unit-normalised text embedding."""
    model = _get_text_model()
    vec = model.encode([text], normalize_embeddings=True)[0]
    return vec.astype(np.float32)


def embed_image(image_source) -> np.ndarray:
    """
    Return a unit-normalised CLIP image embedding.
    image_source: PIL.Image, bytes, or file-like object.
    """
    import torch

    clip_model, clip_processor = _get_clip()

    if isinstance(image_source, (bytes, bytearray)):
        image = Image.open(io.BytesIO(image_source)).convert("RGB")
    elif isinstance(image_source, (str, Path)):
        image = Image.open(image_source).convert("RGB")
    elif hasattr(image_source, "read"):
        image = Image.open(image_source).convert("RGB")
    else:
        image = image_source.convert("RGB")

    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
    vec = features[0].cpu().numpy()
    vec = vec / (np.linalg.norm(vec) + 1e-9)
    return vec.astype(np.float32)


def embed_text_with_clip(text: str) -> np.ndarray:
    """Return a CLIP text embedding (for cross-modal image search)."""
    import torch

    clip_model, clip_processor = _get_clip()
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)
    vec = features[0].cpu().numpy()
    vec = vec / (np.linalg.norm(vec) + 1e-9)
    return vec.astype(np.float32)


def embed_audio(file_path: str) -> tuple[np.ndarray, str]:
    """
    Transcribe audio with Whisper, then embed the transcript.
    Returns (embedding, transcript).
    """
    model = _get_whisper()
    result = model.transcribe(file_path)
    transcript = result.get("text", "").strip()
    if not transcript:
        transcript = "[no speech detected]"
    embedding = embed_text(transcript)
    return embedding, transcript


def embed_pdf(file_path: str) -> tuple[np.ndarray, str]:
    """
    Extract text from PDF, then embed it.
    Returns (embedding, extracted_text_preview).
    """
    import pdfplumber

    text_chunks: list[str] = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                text_chunks.append(page_text.strip())

    full_text = "\n".join(text_chunks)
    if not full_text.strip():
        full_text = "[no text extracted from PDF]"

    # Truncate to avoid model token limits
    preview = full_text[:MAX_PREVIEW_LENGTH]
    embedding = embed_text(full_text[:MAX_EMBEDDING_TEXT_LENGTH])
    return embedding, preview


def embed_video(file_path: str, num_frames: int = 8) -> tuple[np.ndarray, str]:
    """
    Sample frames from a video, embed each with CLIP, average the vectors.
    Returns (averaged_embedding, description).
    """
    import cv2

    cap = cv2.VideoCapture(str(file_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    duration_s = total_frames / fps if fps else 0

    if total_frames == 0:
        cap.release()
        raise ValueError(f"Cannot read frames from video: {file_path}")

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    embeddings: list[np.ndarray] = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        # Convert BGR → RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        vec = embed_image(pil_image)
        embeddings.append(vec)

    cap.release()

    if not embeddings:
        raise ValueError("No frames could be extracted from video")

    avg_vec = np.mean(embeddings, axis=0)
    avg_vec = avg_vec / (np.linalg.norm(avg_vec) + 1e-9)
    description = f"Video ({duration_s:.1f}s, {num_frames} frames sampled)"
    return avg_vec.astype(np.float32), description


def embed_file(file_path: str, file_type: str) -> tuple[np.ndarray, str]:
    """
    Dispatch to the correct embedding function based on file_type.
    Returns (embedding, content_preview).
    """
    t0 = time.perf_counter()
    try:
        if file_type == "text":
            text = Path(file_path).read_text(errors="replace")
            emb = embed_text(text[:MAX_EMBEDDING_TEXT_LENGTH])
            preview = text[:MAX_PREVIEW_LENGTH]
        elif file_type == "image":
            emb = embed_image(file_path)
            preview = f"Image: {Path(file_path).name}"
        elif file_type == "audio":
            emb, preview = embed_audio(file_path)
            preview = f"Transcript: {preview[:500]}"
        elif file_type == "pdf":
            emb, preview = embed_pdf(file_path)
        elif file_type == "video":
            emb, preview = embed_video(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("Embedded %s (%s) in %.1f ms", file_path, file_type, elapsed)
        return emb, preview
    except Exception as exc:
        logger.exception("Embedding failed for %s: %s", file_path, exc)
        raise


def embed_query(query_text: str, modality: str = "text") -> np.ndarray:
    """
    Embed a search query.  When modality is 'image', use CLIP text encoder
    so the query can be compared against image embeddings.
    """
    if modality == "image":
        return embed_text_with_clip(query_text)
    return embed_text(query_text)
