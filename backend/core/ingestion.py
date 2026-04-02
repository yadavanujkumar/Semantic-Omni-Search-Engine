"""
Data ingestion pipeline.

Orchestrates:
1. Save uploaded file to disk
2. Generate embeddings
3. Store embedding in FAISS
4. Store metadata in PostgreSQL
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from config import settings
from core.vector_store import get_vector_store
from models.models import FileRecord

logger = logging.getLogger(__name__)

# Supported MIME types mapped to modality
MIME_TO_MODALITY: dict[str, str] = {
    # Text
    "text/plain": "text",
    "text/csv": "text",
    "text/html": "text",
    "text/markdown": "text",
    # PDF
    "application/pdf": "pdf",
    # Images
    "image/jpeg": "image",
    "image/png": "image",
    "image/gif": "image",
    "image/webp": "image",
    "image/bmp": "image",
    "image/tiff": "image",
    # Audio
    "audio/mpeg": "audio",
    "audio/mp3": "audio",
    "audio/wav": "audio",
    "audio/ogg": "audio",
    "audio/flac": "audio",
    "audio/x-wav": "audio",
    "audio/webm": "audio",
    # Video
    "video/mp4": "video",
    "video/mpeg": "video",
    "video/webm": "video",
    "video/quicktime": "video",
    "video/x-msvideo": "video",
    "video/x-matroska": "video",
}

EXT_TO_MODALITY: dict[str, str] = {
    ".txt": "text",
    ".md": "text",
    ".csv": "text",
    ".html": "text",
    ".htm": "text",
    ".pdf": "pdf",
    ".jpg": "image",
    ".jpeg": "image",
    ".png": "image",
    ".gif": "image",
    ".webp": "image",
    ".bmp": "image",
    ".tiff": "image",
    ".tif": "image",
    ".mp3": "audio",
    ".wav": "audio",
    ".ogg": "audio",
    ".flac": "audio",
    ".m4a": "audio",
    ".mp4": "video",
    ".mpeg": "video",
    ".mpg": "video",
    ".webm": "video",
    ".mov": "video",
    ".avi": "video",
    ".mkv": "video",
}


def detect_modality(filename: str, mime_type: Optional[str] = None) -> str:
    """Determine the modality from MIME type or file extension."""
    if mime_type and mime_type in MIME_TO_MODALITY:
        return MIME_TO_MODALITY[mime_type]
    ext = Path(filename).suffix.lower()
    return EXT_TO_MODALITY.get(ext, "text")


async def ingest_file(
    file_content: bytes,
    original_filename: str,
    mime_type: Optional[str],
    db: Session,
) -> FileRecord:
    """
    Full ingestion pipeline for an uploaded file.
    Returns the persisted FileRecord.
    """
    file_id = str(uuid.uuid4())
    modality = detect_modality(original_filename, mime_type)
    ext = Path(original_filename).suffix.lower()
    stored_filename = f"{file_id}{ext}"

    # 1. Persist file to disk
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_path = settings.UPLOAD_DIR / stored_filename
    file_path.write_bytes(file_content)
    logger.info("Saved file %s → %s (%d bytes)", original_filename, file_path, len(file_content))

    # 2. Generate embeddings
    t_emb_start = time.perf_counter()
    try:
        from core.embeddings import embed_file
        embedding, content_preview = embed_file(str(file_path), modality)
    except Exception as exc:
        logger.exception("Embedding failed, cleaning up: %s", exc)
        file_path.unlink(missing_ok=True)
        raise
    emb_time_ms = (time.perf_counter() - t_emb_start) * 1000

    # 3. Store in FAISS
    t_vec_start = time.perf_counter()
    vector_store = get_vector_store()
    row_id = vector_store.add(file_id, modality, embedding)
    vec_time_ms = (time.perf_counter() - t_vec_start) * 1000

    # 4. Store metadata in PostgreSQL
    record = FileRecord(
        id=file_id,
        filename=stored_filename,
        original_filename=original_filename,
        file_type=modality,
        mime_type=mime_type,
        file_size=len(file_content),
        file_path=str(file_path),
        content_preview=content_preview,
        embedding_id=str(row_id),
        metadata_={
            "embedding_time_ms": round(emb_time_ms, 2),
            "vector_time_ms": round(vec_time_ms, 2),
        },
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    logger.info(
        "Ingested %s → file_id=%s modality=%s emb=%.1fms vec=%.1fms",
        original_filename, file_id, modality, emb_time_ms, vec_time_ms,
    )
    return record
