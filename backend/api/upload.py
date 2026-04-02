"""
Upload endpoint – POST /upload
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from config import settings
from core.ingestion import ingest_file
from db.database import get_db

router = APIRouter(prefix="/upload", tags=["upload"])
logger = logging.getLogger(__name__)

MAX_BYTES = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024


@router.post("", status_code=status.HTTP_201_CREATED)
async def upload_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload a file and trigger the ingestion pipeline."""
    t0 = time.perf_counter()

    # Size check (read in chunks)
    content = await file.read()
    if len(content) > MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum allowed size is {settings.MAX_UPLOAD_SIZE_MB} MB.",
        )

    try:
        record = await ingest_file(
            file_content=content,
            original_filename=file.filename or "unknown",
            mime_type=file.content_type,
            db=db,
        )
    except Exception as exc:
        logger.exception("Ingestion error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        )

    latency_ms = (time.perf_counter() - t0) * 1000
    return {
        "file_id": record.id,
        "filename": record.original_filename,
        "file_type": record.file_type,
        "file_size": record.file_size,
        "content_preview": (record.content_preview or "")[:300],
        "latency_ms": round(latency_ms, 2),
        "message": "File uploaded and indexed successfully.",
    }
