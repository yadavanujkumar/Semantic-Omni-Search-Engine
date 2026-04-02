"""
Files endpoints:
  GET    /files          – list all indexed files
  DELETE /file/{file_id} – soft-delete a file
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from core.vector_store import get_vector_store
from db.database import get_db
from models.models import FileRecord

router = APIRouter(tags=["files"])
logger = logging.getLogger(__name__)


@router.get("/files")
async def list_files(
    modality: Optional[str] = Query(None, description="Filter by modality"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """List all non-deleted indexed files."""
    q = db.query(FileRecord).filter(FileRecord.is_deleted == 0)
    if modality:
        q = q.filter(FileRecord.file_type == modality)
    total = q.count()
    records = q.order_by(FileRecord.created_at.desc()).offset(offset).limit(limit).all()

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "files": [
            {
                "file_id": r.id,
                "filename": r.original_filename,
                "file_type": r.file_type,
                "mime_type": r.mime_type,
                "file_size": r.file_size,
                "content_preview": (r.content_preview or "")[:200],
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in records
        ],
    }


@router.delete("/file/{file_id}", status_code=status.HTTP_200_OK)
async def delete_file(
    file_id: str,
    db: Session = Depends(get_db),
):
    """Soft-delete a file and remove it from the vector index."""
    record = (
        db.query(FileRecord)
        .filter(FileRecord.id == file_id, FileRecord.is_deleted == 0)
        .first()
    )
    if not record:
        raise HTTPException(status_code=404, detail="File not found")

    # Remove from FAISS index
    vector_store = get_vector_store()
    vector_store.delete(file_id, record.file_type)

    # Soft-delete from DB
    record.is_deleted = 1
    db.commit()

    # Optionally remove physical file
    try:
        Path(record.file_path).unlink(missing_ok=True)
    except Exception as exc:
        logger.warning("Could not remove file %s: %s", record.file_path, exc)

    logger.info("Deleted file_id=%s (%s)", file_id, record.original_filename)
    return {"message": f"File '{record.original_filename}' deleted successfully."}
