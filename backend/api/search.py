"""
Search endpoint – POST /search
GET  /search/history
"""
from __future__ import annotations

import logging
import time
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from config import settings
from core.embeddings import embed_query
from core.reranker import rerank
from core.vector_store import get_vector_store
from db.database import get_db
from models.models import FileRecord, SearchHistory

router = APIRouter(prefix="/search", tags=["search"])
logger = logging.getLogger(__name__)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1024)
    modality: Optional[str] = Field(
        None,
        description="Filter by modality: text, image, audio, video, pdf. Omit to search all.",
    )
    top_k: int = Field(settings.TOP_K, ge=1, le=100)
    min_score: float = Field(settings.SIMILARITY_THRESHOLD, ge=0.0, le=1.0)


class SearchResult(BaseModel):
    file_id: str
    filename: str
    file_type: str
    content_preview: Optional[str]
    similarity_score: float
    embedding_distance: float
    matching_modality: str
    matching_keywords: List[str]
    explanation: str
    rank: int
    created_at: Optional[str]
    file_size: Optional[int]
    mime_type: Optional[str]


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total: int
    latency_ms: float
    embedding_time_ms: float
    retrieval_time_ms: float


@router.post("", response_model=SearchResponse)
async def search(
    req: SearchRequest,
    db: Session = Depends(get_db),
):
    """Semantic search across all indexed content."""
    t_total = time.perf_counter()

    # 1. Embed the query
    t_emb = time.perf_counter()
    try:
        query_emb = embed_query(req.query, modality=req.modality or "text")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query embedding failed: {exc}",
        )
    emb_ms = (time.perf_counter() - t_emb) * 1000

    # 2. Vector similarity search
    t_ret = time.perf_counter()
    vector_store = get_vector_store()
    raw_results = vector_store.search(
        query_embedding=query_emb,
        modality=req.modality,
        top_k=req.top_k * 2,  # over-fetch for re-ranking
    )
    ret_ms = (time.perf_counter() - t_ret) * 1000

    # 3. Fetch file metadata from DB
    file_ids = [r["file_id"] for r in raw_results]
    records = (
        db.query(FileRecord)
        .filter(FileRecord.id.in_(file_ids), FileRecord.is_deleted == 0)
        .all()
    )
    record_map = {r.id: r for r in records}

    # 4. Re-rank + explainability
    enriched = rerank(
        query=req.query,
        raw_results=raw_results,
        file_records=record_map,
        top_k=req.top_k,
    )

    # 5. Apply minimum score filter
    enriched = [r for r in enriched if r["similarity_score"] >= req.min_score]

    total_ms = (time.perf_counter() - t_total) * 1000

    # 6. Persist search history
    history = SearchHistory(
        query=req.query,
        query_type=req.modality or "all",
        filters={"modality": req.modality, "top_k": req.top_k, "min_score": req.min_score},
        result_count=len(enriched),
        latency_ms=round(total_ms, 2),
        embedding_time_ms=round(emb_ms, 2),
        retrieval_time_ms=round(ret_ms, 2),
    )
    db.add(history)
    db.commit()

    logger.info(
        "Search query='%s' modality=%s → %d results in %.1fms",
        req.query, req.modality, len(enriched), total_ms,
    )

    return SearchResponse(
        query=req.query,
        results=enriched,
        total=len(enriched),
        latency_ms=round(total_ms, 2),
        embedding_time_ms=round(emb_ms, 2),
        retrieval_time_ms=round(ret_ms, 2),
    )


@router.get("/history")
async def search_history(
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Return recent search history."""
    records = (
        db.query(SearchHistory)
        .order_by(SearchHistory.created_at.desc())
        .limit(limit)
        .all()
    )
    return {
        "history": [
            {
                "id": r.id,
                "query": r.query,
                "query_type": r.query_type,
                "result_count": r.result_count,
                "latency_ms": r.latency_ms,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in records
        ]
    }
