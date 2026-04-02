"""
Result re-ranking and explainability.

For each FAISS hit we:
1. Compute a final score (normalized cosine similarity)
2. Extract matching keywords between query and content preview
3. Produce a human-readable explanation
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for",
    "from", "has", "have", "he", "her", "him", "his", "how", "i", "in",
    "is", "it", "its", "me", "my", "not", "of", "on", "or", "our", "over",
    "she", "so", "that", "the", "their", "them", "they", "this", "to",
    "up", "us", "was", "we", "were", "what", "when", "where", "who",
    "will", "with", "you", "your",
}


def _extract_keywords(text: str) -> set[str]:
    words = re.findall(r"[a-z]+", text.lower())
    return {w for w in words if w not in _STOP_WORDS and len(w) > 2}


def _score_to_label(score: float) -> str:
    if score >= 0.90:
        return "Excellent match"
    if score >= 0.75:
        return "Strong match"
    if score >= 0.55:
        return "Good match"
    if score >= 0.35:
        return "Weak match"
    return "Low relevance"


def explain_result(
    query: str,
    result: Dict[str, Any],
    file_record,
) -> Dict[str, Any]:
    """
    Enrich a raw FAISS result dict with explainability fields.

    Parameters
    ----------
    query       : The user's search query
    result      : Dict with keys: file_id, score, modality, rank
    file_record : ORM FileRecord object

    Returns
    -------
    Enriched result dict
    """
    score = result["score"]
    modality = result["modality"]
    content = (file_record.content_preview or "") if file_record else ""

    # Keyword matching
    query_kws = _extract_keywords(query)
    content_kws = _extract_keywords(content)
    matching_keywords = sorted(query_kws & content_kws)

    # Explain why this was returned
    reason_parts = [
        f"Semantic similarity score of {score:.2%} ({_score_to_label(score)}).",
        f"Matched via {modality} embedding.",
    ]
    if matching_keywords:
        reason_parts.append(f"Shared keywords: {', '.join(matching_keywords[:10])}.")

    explanation = " ".join(reason_parts)

    return {
        "file_id": file_record.id if file_record else result["file_id"],
        "filename": file_record.original_filename if file_record else "unknown",
        "file_type": file_record.file_type if file_record else modality,
        "content_preview": content[:300] if content else None,
        "similarity_score": round(score, 4),
        "embedding_distance": round(1.0 - score, 4),
        "matching_modality": modality,
        "matching_keywords": matching_keywords[:10],
        "explanation": explanation,
        "rank": result["rank"],
        "created_at": (
            file_record.created_at.isoformat() if file_record and file_record.created_at else None
        ),
        "file_size": file_record.file_size if file_record else None,
        "mime_type": file_record.mime_type if file_record else None,
    }


def rerank(
    query: str,
    raw_results: List[Dict],
    file_records: Dict[str, Any],
    top_k: int = 10,
) -> List[Dict]:
    """
    Re-rank and enrich raw FAISS results.

    Parameters
    ----------
    query        : User query string
    raw_results  : List of dicts from VectorStore.search()
    file_records : Dict mapping file_id → FileRecord ORM object
    top_k        : Maximum number of results to return

    Returns
    -------
    List of enriched, re-ranked result dicts
    """
    enriched = []
    for r in raw_results:
        file_record = file_records.get(r["file_id"])
        if file_record and file_record.is_deleted:
            continue
        enriched.append(explain_result(query, r, file_record))

    # Re-rank: primary key = similarity_score, secondary = keyword overlap
    enriched.sort(
        key=lambda x: (x["similarity_score"], len(x["matching_keywords"])),
        reverse=True,
    )

    # Update ranks after re-ranking
    for rank, item in enumerate(enriched[:top_k], 1):
        item["rank"] = rank

    return enriched[:top_k]
