"""
FAISS Vector Store – manages separate per-modality indexes plus a unified index.

Index layout on disk (FAISS_INDEX_DIR):
  unified.index   – all embeddings
  unified.meta    – JSON list of {id, file_id, modality}
  text.index / text.meta
  image.index / image.meta
  audio.index / audio.meta
  video.index / video.meta
  pdf.index / pdf.meta
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_DIMENSION_MAP: Dict[str, int] = {
    "text": 384,    # all-MiniLM-L6-v2
    "image": 512,   # CLIP ViT-B/32
    "audio": 384,   # whisper → text → sentence-transformers
    "video": 512,   # CLIP frames averaged
    "pdf": 384,     # sentence-transformers on extracted text
    "unified": 384, # we store a projected/padded version for the unified index
}

_MODALITIES = ("text", "image", "audio", "video", "pdf")


class VectorStore:
    """Thread-safe FAISS-backed vector store."""

    def __init__(self, index_dir: Path):
        import faiss  # noqa: F401

        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()

        # Per-modality indexes
        self._indexes: Dict[str, Any] = {}
        self._metas: Dict[str, List[Dict]] = {}

        for modality in _MODALITIES:
            dim = _DIMENSION_MAP[modality]
            self._indexes[modality] = self._load_or_create(modality, dim)
            self._metas[modality] = self._load_meta(modality)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _index_path(self, name: str) -> Path:
        return self.index_dir / f"{name}.index"

    def _meta_path(self, name: str) -> Path:
        return self.index_dir / f"{name}.meta.json"

    def _load_or_create(self, name: str, dim: int):
        import faiss

        path = self._index_path(name)
        if path.exists():
            try:
                idx = faiss.read_index(str(path))
                logger.info("Loaded FAISS index '%s' (%d vectors)", name, idx.ntotal)
                return idx
            except Exception as exc:
                logger.warning("Could not load index %s: %s. Creating new.", path, exc)

        idx = faiss.IndexFlatIP(dim)  # Inner-product (cosine on unit vecs)
        logger.info("Created new FAISS index '%s' dim=%d", name, dim)
        return idx

    def _load_meta(self, name: str) -> List[Dict]:
        path = self._meta_path(name)
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception:
                pass
        return []

    def _save(self, name: str):
        import faiss

        faiss.write_index(self._indexes[name], str(self._index_path(name)))
        self._meta_path(name).write_text(json.dumps(self._metas[name]))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, file_id: str, modality: str, embedding: np.ndarray) -> int:
        """
        Add an embedding to the modality-specific index.
        Returns the FAISS row index assigned.
        """
        if modality not in _MODALITIES:
            raise ValueError(f"Unknown modality: {modality}")

        vec = embedding.astype(np.float32).reshape(1, -1)
        with self._lock:
            idx = self._indexes[modality]
            row_id = idx.ntotal
            idx.add(vec)
            self._metas[modality].append({"row_id": row_id, "file_id": file_id})
            self._save(modality)

        logger.debug("Added %s to index '%s' → row %d", file_id, modality, row_id)
        return row_id

    def search(
        self,
        query_embedding: np.ndarray,
        modality: Optional[str],
        top_k: int = 10,
    ) -> List[Dict]:
        """
        Search across the requested modality (or all modalities if None).
        Returns list of {file_id, score, modality, rank} sorted by score desc.
        """
        query_embedding = query_embedding.astype(np.float32)

        modalities_to_search = [modality] if modality and modality in _MODALITIES else list(_MODALITIES)

        results: List[Dict] = []
        with self._lock:
            for mod in modalities_to_search:
                idx = self._indexes[mod]
                if idx.ntotal == 0:
                    continue

                q_dim = query_embedding.shape[0]
                idx_dim = idx.d

                if q_dim != idx_dim:
                    # Pad or truncate query to match index dimension
                    if q_dim < idx_dim:
                        padded = np.zeros(idx_dim, dtype=np.float32)
                        padded[:q_dim] = query_embedding
                        q = padded.reshape(1, -1)
                    else:
                        q = query_embedding[:idx_dim].reshape(1, -1)
                else:
                    q = query_embedding.reshape(1, -1)

                k = min(top_k, idx.ntotal)
                scores, indices = idx.search(q, k)

                for score, row_id in zip(scores[0], indices[0]):
                    if row_id < 0:
                        continue
                    meta = self._metas[mod]
                    entry = next((m for m in meta if m["row_id"] == row_id), None)
                    if entry:
                        results.append({
                            "file_id": entry["file_id"],
                            "score": float(score),
                            "modality": mod,
                        })

        # Sort by score descending, deduplicate by file_id (keep highest score)
        seen: Dict[str, Dict] = {}
        for r in results:
            fid = r["file_id"]
            if fid not in seen or r["score"] > seen[fid]["score"]:
                seen[fid] = r

        ranked = sorted(seen.values(), key=lambda x: x["score"], reverse=True)[:top_k]
        for rank, r in enumerate(ranked, 1):
            r["rank"] = rank
        return ranked

    def delete(self, file_id: str, modality: str) -> bool:
        """
        Remove a vector by file_id from the given modality index.
        FAISS FlatIP does not support in-place removal; we rebuild the index
        without the deleted vector.
        """
        if modality not in _MODALITIES:
            return False

        with self._lock:
            import faiss

            old_metas = self._metas[modality]
            new_metas = [m for m in old_metas if m["file_id"] != file_id]
            if len(new_metas) == len(old_metas):
                return False  # not found

            dim = _DIMENSION_MAP[modality]
            old_idx = self._indexes[modality]

            if new_metas:
                # Extract remaining vectors
                row_ids = [m["row_id"] for m in new_metas]
                vecs = old_idx.reconstruct_batch(row_ids)
                new_idx = faiss.IndexFlatIP(dim)
                new_idx.add(vecs)
                # Remap row_ids
                for i, m in enumerate(new_metas):
                    m["row_id"] = i
            else:
                new_idx = faiss.IndexFlatIP(dim)

            self._indexes[modality] = new_idx
            self._metas[modality] = new_metas
            self._save(modality)

        logger.info("Deleted file_id=%s from '%s' index", file_id, modality)
        return True

    def stats(self) -> Dict[str, int]:
        """Return vector count per modality."""
        with self._lock:
            return {mod: self._indexes[mod].ntotal for mod in _MODALITIES}


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        from config import settings
        _store = VectorStore(settings.FAISS_INDEX_DIR)
    return _store
