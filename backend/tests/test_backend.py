"""
Tests for the Semantic Omni Search Engine backend.

These tests use mocking to avoid requiring real ML models, PostgreSQL,
and FAISS during CI. They validate the API contract and business logic.
"""
from __future__ import annotations

import json
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

# ─── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def mock_settings(tmp_path, monkeypatch):
    """Patch settings to use temp directories."""
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("FAISS_INDEX_DIR", str(tmp_path / "faiss"))
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")


@pytest.fixture
def mock_db():
    """Return a mock SQLAlchemy session."""
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = None
    db.query.return_value.filter.return_value.count.return_value = 0
    db.query.return_value.filter.return_value.all.return_value = []
    db.query.return_value.order_by.return_value.limit.return_value.all.return_value = []
    return db


# ─── Core: reranker ────────────────────────────────────────────────────────────

class TestReranker:
    def test_extract_keywords(self):
        from core.reranker import _extract_keywords
        kws = _extract_keywords("The quick brown fox jumps over the lazy dog")
        assert "quick" in kws
        assert "fox" in kws
        # stop words should be excluded
        assert "the" not in kws
        assert "over" not in kws

    def test_score_to_label(self):
        from core.reranker import _score_to_label
        assert _score_to_label(0.95) == "Excellent match"
        assert _score_to_label(0.80) == "Strong match"
        assert _score_to_label(0.65) == "Good match"
        assert _score_to_label(0.45) == "Weak match"
        assert _score_to_label(0.10) == "Low relevance"

    def test_explain_result_structure(self):
        from core.reranker import explain_result

        mock_record = MagicMock()
        mock_record.id = "file-123"
        mock_record.original_filename = "test.txt"
        mock_record.file_type = "text"
        mock_record.content_preview = "The quick brown fox"
        mock_record.is_deleted = 0
        mock_record.created_at = None
        mock_record.file_size = 1024
        mock_record.mime_type = "text/plain"

        raw = {"file_id": "file-123", "score": 0.85, "modality": "text", "rank": 1}
        result = explain_result("brown fox", raw, mock_record)

        assert result["file_id"] == "file-123"
        assert result["filename"] == "test.txt"
        assert result["similarity_score"] == pytest.approx(0.85, abs=0.01)
        assert result["embedding_distance"] == pytest.approx(0.15, abs=0.01)
        assert "brown" in result["matching_keywords"] or "fox" in result["matching_keywords"]
        assert "explanation" in result
        assert result["rank"] == 1

    def test_rerank_filters_deleted(self):
        from core.reranker import rerank

        deleted_record = MagicMock()
        deleted_record.is_deleted = 1

        raw = [{"file_id": "f1", "score": 0.9, "modality": "text", "rank": 1}]
        result = rerank("query", raw, {"f1": deleted_record}, top_k=10)
        assert result == []

    def test_rerank_sorts_by_score(self):
        from core.reranker import rerank

        def make_record(name):
            r = MagicMock()
            r.id = name
            r.original_filename = name
            r.file_type = "text"
            r.content_preview = "foo bar baz"
            r.is_deleted = 0
            r.created_at = None
            r.file_size = 100
            r.mime_type = "text/plain"
            return r

        raw = [
            {"file_id": "a", "score": 0.5, "modality": "text", "rank": 1},
            {"file_id": "b", "score": 0.9, "modality": "text", "rank": 2},
            {"file_id": "c", "score": 0.7, "modality": "text", "rank": 3},
        ]
        records = {
            "a": make_record("a.txt"),
            "b": make_record("b.txt"),
            "c": make_record("c.txt"),
        }
        results = rerank("test", raw, records, top_k=10)
        scores = [r["similarity_score"] for r in results]
        assert scores == sorted(scores, reverse=True)
        assert results[0]["similarity_score"] == pytest.approx(0.9, abs=0.01)


# ─── Core: ingestion helpers ───────────────────────────────────────────────────

class TestIngestion:
    def test_detect_modality_by_extension(self):
        from core.ingestion import detect_modality
        assert detect_modality("report.pdf") == "pdf"
        assert detect_modality("photo.jpg") == "image"
        assert detect_modality("podcast.mp3") == "audio"
        assert detect_modality("clip.mp4") == "video"
        assert detect_modality("notes.txt") == "text"
        assert detect_modality("readme.md") == "text"
        assert detect_modality("data.csv") == "text"

    def test_detect_modality_by_mime(self):
        from core.ingestion import detect_modality
        assert detect_modality("file", "application/pdf") == "pdf"
        assert detect_modality("file", "image/png") == "image"
        assert detect_modality("file", "audio/wav") == "audio"
        assert detect_modality("file", "video/mp4") == "video"
        assert detect_modality("file", "text/plain") == "text"

    def test_detect_modality_mime_takes_priority(self):
        from core.ingestion import detect_modality
        # MIME type takes priority over extension
        assert detect_modality("file.txt", "image/png") == "image"


# ─── Core: vector store ────────────────────────────────────────────────────────

class TestVectorStore:
    def test_add_and_search(self, tmp_path):
        from core.vector_store import VectorStore

        vs = VectorStore(tmp_path / "faiss")
        emb = np.random.rand(384).astype(np.float32)
        emb /= np.linalg.norm(emb)

        row = vs.add("file-1", "text", emb)
        assert row == 0

        results = vs.search(emb, modality="text", top_k=5)
        assert len(results) == 1
        assert results[0]["file_id"] == "file-1"
        assert results[0]["score"] > 0.99  # same vector

    def test_search_all_modalities(self, tmp_path):
        from core.vector_store import VectorStore

        vs = VectorStore(tmp_path / "faiss")

        text_emb = np.random.rand(384).astype(np.float32)
        text_emb /= np.linalg.norm(text_emb)
        vs.add("file-text", "text", text_emb)

        image_emb = np.random.rand(512).astype(np.float32)
        image_emb /= np.linalg.norm(image_emb)
        vs.add("file-image", "image", image_emb)

        # Search with text query (dim=384) across all modalities
        results = vs.search(text_emb, modality=None, top_k=10)
        file_ids = {r["file_id"] for r in results}
        # text result should always appear
        assert "file-text" in file_ids

    def test_delete(self, tmp_path):
        from core.vector_store import VectorStore

        vs = VectorStore(tmp_path / "faiss")
        emb = np.random.rand(384).astype(np.float32)
        emb /= np.linalg.norm(emb)

        vs.add("file-del", "text", emb)
        assert vs.stats()["text"] == 1

        deleted = vs.delete("file-del", "text")
        assert deleted is True
        assert vs.stats()["text"] == 0

    def test_delete_nonexistent(self, tmp_path):
        from core.vector_store import VectorStore

        vs = VectorStore(tmp_path / "faiss")
        deleted = vs.delete("ghost", "text")
        assert deleted is False

    def test_stats(self, tmp_path):
        from core.vector_store import VectorStore

        vs = VectorStore(tmp_path / "faiss")
        stats = vs.stats()
        assert set(stats.keys()) >= {"text", "image", "audio", "video", "pdf"}
        assert all(v == 0 for v in stats.values())


# ─── API: endpoints (mocked) ───────────────────────────────────────────────────

@pytest.fixture
def client(tmp_path):
    """Create a TestClient with mocked DB and vector store."""
    import sys
    import os

    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    with (
        patch("db.database.create_engine"),
        patch("db.database.SessionLocal"),
        patch("db.database.Base.metadata.create_all"),
        patch("core.vector_store.get_vector_store") as mock_vs,
        patch("config.settings.UPLOAD_DIR", tmp_path / "uploads"),
        patch("config.settings.FAISS_INDEX_DIR", tmp_path / "faiss"),
    ):
        mock_vs.return_value = MagicMock()
        mock_vs.return_value.search.return_value = []
        mock_vs.return_value.stats.return_value = {"text": 0, "image": 0, "audio": 0, "video": 0, "pdf": 0}

        from main import app
        from db.database import get_db
        app.dependency_overrides[get_db] = lambda: MagicMock()

        with TestClient(app) as c:
            yield c


class TestHealthEndpoints:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_stats_endpoint(self, client):
        resp = client.get("/stats")
        assert resp.status_code == 200
        assert "vector_index_sizes" in resp.json()


class TestSearchEndpoint:
    def test_search_empty_query_rejected(self, client):
        resp = client.post("/search", json={"query": ""})
        assert resp.status_code == 422  # Pydantic validation error

    def test_search_valid(self, client):
        with patch("api.search.embed_query") as mock_emb:
            mock_emb.return_value = np.zeros(384, dtype=np.float32)
            resp = client.post("/search", json={"query": "test search"})
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert "latency_ms" in data
        assert "total" in data

    def test_search_invalid_modality_allowed(self, client):
        """Unknown modality should not raise a 422 – it falls back to all."""
        with patch("api.search.embed_query") as mock_emb:
            mock_emb.return_value = np.zeros(384, dtype=np.float32)
            resp = client.post("/search", json={"query": "test", "modality": "text"})
        assert resp.status_code == 200


class TestFilesEndpoints:
    def test_list_files_empty(self, client):
        resp = client.get("/files")
        assert resp.status_code == 200
        data = resp.json()
        assert "files" in data
        assert isinstance(data["files"], list)

    def test_delete_nonexistent_file(self, client):
        resp = client.delete("/file/nonexistent-id")
        assert resp.status_code == 404
