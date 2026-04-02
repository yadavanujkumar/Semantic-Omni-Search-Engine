"""SQLAlchemy ORM models."""
import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, DateTime, Text, JSON
from sqlalchemy import func
from db.database import Base


def generate_uuid():
    return str(uuid.uuid4())


class FileRecord(Base):
    __tablename__ = "file_records"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    filename = Column(String(512), nullable=False)
    original_filename = Column(String(512), nullable=False)
    file_type = Column(String(50), nullable=False)   # text, image, audio, video, pdf
    mime_type = Column(String(100), nullable=True)
    file_size = Column(Integer, nullable=False)       # bytes
    file_path = Column(String(1024), nullable=False)
    content_preview = Column(Text, nullable=True)     # extracted text preview
    embedding_id = Column(String(36), nullable=True)  # FAISS index id
    metadata_ = Column("metadata", JSON, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    is_deleted = Column(Integer, default=0)           # soft delete flag


class SearchHistory(Base):
    __tablename__ = "search_history"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    query = Column(Text, nullable=False)
    query_type = Column(String(50), nullable=False)   # text, image, audio
    filters = Column(JSON, nullable=True)
    result_count = Column(Integer, default=0)
    latency_ms = Column(Float, nullable=True)
    embedding_time_ms = Column(Float, nullable=True)
    retrieval_time_ms = Column(Float, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
