"""Database connection and session management."""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy.pool import NullPool
from config import settings


engine = create_engine(
    str(settings.DATABASE_URL),
    poolclass=NullPool,
    echo=settings.DEBUG,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    """Dependency for FastAPI routes to get a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables in the database."""
    from models.models import FileRecord, SearchHistory  # noqa: F401
    Base.metadata.create_all(bind=engine)
