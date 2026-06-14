"""Engine and session helpers for the index database."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.index.models import Base


def make_engine(url: str) -> Engine:
    """Create a SQLAlchemy engine. ``pool_pre_ping`` survives dropped connections."""
    return create_engine(url, future=True, pool_pre_ping=True)


def create_all(engine: Engine) -> None:
    """Create all index tables if they do not exist (Phase 1 schema bootstrap)."""
    Base.metadata.create_all(engine)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Return a session factory bound to ``engine``."""
    return sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def session_scope(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    """Transactional session scope: commit on success, roll back on error."""
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
