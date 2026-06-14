"""Shared test fixtures.

Database tests run against a real Postgres in an ephemeral container
(testcontainers), so they need a running Docker daemon. The session-scoped engine
creates the schema once; each test gets a session wrapped in a transaction that is
rolled back, keeping tests isolated and fast.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.index.db import create_all, make_engine, make_session_factory
from app.index.models import Base


@pytest.fixture(scope="session")
def pg_engine() -> Iterator[Engine]:
    # Skip the resource-reaper sidecar; the context manager tears the container down.
    os.environ.setdefault("TESTCONTAINERS_RYUK_DISABLED", "true")
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer("postgres:16", driver="psycopg") as postgres:
        engine = make_engine(postgres.get_connection_url())
        create_all(engine)
        try:
            yield engine
        finally:
            engine.dispose()


@pytest.fixture
def db_session(pg_engine: Engine) -> Iterator[Session]:
    connection = pg_engine.connect()
    transaction = connection.begin()
    factory = sessionmaker(bind=connection, expire_on_commit=False)
    session = factory()
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture
def pg_session_factory(pg_engine: Engine) -> Iterator[sessionmaker[Session]]:
    """A real committing session factory for end-to-end script tests.

    Unlike ``db_session`` (which rolls back), scripts open and commit their own
    sessions, so this fixture commits and then truncates all tables afterward to
    keep tests isolated.
    """
    try:
        yield make_session_factory(pg_engine)
    finally:
        with pg_engine.begin() as connection:
            for table in reversed(Base.metadata.sorted_tables):
                connection.execute(table.delete())
