"""Shared pytest fixtures: ephemeral Postgres via testcontainers, FastAPI
test client with rollback-per-test isolation.
"""
from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from testcontainers.postgres import PostgresContainer

import src.config
from src.app import app
from src.db import Base, get_db


@pytest.fixture(scope="session")
def postgres_container() -> Iterator[PostgresContainer]:
    with PostgresContainer("postgres:17-alpine") as pg:
        yield pg


@pytest.fixture(scope="session")
def db_url(postgres_container: PostgresContainer) -> str:
    url = postgres_container.get_connection_url().replace("postgresql+psycopg2://", "postgresql+psycopg://")
    return url


@pytest.fixture(scope="session", autouse=True)
def _patch_settings(db_url: str, monkeypatch_session) -> None:
    monkeypatch_session.setattr(src.config.Settings.model_config, "env_file", None)
    monkeypatch_session.setenv("DATABASE_URL", db_url)
    src.config.get_settings.cache_clear()


@pytest.fixture(scope="session")
def engine(db_url: str):
    engine = create_engine(db_url, future=True)
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def db(engine):
    """Per-test session that rolls back at the end so tests are isolated."""
    connection = engine.connect()
    transaction = connection.begin()
    TestSession = sessionmaker(bind=connection, autoflush=False, autocommit=False)
    session = TestSession()

    def _get_db_override():
        try:
            yield session
        finally:
            pass

    app.dependency_overrides[get_db] = _get_db_override
    yield session
    session.close()
    transaction.rollback()
    connection.close()
    app.dependency_overrides.pop(get_db, None)


@pytest.fixture
def client(db) -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def monkeypatch_session():
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    yield mp
    mp.undo()
