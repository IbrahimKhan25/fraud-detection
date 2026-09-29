"""Postgres connection helpers shared by the API logger, the outcome endpoint
and the monitoring CLI.

DATABASE_URL defaults to the docker-compose Postgres as seen from the host
(port 5433, chosen so it doesn't collide with a Postgres you already run on
5432). Inside compose the API container overrides it to postgres:5432.
"""
import os
from pathlib import Path

import psycopg

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
DEFAULT_DSN = "postgresql://fraud:fraud@localhost:5433/fraud"


def dsn() -> str:
    return os.getenv("DATABASE_URL", DEFAULT_DSN)


def connect(timeout: int = 3) -> psycopg.Connection:
    """New autocommit connection. Short connect timeout so a dead database
    fails fast instead of hanging a request or the logger thread."""
    return psycopg.connect(dsn(), autocommit=True, connect_timeout=timeout)


def ensure_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA_PATH.read_text())
