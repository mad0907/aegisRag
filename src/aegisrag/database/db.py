from contextlib import contextmanager
from pathlib import Path

import psycopg

from aegisrag.config.settings import get_settings

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


@contextmanager
def get_connection():
    settings = get_settings()
    conn = psycopg.connect(settings.postgres_dsn_psycopg)
    try:
        yield conn
    finally:
        conn.close()


def init_schema() -> None:
    ddl = SCHEMA_PATH.read_text()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()


if __name__ == "__main__":
    init_schema()
    print("Schema initialized.")
