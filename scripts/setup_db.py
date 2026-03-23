"""
Initialize the PostgreSQL database and tables for stocknet.
Run once before training:  python scripts/setup_db.py
Reads connection settings from .env at the project root.
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import psycopg2

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

HOST = os.environ.get("PGHOST", "localhost")
PORT = int(os.environ.get("PGPORT", 5432))
USER = os.environ.get("PGUSER", "postgres")
PASSWORD = os.environ.get("PGPASSWORD", "")
DBNAME = os.environ.get("PGDATABASE", "stocknet")


def create_database():
    conn = psycopg2.connect(host=HOST, port=PORT, user=USER, password=PASSWORD, dbname="postgres")
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DBNAME,))
        if cur.fetchone():
            print(f"Database '{DBNAME}' already exists.")
        else:
            cur.execute(f'CREATE DATABASE "{DBNAME}"')
            print(f"Database '{DBNAME}' created.")
    conn.close()


def create_tables():
    conn = psycopg2.connect(host=HOST, port=PORT, user=USER, password=PASSWORD, dbname=DBNAME)
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS training_runs (
                id         SERIAL PRIMARY KEY,
                model_name VARCHAR(255) NOT NULL,
                version    VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT NOW(),
                params     JSONB,
                metadata   JSONB,
                UNIQUE(model_name, version)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS training_logs (
                id           SERIAL PRIMARY KEY,
                run_id       INTEGER REFERENCES training_runs(id) ON DELETE CASCADE,
                epoch        INTEGER NOT NULL,
                timestamp    TIMESTAMP NOT NULL,
                train_loss   DOUBLE PRECISION NOT NULL,
                val_loss     DOUBLE PRECISION NOT NULL,
                elapsed_time DOUBLE PRECISION
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_training_logs_run_id
            ON training_logs(run_id)
        """)
    conn.commit()
    conn.close()
    print("Tables ready.")


if __name__ == "__main__":
    try:
        create_database()
        create_tables()
        print("Done.")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
