#!/usr/bin/env python3
"""
Run a standalone market simulation and store raw tick data in PostgreSQL.

Usage:
    python scripts/run_simulation.py <datasource_json_path> <simulation_id> [--length N]

Arguments:
    datasource_json_path  Path to a registered simulator datasource JSON file.
    simulation_id         Unique identifier for this run (stored as simulation_id in DB).
    --length N            Number of candles to generate (optional).
                          If omitted, runs continuously in batches of 1000
                          candles until killed.

The raw tick data is inserted into the simulation_ticks table.  The web layer
rolls it up to OHLC candles at any granularity on demand.
"""

import argparse
import datetime
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import execute_values

# ── Project paths ─────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _set_low_priority() -> None:
    """Run at below-normal CPU priority so the web server stays responsive."""
    try:
        if os.name == "nt":
            import ctypes
            # BELOW_NORMAL_PRIORITY_CLASS = 0x4000
            ctypes.windll.kernel32.SetPriorityClass(
                ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000
            )
        else:
            os.nice(10)
    except Exception:
        pass  # best-effort — never fail the simulation over a priority hint

# Load simulator module directly to avoid triggering stocknet/__init__.py
# which imports torch (and all nets), unnecessary for simulation.
_sim_path = PROJECT_ROOT / "stocknet" / "datasets" / "simulator.py"
_spec = importlib.util.spec_from_file_location("stocknet.datasets.simulator", _sim_path)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]
DeterministicDealerModelV3 = _mod.DeterministicDealerModelV3

# ── Constants ─────────────────────────────────────────────────────────────────

RULE_SECONDS: dict[str, float] = {
    "min": 60, "T": 60,
    "5min": 300, "5T": 300,
    "15min": 900, "15T": 900,
    "30min": 1800, "30T": 1800,
    "H": 3600,
    "D": 86400,
}

DEFAULT_CANDLES = 1000
TICKS_PER_CANDLE = 100


# ── Database ──────────────────────────────────────────────────────────────────

def _get_connection():
    return psycopg2.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=int(os.environ.get("PGPORT", 5432)),
        database=os.environ.get("PGDATABASE", "stocknet"),
        user=os.environ.get("PGUSER", "postgres"),
        password=os.environ.get("PGPASSWORD", ""),
    )


def _ensure_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS simulation_ticks (
                id            BIGSERIAL PRIMARY KEY,
                datasource    TEXT              NOT NULL,
                simulation_id TEXT              NOT NULL,
                ts            TIMESTAMPTZ       NOT NULL,
                price         DOUBLE PRECISION  NOT NULL
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_sim_ticks_lookup
                ON simulation_ticks (datasource, simulation_id, ts)
        """)
    conn.commit()


def _get_resume_dt(
    conn,
    datasource: str,
    simulation_id: str,
    default_dt: datetime.datetime,
) -> datetime.datetime:
    """Return the timestamp just after the last recorded tick, or default_dt if none."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT MAX(ts) FROM simulation_ticks WHERE datasource = %s AND simulation_id = %s",
            (datasource, simulation_id),
        )
        row = cur.fetchone()
    if row and row[0] is not None:
        last_ts = row[0]
        # Make naive if tz-aware (DB stores TIMESTAMPTZ)
        if hasattr(last_ts, "tzinfo") and last_ts.tzinfo is not None:
            last_ts = last_ts.replace(tzinfo=None)
        resume_dt = last_ts + datetime.timedelta(seconds=1)
        print(f"Resuming from {resume_dt.isoformat()} (last tick: {last_ts.isoformat()})", flush=True)
        return resume_dt
    return default_dt


def _insert_ticks(
    conn,
    datasource: str,
    simulation_id: str,
    tick_series: "pd.Series",
) -> int:
    """Bulk-insert raw tick data into simulation_ticks."""
    rows = [
        (datasource, simulation_id, ts.to_pydatetime(), float(price))
        for ts, price in tick_series.items()
    ]
    if not rows:
        return 0
    with conn.cursor() as cur:
        execute_values(
            cur,
            "INSERT INTO simulation_ticks (datasource, simulation_id, ts, price) VALUES %s",
            rows,
            page_size=10_000,
        )
    conn.commit()
    return len(rows)


# ── Simulation helpers ────────────────────────────────────────────────────────

def _model_params(ds_config: dict) -> tuple:
    model_config: dict = ds_config.get("modelConfig", {})
    agent_per_model: int = int(ds_config.get("agentPerModel", 300))
    sampler_rule: str = ds_config.get("samplerRule", "min")
    secs_per_candle = RULE_SECONDS.get(sampler_rule, 60)
    tick_time = secs_per_candle / TICKS_PER_CANDLE
    return model_config, agent_per_model, sampler_rule, secs_per_candle, tick_time


def _simulate_batch_ticks(
    model_config: dict,
    agent_per_model: int,
    sampler_rule: str,
    secs_per_candle: float,
    tick_time: float,
    target_candles: int,
    start_dt: datetime.datetime,
) -> "pd.Series":
    """Run one simulation batch and return a raw tick Series (datetime index, price values)."""
    total_seconds = int(target_candles * secs_per_candle * 1.2)
    model = DeterministicDealerModelV3(
        num_agent=agent_per_model, tick_time=tick_time, **model_config
    )
    model.reset()
    prices, ticks = model.simulate(total_seconds=total_seconds)
    dt_index = pd.to_datetime(
        [start_dt + datetime.timedelta(seconds=float(t)) for t in ticks]
    )
    tick_series = pd.Series(prices["price"].values, index=dt_index)
    # Trim to exactly target_candles worth of simulated time
    end_dt = start_dt + datetime.timedelta(seconds=target_candles * secs_per_candle)
    return tick_series[tick_series.index <= end_dt]


# ── Public entry point ────────────────────────────────────────────────────────

def run(
    ds_config: dict,
    datasource_name: str,
    simulation_id: str,
    data_length: int | None = None,
) -> int:
    model_config, agent_per_model, sampler_rule, secs_per_candle, tick_time = (
        _model_params(ds_config)
    )

    conn = _get_connection()
    _ensure_schema(conn)

    _BASE_DT = datetime.datetime(2020, 1, 1)
    resume_dt = _get_resume_dt(conn, datasource_name, simulation_id, _BASE_DT)

    if data_length is None:
        _run_continuous(
            conn, datasource_name, simulation_id,
            model_config, agent_per_model, sampler_rule, secs_per_candle, tick_time,
            start_dt=resume_dt,
        )
        conn.close()
        return -1  # never reached normally

    target_candles = data_length
    total_seconds = int(target_candles * secs_per_candle * 1.2)

    print(
        f"Simulator config: agents={agent_per_model}, rule={sampler_rule}, "
        f"target={target_candles} candles (~{total_seconds}s simulated, tick={tick_time:.3f}s)",
        flush=True,
    )

    tick_series = _simulate_batch_ticks(
        model_config, agent_per_model, sampler_rule, secs_per_candle, tick_time,
        target_candles, resume_dt,
    )

    n_ticks = _insert_ticks(conn, datasource_name, simulation_id, tick_series)
    conn.close()

    # Approximate candle count for reporting
    candle_count = len(tick_series.resample(sampler_rule).last().dropna())
    print(
        f"Done. Inserted {n_ticks:,} ticks (~{candle_count:,} candles) "
        f"→ datasource={datasource_name!r}, simulation_id={simulation_id!r}",
        flush=True,
    )
    return candle_count


def _run_continuous(
    conn,
    datasource_name: str,
    simulation_id: str,
    model_config: dict,
    agent_per_model: int,
    sampler_rule: str,
    secs_per_candle: float,
    tick_time: float,
    start_dt: datetime.datetime = datetime.datetime(2020, 1, 1),
) -> None:
    """Generate ticks in batches, inserting into PG indefinitely until killed."""
    print(
        f"Simulator config: agents={agent_per_model}, rule={sampler_rule}, "
        f"continuous mode (no length limit), tick={tick_time:.3f}s",
        flush=True,
    )

    first_batch = True
    total_ticks = 0
    batch_num = 0
    current_dt = start_dt

    try:
        while True:
            batch_num += 1
            print(f"Batch {batch_num}: simulating {DEFAULT_CANDLES} candles…", flush=True)

            tick_series = _simulate_batch_ticks(
                model_config, agent_per_model, sampler_rule, secs_per_candle, tick_time,
                DEFAULT_CANDLES, current_dt,
            )

            n = _insert_ticks(conn, datasource_name, simulation_id, tick_series)
            first_batch = False
            total_ticks += n

            if len(tick_series) > 0:
                current_dt = tick_series.index[-1].to_pydatetime() + datetime.timedelta(
                    seconds=secs_per_candle
                )

            print(
                f"Batch {batch_num}: inserted {n:,} ticks (total {total_ticks:,}) "
                f"→ simulation_id={simulation_id!r}",
                flush=True,
            )
    except KeyboardInterrupt:
        print(
            f"\nStopped after {batch_num} batch(es), {total_ticks:,} ticks total.",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config_path", help="Path to simulator datasource JSON")
    parser.add_argument(
        "simulation_id",
        help="Unique run identifier (stored as simulation_id in the DB)",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=None,
        help=f"Target candle count (default: continuous)",
    )
    args = parser.parse_args()
    _set_low_priority()

    with open(args.config_path, "r", encoding="utf-8") as f:
        ds = json.load(f)

    # Derive datasource name from the config file stem
    datasource_name = Path(args.config_path).stem

    run(ds, datasource_name, args.simulation_id, data_length=args.length)


if __name__ == "__main__":
    main()
