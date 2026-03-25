from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from .config import DB_URL

_SCHEMA_PATH = Path(__file__).parent.parent / "sql" / "schema.sql"


def get_engine() -> Engine:
    return create_engine(DB_URL)


def create_table(engine: Engine) -> None:
    ddl = _SCHEMA_PATH.read_text()
    with engine.connect() as conn:
        conn.execute(text(ddl))
        conn.commit()
    print("Table nfl_prospects created (or already exists).")


def upsert_prospects(df: pd.DataFrame, engine: Engine) -> int:
    """
    Upsert rows into nfl_prospects using pfr_id as the conflict key.
    Returns the number of rows processed.
    """
    if df.empty:
        return 0

    records = df.where(pd.notna(df), other=None).to_dict(orient="records")
    columns = [c for c in df.columns if c != "id"]

    col_list = ", ".join(columns)
    val_list = ", ".join(f":{c}" for c in columns)
    update_set = ", ".join(
        f"{c} = EXCLUDED.{c}"
        for c in columns
        if c not in ("pfr_id", "created_at")
    )

    stmt = text(
        f"""
        INSERT INTO nfl_prospects ({col_list})
        VALUES ({val_list})
        ON CONFLICT (pfr_id) DO UPDATE
        SET {update_set}, updated_at = NOW()
        """
    )

    with engine.connect() as conn:
        conn.execute(stmt, records)
        conn.commit()

    return len(records)
