import sqlite3
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def init_db(db_path):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    c    = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS instrument_events (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     REAL,
            frame_id      INTEGER,
            track_id      INTEGER,
            label         TEXT,
            x1 REAL, y1 REAL, x2 REAL, y2 REAL,
            active_frames INTEGER
        )
    """)
    conn.commit()
    conn.close()
    logger.info(f"DB initialised at {db_path}")


def log_event(db_path, timestamp, frame_id, track_id, label, bbox, active_frames):
    x1, y1, x2, y2 = bbox
    conn = sqlite3.connect(db_path, timeout=10)
    conn.execute("""
        INSERT INTO instrument_events
            (timestamp, frame_id, track_id, label, x1, y1, x2, y2, active_frames)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, (timestamp, frame_id, track_id, label, x1, y1, x2, y2, active_frames))
    conn.commit()
    conn.close()


def get_events_df(db_path):
    try:
        conn = sqlite3.connect(db_path, timeout=10)
        df   = pd.read_sql("SELECT * FROM instrument_events", conn)
        conn.close()
        return df
    except Exception as e:
        logger.error(f"DB read error: {e}")
        return pd.DataFrame()


def clear_session(db_path):
    conn = sqlite3.connect(db_path, timeout=10)
    conn.execute("DELETE FROM instrument_events")
    conn.commit()
    conn.close()
    logger.info("Session data cleared.")