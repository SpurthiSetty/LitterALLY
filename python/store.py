import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         TEXT    NOT NULL,
    label      TEXT    NOT NULL,
    confidence REAL    NOT NULL,
    category   TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS events_ts ON events (ts);
"""


def default_path():
    # Overridable because the container filesystem may not survive a redeploy.
    return Path(os.environ.get("SMARTBIN_DB", Path(__file__).with_name("smartbin.db")))


class EventStore:
    def __init__(self, path=None):
        path = Path(path or default_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.executescript(_SCHEMA)
            self._db.commit()

    def log(self, label, confidence, category):
        with self._lock:
            self._db.execute(
                "INSERT INTO events (ts, label, confidence, category) VALUES (?, ?, ?, ?)",
                (
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    label,
                    float(confidence),
                    category,
                ),
            )
            self._db.commit()

    def recent(self, limit=20):
        with self._lock:
            rows = self._db.execute(
                "SELECT ts, label, confidence, category FROM events"
                " ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def counts_by_category(self):
        with self._lock:
            rows = self._db.execute(
                "SELECT category, COUNT(*) AS n FROM events GROUP BY category"
                " ORDER BY n DESC"
            ).fetchall()
        return {row["category"]: row["n"] for row in rows}
