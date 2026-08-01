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

    # ------------------------------------------------------------------
    # Read side. Used by the chat path, which shares nothing with the
    # real-time path except this database.
    #
    # Timestamps are ISO-8601 UTC strings, which sort lexicographically, so a
    # plain string comparison is a valid time filter. Callers pass datetimes or
    # strings; turning "last week" into one of those belongs in the chat layer,
    # not here.
    # ------------------------------------------------------------------

    @staticmethod
    def _stamp(value):
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.astimezone(timezone.utc).isoformat(timespec="seconds")
        return str(value)

    def _window(self, since, until):
        clauses, params = [], []
        if since is not None:
            clauses.append("ts >= ?")
            params.append(self._stamp(since))
        if until is not None:
            clauses.append("ts < ?")
            params.append(self._stamp(until))
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", params

    def _query(self, sql, params):
        with self._lock:
            return [dict(row) for row in self._db.execute(sql, params).fetchall()]

    def recent(self, limit=20, category=None, since=None, until=None):
        where, params = self._window(since, until)
        if category is not None:
            where = (where + " AND " if where else " WHERE ") + "category = ?"
            params.append(category)
        rows = self._query(
            "SELECT ts, label, confidence, category FROM events"
            f"{where} ORDER BY id DESC LIMIT ?",
            params + [limit],
        )
        return rows

    def counts_by_category(self, since=None, until=None):
        where, params = self._window(since, until)
        rows = self._query(
            f"SELECT category, COUNT(*) AS n FROM events{where}"
            " GROUP BY category ORDER BY n DESC",
            params,
        )
        return {row["category"]: row["n"] for row in rows}

    def counts_by_label(self, since=None, until=None, limit=None):
        where, params = self._window(since, until)
        sql = (
            f"SELECT label, COUNT(*) AS n FROM events{where}"
            " GROUP BY label ORDER BY n DESC"
        )
        if limit is not None:
            sql += " LIMIT ?"
            params = params + [limit]
        return {row["label"]: row["n"] for row in self._query(sql, params)}

    def total(self, since=None, until=None):
        where, params = self._window(since, until)
        rows = self._query(f"SELECT COUNT(*) AS n FROM events{where}", params)
        return rows[0]["n"] if rows else 0

    def last_seen(self, label):
        """When this label was last recorded, or None. Answers 'have I thrown
        out a bottle recently'."""
        rows = self._query(
            "SELECT ts FROM events WHERE label = ? ORDER BY id DESC LIMIT 1",
            [label],
        )
        return rows[0]["ts"] if rows else None

    def span(self):
        """Oldest and newest timestamps, so answers can say what period they
        cover instead of implying the log is complete."""
        rows = self._query("SELECT MIN(ts) AS first, MAX(ts) AS last FROM events", [])
        return (rows[0]["first"], rows[0]["last"]) if rows else (None, None)
