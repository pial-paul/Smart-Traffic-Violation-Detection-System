"""Atomic clip ingestion, indexed queues and single-transition review records."""

import hashlib
import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY, camera TEXT NOT NULL, source_sha256 TEXT NOT NULL,
    provenance TEXT NOT NULL, summary TEXT NOT NULL, UNIQUE(camera, source_sha256)
);
CREATE TABLE IF NOT EXISTS cases (
    id INTEGER PRIMARY KEY, run_id INTEGER NOT NULL REFERENCES runs(id), frame INTEGER NOT NULL,
    track_id INTEGER NOT NULL, occurred_at TEXT NOT NULL, plate TEXT,
    details TEXT NOT NULL, evidence_sha256 TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','issued','dismissed')),
    reviewed_at TEXT, note TEXT, confirmed_plate TEXT, challan TEXT UNIQUE,
    UNIQUE(run_id, frame, track_id)
);
CREATE INDEX IF NOT EXISTS cases_status_id ON cases(status, id DESC);
CREATE TABLE IF NOT EXISTS evidence (
    case_id INTEGER PRIMARY KEY REFERENCES cases(id) ON DELETE CASCADE, jpeg BLOB NOT NULL
);
"""

SELECT = """SELECT c.*, r.camera, r.source_sha256, r.provenance
            FROM cases c JOIN runs r ON r.id=c.run_id"""


class Conflict(Exception):
    """A clip or case has already been processed."""


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(str(self.path), timeout=15)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA foreign_keys=ON")
            with db:
                yield db
        finally:
            db.close()

    def existing(self, camera, source):
        with self.connect() as db:
            row = db.execute(
                "SELECT id,provenance,summary FROM runs WHERE camera=? AND source_sha256=?",
                (camera, source),
            ).fetchone()
            if row:
                return row["id"], json.loads(row["provenance"]), json.loads(row["summary"])

    def save(self, camera, source, provenance, summary, records):
        with self.connect() as db:
            cursor = db.execute(
                """INSERT INTO runs(camera,source_sha256,provenance,summary)
                VALUES(?,?,?,?) ON CONFLICT(camera,source_sha256) DO NOTHING""",
                (camera, source, json.dumps(provenance), json.dumps(summary)),
            )
            if cursor.rowcount == 0:
                raise Conflict("Recording already analyzed for this camera")
            run = cursor.lastrowid
            for event, jpeg in records:
                values = [event[k] for k in ("frame", "track_id", "occurred_at", "plate")]
                case = db.execute(
                    """INSERT INTO cases(run_id,frame,track_id,occurred_at,plate,
                    details,evidence_sha256) VALUES(?,?,?,?,?,?,?)""",
                    [run, *values, json.dumps(event), hashlib.sha256(jpeg).hexdigest()],
                )
                db.execute("INSERT INTO evidence VALUES(?,?)", (case.lastrowid, jpeg))
            return run

    @staticmethod
    def convert(row):
        fields = dict(row)
        fields.update(json.loads(fields.pop("details")))
        fields["provenance"] = json.loads(fields["provenance"])
        return fields

    def cases(self, status=None, before=None, limit=25):
        conditions, args = [], []
        if status:
            conditions.append("c.status=?")
            args.append(status)
        if before is not None:
            conditions.append("c.id<?")
            args.append(before)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        with self.connect() as db:
            rows = db.execute(
                SELECT + where + " ORDER BY c.id DESC LIMIT ?", [*args, limit + 1]
            ).fetchall()
        items = [self.convert(r) for r in rows[:limit]]
        return items, items[-1]["id"] if len(rows) > limit else None

    def get(self, case_id):
        with self.connect() as db:
            row = db.execute(SELECT + " WHERE c.id=?", (case_id,)).fetchone()
            return self.convert(row) if row else None

    def jpeg(self, case_id):
        with self.connect() as db:
            row = db.execute("SELECT jpeg FROM evidence WHERE case_id=?", (case_id,)).fetchone()
            return row[0] if row else None

    def counts(self):
        counts = dict.fromkeys(("pending", "issued", "dismissed"), 0)
        with self.connect() as db:
            counts.update(dict(db.execute("SELECT status,COUNT(*) FROM cases GROUP BY status")))
        return counts

    def review(self, case_id, action, plate, note):
        if action not in ("issue", "dismiss"):
            raise ValueError("Choose issue or dismiss")
        if not isinstance(note, str) or not 1 <= len(note.strip()) <= 500:
            raise ValueError("Provide a review note of 1 to 500 characters")
        confirmed = None
        if action == "issue":
            if not isinstance(plate, str) or not re.fullmatch(r"[A-Za-z0-9]{4,12}", plate.strip()):
                raise ValueError("Confirm a plate of 4 to 12 Latin letters or digits")
            confirmed = plate.strip().upper()
        issued = action == "issue"
        with self.connect() as db:
            result = db.execute(
                """UPDATE cases SET status=?,reviewed_at=?,note=?,confirmed_plate=?,challan=?
                WHERE id=? AND status='pending'""",
                (
                    "issued" if issued else "dismissed",
                    datetime.now(timezone.utc).isoformat(),
                    note.strip(),
                    confirmed,
                    f"CH-{case_id:08d}" if issued else None,
                    case_id,
                ),
            )
            if not result.rowcount:
                if db.execute("SELECT 1 FROM cases WHERE id=?", (case_id,)).fetchone() is None:
                    raise KeyError(case_id)
                raise Conflict("Case already reviewed")
