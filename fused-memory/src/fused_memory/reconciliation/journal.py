"""SQLite-backed journal for reconciliation audit trail and watermark tracking."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

from fused_memory.models.reconciliation import (
    JournalEntry,
    JudgeVerdict,
    ReconciliationRun,
    StageId,
    StageReport,
    Watermark,
)

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS watermarks (
    project_id TEXT PRIMARY KEY,
    last_full_run_id TEXT,
    last_full_run_completed TEXT,
    last_episode_timestamp TEXT,
    last_memory_timestamp TEXT,
    last_task_change_timestamp TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    run_type TEXT NOT NULL,
    trigger_reason TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    events_processed INTEGER DEFAULT 0,
    stage_reports TEXT DEFAULT '{}',
    status TEXT DEFAULT 'running'
);
CREATE INDEX IF NOT EXISTS idx_runs_project ON runs(project_id);
CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at);

CREATE TABLE IF NOT EXISTS journal_entries (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    stage TEXT,
    timestamp TEXT NOT NULL,
    operation TEXT NOT NULL,
    target_system TEXT NOT NULL,
    before_state TEXT,
    after_state TEXT,
    reasoning TEXT DEFAULT '',
    evidence TEXT DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_journal_run ON journal_entries(run_id);

CREATE TABLE IF NOT EXISTS judge_verdicts (
    run_id TEXT PRIMARY KEY,
    reviewed_at TEXT NOT NULL,
    severity TEXT NOT NULL,
    findings TEXT DEFAULT '[]',
    action_taken TEXT DEFAULT 'none'
);
CREATE INDEX IF NOT EXISTS idx_verdicts_reviewed ON judge_verdicts(reviewed_at);
"""


class ReconciliationJournal:
    """Persistent journal backed by SQLite — one database per project directory."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        db_path = self.data_dir / 'reconciliation.db'
        self._db = await aiosqlite.connect(str(db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(SCHEMA_SQL)
        await self._db.commit()
        logger.info(f'Reconciliation journal initialized at {db_path}')

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    def _require_db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError('Journal not initialized — call initialize() first')
        return self._db

    # ── Watermark ──────────────────────────────────────────────────────

    async def get_watermark(self, project_id: str) -> Watermark:
        db = self._require_db()
        async with db.execute(
            'SELECT * FROM watermarks WHERE project_id = ?', (project_id,)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return Watermark(project_id=project_id)
        return Watermark(
            project_id=row['project_id'],
            last_full_run_id=row['last_full_run_id'],
            last_full_run_completed=_parse_dt(row['last_full_run_completed']),
            last_episode_timestamp=_parse_dt(row['last_episode_timestamp']),
            last_memory_timestamp=_parse_dt(row['last_memory_timestamp']),
            last_task_change_timestamp=_parse_dt(row['last_task_change_timestamp']),
        )

    async def update_watermark(self, watermark: Watermark) -> None:
        db = self._require_db()
        await db.execute(
            """INSERT INTO watermarks
               (project_id, last_full_run_id, last_full_run_completed,
                last_episode_timestamp, last_memory_timestamp, last_task_change_timestamp)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(project_id) DO UPDATE SET
                 last_full_run_id = excluded.last_full_run_id,
                 last_full_run_completed = excluded.last_full_run_completed,
                 last_episode_timestamp = excluded.last_episode_timestamp,
                 last_memory_timestamp = excluded.last_memory_timestamp,
                 last_task_change_timestamp = excluded.last_task_change_timestamp
            """,
            (
                watermark.project_id,
                watermark.last_full_run_id,
                _fmt_dt(watermark.last_full_run_completed),
                _fmt_dt(watermark.last_episode_timestamp),
                _fmt_dt(watermark.last_memory_timestamp),
                _fmt_dt(watermark.last_task_change_timestamp),
            ),
        )
        await db.commit()

    # ── Runs ───────────────────────────────────────────────────────────

    async def start_run(self, run: ReconciliationRun) -> None:
        db = self._require_db()
        await db.execute(
            """INSERT INTO runs
               (id, project_id, run_type, trigger_reason, started_at,
                events_processed, stage_reports, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run.id,
                run.project_id,
                run.run_type,
                run.trigger_reason,
                run.started_at.isoformat(),
                run.events_processed,
                json.dumps({}),
                run.status,
            ),
        )
        await db.commit()

    async def complete_run(self, run_id: str, status: str) -> None:
        db = self._require_db()
        await db.execute(
            'UPDATE runs SET status = ?, completed_at = ? WHERE id = ?',
            (status, datetime.now(timezone.utc).isoformat(), run_id),
        )
        await db.commit()

    async def update_run_stage_reports(
        self, run_id: str, stage_reports: dict[str, StageReport]
    ) -> None:
        db = self._require_db()
        serialized = {k: v.model_dump(mode='json') for k, v in stage_reports.items()}
        await db.execute(
            'UPDATE runs SET stage_reports = ? WHERE id = ?',
            (json.dumps(serialized), run_id),
        )
        await db.commit()

    async def get_run(self, run_id: str) -> ReconciliationRun | None:
        db = self._require_db()
        async with db.execute('SELECT * FROM runs WHERE id = ?', (run_id,)) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        reports_raw = json.loads(row['stage_reports'] or '{}')
        stage_reports = {}
        for k, v in reports_raw.items():
            stage_reports[k] = StageReport(**v)
        return ReconciliationRun(
            id=row['id'],
            project_id=row['project_id'],
            run_type=row['run_type'],
            trigger_reason=row['trigger_reason'],
            started_at=datetime.fromisoformat(row['started_at']),
            completed_at=_parse_dt(row['completed_at']),
            events_processed=row['events_processed'],
            stage_reports=stage_reports,
            status=row['status'],
        )

    async def get_recent_runs(
        self, project_id: str, limit: int = 10
    ) -> list[ReconciliationRun]:
        db = self._require_db()
        async with db.execute(
            'SELECT * FROM runs WHERE project_id = ? ORDER BY started_at DESC LIMIT ?',
            (project_id, limit),
        ) as cursor:
            rows = await cursor.fetchall()
        runs = []
        for row in rows:
            reports_raw = json.loads(row['stage_reports'] or '{}')
            stage_reports = {k: StageReport(**v) for k, v in reports_raw.items()}
            runs.append(
                ReconciliationRun(
                    id=row['id'],
                    project_id=row['project_id'],
                    run_type=row['run_type'],
                    trigger_reason=row['trigger_reason'],
                    started_at=datetime.fromisoformat(row['started_at']),
                    completed_at=_parse_dt(row['completed_at']),
                    events_processed=row['events_processed'],
                    stage_reports=stage_reports,
                    status=row['status'],
                )
            )
        return runs

    async def is_run_active(self, project_id: str) -> bool:
        db = self._require_db()
        async with db.execute(
            "SELECT 1 FROM runs WHERE project_id = ? AND status = 'running' LIMIT 1",
            (project_id,),
        ) as cursor:
            return await cursor.fetchone() is not None

    # ── Journal entries ────────────────────────────────────────────────

    async def add_entry(self, entry: JournalEntry) -> None:
        db = self._require_db()
        await db.execute(
            """INSERT INTO journal_entries
               (id, run_id, stage, timestamp, operation, target_system,
                before_state, after_state, reasoning, evidence)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entry.id,
                entry.run_id,
                entry.stage.value if entry.stage else None,
                entry.timestamp.isoformat(),
                entry.operation,
                entry.target_system,
                json.dumps(entry.before_state) if entry.before_state is not None else None,
                json.dumps(entry.after_state) if entry.after_state is not None else None,
                entry.reasoning,
                json.dumps(entry.evidence),
            ),
        )
        await db.commit()

    async def get_entries(self, run_id: str) -> list[JournalEntry]:
        db = self._require_db()
        async with db.execute(
            'SELECT * FROM journal_entries WHERE run_id = ? ORDER BY timestamp',
            (run_id,),
        ) as cursor:
            rows = await cursor.fetchall()
        entries = []
        for row in rows:
            entries.append(
                JournalEntry(
                    id=row['id'],
                    run_id=row['run_id'],
                    stage=StageId(row['stage']) if row['stage'] else None,
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    operation=row['operation'],
                    target_system=row['target_system'],
                    before_state=json.loads(row['before_state'])
                    if row['before_state']
                    else None,
                    after_state=json.loads(row['after_state'])
                    if row['after_state']
                    else None,
                    reasoning=row['reasoning'],
                    evidence=json.loads(row['evidence']),
                )
            )
        return entries

    # ── Judge verdicts ─────────────────────────────────────────────────

    async def add_verdict(self, verdict: JudgeVerdict) -> None:
        db = self._require_db()
        await db.execute(
            """INSERT INTO judge_verdicts
               (run_id, reviewed_at, severity, findings, action_taken)
               VALUES (?, ?, ?, ?, ?)""",
            (
                verdict.run_id,
                verdict.reviewed_at.isoformat(),
                verdict.severity,
                json.dumps(verdict.findings),
                verdict.action_taken,
            ),
        )
        await db.commit()

    async def get_recent_verdicts(
        self, project_id: str, limit: int = 10
    ) -> list[JudgeVerdict]:
        db = self._require_db()
        async with db.execute(
            """SELECT jv.* FROM judge_verdicts jv
               JOIN runs r ON jv.run_id = r.id
               WHERE r.project_id = ?
               ORDER BY jv.reviewed_at DESC LIMIT ?""",
            (project_id, limit),
        ) as cursor:
            rows = await cursor.fetchall()
        return [
            JudgeVerdict(
                run_id=row['run_id'],
                reviewed_at=datetime.fromisoformat(row['reviewed_at']),
                severity=row['severity'],
                findings=json.loads(row['findings']),
                action_taken=row['action_taken'],
            )
            for row in rows
        ]

    # ── Dashboard stats ────────────────────────────────────────────────

    async def get_stats(self, project_id: str, since: datetime) -> dict:
        db = self._require_db()
        since_str = since.isoformat()

        async with db.execute(
            'SELECT COUNT(*) as cnt FROM runs WHERE project_id = ? AND started_at > ?',
            (project_id, since_str),
        ) as cursor:
            row = await cursor.fetchone()
            runs_count = row['cnt'] if row else 0

        async with db.execute(
            """SELECT AVG(
                 CAST((julianday(completed_at) - julianday(started_at)) * 86400 AS REAL)
               ) as avg_dur
               FROM runs WHERE project_id = ? AND started_at > ? AND completed_at IS NOT NULL""",
            (project_id, since_str),
        ) as cursor:
            row = await cursor.fetchone()
            avg_duration = row['avg_dur'] if row else None

        async with db.execute(
            """SELECT severity, COUNT(*) as cnt
               FROM judge_verdicts jv JOIN runs r ON jv.run_id = r.id
               WHERE r.project_id = ? AND jv.reviewed_at > ?
               GROUP BY severity""",
            (project_id, since_str),
        ) as cursor:
            verdict_rows = await cursor.fetchall()
            verdicts = {row['severity']: row['cnt'] for row in verdict_rows}

        return {
            'runs_count': runs_count,
            'avg_duration_seconds': round(avg_duration, 2) if avg_duration else None,
            'verdicts': verdicts,
        }


def _parse_dt(val: str | None) -> datetime | None:
    if val is None:
        return None
    return datetime.fromisoformat(val)


def _fmt_dt(val: datetime | None) -> str | None:
    if val is None:
        return None
    return val.isoformat()
