"""SQLite persistence for orchestrator run results."""

from __future__ import annotations

import logging
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from shared.sqlite_sync_base import CheckpointResult, apply_full_durability_pragmas_sync

from orchestrator.pause_policy import (
    STORED_EWA_PREDICATE_CLASSES,
    PauseClass,
    stored_pause_class,
)

if TYPE_CHECKING:
    from orchestrator.harness import HarnessReport, TaskReport

logger = logging.getLogger(__name__)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS runs (
    run_id         TEXT PRIMARY KEY,
    project_id     TEXT NOT NULL,
    prd_path       TEXT,
    started_at     TEXT NOT NULL,
    completed_at   TEXT,
    total_tasks    INTEGER DEFAULT 0,
    completed      INTEGER DEFAULT 0,
    blocked        INTEGER DEFAULT 0,
    escalated      INTEGER DEFAULT 0,
    total_cost_usd REAL DEFAULT 0.0,
    paused_for_cap INTEGER DEFAULT 0,
    cap_pause_secs REAL DEFAULT 0.0,
    -- Task 6565: the commit the process's verify wire path was imported from
    -- (orchestrator.deployed_sha.DeployedSha.sha); NULL when unreadable or the
    -- row predates 6565.  Migrated by _migrate_runs_code_sha.
    code_sha       TEXT
);

CREATE TABLE IF NOT EXISTS task_results (
    run_id              TEXT NOT NULL REFERENCES runs(run_id),
    task_id             TEXT NOT NULL,
    project_id          TEXT NOT NULL,
    title               TEXT,
    outcome             TEXT NOT NULL,
    cost_usd            REAL DEFAULT 0.0,
    duration_ms         INTEGER DEFAULT 0,
    agent_invocations   INTEGER DEFAULT 0,
    execute_iterations  INTEGER DEFAULT 0,
    verify_attempts     INTEGER DEFAULT 0,
    review_cycles       INTEGER DEFAULT 0,
    steward_cost_usd    REAL DEFAULT 0.0,
    steward_invocations INTEGER DEFAULT 0,
    completed_at        TEXT,
    -- Task 3068 (origin incident: reify esc-5556-1).  The classified block
    -- reason + PRE-block working phase, so a requeue loop leaves a durable,
    -- GROUP-BY-able WHY here and not only in a rotated journald log.
    --
    -- These MUST stay LAST, after completed_at: SQLite's ALTER TABLE ADD
    -- COLUMN can only append, so declaring them anywhere else would give a
    -- freshly-created runs.db a different physical column ORDER than one
    -- migrated by _migrate_task_results_block_context.  Both writers name
    -- their columns explicitly, but an operator forensic `SELECT *` — exactly
    -- the use case this task exists to serve — is positional.
    --
    -- block_detail is deliberately NOT persisted: unbounded raw agent/verify
    -- output in an operationally-queried store (see task 3068 decisions).
    block_reason        TEXT,
    block_phase         TEXT,
    PRIMARY KEY (run_id, task_id)
);

CREATE INDEX IF NOT EXISTS idx_task_results_project
    ON task_results(project_id, completed_at);

CREATE TABLE IF NOT EXISTS scheduler_state (
    project_id     TEXT PRIMARY KEY,
    pause_reason   TEXT NOT NULL,
    pause_at       TEXT NOT NULL,
    set_by_run_id  TEXT NOT NULL,
    -- Task 4559: the live EWA value at pause time, recorded for EVERY pause
    -- class as operator evidence.  Nullable on purpose: NULL means "row
    -- predates task 4559", which the restart predicate treats as unknowable
    -- and therefore restores the halt blind.
    ewa_value      REAL,
    -- Task 4686: the hold's pause_policy.PauseClass value, its owner and its
    -- ISO expiry.  Declared last because ALTER can only append (migrated by
    -- _migrate_scheduler_state_hold_columns); a NULL pause_class reads as
    -- operator.
    pause_class    TEXT,
    hold_owner     TEXT,
    expires_at     TEXT
);

-- Task 5812: the last COLD main-tip sweep that reached a verdict.
CREATE TABLE IF NOT EXISTS main_sweep_cold_control (
    project_id  TEXT PRIMARY KEY,
    verdict_at  TEXT NOT NULL,
    swept_sha   TEXT NOT NULL
);
"""


class RunStore:
    """Synchronous SQLite writer for orchestrator run results."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        """Open a connection to the store's DB with the full durability pragma triad applied."""
        conn = sqlite3.connect(str(self.db_path))
        apply_full_durability_pragmas_sync(conn, busy_timeout_ms=5000)
        return conn

    def checkpoint(self) -> CheckpointResult:
        """Run ``PRAGMA wal_checkpoint(TRUNCATE)`` and return the result.

        Returns:
            A :class:`CheckpointResult` named-tuple ``(busy, log, checkpointed)``.
        """
        conn = self._connect()
        try:
            row = conn.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()
        finally:
            conn.close()
        if row is None:
            raise RuntimeError('PRAGMA wal_checkpoint returned no rows')
        return CheckpointResult(int(row[0]), int(row[1]), int(row[2]))

    def _ensure_schema(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(_SCHEMA)
            # Task 3068: CREATE TABLE IF NOT EXISTS is a no-op on an existing
            # table, so the _SCHEMA additions above reach FRESH DBs only.  Every
            # already-deployed runs.db needs the explicit ALTER.
            self._migrate_task_results_block_context(conn)
            self._migrate_scheduler_state_ewa_value(conn)
            self._migrate_scheduler_state_hold_columns(conn)
            self._migrate_runs_code_sha(conn)
        finally:
            conn.close()

    def _migrate_task_results_block_context(self, conn: sqlite3.Connection) -> None:
        """Add ``block_reason``/``block_phase`` to a pre-3068 ``task_results``.

        Deliberately additive-only, with no ``PRAGMA user_version`` ladder:
        ``RunStore`` has never had one, and this change needs no destructive
        table rebuild, so the light feature-detect idiom
        (``ticket_store._migrate_add_escalated_at`` /
        ``write_journal._migrate``) is sufficient.  Introducing a version
        header to an existing un-versioned DB would itself need a "version 0
        means unknown" special case, for no benefit.

        Idempotent, and a no-op on a fresh DB — ``_SCHEMA`` already declares
        both columns there, so ``PRAGMA table_info`` finds them present.

        Existing rows read NULL for both columns, which is exactly right: NULL
        means "row predates task 3068", while ``''`` (what both writers now
        always store) means "clean exit, no block".

        FAILURE BEHAVIOUR — note what this actually is, not what would be
        tidy.  A genuinely unrecoverable ``ALTER`` propagates out of
        ``__init__``, but the sole production caller (``Harness.run``) wraps
        ``RunStore(db_path)`` in ``except Exception: logger.error(...)`` and
        continues, so a raise here does NOT halt the run — it drops
        ``task_results`` persistence for the whole run.  That is strictly worse
        than writing NULLs, so we do NOT raise on the one failure mode that is
        both plausible and benign: a concurrent construction that already added
        the column.  ``duplicate column name`` means the migration's goal is
        met, and is swallowed per-column.  Everything else (a corrupt DB, a
        lock we could not take inside the busy_timeout) still surfaces, because
        those are real and a silent half-migrated schema would be worse.
        """
        existing = {row[1] for row in conn.execute('PRAGMA table_info(task_results)')}
        added = []
        for column in ('block_reason', 'block_phase'):
            if column in existing:
                continue
            try:
                # Column name is a hard-coded literal, never caller input.
                conn.execute(f'ALTER TABLE task_results ADD COLUMN {column} TEXT')
            except sqlite3.OperationalError as exc:
                # Another RunStore construction won the race between our
                # table_info read and this ALTER.  The column exists, which is
                # all we wanted; raising would cost the caller task_results
                # persistence for an entire run over a no-op.
                if 'duplicate column name' not in str(exc).lower():
                    raise
                logger.debug(
                    'RunStore: task_results.%s already added concurrently', column,
                )
                continue
            added.append(column)
        if added:
            conn.commit()
            logger.info(
                'RunStore: migrated task_results — added %s', ', '.join(added)
            )

    def _migrate_scheduler_state_ewa_value(self, conn: sqlite3.Connection) -> None:
        """Add ``ewa_value`` to a pre-4559 ``scheduler_state``.

        Same shape as :meth:`_migrate_task_results_block_context`, deliberately:
        additive-only, ``PRAGMA table_info`` feature-detect, no
        ``PRAGMA user_version`` ladder.  ``RunStore`` has never had a version
        header, and adding one to an existing un-versioned DB would itself need
        a "version 0 means unknown" special case for no benefit.

        Idempotent, and a no-op on a fresh DB — ``_SCHEMA`` already declares the
        column there, so ``PRAGMA table_info`` finds it present.

        An existing row reads NULL, which is exactly right: NULL means "row
        predates task 4559", and ``Harness._load_persisted_scheduler_pause``
        treats an unknowable predicate as a reason to RE-ASSERT the halt, never
        to release it.

        FAILURE BEHAVIOUR — the same asymmetry the sibling migration documents.
        A raise here propagates out of ``__init__``; the sole production caller
        (``Harness.run``) wraps ``RunStore(db_path)`` in
        ``except Exception: logger.error(...)`` and continues, so it would drop
        persistence for the whole run rather than halt it.  That is strictly
        worse than a NULL column, so the one plausible-and-benign failure — a
        concurrent construction that already added the column between our
        ``table_info`` read and this ``ALTER`` — is swallowed: ``duplicate
        column name`` means the migration's goal is already met.  Everything
        else (a corrupt DB, a lock we could not take inside the busy_timeout)
        still surfaces, because a silent half-migrated schema would be worse.
        """
        existing = {row[1] for row in conn.execute('PRAGMA table_info(scheduler_state)')}
        if 'ewa_value' in existing:
            return
        try:
            # Column name is a hard-coded literal, never caller input.
            conn.execute('ALTER TABLE scheduler_state ADD COLUMN ewa_value REAL')
        except sqlite3.OperationalError as exc:
            if 'duplicate column name' not in str(exc).lower():
                raise
            logger.debug('RunStore: scheduler_state.ewa_value already added concurrently')
            return
        conn.commit()
        logger.info('RunStore: migrated scheduler_state — added ewa_value')

    def _migrate_scheduler_state_hold_columns(self, conn: sqlite3.Connection) -> None:
        """Add ``pause_class``/``hold_owner``/``expires_at`` to a pre-4686 ``scheduler_state``.

        Same shape as :meth:`_migrate_scheduler_state_ewa_value`; a NULL
        ``pause_class`` reads as operator via ``pause_policy.stored_pause_class``.
        """
        existing = {row[1] for row in conn.execute('PRAGMA table_info(scheduler_state)')}
        added = []
        for column in ('pause_class', 'hold_owner', 'expires_at'):
            if column in existing:
                continue
            try:
                # Column name is a hard-coded literal, never caller input.
                conn.execute(f'ALTER TABLE scheduler_state ADD COLUMN {column} TEXT')
            except sqlite3.OperationalError as exc:
                if 'duplicate column name' not in str(exc).lower():
                    raise
                logger.debug(
                    'RunStore: scheduler_state.%s already added concurrently', column,
                )
                continue
            added.append(column)
        if added:
            conn.commit()
            logger.info(
                'RunStore: migrated scheduler_state — added %s', ', '.join(added)
            )

    def _migrate_runs_code_sha(self, conn: sqlite3.Connection) -> None:
        """Add ``code_sha`` to a pre-6565 ``runs``.

        Same shape as :meth:`_migrate_scheduler_state_ewa_value`, deliberately:
        additive-only, ``PRAGMA table_info`` feature-detect, the column added
        LAST, a concurrent construction's ``duplicate column name`` swallowed
        and every other failure raised.  An existing row reads NULL, meaning
        "the run predates task 6565".
        """
        existing = {row[1] for row in conn.execute('PRAGMA table_info(runs)')}
        if 'code_sha' in existing:
            return
        try:
            # Column name is a hard-coded literal, never caller input.
            conn.execute('ALTER TABLE runs ADD COLUMN code_sha TEXT')
        except sqlite3.OperationalError as exc:
            if 'duplicate column name' not in str(exc).lower():
                raise
            logger.debug('RunStore: runs.code_sha already added concurrently')
            return
        conn.commit()
        logger.info('RunStore: migrated runs — added code_sha')

    def start_run(
        self,
        run_id: str,
        project_id: str,
        started_at: str,
        prd_path: str | None = None,
        *,
        code_sha: str | None = None,
    ) -> None:
        """Insert an initial runs row at orchestrator startup.

        *code_sha* is the commit the process's wire path was imported from
        (task 6565), ``None`` when it could not be read.  The row is updated
        with final aggregates via :meth:`finish_run` when the run completes.
        """
        conn = self._connect()
        try:
            conn.execute(
                'INSERT OR IGNORE INTO runs '
                '(run_id, project_id, prd_path, started_at, code_sha) '
                'VALUES (?, ?, ?, ?, ?)',
                (run_id, project_id, prd_path, started_at, code_sha),
            )
            conn.commit()
        finally:
            conn.close()
        logger.info(f'Started run {run_id} for project {project_id}')

    def save_task_result(
        self,
        run_id: str,
        task_report: TaskReport,
        project_id: str,
    ) -> None:
        """Persist a single TaskReport row immediately on task completion."""
        conn = self._connect()
        try:
            conn.execute(
                'INSERT OR REPLACE INTO task_results '
                '(run_id, task_id, project_id, title, outcome, '
                ' cost_usd, duration_ms, agent_invocations, '
                ' execute_iterations, verify_attempts, review_cycles, '
                ' steward_cost_usd, steward_invocations, completed_at, '
                ' block_reason, block_phase) '
                'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (
                    run_id,
                    task_report.task_id,
                    project_id,
                    task_report.title,
                    task_report.outcome.value,
                    task_report.cost_usd,
                    task_report.duration_ms,
                    task_report.agent_invocations,
                    task_report.execute_iterations,
                    task_report.verify_attempts,
                    task_report.review_cycles,
                    task_report.steward_cost_usd,
                    task_report.steward_invocations,
                    task_report.completed_at,
                    # Task 3068 — '' on clean exits, never NULL, so a NULL here
                    # unambiguously means "row predates 3068".
                    task_report.block_reason,
                    task_report.block_phase,
                ),
            )
            conn.commit()
        finally:
            conn.close()
        logger.debug(
            f'Persisted task result {task_report.task_id} '
            f'({task_report.outcome.value}) for run {run_id}'
        )

    def get_task_cost(self, run_id: str, task_id: str) -> float:
        """Return latest persisted ``cost_usd + steward_cost_usd`` for a task.

        Only the most recent ``save_task_result`` row is retained per
        ``(run_id, task_id)`` — prior requeue attempts are overwritten.
        Returns ``0.0`` when no row exists.
        """
        conn = self._connect()
        try:
            cur = conn.execute(
                'SELECT cost_usd, steward_cost_usd FROM task_results '
                'WHERE run_id = ? AND task_id = ?',
                (run_id, task_id),
            )
            row = cur.fetchone()
        finally:
            conn.close()
        if row is None:
            return 0.0
        cost_usd, steward_cost_usd = row
        return float(cost_usd or 0.0) + float(steward_cost_usd or 0.0)

    def finish_run(
        self,
        run_id: str,
        report: HarnessReport,
    ) -> None:
        """Update the runs row with final aggregates at shutdown."""
        conn = self._connect()
        try:
            conn.execute(
                'UPDATE runs SET '
                'completed_at = ?, total_tasks = ?, completed = ?, '
                'blocked = ?, escalated = ?, total_cost_usd = ?, '
                'paused_for_cap = ?, cap_pause_secs = ? '
                'WHERE run_id = ?',
                (
                    report.completed_at,
                    report.total_tasks,
                    report.completed,
                    report.blocked,
                    report.escalated,
                    report.total_cost_usd,
                    int(report.paused_for_cap),
                    report.cap_pause_duration_secs,
                    run_id,
                ),
            )
            conn.commit()
        finally:
            conn.close()
        logger.info(
            f'Finished run {run_id}: {len(report.task_reports)} task results'
        )

    def save_run(
        self,
        report: HarnessReport,
        project_id: str,
        prd_path: str | None = None,
    ) -> str:
        """Persist a HarnessReport and its TaskReports to SQLite.

        Returns the generated run_id.
        """
        run_id = f'run-{uuid.uuid4().hex[:12]}'
        conn = self._connect()
        try:
            conn.execute(
                'INSERT INTO runs '
                '(run_id, project_id, prd_path, started_at, completed_at, '
                ' total_tasks, completed, blocked, escalated, '
                ' total_cost_usd, paused_for_cap, cap_pause_secs) '
                'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (
                    run_id,
                    project_id,
                    prd_path,
                    report.started_at,
                    report.completed_at,
                    report.total_tasks,
                    report.completed,
                    report.blocked,
                    report.escalated,
                    report.total_cost_usd,
                    int(report.paused_for_cap),
                    report.cap_pause_duration_secs,
                ),
            )
            for tr in report.task_reports:
                conn.execute(
                    'INSERT INTO task_results '
                    '(run_id, task_id, project_id, title, outcome, '
                    ' cost_usd, duration_ms, agent_invocations, '
                    ' execute_iterations, verify_attempts, review_cycles, '
                    ' steward_cost_usd, steward_invocations, completed_at, '
                    ' block_reason, block_phase) '
                    'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                    (
                        run_id,
                        tr.task_id,
                        project_id,
                        tr.title,
                        tr.outcome.value,
                        tr.cost_usd,
                        tr.duration_ms,
                        tr.agent_invocations,
                        tr.execute_iterations,
                        tr.verify_attempts,
                        tr.review_cycles,
                        tr.steward_cost_usd,
                        tr.steward_invocations,
                        tr.completed_at,
                        # Task 3068 — the batch path must stay in lockstep with
                        # save_task_result above, or it silently writes NULLs.
                        tr.block_reason,
                        tr.block_phase,
                    ),
                )
            conn.commit()
        finally:
            conn.close()

        logger.info(
            f'Persisted run {run_id}: {len(report.task_reports)} task results '
            f'for project {project_id}'
        )
        return run_id

    # ------------------------------------------------------------------ #
    # Scheduler park-and-stop pause persistence (task 1322)               #
    # ------------------------------------------------------------------ #

    def save_scheduler_pause(
        self,
        project_id: str,
        reason: str,
        pause_at_iso: str,
        set_by_run_id: str,
        ewa_value: float | None = None,
        *,
        pause_class: PauseClass,
        hold_owner: str | None = None,
        expires_at: datetime | None = None,
    ) -> None:
        """Persist (or replace) the scheduler pause state for *project_id*.

        Uses INSERT OR REPLACE so repeated saves are idempotent — the most
        recent call wins, matching UPSERT semantics.

        *pause_class* is required: every persisted hold carries its class
        (PRD §6 invariant 4).  *ewa_value* (task 4559) is the live EWA at pause
        time, recorded for every class as operator evidence; ``None`` stores
        NULL, which reads back as "predates task 4559".  *expires_at* must be
        timezone-aware.

        Raises:
            ValueError: *expires_at* is naive; nothing is written.
        """
        if expires_at is not None and expires_at.tzinfo is None:
            raise ValueError(
                f'expires_at must be timezone-aware; got naive {expires_at.isoformat()!r}'
            )
        conn = self._connect()
        try:
            conn.execute(
                'INSERT OR REPLACE INTO scheduler_state '
                '(project_id, pause_reason, pause_at, set_by_run_id, ewa_value, '
                ' pause_class, hold_owner, expires_at) '
                'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                (
                    project_id, reason, pause_at_iso, set_by_run_id, ewa_value,
                    pause_class.value,
                    hold_owner,
                    expires_at.isoformat() if expires_at is not None else None,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def refresh_scheduler_pause_ewa(self, project_id: str, ewa_value: float) -> bool:
        """Update ONLY ``ewa_value``, and only on a held row whose stored EWA is its predicate.

        Returns True if such a row existed and was updated.  Which classes
        qualify is ``pause_policy.STORED_EWA_PREDICATE_CLASSES`` (today:
        ewa_rate alone).

        Task 4559 amendment.  ``save_scheduler_pause`` writes a trip-TIME
        snapshot, and it is the only writer, so once a halt is asserted the
        stored scalar freezes at the tripping value even as the live EWA decays
        across subsequent digest steps.  That froze two things solid: the
        restart-time predicate re-check in
        ``Harness._load_persisted_scheduler_pause`` could essentially never
        fire (a stored value is by construction >= the threshold at write
        time), and a restart re-seeded live ``_ewa_value`` with the stale high
        number, discarding whatever decay the drain had achieved.  On the 8h
        fleet-redeploy cadence a paused unit therefore reset its healing
        progress on every restart.

        This is deliberately an UPDATE, never an upsert: with no pause row
        there is nothing to refresh, and inserting one would fabricate a halt
        out of a routine digest step.  ``reason``, ``pause_at`` and
        ``set_by_run_id`` are left untouched so the row keeps recording WHEN
        and WHY the halt was asserted while ``ewa_value`` tracks the live
        statistic.  The bool return lets the caller distinguish "refreshed"
        from "no qualifying row to refresh" without a second read.

        The class scope is enforced HERE, in the WHERE clause, not only at the
        call site.  ``_maybe_write_digest`` step (13b) pre-filters on the
        IN-MEMORY ``scheduler.pause_class``, which is first-wins, while this
        row is last-wins (INSERT OR REPLACE): a held ewa_rate halt plus a later
        park-stop ``pause_scheduler`` leaves memory saying ewa_rate over a
        park_stop row.  For a qualifying class alone the stored scalar IS the
        live statistic; for every other class (and a legacy NULL-class row, which
        reads as operator) it is evidence about that pause and is kept.
        """
        predicate_values = sorted(c.value for c in STORED_EWA_PREDICATE_CLASSES)
        placeholders = ', '.join('?' * len(predicate_values))
        conn = self._connect()
        try:
            cursor = conn.execute(
                'UPDATE scheduler_state SET ewa_value = ? '
                f'WHERE project_id = ? AND pause_class IN ({placeholders})',
                (ewa_value, project_id, *predicate_values),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def load_scheduler_pause(self, project_id: str) -> dict | None:
        """Return the persisted pause record for *project_id*, or ``None``.

        Returns a dict with keys ``reason``, ``pause_at``, ``set_by_run_id``,
        ``ewa_value`` (task 4559; ``None`` for a row written before that
        migration), ``pause_class`` (a :class:`PauseClass`; a NULL or
        unrecognised stored value reads as operator), ``hold_owner`` and
        ``expires_at`` (an aware ``datetime`` or ``None``).
        """
        conn = self._connect()
        try:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                'SELECT pause_reason, pause_at, set_by_run_id, ewa_value, '
                '       pause_class, hold_owner, expires_at '
                'FROM scheduler_state WHERE project_id = ?',
                (project_id,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return {
            'reason': row['pause_reason'],
            'pause_at': row['pause_at'],
            'set_by_run_id': row['set_by_run_id'],
            'ewa_value': row['ewa_value'],
            'pause_class': stored_pause_class(row['pause_class']),
            'hold_owner': row['hold_owner'],
            'expires_at': (
                datetime.fromisoformat(row['expires_at'])
                if row['expires_at'] is not None else None
            ),
        }

    def clear_scheduler_pause(self, project_id: str) -> None:
        """Remove any persisted pause state for *project_id*."""
        conn = self._connect()
        try:
            conn.execute(
                'DELETE FROM scheduler_state WHERE project_id = ?',
                (project_id,),
            )
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # Main-tip sweep cold-control cadence (task 5812)                     #
    # ------------------------------------------------------------------ #

    def save_main_sweep_cold_verdict(
        self, project_id: str, *, verdict_at: datetime, swept_sha: str,
    ) -> None:
        """Persist (last wins) when *project_id*'s last COLD sweep reached a verdict.

        Stored here so the cold-control cadence survives an orchestrator restart;
        the consumer is ``orchestrator/main_tip_sweep_cadence.py::MainSweepColdControl``.
        """
        conn = self._connect()
        try:
            conn.execute(
                'INSERT OR REPLACE INTO main_sweep_cold_control '
                '(project_id, verdict_at, swept_sha) VALUES (?, ?, ?)',
                (project_id, verdict_at.isoformat(), swept_sha),
            )
            conn.commit()
        finally:
            conn.close()

    def load_main_sweep_last_cold_verdict(self, project_id: str) -> datetime | None:
        """The aware-UTC time of *project_id*'s last cold sweep verdict, or ``None``."""
        conn = self._connect()
        try:
            row = conn.execute(
                'SELECT verdict_at FROM main_sweep_cold_control WHERE project_id = ?',
                (project_id,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        verdict_at = datetime.fromisoformat(row[0])
        if verdict_at.tzinfo is None:
            return verdict_at.replace(tzinfo=UTC)
        return verdict_at
