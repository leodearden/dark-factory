"""Tests for orchestrator.b3_gate's CLI.

Covers main()'s three verbs and the run_check path, together with the tasks.db
readers (_read_latest_proposal, the task-description read) and the cap config
(_resolve_cap) it consults.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path

import pytest
from _b3_gate_fixtures import diverged_repo
from shared.tasks_db_readonly import TaskDbProblem, TaskDbUnreadable

# ---------------------------------------------------------------------------
# tasks.db and repo scene seeding
# ---------------------------------------------------------------------------

def _seed_tasks_db(
    project_root: Path, task_id: int, proposals: list, *, description: str = '',
) -> None:
    """Seed a minimal tasks.db at <project_root>/.taskmaster/tasks/tasks.db.

    Creates the table with columns: tag, id, status, priority, metadata,
    updated_at, description. Inserts one row with tag='master', the given
    task_id, and the proposals list serialised into metadata JSON as
    {'dry_run_proposals': proposals}. `description` defaults to '' — every
    pre-task-2509 call site omits it and gets the same empty-description row
    as before (b3_gate._read_task_description treats '' as "no description").
    """
    db_dir = project_root / '.taskmaster' / 'tasks'
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / 'tasks.db'
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            tag TEXT NOT NULL,
            id INTEGER NOT NULL,
            status TEXT,
            priority TEXT,
            metadata TEXT,
            updated_at TEXT,
            description TEXT,
            PRIMARY KEY (tag, id)
        )
    """)
    metadata = json.dumps({'dry_run_proposals': proposals})
    conn.execute(
        "INSERT OR REPLACE INTO tasks "
        "(tag, id, status, priority, metadata, updated_at, description) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ('master', task_id, 'blocked', 'medium', metadata,
         '2026-06-04T00:00:00+00:00', description),
    )
    conn.commit()
    conn.close()


def _seed_fresh_check_scene(tmp_path: Path, *, description: str = '') -> tuple[Path, Path]:
    """Seed a repo whose feature branch has diverged from main, and a tasks.db
    whose task 42 carries *description* and one fresh low-risk proposal
    anchored at that repo's head and main shas.

    Returns (repo, project_root).
    """
    repo, head_sha, main_sha = diverged_repo(tmp_path)

    root = tmp_path / 'root'
    root.mkdir()
    entry = {
        'proposal_text': 'Fix the bug',
        'risk_label': 'low',
        'files_referenced': [],
        'block_reason': 'test blocked',
        'investigated_at': '2026-06-04T10:00:00+00:00',
        'head_sha': head_sha,
        'main_sha': main_sha,
    }
    _seed_tasks_db(root, 42, [entry], description=description)
    return repo, root


# ---------------------------------------------------------------------------
# step-13: _read_latest_proposal
# ---------------------------------------------------------------------------

class TestReadLatestProposal:
    def test_returns_last_proposal_entry(self, tmp_path):
        from orchestrator.b3_gate import _read_latest_proposal
        entry_a = {'proposal_text': 'first', 'risk_label': 'low'}
        entry_b = {'proposal_text': 'second', 'risk_label': 'low'}
        _seed_tasks_db(tmp_path, 42, [entry_a, entry_b])
        result = _read_latest_proposal(42, tmp_path)
        assert result == entry_b, f'expected last entry, got {result}'

    def test_missing_task_id_returns_none(self, tmp_path):
        from orchestrator.b3_gate import _read_latest_proposal
        _seed_tasks_db(tmp_path, 42, [{'proposal_text': 'x', 'risk_label': 'low'}])
        result = _read_latest_proposal(99, tmp_path)  # id 99 not seeded
        assert result is None

    def test_missing_db_returns_none(self, tmp_path):
        from orchestrator.b3_gate import _read_latest_proposal
        # No tasks.db at all
        result = _read_latest_proposal(42, tmp_path)
        assert result is None

    def test_empty_proposals_list_returns_none(self, tmp_path):
        from orchestrator.b3_gate import _read_latest_proposal
        _seed_tasks_db(tmp_path, 42, [])
        result = _read_latest_proposal(42, tmp_path)
        assert result is None

    def test_metadata_missing_dry_run_proposals_returns_none(self, tmp_path):
        from orchestrator.b3_gate import _read_latest_proposal
        # Seed with metadata that lacks dry_run_proposals key
        db_dir = tmp_path / '.taskmaster' / 'tasks'
        db_dir.mkdir(parents=True, exist_ok=True)
        db_path = db_dir / 'tasks.db'
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            'CREATE TABLE IF NOT EXISTS tasks '
            '(tag TEXT, id INTEGER, status TEXT, priority TEXT, metadata TEXT, updated_at TEXT)'
        )
        conn.execute(
            "INSERT INTO tasks VALUES ('master', 42, 'blocked', 'medium', ?, ?)",
            (json.dumps({'other_key': 'value'}), '2026-06-04T00:00:00+00:00'),
        )
        conn.commit()
        conn.close()
        result = _read_latest_proposal(42, tmp_path)
        assert result is None

    def test_uses_default_tag_master(self, tmp_path):
        from orchestrator.b3_gate import _read_latest_proposal
        entry = {'proposal_text': 'x', 'risk_label': 'low'}
        _seed_tasks_db(tmp_path, 42, [entry])  # seeds tag='master'
        result = _read_latest_proposal(42, tmp_path)  # default tag='master'
        assert result == entry

    def test_string_task_id_works(self, tmp_path):
        from orchestrator.b3_gate import _read_latest_proposal
        entry = {'proposal_text': 'x', 'risk_label': 'low'}
        _seed_tasks_db(tmp_path, 42, [entry])
        result = _read_latest_proposal('42', tmp_path)
        assert result == entry


# ---------------------------------------------------------------------------
# review amendment: reader failures warn instead of failing open silently
# ---------------------------------------------------------------------------

class TestReaderFailureWarning:
    """_read_task_description / _read_latest_proposal emit a deduped WARNING
    when their read genuinely fails (e.g. tasks.db lacks the expected
    column — an older/foreign schema) rather than degrading silently (task
    2509 review amendment). The benign "no data yet" branches (missing db,
    missing row, empty column) stay silent — only a raised exception warns.
    """

    def _seed_tasks_without_description_column(self, tmp_path, task_id=42):
        """A 'tasks' table that lacks the 'description' column entirely —
        the older/foreign-schema scenario the review finding names."""
        db_dir = tmp_path / '.taskmaster' / 'tasks'
        db_dir.mkdir(parents=True, exist_ok=True)
        db_path = db_dir / 'tasks.db'
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            'CREATE TABLE IF NOT EXISTS tasks '
            '(tag TEXT, id INTEGER, status TEXT, priority TEXT, metadata TEXT, updated_at TEXT)'
        )
        conn.execute(
            "INSERT INTO tasks VALUES ('master', ?, 'blocked', 'medium', ?, ?)",
            (task_id, json.dumps({}), '2026-06-04T00:00:00+00:00'),
        )
        conn.commit()
        conn.close()

    def test_missing_description_column_warns(self, tmp_path, caplog):
        import logging as _logging

        from orchestrator.b3_gate import _read_task_description
        self._seed_tasks_without_description_column(tmp_path, task_id=42)

        with caplog.at_level(_logging.WARNING, logger='orchestrator.b3_gate'):
            result = _read_task_description(42, tmp_path)

        assert result is None
        warn_records = [
            r for r in caplog.records
            if r.levelno == _logging.WARNING and r.name == 'orchestrator.b3_gate'
        ]
        assert len(warn_records) >= 1, (
            f'Expected a WARNING under orchestrator.b3_gate for a missing '
            f'description column; got records: '
            f'{[(r.name, r.message) for r in caplog.records]}'
        )

    def test_missing_description_column_warning_is_deduped(self, tmp_path, caplog):
        """A second call for the SAME (reader, project_root, tag, task_id)
        key must not re-warn within the same process."""
        import logging as _logging

        from orchestrator.b3_gate import _read_task_description
        self._seed_tasks_without_description_column(tmp_path, task_id=42)

        with caplog.at_level(_logging.WARNING, logger='orchestrator.b3_gate'):
            _read_task_description(42, tmp_path)
            _read_task_description(42, tmp_path)

        warn_records = [
            r for r in caplog.records
            if r.levelno == _logging.WARNING and r.name == 'orchestrator.b3_gate'
        ]
        assert len(warn_records) == 1, (
            f'Expected exactly one deduped WARNING across two calls with the '
            f'same key; got {len(warn_records)}: {[r.message for r in warn_records]}'
        )

    def test_missing_db_stays_silent(self, tmp_path, caplog):
        """Benign 'no tasks.db at all' must NOT warn — this is not a failure,
        it's the ordinary first-run-absent case."""
        import logging as _logging

        from orchestrator.b3_gate import _read_task_description

        with caplog.at_level(_logging.WARNING, logger='orchestrator.b3_gate'):
            result = _read_task_description(42, tmp_path)

        assert result is None
        warn_records = [
            r for r in caplog.records
            if r.levelno == _logging.WARNING and r.name == 'orchestrator.b3_gate'
        ]
        assert len(warn_records) == 0, (
            f'missing tasks.db must not warn (benign first-run absence); got: '
            f'{[r.message for r in warn_records]}'
        )

    def test_latest_proposal_missing_metadata_column_stays_silent(self, tmp_path, caplog):
        """_read_latest_proposal is deliberately NOT given the same WARNING
        instrumentation as _read_task_description above, despite having the
        identical fail-open shape (review finding explicitly named it as an
        "ideal" sibling fix). This exact site already has a pre-existing,
        reasoned entry in shared/tests/silent_fallthrough_allowlist.py
        ("narrow fix deferred to follow-up"); adding a WARNING here would
        change the handler's AST shape and require also editing that
        allowlist file — which sits outside this task's locked module scope
        (task 2509 review amendment: scope discipline). Locks in the
        intentional asymmetry so a future edit doesn't "fix" this half
        without the matching allowlist update.
        """
        import logging as _logging

        from orchestrator.b3_gate import _read_latest_proposal

        db_dir = tmp_path / '.taskmaster' / 'tasks'
        db_dir.mkdir(parents=True, exist_ok=True)
        db_path = db_dir / 'tasks.db'
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            'CREATE TABLE IF NOT EXISTS tasks '
            '(tag TEXT, id INTEGER, status TEXT, priority TEXT, description TEXT, updated_at TEXT)'
        )
        conn.execute(
            "INSERT INTO tasks VALUES ('master', 42, 'blocked', 'medium', 'desc', ?)",
            ('2026-06-04T00:00:00+00:00',),
        )
        conn.commit()
        conn.close()

        with caplog.at_level(_logging.WARNING, logger='orchestrator.b3_gate'):
            result = _read_latest_proposal(42, tmp_path)

        assert result is None
        warn_records = [
            r for r in caplog.records
            if r.levelno == _logging.WARNING and r.name == 'orchestrator.b3_gate'
        ]
        assert len(warn_records) == 0, (
            f'_read_latest_proposal is intentionally NOT warning-instrumented '
            f'in this amendment pass (scope discipline — see docstring); got: '
            f'{[r.message for r in warn_records]}'
        )

    def _seed_stub_tasks_db(self, tmp_path):
        """The 0-byte decoy shape: a tasks.db that exists and holds nothing."""
        db = tmp_path / '.taskmaster' / 'tasks' / 'tasks.db'
        db.parent.mkdir(parents=True, exist_ok=True)
        db.write_bytes(b'')
        return db

    def _run_check_warnings(self, tmp_path, capsys, caplog):
        """Run the public 'check' verb; return its verdict and EVERY warning.

        Every warning, not only those naming the store: run_check reads tasks.db
        through both readers, and _read_latest_proposal must stay silent while
        _read_task_description warns once, so a second warning is a regression.
        """
        import argparse
        import logging as _logging

        from orchestrator.b3_gate import run_check
        args = argparse.Namespace(
            task_id='42',
            worktree=str(tmp_path),
            project_root=str(tmp_path),
            config=None,
            now='2026-06-04T12:00:00+00:00',
            tag='master',
            category=None,
            verb='check',
        )

        with caplog.at_level(_logging.WARNING, logger='orchestrator.b3_gate'):
            run_check(args)

        result = json.loads(capsys.readouterr().out)
        warnings = [r.getMessage() for r in caplog.records if r.levelno >= _logging.WARNING]
        return result, warnings

    def test_run_check_on_a_stub_store_warns_once_naming_the_refusal(
        self, tmp_path, capsys, caplog,
    ):
        db = self._seed_stub_tasks_db(tmp_path)

        result, warnings = self._run_check_warnings(tmp_path, capsys, caplog)

        assert 'verdict' in result
        refusal = str(TaskDbUnreadable(db.resolve(), TaskDbProblem.EMPTY_STUB))
        assert len(warnings) == 1, warnings
        assert refusal in warnings[0]

    @pytest.mark.skipif(os.geteuid() == 0, reason='mode 000 does not deny root')
    def test_run_check_on_an_unopenable_store_warns_once(self, tmp_path, capsys, caplog):
        """A store connect_ro cannot OPEN raises sqlite3.Error, not a refusal.

        It propagates out of _open_tasks_db like a non-ABSENT refusal, so the
        description reader warns instead of answering a silent None.
        """
        _seed_tasks_db(tmp_path, 42, [])
        db = tmp_path / '.taskmaster' / 'tasks' / 'tasks.db'
        db.chmod(0o000)
        try:
            result, warnings = self._run_check_warnings(tmp_path, capsys, caplog)
        finally:
            db.chmod(0o600)

        assert 'verdict' in result
        assert len(warnings) == 1, warnings
        assert sqlite3.OperationalError.__name__ in warnings[0]


# ---------------------------------------------------------------------------
# step-15: _resolve_cap
# ---------------------------------------------------------------------------

class TestResolveCap:
    def test_none_returns_default(self):
        from orchestrator.b3_gate import DEFAULT_CAP, _resolve_cap
        assert _resolve_cap(None) == DEFAULT_CAP
        assert _resolve_cap(None) == 6

    def test_config_path_returns_configured_cap(self, tmp_path):
        from orchestrator.b3_gate import _resolve_cap
        cfg_file = tmp_path / 'config.yaml'
        cfg_file.write_text('unblock_auto:\n  b3_merge_cap_per_24h: 2\n')
        assert _resolve_cap(str(cfg_file)) == 2

    def test_config_path_different_cap(self, tmp_path):
        from orchestrator.b3_gate import _resolve_cap
        cfg_file = tmp_path / 'config.yaml'
        cfg_file.write_text('unblock_auto:\n  b3_merge_cap_per_24h: 10\n')
        assert _resolve_cap(str(cfg_file)) == 10

    # task-1808 step-7: bad config → DEFAULT_CAP + WARNING under orchestrator.b3_gate

    def test_bad_config_returns_default_and_warns(self, tmp_path, caplog):
        """Malformed config YAML → DEFAULT_CAP returned AND WARNING logged.

        Config corruption is NOT cap-exhausting (§4.5 — a bad/foreign config
        legitimately falls back to DEFAULT_CAP).  But the fallback must be VISIBLE.
        Fails today: b3_gate has no module logger → zero WARNING records.
        """
        import logging as _logging

        from orchestrator.b3_gate import DEFAULT_CAP, _resolve_cap

        cfg_file = tmp_path / 'bad_config.yaml'
        # Write a YAML string that's structurally invalid / causes load_config to raise
        cfg_file.write_text('unblock_auto: [unterminated\n', encoding='utf-8')

        with caplog.at_level(_logging.WARNING, logger='orchestrator.b3_gate'):
            result = _resolve_cap(str(cfg_file))

        assert result == DEFAULT_CAP, (
            f'Expected DEFAULT_CAP={DEFAULT_CAP} for bad config; got {result}'
        )
        warn_records = [
            r for r in caplog.records
            if r.levelno == _logging.WARNING
            and r.name == 'orchestrator.b3_gate'
        ]
        assert len(warn_records) >= 1, (
            f'Expected WARNING under orchestrator.b3_gate for bad config; '
            f'got records: {[(r.name, r.message) for r in caplog.records]}'
        )

    def test_none_config_stays_silent(self, caplog):
        """_resolve_cap(None) must NOT emit any WARNING (benign no-config path)."""
        import logging as _logging

        from orchestrator.b3_gate import _resolve_cap

        with caplog.at_level(_logging.WARNING, logger='orchestrator.b3_gate'):
            _resolve_cap(None)

        warn_records = [
            r for r in caplog.records
            if r.levelno == _logging.WARNING
            and r.name == 'orchestrator.b3_gate'
        ]
        assert len(warn_records) == 0, (
            f'_resolve_cap(None) must not emit warnings; got: {[r.message for r in warn_records]}'
        )


# ---------------------------------------------------------------------------
# step-17: CLI main() wiring for all three verbs, JSON to stdout
# ---------------------------------------------------------------------------

class TestCLIMain:
    """Test the three CLI verbs via main(argv)."""

    _NOW_ISO = '2026-06-04T12:00:00+00:00'
    _NOW = datetime.fromisoformat(_NOW_ISO)

    def test_check_fresh_verdict_json(self, tmp_path, capsys):
        """check verb: prints a single JSON object with verdict=='fresh'."""
        from orchestrator.b3_gate import main
        repo, root = _seed_fresh_check_scene(tmp_path)

        rc = main([
            'check',
            '--task-id', '42',
            '--worktree', str(repo),
            '--project-root', str(root),
            '--category', 'task_failure',
            '--now', self._NOW_ISO,
        ])
        assert rc == 0, f'expected exit 0, got {rc}'

        out = capsys.readouterr().out.strip()
        data = json.loads(out)  # must be parseable JSON
        assert data['verdict'] == 'fresh', f'expected fresh: {data}'
        assert data['cap_remaining'] == 6
        assert data['already_attempted'] is False
        assert 'age_seconds' in data
        assert 'reason' in data

    def test_record_launch_first_call(self, tmp_path, capsys):
        """record-launch verb: first call prints {recorded:True, already_attempted:False}."""
        from orchestrator.b3_gate import main
        repo, root = _seed_fresh_check_scene(tmp_path)

        rc = main([
            'record-launch',
            '--task-id', '42',
            '--worktree', str(repo),
            '--project-root', str(root),
        ])
        assert rc == 0

        out = capsys.readouterr().out.strip()
        data = json.loads(out)
        assert data['recorded'] is True
        assert data['already_attempted'] is False

    def test_record_launch_idempotent_second_call(self, tmp_path, capsys):
        """record-launch verb: second identical call prints already_attempted:True."""
        from orchestrator.b3_gate import main
        repo, root = _seed_fresh_check_scene(tmp_path)

        argv = ['record-launch', '--task-id', '42',
                '--worktree', str(repo), '--project-root', str(root)]
        main(argv)
        capsys.readouterr()  # discard first call output

        rc = main(argv)
        assert rc == 0
        out = capsys.readouterr().out.strip()
        data = json.loads(out)
        assert data['recorded'] is False
        assert data['already_attempted'] is True

    def test_check_after_record_launch_reports_already_attempted(self, tmp_path, capsys):
        """check verb: after record-launch, already_attempted==True."""
        from orchestrator.b3_gate import main
        repo, root = _seed_fresh_check_scene(tmp_path)

        # Record launch first
        main(['record-launch', '--task-id', '42',
              '--worktree', str(repo), '--project-root', str(root)])
        capsys.readouterr()

        # Now check — already_attempted must be True
        rc = main([
            'check',
            '--task-id', '42',
            '--worktree', str(repo),
            '--project-root', str(root),
            '--now', self._NOW_ISO,
        ])
        assert rc == 0
        out = capsys.readouterr().out.strip()
        data = json.loads(out)
        assert data['already_attempted'] is True

    def test_charge_succeeds_remaining_five(self, tmp_path, capsys):
        """charge verb: first charge prints {charged:True, remaining:5} (default cap=6)."""
        from orchestrator.b3_gate import main
        _, root = _seed_fresh_check_scene(tmp_path)

        rc = main([
            'charge',
            '--task-id', '42',
            '--project-root', str(root),
            '--now', self._NOW_ISO,
        ])
        assert rc == 0
        out = capsys.readouterr().out.strip()
        data = json.loads(out)
        assert data['charged'] is True
        assert data['remaining'] == 5

    def test_each_verb_prints_single_parseable_json(self, tmp_path, capsys):
        """Each verb call prints exactly one parseable JSON line to stdout."""
        from orchestrator.b3_gate import main
        repo, root = _seed_fresh_check_scene(tmp_path)

        for argv in [
            ['check', '--task-id', '42', '--worktree', str(repo),
             '--project-root', str(root), '--now', self._NOW_ISO],
            ['record-launch', '--task-id', '42', '--worktree', str(repo),
             '--project-root', str(root)],
            ['charge', '--task-id', '42', '--project-root', str(root),
             '--now', self._NOW_ISO],
        ]:
            rc = main(argv)
            assert rc == 0, f'expected exit 0 for {argv[0]}'
            out = capsys.readouterr().out.strip()
            # Must be a single parseable JSON object
            parsed = json.loads(out)
            assert isinstance(parsed, dict), f'expected dict, got {type(parsed)}'
            # stdout must not have stray lines
            assert '\n' not in out.strip() or out.strip().count('\n') == 0, \
                f'multiple lines in stdout for {argv[0]}'


# ---------------------------------------------------------------------------
# step-5 (task 2509): run_check reads the task description and gates on it
# ---------------------------------------------------------------------------

class TestRunCheckStopInstruction:
    """run_check must READ the task's description from tasks.db and forward
    it to check_proposal as extra_texts — an explicit stop instruction in
    the description hard-aborts even an otherwise-fresh low-risk proposal
    (task 2509).
    """

    _NOW_ISO = '2026-06-04T12:00:00+00:00'

    def test_stop_instruction_in_description_aborts(self, tmp_path, capsys):
        """Case A: description contains 'do not apply' -> run_check prints
        verdict=='abort' with a stop-instruction reason."""
        from orchestrator.b3_gate import main
        repo, root = _seed_fresh_check_scene(
            tmp_path, description='Investigate the failure, but do not apply any fix yet.',
        )

        rc = main([
            'check',
            '--task-id', '42',
            '--worktree', str(repo),
            '--project-root', str(root),
            '--category', 'task_failure',
            '--now', self._NOW_ISO,
        ])
        assert rc == 0, f'expected exit 0, got {rc}'

        data = json.loads(capsys.readouterr().out.strip())
        assert data['verdict'] == 'abort', f'expected abort, got {data}'
        assert 'stop instruction' in data['reason'].lower(), (
            f'expected "stop instruction" in reason: {data["reason"]!r}'
        )

    def test_benign_description_verdict_unchanged(self, tmp_path, capsys):
        """Case B (regression): a benign description leaves the verdict exactly
        what the proposal+git would yield today (fresh)."""
        from orchestrator.b3_gate import main
        repo, root = _seed_fresh_check_scene(
            tmp_path, description='This task is blocked on a flaky integration test.',
        )

        rc = main([
            'check',
            '--task-id', '42',
            '--worktree', str(repo),
            '--project-root', str(root),
            '--category', 'task_failure',
            '--now', self._NOW_ISO,
        ])
        assert rc == 0, f'expected exit 0, got {rc}'

        data = json.loads(capsys.readouterr().out.strip())
        assert data['verdict'] == 'fresh', f'expected fresh, got {data}'


# ---------------------------------------------------------------------------
# review amendment: run_check shares one sqlite connection (efficiency)
# ---------------------------------------------------------------------------

class TestRunCheckSharedConnection:
    """run_check must open tasks.db at most once per invocation — the proposal
    and description reads share a single connection (task 2509 review
    amendment) rather than each opening (and closing) its own.
    """

    _NOW_ISO = '2026-06-04T12:00:00+00:00'

    def test_run_check_opens_tasks_db_exactly_once(self, tmp_path, capsys, monkeypatch):
        from orchestrator import b3_gate
        from orchestrator.b3_gate import main

        repo, root = _seed_fresh_check_scene(tmp_path, description='benign description')

        real_connect = sqlite3.connect
        connect_calls: list[str] = []

        def _counting_connect(database, *args, **kwargs):
            connect_calls.append(database)
            return real_connect(database, *args, **kwargs)

        monkeypatch.setattr(b3_gate.sqlite3, 'connect', _counting_connect)

        rc = main([
            'check',
            '--task-id', '42',
            '--worktree', str(repo),
            '--project-root', str(root),
            '--category', 'task_failure',
            '--now', self._NOW_ISO,
        ])
        assert rc == 0, f'expected exit 0, got {rc}'

        # tasks.db-targeted opens only — the state-file path (b3-state.json)
        # never goes through sqlite3.connect, so any hit here is a tasks.db read.
        tasks_db_opens = [c for c in connect_calls if 'tasks.db' in c]
        assert len(tasks_db_opens) == 1, (
            f'expected exactly one tasks.db connection, got {len(tasks_db_opens)}: '
            f'{tasks_db_opens!r}'
        )

        data = json.loads(capsys.readouterr().out.strip())
        assert data['verdict'] == 'fresh', f'expected fresh, got {data}'
