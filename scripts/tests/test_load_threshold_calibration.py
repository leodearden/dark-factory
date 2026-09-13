"""Tests for scripts/load-threshold-calibration.py (task 3592, leaf δ of
plans/load-throttle-harmonisation-prd.md; the gate ε1/ε2 run).

Every input is INJECTED — a seeded temp DB, temp yaml files, a defaults
mapping — so no test reads the live corpus, either project's committed
config, or the orchestrator's code. Nothing here imports `sampler`: this suite
runs under `--project shared`, where a probe showed sibling workspace members
can be absent from the venv, and the script under test must itself run under
the system python3 where none of them exist.
"""
from __future__ import annotations

import importlib.util
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]
SCRIPT = REPO_ROOT / 'scripts' / 'load-threshold-calibration.py'

# The real store schema, copied here rather than imported: this suite cannot
# rely on `sampler` being importable (see the module docstring). The lockstep
# guard against sampler.store lives in the sampler suite, where both packages
# do import.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    ts INTEGER NOT NULL,
    metric TEXT NOT NULL,
    value REAL NOT NULL,
    window_mean REAL,
    window_max REAL
);
CREATE INDEX IF NOT EXISTS idx_samples_metric_ts ON samples (metric, ts);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def load_script():
    """Load the hyphen-named script as a module.

    This must work under a STDLIB-ONLY module body: at ε1/ε2 time the
    `#!/usr/bin/env python3` shebang resolves to /usr/bin/python3, which has
    neither `shared` nor `sampler` nor `orchestrator`. A top-level first-party
    import would crash the gate on import fourteen days after this lands, in a
    born-at-L2 escalation path, with no earlier signal.
    """
    spec = importlib.util.spec_from_file_location('load_threshold_calibration', SCRIPT)
    assert spec is not None, f'Could not build spec from {SCRIPT}'
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def seed_db(path: Path, series: dict[str, list[float]], *, start_ts: int = 1_000_000,
            step: int = 5) -> Path:
    """Write a DB with the real schema and one row per value, spaced `step` s."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(_SCHEMA)
        for metric, values in series.items():
            conn.executemany(
                'INSERT INTO samples (ts, metric, value) VALUES (?, ?, ?)',
                [(start_ts + i * step, metric, v) for i, v in enumerate(values)],
            )
        conn.commit()
    finally:
        conn.close()
    return path


def run_script(*argv: str):
    """Run the script as a subprocess, the way the gate will."""
    return subprocess.run(
        [sys.executable, str(SCRIPT), *argv],
        capture_output=True, text=True, timeout=120,
    )


def trailing_json(stdout: str) -> dict:
    """The script's contract: stdout's LAST line parses as JSON."""
    return json.loads(stdout.strip().splitlines()[-1])


# ── the module body is stdlib-only ──────────────────────────────────────────


def test_script_is_executable():
    """ε1/ε2 name this path in metadata.before_done.script, which validates
    path-exists-AND-executable at submit_task time."""
    import os

    assert os.access(SCRIPT, os.X_OK), (
        f'Expected {SCRIPT} to be executable (os.X_OK); run: chmod +x {SCRIPT}')


def test_module_body_imports_no_yaml_and_no_first_party_package():
    """Loading the module must not execute a yaml or first-party import.

    yaml is imported LAZILY inside the drift check so a missing PyYAML becomes
    a named degradation in the report instead of an import crash, and so any
    test in any subproject can load this module by path.
    """
    module = load_script()

    assert not hasattr(module, 'yaml'), (
        'yaml must be imported lazily inside the drift check, not at module level')
    for name in ('shared', 'sampler', 'orchestrator'):
        assert not hasattr(module, name), (
            f'{name} must not be imported: at gate time this script runs under '
            'the system python3, which does not have it')


def test_source_carries_no_first_party_top_level_import():
    """Belt and braces on the same fact, read off the source.

    hasattr alone would miss `from shared.psi import _ARMS`, which binds
    `_ARMS` rather than `shared`.
    """
    source = SCRIPT.read_text()
    body = [
        line for line in source.splitlines()
        if line.startswith(('import ', 'from ')) and not line.lstrip().startswith('#')
    ]
    for line in body:
        for pkg in ('shared', 'sampler', 'orchestrator', 'yaml'):
            assert not line.startswith((f'import {pkg}', f'from {pkg}')), line


# ── the argument surface ────────────────────────────────────────────────────


def test_main_accepts_argv_in_process_and_returns_zero(tmp_path: Path):
    module = load_script()
    db = seed_db(tmp_path / 'db.sqlite', {'runqueue_ratio': [1.0, 2.0, 3.0]})

    rc = module.main(['--db', str(db), '--no-report'])

    assert rc == 0


def test_every_documented_flag_is_accepted(tmp_path: Path):
    module = load_script()
    db = seed_db(tmp_path / 'db.sqlite', {'runqueue_ratio': [1.0, 2.0]})

    rc = module.main([
        '--db', str(db),
        '--arm', 'runqueue_ratio',
        '--config', str(tmp_path / 'missing-local.yaml'),
        '--peer-config', str(tmp_path / 'missing-peer.yaml'),
        '--report-dir', str(tmp_path),
        '--no-report',
    ])

    assert rc == 0
    # --commit is exercised against a throwaway git repo in the report tests;
    # here it only has to PARSE, which parse_args proves without running it.
    assert module.parse_args(['--commit']).commit is True


def test_peer_config_defaults_to_the_reify_checkout():
    module = load_script()

    args = module.parse_args(['--no-report'])

    assert str(args.peer_config) == '/home/leo/src/reify/dark-factory-orchestrator.yaml'


def test_db_defaults_to_the_dark_factory_root_seam(monkeypatch):
    """The same env seam sampler/__main__.py and the installer honour."""
    module = load_script()
    monkeypatch.setenv('DARK_FACTORY_ROOT', '/tmp/some-root')

    args = module.parse_args(['--no-report'])

    assert str(args.db) == '/tmp/some-root/data/load-samples.db'


def test_an_unknown_arm_is_rejected_by_argparse():
    """Rather than silently producing an empty report."""
    module = load_script()

    with pytest.raises(SystemExit):
        module.parse_args(['--arm', 'not_an_arm'])


# ── the percentile section ──────────────────────────────────────────────────


def test_percentiles_are_reported_per_metric_including_colon_stems(tmp_path: Path):
    db = seed_db(tmp_path / 'db.sqlite', {
        'runqueue_ratio': [float(i) for i in range(1, 101)],
        'own_cpu_some10:orchestrator-reify.service': [float(i) for i in range(1, 101)],
    })

    result = run_script('--db', str(db), '--no-report')
    assert result.returncode == 0, result.stderr

    payload = trailing_json(result.stdout)
    percentiles = payload['percentiles']
    assert 'runqueue_ratio' in percentiles
    assert 'own_cpu_some10:orchestrator-reify.service' in percentiles
    for stats in percentiles.values():
        assert set(stats) >= {'n', 'p50', 'p90', 'p95', 'p99', 'max'}
        assert stats['n'] == 100
        assert stats['max'] == pytest.approx(100.0)
        assert stats['p50'] == pytest.approx(50.0, abs=1.0)
        assert stats['p90'] == pytest.approx(90.0, abs=1.0)
        assert stats['p99'] == pytest.approx(99.0, abs=1.0)


def test_the_human_report_reaches_stdout_before_the_json(tmp_path: Path):
    db = seed_db(tmp_path / 'db.sqlite', {'runqueue_ratio': [1.0, 2.0, 3.0]})

    result = run_script('--db', str(db), '--no-report')

    assert result.returncode == 0, result.stderr
    lines = result.stdout.strip().splitlines()
    assert lines[0].startswith('# '), lines[0]
    trailing_json(result.stdout)  # must parse


def test_an_empty_db_exits_zero_with_a_named_degradation(tmp_path: Path):
    """Never a crash and never a report of zeros — the corpus being empty is a
    fact about the corpus, and ε1/ε2's reader has to be told it."""
    db = seed_db(tmp_path / 'db.sqlite', {})

    result = run_script('--db', str(db), '--no-report')

    assert result.returncode == 0, result.stderr
    payload = trailing_json(result.stdout)
    assert 'no_samples_in_window' in payload['degradations'], payload
    assert payload['percentiles'] == {}


def test_a_missing_db_exits_zero_with_a_named_degradation(tmp_path: Path):
    result = run_script('--db', str(tmp_path / 'nope.sqlite'), '--no-report')

    assert result.returncode == 0, result.stderr
    payload = trailing_json(result.stdout)
    assert 'db_unavailable' in payload['degradations'], payload
