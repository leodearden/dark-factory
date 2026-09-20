"""Tests for scripts/load-threshold-calibration.py (task 3592, leaf δ of
plans/load-throttle-harmonisation-prd.md; the gate ε1/ε2 run).

Every input is INJECTED — a seeded temp DB, temp yaml files, a defaults
mapping — so no test reads the live corpus, and the two defaults that DO reach
outside this worktree (the peer project's config, and the uv that fetches the
orchestrator's code defaults by running it) are redirected at absent paths
unless a test names them. ``load_script`` and ``run_script`` do that
redirecting and explain it; exactly two tests opt out, and both say so in
their docstrings. That claim used to be aspirational — ~25 tests that merely
omitted `--uv-bin` spawned `uv run --project orchestrator` against the MAIN
checkout, ~20 more read the real /home/leo/src/reify config, and the suite
took 37 s to say so.

Nothing here imports `sampler`: this suite
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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).parents[2]
SCRIPT = REPO_ROOT / 'scripts' / 'load-threshold-calibration.py'

# The real store schema, copied here rather than imported: this suite cannot
# rely on `sampler` being importable (see the module docstring).
#
# NOTHING IN THIS SUITE RECONCILES THIS COPY, and it cannot: reconciling means
# importing the writer. Every test below reads a DB built from this fixture, so
# if sampler.store's schema moved, all of them would stay green against a shape
# the store no longer writes -- measured, by renaming the `metric` column in
# sampler.store: 73 green here, and the gate returning no_samples_in_window
# against the live corpus.
#
# The reconciler is therefore in the SAMPLER suite, which can import both sides:
# sampler/tests/test_load_metrics.py::TestCalibrationScriptArmTableLockstep
# ::test_the_script_reads_a_corpus_THIS_STORE_wrote drives THIS script's
# read_series over a DB the real store wrote. It guards from the writer's side
# rather than comparing two DDL strings, so a reformatted CREATE TABLE does not
# trip it and a column the store never had does not slip past.
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


# The script's two defaults that reach OUTSIDE this worktree: the peer
# project's committed config, and the uv that fetches the orchestrator's code
# defaults by running it. Absent paths, because "absent" is a contract this
# script already handles by name (`peer_config_missing`,
# `code_defaults_unavailable`) — so redirecting them costs a test nothing but
# the degradation it was already free to ignore.
_ABSENT_PEER_CONFIG = Path('/nonexistent/hermetic-peer/dark-factory-orchestrator.yaml')
_ABSENT_UV_BIN = Path('/nonexistent/hermetic-uv/uv')

REAL_PEER_CONFIG = Path('/home/leo/src/reify/dark-factory-orchestrator.yaml')


def load_script(*, real_host_defaults: bool = False):
    """Load the hyphen-named script as a module, hermetic by default.

    This must work under a STDLIB-ONLY module body: at ε1/ε2 time the
    `#!/usr/bin/env python3` shebang resolves to /usr/bin/python3, which has
    neither `shared` nor `sampler` nor `orchestrator`. A top-level first-party
    import would crash the gate on import fourteen days after this lands, in a
    born-at-L2 escalation path, with no earlier signal.

    The two host-reaching argparse defaults are redirected at absent paths
    unless *real_host_defaults* asks for the shipped ones. Without that, a
    test that merely omits `--peer-config` reads /home/leo/src/reify and one
    that omits `--uv-bin` spawns `uv run --project orchestrator` against the
    MAIN checkout — neither of which the test asked for, both of which the
    module docstring above promises do not happen, and together ~1 s of every
    such test. `parse_args` reads both through the module globals at call
    time, so patching the loaded module IS the production seam rather than a
    parallel one.
    """
    spec = importlib.util.spec_from_file_location('load_threshold_calibration', SCRIPT)
    assert spec is not None, f'Could not build spec from {SCRIPT}'
    assert spec.loader is not None
    # `Any` rather than the inferred ModuleType: pyright knows no attribute a
    # by-path-loaded module defines, so it admits every `module.<name>` READ
    # below through ModuleType.__getattr__ but rejects the two assignments. This
    # checks them the way the reads are already checked. Not `setattr`: bugbear
    # B010 is selected for this directory by the root [tool.ruff].
    module: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    if not real_host_defaults:
        module.DEFAULT_PEER_CONFIG = _ABSENT_PEER_CONFIG
        module.DEFAULT_UV_BIN = _ABSENT_UV_BIN
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
    """Run the script as a subprocess, the way the gate will — hermetically.

    A subprocess cannot inherit the loader's redirected defaults, so the same
    two flags are supplied here instead, and only when the caller has not
    named them itself. See ``load_script`` for why they are redirected at all.
    """
    return subprocess.run(
        [sys.executable, str(SCRIPT), *hermetic(argv)],
        capture_output=True, text=True, timeout=120,
    )


def hermetic(argv) -> list[str]:
    """*argv* with the two host-reaching flags defaulted to absent paths."""
    out = list(argv)
    for flag, absent in (('--peer-config', _ABSENT_PEER_CONFIG),
                         ('--uv-bin', _ABSENT_UV_BIN)):
        if flag not in out:
            out += [flag, str(absent)]
    return out


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


_LOAD_UNDER_AN_IMPORT_BLOCKER = '''
import importlib.util, sys

FORBIDDEN = {'shared', 'sampler', 'orchestrator', 'yaml'}


class Blocker:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in FORBIDDEN:
            raise AssertionError('module-level import of ' + fullname)
        return None


sys.meta_path.insert(0, Blocker())
spec = importlib.util.spec_from_file_location('calibration_under_test', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print('loaded')
'''


def test_loading_the_script_executes_no_first_party_or_yaml_import():
    """The gate-time property, EXECUTED rather than grepped off the source.

    At gate time the shebang resolves to the system python3, which has yaml
    but none of `shared`, `sampler`, `orchestrator`. A top-level first-party
    import would crash ε1/ε2 on import, fourteen days after this lands, in a
    born-at-L2 path with no earlier signal.

    This replaces a rescan of the source text for lines beginning `import x` /
    `from x`, which was strictly weaker in both directions: it missed indented
    and `__import__` forms, and it could only ever report what the source LOOKS
    like. Loading the module behind a meta-path finder that raises reports what
    the module DOES, in every form an import can take. It also closes the hole
    the scan existed for — `from shared.psi import _ARMS` binds `_ARMS`, not
    `shared`, so the hasattr check above cannot see it.
    """
    result = subprocess.run(
        [sys.executable, '-c', _LOAD_UNDER_AN_IMPORT_BLOCKER, str(SCRIPT)],
        capture_output=True, text=True, check=False,
    )

    assert result.returncode == 0, result.stderr
    assert 'loaded' in result.stdout, result.stdout


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
    module = load_script(real_host_defaults=True)

    args = module.parse_args(['--no-report'])

    assert args.peer_config == REAL_PEER_CONFIG


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


def test_the_percentile_is_nearest_rank_with_a_tie_break_that_does_not_move():
    """Two decided properties of `pct`, neither derivable from the ladder tests.

    NEAREST-RANK: every reported figure is a value the corpus contains. The
    ladder these feed is a threshold ladder, so an interpolated p99 would name a
    number no tick produced. `[1, 2]` proves it -- an interpolating p50 is 1.5,
    which is not in the input.

    A TIE-BREAK THAT DOES NOT DEPEND ON n: the index lands on an exact .5 for
    even-sized inputs, and `round` is banker's, so this alternated between the
    lower and upper median as n changed -- 1 for `[1, 2]` but 3 for
    `[1, 2, 3, 4]`. Pinned at both sizes, because one alone cannot see it.
    """
    module = load_script()

    assert module.pct([1.0, 2.0], 0.5) == 2.0
    assert module.pct([1.0, 2.0, 3.0, 4.0], 0.5) == 3.0
    assert module.pct([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 0.5) == 4.0
    # Non-tie indices are untouched by the tie-break rule, and an empty input
    # is NaN rather than a crash or a fabricated 0.0.
    assert module.pct([float(i) for i in range(1, 101)], 0.99) == 99.0
    assert module.pct([], 0.5) != module.pct([], 0.5)  # NaN


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


# ── D11: hold fraction and hold streak ──────────────────────────────────────


def test_hold_fraction_uses_ge_matching_the_gate(tmp_path: Path):
    """`>=`, not `>`, mirroring shared.psi's arm comparison.

    A report that disagreed with the gate on the boundary would recommend a
    threshold the gate then behaves differently at.
    """
    module = load_script()

    assert module.hold_fraction([1.0, 2.0, 3.0, 4.0], 3.0) == pytest.approx(0.5)
    # Every sample exactly AT the candidate holds.
    assert module.hold_fraction([3.0, 3.0, 3.0], 3.0) == pytest.approx(1.0)


def test_hold_fraction_saturates_at_both_ends(tmp_path: Path):
    module = load_script()
    values = [1.0, 2.0, 3.0]

    assert module.hold_fraction(values, 99.0) == pytest.approx(0.0)
    assert module.hold_fraction(values, 1.0) == pytest.approx(1.0)
    assert module.hold_fraction([], 1.0) == pytest.approx(0.0)


def test_each_candidate_is_labelled_against_the_twenty_percent_target():
    """D11: "the gate holds on a minority of ticks (target <= 20%)"."""
    module = load_script()

    assert module.within_d11_target(0.05) is True
    assert module.within_d11_target(0.20) is True
    assert module.within_d11_target(0.35) is False


def test_longest_hold_run_is_reported_in_ticks_and_wall_clock():
    module = load_script()
    # 240 consecutive holds at the pinned 5 s cadence = 20 minutes.
    values = [0.0] * 10 + [9.0] * 240 + [0.0] * 10
    points = [(1000 + i * 5, v) for i, v in enumerate(values)]

    run = module.longest_hold_run(points, 5.0, spacing_seconds=5)

    assert run['ticks'] == 240
    assert run['seconds'] == 1200
    assert run['human'] == '20m'


def test_one_long_block_is_distinguished_from_the_same_fraction_as_blips():
    """The whole reason D11's second clause is reported separately.

    A fraction alone cannot tell 20% delivered as single-tick blips from 20%
    delivered as one continuous block, and those are opposite verdicts for a
    dispatch throttle.
    """
    module = load_script()
    block = [0.0] * 800 + [9.0] * 200
    blips = [9.0 if i % 5 == 0 else 0.0 for i in range(1000)]

    def at_cadence(values):
        return [(1000 + i * 5, v) for i, v in enumerate(values)]

    assert module.hold_fraction(block, 5.0) == pytest.approx(0.2)
    assert module.hold_fraction(blips, 5.0) == pytest.approx(0.2)
    assert module.longest_hold_run(
        at_cadence(block), 5.0, spacing_seconds=5)['ticks'] == 200
    assert module.longest_hold_run(
        at_cadence(blips), 5.0, spacing_seconds=5)['ticks'] == 1


def test_a_run_is_not_welded_across_a_sampler_outage():
    """The number D11's second clause is read off must not span a gap.

    The run walked values alone and multiplied the count by the median
    spacing, so it could not see a hole in the corpus. Holds at ts
    1000/1005/1010, a multi-hour outage, then holds at 100000/100005 reported
    as one run of "5 ticks (25s)" for an interval actually spanning ~27 h.
    "Never sits at the floor for hours" is exactly the question this answers,
    so welding across an outage is wrong in the direction that matters.
    """
    module = load_script()
    points = [
        (1000, 9.0), (1005, 9.0), (1010, 9.0),
        (100_000, 9.0), (100_005, 9.0),          # after a ~27 h outage
    ]

    run = module.longest_hold_run(points, 5.0, spacing_seconds=5)

    assert run['ticks'] == 3, (
        f'the outage was welded into the run: {run}'
    )
    assert run['seconds'] == 15


def test_wall_clock_is_read_off_the_timestamps_not_multiplied_from_ticks():
    """A run whose ticks are late reports the time it actually spanned."""
    module = load_script()
    # Five holding samples, but the sampler ran late: 12 s between them, still
    # inside the gap tolerance, so it is one run spanning 48 s + its own tick.
    points = [(1000 + i * 12, 9.0) for i in range(5)]

    run = module.longest_hold_run(points, 5.0, spacing_seconds=5)

    assert run['ticks'] == 5
    assert run['seconds'] == 53, (
        'ticks x nominal spacing would have reported 25 s for an interval that '
        f'actually spanned 48 s of wall clock: {run}'
    )


def test_a_single_tick_hold_still_reports_its_own_tick_of_wall_clock():
    """`ts_end - ts_start` alone would report 0s for a hold that did happen."""
    module = load_script()

    run = module.longest_hold_run([(1000, 9.0)], 5.0, spacing_seconds=5)

    assert run == {'ticks': 1, 'seconds': 5, 'human': '5s'}


def test_observed_spacing_is_measured_from_the_corpus_not_assumed(tmp_path: Path):
    """A 5 s cadence is the unit's setting, not a property of the corpus."""
    module = load_script()

    assert module.observed_spacing([100, 105, 110, 115]) == 5
    assert module.observed_spacing([100, 130, 160]) == 30
    # Degenerate inputs must not raise; they fall back to the unit's cadence.
    assert module.observed_spacing([100]) == 5
    assert module.observed_spacing([]) == 5


def test_the_ladder_lands_in_the_trailing_json(tmp_path: Path):
    """ε1/ε2's escalation is what a human actually reads."""
    db = seed_db(tmp_path / 'db.sqlite', {
        'runqueue_ratio': [0.5] * 80 + [6.0] * 20,
    })

    result = run_script('--db', str(db), '--arm', 'runqueue_ratio', '--no-report')
    assert result.returncode == 0, result.stderr

    payload = trailing_json(result.stdout)
    candidates = payload['holds']['runqueue_ratio']
    by_threshold = {c['threshold']: c for c in candidates}
    assert set(by_threshold) == {1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0}

    at_six = by_threshold[6.0]
    assert at_six['hold_fraction'] == pytest.approx(0.2)
    assert at_six['within_d11_target'] is True
    assert at_six['longest_hold_run']['ticks'] == 20
    assert at_six['longest_hold_run']['seconds'] == 100

    at_eight = by_threshold[8.0]
    assert at_eight['hold_fraction'] == pytest.approx(0.0)


def test_a_stem_arm_reports_per_leaf_and_is_never_pooled(tmp_path: Path):
    """Averaging seven unrelated projects' CPU pressure describes nothing."""
    db = seed_db(tmp_path / 'db.sqlite', {
        'own_cpu_some10:orchestrator-reify.service': [90.0] * 100,
        'own_cpu_some10:orchestrator-dark-factory.service': [1.0] * 100,
    })

    result = run_script('--db', str(db), '--arm', 'own_cpu_some_avg10', '--no-report')
    assert result.returncode == 0, result.stderr

    payload = trailing_json(result.stdout)
    holds = payload['holds']
    assert set(holds) == {
        'own_cpu_some10:orchestrator-reify.service',
        'own_cpu_some10:orchestrator-dark-factory.service',
    }
    hot = {c['threshold']: c for c in holds['own_cpu_some10:orchestrator-reify.service']}
    cold = {c['threshold']: c
            for c in holds['own_cpu_some10:orchestrator-dark-factory.service']}
    assert hot[80.0]['hold_fraction'] == pytest.approx(1.0)
    assert cold[80.0]['hold_fraction'] == pytest.approx(0.0)


def test_an_unwritable_report_dir_is_a_named_degradation_not_a_traceback(
    tmp_path: Path,
):
    """The one boundary that was not honouring the always-exit-0 contract.

    ε1/ε2 are always_escalates with no target_unit, so a non-zero rc is
    classified an INFRA FAULT and born at L2 — a human paged about a "broken
    script" instead of reading the calibration that already ran. Every other
    boundary in the file (read_series, load_psi_admission_block,
    fetch_code_defaults, commit_report) already returns a named degradation.
    """
    db = seed_db(tmp_path / 'db.sqlite', {'runqueue_ratio': [1.0, 2.0, 3.0]})
    missing = tmp_path / 'no-such-dir' / 'nor-this-one'

    result = run_script('--db', str(db), '--arm', 'runqueue_ratio',
                        '--report-dir', str(missing))

    assert result.returncode == 0, (
        f'a missing --report-dir turned a completed analysis into rc='
        f'{result.returncode}: {result.stderr}'
    )
    payload = trailing_json(result.stdout)
    assert 'report_unwritable' in payload['degradations'], payload['degradations']
    detail = next(d for d in payload['degradation_details']
                  if d.startswith('report_unwritable'))
    assert str(missing) in detail, detail
    # The analysis itself is still delivered, not swallowed with the write.
    assert payload['percentiles']['runqueue_ratio']['n'] == 3


# ── a hold fraction is meaningless without the coverage it was computed over ──


def test_read_series_counts_the_tick_clock_rather_than_loading_it(tmp_path: Path):
    """The corpus tick count is a COUNT(*), not a series fetched for its len().

    Driven in-process because the fact under test is a property of
    ``read_series``'s return value that the trailing JSON cannot show: the
    payload reports the same count either way, and only the returned
    readability dict says whether the clock's rows were MATERIALISED to get it.

    The cost this defends is measured, not stylistic. ``runqueue_read_ok`` is
    written on every completed tick, so at the 30-day steady state the script's
    own docstring cites it is ~518k ``(ts, value)`` tuples — built into a list
    purely so ``coverage_table`` could take its ``len()``. The ε2 cut
    (``--arm own_cpu_some_avg10``) and the four PSI arms never read that series
    at all, and none of them DECLARES the clock as its readability metric; the
    runqueue arm still gets it as a series because it does. The count itself is
    served index-only by ``idx_samples_metric_ts`` — measured plan against this
    exact schema: ``SEARCH samples USING COVERING INDEX idx_samples_metric_ts
    (metric=?)``, no temp B-tree.
    """
    module = load_script()
    db = seed_db(tmp_path / 'db.sqlite', {
        'runqueue_read_ok': [1.0] * 100,
        'own_read_ok:leaf.service': [1.0] * 100,
        'own_cpu_some10:leaf.service': [30.0] * 100,
    })

    read = module.read_series(db, 'own_cpu_some_avg10')

    assert read.degradations == [], read.degradations
    assert read.ticks_in_corpus == 100
    assert module.TICK_METRIC not in read.readability, (
        'the tick clock was materialised as a series for an arm that does not '
        f'declare it as its readability metric: {sorted(read.readability)}'
    )
    # The evidence the arm DID ask for is still fetched, both halves of it.
    assert len(read.readability['own_read_ok:leaf.service']) == 100
    assert len(read.series['own_cpu_some10:leaf.service']) == 100



def test_hold_fraction_is_reported_beside_its_readable_tick_coverage(tmp_path: Path):
    """The denominator is SUCCESSFUL reads, not ticks, so coverage must ship too.

    A failed read emits no value row at all (correctly — persisting α's
    fail-open 0.0 would fabricate an idle host), so hold_fraction divides by the
    number of readable ticks. "Holds on 20% of samples" over a fully-observed
    fortnight and over the 10% of ticks that were readable are opposite verdicts
    for setting a dispatch threshold, and the report could not tell them apart.
    """
    db = seed_db(tmp_path / 'db.sqlite', {
        # runqueue_read_ok is the tick clock: one row every tick, readable or not.
        'runqueue_read_ok': [1.0] * 10 + [0.0] * 90,
        'runqueue_ratio': [5.0] * 10,
    })

    result = run_script('--db', str(db), '--arm', 'runqueue_ratio', '--no-report')
    assert result.returncode == 0, result.stderr

    coverage = trailing_json(result.stdout)['coverage']['runqueue_ratio']
    assert coverage['ticks_in_corpus'] == 100
    assert coverage['ticks_with_a_row'] == 100
    assert coverage['readable'] == 10
    assert coverage['readable_fraction'] == pytest.approx(0.1)


def test_coverage_below_the_floor_is_a_named_degradation(tmp_path: Path):
    """Makes the D11 verdict self-checking instead of silently unsound."""
    db = seed_db(tmp_path / 'db.sqlite', {
        'runqueue_read_ok': [1.0] * 3 + [0.0] * 97,
        'runqueue_ratio': [5.0] * 3,
    })

    result = run_script('--db', str(db), '--arm', 'runqueue_ratio', '--no-report')
    assert result.returncode == 0, result.stderr

    payload = trailing_json(result.stdout)
    assert 'low_readability' in payload['degradations'], payload['degradations']
    # The clock is this arm's own readability metric, so it is present on
    # every tick by construction and its shortfall is all failed reads.
    assert 'partial_presence' not in payload['degradations'], payload['degradations']
    detail = next(d for d in payload['degradation_details']
                  if d.startswith('low_readability'))
    assert 'runqueue_ratio' in detail and '3/100' in detail, detail


def test_full_coverage_raises_no_readability_degradation(tmp_path: Path):
    db = seed_db(tmp_path / 'db.sqlite', {
        'runqueue_read_ok': [1.0] * 50,
        'runqueue_ratio': [5.0] * 50,
    })

    result = run_script('--db', str(db), '--arm', 'runqueue_ratio', '--no-report')
    assert result.returncode == 0, result.stderr

    payload = trailing_json(result.stdout)
    assert 'low_readability' not in payload['degradations'], payload['degradations']
    assert payload['coverage']['runqueue_ratio']['readable_fraction'] == pytest.approx(1.0)


def test_a_stem_arm_reports_coverage_per_leaf_joined_on_the_leaf_tail(tmp_path: Path):
    """own_read_ok:<leaf> is the coverage for own_cpu_some10:<leaf>, per leaf.

    One leaf can be unreadable while its siblings are fine, so pooling the
    stem's coverage would hide exactly the case worth seeing.
    """
    db = seed_db(tmp_path / 'db.sqlite', {
        'runqueue_read_ok': [1.0] * 20,
        'own_read_ok:orchestrator-reify.service': [1.0] * 20,
        'own_cpu_some10:orchestrator-reify.service': [30.0] * 20,
        'own_read_ok:orchestrator-know-live.service': [1.0] * 2 + [0.0] * 18,
        'own_cpu_some10:orchestrator-know-live.service': [30.0] * 2,
    })

    result = run_script('--db', str(db), '--arm', 'own_cpu_some_avg10', '--no-report')
    assert result.returncode == 0, result.stderr

    coverage = trailing_json(result.stdout)['coverage']
    assert coverage['own_cpu_some10:orchestrator-reify.service'][
        'readable_fraction'] == pytest.approx(1.0)
    assert coverage['own_cpu_some10:orchestrator-know-live.service'] == {
        'ticks_in_corpus': 20, 'ticks_with_a_row': 20,
        'readable': 2, 'readable_fraction': 0.1,
        # Named, so a reader of the escalation can tell WHICH series was
        # counted — the per-leaf join is the thing this test is about.
        'readability_metric': 'own_read_ok:orchestrator-know-live.service',
    }


def seed_leaf_present_for_the_last_fifth(path: Path, *, readable: int = 20) -> Path:
    """A 100-tick corpus in which one leaf exists only for the last 20 ticks.

    The shape a restarted, added or renamed ``orchestrator-*.service`` leaves
    behind, and the one task 3394's ``df-*.slice`` migration produces
    mid-corpus: the sampler writes ``own_read_ok:<leaf>`` only on ticks it
    DISCOVERED the leaf, so the late leaf has 20 rows — *readable* of them
    readable — while the tick clock and the steady leaf have 100.
    """
    seed_db(path, {
        'runqueue_read_ok': [1.0] * 100,
        'own_read_ok:orchestrator-reify.service': [1.0] * 100,
        'own_cpu_some10:orchestrator-reify.service': [30.0] * 100,
    })
    return seed_db(path, {
        'own_read_ok:orchestrator-new.service':
            [1.0] * readable + [0.0] * (20 - readable),
        'own_cpu_some10:orchestrator-new.service': [30.0] * readable,
    }, start_ts=1_000_000 + 80 * 5)


def details(payload: dict, cause: str) -> list[str]:
    return [d for d in payload['degradation_details'] if d.startswith(cause)]


def test_a_leaf_present_for_part_of_the_corpus_is_covered_over_all_of_it(
    tmp_path: Path,
):
    """The denominator is the corpus tick count, not the leaf's own row count.

    Dividing by the leaf's own ``own_read_ok`` rows reported 20/20 = 100% for a
    leaf observed over a fifth of the window, and the floor never fired — the
    misreading the coverage block exists to prevent, delivered to the human
    setting a production threshold. Run with ``--arm own_cpu_some_avg10`` on
    purpose: the clock belongs to the runqueue arm, and must be read even when
    that arm is not the one asked for.
    """
    db = seed_leaf_present_for_the_last_fifth(tmp_path / 'db.sqlite')

    result = run_script('--db', str(db), '--arm', 'own_cpu_some_avg10', '--no-report')
    assert result.returncode == 0, result.stderr

    payload = trailing_json(result.stdout)
    assert payload['coverage']['own_cpu_some10:orchestrator-new.service'] == {
        'ticks_in_corpus': 100, 'ticks_with_a_row': 20,
        'readable': 20, 'readable_fraction': 0.2,
        'readability_metric': 'own_read_ok:orchestrator-new.service',
    }
    assert payload['coverage']['own_cpu_some10:orchestrator-reify.service'][
        'readable_fraction'] == pytest.approx(1.0)
    [absent] = details(payload, 'partial_presence')
    assert 'orchestrator-new.service' in absent and '20/100' in absent, absent


def test_absence_is_not_reported_as_failed_reads(tmp_path: Path):
    """A leaf that was simply not there is not a flaky read.

    Every one of the late leaf's reads succeeded, so ``low_readability`` —
    which sends an operator hunting a failing collector — must stay silent
    while ``partial_presence`` names the span.
    """
    db = seed_leaf_present_for_the_last_fifth(tmp_path / 'db.sqlite')

    result = run_script('--db', str(db), '--arm', 'own_cpu_some_avg10', '--no-report')
    assert result.returncode == 0, result.stderr

    payload = trailing_json(result.stdout)
    assert 'partial_presence' in payload['degradations'], payload['degradations']
    assert 'low_readability' not in payload['degradations'], (
        f'absence was reported as failed reads: {payload["degradation_details"]}'
    )


def test_absence_and_failed_reads_are_each_counted_against_their_own_base(
    tmp_path: Path,
):
    """20 rows of 100 ticks, 10 of them readable: 20/100 absent, 10/20 failed.

    With no failed reads in the fixture, counting presence off READABLE rows
    and counting it off ROWS give the same answer, so a fixture that mixes the
    two is the only one that can tell them apart.
    """
    db = seed_leaf_present_for_the_last_fifth(tmp_path / 'db.sqlite', readable=10)

    result = run_script('--db', str(db), '--arm', 'own_cpu_some_avg10', '--no-report')
    assert result.returncode == 0, result.stderr

    payload = trailing_json(result.stdout)
    coverage = payload['coverage']['own_cpu_some10:orchestrator-new.service']
    assert (coverage['ticks_in_corpus'], coverage['ticks_with_a_row'],
            coverage['readable'], coverage['readable_fraction']) == (100, 20, 10, 0.1)
    [absent] = details(payload, 'partial_presence')
    [failed] = details(payload, 'low_readability')
    assert '20/100' in absent, absent
    assert '10/20' in failed, failed


def seed_leaf_present_on(path: Path, *, present: int, readable: int) -> Path:
    """A 100-tick corpus with one leaf missing from its first 100 - *present*."""
    seed_db(path, {'runqueue_read_ok': [1.0] * 100})
    return seed_db(path, {
        'own_read_ok:orchestrator-reify.service':
            [1.0] * readable + [0.0] * (present - readable),
        'own_cpu_some10:orchestrator-reify.service': [30.0] * readable,
    }, start_ts=1_000_000 + (100 - present) * 5)


@pytest.mark.parametrize(('present', 'readable', 'expected'), [
    # A unit restart: a 3-tick gap is not a finding, however badly it read.
    (97, 50, ['low_readability']),
    # Exactly AT the floor is not below it — presence, then readability.
    (95, 95, []),
    (100, 95, []),
    (94, 94, ['partial_presence']),
])
def test_each_cause_is_judged_on_its_own_ratio_at_the_floor(
    tmp_path: Path, present: int, readable: int, expected: list[str],
):
    db = seed_leaf_present_on(tmp_path / 'db.sqlite', present=present, readable=readable)

    result = run_script('--db', str(db), '--arm', 'own_cpu_some_avg10', '--no-report')
    assert result.returncode == 0, result.stderr

    payload = trailing_json(result.stdout)
    fired = [cause for cause in ('partial_presence', 'low_readability')
             if cause in payload['degradations']]
    assert fired == expected, payload['degradation_details']


def test_the_report_line_names_each_cause_below_the_floor(tmp_path: Path):
    db = seed_leaf_present_for_the_last_fifth(tmp_path / 'db.sqlite', readable=10)

    result = run_script('--db', str(db), '--arm', 'own_cpu_some_avg10', '--no-report')
    assert result.returncode == 0, result.stderr

    lines = [line for line in result.stdout.splitlines() if line.startswith('Coverage:')]
    [late] = [line for line in lines if '10/100' in line]
    assert late == (
        'Coverage: readable on 10/100 corpus ticks (10.0%), present on 20/100'
        ' — **BELOW THE FLOOR**: partial_presence, low_readability, see degradations'
    ), late
    [steady] = [line for line in lines if line is not late]
    assert steady == (
        'Coverage: readable on 100/100 corpus ticks (100.0%), present on 100/100'
    ), steady


def test_a_series_with_no_read_ok_rows_is_unknown_even_when_the_clock_ran(
    tmp_path: Path,
):
    """A clock alone is a denominator with no numerator evidence — not 0%.

    Without the guard the fraction is 0/30, and the floor check reports a
    below-floor read rate for a series nothing was ever recorded about.
    """
    db = seed_db(tmp_path / 'db.sqlite', {
        'runqueue_read_ok': [1.0] * 30,
        'own_cpu_some10:orchestrator-reify.service': [30.0] * 30,
    })

    result = run_script('--db', str(db), '--arm', 'own_cpu_some_avg10', '--no-report')
    assert result.returncode == 0, result.stderr

    payload = trailing_json(result.stdout)
    coverage = payload['coverage']['own_cpu_some10:orchestrator-reify.service']
    assert coverage['ticks_in_corpus'] == 30
    assert coverage['ticks_with_a_row'] == 0
    assert coverage['readable_fraction'] is None, coverage
    assert 'unknown_readability' in payload['degradations'], payload['degradations']
    assert 'low_readability' not in payload['degradations'], payload['degradations']


def test_an_arm_whose_read_ok_rows_are_absent_reports_unknown_not_zero(
    tmp_path: Path,
):
    """Zero ticks is an UNKNOWN coverage, not a 0% one.

    ``readable/ticks if ticks else 0.0`` fabricated a 0.0 for a series with no
    ``*_read_ok`` rows at all, and the floor check then emitted
    "readable on 0/0 ticks (0.0%), below the 90% floor" — a verdict about the
    corpus derived from the absence of evidence about it. That is the same
    class of defect this file refuses everywhere else (a fabricated 1.0 for the
    PSI arms; α's fail-open 0.0 persisted as a ratio).

    It is REACHABLE, not hypothetical: the readability metric simply not being
    in the corpus is what a sampler that never ran the load group produces, and
    an operator reading "below the floor" would go looking for a flaky read
    that never happened instead of for a collector that never ran.
    """
    db = seed_db(tmp_path / 'db.sqlite', {'runqueue_ratio': [5.0] * 30})

    result = run_script('--db', str(db), '--arm', 'runqueue_ratio', '--no-report')
    assert result.returncode == 0, result.stderr

    payload = trailing_json(result.stdout)
    coverage = payload['coverage']['runqueue_ratio']
    assert coverage['ticks_in_corpus'] == 0
    assert coverage['ticks_with_a_row'] == 0
    assert coverage['readable_fraction'] is None, (
        'a coverage computed over zero ticks reported a number; '
        f'got {coverage!r}'
    )
    assert 'unknown_readability' in payload['degradations'], payload['degradations']
    assert 'low_readability' not in payload['degradations'], (
        'absence of evidence was reported as evidence of a below-floor read '
        f'rate: {payload["degradation_details"]}'
    )
    detail = next(d for d in payload['degradation_details']
                  if d.startswith('unknown_readability'))
    assert 'runqueue_read_ok' in detail, detail


def test_the_unknown_coverage_line_does_not_claim_a_percentage(tmp_path: Path):
    """The human report must not print 0.0% for a coverage it does not know."""
    db = seed_db(tmp_path / 'db.sqlite', {'runqueue_ratio': [5.0] * 30})

    result = run_script('--db', str(db), '--arm', 'runqueue_ratio', '--no-report')
    assert result.returncode == 0, result.stderr

    assert 'BELOW THE FLOOR' not in result.stdout, result.stdout
    coverage_lines = [
        line for line in result.stdout.splitlines() if line.startswith('Coverage:')
    ]
    assert coverage_lines, result.stdout
    # Scoped to the coverage line on purpose: a 0.0% hold FRACTION on a ladder
    # rung is a real measurement and must keep printing.
    for line in coverage_lines:
        assert 'UNKNOWN' in line, line
        assert '%' not in line, (
            'the report printed a coverage percentage it does not know: ' + line
        )


def test_a_psi_arm_says_it_has_no_readability_metric_rather_than_inventing_one(
    tmp_path: Path,
):
    """collect_psi emits no read_ok, so its coverage is unknown, not 1.0.

    Reporting a fabricated 100% for the four host-PSI arms would be the same
    class of defect as persisting α's fail-open 0.0 as a ratio.
    """
    db = seed_db(tmp_path / 'db.sqlite', {'psi_mem_full_avg10': [5.0] * 30})

    result = run_script('--db', str(db), '--arm', 'mem_full_avg10', '--no-report')
    assert result.returncode == 0, result.stderr

    payload = trailing_json(result.stdout)
    assert payload['coverage']['psi_mem_full_avg10'] is None
    assert 'low_readability' not in payload['degradations']


def seed_a_leaf_that_never_reads(path: Path) -> Path:
    """A 100-tick corpus with one healthy leaf and one discovered-but-dark one.

    Exactly the shape ``collect_load_metrics`` writes for a cgroup leaf found
    on every tick whose ``cpu.pressure`` never reads: ``own_read_ok:<leaf>`` =
    0.0 on each of those ticks, and NO ``own_cpu_some10:<leaf>`` row at all,
    because a failed read persists no value — persisting α's fail-open 0.0
    would fabricate an idle cgroup.
    """
    return seed_db(path, {
        'runqueue_read_ok': [1.0] * 100,
        'own_read_ok:good.service': [1.0] * 100,
        'own_cpu_some10:good.service': [30.0] * 100,
        'own_read_ok:dark.service': [0.0] * 100,
    })


def test_a_leaf_discovered_every_tick_and_readable_on_none_is_still_reported(
    tmp_path: Path,
):
    """The one case ``coverage_table``'s own rationale could not see.

    It keys per leaf because "one cgroup can be unreadable while its siblings
    are fine, which is exactly the case worth seeing" — and the FULLY
    unreadable leaf was invisible, because the block iterated the VALUE series
    and a leaf readable on no tick has no value rows to iterate. Measured
    before this change: the coverage keys were the healthy sibling alone, the
    degradations named nothing, and the string 'dark.service' appeared NOWHERE
    in stdout. A leaf the sampler discovered 100 times and never once read was
    reported identically to one that never existed, and those call for
    opposite next actions.
    """
    db = seed_a_leaf_that_never_reads(tmp_path / 'db.sqlite')

    result = run_script('--db', str(db), '--arm', 'own_cpu_some_avg10', '--no-report')
    assert result.returncode == 0, result.stderr

    payload = trailing_json(result.stdout)
    assert payload['coverage']['own_cpu_some10:dark.service'] == {
        'ticks_in_corpus': 100, 'ticks_with_a_row': 100,
        'readable': 0, 'readable_fraction': 0.0,
        'readability_metric': 'own_read_ok:dark.service',
    }
    # The sibling is untouched: this is per leaf, not a stem-wide verdict.
    assert payload['coverage']['own_cpu_some10:good.service'][
        'readable_fraction'] == pytest.approx(1.0)
    [dark] = [d for d in payload['degradation_details'] if 'dark.service' in d]
    assert '100' in dark, dark


def test_an_arm_readable_on_no_tick_is_reported_in_a_full_run(tmp_path: Path):
    """A non-stem arm goes dark the same way, and hid in a FULL run too.

    ``runqueue_read_ok`` = 0.0 on every tick is a sampler that ran 100 times
    and got no /proc/stat reading, so it wrote no ``runqueue_ratio`` value row
    at all. Measured before this change, with a healthy PSI arm in the same
    corpus so the run was not empty: the only coverage key was that PSI arm's
    ``None``, and the runqueue arm had no row and no degradation — "this arm
    never read" and "this corpus has no such arm" printed identically.
    """
    db = seed_db(tmp_path / 'db.sqlite', {
        'runqueue_read_ok': [0.0] * 100,
        'psi_mem_full_avg10': [5.0] * 100,
    })

    result = run_script('--db', str(db), '--no-report')
    assert result.returncode == 0, result.stderr

    payload = trailing_json(result.stdout)
    assert payload['coverage']['runqueue_ratio'] == {
        'ticks_in_corpus': 100, 'ticks_with_a_row': 100,
        'readable': 0, 'readable_fraction': 0.0,
        'readability_metric': 'runqueue_read_ok',
    }


def test_a_never_readable_series_is_named_apart_from_a_flaky_one(tmp_path: Path):
    """Zero reads is its own finding, not the extreme of a low read rate.

    The operator reading differs, which is the whole reason this vocabulary is
    enumerated. ``low_readability`` ends "read its hold fractions against that
    coverage" — but a series readable on NO tick HAS no hold fractions and no
    candidate-threshold section at all, because a tick with no readable value
    writes no value row. Naming it apart is what tells the reader that the
    series' absence from the ladder sections is failed reads rather than a
    leaf that was never discovered — the confusion the whole fix is about.
    """
    db = seed_a_leaf_that_never_reads(tmp_path / 'db.sqlite')

    result = run_script('--db', str(db), '--arm', 'own_cpu_some_avg10', '--no-report')
    assert result.returncode == 0, result.stderr

    payload = trailing_json(result.stdout)
    assert 'never_readable' in payload['degradations'], payload['degradations']
    assert 'low_readability' not in payload['degradations'], (
        f'zero reads was folded into the flaky-read cause: '
        f'{payload["degradation_details"]}'
    )
    # It was discovered on every tick, so its presence is not a finding.
    assert 'partial_presence' not in payload['degradations'], payload['degradations']
    [dark] = details(payload, 'never_readable')
    assert 'own_cpu_some10:dark.service' in dark and '100/100' in dark, dark
    assert not [d for d in payload['degradation_details'] if 'good.service' in d], (
        'the healthy sibling raised a degradation of its own'
    )


def test_absence_and_never_reading_are_two_independent_causes(tmp_path: Path):
    """Ordered exclusivity would hide half of a leaf that arrived late AND read never.

    ``partial_presence`` is rows over corpus ticks and ``never_readable`` is a
    zero numerator over rows; they are judged on different bases, so a leaf
    discovered for the last fifth of the corpus and readable on none of those
    20 ticks is BOTH. Only ``low_readability`` is displaced — the two are the
    same ratio, reported at a different name.
    """
    db = seed_leaf_present_for_the_last_fifth(tmp_path / 'db.sqlite', readable=0)

    result = run_script('--db', str(db), '--arm', 'own_cpu_some_avg10', '--no-report')
    assert result.returncode == 0, result.stderr

    payload = trailing_json(result.stdout)
    fired = [cause for cause in
             ('partial_presence', 'low_readability', 'never_readable')
             if cause in payload['degradations']]
    assert fired == ['partial_presence', 'never_readable'], (
        payload['degradation_details'])
    [absent] = details(payload, 'partial_presence')
    [dark] = details(payload, 'never_readable')
    assert '20/100' in absent, absent
    assert '20/100' in dark, dark


def test_a_sampler_that_never_read_is_distinguishable_from_one_that_never_ran(
    tmp_path: Path,
):
    """The two readings of an empty arm call for opposite next actions.

    Measured before this change, a corpus holding 100 ``runqueue_read_ok`` =
    0.0 rows and nothing else reported ``no_samples_in_window: no rows for
    ['runqueue_ratio']`` with ``coverage == {}`` — which an operator reads as a
    sampler that never ran, and goes to check the timer unit. It ran 100 times
    and never got a /proc/stat reading, which is a host or collector fault.
    The evidence that separates them was already fetched and then discarded by
    ``read_series``'s no-value-rows early return.

    ``no_samples_in_window`` keeps its name and text here: it is literally
    true, there are no value rows in the window. What was missing is not a
    different name but the readability evidence beside it.
    """
    db = seed_db(tmp_path / 'db.sqlite', {'runqueue_read_ok': [0.0] * 100})

    result = run_script('--db', str(db), '--arm', 'runqueue_ratio', '--no-report')
    assert result.returncode == 0, result.stderr

    payload = trailing_json(result.stdout)
    assert 'no_samples_in_window' in payload['degradations'], payload['degradations']
    assert payload['coverage']['runqueue_ratio'] == {
        'ticks_in_corpus': 100, 'ticks_with_a_row': 100,
        'readable': 0, 'readable_fraction': 0.0,
        'readability_metric': 'runqueue_read_ok',
    }
    assert 'never_readable' in payload['degradations'], payload['degradations']


def test_a_coverage_that_cannot_be_true_is_refused_rather_than_printed(
    tmp_path: Path,
):
    """More readability rows than corpus ticks is a broken invariant, not a ratio.

    Measured before this change, this corpus printed ``Coverage: readable on
    100/50 corpus ticks (200.0%), present on 100/50`` with no degradation at
    all. ``sampler/src/sampler/store.py::write_tick`` writes one whole tick in
    ONE transaction, so no corpus the sampler wrote can produce it — it is the
    hand-seeded or partly-restored one — which is exactly why it belongs in the
    report as an invariant violation rather than folded into a shortfall cause.

    The 200% is only the loudest symptom, and not the thing being guarded: the
    SAME broken clock with half the reads failing yields a perfectly plausible
    0.2 and a ``low_readability`` verdict — a wrong answer delivered quietly,
    which is worse. So the check is on the row counts, not on the fraction.
    """
    db = seed_db(tmp_path / 'db.sqlite', {
        'runqueue_read_ok': [1.0] * 50,
        'own_read_ok:leaf.service': [1.0] * 100,
        'own_cpu_some10:leaf.service': [30.0] * 100,
    })

    result = run_script('--db', str(db), '--arm', 'own_cpu_some_avg10', '--no-report')
    assert result.returncode == 0, result.stderr

    payload = trailing_json(result.stdout)
    coverage = payload['coverage']['own_cpu_some10:leaf.service']
    assert coverage['readable_fraction'] is None, coverage
    # The three counts stay reported verbatim: they are the evidence for the
    # verdict, and without them a reader cannot see WHICH numbers are impossible.
    assert (coverage['ticks_with_a_row'], coverage['readable'],
            coverage['ticks_in_corpus']) == (100, 100, 50), coverage
    assert 'impossible_coverage' in payload['degradations'], payload['degradations']
    for quiet in ('low_readability', 'partial_presence', 'unknown_readability'):
        assert quiet not in payload['degradations'], payload['degradation_details']

    [line] = [ln for ln in result.stdout.splitlines() if ln.startswith('Coverage:')]
    assert 'UNKNOWN' in line, line
    assert '%' not in line, (
        'the report printed a coverage percentage it does not believe: ' + line)
    # Both counts and both metric names, so the reader can see which two
    # numbers cannot both be true and which series they were counted from.
    assert '100' in line and '50' in line, line
    assert 'own_read_ok:leaf.service' in line and 'runqueue_read_ok' in line, line


def test_a_selectors_underscores_are_not_sql_wildcards(tmp_path: Path):
    """`_` is a single-character LIKE wildcard, and every selector contains one.

    A stem selector is matched against the corpus as a ':'-prefixed pattern.
    Spelled with LIKE, `own_cpu_some10:%` also matches `own-cpu-some10:leaf` —
    the underscores match the hyphens — and this repo spells unit and slice
    names with hyphens everywhere (`orchestrator-dark-factory.service`,
    `df-<project>.slice`), so that is a realistic next metric name and not a
    contrived one. Pooling it into this arm's series would shift the arm's
    percentiles and hold fractions with no degradation reported.
    """
    db = seed_db(tmp_path / 'db.sqlite', {
        'own_cpu_some10:orchestrator-dark-factory.service': [1.0] * 50,
        'own-cpu-some10:a-hyphen-spelled-future-metric': [99.0] * 50,
    })

    result = run_script('--db', str(db), '--arm', 'own_cpu_some_avg10', '--no-report')
    assert result.returncode == 0, result.stderr

    payload = trailing_json(result.stdout)
    assert set(payload['holds']) == {
        'own_cpu_some10:orchestrator-dark-factory.service'
    }, (
        'the hyphen-spelled metric was pooled into this arm: a selector\'s '
        f"underscores are being read as wildcards; got {sorted(payload['holds'])}"
    )
    assert set(payload['percentiles']) == {
        'own_cpu_some10:orchestrator-dark-factory.service'
    }


# ── the two-yaml drift check (detail C) ─────────────────────────────────────

_LOCAL_YAML = """
psi_admission:
  enabled: true
  # a comment that must not affect the comparison
  cpu_some_avg10: 70.0
  runqueue_ratio: 4.0
  min_inflight_floor: 3
"""

# Same leaves, same values — different key ORDER, indentation, comments and
# float spellings. Must compare as NO drift.
_PEER_YAML_EQUIVALENT = """
psi_admission:
    min_inflight_floor: 3        # reordered
    runqueue_ratio: 4.00         # 4.00, not 4.0
    cpu_some_avg10: 7.0e+1       # scientific spelling of 70.0 (see below re: the +)
    enabled: yes                 # yaml's other spelling of true
"""


def write_yaml(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def _write_bytes(path: Path, raw: bytes) -> Path:
    """A config that is not decodable text, which write_yaml cannot express."""
    path.write_bytes(raw)
    return path


def test_formatting_differences_are_not_drift(tmp_path: Path):
    """The comparison is over PARSED MAPPINGS, never text (INV-10)."""
    module = load_script()
    local, _ = module.load_psi_admission_block(
        write_yaml(tmp_path / 'local.yaml', _LOCAL_YAML), 'local')
    peer, _ = module.load_psi_admission_block(
        write_yaml(tmp_path / 'peer.yaml', _PEER_YAML_EQUIVALENT), 'peer')

    result = module.compare_blocks(local, peer)

    assert result['drift'] == [], result
    assert result['local_only'] == []
    assert result['peer_only'] == []


def test_a_differing_value_is_reported_naming_both_sides():
    module = load_script()

    result = module.compare_blocks(
        {'cpu_some_avg10': 70.0}, {'cpu_some_avg10': 85.0})

    assert result['drift'] == [
        {'leaf': 'cpu_some_avg10', 'local': 70.0, 'peer': 85.0}
    ], result


def test_a_one_sided_leaf_is_distinct_from_a_value_mismatch():
    module = load_script()

    result = module.compare_blocks(
        {'cpu_some_avg10': 70.0, 'min_inflight_floor': 3},
        {'cpu_some_avg10': 70.0, 'runqueue_ratio': 4.0},
    )

    assert result['drift'] == []
    assert result['local_only'] == ['min_inflight_floor']
    assert result['peer_only'] == ['runqueue_ratio']


def test_int_and_float_spellings_of_the_same_number_are_not_drift():
    module = load_script()

    assert module.compare_blocks({'x': 15}, {'x': 15.0})['drift'] == []


def test_a_numeric_looking_string_is_reported_as_drift_not_silently_coerced():
    """Deliberate, and it rests on a MEASURED PyYAML quirk.

    PyYAML implements YAML 1.1, whose float regex requires a SIGN in the
    exponent: `7.0e+1` parses as 70.0 but `7.0e1` parses as the STRING
    '7.0e1'. So a config author can write what looks like a number and get a
    string. The drift check reports that as drift rather than coercing it,
    because the two files genuinely do differ and the operator should see it —
    coercing would mean interpreting a meaningful string, and would hide a
    real config-authoring mistake behind a clean report.
    """
    module = load_script()

    result = module.compare_blocks({'x': 70.0}, {'x': '7.0e1'})

    assert result['drift'] == [{'leaf': 'x', 'local': 70.0, 'peer': '7.0e1'}]


def test_an_absent_block_is_never_treated_as_an_empty_match():
    """None means "not configured"; {} means "configured empty"."""
    module = load_script()

    assert module.compare_blocks(None, {'cpu_some_avg10': 70.0}) is None
    assert module.compare_blocks({'cpu_some_avg10': 70.0}, None) is None
    assert module.compare_blocks(None, None) is None
    # An actually-empty block on both sides IS a comparison, and finds nothing.
    assert module.compare_blocks({}, {})['drift'] == []


@pytest.mark.parametrize(
    ('setup', 'expected'),
    [
        pytest.param(lambda p: p / 'absent.yaml', 'peer_config_missing', id='missing'),
        pytest.param(
            lambda p: write_yaml(p / 'bad.yaml', 'psi_admission: [not, a, mapping\n'),
            'peer_config_unparseable', id='unparseable'),
        pytest.param(
            lambda p: write_yaml(p / 'noblock.yaml', 'other_key: 1\n'),
            'peer_psi_admission_absent', id='no-block'),
        pytest.param(
            lambda p: write_yaml(p / 'scalar.yaml', 'just-a-string\n'),
            'peer_config_unparseable', id='not-a-mapping'),
        # A UnicodeDecodeError is a ValueError, not an OSError, so before it was
        # named here it left main() by a path with no top-level guard -- turning
        # an odd byte in someone's yaml into an INFRA FAULT page. The read is
        # pinned to utf-8 for the same reason, so this degrades identically
        # under LANG=C and LANG=*.UTF-8 rather than only under one of them.
        pytest.param(
            lambda p: _write_bytes(p / 'latin1.yaml',
                                   'psi_admission:\n  note: caf\xe9\n'.encode('latin-1')),
            'peer_config_unreadable', id='not-utf8'),
    ],
)
def test_every_peer_failure_mode_is_its_own_named_degradation(
    tmp_path: Path, setup, expected
):
    module = load_script()

    block, degradations = module.load_psi_admission_block(setup(tmp_path), 'peer')

    assert block is None
    assert [d.split(':', 1)[0] for d in degradations] == [expected], degradations


def test_the_local_side_gets_the_same_treatment(tmp_path: Path):
    """Measured true on main TODAY: dark-factory-orchestrator.yaml has no
    psi_admission block (γ unlanded) while reify's already carries one, so the
    first real run IS one-sided and must say so rather than report a match."""
    module = load_script()

    block, degradations = module.load_psi_admission_block(
        write_yaml(tmp_path / 'local.yaml', 'other_key: 1\n'), 'local')

    assert block is None
    assert [d.split(':', 1)[0] for d in degradations] == ['local_psi_admission_absent']


def test_a_yaml_date_scalar_does_not_cost_the_rc_after_the_analysis_ran(
    tmp_path: Path,
):
    """compare_blocks copies RAW parsed yaml values through, and yaml has types.

    An unquoted `2026-09-17` in a psi_admission leaf resolves to datetime.date,
    which lands in `drift` untouched and reaches the trailing json.dumps. Without
    `default=`, that raises TypeError AFTER the whole analysis has run and
    printed the human report -- the single most expensive moment to lose the rc,
    because eps1/eps2 read a non-zero rc as an INFRA FAULT with no gate and page
    a human about a broken script that had in fact just delivered its answer.

    Asserted end to end through the real subprocess rather than on json.dumps
    directly: the property is about the SCRIPT's exit code, and the type only
    reaches the serialiser by travelling the whole yaml -> compare_blocks ->
    report path.
    """
    db = seed_db(tmp_path / 'load.db', {'runqueue_ratio': [1.0] * 10})
    local = write_yaml(tmp_path / 'local.yaml',
                       'psi_admission:\n  review_after: 2026-09-17\n')
    peer = write_yaml(tmp_path / 'peer.yaml',
                      'psi_admission:\n  review_after: 2026-10-01\n')

    result = run_script('--db', str(db), '--no-report',
                        '--config', str(local), '--peer-config', str(peer))

    assert result.returncode == 0, result.stderr[-2000:]
    payload = trailing_json(result.stdout)
    drift, = payload['drift']['drift']
    assert drift['leaf'] == 'review_after'
    assert (drift['local'], drift['peer']) == ('2026-09-17', '2026-10-01'), (
        'the date reached the JSON as something other than its str() form'
    )


def test_missing_pyyaml_is_a_named_degradation_not_an_import_crash(
    tmp_path: Path, monkeypatch
):
    module = load_script()
    monkeypatch.setitem(sys.modules, 'yaml', None)

    block, degradations = module.load_psi_admission_block(
        write_yaml(tmp_path / 'local.yaml', _LOCAL_YAML), 'local')

    assert block is None
    assert [d.split(':', 1)[0] for d in degradations] == ['pyyaml_absent'], degradations


def test_a_one_sided_run_reports_the_degradation_and_still_exits_zero(tmp_path: Path):
    db = seed_db(tmp_path / 'db.sqlite', {'runqueue_ratio': [1.0, 2.0]})
    local = write_yaml(tmp_path / 'local.yaml', _LOCAL_YAML)

    result = run_script(
        '--db', str(db), '--config', str(local),
        '--peer-config', str(tmp_path / 'absent.yaml'), '--no-report')

    assert result.returncode == 0, result.stderr
    payload = trailing_json(result.stdout)
    assert 'peer_config_missing' in payload['degradations'], payload
    assert payload['drift'] is None, 'a one-sided comparison must not report a match'


def test_a_two_sided_run_reports_structured_drift(tmp_path: Path):
    db = seed_db(tmp_path / 'db.sqlite', {'runqueue_ratio': [1.0, 2.0]})
    local = write_yaml(tmp_path / 'local.yaml', _LOCAL_YAML)
    peer = write_yaml(tmp_path / 'peer.yaml', """
psi_admission:
  enabled: true
  cpu_some_avg10: 85.0
  min_inflight_floor: 3
""")

    result = run_script(
        '--db', str(db), '--config', str(local),
        '--peer-config', str(peer), '--no-report')

    assert result.returncode == 0, result.stderr
    payload = trailing_json(result.stdout)
    assert payload['drift']['drift'] == [
        {'leaf': 'cpu_some_avg10', 'local': 70.0, 'peer': 85.0}
    ], payload['drift']
    assert payload['drift']['local_only'] == ['runqueue_ratio']


def test_smoke_against_the_two_real_committed_configs(tmp_path: Path):
    """The real committed yaml is found, PARSES, and the analysis runs on it.

    ONE of the two tests here that deliberately reads outside this worktree,
    and it names both configs explicitly rather than inheriting a default —
    everything else runs against injected paths (see ``load_script``). If the
    peer checkout is absent the peer half degrades by name and the local
    assertions below, which are what this test is for, still hold.

    Neither project's current values are pinned — those are operator decisions
    that change, and a test that froze them would go red on an ordinary tuning
    commit. Nor is the report's PROSE pinned: a heading substring goes red on a
    reword and stays green if the section is emptied, so it witnesses nothing.
    What is stable is that the file this script is aimed at by default exists,
    is valid yaml, reaches the yaml parser at all, and yields a populated
    payload. `isinstance(payload['degradations'], list)`, which is what this
    asserted before, is true of any list-valued output and so witnessed none of
    that.
    """
    db = seed_db(tmp_path / 'db.sqlite', {'runqueue_ratio': [1.0, 2.0]})

    result = run_script(
        '--db', str(db),
        '--config', str(REPO_ROOT / 'dark-factory-orchestrator.yaml'),
        '--peer-config', str(REAL_PEER_CONFIG),
        '--no-report')

    assert result.returncode == 0, result.stderr
    payload = trailing_json(result.stdout)
    for cannot_read in ('local_config_missing', 'local_config_unreadable',
                        'local_config_unparseable'):
        assert cannot_read not in payload['degradations'], payload['degradation_details']
    assert payload['percentiles']['runqueue_ratio']['n'] == 2
    assert payload['holds']['runqueue_ratio'], payload
    assert 'pyyaml_absent' not in payload['degradations'], payload['degradation_details']


# ── detail (D): leaves that merely restate the shipped code default ─────────

# INJECTED, never fetched. The point of this whole check is that PRD §6.2
# writes three values that are already the code defaults, giving one fact
# three homes with no reconciler (INV-9). Hard-coding them into the script
# would make it the FOURTH home and defeat the check it exists to be.
_INJECTED_DEFAULTS = {
    'mem_some_avg10': 15.0,
    'mem_full_avg10': 3.0,
    'io_some_avg10': 40.0,
    'cpu_some_avg10': 85.0,
}


def test_leaves_restating_the_code_default_are_flagged():
    module = load_script()
    block = {
        'mem_some_avg10': 15.0,
        'mem_full_avg10': 3.0,
        'io_some_avg10': 40.0,
        'runqueue_ratio': 4.0,
    }

    result = module.compare_to_code_defaults(block, _INJECTED_DEFAULTS)

    assert sorted(result['restates_default']) == [
        'io_some_avg10', 'mem_full_avg10', 'mem_some_avg10']
    assert 'runqueue_ratio' not in result['restates_default']


def test_a_leaf_that_differs_from_the_default_is_not_flagged():
    module = load_script()

    result = module.compare_to_code_defaults(
        {'cpu_some_avg10': 70.0}, _INJECTED_DEFAULTS)

    assert result['restates_default'] == []
    assert result['unknown_to_schema'] == []


def test_int_and_float_spellings_do_not_hide_a_restatement():
    module = load_script()

    result = module.compare_to_code_defaults({'mem_some_avg10': 15}, _INJECTED_DEFAULTS)

    assert result['restates_default'] == ['mem_some_avg10']


def test_a_leaf_absent_from_the_defaults_is_reported_unknown_to_the_schema():
    """Measured: PsiAdmissionConfig has no runqueue_ratio field today (β
    unlanded). Reported, not flagged and not silently dropped — "this arm is
    not in the model yet" is a fact the operator needs."""
    module = load_script()

    result = module.compare_to_code_defaults(
        {'runqueue_ratio': 4.0, 'own_cpu_some_avg10': 50.0}, _INJECTED_DEFAULTS)

    assert sorted(result['unknown_to_schema']) == [
        'own_cpu_some_avg10', 'runqueue_ratio']
    assert result['restates_default'] == []


def test_an_absent_defaults_mapping_yields_no_verdict():
    """When the shell could not obtain the defaults, nothing may be claimed."""
    module = load_script()

    assert module.compare_to_code_defaults({'mem_some_avg10': 15.0}, None) is None
    assert module.compare_to_code_defaults(None, _INJECTED_DEFAULTS) is None


def test_the_flags_are_produced_for_both_the_local_and_peer_blocks(
    tmp_path: Path, monkeypatch, capsys
):
    """Both verdicts must REACH the report, each on its own side's line.

    This asserted on compare_to_code_defaults called directly, having called
    main() and discarded everything it produced — so deleting the main() call
    left it green and nothing checked that either flag was ever rendered. The
    defaults are still injected rather than fetched (the whole point of the
    check is that this script owns no copy of them), but now through the same
    seam main() uses.
    """
    module = load_script()
    local = write_yaml(tmp_path / 'local.yaml', """
psi_admission:
  mem_some_avg10: 15.0
""")
    peer = write_yaml(tmp_path / 'peer.yaml', """
psi_admission:
  io_some_avg10: 40.0
""")
    db = seed_db(tmp_path / 'db.sqlite', {'runqueue_ratio': [1.0, 2.0]})
    monkeypatch.setattr(
        module, 'fetch_code_defaults',
        lambda **_kwargs: (_INJECTED_DEFAULTS, []),
    )

    module.main([
        '--db', str(db), '--config', str(local), '--peer-config', str(peer),
        '--no-report',
    ])

    stdout = capsys.readouterr().out
    payload = trailing_json(stdout)
    assert payload['restates_code_default']['local'][
        'restates_default'] == ['mem_some_avg10']
    assert payload['restates_code_default']['peer'][
        'restates_default'] == ['io_some_avg10']
    # Each side's verdict on its OWN line: the report is what an operator
    # reads, and a local flag rendered under "peer" would be worse than none.
    local_line, = [ln for ln in stdout.splitlines() if ln.startswith('- **local**')]
    peer_line, = [ln for ln in stdout.splitlines() if ln.startswith('- **peer**')]
    assert 'mem_some_avg10' in local_line and 'io_some_avg10' not in local_line
    assert 'io_some_avg10' in peer_line and 'mem_some_avg10' not in peer_line


def test_compare_to_code_defaults_has_no_built_in_defaults_mapping():
    """There is nothing to fall back on, so a failed fetch cannot silently
    become a stale comparison against the script's own copy of the numbers.

    Asserted by CALLING it one argument short rather than by reading the
    signature: introspection would freeze the parameter's NAME, failing for a
    rename that breaks nothing, while proving nothing about a call.
    """
    module = load_script()

    with pytest.raises(TypeError):
        module.compare_to_code_defaults({'mem_some_avg10': 3.0})


def test_the_defaults_shell_degrades_named_when_the_subprocess_fails(tmp_path: Path):
    module = load_script()

    defaults, degradations = module.fetch_code_defaults(
        command=['/bin/false'], cwd=tmp_path)

    assert defaults is None
    assert [d.split(':', 1)[0] for d in degradations] == ['code_defaults_unavailable']


def test_the_defaults_shell_degrades_named_on_unparseable_output(tmp_path: Path):
    module = load_script()

    defaults, degradations = module.fetch_code_defaults(
        command=['/bin/echo', 'not json'], cwd=tmp_path)

    assert defaults is None
    assert [d.split(':', 1)[0] for d in degradations] == ['code_defaults_unavailable']


def test_the_defaults_shell_degrades_named_on_a_missing_binary(tmp_path: Path):
    module = load_script()

    defaults, degradations = module.fetch_code_defaults(
        command=[str(tmp_path / 'no-such-uv')], cwd=tmp_path)

    assert defaults is None
    assert [d.split(':', 1)[0] for d in degradations] == ['code_defaults_unavailable']


@pytest.mark.skipif(
    not Path('/home/leo/.local/bin/uv').exists(),
    reason='the shipped uv is absent on this host',
)
def test_the_real_defaults_shell_answers_on_this_host():
    """The one test that deliberately runs the real `uv run --project orchestrator`.

    Every branch of the defaults shell is covered above with an injected
    command, and `default_defaults_command` pins the argv — but nothing
    checked that the argv, run for real, still answers. Detail (D)'s whole
    premise is that the script owns NO copy of the defaults, so if the model
    moved or the flags stopped working the check would silently degrade to
    `code_defaults_unavailable` on the gate's 14-day clock with nobody the
    wiser.

    This used to be covered by accident, by ~25 tests that merely omitted
    `--uv-bin`; they are hermetic now (see ``load_script``), so the coverage is
    deliberate here instead — one spawn rather than twenty-five, and named for
    what it checks. It reads the MAIN checkout, which is the point.
    """
    module = load_script(real_host_defaults=True)

    defaults, degradations = module.fetch_code_defaults(
        command=module.default_defaults_command(module.DEFAULT_UV_BIN),
        cwd=module.default_project_root(),
    )

    assert degradations == [], degradations
    assert defaults is not None
    # The three leaves PRD §6.2 writes that are already the shipped default —
    # the comparison detail (D) exists to make.
    assert {'mem_some_avg10', 'mem_full_avg10', 'io_some_avg10'} <= set(defaults)


def test_the_whole_report_is_still_produced_when_defaults_are_unavailable(tmp_path: Path):
    """Exit 0 with the degradation named, and every other section intact."""
    db = seed_db(tmp_path / 'db.sqlite', {'runqueue_ratio': [1.0, 2.0, 3.0]})

    result = run_script(
        '--db', str(db), '--no-report', '--uv-bin', str(tmp_path / 'no-such-uv'))

    assert result.returncode == 0, result.stderr
    payload = trailing_json(result.stdout)
    assert 'code_defaults_unavailable' in payload['degradations'], payload
    assert payload['percentiles'], 'the rest of the report must survive'


def test_the_defaults_command_uses_no_sync(tmp_path: Path):
    """--no-sync is load-bearing: a plain `uv run --project shared` was
    measured REMOVING orchestrator from the shared root venv, so a syncing
    invocation here could break live processes."""
    module = load_script()

    command = module.default_defaults_command(Path('/home/leo/.local/bin/uv'))

    assert '--no-sync' in command, command
    assert '--frozen' in command, command
    assert '--project' in command and 'orchestrator' in command, command


def test_the_defaults_subprocess_runs_in_the_project_root_not_beside_the_config(
    tmp_path: Path, monkeypatch
):
    """Two unrelated dimensions had been fused into one flag (heuristic 3).

    The subprocess that asks the LIVE MODEL for its shipped defaults ran with
    ``cwd=args.config.parent``, so `--config /home/leo/src/reify/...` quietly
    compared that config against REIFY's orchestrator code and labelled the
    answer "the shipped code default". Both projects are dark-factory-derived,
    so it succeeds and silently reports the wrong project's numbers — the worst
    shape of failure for a file whose whole job is naming drift.

    Where the config lives and which checkout owns the code are independent
    facts, and each now has its own input.
    """
    module = load_script()
    db = seed_db(tmp_path / 'db.sqlite', {'runqueue_ratio': [1.0, 2.0]})
    elsewhere = tmp_path / 'some' / 'other' / 'checkout'
    elsewhere.mkdir(parents=True)
    (elsewhere / 'cfg.yaml').write_text('psi_admission:\n  mem_some_avg10: 15.0\n')
    project_root = tmp_path / 'the-real-root'
    project_root.mkdir()

    seen: dict = {}

    def capture(*, command, cwd):
        seen['command'], seen['cwd'] = command, cwd
        return None, ['code_defaults_unavailable: stubbed']

    monkeypatch.setattr(module, 'fetch_code_defaults', capture)
    module.main([
        '--db', str(db), '--no-report',
        '--config', str(elsewhere / 'cfg.yaml'),
        '--project-root', str(project_root),
    ])

    assert Path(seen['cwd']) == project_root, (
        f'the defaults subprocess ran in {seen["cwd"]}, which is derived from '
        '--config rather than from the project root'
    )


def test_the_project_root_defaults_through_the_same_env_seam_as_the_corpus(
    monkeypatch,
):
    """One seam for "which checkout is this", already used by default_db."""
    module = load_script()
    monkeypatch.setenv('DARK_FACTORY_ROOT', '/somewhere/else')

    assert module.parse_args([]).project_root == Path('/somewhere/else')
    assert module.default_db() == Path('/somewhere/else/data/load-samples.db')


def test_every_this_checkout_default_follows_the_seam_together(monkeypatch):
    """A path that ignores the seam sends one run across TWO checkouts.

    --db and --project-root honoured $DARK_FACTORY_ROOT while --config and
    --report-dir were hardcoded, so `DARK_FACTORY_ROOT=/other` read the other
    checkout's corpus and code defaults, compared them against the ORIGINAL
    checkout's yaml, and with --commit filed the report into the original repo.
    Nothing in the report said so. Asserted as a SET so a fifth such default
    cannot be added while quietly skipping the seam.

    --peer-config is excluded on purpose and pinned by its own test above: it
    names the OTHER project, which is the entire point of the comparison —
    which is also why this is one of the two tests that ask the loader for the
    SHIPPED defaults rather than the hermetic ones.
    """
    module = load_script(real_host_defaults=True)
    monkeypatch.setenv('DARK_FACTORY_ROOT', '/somewhere/else')

    args = module.parse_args([])

    assert {args.db, args.project_root, args.config, args.report_dir} == {
        Path('/somewhere/else/data/load-samples.db'),
        Path('/somewhere/else'),
        Path('/somewhere/else/dark-factory-orchestrator.yaml'),
        Path('/somewhere/else/plans'),
    }
    assert args.peer_config == REAL_PEER_CONFIG


# ── the --report-dir / --no-report / --commit trio ──────────────────────────


def make_repo(tmp_path: Path) -> Path:
    """A throwaway git repo. Never the real checkout, and never `git stash`."""
    repo = tmp_path / 'repo'
    (repo / 'plans').mkdir(parents=True)
    for args in (
        ['init', '-q', '-b', 'main'],
        ['config', 'user.email', 'test@example.invalid'],
        ['config', 'user.name', 'Test'],
        ['config', 'commit.gpgsign', 'false'],
    ):
        subprocess.run(['git', '-C', str(repo), *args], check=True,
                       capture_output=True)
    (repo / 'seed.txt').write_text('seed\n')
    subprocess.run(['git', '-C', str(repo), 'add', '--', 'seed.txt'],
                   check=True, capture_output=True)
    subprocess.run(['git', '-C', str(repo), 'commit', '-q', '-m', 'seed',
                    '--no-verify'], check=True, capture_output=True)
    return repo


def git_out(repo: Path, *args: str) -> str:
    return subprocess.run(['git', '-C', str(repo), *args], check=True,
                          capture_output=True, text=True).stdout


def test_the_report_is_written_with_a_dated_name_and_announced_on_stderr(
    tmp_path: Path, capsys
):
    """The announcement's CHANNEL is the contract, and it was never checked.

    The module docstring promises stdout's last line is the single-line JSON,
    which is how ε1/ε2 read the run; the report path therefore goes to stderr.
    Moving that print to stdout left every test green, because it is emitted
    BEFORE the JSON and `splitlines()[-1]` still parsed — so the name of this
    test was the only thing asserting the channel.
    """
    module = load_script()
    db = seed_db(tmp_path / 'db.sqlite', {'runqueue_ratio': [1.0, 2.0]})
    report_dir = tmp_path / 'plans'
    report_dir.mkdir()

    rc = module.main(['--db', str(db), '--report-dir', str(report_dir)])

    assert rc == 0
    written = list(report_dir.glob('load-threshold-calibration-*.md'))
    assert len(written) == 1, written
    assert written[0].name.count('-') == 5, written[0].name
    assert written[0].read_text().strip(), 'the report file was created but empty'

    captured = capsys.readouterr()
    assert f'report: {written[0]}' in captured.err, captured.err
    assert str(written[0]) not in captured.out, (
        'the report path reached STDOUT, where the gate reads JSON'
    )
    assert json.loads(captured.out.strip().splitlines()[-1])['db'] == str(db)


def test_the_filename_heading_and_commit_subject_share_one_clock_read(tmp_path: Path,
                                                                     monkeypatch):
    """One `datetime.now(UTC)`, reused — not three reads that can straddle
    midnight and produce a report whose name, heading and commit disagree.

    The clock is REPLACED with one that returns a different day on every read,
    because against the real clock this test passed on every second of the day
    except the midnight boundary it names — which is to say it did not pin the
    property at all. With this clock a second read is visible immediately.
    """
    repo = make_repo(tmp_path)
    db = seed_db(tmp_path / 'db.sqlite', {'runqueue_ratio': [1.0, 2.0]})
    module = load_script()

    instants = iter([
        datetime(2026, 9, 14, 23, 59, 59, tzinfo=UTC),
        datetime(2026, 9, 15, 0, 0, 1, tzinfo=UTC),
        datetime(2026, 9, 16, 0, 0, 1, tzinfo=UTC),
    ])

    class ADifferentDayOnEveryRead:
        @staticmethod
        def now(_tz=None):
            return next(instants)

    monkeypatch.setattr(module, 'datetime', ADifferentDayOnEveryRead)

    module.main(['--db', str(db), '--report-dir', str(repo / 'plans'), '--commit'])

    written, = (repo / 'plans').glob('load-threshold-calibration-*.md')
    assert written.name == 'load-threshold-calibration-2026-09-14.md'
    assert '2026-09-14' in written.read_text().splitlines()[0]
    assert '2026-09-14' in git_out(repo, 'log', '-1', '--pretty=%s')


def test_a_report_dir_at_the_repo_root_still_commits(tmp_path: Path):
    """The repo is discovered, not computed from the report path.

    `repo = path.parent.parent` is right only for the default
    <root>/plans/<file>.md layout. `--report-dir <root>` put the report at
    <root>/<file>.md, whose parent.parent is the directory ABOVE the repo, so
    the commit degraded for no reason the operator could see.
    """
    repo = make_repo(tmp_path)
    db = seed_db(tmp_path / 'db.sqlite', {'runqueue_ratio': [1.0, 2.0]})
    module = load_script()

    module.main(['--db', str(db), '--report-dir', str(repo), '--commit'])

    written, = repo.glob('load-threshold-calibration-*.md')
    committed = git_out(repo, 'show', '--name-only', '--pretty=', 'HEAD').split()
    assert written.name in committed, (
        f'the report at the repo root was not committed; HEAD touched {committed}'
    )


def test_a_report_dir_outside_any_repo_is_still_a_named_degradation(tmp_path: Path):
    """Discovery must not turn "no repo here" into a silent success."""
    outside = tmp_path / 'not-a-repo'
    outside.mkdir()
    db = seed_db(tmp_path / 'db.sqlite', {'runqueue_ratio': [1.0, 2.0]})

    result = run_script('--db', str(db), '--report-dir', str(outside), '--commit')

    assert result.returncode == 0, result.stderr
    payload = trailing_json(result.stdout)
    assert 'report_commit_failed' in payload['degradations'], payload


def test_no_report_prints_everything_but_writes_no_file(tmp_path: Path):
    """print-only, never compute-less."""
    db = seed_db(tmp_path / 'db.sqlite', {'runqueue_ratio': [1.0, 2.0, 3.0]})
    report_dir = tmp_path / 'plans'
    report_dir.mkdir()

    result = run_script('--db', str(db), '--report-dir', str(report_dir),
                        '--no-report')

    assert result.returncode == 0, result.stderr
    assert list(report_dir.iterdir()) == []
    assert trailing_json(result.stdout)['percentiles']
    # "everything": the human report is printed too, not only the JSON line.
    printed = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(printed) > 1, result.stdout


def test_commit_touches_exactly_the_report_and_nothing_else(tmp_path: Path):
    """What `git commit --only` buys, and why a bare `git commit` is wrong.

    An unrelated DIRTY file and an unrelated STAGED file are left in the repo;
    a bare commit would sweep the staged one in. Under the merge worker and
    the startup reconciler, doing that to the real checkout is a live hazard,
    not a stylistic preference.
    """
    repo = make_repo(tmp_path)
    db = seed_db(tmp_path / 'db.sqlite', {'runqueue_ratio': [1.0, 2.0]})
    module = load_script()

    (repo / 'seed.txt').write_text('dirtied\n')
    (repo / 'other.txt').write_text('staged but unrelated\n')
    subprocess.run(['git', '-C', str(repo), 'add', '--', 'other.txt'],
                   check=True, capture_output=True)

    module.main(['--db', str(db), '--report-dir', str(repo / 'plans'), '--commit'])

    touched = git_out(repo, 'show', '--name-only', '--pretty=', 'HEAD').split()
    assert len(touched) == 1, touched
    assert touched[0].startswith('plans/load-threshold-calibration-'), touched
    # The unrelated work is still exactly where it was left.
    assert 'other.txt' in git_out(repo, 'diff', '--cached', '--name-only')
    assert 'seed.txt' in git_out(repo, 'diff', '--name-only')


def test_the_commit_subject_names_the_generating_script(tmp_path: Path):
    repo = make_repo(tmp_path)
    db = seed_db(tmp_path / 'db.sqlite', {'runqueue_ratio': [1.0, 2.0]})
    module = load_script()

    module.main(['--db', str(db), '--report-dir', str(repo / 'plans'), '--commit'])

    subject = git_out(repo, 'log', '-1', '--pretty=%s')
    assert 'scripts/load-threshold-calibration.py' in subject, subject


def test_no_report_with_commit_commits_nothing(tmp_path: Path):
    """--commit nests inside the not-no-report branch, pinned rather than
    left accidental."""
    repo = make_repo(tmp_path)
    db = seed_db(tmp_path / 'db.sqlite', {'runqueue_ratio': [1.0, 2.0]})
    module = load_script()
    before = git_out(repo, 'rev-parse', 'HEAD').strip()

    rc = module.main([
        '--db', str(db), '--report-dir', str(repo / 'plans'),
        '--no-report', '--commit',
    ])

    assert rc == 0
    assert git_out(repo, 'rev-parse', 'HEAD').strip() == before
    assert list((repo / 'plans').iterdir()) == []


def test_a_rejected_commit_leaves_nothing_staged_behind_it(tmp_path: Path):
    """The index must come back, because the repo this runs in is shared.

    `git add` is unavoidable — `git commit --only` rejects an untracked
    pathspec — so a commit that fails AFTER it leaves the report staged in a
    checkout the merge worker and the hooks act on directly, where a
    concurrent bare `git commit` would sweep it into an unrelated commit. A
    rejecting pre-commit hook is the realistic trigger and the one used here;
    `.git/index.lock` held past the grace window is the other.
    """
    module = load_script()
    repo = make_repo(tmp_path)
    hook = repo / '.git' / 'hooks' / 'pre-commit'
    hook.write_text('#!/bin/sh\necho "nope" >&2\nexit 1\n')
    hook.chmod(0o755)
    db = seed_db(tmp_path / 'db.sqlite', {'runqueue_ratio': [1.0, 2.0]})

    rc = module.main([
        '--db', str(db), '--report-dir', str(repo / 'plans'), '--commit',
    ])

    assert rc == 0
    assert git_out(repo, 'diff', '--cached', '--name-only').strip() == '', (
        'the rejected commit left the report staged in a machine-operated '
        'checkout, where a concurrent bare `git commit` would sweep it up'
    )
    assert list((repo / 'plans').glob('load-threshold-calibration-*.md')), (
        'unstaging must not delete the analysis it failed to commit'
    )


def test_a_failing_commit_does_not_take_the_report_down_with_it(tmp_path: Path):
    """The analysis is the deliverable; committing it is a convenience.

    ε1/ε2 classify a non-zero rc as an INFRA FAULT with no gate, so a git
    failure must not turn a delivered calibration into a born-at-L2 page.
    """
    db = seed_db(tmp_path / 'db.sqlite', {'runqueue_ratio': [1.0, 2.0]})
    not_a_repo = tmp_path / 'not-a-repo' / 'plans'
    not_a_repo.mkdir(parents=True)

    result = run_script('--db', str(db), '--report-dir', str(not_a_repo), '--commit')

    assert result.returncode == 0, result.stderr
    assert list(not_a_repo.glob('load-threshold-calibration-*.md')), 'report still written'
    payload = trailing_json(result.stdout)
    assert 'report_commit_failed' in payload['degradations'], payload


# ── arm_thresholds: a botched arm leaf is NAMED, never dropped (esc-3592-8) ──

def test_a_non_numeric_arm_leaf_is_named_rather_than_silently_dropped():
    """The silent-drop this module's own contract forbids.

    The module docstring promises every degradation is a named entry in the
    report AND a key in the trailing JSON. A `runqueue_ratio: "4.0"` used to
    vanish from `configured` with nothing said anywhere, so the report printed
    "no value in force" for a config that visibly sets one.
    """
    module = load_script()

    result = module.arm_thresholds(
        {'enabled': True, 'runqueue_ratio': '4.0', 'cpu_some_avg10': 85.0},
        'local')

    assert result.in_force == {'cpu_some_avg10': 85.0}
    assert result.unusable == {'runqueue_ratio': "'4.0' (str)"}
    assert result.degradations == [
        "local_threshold_not_numeric: runqueue_ratio = '4.0' (str)"
    ]


def test_the_yaml_1_1_exponent_quirk_is_the_realistic_trigger():
    """`7.0e1` is a STRING to PyYAML; `7.0e+1` is 70.0. Measured, not supposed.

    This is why the guard is not hypothetical: an operator writes what reads as
    a number and gets a string. Asserted through yaml itself rather than by
    hand-writing the string, so the test fails if the quirk ever goes away.
    """
    import yaml
    module = load_script()

    block = yaml.safe_load('runqueue_ratio: 7.0e1\ncpu_some_avg10: 7.0e+1\n')
    assert block['runqueue_ratio'] == '7.0e1', 'PyYAML quirk gone; revisit'

    result = module.arm_thresholds(block, 'local')

    assert result.in_force == {'cpu_some_avg10': 70.0}
    assert 'runqueue_ratio' in result.unusable
    assert result.degradations == [
        "local_threshold_not_numeric: runqueue_ratio = '7.0e1' (str)"
    ]


def test_a_bool_on_an_arm_leaf_is_unusable_not_the_threshold_one():
    """bool IS an int in Python, so `runqueue_ratio: true` must not read as 1.0.

    It is also not a silent skip: on an ARM leaf a bool is an authoring
    mistake, so it lands in `unusable` with a degradation.
    """
    module = load_script()

    result = module.arm_thresholds({'runqueue_ratio': True}, 'local')

    assert result.in_force == {}
    assert result.unusable == {'runqueue_ratio': 'True (bool)'}
    assert result.degradations == [
        'local_threshold_not_numeric: runqueue_ratio = True (bool)'
    ]


def test_non_arm_leaves_are_skipped_silently_because_they_are_not_mistakes():
    """`enabled` and `min_inflight_floor` are not thresholds. No degradation."""
    module = load_script()

    result = module.arm_thresholds(
        {'enabled': True, 'min_inflight_floor': 3, 'runqueue_ratio': 4.0},
        'local')

    assert result.in_force == {'runqueue_ratio': 4.0}
    assert result.unusable == {}
    assert result.degradations == []


def test_an_absent_block_yields_no_thresholds_and_no_degradation():
    """`None` in means nothing configured — which is not itself a failure."""
    module = load_script()

    result = module.arm_thresholds(None, 'local')

    assert (result.in_force, result.unusable, result.degradations) == ({}, {}, [])


def test_the_side_prefixes_the_degradation_like_its_sibling_loader():
    """Same `<side>_<name>` vocabulary as load_psi_admission_block."""
    module = load_script()

    result = module.arm_thresholds({'runqueue_ratio': '4.0'}, 'peer')

    assert result.degradations == [
        "peer_threshold_not_numeric: runqueue_ratio = '4.0' (str)"
    ]


def test_an_unconfigured_arm_does_not_misdirect_the_reader_to_degradations():
    """The other half of the same defect.

    The old single label sent the reader to the Degradations section for an
    arm that was simply never configured — where there is nothing to read and
    nothing is wrong. "(see degradations)" is now reserved for the case that
    actually put something there.
    """
    module = load_script()

    assert module._in_force_label('runqueue_ratio', None, {}) == '(not configured)'
    assert module._in_force_label(None, None, {}) == '(not configured)'
    assert module._in_force_label('runqueue_ratio', 4.0, {}) == '4.0'
    assert 'see degradations' in module._in_force_label(
        'runqueue_ratio', None, {'runqueue_ratio': "'4.0' (str)"})


def test_the_botched_leaf_reaches_both_the_report_and_the_json(tmp_path: Path):
    """End to end, which is where the contract is actually owed.

    Reproduces the reviewer's exact config. Before the fix the JSON reported
    `configured: {cpu_some_avg10: 85.0}` with no mention of the runqueue leaf
    in `degradations`, and the report said "(none — see degradations)".
    """
    db = seed_db(tmp_path / 'db.sqlite', {'runqueue_ratio': [1.0, 2.0, 3.0]})
    local = write_yaml(tmp_path / 'local.yaml', """
psi_admission:
  enabled: true
  runqueue_ratio: "4.0"
  cpu_some_avg10: 85.0
""")

    result = run_script('--db', str(db), '--config', str(local),
                        '--peer-config', str(tmp_path / 'absent.yaml'),
                        '--arm', 'runqueue_ratio', '--no-report')

    assert result.returncode == 0, result.stderr
    payload = trailing_json(result.stdout)
    assert 'local_threshold_not_numeric' in payload['degradations'], payload
    assert payload['unusable_thresholds'] == {'runqueue_ratio': "'4.0' (str)"}
    assert 'runqueue_ratio' not in payload['configured']
    assert any('local_threshold_not_numeric' in d
               for d in payload['degradation_details']), payload
    # And the human half of the same promise.
    assert 'UNUSABLE' in result.stdout
