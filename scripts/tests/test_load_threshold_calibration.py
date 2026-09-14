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

    run = module.longest_hold_run(values, 5.0, spacing_seconds=5)

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

    assert module.hold_fraction(block, 5.0) == pytest.approx(0.2)
    assert module.hold_fraction(blips, 5.0) == pytest.approx(0.2)
    assert module.longest_hold_run(block, 5.0, spacing_seconds=5)['ticks'] == 200
    assert module.longest_hold_run(blips, 5.0, spacing_seconds=5)['ticks'] == 1


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


def test_the_human_report_carries_the_unit_label(tmp_path: Path):
    """"4.0" means nothing to the human reading the escalation without it."""
    db = seed_db(tmp_path / 'db.sqlite', {'runqueue_ratio': [1.0, 2.0, 3.0]})

    result = run_script('--db', str(db), '--arm', 'runqueue_ratio', '--no-report')

    assert result.returncode == 0, result.stderr
    assert 'runnable threads per CPU' in result.stdout, result.stdout


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
    """Exits 0 and names whatever degradation is true, pinning neither
    project's current values — those are operator decisions that change."""
    db = seed_db(tmp_path / 'db.sqlite', {'runqueue_ratio': [1.0, 2.0]})

    result = run_script(
        '--db', str(db),
        '--config', str(REPO_ROOT / 'dark-factory-orchestrator.yaml'),
        '--no-report')

    assert result.returncode == 0, result.stderr
    payload = trailing_json(result.stdout)
    assert isinstance(payload['degradations'], list)


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


def test_the_flags_are_produced_for_both_the_local_and_peer_blocks(tmp_path: Path):
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

    module.main([
        '--db', str(db), '--config', str(local), '--peer-config', str(peer),
        '--no-report',
    ])
    local_block, _ = module.load_psi_admission_block(local, 'local')
    peer_block, _ = module.load_psi_admission_block(peer, 'peer')

    assert module.compare_to_code_defaults(
        local_block, _INJECTED_DEFAULTS)['restates_default'] == ['mem_some_avg10']
    assert module.compare_to_code_defaults(
        peer_block, _INJECTED_DEFAULTS)['restates_default'] == ['io_some_avg10']


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
    tmp_path: Path
):
    module = load_script()
    db = seed_db(tmp_path / 'db.sqlite', {'runqueue_ratio': [1.0, 2.0]})
    report_dir = tmp_path / 'plans'
    report_dir.mkdir()

    rc = module.main(['--db', str(db), '--report-dir', str(report_dir)])

    assert rc == 0
    written = list(report_dir.glob('load-threshold-calibration-*.md'))
    assert len(written) == 1, written
    assert written[0].name.count('-') == 5, written[0].name
    assert written[0].read_text().startswith('# Load-threshold calibration')


def test_the_filename_heading_and_commit_subject_share_one_clock_read(tmp_path: Path):
    """One `datetime.now(UTC)`, reused — not three reads that can straddle
    midnight and produce a report whose name, heading and commit disagree."""
    repo = make_repo(tmp_path)
    db = seed_db(tmp_path / 'db.sqlite', {'runqueue_ratio': [1.0, 2.0]})
    module = load_script()

    module.main(['--db', str(db), '--report-dir', str(repo / 'plans'), '--commit'])

    written, = (repo / 'plans').glob('load-threshold-calibration-*.md')
    date = written.stem.rsplit('-', 3)[-3:]
    date_str = '-'.join(date)
    assert date_str in written.read_text().splitlines()[0]
    assert date_str in git_out(repo, 'log', '-1', '--pretty=%s')


def test_no_report_prints_everything_but_writes_no_file(tmp_path: Path):
    """print-only, never compute-less."""
    db = seed_db(tmp_path / 'db.sqlite', {'runqueue_ratio': [1.0, 2.0, 3.0]})
    report_dir = tmp_path / 'plans'
    report_dir.mkdir()

    result = run_script('--db', str(db), '--report-dir', str(report_dir),
                        '--no-report')

    assert result.returncode == 0, result.stderr
    assert list(report_dir.iterdir()) == []
    assert '# Load-threshold calibration' in result.stdout
    assert trailing_json(result.stdout)['percentiles']


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
