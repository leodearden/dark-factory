"""Memory-eval dashboard backend — aggregator, shape fn and route (task 3215).

Covers ``dashboard.data.memory_evals.build_memory_evals``,
``dashboard.data.redux_api.shape_memory_evals`` and
``GET /api/v2/dashboard/memory-evals``.

**Artifact-only boundary.** The builders below emit the memory-eval artifact
formats as *plain JSON bytes* (``json.dumps`` + the writer's canonical
serialization).  They deliberately do NOT import ``shared.memory_eval_metrics``
/ ``shared.memory_eval_limits``: importing the producer module would make this
test agree with the producer's in-memory objects instead of with the on-disk
*file format*, which is the only thing the dashboard actually reads
(``docs/prds/memory-eval-program.md`` M1: "The dashboard consumes artifacts
only, never the module").  The committed exemplars under
:data:`_FIXTURE_ROOT` are the contract this scaffolding mirrors.

Record shapes:

* metrics — ``shared/tests/fixtures/memory_eval/README.md`` §"Record schema (M1)"
* limits  — ``shared/tests/fixtures/memory_eval/e1-retrieval-health/limits-current.json``
* verdicts — the M2 amendment pinned at ``docs/prds/memory-eval-program.md:38``
  (``{eval_id, metric_id, item|window, fingerprint, verdict, value, limit_ref,
  run_stamp}`` plus a ``storm_escape`` block)
"""

from __future__ import annotations

import builtins
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

from dashboard.config import DashboardConfig

# The real committed exemplar tree (3207), consumed read-only by the
# consumer-side boundary test.  dashboard/tests/ -> dashboard/ -> <repo root>.
_FIXTURE_ROOT = Path(__file__).parents[2] / 'shared' / 'tests' / 'fixtures' / 'memory_eval'

# Default corpus block, shaped like every committed exemplar's.
_CORPUS: dict[str, Any] = {
    'project_id': 'dark_factory',
    'counts': {'entities_and_relations': 1204, 'temporal_facts': 588},
}

# Distinguishes "caller passed run_stamp=None (omit it)" from "caller said
# nothing (default it to the filename stamp)".
_UNSET: Any = object()


def _make_config(tmp_path: Path, *, known_project_roots: list[Path] | None = None) -> DashboardConfig:
    """DashboardConfig pointed at *tmp_path* (mirrors test_escalation_lifecycle_gate._make_config)."""
    return DashboardConfig(project_root=tmp_path, known_project_roots=known_project_roots or [])


def _dump(path: Path, payload: Any) -> Path:
    """Write *payload* in the producers' canonical serialization.

    ``json.dumps(..., indent=2, sort_keys=True, ensure_ascii=False)`` plus a
    trailing newline — the exact form the README pins for the committed
    exemplars, so the tmp_path trees are byte-shaped like the real ones.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + '\n')
    return path


def _metric(
    metric_id: str,
    kind: str,
    value: float,
    *,
    n: int = 30,
    denominator: int | None = None,
    direction: str | None = None,
    items: list[dict] | None = None,
    details_path: str | None = None,
) -> dict[str, Any]:
    """One M1 metric record; ``None``-valued optional fields are omitted."""
    record: dict[str, Any] = {'metric_id': metric_id, 'kind': kind, 'value': value, 'n': n}
    if denominator is not None:
        record['denominator'] = denominator
    if direction is not None:
        record['direction'] = direction
    if items is not None:
        record['items'] = items
    if details_path is not None:
        record['details_path'] = details_path
    return record


def _write_metrics(
    root: Path,
    eval_id: str,
    stamp: str,
    metrics: list[dict],
    *,
    corpus: Any = _UNSET,  # default: the shared _CORPUS block
    filename: str | None = None,
    run_stamp: Any = _UNSET,  # default: the in-body stamp == *stamp*
) -> Path:
    """Write ``<root>/<eval_id>/metrics-<stamp>.json``.

    *filename* overrides the whole basename (for the stamp-less names the
    committed ``malformed/`` dir carries).  *run_stamp* overrides the in-body
    stamp; pass ``None`` explicitly to omit it entirely (the
    ``missing_run_stamp`` degraded case).  *corpus* follows the same
    convention — pass ``None`` explicitly to write a run with NO corpus block,
    which the M1 schema permits and which the latest-run rule has to answer for.
    """
    body: dict[str, Any] = {
        'schema_version': 1,
        'eval_id': eval_id,
        'metrics': list(metrics),
    }
    resolved_corpus = _CORPUS if corpus is _UNSET else corpus
    if resolved_corpus is not None:
        body['corpus'] = dict(resolved_corpus)
    in_body = stamp if run_stamp is _UNSET else run_stamp
    if in_body is not None:
        body['run_stamp'] = in_body
    return _dump(root / eval_id / (filename or f'metrics-{stamp}.json'), body)


def _write_limits(
    root: Path,
    eval_id: str,
    *,
    run_stamp: str,
    alpha: float = 0.002777777777777778,
    false_alarm_budget: float = 1.0,
    runs_per_quarter: int = 90,
    min_samples: int = 10,
    baseline_window: int = 3,
    baseline_run_stamps: list[str] | None = None,
    grandfather_set_hash: str = 'f8c4' * 16,
    generator: str = 'shared.memory_eval_limits',
    verdicts: list[dict] | None = None,
    **extra: Any,
) -> Path:
    """Write the per-eval ``<root>/<eval_id>/limits-current.json``.

    Per-eval, NOT at the root: ``shared.memory_eval_limits.limits_artifact_path``
    returns ``<root>/<eval_id>/limits-current.json`` and the committed exemplar
    sits at ``e1-retrieval-health/limits-current.json``.

    *verdicts* is the limits artifact's OWN embedded per-metric array
    (``{metric_id, rule_kind, status, p_value, ...}``).  Only ``rule_kind`` is
    a dashboard input; the ``status``/``p_value``/``alarms`` fields are written
    here precisely so the tests can assert the reader ignores them.
    """
    body: dict[str, Any] = {
        'schema_version': 1,
        'eval_id': eval_id,
        'run_stamp': run_stamp,
        'alpha': alpha,
        'false_alarm_budget': false_alarm_budget,
        'runs_per_quarter': runs_per_quarter,
        'min_samples': min_samples,
        'baseline_window': baseline_window,
        'baseline_run_stamps': list(baseline_run_stamps or []),
        'grandfather_set_hash': grandfather_set_hash,
        'generator': generator,
        'verdicts': list(verdicts or []),
    }
    body.update(extra)
    return _dump(root / eval_id / 'limits-current.json', body)


def _limits_verdict(metric_id: str, rule_kind: str, *, status: str = 'ok', **extra: Any) -> dict[str, Any]:
    """One entry of the LIMITS artifact's embedded ``verdicts[]`` array.

    Note the deliberately different vocabulary from the M2 verdicts artifact
    (``baseline_snapshot|ok|alarm|improved|insufficient_data`` vs
    ``alarm|no_alarm|insufficient_data|grandfathered``) and the absent
    ``fingerprint`` — which is why this is a provenance source only.
    """
    return {'metric_id': metric_id, 'rule_kind': rule_kind, 'status': status, 'alarms': [], **extra}


def _verdict(
    eval_id: str,
    metric_id: str,
    verdict: str,
    *,
    fingerprint: str,
    value: float | None = None,
    limit_ref: str | None = None,
    run_stamp: str | None = None,
    item: str | None = None,
    window: str | None = None,
) -> dict[str, Any]:
    """One entry of the M2-amendment verdicts artifact."""
    entry: dict[str, Any] = {
        'eval_id': eval_id,
        'metric_id': metric_id,
        'verdict': verdict,
        'fingerprint': fingerprint,
        'value': value,
        'limit_ref': limit_ref,
        'run_stamp': run_stamp,
    }
    if item is not None:
        entry['item'] = item
    if window is not None:
        entry['window'] = window
    return entry


def _write_verdicts(
    root: Path,
    entries: list[dict],
    *,
    storm_escape: dict | None = None,
    eval_id: str | None = None,
    run_stamp: str | None = None,
) -> Path:
    """Write the ROOT ``<root>/verdicts-current.json``.

    Root-scoped, not per-eval: ``docs/prds/memory-eval-program.md:38`` pins
    ``fused-memory/data/memory-evals/verdicts-<STAMP>.json``, and the
    ``storm_escape`` block is per-RUN across the whole program.  *eval_id*
    writes a per-EVAL copy instead — used only to prove the reader names a
    misplaced file rather than silently falling back to it.
    """
    body: dict[str, Any] = {'schema_version': 1, 'entries': list(entries)}
    if run_stamp is not None:
        body['run_stamp'] = run_stamp
    if storm_escape is not None:
        body['storm_escape'] = dict(storm_escape)
    target = root / 'verdicts-current.json' if eval_id is None else root / eval_id / 'verdicts-current.json'
    return _dump(target, body)


def _write_escalation(
    esc_dir: Path,
    esc_id: str,
    *,
    category: str = 'eval_regression',
    dedupe_fingerprint: str | None = None,
    summary: str = 'memory-eval regression',
    severity: str = 'blocking',
    level: int = 1,
    timestamp: str = '2026-07-30T03:15:00+00:00',
    status: str = 'pending',
    **extra: Any,
) -> Path:
    """Write one pending escalation JSON into a queue dir (``<esc_dir>/<esc_id>.json``).

    Field names mirror ``escalation.models.Escalation`` (``timestamp``, not
    ``created_at``) since that is what the queue actually serialises and what
    ``load_queue_escalations`` passes through unchanged.
    """
    body: dict[str, Any] = {
        'id': esc_id,
        'task_id': 'memory-eval-e1',
        'agent_role': 'memory-eval-runner',
        'severity': severity,
        'category': category,
        'summary': summary,
        'detail': '',
        'timestamp': timestamp,
        'status': status,
        'level': level,
        'dedupe_fingerprint': dedupe_fingerprint,
    }
    body.update(extra)
    return _dump(esc_dir / f'{esc_id}.json', body)


# ---------------------------------------------------------------------------
# step-1 — the config property
# ---------------------------------------------------------------------------


class TestConfigProperty:
    """``DashboardConfig.memory_evals_dir`` — contract-fixed, NOT env-indirected."""

    def test_memory_evals_dir_is_project_relative(self, tmp_path: Path) -> None:
        config = DashboardConfig(project_root=tmp_path)

        assert config.memory_evals_dir == tmp_path.resolve() / 'fused-memory' / 'data' / 'memory-evals'

    def test_memory_evals_dir_ignores_runtime_data_dir_env_vars(self, tmp_path, monkeypatch) -> None:
        """The M1 artifact path is contract-fixed — no ``_runtime_data_dir`` indirection.

        ``QUEUE_DATA_DIR`` / ``RECONCILIATION_DATA_DIR`` relocate the *managed*
        fused-memory runtime dirs (``Config._runtime_data_dir``) to an
        XDG-rooted path outside
        the watched tree.  The memory-eval artifacts are NOT among them: they
        are written relative to the repo at the path
        ``docs/prds/memory-eval-program.md`` §3 pins, so a relocation env var
        must leave this property untouched.  This test pins the deliberate
        divergence, so a future "make it consistent" refactor fails loudly.
        """
        decoy = tmp_path / 'decoy-runtime'
        for var in ('MEMORY_EVAL_DATA_DIR', 'RECONCILIATION_DATA_DIR', 'QUEUE_DATA_DIR'):
            monkeypatch.setenv(var, str(decoy))

        config = DashboardConfig(project_root=tmp_path)

        assert config.memory_evals_dir == tmp_path.resolve() / 'fused-memory' / 'data' / 'memory-evals'
        assert decoy not in config.memory_evals_dir.parents
        # Control: an env-indirected sibling DOES move, proving the vars are live.
        assert config.reconciliation_escalations_dir == decoy / 'escalations'


# ---------------------------------------------------------------------------
# step-3 — trend assembly and latest-run scalars
# ---------------------------------------------------------------------------

_PAYLOAD_KEYS = {
    'generated_at',
    'root_present',
    # Run-scoped across the whole program, so it is a TOP-LEVEL key: one banner
    # source rather than the UI electing an eval row to read it from, and it
    # still resolves when the root enumerates to zero eval dirs.
    'storm_escape',
    'evals',
    'issues',
    'issue_count',
    'unmatched_escalations',
}


def _two_eval_tree(tmp_path: Path) -> tuple[Path, Path]:
    """A root with ``eval-a`` (3 runs, all four M1 kinds) and ``eval-b`` (1 run).

    ``eval-a``'s newest run introduces ``latecomer``, a metric absent from the
    two older runs — the index-alignment case.  Both evals get limits and a
    root verdicts artifact so the healthy tree reports zero issues.
    """
    root = tmp_path / 'memory-evals'
    esc_dir = tmp_path / 'escalations'
    esc_dir.mkdir(parents=True, exist_ok=True)

    stamps = ['20260701T031500Z', '20260702T031500Z', '20260703T031500Z']
    # Exactly-representable literals: the reader passes values through
    # verbatim, so an expected value computed here would test float
    # arithmetic rather than the passthrough.
    proportions = [0.8, 0.75, 0.5]
    for i, stamp in enumerate(stamps):
        metrics = [
            _metric('canonical-in-top-5', 'proportion', proportions[i], denominator=30, direction='lower_is_worse'),
            _metric('dangling-pointers', 'count', float(4 + i), direction='higher_is_worse'),
            _metric('search-latency-p50-ms', 'scalar', float(40 + i)),
            _metric(
                'topic-canonical-present',
                'tripwire',
                1.0,
                n=2,
                items=[{'item_key': 't-a', 'passed': True}, {'item_key': 't-b', 'passed': False}],
            ),
        ]
        if stamp == stamps[-1]:
            metrics.append(_metric('latecomer', 'count', 7.0, n=5, direction='higher_is_worse'))
        _write_metrics(root, 'eval-a', stamp, metrics)

    _write_metrics(
        root,
        'eval-b',
        '20260801T031500Z',
        [_metric('solo-metric', 'scalar', 3.5, n=1)],
        corpus={'project_id': 'other_project', 'counts': {'temporal_facts': 7}},
    )

    _write_limits(root, 'eval-a', run_stamp=stamps[-1], baseline_run_stamps=stamps[:2])
    _write_limits(root, 'eval-b', run_stamp='20260801T031500Z')
    _write_verdicts(root, [])
    return root, esc_dir


class TestTrendsAndCurrentValues:
    """Enumeration, trend assembly and latest-run scalars — the DD5 generic path."""

    def test_payload_shape_and_generic_enumeration(self, tmp_path: Path) -> None:
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _two_eval_tree(tmp_path)
        payload = build_memory_evals(root, esc_dir)

        assert set(payload) == _PAYLOAD_KEYS
        assert payload['root_present'] is True
        assert isinstance(payload['generated_at'], str) and payload['generated_at']
        # DD5: both eval dirs appear with zero per-eval code, sorted by eval_id.
        assert [e['eval_id'] for e in payload['evals']] == ['eval-a', 'eval-b']

    def test_run_stamps_are_oldest_first_and_counted(self, tmp_path: Path) -> None:
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _two_eval_tree(tmp_path)
        eval_a = build_memory_evals(root, esc_dir)['evals'][0]

        assert eval_a['run_stamps'] == ['20260701T031500Z', '20260702T031500Z', '20260703T031500Z']
        assert eval_a['run_count'] == len(eval_a['run_stamps']) == 3
        assert eval_a['runs_on_disk'] == 3
        assert eval_a['truncated'] is False

    def test_trend_is_chartdata_over_the_shared_run_axis(self, tmp_path: Path) -> None:
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _two_eval_tree(tmp_path)
        eval_a = build_memory_evals(root, esc_dir)['evals'][0]
        by_id = {m['metric_id']: m for m in eval_a['metrics']}

        assert set(by_id) == {
            'canonical-in-top-5',
            'dangling-pointers',
            'search-latency-p50-ms',
            'topic-canonical-present',
            'latecomer',
        }
        for metric in eval_a['metrics']:
            trend = metric['trend']
            assert set(trend) == {'labels', 'values'}
            assert trend['labels'] == eval_a['run_stamps']
            assert len(trend['values']) == len(trend['labels'])

        assert by_id['dangling-pointers']['trend']['values'] == [4.0, 5.0, 6.0]

    def test_metric_absent_from_older_runs_gets_none_holes(self, tmp_path: Path) -> None:
        """Index alignment, not a shifted axis — a gap must read as a gap."""
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _two_eval_tree(tmp_path)
        eval_a = build_memory_evals(root, esc_dir)['evals'][0]
        latecomer = next(m for m in eval_a['metrics'] if m['metric_id'] == 'latecomer')

        assert latecomer['trend']['labels'] == eval_a['run_stamps']
        assert latecomer['trend']['values'] == [None, None, 7.0]

    def test_scalars_come_from_the_latest_run(self, tmp_path: Path) -> None:
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _two_eval_tree(tmp_path)
        eval_a = build_memory_evals(root, esc_dir)['evals'][0]
        by_id = {m['metric_id']: m for m in eval_a['metrics']}

        prop = by_id['canonical-in-top-5']
        assert prop['current_value'] == 0.5  # the 20260703 run, not 0.8/0.75
        assert prop['kind'] == 'proportion'
        assert prop['n'] == 30
        assert prop['denominator'] == 30
        assert prop['direction'] == 'lower_is_worse'

        # Fields the kind does not carry are present-and-None, never absent:
        # beta reads a fixed column set (PRD open question 4).
        scalar = by_id['search-latency-p50-ms']
        assert scalar['kind'] == 'scalar'
        assert scalar['denominator'] is None
        assert scalar['direction'] is None
        assert scalar['current_value'] == 42.0

    def test_corpus_mirrors_the_latest_run(self, tmp_path: Path) -> None:
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _two_eval_tree(tmp_path)
        evals = build_memory_evals(root, esc_dir)['evals']

        assert evals[0]['corpus'] == {
            'project_id': 'dark_factory',
            'counts': {'entities_and_relations': 1204, 'temporal_facts': 588},
        }
        assert evals[1]['corpus'] == {'project_id': 'other_project', 'counts': {'temporal_facts': 7}}

    def test_corpus_is_absent_when_the_latest_run_omits_it(self, tmp_path: Path) -> None:
        """The corpus block describes the LATEST run, or it is absent.

        Carrying the newest run that merely HAPPENED to have a corpus presents
        an older run's counts as current with nothing disclosing the skew —
        while every other latest-run scalar on the row (``current_value``,
        ``n``, ``denominator``, ``kind``) comes from the latest run with no
        fallback, and the module elsewhere goes out of its way to disclose
        exactly this kind of drift (``limits.stale_for_latest_run``,
        ``truncated``/``runs_on_disk``).  ``corpus`` is optional in M1, so
        "this run reported no corpus" is a truthful answer; "here are last
        week's counts, unlabelled" is not.
        """
        from dashboard.data.memory_evals import build_memory_evals

        root = tmp_path / 'memory-evals'
        esc_dir = tmp_path / 'escalations'
        _write_metrics(
            root, 'eval-a', '20260703T031500Z', [_metric('m', 'count', 1.0)],
            corpus={'project_id': 'dark_factory', 'counts': {'temporal_facts': 3}},
        )
        # The newer run carries no corpus block at all.
        _write_metrics(root, 'eval-a', '20260704T031500Z', [_metric('m', 'count', 2.0)], corpus=None)

        row = build_memory_evals(root, esc_dir)['evals'][0]

        assert row['latest_run_stamp'] == '20260704T031500Z'
        assert row['corpus'] is None

    def test_healthy_tree_reports_no_issues(self, tmp_path: Path) -> None:
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _two_eval_tree(tmp_path)
        payload = build_memory_evals(root, esc_dir)

        assert payload['issues'] == []
        assert payload['issue_count'] == 0


# ---------------------------------------------------------------------------
# step-5 — the trend window cap (no silent caps)
# ---------------------------------------------------------------------------


def _n_run_tree(tmp_path: Path, count: int) -> tuple[Path, Path]:
    """A single eval with *count* runs, stamped 2026-07-01 onwards."""
    root = tmp_path / 'memory-evals'
    esc_dir = tmp_path / 'escalations'
    esc_dir.mkdir(parents=True, exist_ok=True)
    for day in range(1, count + 1):
        stamp = f'202607{day:02d}T031500Z'
        _write_metrics(root, 'eval-a', stamp, [_metric('dangling-pointers', 'count', float(day), direction='higher_is_worse')])
    return root, esc_dir


def _run_stamp(index: int) -> str:
    """The *index*-th run stamp, spaced one hour apart from 2026-07-01T00:15Z.

    Hour-spaced rather than day-spaced so a run count larger than a month still
    produces real, parseable, lexicographically-chronological stamps (filename
    order IS the producer's ordering contract, so the stamps must sort right).
    """
    day, hour = divmod(index, 24)
    return f'202607{day + 1:02d}T{hour:02d}1500Z'


def _many_run_tree(tmp_path: Path, count: int) -> tuple[Path, Path]:
    """A single eval with *count* hour-spaced runs — usable past a 31-day month."""
    root = tmp_path / 'memory-evals'
    esc_dir = tmp_path / 'escalations'
    esc_dir.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        _write_metrics(
            root, 'eval-a', _run_stamp(index),
            [_metric('dangling-pointers', 'count', float(index), direction='higher_is_worse')],
        )
    return root, esc_dir


class TestTrendCap:
    """The trailing-window cap discloses what it dropped — it never truncates silently."""

    def test_cap_keeps_the_most_recent_runs_and_says_so(self, tmp_path, monkeypatch) -> None:
        import dashboard.data.memory_evals as memory_evals_module
        from dashboard.data.memory_evals import build_memory_evals

        monkeypatch.setattr(memory_evals_module, '_TREND_RUN_CAP', 3)
        root, esc_dir = _n_run_tree(tmp_path, 5)

        eval_a = build_memory_evals(root, esc_dir)['evals'][0]

        assert eval_a['truncated'] is True
        assert eval_a['runs_on_disk'] == 5
        assert eval_a['run_count'] == 3
        # The OLDEST two are dropped, not the newest — a trend that ends three
        # runs ago would read as current.
        assert eval_a['run_stamps'] == ['20260703T031500Z', '20260704T031500Z', '20260705T031500Z']
        trend = eval_a['metrics'][0]['trend']
        assert trend['labels'] == eval_a['run_stamps']
        assert trend['values'] == [3.0, 4.0, 5.0]

    def test_truncated_reports_the_actual_drop_not_a_constant(self, tmp_path, monkeypatch) -> None:
        import dashboard.data.memory_evals as memory_evals_module
        from dashboard.data.memory_evals import build_memory_evals

        monkeypatch.setattr(memory_evals_module, '_TREND_RUN_CAP', 3)
        root, esc_dir = _n_run_tree(tmp_path, 2)

        eval_a = build_memory_evals(root, esc_dir)['evals'][0]

        assert eval_a['truncated'] is False
        assert eval_a['runs_on_disk'] == eval_a['run_count'] == 2

    def test_the_real_cap_takes_effect_at_ninety_runs(self, tmp_path: Path) -> None:
        """One screen of trend == one alpha-derivation window, unmonkeypatched.

        90 is the PRD's declared lean for open question 2 and matches
        ``runs_per_quarter=90`` in the committed limits artifact, so the
        displayed window is exactly the window the limits govern.  Asserted by
        its EFFECT on a 95-run tree rather than by restating the constant: a
        test that re-reads the knob can only fail when someone deliberately
        turns it, and pins no behaviour at all.
        """
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _many_run_tree(tmp_path, 95)

        eval_a = build_memory_evals(root, esc_dir)['evals'][0]

        assert eval_a['runs_on_disk'] == 95
        assert eval_a['run_count'] == 90
        assert eval_a['truncated'] is True
        # The trend is index-aligned to the capped axis, not to what is on disk.
        assert len(eval_a['run_stamps']) == 90
        assert len(eval_a['metrics'][0]['trend']['values']) == 90
        # The newest run survives the cap; the oldest five are what was dropped.
        assert eval_a['latest_run_stamp'] == eval_a['run_stamps'][-1] == _run_stamp(94)
        assert eval_a['run_stamps'][0] == _run_stamp(5)


# ---------------------------------------------------------------------------
# step-7 — limits provenance (a provenance source, never a verdict source)
# ---------------------------------------------------------------------------

# The whitelist.  The limits artifact on disk carries a good deal more
# (`alarms`, `alarmed_metric_count`, `grandfather_set`, `verdicts`,
# `snapshotted_metric_ids`); asserting the block is EXACTLY these keys is what
# pins that the extra fields are not quietly along for the ride.
_LIMITS_KEYS = {
    'alpha',
    'false_alarm_budget',
    'runs_per_quarter',
    'min_samples',
    'baseline_window',
    'baseline_run_stamps',
    'grandfather_set_hash',
    'run_stamp',
    'generator',
    'stale_for_latest_run',
}

_LIMITS_RUNS = ['20260704T031500Z', '20260705T031500Z']
_LIMITS_BASELINE = ['20260701T031500Z', '20260702T031500Z', '20260703T031500Z']


def _limits_tree(
    tmp_path: Path,
    *,
    limits_run_stamp: str = _LIMITS_RUNS[0],
    write_limits: bool = True,
    write_verdicts: bool = True,
) -> tuple[Path, Path]:
    """One eval, two runs, and a limits artifact whose embedded verdicts disagree.

    ``unruled-metric`` deliberately has NO entry in the limits artifact's
    ``verdicts[]`` array — so ``rule_kind`` has to be a lookup keyed on
    ``metric_id``, not a positional zip over two independently-ordered lists.
    """
    root = tmp_path / 'memory-evals'
    esc_dir = tmp_path / 'escalations'
    esc_dir.mkdir(parents=True, exist_ok=True)

    for i, stamp in enumerate(_LIMITS_RUNS):
        _write_metrics(root, 'eval-a', stamp, [
            _metric('canonical-in-top-5', 'proportion', 0.4, denominator=30, direction='lower_is_worse'),
            _metric('dangling-pointers', 'count', float(4 + i), direction='higher_is_worse'),
            _metric('topic-canonical-present', 'tripwire', 2.0, n=8),
            _metric('search-latency-p50-ms', 'scalar', 44.0),
            _metric('unruled-metric', 'scalar', 1.0),
        ])

    if write_limits:
        _write_limits(
            root,
            'eval-a',
            run_stamp=limits_run_stamp,
            baseline_run_stamps=_LIMITS_BASELINE,
            verdicts=[
                _limits_verdict('canonical-in-top-5', 'proportion', status='alarm', p_value=1.8424481488582588e-06),
                _limits_verdict('dangling-pointers', 'count', status='ok', p_value=1.0),
                _limits_verdict('topic-canonical-present', 'tripwire', status='alarm', p_value=None),
                _limits_verdict('search-latency-p50-ms', 'scalar', status='ok', p_value=None),
            ],
        )
    if write_verdicts:
        _write_verdicts(root, [])
    return root, esc_dir


def _only(rows: list[dict], metric_id: str) -> dict:
    """The single metric row with *metric_id* (fails loudly if absent)."""
    matches = [row for row in rows if row['metric_id'] == metric_id]
    assert len(matches) == 1, f'expected exactly one {metric_id!r} row, got {len(matches)}'
    return matches[0]


class TestLimitsProvenance:
    """The limits artifact contributes provenance + ``rule_kind``. Nothing else.

    Design decision: ``verdicts-current.json`` is the SOLE verdict source.  The
    limits artifact embeds its own ``verdicts[]``/``alarms[]`` in a DIFFERENT
    vocabulary (``baseline_snapshot|ok|alarm|improved|insufficient_data``) and
    carries no ``fingerprint``, so it cannot participate in the escalation join
    at all.  Reading both would hand the UI two disagreeing alarm truths.
    """

    def test_provenance_passes_through_verbatim(self, tmp_path: Path) -> None:
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _limits_tree(tmp_path)

        limits = build_memory_evals(root, esc_dir)['evals'][0]['limits']

        assert set(limits) == _LIMITS_KEYS
        assert limits['alpha'] == 0.002777777777777778
        assert limits['false_alarm_budget'] == 1.0
        assert limits['runs_per_quarter'] == 90
        assert limits['min_samples'] == 10
        assert limits['baseline_window'] == 3
        assert limits['baseline_run_stamps'] == _LIMITS_BASELINE
        assert limits['grandfather_set_hash'] == 'f8c4' * 16
        assert limits['run_stamp'] == _LIMITS_RUNS[0]
        assert limits['generator'] == 'shared.memory_eval_limits'

    def test_rule_kind_is_looked_up_per_metric_id(self, tmp_path: Path) -> None:
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _limits_tree(tmp_path)

        rows = build_memory_evals(root, esc_dir)['evals'][0]['metrics']

        assert _only(rows, 'canonical-in-top-5')['rule_kind'] == 'proportion'
        assert _only(rows, 'dangling-pointers')['rule_kind'] == 'count'
        assert _only(rows, 'topic-canonical-present')['rule_kind'] == 'tripwire'
        assert _only(rows, 'search-latency-p50-ms')['rule_kind'] == 'scalar'
        # No limits entry for this one — absent means absent, not the
        # neighbouring row's rule_kind.
        assert _only(rows, 'unruled-metric')['rule_kind'] is None

    def test_stale_for_latest_run_discloses_the_skew(self, tmp_path: Path) -> None:
        """Provenance stamped at an older run does not govern the displayed one."""
        from dashboard.data.memory_evals import build_memory_evals

        stale_root, esc_dir = _limits_tree(tmp_path / 'stale', limits_run_stamp=_LIMITS_RUNS[0])
        current_root, _ = _limits_tree(tmp_path / 'current', limits_run_stamp=_LIMITS_RUNS[-1])

        stale = build_memory_evals(stale_root, esc_dir)['evals'][0]
        current = build_memory_evals(current_root, esc_dir)['evals'][0]

        assert stale['latest_run_stamp'] == _LIMITS_RUNS[-1]
        assert stale['limits']['stale_for_latest_run'] is True
        assert current['limits']['stale_for_latest_run'] is False

    def test_limits_status_is_never_read_as_a_verdict(self, tmp_path: Path) -> None:
        """Negative space: the limits vocabulary does not leak onto metric rows."""
        from dashboard.data.memory_evals import build_memory_evals

        # No root verdicts artifact at all — so the ONLY 'alarm' string on disk
        # for this metric is the limits artifact's, and it must not be read.
        root, esc_dir = _limits_tree(tmp_path, write_verdicts=False)

        rows = build_memory_evals(root, esc_dir)['evals'][0]['metrics']

        alarmed_in_limits = _only(rows, 'canonical-in-top-5')
        assert 'verdict' in alarmed_in_limits, 'the verdict column must exist even when empty'
        assert alarmed_in_limits['verdict'] is None
        for row in rows:
            assert 'status' not in row
            assert 'p_value' not in row
            assert 'alarms' not in row
            assert 'baseline' not in row

    def test_missing_limits_is_named_not_silent(self, tmp_path: Path) -> None:
        """Metrics with no limits beside them = the evaluator is not completing."""
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _limits_tree(tmp_path, write_limits=False)

        payload = build_memory_evals(root, esc_dir)

        assert payload['evals'][0]['limits'] is None
        assert payload['issue_count'] == len(payload['issues']) == 1
        issue = payload['issues'][0]
        assert issue['kind'] == 'missing_limits'
        assert issue['eval_id'] == 'eval-a'
        assert issue['path'] == str(root / 'eval-a' / 'limits-current.json')


# ---------------------------------------------------------------------------
# step-9 — verdicts (the sole verdict source), joined on (eval_id, metric_id)
# ---------------------------------------------------------------------------

_VERDICT_RUN = '20260705T031500Z'

# The M2 verdict vocabulary in full.  All four must survive the read verbatim:
# any dashboard-side mapping table would be a second vocabulary to keep in sync
# with the evaluator's, and the first divergence would silently mislabel an
# alarm.
_VERDICT_VALUES = ('alarm', 'no_alarm', 'insufficient_data', 'grandfathered')


def _good_verdict_entries() -> list[dict]:
    """One entry per verdict value, all resolving onto real ``eval-a`` metrics."""
    return [
        _verdict(
            'eval-a', 'canonical-in-top-5', 'alarm',
            fingerprint='fp-canonical-alarm', value=0.4,
            limit_ref='alpha=0.002777777777777778', run_stamp=_VERDICT_RUN,
        ),
        _verdict(
            'eval-a', 'dangling-pointers', 'no_alarm',
            fingerprint='fp-dangling-clear', value=4.0,
            limit_ref='alpha=0.002777777777777778', run_stamp=_VERDICT_RUN,
        ),
        _verdict(
            'eval-a', 'search-latency-p50-ms', 'insufficient_data',
            fingerprint='fp-latency-thin', value=44.0,
            limit_ref='min_samples=10', run_stamp=_VERDICT_RUN,
        ),
        _verdict(
            'eval-a', 'topic-canonical-present', 'grandfathered',
            fingerprint='fp-topic-grandfathered', value=2.0,
            limit_ref='grandfather_set_hash=f8c46981', run_stamp=_VERDICT_RUN,
            item='t-recon-watcher-triage',
        ),
    ]


def _verdicts_tree(
    tmp_path: Path,
    *,
    entries: list[dict] | None = None,
    write_verdicts: bool = True,
    per_eval_copy: bool = False,
) -> tuple[Path, Path]:
    """``eval-a`` with one run, limits, and a root verdicts artifact.

    ``unjudged-metric`` has no verdict entry.  *per_eval_copy* writes the
    verdicts artifact to the WRONG place (``<root>/eval-a/``) to prove the
    reader names it rather than silently falling back to it.
    """
    root = tmp_path / 'memory-evals'
    esc_dir = tmp_path / 'escalations'
    esc_dir.mkdir(parents=True, exist_ok=True)

    _write_metrics(root, 'eval-a', _VERDICT_RUN, [
        # current_value 0.5 vs the verdict entry's value 0.4 — deliberately
        # different, so the two sources cannot be confused for one another.
        _metric('canonical-in-top-5', 'proportion', 0.5, denominator=30, direction='lower_is_worse'),
        _metric('dangling-pointers', 'count', 4.0, direction='higher_is_worse'),
        _metric('topic-canonical-present', 'tripwire', 2.0, n=8),
        _metric('search-latency-p50-ms', 'scalar', 44.0),
        _metric('unjudged-metric', 'scalar', 1.0),
    ])
    _write_limits(root, 'eval-a', run_stamp=_VERDICT_RUN)

    resolved = _good_verdict_entries() if entries is None else entries
    if write_verdicts:
        _write_verdicts(root, resolved, run_stamp=_VERDICT_RUN)
    if per_eval_copy:
        _write_verdicts(root, resolved, run_stamp=_VERDICT_RUN, eval_id='eval-a')
    return root, esc_dir


class TestVerdicts:
    """``verdicts-current.json`` at the ROOT is the sole verdict source.

    Root-scoped, not per-eval: the PRD pins
    ``fused-memory/data/memory-evals/verdicts-<STAMP>.json`` and the
    ``storm_escape`` block is per-RUN across the whole program, which only
    makes sense above the eval dirs.
    """

    def test_all_four_verdict_values_survive_verbatim(self, tmp_path: Path) -> None:
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _verdicts_tree(tmp_path)

        rows = build_memory_evals(root, esc_dir)['evals'][0]['metrics']

        assert _only(rows, 'canonical-in-top-5')['verdict'] == 'alarm'
        assert _only(rows, 'dangling-pointers')['verdict'] == 'no_alarm'
        assert _only(rows, 'search-latency-p50-ms')['verdict'] == 'insufficient_data'
        assert _only(rows, 'topic-canonical-present')['verdict'] == 'grandfathered'
        # Nothing was translated on the way through.
        assert {row['verdict'] for row in rows if row['verdict']} == set(_VERDICT_VALUES)

    def test_verdict_fields_land_on_the_matching_row(self, tmp_path: Path) -> None:
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _verdicts_tree(tmp_path)

        rows = build_memory_evals(root, esc_dir)['evals'][0]['metrics']

        alarmed = _only(rows, 'canonical-in-top-5')
        assert alarmed['fingerprint'] == 'fp-canonical-alarm'
        assert alarmed['limit_ref'] == 'alpha=0.002777777777777778'
        assert alarmed['run_stamp'] == _VERDICT_RUN
        # Two distinct sources, two distinct fields: `value` is what the
        # evaluator judged, `current_value` is what the metrics artifact says.
        assert alarmed['value'] == 0.4
        assert alarmed['current_value'] == 0.5

        grandfathered = _only(rows, 'topic-canonical-present')
        assert grandfathered['item'] == 't-recon-watcher-triage'
        # An entry with no `item` leaves the field empty rather than borrowing
        # the previous entry's.
        assert alarmed['item'] is None

    def test_metric_with_no_entry_has_no_verdict(self, tmp_path: Path) -> None:
        """Absent means absent — never defaulted to ``no_alarm``."""
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _verdicts_tree(tmp_path)

        unjudged = _only(build_memory_evals(root, esc_dir)['evals'][0]['metrics'], 'unjudged-metric')

        assert unjudged['verdict'] is None
        assert unjudged['fingerprint'] is None
        assert unjudged['limit_ref'] is None
        assert unjudged['value'] is None

    def test_orphan_verdict_is_named_not_dropped(self, tmp_path: Path) -> None:
        """A verdict pointing at nothing is a contract drift, not a no-op."""
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _verdicts_tree(tmp_path, entries=[
            _verdict('eval-a', 'no-such-metric', 'alarm', fingerprint='fp-orphan-metric'),
            _verdict('no-such-eval', 'canonical-in-top-5', 'alarm', fingerprint='fp-orphan-eval'),
        ])

        payload = build_memory_evals(root, esc_dir)

        assert payload['issue_count'] == len(payload['issues']) == 2
        assert {issue['kind'] for issue in payload['issues']} == {'orphan_verdict'}
        assert {issue['eval_id'] for issue in payload['issues']} == {'eval-a', 'no-such-eval'}
        detail = ' '.join(issue['detail'] for issue in payload['issues'])
        assert 'no-such-metric' in detail
        assert 'canonical-in-top-5' in detail
        for issue in payload['issues']:
            assert issue['path'] == str(root / 'verdicts-current.json')

    def test_missing_root_verdicts_is_named(self, tmp_path: Path) -> None:
        """Trends with a blank verdict column would otherwise look healthy."""
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _verdicts_tree(tmp_path, write_verdicts=False)

        payload = build_memory_evals(root, esc_dir)

        assert payload['issue_count'] == len(payload['issues']) == 1
        issue = payload['issues'][0]
        assert issue['kind'] == 'missing_verdicts'
        assert issue['path'] == str(root / 'verdicts-current.json')
        assert all(row['verdict'] is None for row in payload['evals'][0]['metrics'])

    def test_misplaced_per_eval_verdicts_is_named_never_used(self, tmp_path: Path) -> None:
        """No silent fallback to the wrong location — drift stays visible."""
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _verdicts_tree(tmp_path, write_verdicts=False, per_eval_copy=True)

        payload = build_memory_evals(root, esc_dir)

        assert payload['issue_count'] == len(payload['issues']) == 1
        issue = payload['issues'][0]
        assert issue['kind'] == 'missing_verdicts'
        assert str(root / 'eval-a' / 'verdicts-current.json') in issue['detail']
        # Named, not read.
        assert all(row['verdict'] is None for row in payload['evals'][0]['metrics'])


# ---------------------------------------------------------------------------
# step-11 — the escalation join, on the whole fingerprint string
# ---------------------------------------------------------------------------

_JOIN_RUN = '20260705T031500Z'

# Flat hex, no delimiters: a reader that tried to derive an eval_id or
# metric_id by SPLITTING the fingerprint would find nothing to split on.  The
# join is `==` over the whole opaque string — the fingerprint is the producer's
# private construction and the dashboard never parses its substructure.
_FP_ALARMED_OPEN = 'a1b2c3d4e5f60718293a4b5c6d7e8f90'
_FP_RECOVERED_OPEN = '0f9e8d7c6b5a49382716f5e4d3c2b1a0'
_FP_ALARMED_UNLINKED = 'ffeeddccbbaa99887766554433221100'
_FP_CLEAR = '00112233445566778899aabbccddeeff'
_FP_WRONG_CATEGORY = 'deadbeefdeadbeefdeadbeefdeadbeef'
_FP_UNMATCHED = 'cafebabecafebabecafebabecafebabe'
_FP_UNJUDGED_OPEN = '13579bdf02468ace13579bdf02468ace'
# The non-alarm, non-no_alarm half of the M2 vocabulary.  Each one gets both a
# linked and an unlinked case: the two answer different questions, and the
# collapsed mapping used to give them the same badge.
_FP_INSUFFICIENT_OPEN = '2468ace02468ace02468ace02468ace0'
_FP_INSUFFICIENT = 'bdf13579bdf13579bdf13579bdf13579'
_FP_GRANDFATHERED_OPEN = 'aabbccdd00112233aabbccdd00112233'
_FP_GRANDFATHERED = '33221100ddccbbaa33221100ddccbbaa'

# The escalation projection carried onto a metric row (and into
# `unmatched_escalations`).  `created_at` is sourced from the queue record's
# `timestamp` — that is the field `escalation.models.Escalation` serialises.
_ESCALATION_KEYS = {'id', 'summary', 'severity', 'level', 'created_at', 'dedupe_fingerprint'}

# An unmatched escalation carries one extra field: WHY it is unlinked.  Without
# it, a storm-suppressed escalation and a genuine parity orphan are one
# undifferentiated list the UI labels "nothing explains this".
_UNMATCHED_KEYS = _ESCALATION_KEYS | {'reason'}

_JOIN_ESC_TIMESTAMP = '2026-07-30T03:15:00+00:00'


def _join_tree(tmp_path: Path) -> tuple[Path, Path]:
    """One eval whose metrics span every parity state alpha owns.

    Covers the full ``(verdict class) x (linked?)`` matrix the payload can
    reach from a single artifact tree: each of the four M2 verdict values with
    and without a linked open escalation, plus the no-entry-at-all row.  The
    storm states are ``_storm_tree``'s.
    """
    root = tmp_path / 'memory-evals'
    esc_dir = tmp_path / 'escalations'
    esc_dir.mkdir(parents=True, exist_ok=True)

    _write_metrics(root, 'eval-a', _JOIN_RUN, [
        _metric(name, 'count', 3.0, direction='higher_is_worse')
        for name in (
            'alarmed-open',
            'recovered-open',
            'alarmed-unlinked',
            'clear-metric',
            'wrong-category',
            'unjudged-metric',
            'insufficient-open',
            'insufficient-metric',
            'grandfathered-open',
            'grandfathered-metric',
        )
    ])
    _write_limits(root, 'eval-a', run_stamp=_JOIN_RUN)
    _write_verdicts(root, [
        _verdict('eval-a', 'alarmed-open', 'alarm', fingerprint=_FP_ALARMED_OPEN, run_stamp=_JOIN_RUN),
        _verdict('eval-a', 'recovered-open', 'no_alarm', fingerprint=_FP_RECOVERED_OPEN, run_stamp=_JOIN_RUN),
        _verdict('eval-a', 'alarmed-unlinked', 'alarm', fingerprint=_FP_ALARMED_UNLINKED, run_stamp=_JOIN_RUN),
        _verdict('eval-a', 'clear-metric', 'no_alarm', fingerprint=_FP_CLEAR, run_stamp=_JOIN_RUN),
        _verdict('eval-a', 'wrong-category', 'alarm', fingerprint=_FP_WRONG_CATEGORY, run_stamp=_JOIN_RUN),
        _verdict(
            'eval-a', 'insufficient-open', 'insufficient_data',
            fingerprint=_FP_INSUFFICIENT_OPEN, run_stamp=_JOIN_RUN,
        ),
        _verdict(
            'eval-a', 'insufficient-metric', 'insufficient_data',
            fingerprint=_FP_INSUFFICIENT, run_stamp=_JOIN_RUN,
        ),
        _verdict(
            'eval-a', 'grandfathered-open', 'grandfathered',
            fingerprint=_FP_GRANDFATHERED_OPEN, run_stamp=_JOIN_RUN,
        ),
        _verdict(
            'eval-a', 'grandfathered-metric', 'grandfathered',
            fingerprint=_FP_GRANDFATHERED, run_stamp=_JOIN_RUN,
        ),
    ], run_stamp=_JOIN_RUN)

    _write_escalation(
        esc_dir, 'esc-alarmed-open',
        dedupe_fingerprint=_FP_ALARMED_OPEN,
        summary='canonical-in-top-5 regressed', severity='blocking', level=1,
        timestamp=_JOIN_ESC_TIMESTAMP,
    )
    _write_escalation(
        esc_dir, 'esc-recovered-open',
        dedupe_fingerprint=_FP_RECOVERED_OPEN,
        summary='dangling-pointers regressed', severity='blocking', level=0,
        timestamp=_JOIN_ESC_TIMESTAMP,
    )
    # Same fingerprint value, different category — must NOT be joined.
    _write_escalation(
        esc_dir, 'esc-wrong-category',
        category='schema_drift', dedupe_fingerprint=_FP_WRONG_CATEGORY,
        timestamp=_JOIN_ESC_TIMESTAMP,
    )
    # An open eval_regression escalation no verdict claims — the reverse
    # direction of the parity question.
    _write_escalation(
        esc_dir, 'esc-unmatched',
        dedupe_fingerprint=_FP_UNMATCHED,
        summary='an eval_regression nothing on disk explains',
        timestamp=_JOIN_ESC_TIMESTAMP,
    )
    # A still-open escalation on a metric the evaluator could NOT judge, and
    # one on a metric it deliberately exempted.  Neither is a recovery.
    _write_escalation(
        esc_dir, 'esc-insufficient-open',
        dedupe_fingerprint=_FP_INSUFFICIENT_OPEN,
        summary='search-latency regressed', timestamp=_JOIN_ESC_TIMESTAMP,
    )
    _write_escalation(
        esc_dir, 'esc-grandfathered-open',
        dedupe_fingerprint=_FP_GRANDFATHERED_OPEN,
        summary='topic-canonical-present regressed', timestamp=_JOIN_ESC_TIMESTAMP,
    )
    return root, esc_dir


def _unjudged_open_tree(tmp_path: Path) -> tuple[Path, Path]:
    """One metric whose verdict entry keys and fingerprints but judges nothing.

    The ONE reachable way a row can be both unjudged and linked: the join only
    runs when the verdict ENTRY carries a fingerprint, so a metric with no
    entry at all can never link.
    """
    root = tmp_path / 'memory-evals'
    esc_dir = tmp_path / 'escalations'
    esc_dir.mkdir(parents=True, exist_ok=True)
    _write_metrics(root, 'eval-a', _JOIN_RUN, [
        _metric('unjudged-open', 'count', 3.0, direction='higher_is_worse'),
    ])
    _write_limits(root, 'eval-a', run_stamp=_JOIN_RUN)
    # Written out literally: ``_verdict()`` always sets a ``verdict``, and the
    # artifact is unvalidated JSON, so an entry missing that one field is a
    # shape the reader has to answer for.
    _write_verdicts(root, [{
        'eval_id': 'eval-a',
        'metric_id': 'unjudged-open',
        'fingerprint': _FP_UNJUDGED_OPEN,
        'run_stamp': _JOIN_RUN,
    }], run_stamp=_JOIN_RUN)
    _write_escalation(
        esc_dir, 'esc-unjudged-open',
        dedupe_fingerprint=_FP_UNJUDGED_OPEN,
        summary='unjudged-open regressed', timestamp=_JOIN_ESC_TIMESTAMP,
    )
    return root, esc_dir


class TestEscalationJoin:
    """Verdict/escalation parity, joined on the whole fingerprint string.

    Alpha owns the two states it can produce from a single artifact tree:
    alarmed+open and recovered+open.  The storm case and the full both-
    directions matrix are gamma's gate and are deliberately not duplicated
    here.
    """

    def test_alarmed_and_open_links_the_escalation(self, tmp_path: Path) -> None:
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _join_tree(tmp_path)

        row = _only(build_memory_evals(root, esc_dir)['evals'][0]['metrics'], 'alarmed-open')

        assert row['parity'] == 'alarmed_open'
        assert set(row['escalation']) == _ESCALATION_KEYS
        assert row['escalation']['id'] == 'esc-alarmed-open'
        assert row['escalation']['summary'] == 'canonical-in-top-5 regressed'
        assert row['escalation']['severity'] == 'blocking'
        assert row['escalation']['level'] == 1
        assert row['escalation']['created_at'] == _JOIN_ESC_TIMESTAMP
        assert row['escalation']['dedupe_fingerprint'] == _FP_ALARMED_OPEN

    def test_recovered_but_still_open_keeps_the_link(self, tmp_path: Path) -> None:
        """The join is on fingerprint, never gated on the verdict.

        A metric that has recovered while its escalation is still open is a
        real, transient state — the watcher has not closed it yet.  Dropping
        the link would make the escalation look orphaned in exactly the window
        an operator is most likely to be looking at it.

        ``recovered_open`` is reserved for THIS case and no other: the
        escalation was filed on a prior alarm and the metric now reads
        ``no_alarm``, so a recovery genuinely happened.  Every other verdict
        class linked to an open escalation gets its own state — see the
        ``insufficient_data``/``grandfathered``/``unjudged`` cases below —
        because calling those a recovery asserts something that never occurred.
        """
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _join_tree(tmp_path)

        row = _only(build_memory_evals(root, esc_dir)['evals'][0]['metrics'], 'recovered-open')

        assert row['verdict'] == 'no_alarm'
        assert row['parity'] == 'recovered_open'
        assert row['escalation']['id'] == 'esc-recovered-open'

    def test_insufficient_data_with_an_open_escalation_is_not_a_recovery(
        self, tmp_path: Path,
    ) -> None:
        """"We could not judge this" is not "this recovered".

        ``insufficient_data`` means the evaluator lacked the samples to reach a
        verdict at all.  Rendering a still-open escalation on such a metric as
        ``recovered_open`` asserts a recovery that never happened — the alarm
        is live and nothing has been shown to have improved.
        """
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _join_tree(tmp_path)

        row = _only(build_memory_evals(root, esc_dir)['evals'][0]['metrics'], 'insufficient-open')

        assert row['verdict'] == 'insufficient_data'
        assert row['parity'] == 'insufficient_data_open'
        assert row['escalation']['id'] == 'esc-insufficient-open'

    def test_insufficient_data_without_a_link_is_not_clear(self, tmp_path: Path) -> None:
        """An unjudgeable metric must not wear the healthy badge either."""
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _join_tree(tmp_path)

        row = _only(build_memory_evals(root, esc_dir)['evals'][0]['metrics'], 'insufficient-metric')

        assert row['verdict'] == 'insufficient_data'
        assert row['parity'] == 'insufficient_data'
        assert row['escalation'] is None

    def test_grandfathered_with_an_open_escalation_is_not_a_recovery(
        self, tmp_path: Path,
    ) -> None:
        """A standing exception with a live alarm is neither clear nor recovered."""
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _join_tree(tmp_path)

        row = _only(build_memory_evals(root, esc_dir)['evals'][0]['metrics'], 'grandfathered-open')

        assert row['verdict'] == 'grandfathered'
        assert row['parity'] == 'grandfathered_open'
        assert row['escalation']['id'] == 'esc-grandfathered-open'

    def test_grandfathered_without_a_link_is_not_clear(self, tmp_path: Path) -> None:
        """A known-bad measurement, deliberately exempted, is not a healthy one.

        ``grandfathered`` is the ratchet's standing exception: the metric IS
        failing and the program has chosen not to alarm on it yet.  Labelling
        it ``clear`` calls that exception healthy, which erases the one signal
        telling an operator the exception is still outstanding.
        """
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _join_tree(tmp_path)

        row = _only(build_memory_evals(root, esc_dir)['evals'][0]['metrics'], 'grandfathered-metric')

        assert row['verdict'] == 'grandfathered'
        assert row['parity'] == 'grandfathered'
        assert row['escalation'] is None

    def test_alarm_with_no_escalation_is_flagged_unlinked(self, tmp_path: Path) -> None:
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _join_tree(tmp_path)

        row = _only(build_memory_evals(root, esc_dir)['evals'][0]['metrics'], 'alarmed-unlinked')

        assert row['escalation'] is None
        assert row['parity'] == 'alarmed_unlinked'

    def test_quiet_and_unjudged_metrics_are_distinguishable(self, tmp_path: Path) -> None:
        """``clear`` means "judged, and not alarming" — nothing weaker.

        A metric NOTHING has judged is materially different from one the
        evaluator looked at and passed: the first says the evaluator never
        reached this metric, the second says it did and found no regression.
        Folding the two into one label makes an un-run judgement render with
        the healthy badge, which is the module's own stated failure mode
        (fail toward "visibly unrenderable", never toward the healthy label).
        """
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _join_tree(tmp_path)
        rows = build_memory_evals(root, esc_dir)['evals'][0]['metrics']

        assert _only(rows, 'clear-metric')['verdict'] == 'no_alarm'
        assert _only(rows, 'clear-metric')['parity'] == 'clear'
        assert _only(rows, 'clear-metric')['escalation'] is None
        # No verdict entry at all: not an alarm, but not "clear" either.
        assert _only(rows, 'unjudged-metric')['verdict'] is None
        assert _only(rows, 'unjudged-metric')['parity'] == 'unjudged'
        assert _only(rows, 'unjudged-metric')['escalation'] is None

    def test_fingerprinted_entry_with_no_verdict_is_unjudged_open(self, tmp_path: Path) -> None:
        """A link with no judgement behind it must not claim a recovery.

        An entry that keys and fingerprints fine but carries no ``verdict``
        field is the ONE reachable way a row can be both unjudged and linked:
        the join only runs when the verdict ENTRY carries a fingerprint, so a
        metric with no entry at all can never link.  Reporting this as
        ``recovered_open`` asserts a recovery that never happened — the
        evaluator did not judge this metric, so nothing recovered.
        """
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _unjudged_open_tree(tmp_path)

        row = _only(build_memory_evals(root, esc_dir)['evals'][0]['metrics'], 'unjudged-open')

        assert row['verdict'] is None
        assert row['parity'] == 'unjudged_open'
        # The link is still surfaced — the escalation is real and open, and
        # hiding it would make it look orphaned.
        assert row['escalation']['id'] == 'esc-unjudged-open'

    def test_matching_fingerprint_in_another_category_is_not_joined(self, tmp_path: Path) -> None:
        """Only ``eval_regression`` escalations participate in this join."""
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _join_tree(tmp_path)

        row = _only(build_memory_evals(root, esc_dir)['evals'][0]['metrics'], 'wrong-category')

        assert row['escalation'] is None
        assert row['parity'] == 'alarmed_unlinked'

    def test_escalation_claimed_by_no_verdict_is_surfaced(self, tmp_path: Path) -> None:
        """The reverse direction: every open eval_regression must be explained."""
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _join_tree(tmp_path)

        unmatched = build_memory_evals(root, esc_dir)['unmatched_escalations']

        assert [entry['id'] for entry in unmatched] == ['esc-unmatched']
        assert set(unmatched[0]) == _UNMATCHED_KEYS
        assert unmatched[0]['dedupe_fingerprint'] == _FP_UNMATCHED
        # No storm here, so this really is the "nothing claims it" case.
        assert unmatched[0]['reason'] == 'no_matching_verdict'

    def test_resolved_escalation_is_not_rendered_as_open(self, tmp_path: Path) -> None:
        """A CLOSED escalation must not produce an ``alarmed_open`` row.

        The join originally inferred openness from ``load_queue_escalations``
        not walking the archive subtree.  That inference does not hold:
        ``escalation.queue._archive_resolved`` is best-effort — on ``OSError``
        it logs a warning and leaves the resolved record in the queue root —
        and ``dashboard.data.escalations._bucket`` counts ``resolved`` among
        the very records this same reader returns.  A resolved alarm rendering
        as open is the parity view asserting the exact falsehood it exists to
        catch.
        """
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _join_tree(tmp_path)
        # Same record, same fingerprint, same place on disk — only closed.
        _write_escalation(
            esc_dir, 'esc-alarmed-open',
            dedupe_fingerprint=_FP_ALARMED_OPEN,
            summary='canonical-in-top-5 regressed', severity='blocking', level=1,
            timestamp=_JOIN_ESC_TIMESTAMP,
            status='resolved', resolved_at='2026-07-30T04:00:00+00:00',
        )

        payload = build_memory_evals(root, esc_dir)
        row = _only(payload['evals'][0]['metrics'], 'alarmed-open')

        assert row['escalation'] is None
        assert row['parity'] == 'alarmed_unlinked'
        # And the reverse direction: a closed escalation is not a still-open
        # orphan either, so it must not appear as unmatched.
        assert 'esc-alarmed-open' not in [e['id'] for e in payload['unmatched_escalations']]
        # Excluding a KNOWN terminal state is the filter working, not a
        # discard — so it is not reported as an issue.
        assert payload['issue_count'] == len(payload['issues']) == 0

    def test_dismissed_escalation_is_not_rendered_as_open(self, tmp_path: Path) -> None:
        """``dismissed`` is the other known terminal state, and is treated alike."""
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _join_tree(tmp_path)
        _write_escalation(
            esc_dir, 'esc-recovered-open',
            dedupe_fingerprint=_FP_RECOVERED_OPEN,
            summary='dangling-pointers regressed', severity='blocking', level=0,
            timestamp=_JOIN_ESC_TIMESTAMP,
            status='dismissed',
        )

        payload = build_memory_evals(root, esc_dir)
        row = _only(payload['evals'][0]['metrics'], 'recovered-open')

        assert row['escalation'] is None
        assert row['parity'] == 'clear'
        assert payload['issue_count'] == 0

    def test_escalation_with_no_status_field_still_joins(self, tmp_path: Path) -> None:
        """Absent ``status`` reads as pending, matching the model default.

        ``Escalation.status`` defaults to ``'pending'``, so a record that omits
        the field is an open one — the filter must not swallow it.
        """
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _join_tree(tmp_path)
        body = json.loads((esc_dir / 'esc-alarmed-open.json').read_text())
        del body['status']
        (esc_dir / 'esc-alarmed-open.json').write_text(json.dumps(body))

        payload = build_memory_evals(root, esc_dir)
        row = _only(payload['evals'][0]['metrics'], 'alarmed-open')

        assert row['parity'] == 'alarmed_open'
        assert row['escalation']['id'] == 'esc-alarmed-open'
        assert payload['issue_count'] == 0

    def test_unrecognised_status_is_skipped_and_named(self, tmp_path: Path) -> None:
        """The openness test is POSITIVE, and the resulting skip is a real discard.

        A status vocabulary that grows a new terminal state must not silently
        start rendering as open — hence the positive ``pending`` test rather
        than a blocklist of closed values.  But unlike ``resolved`` /
        ``dismissed``, an unrecognised value drops a record the reader cannot
        classify, so it is NAMED (the module's no-silent-discard invariant).
        """
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _join_tree(tmp_path)
        _write_escalation(
            esc_dir, 'esc-alarmed-open',
            dedupe_fingerprint=_FP_ALARMED_OPEN,
            summary='canonical-in-top-5 regressed', severity='blocking', level=1,
            timestamp=_JOIN_ESC_TIMESTAMP,
            status='quarantined',
        )

        payload = build_memory_evals(root, esc_dir)
        row = _only(payload['evals'][0]['metrics'], 'alarmed-open')

        # Not joined: unknown openness must not be rendered as open.
        assert row['escalation'] is None
        assert row['parity'] == 'alarmed_unlinked'

        assert payload['issue_count'] == len(payload['issues']) == 1
        issue = payload['issues'][0]
        assert issue['kind'] == 'unknown_escalation_status'
        # Both the id and the offending value, so an operator can find the
        # record without re-deriving which one was dropped.
        assert 'esc-alarmed-open' in issue['detail']
        assert 'quarantined' in issue['detail']

    def test_unhashable_status_is_named_not_raised(self, tmp_path: Path) -> None:
        """A non-string ``status`` is a classification failure, not a crash.

        ``status`` arrives from an unvalidated JSON artifact, so it can be any
        JSON type.  An unhashable one (``[]``, ``{}``) reaching the closed-set
        membership test raises ``TypeError`` — which escapes ``build_memory_evals``
        (documented "never raises") and 500s the whole section.  It takes the
        same named-discard path as an unrecognised string: a status this reader
        cannot classify must never render as open.
        """
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _join_tree(tmp_path)
        _write_escalation(
            esc_dir, 'esc-alarmed-open',
            dedupe_fingerprint=_FP_ALARMED_OPEN,
            summary='canonical-in-top-5 regressed', severity='blocking', level=1,
            timestamp=_JOIN_ESC_TIMESTAMP,
            # Deliberately the artifact shape the model forbids — the point of
            # the test is that the reader survives a producer that violates it.
            status=cast(Any, ['weird']),
        )

        payload = build_memory_evals(root, esc_dir)
        row = _only(payload['evals'][0]['metrics'], 'alarmed-open')

        assert row['escalation'] is None
        assert row['parity'] == 'alarmed_unlinked'
        assert payload['issue_count'] == len(payload['issues']) == 1
        issue = payload['issues'][0]
        assert issue['kind'] == 'unknown_escalation_status'
        assert 'esc-alarmed-open' in issue['detail']
        assert set(payload) == _PAYLOAD_KEYS

    def test_open_escalation_with_no_fingerprint_is_named_and_surfaced(self, tmp_path: Path) -> None:
        """The fingerprint filter is the last place an OPEN alarm can vanish.

        An absent or non-string ``dedupe_fingerprint`` cannot key the index —
        and ``unmatched_escalations`` is derived FROM that index, so a bare
        skip erases the escalation from the metric rows and from the
        reverse-direction list in one move.  A live alarm that appears NOWHERE
        in the payload is the precise blind spot the parity view exists to
        close, and the module's own contract forbids discarding a parsed record
        without naming it.  Unlike ``resolved``/``dismissed`` (the filter doing
        its job) this record is genuinely open, so it is both named in
        ``issues`` and carried into ``unmatched_escalations`` with the reason
        it carries no link.
        """
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _join_tree(tmp_path)
        _write_escalation(
            esc_dir, 'esc-no-fingerprint',
            dedupe_fingerprint=None,
            summary='an open alarm the producer never fingerprinted',
            timestamp=_JOIN_ESC_TIMESTAMP,
        )
        _write_escalation(
            esc_dir, 'esc-bad-fingerprint',
            # Unvalidated JSON: the field can be any type, not merely absent.
            dedupe_fingerprint=cast(Any, {'eval': 'eval-a'}),
            summary='an open alarm whose fingerprint is not a string',
            timestamp=_JOIN_ESC_TIMESTAMP,
        )

        payload = build_memory_evals(root, esc_dir)

        named = [i for i in payload['issues'] if i['kind'] == 'unfingerprinted_escalation']
        assert len(named) == 2
        details = ' '.join(i['detail'] for i in named)
        assert 'esc-no-fingerprint' in details
        assert 'esc-bad-fingerprint' in details
        assert payload['issue_count'] == len(payload['issues'])

        # Named in `issues` is not enough: the open alarm itself must still be
        # visible where an operator looks for unexplained escalations.
        unmatched = {entry['id']: entry for entry in payload['unmatched_escalations']}
        assert unmatched['esc-no-fingerprint']['reason'] == 'no_fingerprint'
        assert unmatched['esc-bad-fingerprint']['reason'] == 'no_fingerprint'
        assert set(unmatched['esc-no-fingerprint']) == _UNMATCHED_KEYS
        # The joinable records are untouched by the new path.
        assert _only(payload['evals'][0]['metrics'], 'alarmed-open')['escalation'] is not None
        assert unmatched['esc-unmatched']['reason'] == 'no_matching_verdict'


# ---------------------------------------------------------------------------
# a verdict value outside the closed M2 vocabulary
# ---------------------------------------------------------------------------

_FP_UNKNOWN_VERDICT = '9876543210fedcba9876543210fedcba'

# The parity vocabulary in full — the closed set the UI switches on, mirroring
# the `_VERDICT_VALUES` contract test one level down.  Deriving parity
# server-side only buys the frontend anything if the output set is CLOSED and
# knowable; an open-ended one would force every consumer to re-derive from
# `verdict`, which is the exact drift this field exists to prevent.
_PARITY_VALUES = (
    'alarmed_open',
    'alarmed_unlinked',
    'recovered_open',
    'clear',
    'insufficient_data',
    'insufficient_data_open',
    'grandfathered',
    'grandfathered_open',
    'unjudged',
    'unjudged_open',
    'unknown_verdict',
    'unknown_verdict_open',
    'storm_collapsed',
)


def _unknown_verdict_tree(tmp_path: Path, verdict: Any, *, linked: bool) -> tuple[Path, Path]:
    """One metric whose verdict value is outside the M2 vocabulary.

    Built standalone rather than folded into ``_join_tree``: that tree is
    asserted to be issue-free by several tests, and this one exists precisely
    to raise an issue.  The entry is written as a literal dict because
    ``_verdict()`` takes a ``str`` and one of these cases is not a string at
    all — the artifact is unvalidated JSON, so that shape is reachable.
    """
    root = tmp_path / 'memory-evals'
    esc_dir = tmp_path / 'escalations'
    esc_dir.mkdir(parents=True, exist_ok=True)
    _write_metrics(root, 'eval-a', _JOIN_RUN, [
        _metric('drifted-metric', 'count', 3.0, direction='higher_is_worse'),
    ])
    _write_limits(root, 'eval-a', run_stamp=_JOIN_RUN)
    _write_verdicts(root, [{
        'eval_id': 'eval-a',
        'metric_id': 'drifted-metric',
        'verdict': verdict,
        'fingerprint': _FP_UNKNOWN_VERDICT,
        'run_stamp': _JOIN_RUN,
    }], run_stamp=_JOIN_RUN)
    if linked:
        _write_escalation(
            esc_dir, 'esc-drifted',
            dedupe_fingerprint=_FP_UNKNOWN_VERDICT,
            summary='drifted-metric regressed', timestamp=_JOIN_ESC_TIMESTAMP,
        )
    return root, esc_dir


class TestUnknownVerdict:
    """A verdict this reader cannot render is NAMED, never rendered healthy.

    The same shape the module already applies twice: ``unknown_kind`` for a
    metric kind with no chart primitive, and ``unknown_escalation_status`` for
    a status it cannot classify.  Both are justified by one rule — an
    unrecognised value must fail toward "visibly unrenderable", never toward
    the healthy label.  A verdict outside the closed M2 set is the third
    instance, and it used to fail the other way: straight to ``clear``.

    The drift is realistic rather than hypothetical.  The LIMITS artifact
    carries its OWN verdict vocabulary (``baseline_snapshot|ok|alarm|improved|
    insufficient_data``), so a producer-side mix-up lands ``improved`` in this
    field — and ``improved`` reading as ``clear`` is very nearly plausible,
    which is what makes the silent version of this dangerous.
    """

    def test_unknown_verdict_is_named_not_rendered_clear(self, tmp_path: Path) -> None:
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _unknown_verdict_tree(tmp_path, 'improved', linked=False)

        payload = build_memory_evals(root, esc_dir)
        row = _only(payload['evals'][0]['metrics'], 'drifted-metric')

        assert row['parity'] == 'unknown_verdict'
        # The reader NAMES the value it cannot render; it never rewrites it.
        assert row['verdict'] == 'improved'

        named = [i for i in payload['issues'] if i['kind'] == 'unknown_verdict']
        assert len(named) == 1
        assert named[0]['eval_id'] == 'eval-a'
        assert 'drifted-metric' in named[0]['detail']
        assert "'improved'" in named[0]['detail']
        # The ROOT verdicts artifact is where the offending value lives.
        assert named[0]['path'] == str(root / 'verdicts-current.json')

    def test_unknown_verdict_with_an_open_escalation_keeps_the_link(
        self, tmp_path: Path,
    ) -> None:
        """Unrenderable is not unlinked: the escalation is real and still open."""
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _unknown_verdict_tree(tmp_path, 'improved', linked=True)

        payload = build_memory_evals(root, esc_dir)
        row = _only(payload['evals'][0]['metrics'], 'drifted-metric')

        assert row['parity'] == 'unknown_verdict_open'
        assert row['verdict'] == 'improved'
        assert row['escalation']['id'] == 'esc-drifted'
        assert len([i for i in payload['issues'] if i['kind'] == 'unknown_verdict']) == 1

    def test_non_string_verdict_is_named_not_raised(self, tmp_path: Path) -> None:
        """An unhashable verdict must not raise on the vocabulary lookup.

        Mirrors ``test_unhashable_status_is_named_not_raised``: the artifact is
        unvalidated JSON, so ``verdict`` can be any type, and a bare
        ``in _KNOWN_VERDICTS`` on a dict would raise ``TypeError``.
        """
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _unknown_verdict_tree(tmp_path, {}, linked=False)

        payload = build_memory_evals(root, esc_dir)
        row = _only(payload['evals'][0]['metrics'], 'drifted-metric')

        assert row['parity'] == 'unknown_verdict'
        # Passed through verbatim, exactly as a recognised value would be.
        assert row['verdict'] == {}
        named = [i for i in payload['issues'] if i['kind'] == 'unknown_verdict']
        assert len(named) == 1
        assert '{}' in named[0]['detail']

    def test_absent_verdict_earns_no_unknown_issue(self, tmp_path: Path) -> None:
        """"Nothing judged this" is a legitimate state, not a discard.

        The payload already models it — ``unjudged`` parity, plus
        ``missing_verdicts``/``orphan_verdict`` at the artifact level — so
        naming it again here would put a standing issue on a healthy tree and
        train operators past the list.
        """
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _join_tree(tmp_path)

        payload = build_memory_evals(root, esc_dir)

        assert _only(payload['evals'][0]['metrics'], 'unjudged-metric')['parity'] == 'unjudged'
        assert [i for i in payload['issues'] if i['kind'] == 'unknown_verdict'] == []


class TestParityVocabularyIsClosedAndExported:
    """``PARITY_STATES`` is the single source of truth for the badge vocabulary.

    Deriving parity server-side only buys the frontend anything if the output
    set is CLOSED and knowable — otherwise every consumer has to re-derive
    badge state from ``verdict``, which is the drift this field exists to
    prevent.  Exporting the set gives the two known consumers (the dashboard
    tab, task 3216; the parity gate, task 3217) something to assert against the
    PRODUCER instead of hardcoding a subset in their own test files, where it
    rots silently the moment a state is added.
    """

    def test_the_module_exports_the_vocabulary(self) -> None:
        from dashboard.data import memory_evals

        assert set(_PARITY_VALUES) == memory_evals.PARITY_STATES

    def test_the_literal_and_the_builder_agree(self, tmp_path: Path) -> None:
        """Every state the builder emits is a member, and every member is reachable.

        Both directions matter and neither implies the other.  A state escaping
        the set means a consumer switching on ``PARITY_STATES`` silently drops
        a row; a member no fixture can produce is either dead vocabulary or an
        untested state, and the literal set cannot tell those apart on its own.
        This is what keeps the hand-written literal honest.
        """
        from dashboard.data.memory_evals import build_memory_evals

        trees = [
            _join_tree(tmp_path / 'join'),
            _storm_tree(
                tmp_path / 'storm',
                storm={'triggered': True, 'alarm_count': 2, 'aggregate_fingerprint': _FP_AGGREGATE},
            ),
            _unjudged_open_tree(tmp_path / 'unjudged-open'),
            _unknown_verdict_tree(tmp_path / 'unknown', 'improved', linked=False),
            _unknown_verdict_tree(tmp_path / 'unknown-open', 'improved', linked=True),
        ]

        observed = {
            metric['parity']
            for root, esc_dir in trees
            for row in build_memory_evals(root, esc_dir)['evals']
            for metric in row['metrics']
        }

        from dashboard.data import memory_evals

        assert observed - memory_evals.PARITY_STATES == set(), 'a parity escaped the exported set'
        assert memory_evals.PARITY_STATES - observed == set(), 'an exported parity is unreachable'


# ---------------------------------------------------------------------------
# step-13 — storm collapse, modelled (gamma's gate owns the full matrix)
# ---------------------------------------------------------------------------

_STORM_RUN = '20260706T031500Z'
_FP_AGGREGATE = '77665544332211009988aabbccddeeff'
_FP_STORM_A = '11223344556677889900112233445566'
_FP_STORM_B = '66554433221100998877665544332211'

_STORM_KEYS = {'triggered', 'alarm_count', 'aggregate_fingerprint', 'escalation'}


def _storm_tree(
    tmp_path: Path,
    *,
    storm: dict | None,
    aggregate_escalation: bool = True,
    per_metric_escalation: bool = False,
) -> tuple[Path, Path]:
    """Two alarming metrics and one quiet one, under an optional storm block."""
    root = tmp_path / 'memory-evals'
    esc_dir = tmp_path / 'escalations'
    esc_dir.mkdir(parents=True, exist_ok=True)

    _write_metrics(root, 'eval-a', _STORM_RUN, [
        _metric('storm-a', 'count', 9.0, direction='higher_is_worse'),
        _metric('storm-b', 'count', 8.0, direction='higher_is_worse'),
        _metric('quiet-metric', 'scalar', 44.0),
    ])
    _write_limits(root, 'eval-a', run_stamp=_STORM_RUN)
    _write_verdicts(
        root,
        [
            _verdict('eval-a', 'storm-a', 'alarm', fingerprint=_FP_STORM_A, run_stamp=_STORM_RUN),
            _verdict('eval-a', 'storm-b', 'alarm', fingerprint=_FP_STORM_B, run_stamp=_STORM_RUN),
            _verdict('eval-a', 'quiet-metric', 'no_alarm', fingerprint='quietfingerprint0000000000000000', run_stamp=_STORM_RUN),
        ],
        storm_escape=storm,
        run_stamp=_STORM_RUN,
    )

    if aggregate_escalation:
        _write_escalation(
            esc_dir, 'esc-storm-aggregate',
            dedupe_fingerprint=_FP_AGGREGATE,
            summary='memory-eval storm: 2 metrics alarmed in one run',
            severity='blocking', level=1,
            timestamp=_JOIN_ESC_TIMESTAMP,
        )
    if per_metric_escalation:
        _write_escalation(
            esc_dir, 'esc-storm-a',
            dedupe_fingerprint=_FP_STORM_A,
            summary='storm-a regressed', severity='blocking', level=1,
            timestamp=_JOIN_ESC_TIMESTAMP,
        )
    return root, esc_dir


class TestStormCollapseModelled:
    """The storm-escape state is MODELLED here; gamma's gate owns the matrix.

    Scope note: alpha proves the payload can represent a storm honestly — the
    aggregate resolves, per-metric links collapse, and the aggregate is not
    double-counted as unexplained.  The full both-directions parity matrix and
    the re-verification against the committed verdicts exemplar are gamma's
    gate and are deliberately not duplicated here.
    """

    def test_storm_block_is_surfaced_and_its_aggregate_resolved(self, tmp_path: Path) -> None:
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _storm_tree(tmp_path, storm={
            'triggered': True, 'alarm_count': 2, 'aggregate_fingerprint': _FP_AGGREGATE,
        })

        payload = build_memory_evals(root, esc_dir)
        storm = payload['evals'][0]['storm_escape']

        assert set(storm) == _STORM_KEYS
        assert storm['triggered'] is True
        assert storm['alarm_count'] == 2
        assert storm['aggregate_fingerprint'] == _FP_AGGREGATE
        assert storm['escalation']['id'] == 'esc-storm-aggregate'
        assert set(storm['escalation']) == _ESCALATION_KEYS
        # The block is run-scoped across the whole PROGRAM, so the same
        # resolved block is the payload's own — the per-eval copy explains one
        # row's collapsed link, the top-level one is the banner source.
        assert payload['storm_escape'] == storm

    def test_the_program_wide_block_survives_a_root_with_no_eval_dirs(self, tmp_path: Path) -> None:
        """The banner (and its aggregate link) must not depend on an eval row existing.

        A verdicts artifact can name a triggered storm while the root
        enumerates zero eval dirs — nothing has been written yet, or every dir
        failed to enumerate.  Nested only inside the eval rows, the whole block
        and its resolved aggregate escalation would vanish while the aggregate
        escalation itself stayed open.
        """
        from dashboard.data.memory_evals import build_memory_evals

        root = tmp_path / 'memory-evals'
        esc_dir = tmp_path / 'escalations'
        esc_dir.mkdir(parents=True, exist_ok=True)
        _write_verdicts(
            root, [],
            storm_escape={'triggered': True, 'alarm_count': 2, 'aggregate_fingerprint': _FP_AGGREGATE},
            run_stamp=_STORM_RUN,
        )
        _write_escalation(
            esc_dir, 'esc-storm-aggregate', dedupe_fingerprint=_FP_AGGREGATE,
            summary='memory-eval storm: 2 metrics alarmed in one run',
            timestamp=_JOIN_ESC_TIMESTAMP,
        )

        payload = build_memory_evals(root, esc_dir)

        assert payload['evals'] == []
        assert payload['storm_escape']['triggered'] is True
        assert payload['storm_escape']['escalation']['id'] == 'esc-storm-aggregate'
        # Resolved by the block, so it is not ALSO reported as unexplained.
        assert payload['unmatched_escalations'] == []

    def test_no_storm_leaves_the_top_level_block_absent(self, tmp_path: Path) -> None:
        """An untriggered (or absent) storm renders no banner at all."""
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _storm_tree(tmp_path, storm=None, aggregate_escalation=False)

        assert build_memory_evals(root, esc_dir)['storm_escape'] is None

    def test_per_metric_links_collapse_into_the_aggregate(self, tmp_path: Path) -> None:
        """No per-metric links during a storm — the aggregate is the one alert."""
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _storm_tree(
            tmp_path,
            storm={'triggered': True, 'alarm_count': 2, 'aggregate_fingerprint': _FP_AGGREGATE},
            per_metric_escalation=True,
        )

        payload = build_memory_evals(root, esc_dir)
        rows = payload['evals'][0]['metrics']

        for metric_id in ('storm-a', 'storm-b'):
            row = _only(rows, metric_id)
            assert row['verdict'] == 'alarm'
            assert row['escalation'] is None
            assert row['parity'] == 'storm_collapsed'
        # A quiet metric is still quiet during a storm.
        assert _only(rows, 'quiet-metric')['parity'] == 'clear'

        # Suppressed is not unexplained.  The per-metric escalation is still
        # OPEN (per-metric filings from earlier runs stay pending across a
        # storm) and it IS reported — but as storm-suppressed, so the UI's
        # "no metric row explains this" signal does not fire on an escalation
        # a row explains perfectly well.  Reporting it undifferentiated would
        # train operators to ignore the one signal that catches a real orphan.
        (unmatched,) = payload['unmatched_escalations']
        assert unmatched['id'] == 'esc-storm-a'
        assert unmatched['reason'] == 'storm_suppressed'
        assert set(unmatched) == _UNMATCHED_KEYS

    def test_aggregate_is_not_also_reported_unexplained(self, tmp_path: Path) -> None:
        """It IS explained — by the storm block, not by a metric row."""
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _storm_tree(tmp_path, storm={
            'triggered': True, 'alarm_count': 2, 'aggregate_fingerprint': _FP_AGGREGATE,
        })

        payload = build_memory_evals(root, esc_dir)

        assert [entry['id'] for entry in payload['unmatched_escalations']] == []

    def test_untriggered_storm_leaves_normal_parity(self, tmp_path: Path) -> None:
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _storm_tree(
            tmp_path,
            storm={'triggered': False, 'alarm_count': 0, 'aggregate_fingerprint': None},
            aggregate_escalation=False,
            per_metric_escalation=True,
        )

        payload = build_memory_evals(root, esc_dir)

        assert payload['evals'][0]['storm_escape'] is None
        rows = payload['evals'][0]['metrics']
        assert _only(rows, 'storm-a')['parity'] == 'alarmed_open'
        assert _only(rows, 'storm-a')['escalation']['id'] == 'esc-storm-a'
        assert _only(rows, 'storm-b')['parity'] == 'alarmed_unlinked'

    def test_absent_storm_block_is_none(self, tmp_path: Path) -> None:
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _storm_tree(tmp_path, storm=None, aggregate_escalation=False)

        payload = build_memory_evals(root, esc_dir)

        assert payload['evals'][0]['storm_escape'] is None
        assert _only(payload['evals'][0]['metrics'], 'storm-a')['parity'] == 'alarmed_unlinked'


# ---------------------------------------------------------------------------
# step-15 — staleness (displayed, never alarmed on) and every degraded state
# ---------------------------------------------------------------------------

_AGE_RUN = '20260705T031500Z'
_AGE_RUN_AT = datetime(2026, 7, 5, 3, 15, 0, tzinfo=UTC)


def _healthy_tree(tmp_path: Path, *, metrics: list[dict] | None = None, **kwargs: Any) -> tuple[Path, Path]:
    """One eval, one run, limits AND verdicts present — zero issues by default.

    Every degraded-state test starts from this and breaks exactly one thing, so
    ``issue_count == 1`` is an assertion about the break rather than about the
    surrounding fixture.
    """
    root = tmp_path / 'memory-evals'
    esc_dir = tmp_path / 'escalations'
    esc_dir.mkdir(parents=True, exist_ok=True)
    rows = metrics if metrics is not None else [_metric('dangling-pointers', 'count', 4.0)]
    _write_metrics(root, 'eval-a', _AGE_RUN, rows, **kwargs)
    _write_limits(root, 'eval-a', run_stamp=_AGE_RUN)
    _write_verdicts(root, [], run_stamp=_AGE_RUN)
    return root, esc_dir


def _corrupt(path: Path) -> Path:
    """Make *path* syntactically unparseable (it is still a file that exists)."""
    path.write_text('{"metrics": [ this is not json')
    return path


def _raiser(exc: Exception) -> Any:
    """A stand-in for a module-level reader that raises *exc* however it is called."""
    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise exc
    return _boom


class TestCodeBugsAreNotArtifactDegradation:
    """A bug in this module must not be reported to the operator as a bad file.

    The narrow tuple and the outermost guard are two halves of one contract.
    ``_ARTIFACT_ERRORS`` says what READING AND PARSING one artifact off disk
    can raise — nothing more — so a ``TypeError`` or ``AttributeError`` out of
    a typo in ``_read_limits`` can no longer masquerade as ``unreadable_limits``
    and send the operator to inspect a file that is perfectly fine.

    The module's never-raises contract is unchanged and still load-bearing: a
    degraded tree yields a degraded payload, never a 500.  It is now delivered
    by ONE named boundary rather than by four broad per-artifact catches, and
    a bug that reaches it is reported AS a bug (``internal_error``) with a real
    traceback in the dashboard log, instead of being silently relabelled.
    """

    def test_artifact_errors_is_narrowed_to_io_and_parse(self) -> None:
        """Pinned structurally so the tuple cannot silently widen back.

        ``json.JSONDecodeError`` is dropped as a redundant ``ValueError``
        subclass; ``TypeError``/``AttributeError`` are dropped because they are
        what a code bug raises, not what a bad file raises.
        """
        from dashboard.data import memory_evals

        assert (OSError, ValueError) == memory_evals._ARTIFACT_ERRORS

    def test_a_bug_in_read_limits_is_an_internal_error(self, tmp_path: Path, monkeypatch) -> None:
        from dashboard.data import memory_evals as memory_evals_mod
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _healthy_tree(tmp_path)
        monkeypatch.setattr(
            memory_evals_mod, '_read_limits', _raiser(AttributeError("no attribute 'alpah'")),
        )

        payload = build_memory_evals(root, esc_dir)

        # The never-500 contract still holds.
        assert set(payload) == _PAYLOAD_KEYS
        assert payload['issue_count'] == len(payload['issues']) == 1
        # Named as what it is — NOT as an unreadable limits artifact.
        assert payload['issues'][0]['kind'] == 'internal_error'
        # The exception type is in the detail so an operator can tell a
        # dashboard bug from a broken artifact without reading the log.
        assert 'AttributeError' in payload['issues'][0]['detail']

    def test_a_bug_in_read_verdicts_is_an_internal_error(self, tmp_path: Path, monkeypatch) -> None:
        from dashboard.data import memory_evals as memory_evals_mod
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _healthy_tree(tmp_path)
        monkeypatch.setattr(
            memory_evals_mod, '_read_verdicts', _raiser(TypeError('unhashable type: dict')),
        )

        payload = build_memory_evals(root, esc_dir)

        assert set(payload) == _PAYLOAD_KEYS
        assert payload['issue_count'] == len(payload['issues']) == 1
        assert payload['issues'][0]['kind'] == 'internal_error'
        assert 'TypeError' in payload['issues'][0]['detail']

    def test_a_bug_in_index_escalations_is_an_internal_error(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        from dashboard.data import memory_evals as memory_evals_mod
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _healthy_tree(tmp_path)
        monkeypatch.setattr(
            memory_evals_mod, '_index_escalations',
            _raiser(AttributeError("'list' object has no attribute 'get'")),
        )

        payload = build_memory_evals(root, esc_dir)

        assert set(payload) == _PAYLOAD_KEYS
        assert payload['issue_count'] == len(payload['issues']) == 1
        assert payload['issues'][0]['kind'] == 'internal_error'
        assert 'AttributeError' in payload['issues'][0]['detail']

    def test_the_traceback_reaches_the_log(self, tmp_path: Path, monkeypatch, caplog) -> None:
        """A swallowed code bug must not be invisible.

        The payload names the bug for the operator; the log carries the
        traceback for whoever has to fix it.  Without the second half, keeping
        the never-500 contract would mean a bug that silently degrades the
        dashboard forever with nothing to debug from.
        """
        from dashboard.data import memory_evals as memory_evals_mod
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _healthy_tree(tmp_path)
        monkeypatch.setattr(
            memory_evals_mod, '_read_verdicts', _raiser(TypeError('unhashable type: dict')),
        )

        with caplog.at_level(logging.ERROR, logger='dashboard.data.memory_evals'):
            build_memory_evals(root, esc_dir)

        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(errors) == 1
        assert errors[0].exc_info is not None

    def test_io_failures_keep_their_granular_kinds(self, tmp_path: Path, monkeypatch) -> None:
        """The outer guard must not swallow what the narrow catches still handle.

        A parse failure is a ``ValueError`` and a permission failure is an
        ``OSError``, so both stay inside ``_ARTIFACT_ERRORS`` and keep degrading
        one artifact at a time.  If any of these started reading
        ``internal_error``, the narrowing would have gone too far and the
        operator would lose the pointer to the actual broken file.
        """
        from dashboard.data import memory_evals as memory_evals_mod
        from dashboard.data.memory_evals import build_memory_evals

        # Every tree is built BEFORE any monkeypatch, so the patched Path
        # method can never affect the fixture that exercises it.
        root, esc_dir = _healthy_tree(tmp_path)
        root2, esc_dir2 = _healthy_tree(tmp_path / 'second')
        root3, esc_dir3 = _healthy_tree(tmp_path / 'third')
        _corrupt(root / 'eval-a' / 'limits-current.json')
        _corrupt(root / 'verdicts-current.json')

        # Two ValueError-induced boundaries (a corrupt file is a parse error).
        assert {i['kind'] for i in build_memory_evals(root, esc_dir)['issues']} == {
            'unreadable_limits', 'unreadable_verdicts',
        }

        # And the two OSError-induced boundaries, one at a time.
        monkeypatch.setattr(
            memory_evals_mod, 'load_queue_escalations',
            _raiser(PermissionError(13, 'Permission denied')),
        )
        assert [i['kind'] for i in build_memory_evals(root2, esc_dir2)['issues']] == [
            'unreadable_escalations',
        ]

        monkeypatch.setattr(Path, 'iterdir', _raiser(PermissionError(13, 'Permission denied')))
        assert [i['kind'] for i in build_memory_evals(root3, esc_dir3)['issues']] == [
            'unreadable_root',
        ]


class TestStalenessAndDegradedStates:
    """A degraded tree yields a degraded payload — never an exception, never a lie.

    Staleness here is DISPLAYED, never alarmed on: the runner-failure tripwire
    belongs to the eval program, and this module files nothing.
    """

    def test_age_is_measured_against_the_injected_now(self, tmp_path: Path) -> None:
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _healthy_tree(tmp_path)

        eval_a = build_memory_evals(root, esc_dir, now=_AGE_RUN_AT + timedelta(hours=1))['evals'][0]

        assert eval_a['latest_run_stamp'] == _AGE_RUN
        assert eval_a['latest_run_age_seconds'] == 3600.0

    def test_stale_flips_at_the_threshold(self, tmp_path: Path, monkeypatch) -> None:
        import dashboard.data.memory_evals as memory_evals_module
        from dashboard.data.memory_evals import build_memory_evals

        monkeypatch.setattr(memory_evals_module, '_STALE_AFTER_SECONDS', 3600.0)
        root, esc_dir = _healthy_tree(tmp_path)

        fresh = build_memory_evals(root, esc_dir, now=_AGE_RUN_AT + timedelta(seconds=3599))
        stale = build_memory_evals(root, esc_dir, now=_AGE_RUN_AT + timedelta(seconds=3601))

        assert fresh['evals'][0]['stale'] is False
        assert stale['evals'][0]['stale'] is True

    def test_unreadable_metrics_run_is_named(self, tmp_path: Path) -> None:
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _healthy_tree(tmp_path)
        bad = _corrupt(root / 'eval-a' / 'metrics-20260706T031500Z.json')

        payload = build_memory_evals(root, esc_dir)

        assert payload['issue_count'] == len(payload['issues']) == 1
        assert payload['issues'][0]['kind'] == 'unreadable_metrics'
        assert payload['issues'][0]['path'] == str(bad)
        assert payload['issues'][0]['eval_id'] == 'eval-a'
        # The readable run still renders.
        assert payload['evals'][0]['run_stamps'] == [_AGE_RUN]

    def test_unreadable_limits_is_named(self, tmp_path: Path) -> None:
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _healthy_tree(tmp_path)
        bad = _corrupt(root / 'eval-a' / 'limits-current.json')

        payload = build_memory_evals(root, esc_dir)

        assert payload['issue_count'] == len(payload['issues']) == 1
        assert payload['issues'][0]['kind'] == 'unreadable_limits'
        assert payload['issues'][0]['path'] == str(bad)
        assert payload['evals'][0]['limits'] is None

    def test_unreadable_verdicts_never_defaults_the_column(self, tmp_path: Path) -> None:
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _healthy_tree(tmp_path)
        bad = _corrupt(root / 'verdicts-current.json')

        payload = build_memory_evals(root, esc_dir)

        assert payload['issue_count'] == len(payload['issues']) == 1
        assert payload['issues'][0]['kind'] == 'unreadable_verdicts'
        assert payload['issues'][0]['path'] == str(bad)
        # An unreadable verdict artifact must not read as "nothing alarmed".
        assert all(row['verdict'] is None for row in payload['evals'][0]['metrics'])

    def test_top_level_list_is_malformed_metrics(self, tmp_path: Path) -> None:
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _healthy_tree(tmp_path)
        bad = _dump(root / 'eval-a' / 'metrics-20260706T031500Z.json', [{'metric_id': 'x'}])

        payload = build_memory_evals(root, esc_dir)

        assert payload['issue_count'] == len(payload['issues']) == 1
        assert payload['issues'][0]['kind'] == 'malformed_metrics'
        assert payload['issues'][0]['path'] == str(bad)

    def test_non_list_metrics_field_is_malformed_metrics(self, tmp_path: Path) -> None:
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _healthy_tree(tmp_path)
        bad = _dump(
            root / 'eval-a' / 'metrics-20260706T031500Z.json',
            {'schema_version': 1, 'run_stamp': '20260706T031500Z', 'metrics': {'oops': 'a dict'}},
        )

        payload = build_memory_evals(root, esc_dir)

        assert payload['issue_count'] == len(payload['issues']) == 1
        assert payload['issues'][0]['kind'] == 'malformed_metrics'
        assert payload['issues'][0]['path'] == str(bad)

    def test_wrong_type_verdicts_is_malformed_verdicts(self, tmp_path: Path) -> None:
        """A verdicts artifact that PARSES but is not an object must still be named.

        The third disposal path, and the one the reviewer found missing: absent
        is ``missing_verdicts``, unparseable is ``unreadable_verdicts``, and
        valid-JSON-but-wrong-type was silently discarded — which made a broken
        verdicts file structurally identical to a healthy no-alarm tree while a
        real alarm sat on disk.
        """
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _healthy_tree(tmp_path)
        # Captured BEFORE the corruption: the healthy tree this payload must
        # remain distinguishable from.
        healthy = build_memory_evals(root, esc_dir)

        bad = _dump(
            root / 'verdicts-current.json',
            [_verdict('eval-a', 'dangling-pointers', 'alarm', fingerprint='opaque-fp-1')],
        )

        payload = build_memory_evals(root, esc_dir)

        assert payload['issue_count'] == len(payload['issues'])
        malformed = [i for i in payload['issues'] if i['kind'] == 'malformed_verdicts']
        assert len(malformed) == 1
        assert malformed[0]['path'] == str(bad)
        # The received type is named, so an operator reading the issue knows
        # what shape landed rather than only that something was wrong.
        assert 'list' in malformed[0]['detail']

        # Absent, never defaulted to 'no_alarm' — the promise the comment at the
        # unreadable-verdicts handler already made.
        assert all(row['verdict'] is None for row in payload['evals'][0]['metrics'])

        # The exact confusion the reviewer reproduced: a broken verdicts file
        # must not read as "nothing alarmed".
        assert healthy['issue_count'] == 0
        assert payload['issue_count'] > healthy['issue_count']

    def test_wrong_type_entries_is_malformed_verdicts(self, tmp_path: Path) -> None:
        """The body is an OBJECT but ``entries`` is not a list — still a discard.

        The narrower sibling of the wrong-type-body case, and the one the
        original guard missed: iterating a dict-valued ``entries`` walks its
        KEYS, fails every per-entry ``isinstance`` test, and yields an empty
        index with no signal — byte-identical to a healthy no-alarm tree while
        a real alarm sits on disk.  The guard's own detail string already
        claimed to be checking for "an object with an 'entries' list", so the
        intent was there but the check was not.
        """
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _healthy_tree(tmp_path)
        healthy = build_memory_evals(root, esc_dir)

        bad = _dump(root / 'verdicts-current.json', {
            'schema_version': 1,
            # A plausible producer bug: entries keyed by eval rather than listed.
            'entries': {'eval-a': {'dangling-pointers': 'alarm'}},
        })

        payload = build_memory_evals(root, esc_dir)

        malformed = [i for i in payload['issues'] if i['kind'] == 'malformed_verdicts']
        assert len(malformed) == 1
        assert malformed[0]['path'] == str(bad)
        assert 'dict' in malformed[0]['detail']

        assert all(row['verdict'] is None for row in payload['evals'][0]['metrics'])
        # The whole point: distinguishable from the healthy tree.
        assert healthy['issue_count'] == 0
        assert payload['issue_count'] == len(payload['issues']) > 0

    def test_absent_entries_field_is_malformed_verdicts(self, tmp_path: Path) -> None:
        """``entries`` missing entirely has the same standing as a wrong type."""
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _healthy_tree(tmp_path)
        _dump(root / 'verdicts-current.json', {'schema_version': 1})

        payload = build_memory_evals(root, esc_dir)

        malformed = [i for i in payload['issues'] if i['kind'] == 'malformed_verdicts']
        assert len(malformed) == 1
        # Named as absent rather than as a type, so the operator is not sent
        # looking for a field that was never written.
        assert 'no such field' in malformed[0]['detail']

    def test_non_object_entry_is_named_not_dropped(self, tmp_path: Path) -> None:
        """One unusable element must not vanish while its siblings are read."""
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _healthy_tree(tmp_path)
        # Written directly rather than through `_write_verdicts`: the whole
        # point is an element the typed helper would not let a producer emit.
        _dump(root / 'verdicts-current.json', {
            'schema_version': 1,
            'run_stamp': _AGE_RUN,
            'entries': [
                'not-an-entry',
                _verdict('eval-a', 'dangling-pointers', 'alarm', fingerprint='opaque-fp-1', run_stamp=_AGE_RUN),
            ],
        })

        payload = build_memory_evals(root, esc_dir)

        # The good sibling is still read — the bad element is skipped, not the file.
        assert _only(payload['evals'][0]['metrics'], 'dangling-pointers')['verdict'] == 'alarm'

        assert payload['issue_count'] == len(payload['issues']) == 1
        assert payload['issues'][0]['kind'] == 'malformed_verdict_entry'
        assert 'str' in payload['issues'][0]['detail']

    def test_unkeyable_verdict_entries_are_counted_not_dropped(self, tmp_path: Path) -> None:
        """An entry that IS an object but carries no usable key is still a discard.

        ``malformed_verdict_entry`` covers a non-object element.  An element
        that IS an object but whose ``eval_id``/``metric_id`` is not a string
        can never be keyed either, and was dropped by a bare ``continue`` — the
        record-level form of the same silent discard the artifact-level guards
        above exist to prevent.  Milder in blast radius (one unjudged row
        rather than a wholly unjudged tree) but the same class: the row renders
        ``verdict: None``, which is indistinguishable from "no entry was ever
        written for it".

        Counted once per file rather than one issue apiece, mirroring
        ``unidentified_metrics`` — a systematically broken artifact should
        degrade one row of the issues list, not flood it.
        """
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _healthy_tree(tmp_path)
        good = _verdict('eval-a', 'dangling-pointers', 'alarm', fingerprint='opaque-fp-1')
        _write_verdicts(root, [
            good,
            {'eval_id': 'eval-a', 'metric_id': None, 'verdict': 'alarm', 'fingerprint': 'fp-x'},
            {'eval_id': 42, 'metric_id': 'dangling-pointers', 'verdict': 'alarm', 'fingerprint': 'fp-y'},
        ])

        payload = build_memory_evals(root, esc_dir)

        # The keyable sibling is still read — bad records cost only themselves.
        assert _only(payload['evals'][0]['metrics'], 'dangling-pointers')['verdict'] == 'alarm'

        assert payload['issue_count'] == len(payload['issues'])
        unidentified = [i for i in payload['issues'] if i['kind'] == 'unidentified_verdicts']
        assert len(unidentified) == 1, [i['kind'] for i in payload['issues']]
        # The COUNT is what makes the issue actionable: two records vanished.
        assert '2' in unidentified[0]['detail']
        assert unidentified[0]['path'] == str(root / 'verdicts-current.json')

    def test_unkeyable_limits_verdict_records_are_counted_not_dropped(self, tmp_path: Path) -> None:
        """The same record-level discard sits in the limits reader.

        A non-object record, or one whose ``metric_id`` is not a string, is
        skipped when indexing the limits artifact's embedded ``verdicts[]``.
        The cost is the metric's ``rule_kind`` — the rule that governs it —
        silently reading as ``None``, which is indistinguishable from a limits
        artifact that simply never mentioned that metric.

        Carries an ``eval_id`` (unlike the root-scoped verdicts issue) because
        the limits artifact is per-eval, matching its sibling issues.
        """
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _healthy_tree(tmp_path)
        # Annotated `list[Any]`: the whole point is that two of these records
        # are NOT the shape the artifact promises, which the writer's own
        # parameter type would otherwise refuse to express.
        records: list[Any] = [
            {'metric_id': 'dangling-pointers', 'rule_kind': 'count'},
            'not-an-object',
            {'metric_id': None, 'rule_kind': 'count'},
        ]
        limits_path = _write_limits(root, 'eval-a', run_stamp=_AGE_RUN, verdicts=records)

        payload = build_memory_evals(root, esc_dir)

        # Provenance survives — one bad record must not cost the whole block.
        assert payload['evals'][0]['limits'] is not None
        assert _only(payload['evals'][0]['metrics'], 'dangling-pointers')['rule_kind'] == 'count'

        assert payload['issue_count'] == len(payload['issues'])
        unidentified = [i for i in payload['issues'] if i['kind'] == 'unidentified_limits_verdicts']
        assert len(unidentified) == 1, [i['kind'] for i in payload['issues']]
        assert unidentified[0]['eval_id'] == 'eval-a'
        assert unidentified[0]['path'] == str(limits_path)
        assert '2' in unidentified[0]['detail']

    def test_wrong_type_limits_verdicts_field_is_named(self, tmp_path: Path) -> None:
        """A non-list ``verdicts`` field loses every ``rule_kind`` at once.

        ``body.get('verdicts') or []`` iterates a dict's KEYS (or a string's
        characters), so every record check below fails and the index comes back
        empty with no signal — the artifact-level analogue of the ``entries``
        defect, one function earlier.

        Distinct from ``malformed_limits``: the body IS an object here, so the
        alpha/baseline/grandfather provenance is perfectly usable and must
        still be returned.  Only ``rule_kind`` is lost, so the issue is scoped
        to that rather than discarding a block that parsed fine.
        """
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _healthy_tree(tmp_path)
        limits_path = _write_limits(root, 'eval-a', run_stamp=_AGE_RUN)
        body = json.loads(limits_path.read_text())
        body['verdicts'] = {'dangling-pointers': 'count'}
        _dump(limits_path, body)

        payload = build_memory_evals(root, esc_dir)

        # The provenance block is NOT discarded — the body parsed and is an object.
        assert payload['evals'][0]['limits'] is not None
        assert payload['evals'][0]['limits']['alpha'] == 0.002777777777777778
        # But the rule this metric is governed by is genuinely gone.
        assert _only(payload['evals'][0]['metrics'], 'dangling-pointers')['rule_kind'] is None

        assert payload['issue_count'] == len(payload['issues'])
        malformed = [i for i in payload['issues'] if i['kind'] == 'malformed_limits_verdicts']
        assert len(malformed) == 1, [i['kind'] for i in payload['issues']]
        assert malformed[0]['eval_id'] == 'eval-a'
        assert malformed[0]['path'] == str(limits_path)
        assert 'dict' in malformed[0]['detail']
        # A body that IS an object must not also be reported as the wrong type.
        assert not [i for i in payload['issues'] if i['kind'] == 'malformed_limits']

    def test_malformed_entries_does_not_suppress_the_storm_block(self, tmp_path: Path) -> None:
        """One named degradation must not become a second silent one.

        ``storm_escape`` is run-scoped and parses independently of ``entries``,
        so a broken entries list has no bearing on whether a storm is in
        effect — and a suppressed storm banner is exactly the kind of quiet
        loss this reader is built to refuse.
        """
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _healthy_tree(tmp_path)
        _dump(root / 'verdicts-current.json', {
            'schema_version': 1,
            'entries': 'not-a-list',
            'storm_escape': {
                'triggered': True,
                'alarm_count': 7,
                'aggregate_fingerprint': _FP_AGGREGATE,
            },
        })

        payload = build_memory_evals(root, esc_dir)

        assert payload['storm_escape'] is not None
        assert payload['storm_escape']['triggered'] is True
        assert payload['storm_escape']['alarm_count'] == 7
        # And the entries breakage is still reported.
        assert [i['kind'] for i in payload['issues']] == ['malformed_verdicts']

    def test_wrong_type_limits_is_malformed_limits(self, tmp_path: Path) -> None:
        """The limits reader carries the identical defect, and the same fix.

        Found while verifying the verdicts finding: a limits artifact that
        parses but is a bare list falls through with no issue, and because
        ``missing_limits`` is only recorded when the file is ABSENT, a
        present-but-wrong-type artifact produced NO signal at all — losing the
        alpha/baseline/grandfather-hash provenance silently.
        """
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _healthy_tree(tmp_path)
        bad = _dump(root / 'eval-a' / 'limits-current.json', ['not', 'an', 'object'])

        payload = build_memory_evals(root, esc_dir)

        assert payload['issue_count'] == len(payload['issues'])
        malformed = [i for i in payload['issues'] if i['kind'] == 'malformed_limits']
        assert len(malformed) == 1
        # Per-eval, so it carries an eval_id — unlike the root-scoped verdicts issue.
        assert malformed[0]['eval_id'] == 'eval-a'
        assert malformed[0]['path'] == str(bad)
        assert 'list' in malformed[0]['detail']

        # Provenance stays absent rather than half-populated.
        assert payload['evals'][0]['limits'] is None

        # Malformed, not missing: reporting both would misdirect the operator
        # toward an evaluator that never ran, when the file is right there.
        assert [i for i in payload['issues'] if i['kind'] == 'missing_limits'] == []

    def test_unknown_kind_is_flagged_but_the_value_still_shows(self, tmp_path: Path) -> None:
        """A kind outside the closed vocabulary is a RENDERING failure.

        There is no chart primitive for it, so the dashboard genuinely cannot
        resolve it and says so.  The number itself is still real, so it is
        still displayed rather than blanked.
        """
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _healthy_tree(tmp_path, metrics=[_metric('odd-one', 'histogram', 1.4)])

        payload = build_memory_evals(root, esc_dir)

        assert payload['issue_count'] == len(payload['issues']) == 1
        issue = payload['issues'][0]
        assert issue['kind'] == 'unknown_kind'
        assert issue['eval_id'] == 'eval-a'
        assert 'histogram' in issue['detail']
        row = _only(payload['evals'][0]['metrics'], 'odd-one')
        assert row['kind'] == 'histogram'
        assert row['current_value'] == 1.4

    def test_metric_record_with_no_kind_is_named(self, tmp_path: Path) -> None:
        """A missing ``kind`` is as unrenderable as an unknown one.

        ``kind`` is a REQUIRED M1 field, and the reader's whole rationale for
        flagging ``histogram`` — there is no chart primitive for it — applies
        identically when the field is simply absent.  Skipping a ``None`` kind
        let an unclassifiable series render with nothing said anywhere.
        """
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _healthy_tree(tmp_path, metrics=[{'metric_id': 'kindless', 'value': 5.0, 'n': 1}])

        payload = build_memory_evals(root, esc_dir)

        assert payload['issue_count'] == len(payload['issues']) == 1
        issue = payload['issues'][0]
        # Its own kind, not `unknown_kind`: the operator-facing fix is a field
        # to add, not a chart vocabulary to widen.
        assert issue['kind'] == 'missing_kind'
        assert issue['eval_id'] == 'eval-a'
        assert issue['path'] == str(root / 'eval-a' / f'metrics-{_AGE_RUN}.json')
        assert 'kindless' in issue['detail']

        # The value is still real, so it is still shown.
        row = _only(payload['evals'][0]['metrics'], 'kindless')
        assert row['kind'] is None
        assert row['current_value'] == 5.0

    def test_a_metric_absent_from_a_run_is_not_a_missing_kind(self, tmp_path: Path) -> None:
        """The hole case and the defect case both read as ``None`` — only one is a defect.

        A metric introduced mid-window is absent from the older runs, and both
        "absent from this run" and "present with no kind" collapse to ``None``
        off a ``.get`` chain.  Conflating them would make every legitimate
        trend hole raise a ``missing_kind``.
        """
        from dashboard.data.memory_evals import build_memory_evals

        root = tmp_path / 'memory-evals'
        esc_dir = tmp_path / 'escalations'
        esc_dir.mkdir(parents=True, exist_ok=True)
        _write_metrics(root, 'eval-a', '20260701T031500Z', [_metric('old-timer', 'count', 1.0)])
        _write_metrics(root, 'eval-a', '20260702T031500Z', [
            _metric('old-timer', 'count', 2.0), _metric('latecomer', 'count', 7.0),
        ])
        _write_limits(root, 'eval-a', run_stamp='20260702T031500Z')
        _write_verdicts(root, [], run_stamp='20260702T031500Z')

        payload = build_memory_evals(root, esc_dir)

        assert payload['issue_count'] == len(payload['issues']) == 0
        assert _only(payload['evals'][0]['metrics'], 'latecomer')['trend']['values'] == [None, 7.0]

    def test_duplicate_metric_id_in_one_run_is_named(self, tmp_path: Path) -> None:
        """Two records for one series collapse — deterministically, and loudly.

        ``metric_id`` is unique within a run (M1), so a dict comprehension over
        the array reads as safe; on a duplicate it silently kept whichever
        record came LAST, so the row's ``current_value`` was one of two
        candidates with nothing saying a choice had been made.
        """
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _healthy_tree(tmp_path, metrics=[
            _metric('dup', 'count', 1.0),
            _metric('dup', 'count', 99.0),
        ])

        payload = build_memory_evals(root, esc_dir)

        assert payload['issue_count'] == len(payload['issues']) == 1
        issue = payload['issues'][0]
        assert issue['kind'] == 'duplicate_metric_id'
        assert issue['eval_id'] == 'eval-a'
        assert issue['path'] == str(root / 'eval-a' / f'metrics-{_AGE_RUN}.json')
        assert 'dup' in issue['detail']

        # One row, and the FIRST record is the one kept — a stated rule rather
        # than whatever array order happened to deliver.
        (row,) = payload['evals'][0]['metrics']
        assert row['metric_id'] == 'dup'
        assert row['current_value'] == 1.0

    def test_metric_record_with_no_metric_id_is_counted_not_dropped(self, tmp_path: Path) -> None:
        """An unidentifiable record cannot be charted — but its loss is reported.

        One issue per FILE with a count, not one per record: a systematically
        broken artifact degrades one row instead of flooding the issues list.
        """
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _healthy_tree(tmp_path, metrics=[
            _metric('real-one', 'count', 4.0),
            {'kind': 'count', 'value': 1.0, 'n': 1},
            {'metric_id': '', 'kind': 'count', 'value': 2.0, 'n': 1},
        ])

        payload = build_memory_evals(root, esc_dir)

        assert payload['issue_count'] == len(payload['issues']) == 1
        assert payload['issues'][0]['kind'] == 'unidentified_metrics'
        assert payload['issues'][0]['eval_id'] == 'eval-a'
        assert '2' in payload['issues'][0]['detail']
        assert [row['metric_id'] for row in payload['evals'][0]['metrics']] == ['real-one']

    def test_duplicate_verdict_entry_is_named(self, tmp_path: Path) -> None:
        """Two judgements of one metric would pick the row's verdict by array order."""
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _healthy_tree(tmp_path)
        _write_verdicts(root, [
            _verdict('eval-a', 'dangling-pointers', 'alarm', fingerprint='a' * 32, run_stamp=_AGE_RUN),
            _verdict('eval-a', 'dangling-pointers', 'no_alarm', fingerprint='b' * 32, run_stamp=_AGE_RUN),
        ], run_stamp=_AGE_RUN)

        payload = build_memory_evals(root, esc_dir)

        assert payload['issue_count'] == len(payload['issues']) == 1
        issue = payload['issues'][0]
        assert issue['kind'] == 'duplicate_verdict_entry'
        assert issue['eval_id'] == 'eval-a'
        assert issue['path'] == str(root / 'verdicts-current.json')
        # First wins — and it is the verdict AND the fingerprint (hence the
        # escalation link) that array order would otherwise have chosen.
        row = _only(payload['evals'][0]['metrics'], 'dangling-pointers')
        assert row['verdict'] == 'alarm'
        assert row['fingerprint'] == 'a' * 32

    def test_two_escalations_sharing_a_fingerprint_are_named(self, tmp_path: Path) -> None:
        """A dropped escalation is exactly what the parity view exists to catch.

        Indexed last-wins, the loser was invisible EVERYWHERE in the payload —
        no row link and no ``unmatched_escalations`` entry.
        """
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _healthy_tree(tmp_path)
        _write_escalation(esc_dir, 'esc-1', dedupe_fingerprint='c' * 32)
        _write_escalation(esc_dir, 'esc-2', dedupe_fingerprint='c' * 32)

        payload = build_memory_evals(root, esc_dir)

        collisions = [i for i in payload['issues'] if i['kind'] == 'duplicate_escalation_fingerprint']
        assert len(collisions) == 1
        assert payload['issue_count'] == len(payload['issues'])
        assert collisions[0]['path'] == str(esc_dir)
        # BOTH ids are named — which of the two the queue scan reached first is
        # ``Path.glob`` order and so is not the test's to pin, but an operator
        # must be able to see exactly which pair collided either way.
        assert 'esc-1' in collisions[0]['detail']
        assert 'esc-2' in collisions[0]['detail']
        # Exactly one survivor, still reported in the usual way — the point is
        # that the loser is no longer invisible, not which one lost.
        assert [e['id'] for e in payload['unmatched_escalations']] in (['esc-1'], ['esc-2'])

    def test_unreadable_root_degrades_the_payload_not_the_response(self, tmp_path, monkeypatch) -> None:
        """The enumeration is an artifact boundary like every other one here.

        ``is_dir()`` can answer True and the walk still fail — a mode change
        between the two, or an NFS/permission hiccup.  Unguarded, that turned a
        degraded tree into a 500 on the dashboard poll.
        """
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _healthy_tree(tmp_path)

        def _boom(self: Path) -> Any:
            raise PermissionError(13, 'Permission denied')

        monkeypatch.setattr(Path, 'iterdir', _boom)

        payload = build_memory_evals(root, esc_dir)

        # Present, but unreadable — NOT reported as an empty root, which is the
        # healthy "no eval has ever run" state.
        assert payload['root_present'] is True
        assert payload['evals'] == []
        assert payload['issue_count'] == len(payload['issues']) == 1
        assert payload['issues'][0]['kind'] == 'unreadable_root'
        assert payload['issues'][0]['path'] == str(root)
        assert set(payload) == _PAYLOAD_KEYS

    def test_unreadable_escalations_degrades_the_payload_not_the_response(
        self, tmp_path: Path, monkeypatch: Any,
    ) -> None:
        """The escalation join is an artifact boundary like every other one here.

        Every other boundary in ``build_memory_evals`` (verdicts, limits, the
        root walk, each metrics run) is wrapped, so the "never raises" contract
        holds structurally.  Leaving the join open made that contract depend on
        auditing every field access inside the index for an unvalidated JSON
        type — which is how the unhashable-``status`` 500 got in.

        The stub takes ``**_kwargs`` deliberately, and must keep doing so: the
        caller passes ``skipped=`` now, and a stub that rejected it would raise
        an arity ``TypeError`` instead of the ``PermissionError`` this test is
        about.  ``TypeError`` is in ``_ARTIFACT_ERRORS``, so the test would
        still go GREEN while silently no longer exercising the read boundary at
        all.  The ``unreadable_escalation_file`` assertion below is the tell:
        it distinguishes "the whole read failed" (this case) from "the reader
        ran and skipped some files", which an arity failure could never reach.
        """
        from dashboard.data import memory_evals as memory_evals_mod
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _healthy_tree(tmp_path)

        def _boom(_dir: Path, **_kwargs: Any) -> Any:
            raise PermissionError(13, 'Permission denied')

        monkeypatch.setattr(memory_evals_mod, 'load_queue_escalations', _boom)

        payload = build_memory_evals(root, esc_dir)

        # The eval rows still render — only the join is lost, and it is NAMED
        # rather than silently reading as "nothing is escalated".
        assert payload['root_present'] is True
        assert payload['evals'] != []
        assert payload['unmatched_escalations'] == []
        assert payload['issue_count'] == len(payload['issues']) == 1
        assert payload['issues'][0]['kind'] == 'unreadable_escalations'
        assert payload['issues'][0]['path'] == str(esc_dir)
        # The failure that was caught is the INJECTED one.  ``detail`` is
        # ``str(exc)``, so this is what makes an arity ``TypeError`` (which
        # ``_ARTIFACT_ERRORS`` would swallow, greening this test while
        # exercising nothing) distinguishable from the read failure the test is
        # actually about — see the stub's ``**_kwargs`` note above.
        assert 'Permission denied' in payload['issues'][0]['detail']
        # The whole read failed, so no per-file skip could have been collected:
        # this is the plural "the join blew up" kind, never the singular one.
        assert 'unreadable_escalation_file' not in {i['kind'] for i in payload['issues']}
        assert set(payload) == _PAYLOAD_KEYS

    def test_partial_skips_survive_a_reader_that_dies_mid_scan(
        self, tmp_path: Path, monkeypatch: Any,
    ) -> None:
        """Skips recorded before the reader blew up are still named.

        ``skipped`` is filled inside the reader but drained by the caller, so
        the two are only atomic if the drain runs on the exception path too.
        A reader that dies part-way through the scan — ``glob`` hitting a mode
        change or an NFS hiccup, the case the ``_ARTIFACT_ERRORS`` wrapper
        exists for — would otherwise take every skip it had already recorded
        out with it, leaving only the coarse ``unreadable_escalations`` issue:
        this change's own silent discard, reappearing one frame up in exactly
        the degraded scenario it was written to close.

        Both kinds are expected together and they say different things: the
        plural one means the scan died, the singular ones name the files it had
        already given up on before dying.
        """
        from dashboard.data import memory_evals as memory_evals_mod
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _healthy_tree(tmp_path)
        doomed = esc_dir / 'esc-half-read.json'

        def _die_mid_scan(
            _dir: Path, *, skipped: list[dict[str, Any]] | None = None, **_kwargs: Any,
        ) -> Any:
            # One file was read and skipped; the NEXT dir entry blew the scan up.
            if skipped is not None:
                skipped.append({'path': doomed, 'error': 'Expecting value: line 1 column 1'})
            raise PermissionError(13, 'Permission denied')

        monkeypatch.setattr(memory_evals_mod, 'load_queue_escalations', _die_mid_scan)

        payload = build_memory_evals(root, esc_dir)

        assert payload['issue_count'] == len(payload['issues']) == 2
        per_file = [i for i in payload['issues'] if i['kind'] == 'unreadable_escalation_file']
        coarse = [i for i in payload['issues'] if i['kind'] == 'unreadable_escalations']
        assert len(per_file) == len(coarse) == 1
        assert per_file[0]['path'] == str(doomed)
        assert 'Expecting value' in per_file[0]['detail']
        assert coarse[0]['path'] == str(esc_dir)
        # The drain does not SWALLOW the reader's exception — it re-raises into
        # the existing wrapper, which is what produces the coarse issue.
        assert 'Permission denied' in coarse[0]['detail']
        assert payload['evals'] != []
        assert payload['unmatched_escalations'] == []
        assert set(payload) == _PAYLOAD_KEYS

    def test_unreadable_queue_file_is_named_in_issues(self, tmp_path: Path) -> None:
        """The one discard that happens a frame DOWN is named too.

        ``load_queue_escalations`` skips an unparseable queue file — correct,
        one corrupt escalation must not crash the join.  But the skip used to
        stop at a WARNING log, so a corrupt file holding an open alarm left
        ``unmatched_escalations`` silently short one record while
        ``issue_count`` stayed 0: the parity view reporting "nothing
        unexplained" in exactly the case it exists to catch.  The module
        docstring's no-silent-discard invariant covers every discard INSIDE
        this module; this pins the one it reaches through the shared reader.
        """
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _healthy_tree(tmp_path)
        _write_escalation(esc_dir, 'esc-good', dedupe_fingerprint='g' * 32)
        bad = _corrupt(_write_escalation(esc_dir, 'esc-bad', dedupe_fingerprint='b' * 32))

        payload = build_memory_evals(root, esc_dir)

        assert payload['issue_count'] == len(payload['issues']) == 1
        issue = payload['issues'][0]
        assert issue['kind'] == 'unreadable_escalation_file'
        # The FILE, not the queue dir — the discriminator against the sibling
        # record-level issues (``unknown_escalation_status`` and friends), which
        # name the dir because they describe records inside a file that DID
        # parse.  Here the reader knows exactly which file to go repair.
        assert issue['path'] == str(bad)
        # The underlying parse error reaches the PAYLOAD, not just the WARNING
        # log — that is the whole point of the channel, and the operator needs
        # it to tell bad JSON from a permissions problem from a directory.  A
        # truthiness check would stay green if a regression dropped
        # ``entry['error']`` from the f-string, so assert the text.
        assert 'Expecting value' in issue['detail']
        # ...and the claim about what was LOST stays conditional.  The file is
        # unreadable, so its category and status are exactly what is unknown,
        # and this queue is shared across every escalation category — the
        # likeliest corrupt file is one this join would have filtered out
        # anyway.  Asserting the loss as fact would put a falsehood in the
        # payload of the view whose job is to never tell one.
        assert 'if it held' in issue['detail']
        # The readable escalation still joins — one bad file degrades one
        # record, never the queue.
        assert [e['id'] for e in payload['unmatched_escalations']] == ['esc-good']
        assert set(payload) == _PAYLOAD_KEYS

    def test_every_unreadable_queue_file_gets_its_own_issue(self, tmp_path: Path) -> None:
        """One issue per file, and emission is not gated on any record surviving.

        File-level failures get one issue apiece here (like
        ``unreadable_metrics``); only RECORD-level failures inside a parsed
        file are collapsed with a count, because one artifact can hold
        thousands of records.  A queue file is a file-level artifact, and each
        corrupt one is a distinct operator action — go read and repair *that*
        file — which a collapsed count would erase.
        """
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _healthy_tree(tmp_path)
        first = _corrupt(_write_escalation(esc_dir, 'esc-bad-1'))
        second = _corrupt(_write_escalation(esc_dir, 'esc-bad-2'))

        payload = build_memory_evals(root, esc_dir)

        named = [i for i in payload['issues'] if i['kind'] == 'unreadable_escalation_file']
        assert len(named) == 2
        assert {i['path'] for i in named} == {str(first), str(second)}
        assert payload['issue_count'] == len(payload['issues']) == 2
        # Nothing survived the read, and the issues are emitted anyway.
        assert payload['unmatched_escalations'] == []
        assert payload['evals'] != []
        assert set(payload) == _PAYLOAD_KEYS

    def test_missing_run_stamp_is_named(self, tmp_path: Path) -> None:
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _healthy_tree(tmp_path, run_stamp=None)

        payload = build_memory_evals(root, esc_dir, now=_AGE_RUN_AT)

        assert payload['issue_count'] == len(payload['issues']) == 1
        assert payload['issues'][0]['kind'] == 'missing_run_stamp'
        assert payload['evals'][0]['latest_run_age_seconds'] is None
        assert payload['evals'][0]['stale'] is False

    def test_unparseable_run_stamp_degrades_the_age_not_the_payload(self, tmp_path: Path) -> None:
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _healthy_tree(tmp_path, run_stamp='not-a-stamp')

        payload = build_memory_evals(root, esc_dir, now=_AGE_RUN_AT)

        assert payload['issue_count'] == len(payload['issues']) == 1
        assert payload['issues'][0]['kind'] == 'unparseable_run_stamp'
        assert payload['evals'][0]['latest_run_stamp'] == 'not-a-stamp'
        assert payload['evals'][0]['latest_run_age_seconds'] is None
        assert payload['evals'][0]['stale'] is False

    def test_missing_root_is_empty_but_healthy(self, tmp_path: Path) -> None:
        """"No eval has ever run" is a legitimate state, not a degradation.

        Flagging it would train operators to ignore the issues list.
        """
        from dashboard.data.memory_evals import build_memory_evals

        payload = build_memory_evals(tmp_path / 'never-created', tmp_path / 'escalations')

        assert payload['root_present'] is False
        assert payload['evals'] == []
        assert payload['issues'] == []
        assert payload['issue_count'] == 0

    def test_empty_root_is_present_with_no_evals(self, tmp_path: Path) -> None:
        from dashboard.data.memory_evals import build_memory_evals

        root = tmp_path / 'memory-evals'
        root.mkdir(parents=True)

        payload = build_memory_evals(root, tmp_path / 'escalations')

        assert payload['root_present'] is True
        assert payload['evals'] == []
        # No eval dirs means nothing is waiting on a verdict.
        assert payload['issues'] == []
        assert payload['issue_count'] == 0

    def test_eval_dir_that_has_never_run_is_not_a_missing_verdict(self, tmp_path: Path) -> None:
        """An eval dir with no runs is pre-deployment, not contract drift.

        ``missing_verdicts`` answers "there are metrics on disk with nothing
        judging them" — a real silent-failure mode.  Gating it on the eval
        DIRECTORY existing instead of on any RUN existing fires it for a
        freshly-registered eval that has not run yet, which is the same
        empty-but-healthy state the absent-root case is explicitly not allowed
        to flag.  ``_read_limits`` already draws the line at ``has_runs``; the
        two readers must not disagree about what "never run" means, or one
        harmless tree produces a standing issue operators learn to ignore.
        """
        from dashboard.data.memory_evals import build_memory_evals

        root = tmp_path / 'memory-evals'
        (root / 'e1-retrieval-health').mkdir(parents=True)

        payload = build_memory_evals(root, tmp_path / 'escalations')

        assert payload['root_present'] is True
        assert [row['eval_id'] for row in payload['evals']] == ['e1-retrieval-health']
        assert payload['evals'][0]['run_stamps'] == []
        assert payload['issues'] == []
        assert payload['issue_count'] == 0


# ---------------------------------------------------------------------------
# step-17 — the redux shape fn
# ---------------------------------------------------------------------------


class TestShapeMemoryEvals:
    """``shape_memory_evals`` — one Redux key, pure, and never sharing state.

    The shape layer is the seam beta (the React section) consumes: these field
    names ARE the contract (PRD open question 4 — alpha's shape fn plus these
    tests pin them, beta consumes them).  Everything asserted here is about the
    seam itself; the payload's *content* is pinned by the builder tests above.
    """

    def test_returns_exactly_one_redux_key(self, tmp_path: Path) -> None:
        from dashboard.data import redux_api
        from dashboard.data.memory_evals import build_memory_evals

        payload = build_memory_evals(*_two_eval_tree(tmp_path))

        shaped = redux_api.shape_memory_evals(**payload)

        # One top-level key, exactly like shape_curator/shape_scheduler — the
        # route returns this dict verbatim, so an extra key here would land in
        # the Redux store as an unowned slice.
        assert set(shaped) == {'MEMORY_EVALS'}

    def test_every_top_level_payload_key_survives(self, tmp_path: Path) -> None:
        """The builder's whole payload reaches the UI — nothing is dropped in transit.

        A shape fn that silently omits a key would leave the section rendering
        a blank column with no error anywhere, which is exactly the silent
        degradation DD6 forbids.
        """
        from dashboard.data import redux_api
        from dashboard.data.memory_evals import build_memory_evals

        payload = build_memory_evals(*_two_eval_tree(tmp_path))

        body = redux_api.shape_memory_evals(**payload)['MEMORY_EVALS']

        assert set(body) == _PAYLOAD_KEYS
        # Value-for-value, not merely key-for-key: the shape layer reshapes
        # nothing, it only names the slice.
        for key in _PAYLOAD_KEYS:
            assert body[key] == payload[key], key

    def test_top_level_containers_are_shallow_copied(self) -> None:
        """The ``shape_curator`` shallow-copy contract, pinned.

        The route hands the builder's payload straight in; if the shape fn
        aliased the caller's lists, a later mutation of the cached builder
        result would retroactively rewrite an already-returned response.
        """
        from dashboard.data import redux_api

        evals: list[dict] = [{'eval_id': 'eval-a'}]
        issues: list[dict] = [{'kind': 'missing_limits'}]
        unmatched: list[dict] = [{'id': 'esc-1'}]
        storm: dict = {'triggered': True, 'alarm_count': 2}

        body = redux_api.shape_memory_evals(
            generated_at='2026-07-30T03:15:00+00:00',
            root_present=True,
            storm_escape=storm,
            evals=evals,
            issues=issues,
            issue_count=1,
            unmatched_escalations=unmatched,
        )['MEMORY_EVALS']

        evals.append({'eval_id': 'eval-b'})
        issues.append({'kind': 'orphan_verdict'})
        unmatched.append({'id': 'esc-2'})
        storm['alarm_count'] = 99

        assert body['evals'] == [{'eval_id': 'eval-a'}]
        assert body['issues'] == [{'kind': 'missing_limits'}]
        assert body['unmatched_escalations'] == [{'id': 'esc-1'}]
        assert body['storm_escape'] == {'triggered': True, 'alarm_count': 2}

    def test_defaults_are_fresh_literals_not_a_shared_constant(self) -> None:
        """Two default calls must not hand out the same mutable containers.

        A module-level empty-payload constant would let one request's response
        object be mutated by another's — the bug ``shape_curator``'s
        "fresh literal each call" comment already records.
        """
        from dashboard.data import redux_api

        first = redux_api.shape_memory_evals()['MEMORY_EVALS']
        second = redux_api.shape_memory_evals()['MEMORY_EVALS']

        assert first == second
        assert first is not second
        for key in ('evals', 'issues', 'unmatched_escalations'):
            assert first[key] == []
            assert first[key] is not second[key], key

        first['evals'].append({'eval_id': 'leaked'})
        assert second['evals'] == []

    def test_empty_default_payload_is_the_missing_root_shape(self) -> None:
        """Called with nothing, the shape fn agrees with the builder's empty payload.

        Same keys, same "nothing has run yet" semantics — so a default-shaped
        response and a real one are never structurally different to the UI.
        """
        from dashboard.data import redux_api

        body = redux_api.shape_memory_evals()['MEMORY_EVALS']

        assert set(body) == _PAYLOAD_KEYS
        assert body['root_present'] is False
        assert body['storm_escape'] is None
        assert body['evals'] == []
        assert body['issues'] == []
        assert body['issue_count'] == 0
        assert body['unmatched_escalations'] == []
        assert body['generated_at'] is None

    def test_shape_is_io_free(self, tmp_path: Path, monkeypatch) -> None:
        """No filesystem read in the shape layer — all I/O belongs to the builder.

        The route runs the builder in a worker thread precisely because it
        touches disk; a shape fn that also read would do that I/O back on the
        event loop.
        """
        from dashboard.data import redux_api
        from dashboard.data.memory_evals import build_memory_evals

        payload = build_memory_evals(*_two_eval_tree(tmp_path))

        def _no_io(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError('shape_memory_evals must not touch the filesystem')

        monkeypatch.setattr(builtins, 'open', _no_io)
        monkeypatch.setattr(Path, 'open', _no_io)
        monkeypatch.setattr(Path, 'read_text', _no_io)

        shaped = redux_api.shape_memory_evals(**payload)

        assert set(shaped) == {'MEMORY_EVALS'}


# ---------------------------------------------------------------------------
# step-19 — GET /api/v2/dashboard/memory-evals
# ---------------------------------------------------------------------------

_ROUTE_URL = '/api/v2/dashboard/memory-evals'
_ROUTE_RUNS = ('20260703T031500Z', '20260704T031500Z', '20260705T031500Z')
_ROUTE_FINGERPRINT = 'eval:e1-retrieval-health|metric:dangling-pointers|item:node-7'


def _route_tree(tmp_path: Path) -> DashboardConfig:
    """A full artifact tree at the paths the ROUTE resolves from config.

    Built through ``config.memory_evals_dir`` / ``config.reconciliation_escalations_dir``
    rather than hand-spelled paths, so the test exercises the route's own
    resolution instead of agreeing with a second copy of it.
    """
    config = _make_config(tmp_path)
    root = config.memory_evals_dir
    esc_dir = config.reconciliation_escalations_dir
    esc_dir.mkdir(parents=True, exist_ok=True)

    for stamp in _ROUTE_RUNS:
        _write_metrics(root, 'e1-retrieval-health', stamp, [
            _metric('canonical-in-top-5', 'proportion', 0.94, n=50, denominator=50),
            _metric('dangling-pointers', 'count', 4.0, n=4),
        ])
    _write_limits(
        root, 'e1-retrieval-health',
        run_stamp=_ROUTE_RUNS[-1],
        baseline_run_stamps=list(_ROUTE_RUNS[:2]),
        verdicts=[
            _limits_verdict('canonical-in-top-5', 'proportion'),
            _limits_verdict('dangling-pointers', 'count'),
        ],
    )
    _write_verdicts(root, [
        _verdict(
            'e1-retrieval-health', 'dangling-pointers', 'alarm',
            fingerprint=_ROUTE_FINGERPRINT, value=4.0,
            limit_ref='count>=3', run_stamp=_ROUTE_RUNS[-1], item='node-7',
        ),
        _verdict(
            'e1-retrieval-health', 'canonical-in-top-5', 'no_alarm',
            fingerprint='eval:e1-retrieval-health|metric:canonical-in-top-5',
            value=0.94, run_stamp=_ROUTE_RUNS[-1],
        ),
    ], run_stamp=_ROUTE_RUNS[-1])
    _write_escalation(esc_dir, 'esc-eval-1', dedupe_fingerprint=_ROUTE_FINGERPRINT)
    return config


def _payload(**overrides: Any) -> dict[str, Any]:
    """A healthy builder-shaped payload, overridable one key at a time.

    Shaped as ``build_memory_evals`` returns it (``_PAYLOAD_KEYS`` exactly), so
    a stub standing in for the builder cannot pass the route something the
    route's own ``shape_memory_evals(**result)`` call would reject.
    """
    payload: dict[str, Any] = {
        'generated_at': '2026-07-06T03:15:00+00:00',
        'root_present': True,
        'storm_escape': None,
        'evals': [],
        'issues': [],
        'issue_count': 0,
        'unmatched_escalations': [],
    }
    payload.update(overrides)
    payload['issue_count'] = len(payload['issues'])
    return payload


# ---------------------------------------------------------------------------
# a scan that never reached the tree must not be pinned for the TTL window
# ---------------------------------------------------------------------------


class TestRootScanCacheability:
    """``root_scan_succeeded`` — "did this scan actually reach the tree?"

    The TTL cache in front of this route exists because the artifact walk is
    expensive.  That reasoning does not extend to the two payloads produced
    WITHOUT walking anything: an absent root returns immediately from
    ``is_dir()``, and an unwalkable one raises immediately from ``iterdir()``.
    Both are O(1) to recompute, so retrying them on every poll costs nothing
    and there is no thundering herd to protect — while caching them pins a
    "there is no data" view of a tree that may exist by the next poll (a mount
    landing, a mode being fixed) for the full TTL window.

    The predicate is deliberately narrow.  A degraded ROW — an unreadable
    metrics run, an unknown kind, an orphan verdict — stays cacheable: the scan
    DID reach the tree, re-running it would return the same degradation, and
    that re-run is the expensive walk this cache is here to prevent.
    """

    def test_healthy_payload_is_cacheable(self) -> None:
        from dashboard.data.memory_evals import root_scan_succeeded

        assert root_scan_succeeded(_payload()) is True

    def test_absent_root_is_not_cacheable(self) -> None:
        """``root_present: False`` — ``is_dir()`` said no, so nothing was walked."""
        from dashboard.data.memory_evals import root_scan_succeeded

        assert root_scan_succeeded(_payload(root_present=False)) is False

    def test_unreadable_root_is_not_cacheable(self) -> None:
        """The root IS present but the enumeration raised — also nothing walked."""
        from dashboard.data.memory_evals import root_scan_succeeded

        payload = _payload(issues=[{
            'kind': 'unreadable_root',
            'path': '/tmp/memory-evals',
            'detail': '[Errno 13] Permission denied',
        }])

        assert payload['root_present'] is True
        assert root_scan_succeeded(payload) is False

    def test_a_degraded_row_stays_cacheable(self) -> None:
        """Row-level degradation is not a failed scan — the walk happened.

        This is the load-bearing half.  Keying the cache on "any issue at all"
        would make a single permanently-corrupt metrics file defeat the cache
        forever, re-running the full walk on every poll to rediscover a
        degradation that re-scanning cannot fix.
        """
        from dashboard.data.memory_evals import root_scan_succeeded

        payload = _payload(issues=[
            {'kind': 'unreadable_metrics', 'path': '/tmp/m.json', 'detail': 'x'},
            {'kind': 'unknown_kind', 'path': '/tmp/m.json', 'detail': 'y'},
            {'kind': 'orphan_verdict', 'path': '/tmp/v.json', 'detail': 'z'},
        ])

        assert payload['issue_count'] == 3
        assert root_scan_succeeded(payload) is True

    def test_a_partial_payload_cannot_raise_inside_the_predicate(self) -> None:
        """It runs inside the cache write path, where raising would 500 the poll.

        ``build_memory_evals`` returns every key on every path, so this is
        defence rather than a live case — but a predicate that trusts that
        invariant turns any future violation of it into a crash at the least
        debuggable point in the request.
        """
        from dashboard.data.memory_evals import root_scan_succeeded

        assert root_scan_succeeded({}) is False
        assert root_scan_succeeded({'root_present': True}) is True


class TestMemoryEvalsEndpoint:
    """``GET /api/v2/dashboard/memory-evals`` end-to-end through the real route.

    Sync tests driven by the starlette ``TestClient`` (``client`` conftest
    fixture), mirroring the escalation-analytics route suite: the config is
    swapped onto ``app.state`` and the TTL cache is cleared before each GET so
    one test's payload can never be served to the next.
    """

    def test_full_payload_reaches_the_client(self, client, tmp_path: Path) -> None:
        from dashboard.app import _memory_evals_cache_clear

        config = _route_tree(tmp_path)
        client.app.state.config = config
        _memory_evals_cache_clear()

        resp = client.get(_ROUTE_URL)

        assert resp.status_code == 200
        body = resp.json()
        assert set(body) == {'MEMORY_EVALS'}
        payload = body['MEMORY_EVALS']
        assert set(payload) == _PAYLOAD_KEYS
        assert payload['root_present'] is True
        assert payload['issue_count'] == len(payload['issues']) == 0

        (row,) = payload['evals']
        assert row['eval_id'] == 'e1-retrieval-health'
        assert row['run_stamps'] == list(_ROUTE_RUNS)
        assert row['latest_run_stamp'] == _ROUTE_RUNS[-1]

        # Limits provenance survived the JSON round trip.
        assert row['limits']['alpha'] == 0.002777777777777778
        assert row['limits']['baseline_run_stamps'] == list(_ROUTE_RUNS[:2])
        assert row['limits']['stale_for_latest_run'] is False

        by_id = {m['metric_id']: m for m in row['metrics']}
        assert set(by_id) == {'canonical-in-top-5', 'dangling-pointers'}

        # Trends: a full-width series over the shared run axis.
        trend = by_id['canonical-in-top-5']['trend']
        assert trend['labels'] == list(_ROUTE_RUNS)
        assert trend['values'] == [0.94, 0.94, 0.94]
        assert by_id['canonical-in-top-5']['current_value'] == 0.94
        assert by_id['canonical-in-top-5']['rule_kind'] == 'proportion'

        # Verdict + the fingerprint-matched escalation, all the way to the UI.
        alarmed = by_id['dangling-pointers']
        assert alarmed['verdict'] == 'alarm'
        assert alarmed['fingerprint'] == _ROUTE_FINGERPRINT
        assert alarmed['escalation']['id'] == 'esc-eval-1'
        assert alarmed['parity'] == 'alarmed_open'
        assert by_id['canonical-in-top-5']['parity'] == 'clear'
        assert payload['unmatched_escalations'] == []

    def test_malformed_artifact_never_500s_and_is_counted(self, client, tmp_path: Path) -> None:
        """The row-9 contract, for this route: degrade loudly, never crash.

        Mirrors ``test_row9_malformed_regime_markers_never_500s`` — a corrupt
        artifact must raise ``issue_count``, not the status code.
        """
        from dashboard.app import _memory_evals_cache_clear

        config = _route_tree(tmp_path)
        client.app.state.config = config
        _memory_evals_cache_clear()

        pre = client.get(_ROUTE_URL).json()['MEMORY_EVALS']['issue_count']
        assert pre == 0

        _corrupt(config.memory_evals_dir / 'e1-retrieval-health' / f'metrics-{_ROUTE_RUNS[0]}.json')
        _memory_evals_cache_clear()
        resp = client.get(_ROUTE_URL)

        assert resp.status_code == 200
        payload = resp.json()['MEMORY_EVALS']
        assert payload['issue_count'] > pre
        assert 'unreadable_metrics' in {i['kind'] for i in payload['issues']}
        # Still serving: the other two runs render.
        assert payload['evals'][0]['run_stamps'] == list(_ROUTE_RUNS[1:])

    def test_corrupt_escalation_file_is_a_200_not_a_500(self, client, tmp_path: Path) -> None:
        """The other artifact tree the route reads must degrade the same way.

        ``test_malformed_artifact_never_500s_and_is_counted`` corrupts a
        METRICS artifact; the escalation queue is the second unvalidated tree
        this route walks, and it reaches the reader through a different code
        path (``load_queue_escalations``, reused per INV-5 rather than
        reimplemented).  This pins the half of the contract this module
        depends on but does not own: an unparseable queue file is skipped, so
        the section still renders and the row degrades to unlinked rather than
        the whole poll 500ing.

        The reader-level contract it depends on is a dependency worth pinning
        here: if ``load_queue_escalations`` ever started propagating a
        ``JSONDecodeError``, this route would be one of the callers that
        breaks, and nothing else here would catch it.

        This test also holds the LOUD half closed.  The skip used to produce no
        payload issue at all — the discard happened inside the shared reader,
        one frame below the module that owns the ``issues`` channel — so a
        corrupt file holding an open alarm left the parity view silently short
        one escalation and ``unmatched_escalations`` could not be read as
        exhaustive.  The reader now reports its skips through an opt-in
        ``skipped`` accumulator and this module names each one; the assertions
        below (``issue_count`` rising from 0, and the
        ``unreadable_escalation_file`` kind present) are what keep the gap
        closed at the route level, where the operator actually sees it.
        """
        from dashboard.app import _memory_evals_cache_clear

        config = _route_tree(tmp_path)
        client.app.state.config = config
        _memory_evals_cache_clear()

        pre = client.get(_ROUTE_URL).json()['MEMORY_EVALS']
        assert _only(pre['evals'][0]['metrics'], 'dangling-pointers')['parity'] == 'alarmed_open'
        assert pre['issue_count'] == 0

        _corrupt(config.reconciliation_escalations_dir / 'esc-eval-1.json')
        _memory_evals_cache_clear()
        resp = client.get(_ROUTE_URL)

        assert resp.status_code == 200
        payload = resp.json()['MEMORY_EVALS']
        # The eval section still renders in full; only the join is lost.
        assert payload['evals'][0]['run_stamps'] == list(_ROUTE_RUNS)
        row = _only(payload['evals'][0]['metrics'], 'dangling-pointers')
        assert row['escalation'] is None
        # Degraded to unlinked — never silently re-rendered as still-linked.
        assert row['parity'] == 'alarmed_unlinked'
        # ...and LOUDLY: the lost alarm is named, not merely absent.
        assert payload['issue_count'] == 1
        assert 'unreadable_escalation_file' in {i['kind'] for i in payload['issues']}

    def test_ttl_cache_single_flights_the_scan(self, client, tmp_path: Path) -> None:
        """Within the TTL window the disk is not re-scanned.

        Asserted by mutating the tree and getting the OLD payload back — the
        only observable proof that the ~60s single-flight cache is actually in
        the path, and the reason a cold artifact scan cannot be re-run per poll.
        """
        from dashboard.app import _memory_evals_cache_clear

        config = _route_tree(tmp_path)
        client.app.state.config = config
        _memory_evals_cache_clear()

        first = client.get(_ROUTE_URL).json()

        _write_metrics(
            config.memory_evals_dir, 'e1-retrieval-health', '20260706T031500Z',
            [_metric('dangling-pointers', 'count', 9.0, n=9)],
        )

        assert client.get(_ROUTE_URL).json() == first

        _memory_evals_cache_clear()
        after = client.get(_ROUTE_URL).json()['MEMORY_EVALS']

        assert after['evals'][0]['run_stamps'][-1] == '20260706T031500Z'

    def test_a_healthy_scan_is_still_served_from_cache(
        self, client, monkeypatch, tmp_path: Path,
    ) -> None:
        """Gating the cache must not disable it — the ordinary payload still caches.

        ``test_ttl_cache_single_flights_the_scan`` above proves this over the
        real disk; this proves it at the builder boundary, so a ``cache_ok``
        predicate that accidentally rejected everything would fail HERE with a
        call count rather than silently turning every dashboard poll into a
        full artifact walk.
        """
        import dashboard.app as app_module
        from dashboard.app import _memory_evals_cache_clear

        client.app.state.config = _make_config(tmp_path)
        _memory_evals_cache_clear()
        # SYNC, not AsyncMock: the route calls the builder through
        # asyncio.to_thread, which hands it a plain callable.
        build = MagicMock(return_value=_payload())
        monkeypatch.setattr(app_module, 'build_memory_evals', build)

        assert client.get(_ROUTE_URL).status_code == 200
        assert client.get(_ROUTE_URL).status_code == 200

        assert build.call_count == 1, f'expected 1 build call, got {build.call_count}'

    def test_an_absent_root_is_not_pinned_for_the_ttl(
        self, client, monkeypatch, tmp_path: Path,
    ) -> None:
        """``root_present: False`` is re-checked every poll.

        A root that does not exist yet is the pre-deployment state AND the
        transient one (an unmounted volume, a tree being created).  Caching it
        means the dashboard keeps reporting "no evals have ever run" for a full
        TTL window after the tree lands.  Re-checking costs one ``is_dir()``.
        """
        import dashboard.app as app_module
        from dashboard.app import _memory_evals_cache_clear

        client.app.state.config = _make_config(tmp_path)
        _memory_evals_cache_clear()
        build = MagicMock(return_value=_payload(root_present=False))
        monkeypatch.setattr(app_module, 'build_memory_evals', build)

        r1 = client.get(_ROUTE_URL)
        r2 = client.get(_ROUTE_URL)

        assert r1.status_code == r2.status_code == 200
        assert r2.json()['MEMORY_EVALS']['root_present'] is False
        assert build.call_count == 2, f'expected 2 build calls, got {build.call_count}'

    def test_an_unreadable_root_is_not_pinned_for_the_ttl(
        self, client, monkeypatch, tmp_path: Path,
    ) -> None:
        """The other never-walked payload: present, but the enumeration raised.

        Distinct from the absent case because ``root_present`` stays True — the
        signal is the ``unreadable_root`` issue, which is why the predicate
        reads both and not just the flag.
        """
        import dashboard.app as app_module
        from dashboard.app import _memory_evals_cache_clear

        client.app.state.config = _make_config(tmp_path)
        _memory_evals_cache_clear()
        build = MagicMock(return_value=_payload(issues=[{
            'kind': 'unreadable_root',
            'path': str(tmp_path),
            'detail': '[Errno 13] Permission denied',
        }]))
        monkeypatch.setattr(app_module, 'build_memory_evals', build)

        r1 = client.get(_ROUTE_URL)
        r2 = client.get(_ROUTE_URL)

        assert r1.status_code == r2.status_code == 200
        assert build.call_count == 2, f'expected 2 build calls, got {build.call_count}'

    def test_a_degraded_row_is_still_cached(
        self, client, monkeypatch, tmp_path: Path,
    ) -> None:
        """A corrupt metrics file must not defeat the cache in front of the walk.

        The failure this pins is a perf cliff, not a wrong answer: gate the
        cache on ``issue_count`` instead of on the root and one permanently
        broken artifact re-runs the full scan on every poll, forever.
        """
        import dashboard.app as app_module
        from dashboard.app import _memory_evals_cache_clear

        client.app.state.config = _make_config(tmp_path)
        _memory_evals_cache_clear()
        build = MagicMock(return_value=_payload(issues=[
            {'kind': 'unreadable_metrics', 'path': str(tmp_path), 'detail': 'boom'},
        ]))
        monkeypatch.setattr(app_module, 'build_memory_evals', build)

        assert client.get(_ROUTE_URL).status_code == 200
        r2 = client.get(_ROUTE_URL)

        assert r2.json()['MEMORY_EVALS']['issue_count'] == 1
        assert build.call_count == 1, f'expected 1 build call, got {build.call_count}'

    def test_cache_clear_hook_is_exported(self) -> None:
        """The test hook is part of the module's published surface, like the analytics one."""
        import dashboard.app as app_module

        assert callable(app_module._memory_evals_cache_clear)
        assert '_memory_evals_cache_clear' in app_module.__all__

    def test_missing_root_is_a_200_not_a_500(self, client, tmp_path: Path) -> None:
        """A project that has never run an eval renders an empty section, not an error."""
        from dashboard.app import _memory_evals_cache_clear

        client.app.state.config = _make_config(tmp_path)
        _memory_evals_cache_clear()

        resp = client.get(_ROUTE_URL)

        assert resp.status_code == 200
        payload = resp.json()['MEMORY_EVALS']
        assert payload['root_present'] is False
        assert payload['evals'] == []
        assert payload['issues'] == []


# ---------------------------------------------------------------------------
# step-21 — the consumer-side boundary test over 3207's committed exemplars
# ---------------------------------------------------------------------------

# The newest stamp anywhere in the committed tree (e1-dual-tripwire's second
# run), injected so staleness is measured against the artifacts rather than
# against the wall clock.
_EXEMPLAR_NOW = datetime(2026, 8, 2, 3, 15, 0, tzinfo=UTC)

_EXEMPLAR_EVAL_IDS = ['e1-dual-tripwire', 'e1-retrieval-health', 'e1-thin', 'malformed']

_RH_RUNS = [
    '20260701T031500Z', '20260702T031500Z', '20260703T031500Z',
    '20260704T031500Z', '20260705T031500Z',
]


def _exemplar_payload(tmp_path: Path) -> dict:
    """``build_memory_evals`` over the REAL committed tree, read-only.

    The escalations dir is an empty tmp path: 3207's fixtures are artifacts
    only, and the join's own behaviour is pinned by ``TestEscalationJoin``.
    """
    from dashboard.data.memory_evals import build_memory_evals

    return build_memory_evals(_FIXTURE_ROOT, tmp_path / 'escalations', now=_EXEMPLAR_NOW)


def _by_eval(payload: dict) -> dict[str, dict]:
    return {row['eval_id']: row for row in payload['evals']}


class TestCommittedExemplarBoundary:
    """The dashboard-shaped reader the memory-eval PRD's M1 promised, delivered.

    Everything below runs the PRODUCTION entry point against 3207's committed
    bytes under ``shared/tests/fixtures/memory_eval/`` — the only test in this
    file that reads artifacts it did not write.  That is the point: the tmp_path
    trees above agree with this file's own idea of the format, and only this
    class can catch the two drifting apart.

    Read-only by construction; the fixtures are committed and shared with the
    producer's suite.
    """

    def test_every_committed_eval_dir_appears_with_no_per_eval_code(self, tmp_path: Path) -> None:
        """Generic directory enumeration (DD5) — eval ids are data, never code.

        The root ``README.md`` is a file, not a dir, and must not be mistaken
        for an eval.
        """
        payload = _exemplar_payload(tmp_path)

        assert [row['eval_id'] for row in payload['evals']] == _EXEMPLAR_EVAL_IDS
        assert payload['root_present'] is True
        assert payload['unmatched_escalations'] == []

    def test_the_reader_never_writes_to_the_committed_tree(self, tmp_path: Path) -> None:
        """A consumer of a shared fixture tree must leave it byte-identical."""
        before = {p: p.stat().st_mtime_ns for p in sorted(_FIXTURE_ROOT.rglob('*')) if p.is_file()}

        _exemplar_payload(tmp_path)

        after = {p: p.stat().st_mtime_ns for p in sorted(_FIXTURE_ROOT.rglob('*')) if p.is_file()}
        assert after == before

    def test_retrieval_health_series_is_read_from_the_committed_bytes(self, tmp_path: Path) -> None:
        row = _by_eval(_exemplar_payload(tmp_path))['e1-retrieval-health']

        assert row['run_count'] == 5
        assert row['runs_on_disk'] == 5
        assert row['truncated'] is False
        assert row['run_stamps'] == _RH_RUNS
        assert row['latest_run_stamp'] == '20260705T031500Z'

        by_id = {m['metric_id']: m for m in row['metrics']}
        assert set(by_id) == {
            'canonical-in-top-5', 'dangling-pointers', 'superseded-above-successor',
            'topic-canonical-present', 'search-latency-p50-ms',
        }
        for metric_id, metric in by_id.items():
            assert metric['trend']['labels'] == _RH_RUNS, metric_id
            assert len(metric['trend']['values']) == 5, metric_id

        # The committed series, verbatim — including the 0704 regression spike
        # and the partial 0705 recovery the limits artifact alarms on.
        assert by_id['canonical-in-top-5']['trend']['values'] == [
            0.8, 0.8, 0.8, 0.4, 0.6666666666666666,
        ]
        assert by_id['dangling-pointers']['trend']['values'] == [4.0, 5.0, 6.0, 20.0, 8.0]
        assert by_id['search-latency-p50-ms']['trend']['values'] == [41.5, 39.0, 43.25, 44.0, 40.0]

        # Scalars come from the NEWEST run (0705), not the alarmed 0704 one.
        assert by_id['canonical-in-top-5']['current_value'] == 0.6666666666666666
        assert by_id['dangling-pointers']['current_value'] == 8.0
        assert by_id['search-latency-p50-ms']['current_value'] == 40.0
        assert by_id['canonical-in-top-5']['kind'] == 'proportion'
        assert by_id['canonical-in-top-5']['denominator'] == 30
        assert by_id['canonical-in-top-5']['direction'] == 'lower_is_worse'
        assert by_id['search-latency-p50-ms']['kind'] == 'scalar'

        assert row['corpus']['project_id'] == 'dark_factory'
        assert row['corpus']['counts']['entities_and_relations'] == 1204

    def test_committed_limits_provenance_passes_through_verbatim(self, tmp_path: Path) -> None:
        """The provenance block reproduces the committed artifact, value for value.

        These numbers are READ, never computed: the dashboard does no statistics
        (G6/INV-5), so an alpha that disagrees with the artifact could only come
        from a re-derivation that does not belong here.
        """
        limits = _by_eval(_exemplar_payload(tmp_path))['e1-retrieval-health']['limits']

        assert limits['alpha'] == 0.002777777777777778
        assert limits['false_alarm_budget'] == 1.0
        assert limits['runs_per_quarter'] == 90
        assert limits['min_samples'] == 10
        assert limits['baseline_window'] == 3
        assert limits['baseline_run_stamps'] == _RH_RUNS[:3]
        assert limits['grandfather_set_hash'] == (
            'f8c46981970a2cc2265806d08e9705ecceb66889e30476dcba0921a45a05dec5'
        )
        assert limits['run_stamp'] == '20260704T031500Z'
        assert limits['generator'] == 'shared.memory_eval_limits'

        # The committed exemplar exhibits the skew itself: limits stamped at the
        # 0704 run, newest metrics at 0705.  Displaying that provenance beside a
        # newer current value without disclosing it would present stale limits
        # as though they governed the displayed run.
        assert limits['stale_for_latest_run'] is True

        # Whitelist, not passthrough: the artifact's own alarm vocabulary stays
        # out of the payload entirely (it is not the M2 verdict vocabulary and
        # carries no fingerprint, so it cannot join an escalation).
        assert set(limits) == {
            'alpha', 'false_alarm_budget', 'runs_per_quarter', 'min_samples',
            'baseline_window', 'baseline_run_stamps', 'grandfather_set_hash',
            'run_stamp', 'generator', 'stale_for_latest_run',
        }

    def test_rule_kind_comes_from_the_committed_limits_artifact(self, tmp_path: Path) -> None:
        row = _by_eval(_exemplar_payload(tmp_path))['e1-retrieval-health']

        assert {m['metric_id']: m['rule_kind'] for m in row['metrics']} == {
            'canonical-in-top-5': 'proportion',
            'dangling-pointers': 'count',
            'superseded-above-successor': 'count',
            'topic-canonical-present': 'tripwire',
            'search-latency-p50-ms': 'scalar',
        }

    def test_thin_and_dual_tripwire_parse_cleanly(self, tmp_path: Path) -> None:
        """The two shape edge cases 3207 committed: a one-run eval and a two-tripwire eval.

        ``e1-dual-tripwire``'s two metrics share the item key ``t-shared`` — a
        producer-side fixture property (it exists to exercise per-item
        grandfathering), not something this payload carries: the dashboard reads
        metric-level values, not per-item rows.
        """
        evals = _by_eval(_exemplar_payload(tmp_path))

        thin = evals['e1-thin']
        assert thin['run_count'] == 1
        assert thin['run_stamps'] == ['20260704T031500Z']
        thin_by_id = {m['metric_id']: m for m in thin['metrics']}
        assert set(thin_by_id) == {'canonical-in-top-5', 'dangling-pointers'}
        assert all(m['n'] == 6 for m in thin['metrics'])
        assert thin_by_id['canonical-in-top-5']['trend']['values'] == [0.6666666666666666]

        dual = evals['e1-dual-tripwire']
        assert dual['run_count'] == 2
        assert dual['run_stamps'] == ['20260801T031500Z', '20260802T031500Z']
        dual_by_id = {m['metric_id']: m for m in dual['metrics']}
        assert set(dual_by_id) == {'topic-canonical-present', 'successor-pointer-present'}
        assert all(m['kind'] == 'tripwire' for m in dual['metrics'])
        assert dual_by_id['topic-canonical-present']['trend']['values'] == [2.0, 2.0]

        # Staleness, measured against the artifacts: the freshest eval is current,
        # the month-old one is displayed as stale.  Displayed, never alarmed on.
        assert dual['latest_run_age_seconds'] == 0.0
        assert dual['stale'] is False
        assert evals['e1-retrieval-health']['stale'] is True

    def test_missing_limits_and_verdicts_are_named_for_this_tree(self, tmp_path: Path) -> None:
        """3207's scope is M1 metrics + M2 limits; verdicts are 3211's (gamma re-verifies).

        Their absence is therefore expected here — and is exactly the kind of
        gap that must be NAMED rather than rendered as a blank column.
        """
        payload = _exemplar_payload(tmp_path)

        missing_limits = {i['eval_id'] for i in payload['issues'] if i['kind'] == 'missing_limits'}
        assert missing_limits == {'e1-dual-tripwire', 'e1-thin', 'malformed'}
        assert _by_eval(payload)['e1-retrieval-health']['limits'] is not None
        for eval_id in missing_limits:
            assert _by_eval(payload)[eval_id]['limits'] is None

        assert len([i for i in payload['issues'] if i['kind'] == 'missing_verdicts']) == 1

        # No verdicts artifact means no verdict — never a defaulted "no_alarm",
        # and never the healthy ``clear`` badge either: nothing judged these.
        for row in payload['evals']:
            for metric in row['metrics']:
                assert metric['verdict'] is None, (row['eval_id'], metric['metric_id'])
                assert metric['escalation'] is None
                assert metric['parity'] == 'unjudged'
            assert row['storm_escape'] is None

    def test_negative_exemplars_split_rendering_from_producer_validation(self, tmp_path: Path) -> None:
        """The asymmetry the design decision pins, over the two committed negatives.

        ``metrics-bad-kind.json``'s ``histogram`` is a RENDERING failure — there
        is no chart primitive for a kind outside the closed vocabulary — so the
        dashboard says so.  ``metrics-proportion-out-of-range.json``'s 1.4 is a
        PRODUCER-side schema rule (M1 rejects it at emit time, not read time);
        restating it here would be a second implementation of a decision the
        producer already owns, so the value passes through verbatim.
        """
        payload = _exemplar_payload(tmp_path)
        row = _by_eval(payload)['malformed']

        # Both files parse: they are semantically malformed, not syntactically.
        assert row['run_count'] == 2
        assert [i['kind'] for i in payload['issues'] if i['eval_id'] == 'malformed'] == [
            'missing_limits', 'unknown_kind',
        ]

        unknown = next(i for i in payload['issues'] if i['kind'] == 'unknown_kind')
        assert 'histogram' in unknown['detail']
        assert 'metrics-bad-kind.json' in unknown['path']

        # The out-of-range proportion: displayed, uncommented.
        metric = _only(row['metrics'], 'canonical-in-top-5')
        assert metric['current_value'] == 1.4
        assert metric['kind'] == 'proportion'

    def test_the_whole_committed_tree_yields_exactly_the_known_gaps(self, tmp_path: Path) -> None:
        """One count over the whole tree, so a new silent degradation cannot hide.

        Three evals without limits, one absent root verdicts artifact, one
        unrenderable kind — five, and nothing else.
        """
        payload = _exemplar_payload(tmp_path)

        assert payload['issue_count'] == len(payload['issues']) == 5
        assert sorted(i['kind'] for i in payload['issues']) == [
            'missing_limits', 'missing_limits', 'missing_limits',
            'missing_verdicts', 'unknown_kind',
        ]

    def test_the_reader_never_touches_the_producer(self) -> None:
        """G6/INV-5 — artifacts only, never the module; and no statistics anywhere.

        Scoped to PRODUCTION code: this asserts over
        ``dashboard/data/memory_evals.py`` only.  A version that also parsed
        this test file's own AST could never catch a regression — an editor
        adding a ``shared`` import to this test would simply be editing the
        assertion that forbids it.  The artifact-only boundary of the *tests*
        is a review property, stated in the module docstring, not something a
        test can police about itself.

        Checked over the AST rather than the raw text because a text grep
        matches prose as readily as code — this assertion's own literals, and
        any future docstring that has to *name* a forbidden symbol in order to
        record that it must never appear.  The AST sees code, which is what the
        sidecar's ``expect: absent`` grep over ``dashboard/src/dashboard/data/``
        is actually about.

        The ban is on the IMPORT, not only on attribute access, because
        ``from math import comb`` leaves no ``math.`` prefix for the
        attribute check below to find.  That is also why ``math`` is banned
        whole rather than narrowed to its statistical surface: this reader
        renders values it has already been handed, so it has no established
        need for any of ``math``, and a ban with no exceptions cannot be
        eroded one benign-looking import at a time.  If a real need appears
        (``math.isnan`` on a NaN value, say), narrowing this to
        ``{'statistics'}`` is the deliberate decision to make then — with the
        attribute check still standing guard over the re-derivation risk.
        """
        import ast

        import dashboard.data.memory_evals as memory_evals_module

        reader_path = Path(memory_evals_module.__file__)
        reader_imports: set[str] = set()
        for node in ast.walk(ast.parse(reader_path.read_text())):
            if isinstance(node, ast.Import):
                reader_imports.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                reader_imports.add(node.module)

        # The on-disk artifact IS the contract; importing the producer would
        # couple the dashboard to its in-memory objects instead.
        assert not [m for m in reader_imports if m.split('.')[0] == 'shared'], sorted(reader_imports)
        # Verdicts are READ, never re-derived — so no stdlib stats surface.
        assert not reader_imports & {'math', 'statistics'}, sorted(reader_imports)

        # And no statistics by attribute access either — verdicts are read,
        # never re-derived (INV-1: the same file the evaluator read).
        stats_calls = [
            f'{node.value.id}.{node.attr}'
            for node in ast.walk(ast.parse(reader_path.read_text()))
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in {'math', 'statistics'}
        ]
        assert stats_calls == []
