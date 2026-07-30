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
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

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
    corpus: dict | None = None,
    filename: str | None = None,
    run_stamp: Any = _UNSET,  # default: the in-body stamp == *stamp*
) -> Path:
    """Write ``<root>/<eval_id>/metrics-<stamp>.json``.

    *filename* overrides the whole basename (for the stamp-less names the
    committed ``malformed/`` dir carries).  *run_stamp* overrides the in-body
    stamp; pass ``None`` explicitly to omit it entirely (the
    ``missing_run_stamp`` degraded case).
    """
    body: dict[str, Any] = {
        'schema_version': 1,
        'eval_id': eval_id,
        'corpus': dict(corpus) if corpus is not None else dict(_CORPUS),
        'metrics': list(metrics),
    }
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
        fused-memory runtime dirs (config.py:156) to an XDG-rooted path outside
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

    def test_cap_is_ninety_runs(self) -> None:
        """One screen of trend == one alpha-derivation window.

        90 is the PRD's declared lean for open question 2 and matches
        ``runs_per_quarter=90`` in the committed limits artifact, so the
        displayed window is exactly the window the limits govern.
        """
        import dashboard.data.memory_evals as memory_evals_module

        assert memory_evals_module._TREND_RUN_CAP == 90


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

# The escalation projection carried onto a metric row (and into
# `unmatched_escalations`).  `created_at` is sourced from the queue record's
# `timestamp` — that is the field `escalation.models.Escalation` serialises.
_ESCALATION_KEYS = {'id', 'summary', 'severity', 'level', 'created_at', 'dedupe_fingerprint'}

_JOIN_ESC_TIMESTAMP = '2026-07-30T03:15:00+00:00'


def _join_tree(tmp_path: Path) -> tuple[Path, Path]:
    """One eval whose five metrics span every parity state alpha owns."""
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
        )
    ])
    _write_limits(root, 'eval-a', run_stamp=_JOIN_RUN)
    _write_verdicts(root, [
        _verdict('eval-a', 'alarmed-open', 'alarm', fingerprint=_FP_ALARMED_OPEN, run_stamp=_JOIN_RUN),
        _verdict('eval-a', 'recovered-open', 'no_alarm', fingerprint=_FP_RECOVERED_OPEN, run_stamp=_JOIN_RUN),
        _verdict('eval-a', 'alarmed-unlinked', 'alarm', fingerprint=_FP_ALARMED_UNLINKED, run_stamp=_JOIN_RUN),
        _verdict('eval-a', 'clear-metric', 'no_alarm', fingerprint=_FP_CLEAR, run_stamp=_JOIN_RUN),
        _verdict('eval-a', 'wrong-category', 'alarm', fingerprint=_FP_WRONG_CATEGORY, run_stamp=_JOIN_RUN),
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
        """
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _join_tree(tmp_path)

        row = _only(build_memory_evals(root, esc_dir)['evals'][0]['metrics'], 'recovered-open')

        assert row['verdict'] == 'no_alarm'
        assert row['parity'] == 'recovered_open'
        assert row['escalation']['id'] == 'esc-recovered-open'

    def test_alarm_with_no_escalation_is_flagged_unlinked(self, tmp_path: Path) -> None:
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _join_tree(tmp_path)

        row = _only(build_memory_evals(root, esc_dir)['evals'][0]['metrics'], 'alarmed-unlinked')

        assert row['escalation'] is None
        assert row['parity'] == 'alarmed_unlinked'

    def test_quiet_metrics_are_clear(self, tmp_path: Path) -> None:
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _join_tree(tmp_path)
        rows = build_memory_evals(root, esc_dir)['evals'][0]['metrics']

        assert _only(rows, 'clear-metric')['parity'] == 'clear'
        assert _only(rows, 'clear-metric')['escalation'] is None
        # No verdict at all is also not an alarm.
        assert _only(rows, 'unjudged-metric')['parity'] == 'clear'

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
        assert set(unmatched[0]) == _ESCALATION_KEYS
        assert unmatched[0]['dedupe_fingerprint'] == _FP_UNMATCHED


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

        storm = build_memory_evals(root, esc_dir)['evals'][0]['storm_escape']

        assert set(storm) == _STORM_KEYS
        assert storm['triggered'] is True
        assert storm['alarm_count'] == 2
        assert storm['aggregate_fingerprint'] == _FP_AGGREGATE
        assert storm['escalation']['id'] == 'esc-storm-aggregate'
        assert set(storm['escalation']) == _ESCALATION_KEYS

    def test_per_metric_links_collapse_into_the_aggregate(self, tmp_path: Path) -> None:
        """No per-metric links during a storm — the aggregate is the one alert."""
        from dashboard.data.memory_evals import build_memory_evals

        root, esc_dir = _storm_tree(
            tmp_path,
            storm={'triggered': True, 'alarm_count': 2, 'aggregate_fingerprint': _FP_AGGREGATE},
            per_metric_escalation=True,
        )

        rows = build_memory_evals(root, esc_dir)['evals'][0]['metrics']

        for metric_id in ('storm-a', 'storm-b'):
            row = _only(rows, metric_id)
            assert row['verdict'] == 'alarm'
            assert row['escalation'] is None
            assert row['parity'] == 'storm_collapsed'
        # A quiet metric is still quiet during a storm.
        assert _only(rows, 'quiet-metric')['parity'] == 'clear'

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

        body = redux_api.shape_memory_evals(
            generated_at='2026-07-30T03:15:00+00:00',
            root_present=True,
            evals=evals,
            issues=issues,
            issue_count=1,
            unmatched_escalations=unmatched,
        )['MEMORY_EVALS']

        evals.append({'eval_id': 'eval-b'})
        issues.append({'kind': 'orphan_verdict'})
        unmatched.append({'id': 'esc-2'})

        assert body['evals'] == [{'eval_id': 'eval-a'}]
        assert body['issues'] == [{'kind': 'missing_limits'}]
        assert body['unmatched_escalations'] == [{'id': 'esc-1'}]

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
