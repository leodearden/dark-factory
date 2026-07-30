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

import json
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
