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
