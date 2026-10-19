"""EvalMetrics' pointer to its shadow cell (plans/live-shadow-eval-prd.md C1/C4)."""

from __future__ import annotations

import json

import pytest

from orchestrator.evals.metrics import EvalMetrics
from orchestrator.evals.rereview import recompute_composite_from_metrics
from orchestrator.shadow_types import ReferenceKind

_LEGACY_METRICS = {
    'tests_pass': True,
    'lint_clean': True,
    'typecheck_clean': False,
    'plan_completion_pct': 0.75,
    'plan_steps': 4,
    'cost_usd': 1.25,
    'workflow_duration_ms': 90_000,
    'review_blocking_issues': 1,
    'review_suggestions': 2,
    'composite_score': 0.0,
    'legacy_sonnet_review_blocking_issues': 0,
}


def _known_fields(metrics: dict) -> dict:
    return {k: v for k, v in metrics.items() if k in EvalMetrics.__dataclass_fields__}


def _read_back(persisted: dict) -> EvalMetrics:
    return EvalMetrics(**json.loads(json.dumps(persisted)))


def test_defaults_are_none() -> None:
    metrics = EvalMetrics()
    assert metrics.shadow_cell_id is None
    assert metrics.reference_kind is None


def test_to_dict_carries_both_keys_unconditionally() -> None:
    as_dict = EvalMetrics().to_dict()
    assert as_dict['shadow_cell_id'] is None
    assert as_dict['reference_kind'] is None


def test_reference_kind_read_back_from_json_is_coerced() -> None:
    assert _read_back({'reference_kind': 'landed'}).reference_kind is ReferenceKind.LANDED


def test_unknown_reference_kind_is_rejected() -> None:
    with pytest.raises(ValueError, match='merged'):
        _read_back({'reference_kind': 'merged'})


def test_json_round_trip_gives_bare_strings() -> None:
    metrics = EvalMetrics(shadow_cell_id='c1', reference_kind=ReferenceKind.NONE)
    read_back = json.loads(json.dumps(metrics.to_dict()))
    assert read_back['shadow_cell_id'] == 'c1'
    assert read_back['reference_kind'] == 'none'
    assert type(read_back['reference_kind']) is str


def test_legacy_metrics_without_the_keys_rescore_unchanged() -> None:
    shadowed = {**_LEGACY_METRICS, 'shadow_cell_id': 'c1', 'reference_kind': 'landed'}

    composite = recompute_composite_from_metrics(dict(_LEGACY_METRICS))

    assert composite == recompute_composite_from_metrics(shadowed)
    assert composite > 0.0
    legacy = EvalMetrics(**_known_fields(_LEGACY_METRICS))
    assert legacy.shadow_cell_id is None
    assert legacy.reference_kind is None
