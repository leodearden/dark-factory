"""Tests for scripts/cgroup-stall-ratio.py: which headline class a process lands in.

Task 5208's AFTER gate compares these classes against the BEFORE samples task
5206 recorded, so a dark_factory verify worker must land in the same df_* class
whether it runs inside orchestrator-dark-factory.service or in its own scope.
The script is stdlib-only and is loaded by path, without orchestrator importable.
"""
from __future__ import annotations

import collections
import importlib.util
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).parents[2]
SCRIPT = REPO_ROOT / 'scripts' / 'cgroup-stall-ratio.py'

APP_SLICE = '0::/user.slice/user-1000.slice/user@1000.service/app.slice'
# Scope unit-name shape owner: orchestrator/src/orchestrator/verify.py::_verify_scope_name / ::_scope_tag_for.
DF_SCOPE = f'{APP_SLICE}/df-verify-dark-factory-c889f090-a3825b3fefa0.scope'
REIFY_SCOPE = f'{APP_SLICE}/df-verify-reify-4ae45bbd-d3b5fc367ef0.scope'
HEX_SLUG_SCOPE = f'{APP_SLICE}/df-verify-proj-deadbeef-0123abcd-d3b5fc367ef0.scope'
DF_UNIT = f'{APP_SLICE}/orchestrator-dark-factory.service'

ONE_CORE_SECOND_NS = 1_000_000_000
EVEN_SPLIT = {'run_cores': 1.0, 'wait_cores': 1.0, 'stall_pct': 50}

CLASS_KEYS = (
    'df_agents', 'df_merge_workers', 'df_task_workers', 'df_background_workers',
    'reify_agents', 'reify_merge_verify', 'reify_task_verify',
)
HEADLINE_KEYS = {'psi_cpu_some_avg10', 'procs_running', 'window_s', *CLASS_KEYS}


def _load_script() -> Any:
    spec = importlib.util.spec_from_file_location('cgroup_stall_ratio', SCRIPT)
    assert spec is not None and spec.loader is not None, f'Could not build spec from {SCRIPT}'
    module: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stall_ratio = _load_script()


def _headline(*keys: tuple[str, str, int]) -> dict:
    run = collections.Counter({key: ONE_CORE_SECOND_NS for key in keys})
    wait = collections.Counter(run)
    return stall_ratio.headline(run, wait, 1.0, '50.00', '100')


def _populated(headline: dict) -> set[str]:
    return {name for name in CLASS_KEYS if headline[name]['stall_pct'] is not None}


@pytest.mark.parametrize(
    ('cgroup', 'label'),
    [
        (DF_SCOPE, 'verify-scope:dark-factory'),
        (REIFY_SCOPE, 'verify-scope:reify'),
        (HEX_SLUG_SCOPE, 'verify-scope:proj-deadbeef'),
        (DF_UNIT, 'df'),
    ],
    ids=['df-scope', 'reify-scope', 'hex-slug-scope', 'df-unit'],
)
def test_bucket_labels_a_verify_scope_by_its_project_slug(cgroup: str, label: str) -> None:
    assert stall_ratio.bucket(cgroup) == label


@pytest.mark.parametrize('cgroup', [DF_SCOPE, DF_UNIT], ids=['scope', 'unit'])
@pytest.mark.parametrize(
    ('nice', 'expected_class'),
    [(5, 'df_merge_workers'), (15, 'df_task_workers'), (19, 'df_background_workers')],
)
def test_a_dark_factory_verify_worker_is_classed_by_nice_tier_in_unit_or_scope(
    cgroup: str, nice: int, expected_class: str,
) -> None:
    headline = _headline((stall_ratio.bucket(cgroup), 'xdist-worker', nice))

    assert headline[expected_class] == EVEN_SPLIT
    assert _populated(headline) == {expected_class}


def test_verify_scopes_of_two_projects_stay_in_their_own_classes() -> None:
    headline = _headline(
        (stall_ratio.bucket(DF_SCOPE), 'xdist-worker', 5),
        (stall_ratio.bucket(REIFY_SCOPE), 'rustc', 5),
    )

    assert headline['df_merge_workers'] == EVEN_SPLIT
    assert headline['reify_merge_verify'] == EVEN_SPLIT
    assert _populated(headline) == {'df_merge_workers', 'reify_merge_verify'}


def test_the_headline_key_set_is_the_one_the_after_gate_parses() -> None:
    assert set(_headline()) == HEADLINE_KEYS
