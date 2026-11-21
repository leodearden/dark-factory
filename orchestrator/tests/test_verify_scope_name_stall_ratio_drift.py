"""scripts/cgroup-stall-ratio.py classes the verify scopes verify.py actually names.

The script is stdlib-only, so it re-encodes the unit-name shape owned by
``verify.py::_verify_scope_name`` and ``verify.py::_scope_tag_for`` instead of
importing it. Here, where orchestrator is importable, the script's parser reads
names built by that producer. If the shape moves, a project's verify worker
falls into the script's 'other' bucket, and the per-project classes that task
5208's AFTER gate compares read null.
"""
from __future__ import annotations

import collections
import importlib.util
from pathlib import Path
from typing import Any

import pytest

from orchestrator.verify import _scope_tag_for, _verify_scope_name

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts' / 'cgroup-stall-ratio.py'
APP_SLICE = '0::/user.slice/user-1000.slice/user@1000.service/app.slice'
MERGE_NICE = 5


def _load_stall_ratio() -> Any:
    spec = importlib.util.spec_from_file_location('cgroup_stall_ratio', SCRIPT)
    assert spec is not None and spec.loader is not None, f'Could not build spec from {SCRIPT}'
    module: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stall_ratio = _load_stall_ratio()


@pytest.mark.parametrize(
    ('project_root', 'bucket_label', 'merge_family', 'merge_class'),
    [
        (Path('/x/dark-factory'), 'verify-scope:dark-factory', 'xdist-worker', 'df_merge_workers'),
        (Path('/x/reify'), 'verify-scope:reify', 'rustc', 'reify_merge_verify'),
    ],
    ids=['dark-factory', 'reify'],
)
def test_a_merge_verify_in_a_produced_scope_lands_in_its_project_class(
    project_root: Path, bucket_label: str, merge_family: str, merge_class: str,
) -> None:
    scope_cgroup = f'{APP_SLICE}/{_verify_scope_name(_scope_tag_for(project_root))}'
    scope_bucket = stall_ratio.bucket(scope_cgroup)
    one_second = collections.Counter({(scope_bucket, merge_family, MERGE_NICE): 1_000_000_000})

    headline = stall_ratio.headline(one_second, one_second, 1.0, '50.00', '100')

    assert scope_bucket == bucket_label, scope_cgroup
    assert headline[merge_class]['stall_pct'] == 50, scope_cgroup
