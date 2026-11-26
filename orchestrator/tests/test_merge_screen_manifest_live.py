"""INV-10: every check in the shipped merge screen manifest still screens something.

Loads the SHIPPED config (conftest pins ORCH_CONFIG_PATH to
dark-factory-orchestrator.yaml) with this tree as its project root, so
``load_config`` discovers the real module registry, and runs the manifest
through the real node-id runner on this tree. A missing, renamed,
uncollectable or skipping node id reads ``error``, so this fails exactly when
a manifest check has silently stopped screening. A screen the shipped config
leaves disabled is a failure, never a skip.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest
import yaml
from _orch_helpers import git_env_with_ceiling
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

from orchestrator.config import load_config
from orchestrator.merge_lane.screen import run_screen

REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = [
    pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT),
    pytest.mark.xdist_group('merge_screen_manifest_live'),
]


def _head(tree: Path) -> str:
    return subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=tree, env=git_env_with_ceiling(tree),
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def test_every_shipped_manifest_check_passes_on_this_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('ORCH_PROJECT_ROOT', str(REPO_ROOT))
    config = load_config()
    manifest = config.merge_screen.manifest
    assert manifest is not None, (
        'the shipped dark-factory-orchestrator.yaml must enable the merge screen '
        '(merge_screen.manifest); a disabled screen is a failure here, not a skip'
    )
    shipped_ids = [entry['id'] for entry in yaml.safe_load((REPO_ROOT / manifest).read_text())['checks']]

    verdict = asyncio.run(run_screen(REPO_ROOT, config, tree_sha=_head(REPO_ROOT)))

    assert sorted(check.id for check in verdict.checks) == sorted(shipped_ids)
    not_passing = [
        f'{check.id}: {check.status} {list(check.findings or check.causes)}'
        for check in verdict.checks if check.status != 'pass'
    ]
    assert not_passing == [], 'manifest checks that no longer pass on this tree:\n' + '\n'.join(not_passing)
