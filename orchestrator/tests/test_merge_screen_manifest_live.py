"""INV-10: every check in the shipped merge screen manifest still screens something.

Loads the SHIPPED config (conftest pins ORCH_CONFIG_PATH to
dark-factory-orchestrator.yaml) with this tree as its project root, so
``load_config`` discovers the real module registry, and runs the manifest
through the real node-id runner on this tree. A missing, renamed,
uncollectable or skipping node id reads ``error``, so this fails exactly when
a manifest check has silently stopped screening. A ``finding`` is left to the
check's own module test, which reports that defect once. A screen the shipped
config leaves disabled is a failure, never a skip.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml
from _orch_helpers import VERIFY_CLI_PER_TEST_TIMEOUT
from _screen_fixtures import head

from orchestrator.config import load_config
from orchestrator.merge_lane.screen import run_screen

REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = [
    pytest.mark.timeout(VERIFY_CLI_PER_TEST_TIMEOUT),
    pytest.mark.xdist_group('merge_screen_manifest_live'),
]


def test_every_shipped_manifest_check_reaches_a_verdict_on_this_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('ORCH_PROJECT_ROOT', str(REPO_ROOT))
    config = load_config()
    manifest = config.merge_screen.manifest
    assert manifest is not None, (
        'the shipped dark-factory-orchestrator.yaml must enable the merge screen '
        '(merge_screen.manifest); a disabled screen is a failure here, not a skip'
    )
    shipped_ids = [entry['id'] for entry in yaml.safe_load((REPO_ROOT / manifest).read_text())['checks']]

    verdict = asyncio.run(run_screen(REPO_ROOT, config, tree_sha=head(REPO_ROOT)))

    assert sorted(check.id for check in verdict.checks) == sorted(shipped_ids)
    no_verdict = [f'{check.id}: {list(check.causes)}' for check in verdict.error_checks]
    assert no_verdict == [], 'manifest checks that no longer screen this tree:\n' + '\n'.join(no_verdict)
