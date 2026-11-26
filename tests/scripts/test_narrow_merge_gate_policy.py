"""The LIVE narrow merge gate policy, pinned where it is declared: dark-factory-orchestrator.yaml.

Task 5411; the rationale is docs/narrow-merge-gate.md. The gate is on, and its
docs allowlist keeps the ruled boundary. That boundary is R4 (2026-09-11) as
amended on 2026-10-09 and 2026-11-02: prose is docs, while yaml, toml, json,
shell scripts and prompts are not, because tests read them. The one admitted
yaml is the stamped capability-manifest sidecar. The allowlist must also never
shadow a task-2838 full-gate glob.

The classification goes through the production classifier,
orchestrator/src/orchestrator/narrow_gate_class.py::classify_docs_change, so this
guard tests the policy and not a re-implementation of the glob rules.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from orchestrator.narrow_gate_class import ChangedPath, MergeGateResolution, classify_docs_change
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

if TYPE_CHECKING:
    from orchestrator.config import OrchestratorConfig

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

PROSE = (
    'README.md',
    'CLAUDE.md',
    'docs/x/y.md',
    'plans/p-prd.md',
    'plans/p-prd.capability-manifest.yaml',
    'plans/p-prd.capability-manifest.md',
    'skills/a/SKILL.md',
)

NOT_PROSE = (
    'dark-factory-orchestrator.yaml',
    'plans/p.yaml',
    'plans/p-prd.evidence/seat.md',
    'skills/spawn/spawn-claude.sh',
    'fused-memory/src/fused_memory/reconciliation/prompts/stage1.md',
    'orchestrator/src/orchestrator/defaults.yaml',
    'pyproject.toml',
    'uv.lock',
    'package.json',
    '.claude/settings.json',
)


def _gate_for_added(path: str, config: OrchestratorConfig) -> MergeGateResolution:
    added = ChangedPath(path, 'A', '000000', '100644')
    return classify_docs_change([added], config.narrow_merge_gate.docs_paths).resolution


def test_the_narrow_merge_gate_is_on(root_config: OrchestratorConfig) -> None:
    assert root_config.narrow_merge_gate.enabled is True


@pytest.mark.parametrize('path', PROSE)
def test_a_prose_file_takes_the_docs_gate(root_config: OrchestratorConfig, path: str) -> None:
    assert _gate_for_added(path, root_config) is MergeGateResolution.DOCS


@pytest.mark.parametrize('path', NOT_PROSE)
def test_a_file_tests_read_takes_the_full_gate(root_config: OrchestratorConfig, path: str) -> None:
    assert _gate_for_added(path, root_config) is MergeGateResolution.FULL


def test_no_2838_full_gate_glob_is_shadowed_by_the_docs_allowlist(root_config: OrchestratorConfig) -> None:
    shadowed = [
        glob for glob in root_config.git.merge_config_only_full_gate_globs
        if _gate_for_added(glob, root_config) is not MergeGateResolution.FULL
    ]

    assert shadowed == []
