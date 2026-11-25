"""The shipped curator registries load and decide when resolved as the deployed unit resolves them.

The deployed fused-memory unit (scripts/fused-memory.service.template) runs
with CONFIG_PATH at the shipped fused-memory/config/config.yaml and
PROJECT_ROOT at the repo root, and the server builds its TaskCurator with
cwd = taskmaster.project_root. This module composes exactly that: the shipped
config, the template's PROJECT_ROOT and a real TaskCurator. It drives the
public seams, curate_prepared and registry_health, with an LLM tripwire, so a
registry that does not resolve fails the test instead of reaching the LLM.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _curator_helpers import escalator_double, must_not_invoke, prepared
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.middleware.curator_registry import RegistryState
from fused_memory.middleware.curator_types import CandidateTask
from fused_memory.middleware.task_curator import TaskCurator

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SERVICE_TEMPLATE = _REPO_ROOT / 'scripts' / 'fused-memory.service.template'
_PROJECT_ROOT_LINE = 'Environment=PROJECT_ROOT='
_REPO_ROOT_PLACEHOLDER = '__REPO_ROOT__'
_PROJECT_ID = 'dark_factory'


def _deployed_project_root() -> str:
    """The PROJECT_ROOT the rendered unit sets, with the repo-root placeholder filled in."""
    found = [
        line.strip()
        for line in _SERVICE_TEMPLATE.read_text().splitlines()
        if line.strip().startswith(_PROJECT_ROOT_LINE)
    ]
    assert len(found) == 1, (
        f'{_SERVICE_TEMPLATE} must set PROJECT_ROOT exactly once; found {found!r}'
    )
    value = found[0].removeprefix(_PROJECT_ROOT_LINE)
    return value.replace(_REPO_ROOT_PLACEHOLDER, str(_REPO_ROOT))


def _deployed_curator(monkeypatch: pytest.MonkeyPatch) -> tuple[TaskCurator, Path]:
    monkeypatch.setenv('PROJECT_ROOT', _deployed_project_root())
    config = FusedMemoryConfig()
    # Mirrors fused-memory/src/fused_memory/middleware/task_interceptor.py::TaskInterceptor._get_curator.
    cwd = Path(config.taskmaster.project_root)
    curator = TaskCurator(
        config=config, taskmaster=None, cwd=cwd,
        escalator=escalator_double(), invoke=must_not_invoke(),
    )
    return curator, cwd


_OPERATIONAL_ASK = CandidateTask(
    title=(
        'Operational: rerun prune_recon_cycle_summaries.py --project-id '
        'know_live (dry-run then --apply)'
    ),
    description=(
        'Collapse the pre-existing untagged Stage 1 cycle-summary pile: '
        'dry-run first, review, then --apply against live Mem0/Qdrant.'
    ),
)

_GUARD_HITS: dict[str, tuple[CandidateTask, str, str]] = {
    'cancelled-premise-blocklist': (
        CandidateTask(
            title='Convert FIX C relay-flag deletion in task_knowledge_sync.py: search-then-delete',
            description=(
                'The metric fixc_flags_deleted_not_found shows that delete-by-id fails '
                'when the id is not in the store. Rework to search-then-delete.'
            ),
        ),
        'refuse',
        'cancelled-premise-blocklist: fixc_flag_marker_search_then_delete:',
    ),
    'recon-premise-refuted': (
        CandidateTask(
            title='Fix entity-summary rebuild missing invalid_at filter',
            description='Rebuild does not check missing invalid_at filter before writing.',
        ),
        'refuse',
        'recon-premise-refuted: entity_summary_rebuild_invalid_at_filter_already_present:',
    ),
    'operational-ask-registry': (
        _OPERATIONAL_ASK,
        'route_deterministic',
        'operational-ask-registry: prune_recon_cycle_summaries_apply:',
    ),
}


@pytest.mark.asyncio
@pytest.mark.parametrize('guard', sorted(_GUARD_HITS))
async def test_each_shipped_guard_decides_under_the_deployed_root(guard, monkeypatch):
    candidate, expected_action, expected_prefix = _GUARD_HITS[guard]
    curator, cwd = _deployed_curator(monkeypatch)

    [decision] = await curator.curate_prepared(
        [prepared(candidate.title, candidate=candidate)], _PROJECT_ID, str(cwd),
    )

    assert decision.action == expected_action, f'{guard}: got {decision!r}'
    assert decision.justification.startswith(expected_prefix), f'{guard}: got {decision!r}'


@pytest.mark.asyncio
async def test_all_three_registries_load_under_the_deployed_root(monkeypatch):
    """Pre-merge equivalent of esc-task-curator-37's post-restart "all three registries load"."""
    curator, cwd = _deployed_curator(monkeypatch)

    await curator.curate_prepared(
        [prepared(_OPERATIONAL_ASK.title, candidate=_OPERATIONAL_ASK)], _PROJECT_ID, str(cwd),
    )

    health = curator.registry_health()
    wire = [h.to_wire() for h in health]
    assert len(health) == 3, wire
    for h in health:
        assert h.state is RegistryState.LOADED, wire
        assert h.entry_count is not None and h.entry_count >= 1, wire
