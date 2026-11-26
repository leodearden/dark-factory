"""The shipped curator registries load and decide when resolved as the deployed unit resolves them.

The deployed fused-memory unit (scripts/fused-memory.service.template) runs
with the template's CONFIG_PATH and PROJECT_ROOT, and its TaskInterceptor
builds the curator lazily, choosing the base the curator resolves relative
registry paths against. This module composes exactly that: the rendered
template's environment, the shipped config and a real TaskInterceptor that
builds its own curator. Only the curator's LLM invoker and Qdrant client are
swapped, for a tripwire and an in-memory store, so a registry that does not
resolve fails the test instead of reaching the LLM.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from _curator_helpers import escalator_double, must_not_invoke, prepared
from _fm_helpers import load_script_module
from qdrant_client import AsyncQdrantClient
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.middleware import task_interceptor
from fused_memory.middleware.curator_registry import RegistryState
from fused_memory.middleware.curator_types import CandidateTask
from fused_memory.middleware.task_curator import TaskCurator
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.reconciliation.event_buffer import EventBuffer

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / 'scripts'
_SERVICE_TEMPLATE = _SCRIPTS / 'fused-memory.service.template'
# Which config loads, and the base its relative registry paths resolve against.
_RESOLUTION_VARIABLES = ('CONFIG_PATH', 'PROJECT_ROOT')
_PROJECT_ID = 'dark_factory'


def _deployed_environment() -> dict[str, str]:
    """The [Service] environment of the unit setup-host.sh renders from the template for this checkout."""
    unit_parity = load_script_module(_SCRIPTS / 'systemd_unit_parity.py')
    renderer = load_script_module(_SCRIPTS / 'render_systemd_unit.py')
    rendered = renderer.render_template(
        _SERVICE_TEMPLATE.read_text(), repo_root=str(_REPO_ROOT), uv_path='uv',
    )
    return unit_parity.environment_map(
        unit_parity.parse_unit_directives(rendered), unit_parity.SERVICE_SECTION,
    )


@dataclass(frozen=True)
class _Deployed:
    interceptor: TaskInterceptor
    curator: TaskCurator
    project_root: str


@pytest_asyncio.fixture
async def deployed(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[_Deployed]:
    """A TaskInterceptor over the shipped config, and the curator it built for itself."""
    environment = _deployed_environment()
    for name in _RESOLUTION_VARIABLES:
        monkeypatch.setenv(name, environment[name])
    config = FusedMemoryConfig()
    assert config.taskmaster is not None, 'the shipped config.yaml must carry a taskmaster section'

    built: list[TaskCurator] = []

    def build_without_llm_or_live_qdrant(**kwargs: Any) -> TaskCurator:
        curator = TaskCurator(
            **kwargs, invoke=must_not_invoke(),
            qdrant_client=AsyncQdrantClient(location=':memory:'),
        )
        built.append(curator)
        return curator

    monkeypatch.setattr(task_interceptor, 'TaskCurator', build_without_llm_or_live_qdrant)
    interceptor = TaskInterceptor(
        None, None, EventBuffer(db_path=None), config=config, escalator=escalator_double(),
    )
    project_root = config.taskmaster.project_root
    try:
        # With no tasks to index, this public seam only builds the interceptor's lazy curator.
        await interceptor.index_committed_tasks([], project_root)
        assert len(built) == 1, 'the interceptor must build exactly one curator'
        yield _Deployed(interceptor=interceptor, curator=built[0], project_root=project_root)
    finally:
        await interceptor.close()


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
async def test_each_shipped_guard_decides_under_the_deployed_root(guard, deployed):
    candidate, expected_action, expected_prefix = _GUARD_HITS[guard]

    [decision] = await deployed.curator.curate_prepared(
        [prepared(candidate.title, candidate=candidate)], _PROJECT_ID, deployed.project_root,
    )

    assert decision.action == expected_action, f'{guard}: got {decision!r}'
    assert decision.justification.startswith(expected_prefix), f'{guard}: got {decision!r}'


@pytest.mark.asyncio
async def test_all_three_registries_load_under_the_deployed_root(deployed):
    """Pre-merge equivalent of esc-task-curator-37's post-restart "all three registries load".

    The startup self-check the interceptor schedules when it builds the curator
    is what loads each registry; drain() awaits it.
    """
    await deployed.interceptor.drain()

    health = deployed.interceptor.curator_registry_health()
    wire = [h.to_wire() for h in health]
    assert len(health) == 3, wire
    for h in health:
        assert h.state is RegistryState.LOADED, wire
        assert h.entry_count is not None and h.entry_count >= 1, wire
