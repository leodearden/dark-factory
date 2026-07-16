"""Tests for Harness._tag_task_modules()."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from _orch_helpers import assert_update_wire_mode
from shared.cli_invoke import AllAccountsCappedException

import orchestrator.harness as harness_module
from orchestrator.agents.invoke import AgentResult
from orchestrator.config import OrchestratorConfig
from orchestrator.harness import Harness
from orchestrator.module_charter import sanitize_files_for_persist
from orchestrator.scheduler import Scheduler
from orchestrator.task_status import ACTIVE_TASK_STATUSES


@pytest.fixture
def config(tmp_path: Path) -> OrchestratorConfig:
    return OrchestratorConfig(project_root=tmp_path)


@pytest.fixture
def harness(config: OrchestratorConfig) -> Harness:
    return Harness(config)


SAMPLE_TASKS = [
    {
        'id': '1',
        'title': 'Add health endpoint',
        'description': 'Add GET /health to the server',
        'status': 'pending',
        'metadata': {},
        'dependencies': [],
    },
    {
        'id': '2',
        'title': 'Refactor config schema',
        'description': 'Split config into sub-models',
        'status': 'pending',
        'metadata': {},
        'dependencies': [],
    },
    {
        'id': '3',
        'title': 'Already tagged task',
        'description': 'This one has files already',
        'status': 'pending',
        'metadata': {'files': ['src/server']},
        'dependencies': [],
    },
]

AGENT_MODULE_RESPONSE = {
    'tasks': [
        {'id': '1', 'files': ['src/server/app.py', 'tests/test_app.py']},
        {'id': '2', 'files': ['src/config/schema.py']},
    ],
}


@pytest.mark.asyncio
async def test_tag_task_modules_calls_update_for_untagged(harness, config):
    """Should invoke agent and call update_task for each untagged task."""
    harness.scheduler.get_tasks = AsyncMock(return_value=SAMPLE_TASKS)
    harness.scheduler.update_task = AsyncMock()

    agent_result = AgentResult(
        success=True,
        output=json.dumps(AGENT_MODULE_RESPONSE),
        structured_output=AGENT_MODULE_RESPONSE,
        cost_usd=0.01, duration_ms=6000, turns=2,
    )

    with patch('orchestrator.harness.invoke_agent', AsyncMock(return_value=agent_result)) as mock_invoke:
        await harness._tag_task_modules()

        # Agent should be called once with all untagged tasks
        mock_invoke.assert_called_once()
        prompt = mock_invoke.call_args.kwargs.get('prompt', mock_invoke.call_args[0][0] if mock_invoke.call_args[0] else '')
        # Should include task 1 and 2 but not task 3 (already tagged)
        assert '"1"' in prompt
        assert '"2"' in prompt

    # update_task called for tasks 1 and 2 only
    assert harness.scheduler.update_task.call_count == 2
    calls = harness.scheduler.update_task.call_args_list
    call_ids = {c.args[0] for c in calls}
    assert call_ids == {'1', '2'}

    # Verify file metadata content
    for call in calls:
        task_id, metadata_json = call.args
        metadata = json.loads(metadata_json)
        assert 'files' in metadata
        assert isinstance(metadata['files'], list)


@pytest.mark.asyncio
async def test_tag_task_modules_skips_when_all_tagged(harness):
    """Should skip agent invocation if all tasks already have files."""
    all_tagged = [
        {
            'id': '1',
            'title': 'Task',
            'description': '',
            'status': 'pending',
            'metadata': {'files': ['src/server']},
            'dependencies': [],
        },
    ]
    harness.scheduler.get_tasks = AsyncMock(return_value=all_tagged)
    harness.scheduler.update_task = AsyncMock()

    with patch('orchestrator.harness.invoke_agent', AsyncMock()) as mock_invoke:
        await harness._tag_task_modules()
        mock_invoke.assert_not_called()

    harness.scheduler.update_task.assert_not_called()


@pytest.mark.asyncio
async def test_tag_task_modules_handles_agent_failure(harness):
    """Should log warning and continue if agent fails."""
    harness.scheduler.get_tasks = AsyncMock(return_value=SAMPLE_TASKS)
    harness.scheduler.update_task = AsyncMock()

    # Non-zero cost/turns/duration avoid the zero-cost instant-exit heuristic
    # that would otherwise classify this as a cap hit in the unified loop.
    agent_result = AgentResult(
        success=False, output='Agent error',
        cost_usd=0.01, duration_ms=6000, turns=2,
    )

    with patch('orchestrator.harness.invoke_agent', AsyncMock(return_value=agent_result)):
        await harness._tag_task_modules()

    harness.scheduler.update_task.assert_not_called()


@pytest.mark.asyncio
async def test_tag_task_modules_handles_bad_json(harness):
    """Should handle unparseable agent output gracefully."""
    harness.scheduler.get_tasks = AsyncMock(return_value=SAMPLE_TASKS)
    harness.scheduler.update_task = AsyncMock()

    agent_result = AgentResult(
        success=True,
        output='not valid json',
        structured_output=None,
        cost_usd=0.01, duration_ms=6000, turns=2,
    )

    with patch('orchestrator.harness.invoke_agent', AsyncMock(return_value=agent_result)):
        await harness._tag_task_modules()

    harness.scheduler.update_task.assert_not_called()


@pytest.mark.asyncio
async def test_tag_task_modules_uses_correct_config(harness, config):
    """Should use module_tagger model/budget/turns from config."""
    harness.scheduler.get_tasks = AsyncMock(return_value=SAMPLE_TASKS)
    harness.scheduler.update_task = AsyncMock()

    agent_result = AgentResult(
        success=True,
        output='{}',
        structured_output={'tasks': []},
        cost_usd=0.01, duration_ms=6000, turns=2,
    )

    with patch('orchestrator.harness.invoke_agent', AsyncMock(return_value=agent_result)) as mock_invoke:
        await harness._tag_task_modules()

        call_kwargs = mock_invoke.call_args.kwargs
        assert call_kwargs['model'] == config.models.module_tagger
        assert call_kwargs['max_turns'] == config.max_turns.module_tagger
        assert call_kwargs['max_budget_usd'] == config.budgets.module_tagger


@pytest.mark.asyncio
async def test_tag_task_modules_handles_all_accounts_capped(harness, caplog):
    """AllAccountsCappedException from invoke_with_cap_retry must return None gracefully.

    Before the explicit handler is added (step-4), the exception propagates and
    crashes _tag_task_modules, causing it to raise rather than return None.
    """
    import logging

    harness.scheduler.get_tasks = AsyncMock(return_value=SAMPLE_TASKS)
    harness.scheduler.update_task = AsyncMock()

    cap_exc = AllAccountsCappedException(
        retries=5, elapsed_secs=60.0, label='Module tagging'
    )

    with caplog.at_level(logging.WARNING, logger='orchestrator.harness'), patch(
        'orchestrator.harness.invoke_with_cap_retry',
        AsyncMock(side_effect=cap_exc),
    ):
        result = await harness._tag_task_modules()

    # Must return None (no exception propagation)
    assert result is None

    # Must NOT have tagged any tasks
    harness.scheduler.update_task.assert_not_called()

    # Must emit a warning containing 'all accounts capped' (case-insensitive)
    warning_texts = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        'all accounts capped' in t.lower() for t in warning_texts
    ), f'Expected warning containing "all accounts capped", got: {warning_texts}'


@pytest.mark.asyncio
async def test_tag_task_modules_fetches_active_only(harness, config):
    """_tag_task_modules must pass statuses=ACTIVE_TASK_STATUSES to get_tasks (γ3b).

    Fails until step-6 converts the get_tasks() call — today it passes no
    statuses kwarg.
    """
    harness.scheduler.get_tasks = AsyncMock(return_value=SAMPLE_TASKS)
    harness.scheduler.update_task = AsyncMock()

    agent_result = AgentResult(
        success=True,
        output='{}',
        structured_output={'tasks': []},
        cost_usd=0.01, duration_ms=6000, turns=2,
    )

    with patch('orchestrator.harness.invoke_agent', AsyncMock(return_value=agent_result)):
        await harness._tag_task_modules()

    harness.scheduler.get_tasks.assert_awaited_once()
    call_kwargs = harness.scheduler.get_tasks.call_args.kwargs
    assert 'statuses' in call_kwargs, (
        f'_tag_task_modules must call get_tasks with statuses=ACTIVE_TASK_STATUSES, '
        f'but call_args.kwargs was: {call_kwargs}'
    )
    assert set(call_kwargs['statuses']) == ACTIVE_TASK_STATUSES, (
        f'Expected statuses {ACTIVE_TASK_STATUSES}, got {set(call_kwargs["statuses"])}'
    )


@pytest.mark.asyncio
async def test_tag_task_modules_forwards_merge_mode_on_wire(harness, monkeypatch):
    """_tag_task_modules update_task wire call must carry metadata_mode='merge' (#4271 fix).

    A module re-tag writes only {'files': [...]}, a partial key.  Under merge
    (shallow last-write-wins) sibling keys like _causation_id and memory_hints
    are preserved; under replace they are silently deleted.

    RED today: the wrapper forwards nothing instead of metadata_mode='merge'.
    GREEN after step-5 impl (scheduler.py update_task gains metadata_mode).
    """
    tasks_with_one_untagged = [
        {'id': '1', 'title': 'Task A', 'status': 'pending', 'metadata': {}, 'dependencies': []},
    ]

    captured_args: list[dict] = []

    async def mock_mcp_call(url, method, payload, **kwargs):
        captured_args.append(payload)
        return {}

    # Wire harness to a real Scheduler so the full update_task wire logic runs.
    real_scheduler = Scheduler(OrchestratorConfig(max_per_module=1))
    real_scheduler.get_tasks = AsyncMock(return_value=tasks_with_one_untagged)
    harness.scheduler = real_scheduler

    monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock_mcp_call)

    agent_response = {'tasks': [{'id': '1', 'files': ['src/server/app.py']}]}
    agent_result = AgentResult(
        success=True,
        output=json.dumps(agent_response),
        structured_output=agent_response,
        cost_usd=0.01, duration_ms=6000, turns=2,
    )

    with patch('orchestrator.harness.invoke_agent', AsyncMock(return_value=agent_result)):
        await harness._tag_task_modules()

    # The wire call for update_task (payload['name']=='update_task') should carry
    # metadata_mode='merge' so sibling keys survive the partial-key re-tag.
    assert_update_wire_mode(captured_args, 'merge')


# ---------------------------------------------------------------------------
# α strip: harness cache-population site (task 1906 step-5/6)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tag_task_modules_does_not_cache_directory_wide_module(tmp_path):
    """Module tagger must NOT populate _module_cache with directory-derived wide modules.

    The α strip (task 1906) coerces directory entries to empty before lock
    derivation in _get_modules. The harness cache-population site (harness.py:1477)
    must apply the same strip so a directory charter cannot poison the in-memory
    cache and bypass the α strip (cache is checked first in _get_modules).

    Fails under current code: files_to_modules(['crates/reify-eval/src'], depth)
    → ['crates/reify-eval'] (a subtree-wide prefix lock) is cached for task 1.
    Passes after step-6 adds strip_directory_locks at the cache-write site.
    """
    config = OrchestratorConfig(project_root=tmp_path, lock_depth=2)
    h = Harness(config)

    tasks = [
        {
            'id': '1',
            'title': 'Directory-charter task',
            'description': 'Has a directory entry only',
            'status': 'pending',
            'metadata': {},
            'dependencies': [],
        },
        {
            'id': '2',
            'title': 'File task',
            'description': 'Has a real file entry',
            'status': 'pending',
            'metadata': {},
            'dependencies': [],
        },
    ]
    # Agent returns a directory entry for task 1, a real file for task 2.
    agent_response = {
        'tasks': [
            {'id': '1', 'files': ['crates/reify-eval/src']},   # directory — no extension
            {'id': '2', 'files': ['src/config/schema.py']},    # real file
        ],
    }

    h.scheduler.get_tasks = AsyncMock(return_value=tasks)
    h.scheduler.update_task = AsyncMock()

    agent_result = AgentResult(
        success=True,
        output=json.dumps(agent_response),
        structured_output=agent_response,
        cost_usd=0.01, duration_ms=6000, turns=2,
    )

    with patch('orchestrator.harness.invoke_agent', AsyncMock(return_value=agent_result)):
        await h._tag_task_modules()

    # Task 1: directory entry must NOT produce a cached wide module.
    # Before fix: cache['1'] = ['crates/reify-eval'] — subtree-wide prefix lock.
    cached_1 = h.scheduler._module_cache.get('1', [])
    assert cached_1 == [], (
        f'Directory charter must NOT be cached as a wide module; '
        f'got _module_cache["1"]={cached_1!r}. '
        f'Expected empty (directory entry stripped → no derived module → cache skipped).'
    )

    # Task 2: real file IS cached (strip keeps it, files_to_modules derives a module).
    # At lock_depth=2, src/config/schema.py → 'src/config'.
    cached_2 = h.scheduler._module_cache.get('2', [])
    assert cached_2 != [], (
        f'Real file must produce a cached module; got _module_cache["2"]={cached_2!r}.'
    )
    assert cached_2 == ['src/config'], (
        f'Expected [\"src/config\"] for src/config/schema.py at depth=2; got {cached_2!r}.'
    )


@pytest.mark.asyncio
async def test_tag_task_modules_strips_directories_in_writeback(tmp_path):
    """Module-tagger writeback must persist FILE-LEVEL paths only.

    The lock-charter guard on update_task / task_interceptor.update_task
    REJECTS any metadata.files containing a directory-shaped entry
    (LockCharterViolation), which would drop the entire payload — including the
    valid file-level entries in the same write. The writeback must therefore
    call strip_directory_locks before persisting, consistent with
    _persist_files_metadata and _reconcile_metadata_files_for_done.

    A mixed file+directory list must persist only the file-level entries.
    """
    config = OrchestratorConfig(project_root=tmp_path, lock_depth=2)
    h = Harness(config)

    tasks = [
        {
            'id': '1',
            'title': 'Mixed-charter task',
            'description': 'LLM tags a real file and a directory',
            'status': 'pending',
            'metadata': {},
            'dependencies': [],
        },
    ]
    # Agent returns a mixed list: a real file plus a directory-shaped entry.
    agent_response = {
        'tasks': [
            {'id': '1', 'files': ['src/config/schema.py', 'crates/reify-eval/src']},
        ],
    }

    h.scheduler.get_tasks = AsyncMock(return_value=tasks)
    h.scheduler.update_task = AsyncMock()

    agent_result = AgentResult(
        success=True,
        output=json.dumps(agent_response),
        structured_output=agent_response,
        cost_usd=0.01, duration_ms=6000, turns=2,
    )

    with patch('orchestrator.harness.invoke_agent', AsyncMock(return_value=agent_result)):
        await h._tag_task_modules()

    h.scheduler.update_task.assert_awaited_once()
    task_id, metadata_json = h.scheduler.update_task.call_args.args
    assert task_id == '1'
    persisted = json.loads(metadata_json)['files']
    # The real file survives; the directory entry is stripped so the guard
    # accepts the write.
    assert persisted == ['src/config/schema.py'], (
        f'Writeback must persist file-level entries only (directory stripped); '
        f'got {persisted!r}.'
    )
    from shared.locking import directory_locks
    assert directory_locks(persisted) == [], (
        f'Persisted files must contain no directory-shaped entries; '
        f'got {directory_locks(persisted)!r}.'
    )


@pytest.mark.asyncio
async def test_tag_task_modules_routes_through_seed_modules(tmp_path):
    """_tag_task_modules must route through scheduler.seed_modules and
    sanitize_files_for_persist instead of the old inline
    directory_locks/files_to_modules block + direct _module_cache poke
    (task 2122 step-9/10).

    seed_modules is replaced with a no-op Mock: post-migration,
    _tag_task_modules has no parallel direct _module_cache write — it routes
    solely through seed_modules — so mocking seed_modules to a no-op must
    leave the cache untouched, proving there is no other writer left.
    """
    config = OrchestratorConfig(project_root=tmp_path, lock_depth=2)
    h = Harness(config)

    tasks = [
        {
            'id': '1',
            'title': 'Mixed-charter task',
            'description': '',
            'status': 'pending',
            'metadata': {},
            'dependencies': [],
        },
    ]
    raw_files = ['src/config/schema.py', 'crates/reify-eval/src']
    agent_response = {'tasks': [{'id': '1', 'files': raw_files}]}

    h.scheduler.get_tasks = AsyncMock(return_value=tasks)
    h.scheduler.update_task = AsyncMock()
    mock_seed_modules = Mock(return_value=[])
    h.scheduler.seed_modules = mock_seed_modules

    agent_result = AgentResult(
        success=True,
        output=json.dumps(agent_response),
        structured_output=agent_response,
        cost_usd=0.01, duration_ms=6000, turns=2,
    )

    with (
        patch(
            'orchestrator.harness.sanitize_files_for_persist',
            wraps=sanitize_files_for_persist,
        ) as mock_sanitize,
        patch('orchestrator.harness.invoke_agent', AsyncMock(return_value=agent_result)),
    ):
        await h._tag_task_modules()

    # Routes through seed_modules with the RAW agent-predicted files (the α
    # strip lives inside derive_modules, called by seed_modules itself).
    mock_seed_modules.assert_called_once_with('1', raw_files)

    # No other code path writes _module_cache directly: with seed_modules
    # mocked to a no-op, the cache must stay empty.
    assert '1' not in h.scheduler._module_cache, (
        f'_tag_task_modules must not assign scheduler._module_cache directly; '
        f'got _module_cache={h.scheduler._module_cache!r} despite seed_modules '
        f'being mocked out'
    )

    # Writeback sanitizes via sanitize_files_for_persist, not a raw
    # strip_directory_locks call.
    mock_sanitize.assert_called_once_with(raw_files)
    h.scheduler.update_task.assert_awaited_once()
    task_id, metadata_json = h.scheduler.update_task.call_args.args
    assert task_id == '1'
    assert json.loads(metadata_json)['files'] == ['src/config/schema.py']


# ---------------------------------------------------------------------------
# StructuredOutput schema rename 'tasks'->'predictions' + double-wrap
# acceptance (task 2561 defect 3 / acceptance c)
# ---------------------------------------------------------------------------

_ONE_UNTAGGED_TASK = [
    {
        'id': '1',
        'title': 'Task',
        'description': 'desc',
        'status': 'pending',
        'metadata': {},
        'dependencies': [],
    },
]


@pytest.mark.asyncio
async def test_output_schema_uses_predictions_root(harness):
    """The output_schema's top-level key must be 'predictions', not 'tasks'.

    The StructuredOutput tool's sole parameter is named after the schema's
    top-level key; when that key is 'tasks' — the same heavily-repeated
    domain noun used throughout the prompt — the model double-wraps its
    answer as {"tasks": {"tasks": [...]}} (9/11 mined sessions). Renaming
    the key removes the param-name/schema-key/domain-noun collision.

    Fails now: schema is still keyed 'tasks'.
    """
    harness.scheduler.get_tasks = AsyncMock(return_value=_ONE_UNTAGGED_TASK)
    harness.scheduler.update_task = AsyncMock()

    agent_response = {'predictions': [{'id': '1', 'files': ['a.py']}]}
    agent_result = AgentResult(
        success=True,
        output=json.dumps(agent_response),
        structured_output=agent_response,
        cost_usd=0.01, duration_ms=6000, turns=2,
    )

    with patch('orchestrator.harness.invoke_agent', AsyncMock(return_value=agent_result)) as mock_invoke:
        await harness._tag_task_modules()

    schema = mock_invoke.call_args.kwargs['output_schema']
    assert 'predictions' in schema['properties'], (
        f"Expected 'predictions' key in schema['properties']; got {schema['properties']!r}"
    )
    assert 'tasks' not in schema['properties'], (
        f"schema['properties'] must not retain the colliding 'tasks' key; got {schema['properties']!r}"
    )
    assert schema['required'] == ['predictions'], (
        f"Expected schema['required'] == ['predictions']; got {schema['required']!r}"
    )

    prompt = mock_invoke.call_args.kwargs['prompt']
    assert 'predictions' in prompt, "Expected the literal token 'predictions' in the tagger prompt"


@pytest.mark.asyncio
async def test_tag_accepts_legacy_double_wrap(harness):
    """The parser must still accept the legacy double-wrap {"tasks": {"tasks": [...]}}.

    A defensive normalizer (_extract_tagger_entries) tolerates old-format
    stragglers during rollout, even after the schema key is renamed.

    Fails now: the parser does `mapping.get('tasks', [])`, which on this
    double-wrap returns the inner dict {'tasks': [...]}, not a list.
    """
    harness.scheduler.get_tasks = AsyncMock(return_value=_ONE_UNTAGGED_TASK)
    harness.scheduler.update_task = AsyncMock()

    agent_response = {'tasks': {'tasks': [{'id': '1', 'files': ['a.py']}]}}
    agent_result = AgentResult(
        success=True,
        output=json.dumps(agent_response),
        structured_output=agent_response,
        cost_usd=0.01, duration_ms=6000, turns=2,
    )

    with patch('orchestrator.harness.invoke_agent', AsyncMock(return_value=agent_result)):
        await harness._tag_task_modules()

    harness.scheduler.update_task.assert_awaited_once()
    task_id, metadata_json = harness.scheduler.update_task.call_args.args
    assert task_id == '1'
    assert json.loads(metadata_json)['files'] == ['a.py']


@pytest.mark.asyncio
async def test_tag_accepts_new_flat_predictions(harness):
    """The parser must accept the new flat {"predictions": [...]} shape.

    Fails now: the parser does `mapping.get('tasks', [])`, which finds
    nothing under the new 'predictions' key and silently tags 0 tasks.
    """
    harness.scheduler.get_tasks = AsyncMock(return_value=_ONE_UNTAGGED_TASK)
    harness.scheduler.update_task = AsyncMock()

    agent_response = {'predictions': [{'id': '1', 'files': ['a.py']}]}
    agent_result = AgentResult(
        success=True,
        output=json.dumps(agent_response),
        structured_output=agent_response,
        cost_usd=0.01, duration_ms=6000, turns=2,
    )

    with patch('orchestrator.harness.invoke_agent', AsyncMock(return_value=agent_result)):
        await harness._tag_task_modules()

    harness.scheduler.update_task.assert_awaited_once()
    task_id, metadata_json = harness.scheduler.update_task.call_args.args
    assert task_id == '1'
    assert json.loads(metadata_json)['files'] == ['a.py']


# ---------------------------------------------------------------------------
# Sentinel persist for empty/omitted predictions (task 2561 defect 1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tag_persists_sentinel_for_empty_and_omitted(harness):
    """Every task in the untagged batch must get files_tagged_at persisted,
    even when the agent predicts empty files or omits the task entirely.

    Without this sentinel, a task the agent can't (or won't) predict files
    for re-enters the tagging batch every cycle forever — 73 tagger
    sessions mined in one month, an LLM-spend leak.

    Fails now: the persist gate `if task_id and files:` writes nothing for
    an empty or omitted prediction, so neither '1' nor '2' get an
    update_task call.
    """
    tasks = [
        {
            'id': '1', 'title': 'Task A', 'description': '',
            'status': 'pending', 'metadata': {}, 'dependencies': [],
        },
        {
            'id': '2', 'title': 'Task B', 'description': '',
            'status': 'pending', 'metadata': {}, 'dependencies': [],
        },
    ]
    harness.scheduler.get_tasks = AsyncMock(return_value=tasks)
    harness.scheduler.update_task = AsyncMock()
    mock_seed_modules = Mock(return_value=[])
    harness.scheduler.seed_modules = mock_seed_modules

    # '1' gets an explicit empty files prediction; '2' is omitted entirely.
    agent_response = {'predictions': [{'id': '1', 'files': []}]}
    agent_result = AgentResult(
        success=True,
        output=json.dumps(agent_response),
        structured_output=agent_response,
        cost_usd=0.01, duration_ms=6000, turns=2,
    )

    with patch('orchestrator.harness.invoke_agent', AsyncMock(return_value=agent_result)):
        await harness._tag_task_modules()

    assert harness.scheduler.update_task.call_count == 2, (
        f'Expected update_task for both tasks; got '
        f'{harness.scheduler.update_task.call_args_list!r}'
    )
    calls = harness.scheduler.update_task.call_args_list
    call_ids = {c.args[0] for c in calls}
    assert call_ids == {'1', '2'}

    for call in calls:
        task_id, metadata_json = call.args
        metadata = json.loads(metadata_json)
        tagged_at = metadata.get('files_tagged_at')
        assert isinstance(tagged_at, str) and tagged_at, (
            f'Expected a truthy files_tagged_at string for task {task_id}; '
            f'got metadata={metadata!r}'
        )
        assert not metadata.get('files'), (
            f'Expected no non-empty files for task {task_id} (empty/omitted '
            f'prediction); got metadata={metadata!r}'
        )

    mock_seed_modules.assert_not_called()


# ---------------------------------------------------------------------------
# _extract_tagger_entries: StructuredOutput wrapper-peeling normalizer
# (task 2561 defect 3 / acceptance c)
# ---------------------------------------------------------------------------

_EXTRACT_TAGGER_ENTRIES_CASES = [
    pytest.param(
        {'predictions': [{'id': '1', 'files': ['a.py']}]},
        [{'id': '1', 'files': ['a.py']}],
        id='new_flat_predictions',
    ),
    pytest.param(
        {'tasks': [{'id': '1', 'files': ['a.py']}]},
        [{'id': '1', 'files': ['a.py']}],
        id='legacy_single_wrap_tasks',
    ),
    pytest.param(
        {'tasks': {'tasks': [{'id': '1', 'files': ['a.py']}]}},
        [{'id': '1', 'files': ['a.py']}],
        id='legacy_double_wrap_tasks_tasks',
    ),
    pytest.param({}, [], id='empty_dict'),
    pytest.param(None, [], id='none'),
    pytest.param({'tasks': {}}, [], id='tasks_maps_to_empty_dict'),
]


@pytest.mark.parametrize('payload, expected', _EXTRACT_TAGGER_ENTRIES_CASES)
def test_extract_tagger_entries(payload, expected):
    """_extract_tagger_entries must peel known wrapper keys up to a bounded depth.

    Accepts the new flat {"predictions": [...]} shape (post-rename), the
    legacy single-wrap {"tasks": [...]}, and the legacy double-wrap
    {"tasks": {"tasks": [...]}} produced when the StructuredOutput tool's
    sole parameter collides with the schema's top-level key. Anything that
    doesn't resolve to a list fails safe to [] (no tagging, no crash).

    Fails now with AttributeError: orchestrator.harness has no
    _extract_tagger_entries yet (added in step-4).
    """
    assert harness_module._extract_tagger_entries(payload) == expected


# ---------------------------------------------------------------------------
# Sentinel skip: a sentinel-but-no-files task must not re-enter the batch
# (task 2561 acceptance a-skip)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tag_skips_task_with_sentinel(harness):
    """A task carrying only files_tagged_at (no files) must be skipped.

    Without this, a task the agent couldn't predict files for — which only
    ever gets the sentinel, never a files list — would re-enter the
    tagging batch on every cycle forever.

    Fails now: the skip-check keys off truthy `files` only, so a
    sentinel-but-no-files task still enters the batch and invokes the
    agent.
    """
    tasks = [
        {
            'id': '1', 'title': 'Task', 'description': '',
            'status': 'pending',
            'metadata': {'files_tagged_at': '2026-07-16T00:00:00+00:00'},
            'dependencies': [],
        },
    ]
    harness.scheduler.get_tasks = AsyncMock(return_value=tasks)
    harness.scheduler.update_task = AsyncMock()

    with patch('orchestrator.harness.invoke_agent', AsyncMock()) as mock_invoke:
        await harness._tag_task_modules()
        mock_invoke.assert_not_called()

    harness.scheduler.update_task.assert_not_called()


# ---------------------------------------------------------------------------
# Deterministic task exclusion (task 2561 defect 2 / acceptance b)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tag_excludes_deterministic_tasks(harness):
    """task_kind='deterministic' tasks must never enter the tagging batch.

    Deterministic tasks carry no worktree and no code (CLAUDE.md
    "Deterministic task kind"), so a file-lock prediction is meaningless
    for them.

    Fails now: the batch loop has no task_kind filter, so the
    deterministic task '2' is included in the prompt and can be updated.
    """
    tasks = [
        {
            'id': '1', 'title': 'Normal task', 'description': '',
            'status': 'pending', 'metadata': {}, 'dependencies': [],
        },
        {
            'id': '2', 'title': 'Deterministic task', 'description': '',
            'status': 'pending', 'metadata': {'task_kind': 'deterministic'},
            'dependencies': [],
        },
    ]
    harness.scheduler.get_tasks = AsyncMock(return_value=tasks)
    harness.scheduler.update_task = AsyncMock()

    agent_response = {'predictions': [{'id': '1', 'files': ['a.py']}]}
    agent_result = AgentResult(
        success=True,
        output=json.dumps(agent_response),
        structured_output=agent_response,
        cost_usd=0.01, duration_ms=6000, turns=2,
    )

    with patch('orchestrator.harness.invoke_agent', AsyncMock(return_value=agent_result)) as mock_invoke:
        await harness._tag_task_modules()

    prompt = mock_invoke.call_args.kwargs['prompt']
    assert '"1"' in prompt
    assert '"2"' not in prompt, (
        f'Deterministic task "2" must not appear in the tagger prompt; got prompt={prompt!r}'
    )

    update_ids = {c.args[0] for c in harness.scheduler.update_task.call_args_list}
    assert '2' not in update_ids, (
        f'update_task must not be awaited for deterministic task "2"; got ids={update_ids!r}'
    )


@pytest.mark.asyncio
async def test_tag_skips_when_only_deterministic(harness):
    """If the only untagged task is deterministic, the batch is empty and
    the agent must not be invoked at all.

    Fails now: no task_kind filter, so the deterministic task fills the
    batch and the agent is invoked.
    """
    tasks = [
        {
            'id': '2', 'title': 'Deterministic task', 'description': '',
            'status': 'pending', 'metadata': {'task_kind': 'deterministic'},
            'dependencies': [],
        },
    ]
    harness.scheduler.get_tasks = AsyncMock(return_value=tasks)
    harness.scheduler.update_task = AsyncMock()

    with patch('orchestrator.harness.invoke_agent', AsyncMock()) as mock_invoke:
        await harness._tag_task_modules()
        mock_invoke.assert_not_called()

    harness.scheduler.update_task.assert_not_called()


# ---------------------------------------------------------------------------
# details fallback in the prompt when description is empty (task 2561
# defect 3 prompt fix)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tag_prompt_uses_details_when_description_empty(harness):
    """When a task's description is empty, the prompt must fall back to
    its details field so the tagger still has context to predict from.

    Fails now: task_summaries reads t.get('description', '') only, so
    details is never surfaced in the prompt.
    """
    tasks = [
        {
            'id': '1', 'title': 'Task', 'description': '',
            'details': 'SPECIAL_DETAILS_TOKEN',
            'status': 'pending', 'metadata': {}, 'dependencies': [],
        },
    ]
    harness.scheduler.get_tasks = AsyncMock(return_value=tasks)
    harness.scheduler.update_task = AsyncMock()

    agent_response = {'predictions': []}
    agent_result = AgentResult(
        success=True,
        output=json.dumps(agent_response),
        structured_output=agent_response,
        cost_usd=0.01, duration_ms=6000, turns=2,
    )

    with patch('orchestrator.harness.invoke_agent', AsyncMock(return_value=agent_result)) as mock_invoke:
        await harness._tag_task_modules()

    prompt = mock_invoke.call_args.kwargs['prompt']
    assert 'SPECIAL_DETAILS_TOKEN' in prompt, (
        f'Expected details fallback text in the prompt when description is empty; got prompt={prompt!r}'
    )
