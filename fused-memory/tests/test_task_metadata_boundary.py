"""W3-ζ two-way boundary/integration-gate suite for the TaskMetadata contract (PRD §6).

Unlike the rest of ``fused-memory/tests``, this suite is deliberately an
INTEGRATION gate, not a synthetic-input unit test: it drives the REAL
orchestrator producer (``orchestrator.deterministic_runner._build_done_provenance``,
``orchestrator.workflow.TaskWorkflow``) across the package boundary into the
REAL fused-memory consumer (:class:`SqliteTaskBackend`), via the cross-package
``pythonpath = ["src", "../orchestrator/src"]`` entry in ``pyproject.toml``.
Each row exercises one invariant from ``plans/task-metadata-schema-prd.md``
§5/§6 (I1 round-trip, I2 single-enum symmetry, I3 post-merge validation) and
must be green against the already-landed producer+consumer stack (α/2158,
β/2162, γ/2166, δ/2167, ε/2172) — a RED result here is a genuine integration
gap, not a synthetic-input failure.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from orchestrator.deterministic_runner import (  # type: ignore[import-not-found]
    _build_done_provenance,
)
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome  # type: ignore[import-not-found]
from pydantic import ValidationError
from shared.task_metadata import (
    MemoryHints,
    RetryLedger,
    TaskMetadata,
    apply_migrations,
    parse_metadata,
)

from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.config.schema import TaskmasterConfig


@pytest_asyncio.fixture
async def make_backend(tmp_path):
    """Factory fixture: ``make_backend(enforce=False)`` -> a started :class:`SqliteTaskBackend`.

    A callable factory (rather than a single fixture instance) so one test
    can construct BOTH a warn-mode and an enforce-mode backend to exercise
    the staged-rollout contract (rows 5-6) against the same malformed write.
    Mirrors test_sqlite_task_backend.py's ``backend`` fixture; every backend
    the factory creates is closed at teardown.
    """
    created: list[SqliteTaskBackend] = []

    async def _make(enforce: bool = False) -> SqliteTaskBackend:
        cfg = TaskmasterConfig(project_root=str(tmp_path))
        backend = SqliteTaskBackend(cfg, task_metadata_enforce=enforce)
        await backend.start()
        created.append(backend)
        return backend

    yield _make

    for backend in created:
        await backend.close()


def _schema_warning_messages(caplog) -> list[str]:
    """Return WARNING+ log messages carrying the write-boundary census token.

    Mirrors the exact recipe in test_sqlite_task_backend.py: records on the
    ``fused_memory.backends.sqlite_task_backend`` logger whose message
    contains the literal ``task_metadata.schema_warning`` token (task 2162).
    Call within a ``caplog.at_level(logging.WARNING,
    logger='fused_memory.backends.sqlite_task_backend')`` block.
    """
    return [
        r.message for r in caplog.records
        if r.levelno >= logging.WARNING
        and r.name == 'fused_memory.backends.sqlite_task_backend'
        and 'task_metadata.schema_warning' in r.message
    ]


# ── Row 1 — producer/legacy-read (I1 round-trip) ─────────────────────


def test_legacy_blob_roundtrip_preserves_unknown_keys_and_upgrades_memory_hints():
    """A v0-shaped legacy blob upgrades memory_hints and never silently drops keys.

    No ``schema_version`` key (a legacy row) carrying a legacy list-shaped
    ``memory_hints``, an unknown non-namespaced key, and an ``x_``-prefixed
    key. ``parse_metadata(direction='read')`` is the single legacy-read seam
    both fused-memory and the orchestrator use — I1 requires every unknown
    key survive the round-trip, and the v0->v1 migration upgrades the legacy
    list shape to the canonical ``{entities, queries}`` dict.
    """
    blob = {
        'memory_hints': [
            {'entity': 'E1', 'query': 'Q1'},
            {'entity': 'E2', 'query': 'Q2'},
        ],
        'legacy_only_key': 'v',
        'x_private': {'a': 1},
    }

    # apply_migrations is the standalone migration seam parse_metadata calls
    # internally — exercised directly here as well as end-to-end below.
    migrated = apply_migrations(blob)
    assert migrated['memory_hints'] == {'entities': ['E1', 'E2'], 'queries': ['Q1', 'Q2']}

    model, warnings = parse_metadata(blob, direction='read')
    assert isinstance(model, TaskMetadata)
    dumped = model.model_dump()

    # (a) Unknown keys survive byte-for-value (I1 — extra='allow', no silent drop).
    assert dumped['legacy_only_key'] == 'v'
    assert dumped['x_private'] == {'a': 1}

    # (b) Legacy list memory_hints upgraded to the canonical {entities, queries} shape.
    assert dumped['memory_hints'] == {'entities': ['E1', 'E2'], 'queries': ['Q1', 'Q2']}
    assert model.memory_hints == MemoryHints(entities=['E1', 'E2'], queries=['Q1', 'Q2'])

    # (c) 'legacy_only_key' is warned as unknown; 'x_private' (x_-namespaced) is not.
    unknown_key_fields = {w.field for w in warnings if w.code == 'unknown_key'}
    assert 'legacy_only_key' in unknown_key_fields
    assert 'x_private' not in unknown_key_fields


# ── Row 2 — consumer/write, I2: all four kinds accepted ──────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'kind,fields',
    [
        ('merged', {'commit': 'abc123'}),
        ('found_on_main', {'commit': 'abc123', 'note': 'found it'}),
        ('deterministic-deploy', {'pid': 4242, 'unit': 'fused-memory.service'}),
        ('deterministic-deploy-scheduled', {'unit': 'fused-memory.service'}),
    ],
)
async def test_orchestrator_done_provenance_all_kinds_accepted_by_backend(
    make_backend, tmp_path, kind, fields,
):
    """Every kind the orchestrator can construct is accepted by the validator.

    Builds ``done_provenance`` via the REAL orchestrator seam
    (``_build_done_provenance`` — task 2167 W3-δ SEAM B) for each of the four
    ``DoneProvenance.kind`` values, then writes it through an enforce-mode
    backend. I2: the orchestrator and fused-memory share ONE valid-kinds
    enum, so nothing the producer can legitimately build is ever rejected by
    the consumer (structurally prevents the 1902/1976/1982 class of bug).
    """
    backend = await make_backend(enforce=True)
    project_root = str(tmp_path / 'proj')

    built = _build_done_provenance(kind, **fields)

    dto = await backend.add_task(project_root=project_root, title='t')
    await backend.update_task(
        dto['id'], project_root=project_root,
        metadata=json.dumps({'done_provenance': built}),
    )

    task = await backend.get_task(dto['id'], project_root=project_root)
    assert task['metadata']['done_provenance']['kind'] == kind
    # Full dict survives byte-for-value — a consumer that silently dropped
    # or mangled a non-kind field (commit/note/pid/unit) must not pass.
    assert task['metadata']['done_provenance'] == built


# ── Row 3 — consumer/write negative: symmetric rejection ─────────────


@pytest.mark.asyncio
async def test_bogus_done_provenance_kind_rejected_symmetrically(make_backend, tmp_path):
    """A kind unknown to the shared model is rejected identically on both sides.

    Producer side: the orchestrator's ``_build_done_provenance`` refuses to
    construct a bogus kind (``DoneProvenance.kind`` is a closed Literal).
    Consumer side: an enforce-mode backend refuses to store the same bogus
    kind, and the rolled-back txn leaves the task's metadata untouched. This
    is the single-enum symmetry (I2) proven negatively.
    """
    with pytest.raises(ValidationError):
        _build_done_provenance('bogus')

    backend = await make_backend(enforce=True)
    project_root = str(tmp_path / 'proj')
    dto = await backend.add_task(project_root=project_root, title='t')

    with pytest.raises(ValidationError):
        await backend.update_task(
            dto['id'], project_root=project_root,
            metadata=json.dumps({'done_provenance': {'kind': 'bogus'}}),
        )

    task = await backend.get_task(dto['id'], project_root=project_root)
    assert task['metadata'] == {}


# ── Row 4a — producer↔consumer ledger round-trip ──────────────────────


@pytest.mark.asyncio
async def test_retry_ledger_roundtrips_through_backend(make_backend, tmp_path):
    """A typed RetryLedger survives the real durable write chokepoint intact.

    Constructs a RetryLedger with real non-sentinel counter values exactly
    as the workflow persists it (``ledger.model_dump()``), writes it through
    an enforce-mode backend, and reads it back. This is the two-way boundary
    the ε/2172 mock-scheduler unit tests do not cover: both the raw
    byte-for-value round-trip AND re-parsing back to an equal typed
    instance.
    """
    ledger = RetryLedger(
        consecutive_infra_resume_failures=3,
        last_infra_resume_iteration_count=7,
        total_no_plan_failures=2,
        last_no_plan_main_sha='deadbeef',
        last_merge_outcome_signature='sig-abc123',
    )
    dumped = ledger.model_dump()

    backend = await make_backend(enforce=True)
    project_root = str(tmp_path / 'proj')
    dto = await backend.add_task(project_root=project_root, title='t')

    await backend.update_task(
        dto['id'], project_root=project_root,
        metadata=json.dumps({'retry_ledger': dumped}),
    )

    task = await backend.get_task(dto['id'], project_root=project_root)
    reread = task['metadata']['retry_ledger']

    assert reread == dumped
    assert RetryLedger(**reread) == ledger


# ── Row 4b — producer↔consumer ledger: persist failure escalates ─────


def _make_workflow(
    *,
    task_id: str = '99',
    main_sha: str = 'SHA-A',
    update_task_raises: bool = False,
) -> tuple[TaskWorkflow, AsyncMock]:
    """Build a REAL TaskWorkflow around mocked collaborators.

    A local, trimmed port of ``orchestrator/tests/test_workflow_no_plan_cycle.py``'s
    ``_make`` — that module (and its ``_orch_helpers.pydantic_spec``) is not
    on the fused-memory pythonpath, so the harness is reconstructed here.
    ``config`` is a plain (non-spec'd) MagicMock: ``_handle_no_plan_failure``
    and the persist-fail path it exercises never read ``self.config``, only
    ``TaskWorkflow.__init__``'s ``_resolve_module_configs()`` does (via
    ``config.for_module``), and a bare MagicMock satisfies that without
    stubbing.
    """
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {'id': task_id, 'title': 'T', 'description': 'd', 'metadata': {}}
    assignment.modules = ['mod_a']

    config = MagicMock()

    if update_task_raises:
        update_task = AsyncMock(side_effect=RuntimeError('mcp down'))
    else:
        update_task = AsyncMock(return_value=True)

    scheduler = MagicMock()
    scheduler.update_task = update_task
    scheduler.get_task = AsyncMock(return_value={'id': task_id, 'metadata': {}})

    git_ops = MagicMock()
    git_ops.get_main_sha = AsyncMock(return_value=main_sha)

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=MagicMock(),
        mcp=MagicMock(),
    )

    # Stub _mark_blocked — this test only asserts how _handle_no_plan_failure
    # routes into it, not _mark_blocked's own internals.
    mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
    wf._mark_blocked = mark_blocked  # type: ignore[method-assign]

    return wf, mark_blocked


@pytest.mark.asyncio
async def test_retry_ledger_persist_failure_escalates_to_human():
    """A failed retry_ledger persist escalates to a human rather than under-firing.

    Drives the REAL orchestrator guard ``TaskWorkflow._handle_no_plan_failure``
    with ``scheduler.update_task`` raising. If the counter can't be trusted
    to have landed, the guard must escalate immediately rather than silently
    losing the increment (which would let the no-plan loop under-fire) —
    even on what would otherwise be a first, non-escalating failure.
    """
    wf, mark_blocked = _make_workflow(update_task_raises=True)

    outcome = await wf._handle_no_plan_failure('no plan.json produced', detail='')

    assert outcome == WorkflowOutcome.BLOCKED
    mark_blocked.assert_awaited_once()
    assert mark_blocked.await_args is not None
    _, kwargs = mark_blocked.await_args
    assert kwargs.get('escalate_to_human') is True


# ── Row 5 — staged rollout: same malformed write, warn vs enforce ────


@pytest.mark.asyncio
async def test_done_provenance_malformed_write_warn_vs_enforce_staged_rollout(
    make_backend, tmp_path, caplog,
):
    """The warn->enforce staged-rollout contract that the θ2 gate flips.

    ONE malformed blob (``done_provenance.kind='bogus'``, a known field with
    an invalid value) written through both backend modes: warn-mode does not
    raise — the write proceeds and exactly one
    ``task_metadata.schema_warning`` census line is emitted (``done_provenance``
    is a known field, so no additional ``unknown_key`` warning) — while
    enforce-mode raises ValidationError and rolls back.
    """
    malformed = {'done_provenance': {'kind': 'bogus'}}

    # (a) Warn-mode: does not raise; write proceeds; exactly one census line.
    warn_backend = await make_backend(enforce=False)
    warn_root = str(tmp_path / 'warn')
    dto = await warn_backend.add_task(project_root=warn_root, title='t')

    with caplog.at_level(logging.WARNING, logger='fused_memory.backends.sqlite_task_backend'):
        await warn_backend.update_task(
            dto['id'], project_root=warn_root, metadata=json.dumps(malformed),
        )

    census = _schema_warning_messages(caplog)
    assert len(census) == 1, f'Expected exactly one census line; got {census}'

    task = await warn_backend.get_task(dto['id'], project_root=warn_root)
    assert task['metadata'] == malformed

    # (b) Enforce-mode: raises; write rolled back.
    enforce_backend = await make_backend(enforce=True)
    enforce_root = str(tmp_path / 'enforce')
    dto2 = await enforce_backend.add_task(project_root=enforce_root, title='t')

    with pytest.raises(ValidationError):
        await enforce_backend.update_task(
            dto2['id'], project_root=enforce_root, metadata=json.dumps(malformed),
        )


# ── Row 6 — update-path invariant: post-merge (I3) ────────────────────


@pytest.mark.asyncio
async def test_update_task_kind_deterministic_on_normal_task_rejected_post_merge(
    make_backend, tmp_path, caplog,
):
    """update_task validates the MERGED whole, not just the incoming delta.

    Seeds a normal task with benign existing metadata so the default
    shallow-merge is non-trivial, then updates with only
    ``task_kind='deterministic'``. The POST-MERGE blob
    (``{'files': [...], 'task_kind': 'deterministic'}``) violates the
    deterministic invariant (requires ``before_done`` or
    ``always_escalates``) even though neither half violates it alone —
    proving I3 is caught on update, not only on submit.
    """
    seed_metadata = json.dumps({'files': ['a.py']})

    # (a) Enforce-mode: raises; rolled back — metadata stays at the seed.
    enforce_backend = await make_backend(enforce=True)
    enforce_root = str(tmp_path / 'enforce')
    dto = await enforce_backend.add_task(
        project_root=enforce_root, title='t', metadata=seed_metadata,
    )

    with pytest.raises(ValidationError):
        await enforce_backend.update_task(
            dto['id'], project_root=enforce_root,
            metadata=json.dumps({'task_kind': 'deterministic'}),
        )

    task = await enforce_backend.get_task(dto['id'], project_root=enforce_root)
    assert task['metadata'] == {'files': ['a.py']}

    # (b) Warn-mode: does not raise; exactly one whole-metadata census line;
    # the merged write proceeds.
    warn_backend = await make_backend(enforce=False)
    warn_root = str(tmp_path / 'warn')
    dto2 = await warn_backend.add_task(
        project_root=warn_root, title='t', metadata=seed_metadata,
    )

    with caplog.at_level(logging.WARNING, logger='fused_memory.backends.sqlite_task_backend'):
        await warn_backend.update_task(
            dto2['id'], project_root=warn_root,
            metadata=json.dumps({'task_kind': 'deterministic'}),
        )

    census = _schema_warning_messages(caplog)
    assert len(census) == 1, f'Expected exactly one census line; got {census}'
    assert '<metadata>' in census[0]
    assert 'before_done' in census[0]

    task2 = await warn_backend.get_task(dto2['id'], project_root=warn_root)
    assert task2['metadata'] == {'files': ['a.py'], 'task_kind': 'deterministic'}
