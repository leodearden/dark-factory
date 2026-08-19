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
    """Every kind the orchestrator can construct is accepted by the backend.

    Builds ``done_provenance`` via the REAL orchestrator seam
    (``_build_done_provenance`` — task 2167 W3-δ SEAM B) for each of the four
    ``DoneProvenance.kind`` values, then writes it through the REAL public
    ``add_task`` write boundary. Task 2201/C1's write-authority floor lives
    on ``update_task`` only — ``add_task`` has no done_provenance floor — so
    a fresh insert carrying ``metadata.done_provenance`` is validated
    (whole-blob enforce, via ``_validate_metadata_on_write``) AND persisted
    in the very same call, exactly as production's insert path would treat
    it. I2: the orchestrator and fused-memory share ONE valid-kinds enum, so
    nothing the producer can legitimately build is ever rejected by the
    consumer (structurally prevents the 1902/1976/1982 class of bug).
    """
    backend = await make_backend(enforce=True)
    project_root = str(tmp_path / 'proj')

    built = _build_done_provenance(kind, **fields)

    dto = await backend.add_task(
        project_root=project_root, title='t',
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
    Consumer side: the backend's REAL public ``add_task`` write boundary
    refuses the same bogus kind (enforce-mode ``_validate_metadata_on_write``,
    which ``add_task`` always calls whole-blob) and rolls back its INSERT.
    This is the single-enum symmetry (I2) proven negatively, together with
    the transactional-rollback guarantee. ``update_task`` is not usable for
    this any more: task 2201/C1 made it reject ``metadata.done_provenance``
    unconditionally as a write-authority floor, before any kind is ever
    validated — so this exercises the bogus-kind rejection through
    ``add_task``, the one public seam where a fresh done_provenance write
    still reaches the validator, and confirms no row was left behind
    (nothing was ever staged for the privileged ``stamp_audit_metadata``
    seam, which performs no schema validation of its own, to persist).
    """
    with pytest.raises(ValidationError):
        _build_done_provenance('bogus')

    backend = await make_backend(enforce=True)
    project_root = str(tmp_path / 'proj')

    with pytest.raises(ValidationError):
        await backend.add_task(
            project_root=project_root, title='t',
            metadata=json.dumps({'done_provenance': {'kind': 'bogus'}}),
        )

    # The rejected add_task's INSERT never landed — no row/metadata leaked.
    tasks = (await backend.get_tasks(project_root=project_root))['tasks']
    assert tasks == []


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

    # noqa: bare-magicmock — pydantic_spec unavailable on fused-memory pythonpath; only config.for_module is read (see docstring above)
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
    an invalid value) run through the backend's write-boundary validator
    (``_validate_metadata_on_write`` — the exact seam ``add_task``/
    ``update_task`` call internally) in both modes: warn-mode does not raise
    — exactly one ``task_metadata.schema_warning`` census line is emitted
    (``done_provenance`` is a known field, so no additional ``unknown_key``
    warning) — while enforce-mode raises ValidationError. Task 2201/C1 made
    ``update_task`` reject ``metadata.done_provenance`` unconditionally
    before validation would ever run, and made the privileged
    ``stamp_audit_metadata`` seam (the sole sanctioned done_provenance
    writer) a raw, trusted writer with no validation of its own — so the
    warn-mode write-proceeds half of this contract is now demonstrated by
    persisting through that seam directly, mirroring the pre-2201 behaviour
    where a warn-mode ``update_task`` write landed the malformed blob.
    """
    malformed = {'done_provenance': {'kind': 'bogus'}}

    # (a) Warn-mode: validator does not raise; exactly one census line; the
    # write proceeds via the privileged seam.
    warn_backend = await make_backend(enforce=False)
    warn_root = str(tmp_path / 'warn')
    dto = await warn_backend.add_task(project_root=warn_root, title='t')
    tid = int(dto['id'])

    with caplog.at_level(logging.WARNING, logger='fused_memory.backends.sqlite_task_backend'):
        # Scope the census to exactly this validation call, not whatever the
        # prior add_task call in this test happened to (not) log.
        caplog.clear()
        await warn_backend._validate_metadata_on_write(
            json.dumps(malformed),
            project_root=warn_root, tag='master', task_id=tid,
        )

    census = _schema_warning_messages(caplog)
    assert len(census) == 1, f'Expected exactly one census line; got {census}'

    await warn_backend.stamp_audit_metadata(dto['id'], warn_root, malformed)
    task = await warn_backend.get_task(dto['id'], project_root=warn_root)
    assert task['metadata'] == malformed

    # (b) Enforce-mode: validator raises; nothing is staged for the
    # privileged seam to persist.
    enforce_backend = await make_backend(enforce=True)
    enforce_root = str(tmp_path / 'enforce')
    dto2 = await enforce_backend.add_task(project_root=enforce_root, title='t')
    tid2 = int(dto2['id'])

    with pytest.raises(ValidationError):
        await enforce_backend._validate_metadata_on_write(
            json.dumps(malformed),
            project_root=enforce_root, tag='master', task_id=tid2,
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
        # Scope the census to exactly this write, not whatever the prior
        # add_task calls in this test happened to (not) log.
        caplog.clear()
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


# ── Row 7 — capstone: census code= token + vocabulary reconciliation ─


@pytest.mark.asyncio
async def test_census_code_token_and_vocabulary_reconciliation_end_to_end(
    make_backend, tmp_path, caplog,
):
    """Capstone regression guard for the near-empty enforcement census (task 2330).

    Drives the REAL write-boundary validator (``_validate_metadata_on_write``
    — the exact seam ``add_task``/``update_task`` call internally) with three
    blobs, confirming the task's user-observable signal end-to-end:

    (a) a blob with legacy TOP-LEVEL infra-resume counters emits ZERO
        ``task_metadata.schema_warning`` census lines — the v1->v2
        migration lifts them into ``retry_ledger`` at parse-time;
    (b) a blob of Tier-A blessed conventional keys emits ZERO census lines
        — the ``_BLESSED_METADATA_KEYS`` allowlist suppresses them;
    (c) a blob with a genuine unknown key still emits exactly one census
        line, and that line now carries the ``code=unknown_key`` token —
        WORK ITEM 1's discriminator flowing through the real backend, with
        genuine drift still surfacing.

    This is the regression guard for
    ``grep 'task_metadata.schema_warning' | grep -v code=unknown_key``
    being near-empty in the live journal.
    """
    backend = await make_backend(enforce=False)
    root = str(tmp_path / 'capstone')
    dto = await backend.add_task(project_root=root, title='t')
    tid = int(dto['id'])

    # (a) legacy top-level infra-resume counters -- suppressed by the
    # v1->v2 migration at parse-time.
    legacy_counters = json.dumps({
        'consecutive_infra_resume_failures': 3,
        'last_infra_resume_iteration_count': 7,
    })
    with caplog.at_level(logging.WARNING, logger='fused_memory.backends.sqlite_task_backend'):
        caplog.clear()
        await backend._validate_metadata_on_write(
            legacy_counters, project_root=root, tag='master', task_id=tid,
        )
    assert _schema_warning_messages(caplog) == [], (
        'Expected zero census lines for legacy top-level infra-resume counters '
        '(the v1->v2 migration should lift them into retry_ledger at parse-time)'
    )

    # (b) a representative spread of Tier-A blessed conventional keys --
    # suppressed by the _BLESSED_METADATA_KEYS allowlist.
    blessed_blob = json.dumps({
        'source': 'x',
        'modules': ['a'],
        'complexity': 'simple',
        'prd_path': 'plans/x.md',
        'gate_escalated_at': '2026-01-01T00:00:00+00:00',
    })
    with caplog.at_level(logging.WARNING, logger='fused_memory.backends.sqlite_task_backend'):
        caplog.clear()
        await backend._validate_metadata_on_write(
            blessed_blob, project_root=root, tag='master', task_id=tid,
        )
    assert _schema_warning_messages(caplog) == [], (
        'Expected zero census lines for a Tier-A blessed conventional-key blob'
    )

    # (c) a genuine unknown key -- still surfaces, and now carries the new
    # code=unknown_key discriminator token.
    unknown_blob = json.dumps({'mystery_zzz': 'control'})
    with caplog.at_level(logging.WARNING, logger='fused_memory.backends.sqlite_task_backend'):
        caplog.clear()
        await backend._validate_metadata_on_write(
            unknown_blob, project_root=root, tag='master', task_id=tid,
        )
    census = _schema_warning_messages(caplog)
    assert len(census) == 1, f'Expected exactly one census line; got {census}'
    assert 'code=unknown_key' in census[0], f'Expected code=unknown_key token; got: {census[0]!r}'
    assert 'mystery_zzz' in census[0]


# ── Row 8 — curator-gate contradiction reaches the real write boundary ──


class TestCuratorGateContradictionAtWriteBoundary:
    """``human_curator_gate`` + ``before_done`` is rejected at the REAL boundary (task 3369).

    The unit tests in ``shared/tests/test_task_metadata.py`` exercise
    ``TaskMetadata`` and ``parse_metadata``; neither proves that
    :meth:`SqliteTaskBackend._validate_metadata_on_write` routes THIS finding
    to a raise. That routing is non-obvious and load-bearing: the method
    re-raises only when a warning's ``field`` is in ``incoming_keys`` OR is the
    whole-blob sentinel :data:`_WHOLE_METADATA_FIELD`, and only when the code
    is not in ``_NON_FATAL_WRITE_WARNING_CODES``. This invariant is a
    whole-model validator, so it produces ``loc == ()`` -> the sentinel ->
    unconditionally fatal. Row (c) is what that asymmetry buys: an
    ``update_task`` supplying ONLY ``human_curator_gate`` is still rejected
    even though ``before_done`` is an untouched legacy field never named in
    ``incoming_keys``.

    Rows (d)-(f) pin the OTHER edge of that same asymmetry: once a
    contradictory row exists — which needs a warn-mode write or a non-backend
    writer, see :meth:`_land_a_contradictory_row` — the sentinel locks out
    every later merge-mode write on it, and (e)/(f) are the two documented
    ways back out.

    Per this module's docstring these rows are an INTEGRATION gate, not a
    synthetic-input unit test: they may well be GREEN the moment the shared
    validator lands, because the generic policy already routes the sentinel
    correctly. That is the expected and desired outcome — a RED here would
    mean a genuine integration gap between the new invariant and the write
    boundary, which is exactly what this row exists to detect.
    """

    _BEFORE_DONE = {'script': 'scripts/deploy.sh', 'timeout_secs': 60}

    # A pure gate's marker on a task that also carries a machine step that
    # closes it — the self-contradiction the validator rejects.
    _CONTRADICTION = {
        'task_kind': 'deterministic',
        'always_escalates': True,
        'before_done': _BEFORE_DONE,
        'human_curator_gate': True,
    }

    # A legitimate deterministic deploy task: before_done, no marker. One of
    # the 33 shapes that exist on the live store today.
    _VALID_DEPLOY = {
        'task_kind': 'deterministic',
        'always_escalates': False,
        'before_done': _BEFORE_DONE,
    }

    @pytest.mark.asyncio
    async def test_add_task_with_both_keys_rejected_and_rolled_back(
        self, make_backend, tmp_path,
    ):
        """(a) enforce-mode add_task raises and leaves no row behind."""
        backend = await make_backend(enforce=True)
        project_root = str(tmp_path / 'enforce')

        with pytest.raises(ValidationError):
            await backend.add_task(
                project_root=project_root, title='t',
                metadata=json.dumps(self._CONTRADICTION),
            )

        # The rejected INSERT never landed — the _txn rolled back.
        tasks = (await backend.get_tasks(project_root=project_root))['tasks']
        assert tasks == []

    @pytest.mark.asyncio
    async def test_add_task_with_both_keys_warn_mode_writes_and_censuses(
        self, make_backend, tmp_path, caplog,
    ):
        """(b) warn-mode accepts the write and emits exactly one census line.

        The staged-rollout half of the contract, mirroring rows 5-6:
        ``task_metadata.enforce`` is a RED-TIER restart-only flag, so a
        deployment running in warn-mode must still surface the finding rather
        than swallowing it.
        """
        backend = await make_backend(enforce=False)
        project_root = str(tmp_path / 'warn')

        with caplog.at_level(
            logging.WARNING, logger='fused_memory.backends.sqlite_task_backend'
        ):
            # Scope the census to exactly this write.
            caplog.clear()
            dto = await backend.add_task(
                project_root=project_root, title='t',
                metadata=json.dumps(self._CONTRADICTION),
            )

        census = _schema_warning_messages(caplog)
        assert len(census) == 1, f'Expected exactly one census line; got {census}'
        assert '<metadata>' in census[0]
        assert 'human_curator_gate' in census[0]

        # The write proceeded, and I1 held through the durable round-trip.
        task = await backend.get_task(dto['id'], project_root=project_root)
        assert task['metadata'] == self._CONTRADICTION

    @pytest.mark.asyncio
    async def test_update_task_adding_only_the_marker_rejected(self, make_backend, tmp_path):
        """(c) the realistic reconciliation-Stage-2 write that motivated 3369.

        A VALID deterministic deploy task already exists; a later writer adds
        only ``human_curator_gate``. ``update_task`` shallow-merges and
        validates the merged WHOLE, and the whole-model sentinel bypasses
        ``incoming_keys`` scoping — so this is rejected even though the
        offending ``before_done`` is an untouched legacy field.
        """
        backend = await make_backend(enforce=True)
        project_root = str(tmp_path / 'enforce')

        dto = await backend.add_task(
            project_root=project_root, title='t',
            metadata=json.dumps(self._VALID_DEPLOY),
        )

        with pytest.raises(ValidationError):
            await backend.update_task(
                dto['id'], project_root=project_root,
                metadata=json.dumps({'human_curator_gate': True}),
            )

        # Rolled back — the marker never landed on the deploy task.
        task = await backend.get_task(dto['id'], project_root=project_root)
        assert task['metadata'] == self._VALID_DEPLOY

    async def _land_a_contradictory_row(self, make_backend, project_root: str) -> str:
        """Write a row carrying BOTH keys the only way that is still possible.

        Row (a) proves enforce-mode rejects this shape, so such a row can only
        exist because it was written while ``task_metadata.enforce`` was false
        — a RED-TIER restart-only flag, per row (b) — or by a writer that never
        went through :class:`SqliteTaskBackend` at all. A live-store census run
        for task 3369 found ZERO such rows today (tasks 3053/3063/3181/3234 all
        carry the marker with ``before_done`` absent), so rows (d)-(f) pin
        FORWARD risk, not a current breakage. Returns the task id.
        """
        warn_backend = await make_backend(enforce=False)
        dto = await warn_backend.add_task(
            project_root=project_root, title='t',
            metadata=json.dumps(self._CONTRADICTION),
        )
        return dto['id']

    @pytest.mark.asyncio
    async def test_unrelated_merge_on_a_contradictory_row_rejected(
        self, make_backend, tmp_path,
    ):
        """(d) the lock-out row (c)'s asymmetry implies — pinned, not incidental.

        The same whole-blob-sentinel-bypasses-``incoming_keys`` rule that makes
        row (c) desirable cuts the other way once a contradictory row EXISTS:
        every subsequent ``metadata_mode='merge'`` write on it is rejected,
        including writes with nothing to do with the contradiction. Here that
        is a plain ``files`` edit; the same applies to ``retry_ledger`` updates
        and any other merge-mode field write.

        This is the documented consequence of the design, not an accident — so
        it is asserted rather than left for a future reader to rediscover when
        a warn->enforce flip strands a row. Row (e) is the way out.
        """
        project_root = str(tmp_path / 'legacy')
        task_id = await self._land_a_contradictory_row(make_backend, project_root)

        enforce_backend = await make_backend(enforce=True)
        with pytest.raises(ValidationError) as excinfo:
            await enforce_backend.update_task(
                task_id, project_root=project_root,
                metadata=json.dumps({'files': ['orchestrator/src/orchestrator/thing.py']}),
                metadata_mode='merge',
            )
        # Pin the CAUSE, not just the raise: a bare `pytest.raises` here would
        # stay green if the `files` write started failing for some unrelated
        # reason, which would make this row prove nothing. (Row (e) supplies
        # the control: the identical write SUCCEEDS once the row is repaired.)
        assert 'human_curator_gate' in str(excinfo.value), str(excinfo.value)

        # Rolled back — the unrelated edit did not land either.
        task = await enforce_backend.get_task(task_id, project_root=project_root)
        assert task['metadata'] == self._CONTRADICTION

    @pytest.mark.asyncio
    async def test_contradictory_row_repaired_by_merging_a_falsy_marker(
        self, make_backend, tmp_path,
    ):
        """(e) the escape hatch out of (d), as documented in docs/task-authoring.md.

        The repair must ride along in the SAME write that clears the
        contradiction: an explicitly falsy ``human_curator_gate`` no longer
        contradicts ``before_done`` (the shared-side counterpart is
        ``test_falsy_curator_marker_with_before_done_accepted``), so the merged
        whole validates and the row unsticks. Asserting the FOLLOW-ON unrelated
        merge is the point — a repair that left the row still locked out would
        be no repair.
        """
        project_root = str(tmp_path / 'legacy')
        task_id = await self._land_a_contradictory_row(make_backend, project_root)
        enforce_backend = await make_backend(enforce=True)

        await enforce_backend.update_task(
            task_id, project_root=project_root,
            metadata=json.dumps({'human_curator_gate': False}),
            metadata_mode='merge',
        )
        task = await enforce_backend.get_task(task_id, project_root=project_root)
        assert task['metadata']['human_curator_gate'] is False
        # The legitimate deploy half is untouched — the repair clears the
        # contradiction, it does not discard the machine step.
        assert task['metadata']['before_done'] == self._BEFORE_DONE

        # ...and ordinary merge-mode writes work again.
        await enforce_backend.update_task(
            task_id, project_root=project_root,
            metadata=json.dumps({'files': ['orchestrator/src/orchestrator/thing.py']}),
            metadata_mode='merge',
        )
        task = await enforce_backend.get_task(task_id, project_root=project_root)
        assert task['metadata']['files'] == ['orchestrator/src/orchestrator/thing.py']

    @pytest.mark.asyncio
    async def test_contradictory_row_repaired_by_replace_mode(self, make_backend, tmp_path):
        """(f) the other documented repair: overwrite the whole blob.

        ``metadata_mode='replace'`` validates the blob the caller supplies
        rather than a merge of it with the stranded row, so a replacement that
        simply omits one of the two keys is accepted. Named alongside (e) in
        docs/task-authoring.md so an operator facing a stranded row has a path
        whether or not they want to keep the marker.
        """
        project_root = str(tmp_path / 'legacy')
        task_id = await self._land_a_contradictory_row(make_backend, project_root)
        enforce_backend = await make_backend(enforce=True)

        await enforce_backend.update_task(
            task_id, project_root=project_root,
            metadata=json.dumps(self._VALID_DEPLOY),
            metadata_mode='replace',
        )

        task = await enforce_backend.get_task(task_id, project_root=project_root)
        assert task['metadata'] == self._VALID_DEPLOY


class TestSliceCardinalityAtWriteBoundary:
    """A list-shaped ``milestone`` is censused and rejected at the REAL boundary (task 4142).

    ``shared/tests/test_task_metadata.py`` proves ``parse_metadata`` emits the
    new ``wrong_cardinality`` finding; neither that nor
    ``test_capability_manifest.py`` proves the fused-memory write boundary
    ROUTES it to a census line and a raise. That routing is what the task's
    "why it matters" claim rests on, and it is entirely generic: the new code
    is fatal only because ``_NON_FATAL_WRITE_WARNING_CODES`` is
    ``frozenset({'unknown_key'})`` and ``milestone`` is in ``incoming_keys``.
    No ``sqlite_task_backend.py`` change was made — these rows are what
    verifies that, rather than assuming it.

    Before the fix, the blob in ``_LIST_MILESTONE`` produced ZERO warnings, so
    it was invisible to BOTH the census and the enforce gate and landed in the
    row; the resulting non-dict ``metadata.milestone`` then made
    ``scheduler._milestone_time_gated`` fail-safe-withhold the task from
    dispatch indefinitely, with no escalation path.

    Rows (d)-(e) cover the case the live store actually cares about: a row that
    ALREADY carries the bad shape, written before the gate existed (see
    :meth:`_land_a_list_milestone_row`). They pin the mirror image of
    :class:`TestCuratorGateContradictionAtWriteBoundary`'s (d)-(f): this
    finding is field-scoped, not the whole-blob sentinel, so an existing bad
    row is NOT locked out of unrelated merge-mode writes — which is what makes
    flipping this code fatal safe for the 378 live rows carrying the slice.
    (e) is the repair, and asserts the row is genuinely CLEAN afterwards rather
    than merely tolerated.

    Per this module's docstring these rows are an INTEGRATION gate, not a
    synthetic-input unit test: they are expected to be GREEN the moment the
    shared change lands, because the generic fatal-code policy already routes
    any new code correctly. That is the desired outcome — a RED here would
    mean a genuine integration gap.
    """

    # The verbatim repro blob: a list where a single mapping is declared.
    _LIST_MILESTONE = {'milestone': [{'mode': 'delayed', 'after_secs': 604800}]}

    # The same slice, correctly shaped.
    _DICT_MILESTONE = {'milestone': {'mode': 'delayed', 'after_secs': 604800}}

    @pytest.mark.asyncio
    async def test_list_milestone_censused_with_wrong_cardinality_code(
        self, make_backend, tmp_path, caplog,
    ):
        """(a) exactly one census line, carrying the code and the field.

        The census line is emitted BEFORE the enforce-mode re-raise
        (``_validate_metadata_on_write`` censuses every warning, then decides
        whether to raise), so an enforce-mode backend surfaces the finding to
        an operator as well as rejecting it.
        """
        backend = await make_backend(enforce=True)
        project_root = str(tmp_path / 'enforce')

        with caplog.at_level(
            logging.WARNING, logger='fused_memory.backends.sqlite_task_backend'
        ):
            # Scope the census to exactly this write.
            caplog.clear()
            with pytest.raises(TypeError):
                await backend.add_task(
                    project_root=project_root, title='t',
                    metadata=json.dumps(self._LIST_MILESTONE),
                )

        census = _schema_warning_messages(caplog)
        assert len(census) == 1, f'Expected exactly one census line; got {census}'
        assert 'code=wrong_cardinality' in census[0]
        assert 'milestone' in census[0]

    @pytest.mark.asyncio
    async def test_list_milestone_add_rejected_and_rolled_back(self, make_backend, tmp_path):
        """(b1) the defect can no longer LAND: add_task raises, no row behind."""
        backend = await make_backend(enforce=True)
        project_root = str(tmp_path / 'enforce')

        with pytest.raises(TypeError):
            await backend.add_task(
                project_root=project_root, title='t',
                metadata=json.dumps(self._LIST_MILESTONE),
            )

        tasks = (await backend.get_tasks(project_root=project_root))['tasks']
        assert tasks == []

    @pytest.mark.asyncio
    async def test_list_milestone_update_rejected_row_unchanged(self, make_backend, tmp_path):
        """(b2) a later writer cannot corrupt a good row's milestone either.

        The realistic shape of the defect: a well-formed row already exists
        and a subsequent writer supplies the list. ``milestone`` is in
        ``incoming_keys``, so the generic re-raise gate fires and ``_txn``
        rolls back — a follow-up ``get_task`` shows the dict value intact.
        """
        backend = await make_backend(enforce=True)
        project_root = str(tmp_path / 'enforce')

        dto = await backend.add_task(
            project_root=project_root, title='t',
            metadata=json.dumps(self._DICT_MILESTONE),
        )

        with pytest.raises(TypeError):
            await backend.update_task(
                dto['id'], project_root=project_root,
                metadata=json.dumps(self._LIST_MILESTONE),
                metadata_mode='merge',
            )

        task = await backend.get_task(dto['id'], project_root=project_root)
        assert task['metadata'] == self._DICT_MILESTONE

    @pytest.mark.asyncio
    async def test_well_shaped_dict_milestone_writes_clean(
        self, make_backend, tmp_path, caplog,
    ):
        """(c) the happy path is untouched: no census, no raise, I1 holds."""
        backend = await make_backend(enforce=True)
        project_root = str(tmp_path / 'enforce')

        with caplog.at_level(
            logging.WARNING, logger='fused_memory.backends.sqlite_task_backend'
        ):
            caplog.clear()
            dto = await backend.add_task(
                project_root=project_root, title='t',
                metadata=json.dumps(self._DICT_MILESTONE),
            )

        assert _schema_warning_messages(caplog) == []

        task = await backend.get_task(dto['id'], project_root=project_root)
        assert task['metadata'] == self._DICT_MILESTONE

    async def _land_a_list_milestone_row(self, make_backend, project_root: str) -> str:
        """Write a row carrying the bad shape the only way that is still possible.

        Rows (a)/(b1) prove enforce-mode rejects this blob, so such a row can
        only exist because it was written while ``task_metadata.enforce`` was
        false — a RED-TIER restart-only flag — or by a writer that never went
        through :class:`SqliteTaskBackend`. A live-store census for task 4142
        found ZERO mismatched rows among the 378 carrying ``milestone`` or
        ``delivered_checks``, so rows (d)-(e) pin FORWARD risk, not a current
        breakage. Returns the task id.
        """
        warn_backend = await make_backend(enforce=False)
        dto = await warn_backend.add_task(
            project_root=project_root, title='t',
            metadata=json.dumps(self._LIST_MILESTONE),
        )
        # The warn-mode write really did land the bad shape (I1: the raw list
        # is retained unswapped) — otherwise (d)/(e) would prove nothing.
        landed = await warn_backend.get_task(dto['id'], project_root=project_root)
        assert landed['metadata'] == self._LIST_MILESTONE
        return dto['id']

    @pytest.mark.asyncio
    async def test_unrelated_merge_on_a_list_milestone_row_succeeds(
        self, make_backend, tmp_path, caplog,
    ):
        """(d) an already-bad row is NOT stranded — the fatal code is field-scoped.

        This is the assertion that makes flipping ``wrong_cardinality`` fatal
        safe under the already-live ``task_metadata.enforce: true``: the
        warning's ``field`` is ``milestone``, not the whole-blob sentinel, so
        the ``w.field in incoming_keys`` gate leaves a ``files`` edit alone.
        Deliberately the OPPOSITE outcome from
        ``TestCuratorGateContradictionAtWriteBoundary`` row (d), where a
        whole-model invariant DOES lock the row out — recorded here so the
        difference reads as designed rather than accidental.

        Non-fatal is not silent, though: the census line still fires on every
        write that touches the row, which is the operator's signal to repair it.
        """
        project_root = str(tmp_path / 'legacy')
        task_id = await self._land_a_list_milestone_row(make_backend, project_root)

        enforce_backend = await make_backend(enforce=True)
        with caplog.at_level(
            logging.WARNING, logger='fused_memory.backends.sqlite_task_backend'
        ):
            caplog.clear()
            await enforce_backend.update_task(
                task_id, project_root=project_root,
                metadata=json.dumps({'files': ['orchestrator/src/orchestrator/thing.py']}),
                metadata_mode='merge',
            )

        census = _schema_warning_messages(caplog)
        assert len(census) == 1, f'Expected exactly one census line; got {census}'
        assert 'code=wrong_cardinality' in census[0]

        # The unrelated edit landed, and the untouched bad slice is retained
        # verbatim rather than being coerced or dropped behind the operator.
        task = await enforce_backend.get_task(task_id, project_root=project_root)
        assert task['metadata']['files'] == ['orchestrator/src/orchestrator/thing.py']
        assert task['metadata']['milestone'] == self._LIST_MILESTONE['milestone']

    @pytest.mark.asyncio
    async def test_list_milestone_row_repaired_by_merging_a_dict_milestone(
        self, make_backend, tmp_path, caplog,
    ):
        """(e) the repair path: a merge supplying the correct dict fixes the row.

        The merged whole validates clean, so ``milestone`` being in
        ``incoming_keys`` costs nothing — the write a human or a fixer script
        would make is exactly the write that is accepted. The follow-on
        unrelated merge is the point: it now emits ZERO census lines, proving
        the row is genuinely repaired rather than merely tolerated the way (d)
        tolerates it.
        """
        project_root = str(tmp_path / 'legacy')
        task_id = await self._land_a_list_milestone_row(make_backend, project_root)
        enforce_backend = await make_backend(enforce=True)

        await enforce_backend.update_task(
            task_id, project_root=project_root,
            metadata=json.dumps(self._DICT_MILESTONE),
            metadata_mode='merge',
        )
        task = await enforce_backend.get_task(task_id, project_root=project_root)
        assert task['metadata']['milestone'] == self._DICT_MILESTONE['milestone']

        with caplog.at_level(
            logging.WARNING, logger='fused_memory.backends.sqlite_task_backend'
        ):
            caplog.clear()
            await enforce_backend.update_task(
                task_id, project_root=project_root,
                metadata=json.dumps({'files': ['orchestrator/src/orchestrator/thing.py']}),
                metadata_mode='merge',
            )

        assert _schema_warning_messages(caplog) == []
        task = await enforce_backend.get_task(task_id, project_root=project_root)
        assert task['metadata']['files'] == ['orchestrator/src/orchestrator/thing.py']
