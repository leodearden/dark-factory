"""Direct-call tests for ``middleware/done_provenance.py``.

The judgement is driven through its public functions with an injected git
runner and stamp clock. Tests of the same rules reached through
``TaskInterceptor.set_task_status`` stay in test_task_interceptor.py.
"""

import logging
from pathlib import Path
from typing import get_args

import pytest
from _fm_git_repos import _init_git_repo
from _import_closure import SRC, modules_loaded_by_importing
from _provenance_stamp_clock import FIXED_STAMP, fixed_stamp_clock
from shared.git_async import run_git
from shared.task_metadata import DoneProvenance
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

from fused_memory.config.schema import FusedMemoryConfig, ReconciliationConfig
from fused_memory.middleware.done_provenance import (
    StaleEvidenceMode,
    validate_done_provenance,
)

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

# ── Layering of middleware/done_provenance.py (task 6236) ──


def test_done_provenance_module_imports_without_the_interceptor():
    """The done-provenance judgement layers below its consumer: importing it
    in a fresh interpreter never loads task_interceptor."""
    probe = modules_loaded_by_importing(
        'fused_memory.middleware.done_provenance',
        ('fused_memory.middleware.task_interceptor',),
    )

    assert Path(probe.file).resolve() == SRC / 'fused_memory' / 'middleware' / 'done_provenance.py'
    assert probe.loaded == ()


# ── Tests for the 'operational-verified' done_provenance kind (task 2679) ──
#
# Commitless kind for no-code operational tasks (2648/2650 task_kind-thrash
# class): requires escalation_id (the resolving escalation, recorded
# verbatim) + note, and is accepted only from non-recon-stage callers (reuse
# of recon_write_policy's caller classification) -- a recon stage must not
# self-authorize an operational close.


@pytest.mark.asyncio
async def test_validate_done_provenance_accepts_operational_verified(tmp_path):
    """Direct-call: non-recon caller with escalation_id+note is accepted commitless."""
    err, resolved = await validate_done_provenance(
        '1',
        {'kind': 'operational-verified', 'escalation_id': 'esc-123', 'note': 'restarted fused-memory'},
        str(tmp_path),
        require=False,
        is_recon_stage=False,
        git_runner=run_git,
        stamp_clock=fixed_stamp_clock,
    )

    assert err is None
    assert resolved is not None
    assert resolved == {
        'kind': 'operational-verified',
        'escalation_id': 'esc-123',
        'note': 'restarted fused-memory',
    }
    assert 'commit' not in resolved


@pytest.mark.asyncio
async def test_validate_done_provenance_operational_verified_requires_escalation_id(tmp_path):
    """Direct-call: missing escalation_id is a structured rejection."""
    err, resolved = await validate_done_provenance(
        '1',
        {'kind': 'operational-verified', 'note': 'restarted fused-memory'},
        str(tmp_path),
        require=False,
        is_recon_stage=False,
        git_runner=run_git,
        stamp_clock=fixed_stamp_clock,
    )

    assert resolved is None
    assert err is not None
    assert err['error'] == 'done_provenance_invalid'
    assert 'escalation_id' in err['reason']


@pytest.mark.asyncio
async def test_validate_done_provenance_operational_verified_requires_note(tmp_path):
    """Direct-call: missing note is a structured rejection."""
    err, resolved = await validate_done_provenance(
        '1',
        {'kind': 'operational-verified', 'escalation_id': 'esc-123'},
        str(tmp_path),
        require=False,
        is_recon_stage=False,
        git_runner=run_git,
        stamp_clock=fixed_stamp_clock,
    )

    assert resolved is None
    assert err is not None
    assert err['error'] == 'done_provenance_invalid'
    assert 'note' in err['reason']


@pytest.mark.asyncio
async def test_validate_done_provenance_operational_verified_rejects_recon_stage(tmp_path):
    """Direct-call: a recon-stage caller cannot record operational-verified,
    even with both required fields present."""
    err, resolved = await validate_done_provenance(
        '1',
        {'kind': 'operational-verified', 'escalation_id': 'esc-123', 'note': 'restarted fused-memory'},
        str(tmp_path),
        require=False,
        is_recon_stage=True,
        git_runner=run_git,
        stamp_clock=fixed_stamp_clock,
    )

    assert resolved is None
    assert err is not None
    assert err['error'] == 'done_provenance_invalid'
    assert 'recon' in err['reason'].lower()


@pytest.mark.asyncio
async def test_validate_done_provenance_accepts_every_declared_kind(tmp_path):
    """Every kind declared by DoneProvenance.kind is ACCEPTED given its
    minimal valid payload (behavioral acceptance, not message wording).

    The rejection paths are covered elsewhere
    (test_task_interceptor.py::test_done_provenance_rejects_missing_kind,
    the invalid-kind repair test), and 'merged'/'found_on_main'/'operational-verified' each have
    dedicated tests, but nothing previously asserted that
    'deterministic-gate' and 'deterministic-milestone' — commitless,
    runner-stamped kinds with no dedicated acceptance test at this boundary
    — are actually accepted. The lockstep assertion below means an 8th kind
    added to the Literal without a matching entry here fails loudly (a
    KeyError-free, explicit mismatch) instead of being silently skipped;
    a conditional-requirement branch that accidentally rejects an
    already-covered kind fails just as loudly via the per-kind assertion.
    """
    sha = _init_git_repo(tmp_path)  # backs the 'merged'/'found_on_main' commit field

    minimal_payloads = {
        'merged': {'kind': 'merged', 'commit': sha},
        'found_on_main': {
            'kind': 'found_on_main',
            'commit': sha,
            'note': 'covered by a sibling task',
        },
        'deterministic-deploy': {'kind': 'deterministic-deploy'},
        'deterministic-deploy-scheduled': {'kind': 'deterministic-deploy-scheduled'},
        # escalation_id is REQUIRED for this kind (task 5241) — it cites the
        # resolving gate escalation. This table's lockstep assertion is the
        # repo's own guard that it tracks the contract, so it is corrected
        # here rather than the requirement being relaxed.
        'deterministic-gate': {'kind': 'deterministic-gate', 'escalation_id': 'esc-123'},
        'deterministic-milestone': {'kind': 'deterministic-milestone'},
        'operational-verified': {
            'kind': 'operational-verified',
            'escalation_id': 'esc-123',
            'note': 'restarted fused-memory',
        },
    }

    declared_kinds = get_args(DoneProvenance.model_fields['kind'].annotation)
    assert set(minimal_payloads) == set(declared_kinds), (
        'minimal_payloads must stay in lockstep with the DoneProvenance.kind '
        f'Literal; declared={sorted(declared_kinds)} vs covered='
        f'{sorted(minimal_payloads)}'
    )

    for kind in declared_kinds:
        err, resolved = await validate_done_provenance(
            '1', minimal_payloads[kind], str(tmp_path), require=True,
            # An AUTHORIZED caller (task 5241): the deterministic-* kinds are
            # accepted only from an allowlisted, non-recon-stage identity, so
            # this table tests the SHAPE contract rather than the caller bar,
            # which test_task_interceptor.py::TestDeterministicProvenanceCallerBar
            # owns.
            agent_id='orchestrator', config=FusedMemoryConfig(),
            git_runner=run_git,
            stamp_clock=fixed_stamp_clock,
        )
        assert err is None, f'kind={kind!r} minimal payload rejected: {err}'
        assert resolved is not None, f'kind={kind!r} resolved to None with no error'
        assert resolved['kind'] == kind


# ── Tests for the server-written `stamped_at` field (task 3576) ──
#
# `validate_done_provenance` is the SINGLE write site: every found_on_main
# producer funnels through it (the fresh-done path, the same-status repair
# seam, agent-authored set_task_status calls, and the orchestrator's
# Scheduler.mark_done). The field is scoped to kind='found_on_main' only —
# it is the attribution-by-inference kind the soak gate watches; `merged`
# already carries independent landing evidence.


@pytest.mark.asyncio
async def test_validate_done_provenance_stamps_found_on_main(tmp_path):
    """Direct-call: a valid found_on_main blob comes back carrying stamped_at."""
    sha = _init_git_repo(tmp_path)

    err, resolved = await validate_done_provenance(
        '1',
        {'kind': 'found_on_main', 'commit': sha, 'note': 'sibling task 99 landed this'},
        str(tmp_path),
        require=False,
        git_runner=run_git,
        stamp_clock=fixed_stamp_clock,
    )

    assert err is None
    assert resolved is not None
    assert resolved['stamped_at'] == FIXED_STAMP


@pytest.mark.asyncio
async def test_validate_done_provenance_does_not_stamp_merged(tmp_path):
    """kind='merged' is deliberately NOT stamped — the field is found_on_main-scoped.

    A leak here would perturb the exact-equality merged-kind assertions
    elsewhere in this suite, which is precisely why the scoping exists.
    """
    sha = _init_git_repo(tmp_path)

    err, resolved = await validate_done_provenance(
        '1',
        {'kind': 'merged', 'commit': sha},
        str(tmp_path),
        require=False,
        git_runner=run_git,
        stamp_clock=fixed_stamp_clock,
    )

    assert err is None
    assert resolved is not None
    assert 'stamped_at' not in resolved


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'raw',
    [
        {'kind': 'deterministic-deploy', 'pid': 4242, 'unit': 'fused-memory.service'},
        {'kind': 'deterministic-deploy-scheduled', 'unit': 'fused-memory.service'},
        # escalation_id is required for the gate kind (task 5241).
        {
            'kind': 'deterministic-gate',
            'note': 'pure gate resolved',
            'escalation_id': 'esc-123',
        },
        {'kind': 'deterministic-milestone'},
        {'kind': 'operational-verified', 'escalation_id': 'esc-123', 'note': 'restarted'},
    ],
    ids=['deploy', 'deploy-scheduled', 'gate', 'milestone', 'operational-verified'],
)
async def test_validate_done_provenance_does_not_stamp_other_kinds(
    tmp_path, raw
):
    """Every non-found_on_main kind omits stamped_at."""
    err, resolved = await validate_done_provenance(
        '1',
        raw,
        str(tmp_path),
        require=False,
        is_recon_stage=False,
        agent_id='orchestrator',
        config=FusedMemoryConfig(),
        git_runner=run_git,
        stamp_clock=fixed_stamp_clock,
    )

    assert err is None, f'expected acceptance but got: {err}'
    assert resolved is not None
    assert 'stamped_at' not in resolved


@pytest.mark.asyncio
async def test_validate_done_provenance_discards_caller_supplied_stamp(
    tmp_path, caplog
):
    """A caller-supplied stamped_at is overwritten by the server value — loudly.

    Warn-and-overwrite rather than reject: the same-status repair seam
    re-submits a previously-STORED blob, which post-3576 carries its own
    stamped_at, so a hard rejection would break every repair of a stamped
    blob. The warning keeps the override visible instead of silent.
    """
    sha = _init_git_repo(tmp_path)
    forged = '1999-01-01T00:00:00+00:00'

    with caplog.at_level(logging.WARNING):
        err, resolved = await validate_done_provenance(
            '4242',
            {
                'kind': 'found_on_main',
                'commit': sha,
                'note': 'sibling task 99 landed this',
                'stamped_at': forged,
            },
            str(tmp_path),
            require=False,
            git_runner=run_git,
            stamp_clock=fixed_stamp_clock,
        )

    assert err is None
    assert resolved is not None
    assert resolved['stamped_at'] == FIXED_STAMP
    assert resolved['stamped_at'] != forged

    warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    matching = [m for m in warnings if 'stamped_at' in m]
    assert matching, f'expected a stamped_at override warning, got: {warnings}'
    assert '4242' in matching[0], 'warning must name the task id'
    assert forged in matching[0], 'warning must name the discarded value'


@pytest.mark.asyncio
async def test_validate_done_provenance_no_stamp_warning_when_absent(
    tmp_path, caplog
):
    """The ordinary path (no caller-supplied stamp) stays quiet."""
    sha = _init_git_repo(tmp_path)

    with caplog.at_level(logging.WARNING):
        err, resolved = await validate_done_provenance(
            '1',
            {'kind': 'found_on_main', 'commit': sha, 'note': 'sibling task 99 landed this'},
            str(tmp_path),
            require=False,
            git_runner=run_git,
            stamp_clock=fixed_stamp_clock,
        )

    assert err is None
    assert resolved is not None
    assert not [
        r.message
        for r in caplog.records
        if r.levelno >= logging.WARNING and 'stamped_at' in r.message
    ]


def test_stale_evidence_mode_names_the_configured_modes():
    """The reopen-freshness gate's mode type and the config field feeding it
    name the same modes, so a mode the config accepts is one the gate types."""
    field = ReconciliationConfig.model_fields['reject_stale_done_evidence']

    assert get_args(StaleEvidenceMode) == get_args(field.annotation)
