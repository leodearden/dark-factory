"""Tests for the failed-only merge-verify retry producer (PRD verify-retry-failed-only D2).

The DF orchestrator PRODUCES the reify retry contract: per-profile nextest
filter files ({did-not-pass} exact ids) written into the shared merge_wt, plus
the brand-new REIFY_VERIFY_RETRY_* / REIFY_RUN_ALL_MEMBER_SUBSET /
REIFY_GUI_RETRY_SPECS env keys threaded through MergeVerifySpec.verify_env.
reify's verify.sh (α/β/γ) is the CONSUMER — out of scope here.

Covers:
  * ``_build_retry_verify_env`` — writes the debug/release nextest filter files
    and returns the REIFY_* env dict (this module, step 5/6).
  * ``_assemble_retry_verify_env`` — INV-3 tree-OID corroboration gate
    (step 7/8).
  * ``_run_post_merge_verify`` wiring under ``req.retry_failed_only`` (step 9/10).
"""
from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from test_merge_queue_concurrent_verify import _make_request, _mock_verify_result

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.verify_runner import build_merge_verify_spec

if TYPE_CHECKING:
    from orchestrator.merge_queue import MergeRequest
    from orchestrator.verify_runner import MergeVerifySpec

# The reify-written attempt-0 sidecar contract (path + schema).  reify owns the
# authoritative schema (verify-retry-failed-only α/β/γ); this producer reads it
# tolerantly and stays a no-op until reify lands, so these strings define the
# DF-side expectation the loader is written against.
_SIDECAR_SUBDIR = '.reify-verify-retry'
_SIDECAR_NAME = 'attempt0.json'


def _attempt0(tree_oid: str):
    """A fixture attempt-0 payload with fail-fast-cancelled tests per profile."""
    from orchestrator.merge_queue import _Attempt0Payload

    return _Attempt0Payload(
        tree_oid=tree_oid,
        # debug: 'c a::z' cancelled by fail-fast (absent from verdicts) → not-started
        debug_planned=['c a::x', 'c a::y', 'c a::z'],
        debug_verdicts={'c a::x': 'pass', 'c a::y': 'fail'},
        # release: 'c b::q' cancelled → not-started
        release_planned=['c b::p', 'c b::q'],
        release_verdicts={'c b::p': 'pass'},
        run_all_members=['mem_fail'],
        gui_specs=['ui/x.ts'],
    )


def _git_ops_returning(oid: str | None):
    git_ops = MagicMock()
    git_ops.get_head_tree_hash = AsyncMock(return_value=oid)
    return git_ops


@pytest.mark.asyncio
async def test_assemble_retry_verify_env_tree_pinned(tmp_path: Path) -> None:
    """Case A: current tree OID matches attempt-0 → build the retry env.

    The nextest filter files carry the {did-not-pass} ids (failed ∪ not-started),
    demonstrating the soundness core end-to-end through the gate.
    """
    from orchestrator.merge_queue import _assemble_retry_verify_env

    git_ops = _git_ops_returning('abc123')
    req = cast('MergeRequest', SimpleNamespace(task_id='t-1', retry_failed_only=True))
    env = await _assemble_retry_verify_env(git_ops, req, tmp_path, _attempt0('abc123'))

    assert env is not None
    assert env['REIFY_VERIFY_RETRY_SCOPE'] == 'failed_only'
    assert env['REIFY_VERIFY_RETRY_TREE_OID'] == 'abc123'

    debug_path = Path(env['REIFY_VERIFY_RETRY_NEXTEST_FILTER_FILE_DEBUG'])
    release_path = Path(env['REIFY_VERIFY_RETRY_NEXTEST_FILTER_FILE_RELEASE'])
    assert tmp_path in debug_path.parents  # written under merge_wt
    # {did-not-pass} = failed ∪ not-started (NOT just failed).
    assert debug_path.read_text() == 'c a::y\nc a::z'
    assert release_path.read_text() == 'c b::q'
    git_ops.get_head_tree_hash.assert_awaited_once_with(tmp_path)


@pytest.mark.asyncio
async def test_assemble_retry_verify_env_rebased_returns_none(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Case B: a rebased tree (OID mismatch) → None + WARNING; defer to full verify."""
    from orchestrator.merge_queue import _assemble_retry_verify_env

    git_ops = _git_ops_returning('different-oid')
    req = cast('MergeRequest', SimpleNamespace(task_id='t-2', retry_failed_only=True))
    with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
        env = await _assemble_retry_verify_env(git_ops, req, tmp_path, _attempt0('abc123'))

    assert env is None
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any('full verify' in m for m in warnings), warnings
    assert any(('rebas' in m or 'does not match' in m) for m in warnings), warnings


@pytest.mark.asyncio
async def test_assemble_retry_verify_env_unknown_tree_returns_none(tmp_path: Path) -> None:
    """Case C: get_head_tree_hash returns None → None (fail-safe full verify)."""
    from orchestrator.merge_queue import _assemble_retry_verify_env

    git_ops = _git_ops_returning(None)
    req = cast('MergeRequest', SimpleNamespace(task_id='t-3', retry_failed_only=True))
    env = await _assemble_retry_verify_env(git_ops, req, tmp_path, _attempt0('abc123'))
    assert env is None


def test_build_retry_verify_env_writes_filter_files_and_env(tmp_path: Path) -> None:
    """_build_retry_verify_env writes per-profile filter files + the REIFY_* env.

    The nextest subsets (potentially thousands of ids) ship as newline filter
    FILES; the small run_all-member / gui-spec lists ship as comma-delimited
    env VALUES; tree OID + scope ship as env values.
    """
    from orchestrator.merge_queue import _build_retry_verify_env

    debug = ['c a::y', 'c a::z']
    release = ['c b::q']
    env = _build_retry_verify_env(
        nextest_subset_debug=debug,
        nextest_subset_release=release,
        run_all_members=['mem1', 'mem2'],
        gui_specs=['ui/spec_a.ts'],
        tree_oid='deadbeef',
        filter_dir=tmp_path,
    )

    # (2) filter-file env keys are absolute paths under filter_dir.
    debug_path = Path(env['REIFY_VERIFY_RETRY_NEXTEST_FILTER_FILE_DEBUG'])
    release_path = Path(env['REIFY_VERIFY_RETRY_NEXTEST_FILTER_FILE_RELEASE'])
    assert debug_path.is_absolute()
    assert release_path.is_absolute()
    assert tmp_path in debug_path.parents
    assert tmp_path in release_path.parents

    # (1) the two filter files exist with EXACTLY the newline-joined ids.
    assert debug_path.read_text() == 'c a::y\nc a::z'
    assert release_path.read_text() == 'c b::q'

    # (3) run_all members / gui specs ship comma-delimited.
    assert env['REIFY_RUN_ALL_MEMBER_SUBSET'] == 'mem1,mem2'
    assert env['REIFY_GUI_RETRY_SPECS'] == 'ui/spec_a.ts'

    # (4) tree OID + scope.
    assert env['REIFY_VERIFY_RETRY_TREE_OID'] == 'deadbeef'
    assert env['REIFY_VERIFY_RETRY_SCOPE'] == 'failed_only'


def test_build_retry_verify_env_empty_subsets_still_write_files(tmp_path: Path) -> None:
    """Empty nextest subsets still write (empty) filter files and set env keys.

    The contract is deterministic: reify's verify.sh always finds the filter
    files at the advertised paths, even when a profile has nothing to retry.
    """
    from orchestrator.merge_queue import _build_retry_verify_env

    env = _build_retry_verify_env(
        nextest_subset_debug=[],
        nextest_subset_release=[],
        run_all_members=[],
        gui_specs=[],
        tree_oid='cafef00d',
        filter_dir=tmp_path,
    )

    debug_path = Path(env['REIFY_VERIFY_RETRY_NEXTEST_FILTER_FILE_DEBUG'])
    release_path = Path(env['REIFY_VERIFY_RETRY_NEXTEST_FILTER_FILE_RELEASE'])
    assert debug_path.exists()
    assert release_path.exists()
    assert debug_path.read_text() == ''
    assert release_path.read_text() == ''
    assert env['REIFY_RUN_ALL_MEMBER_SUBSET'] == ''
    assert env['REIFY_GUI_RETRY_SPECS'] == ''
    assert env['REIFY_VERIFY_RETRY_TREE_OID'] == 'cafef00d'
    assert env['REIFY_VERIFY_RETRY_SCOPE'] == 'failed_only'


# ---------------------------------------------------------------------------
# _run_post_merge_verify wiring (step 9/10): guarded by req.retry_failed_only,
# merges the assembled retry env into MergeVerifySpec.verify_env before dispatch.
# Driven on the LOCAL path (runner=None) so task 2822's remote-green cross-check
# block (runs ONLY when runner is not None) stays inert and the wiring is isolated.
# ---------------------------------------------------------------------------


def _make_git_ops_mock() -> MagicMock:
    m = MagicMock()
    m.get_main_sha = AsyncMock(return_value='main-sha')
    m.get_free_disk_bytes = AsyncMock(return_value=100 * 1024 ** 3)
    m.cleanup_merge_worktree = AsyncMock()
    m.create_throwaway_verify_worktree = AsyncMock(return_value='/repo/_throwaway')
    m.get_head_tree_hash = AsyncMock(return_value='deadbeef')
    return m


def _write_attempt0_sidecar(merge_wt: Path) -> None:
    """Write a minimal schema-valid attempt-0 sidecar so the real loader succeeds."""
    d = merge_wt / _SIDECAR_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    (d / _SIDECAR_NAME).write_text(
        json.dumps(
            {
                'tree_oid': 'deadbeef',
                'debug': {'planned': ['c a::x', 'c a::y'], 'verdicts': {'c a::x': 'pass'}},
                'release': {'planned': [], 'verdicts': {}},
                'run_all_members': [],
                'gui_specs': [],
            }
        )
    )


@pytest.mark.asyncio
async def test_run_post_merge_verify_wires_retry_env_when_flag_on(tmp_path: Path) -> None:
    """retry_failed_only=True → the assembled retry env is MERGED into spec.verify_env."""
    from orchestrator import merge_queue as mq

    config = OrchestratorConfig(project_root=tmp_path, git=GitConfig(main_branch='main'))
    req = dataclasses.replace(
        _make_request('r-on', 'task/r-on', tmp_path, config), retry_failed_only=True
    )
    git_ops = _make_git_ops_mock()
    _write_attempt0_sidecar(tmp_path)

    retry_env = {
        'REIFY_VERIFY_RETRY_SCOPE': 'failed_only',
        'REIFY_VERIFY_RETRY_TREE_OID': 'deadbeef',
    }
    captured: dict[str, MergeVerifySpec] = {}

    async def _fake_dispatch(merge_sha, spec, **kw):  # type: ignore[no-untyped-def]
        captured['spec'] = spec
        return _mock_verify_result(True)

    with patch.object(mq, '_ensure_verify_disk_space', new=AsyncMock(return_value=None)), \
         patch.object(mq.VerifyRunnerPool, 'dispatch', new=AsyncMock(side_effect=_fake_dispatch)), \
         patch.object(
             mq, '_assemble_retry_verify_env', new=AsyncMock(return_value=retry_env)
         ) as assemble_mock:
        outcome = await mq._run_post_merge_verify(
            git_ops, req, tmp_path,
            timeouts={}, enospc_retries={},
            max_timeouts=2, max_enospc=1,
            merge_sha='abc123', runner=None,
        )

    assert outcome is None  # verify passed
    assemble_mock.assert_awaited_once()
    spec = captured['spec']
    # (1) retry env keys present — merged, not dropped.
    for k, v in retry_env.items():
        assert spec.verify_env[k] == v
    # (2) pre-existing effective_verify_env keys preserved (merge, NOT replace).
    base = build_merge_verify_spec(config, req.module_configs, None)
    for k, v in base.verify_env.items():
        assert spec.verify_env[k] == v


@pytest.mark.asyncio
async def test_run_post_merge_verify_flag_off_is_byte_identical(tmp_path: Path) -> None:
    """retry_failed_only=False (D1 no-op parity) → assemble not called, spec unchanged."""
    from orchestrator import merge_queue as mq

    config = OrchestratorConfig(project_root=tmp_path, git=GitConfig(main_branch='main'))
    req = _make_request('r-off', 'task/r-off', tmp_path, config)  # flag defaults False
    git_ops = _make_git_ops_mock()

    captured: dict[str, MergeVerifySpec] = {}

    async def _fake_dispatch(merge_sha, spec, **kw):  # type: ignore[no-untyped-def]
        captured['spec'] = spec
        return _mock_verify_result(True)

    with patch.object(mq, '_ensure_verify_disk_space', new=AsyncMock(return_value=None)), \
         patch.object(mq.VerifyRunnerPool, 'dispatch', new=AsyncMock(side_effect=_fake_dispatch)), \
         patch.object(mq, '_assemble_retry_verify_env', new=AsyncMock()) as assemble_mock:
        outcome = await mq._run_post_merge_verify(
            git_ops, req, tmp_path,
            timeouts={}, enospc_retries={},
            max_timeouts=2, max_enospc=1,
            merge_sha='abc123', runner=None,
        )

    assert outcome is None
    assemble_mock.assert_not_awaited()
    base = build_merge_verify_spec(config, req.module_configs, None)
    assert dict(captured['spec'].verify_env) == dict(base.verify_env)


@pytest.mark.asyncio
async def test_run_post_merge_verify_flag_on_no_sidecar_is_byte_identical(
    tmp_path: Path,
) -> None:
    """retry_failed_only=True but NO sidecar → tolerant loader None → spec unchanged.

    This is the real-world DEFAULT state until reify's α/β/γ land and write a
    well-formed attempt-0 sidecar: the flag is on, but the tolerant loader
    returns None, the ``if attempt0 is not None:`` guard short-circuits, and the
    retry producer must leave ``spec.verify_env`` byte-identical to a full
    verify.  Crucially, NEITHER ``_load_attempt0_sidecar`` NOR
    ``_assemble_retry_verify_env`` is patched here — the None short-circuit runs
    through the REAL loader end-to-end (the entire justification for the tolerant
    loader), which the flag-on/valid-sidecar test cannot exercise because it
    mocks the assembler.
    """
    from orchestrator import merge_queue as mq

    config = OrchestratorConfig(project_root=tmp_path, git=GitConfig(main_branch='main'))
    req = dataclasses.replace(
        _make_request('r-on-nosidecar', 'task/r-on-nosidecar', tmp_path, config),
        retry_failed_only=True,
    )
    git_ops = _make_git_ops_mock()
    # NOTE: deliberately do NOT write the attempt-0 sidecar.

    captured: dict[str, MergeVerifySpec] = {}

    async def _fake_dispatch(merge_sha, spec, **kw):  # type: ignore[no-untyped-def]
        captured['spec'] = spec
        return _mock_verify_result(True)

    with patch.object(mq, '_ensure_verify_disk_space', new=AsyncMock(return_value=None)), \
         patch.object(mq.VerifyRunnerPool, 'dispatch', new=AsyncMock(side_effect=_fake_dispatch)):
        outcome = await mq._run_post_merge_verify(
            git_ops, req, tmp_path,
            timeouts={}, enospc_retries={},
            max_timeouts=2, max_enospc=1,
            merge_sha='abc123', runner=None,
        )

    assert outcome is None
    # The tolerant loader short-circuited BEFORE the INV-3 tree-OID probe.
    git_ops.get_head_tree_hash.assert_not_awaited()
    base = build_merge_verify_spec(config, req.module_configs, None)
    assert dict(captured['spec'].verify_env) == dict(base.verify_env)


# ---------------------------------------------------------------------------
# shadow_baseline_sink out-param (PRD verify-retry-failed-only D4, §5.4): on the
# corroborated-narrowed path _run_post_merge_verify copies the attempt-0
# debug ∪ release verdict map into a caller-supplied sink dict, so
# _run_inflight_verify can union it with the PARTIAL narrowed warm output before
# storing the warm shadow baseline (else the full cold shadow compare flags every
# attempt-0-passed test as only_cold → phantom born-at-L2 divergence).  Populated
# ONLY when the narrow actually applied (assemble → non-None); left untouched on
# every non-narrowed path.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_post_merge_verify_populates_shadow_baseline_sink_when_narrowed(
    tmp_path: Path,
) -> None:
    """Narrowed retry → attempt-0 debug∪release verdicts land in shadow_baseline_sink."""
    from orchestrator import merge_queue as mq

    config = OrchestratorConfig(project_root=tmp_path, git=GitConfig(main_branch='main'))
    req = dataclasses.replace(
        _make_request('r-sink-on', 'task/r-sink-on', tmp_path, config),
        retry_failed_only=True,
    )
    git_ops = _make_git_ops_mock()
    # debug.verdicts={'c a::x':'pass'}, release.verdicts={} → sink == {'c a::x':'pass'}.
    _write_attempt0_sidecar(tmp_path)

    retry_env = {'REIFY_VERIFY_RETRY_SCOPE': 'failed_only'}
    sink: dict[str, str] = {}

    async def _fake_dispatch(merge_sha, spec, **kw):  # type: ignore[no-untyped-def]
        return _mock_verify_result(True)

    with patch.object(mq, '_ensure_verify_disk_space', new=AsyncMock(return_value=None)), \
         patch.object(mq.VerifyRunnerPool, 'dispatch', new=AsyncMock(side_effect=_fake_dispatch)), \
         patch.object(
             mq, '_assemble_retry_verify_env', new=AsyncMock(return_value=retry_env)
         ):
        outcome = await mq._run_post_merge_verify(
            git_ops, req, tmp_path,
            timeouts={}, enospc_retries={},
            max_timeouts=2, max_enospc=1,
            merge_sha='abc123', runner=None,
            shadow_baseline_sink=sink,
        )

    assert outcome is None
    assert sink == {'c a::x': 'pass'}


@pytest.mark.asyncio
async def test_run_post_merge_verify_leaves_sink_empty_when_not_narrowed(
    tmp_path: Path,
) -> None:
    """assemble → None (uncorroborated/rebased tree) → sink stays EMPTY.

    The sink is the narrowed-decision's mirror: a full re-verify (assemble
    returned None so ``narrowed`` stays False) must NOT seed a stale attempt-0
    baseline, which would ADD phantom tests → false only_warm divergence.
    """
    from orchestrator import merge_queue as mq

    config = OrchestratorConfig(project_root=tmp_path, git=GitConfig(main_branch='main'))
    req = dataclasses.replace(
        _make_request('r-sink-off', 'task/r-sink-off', tmp_path, config),
        retry_failed_only=True,
    )
    git_ops = _make_git_ops_mock()
    _write_attempt0_sidecar(tmp_path)  # sidecar present, but assemble → None below.

    sink: dict[str, str] = {}

    async def _fake_dispatch(merge_sha, spec, **kw):  # type: ignore[no-untyped-def]
        return _mock_verify_result(True)

    with patch.object(mq, '_ensure_verify_disk_space', new=AsyncMock(return_value=None)), \
         patch.object(mq.VerifyRunnerPool, 'dispatch', new=AsyncMock(side_effect=_fake_dispatch)), \
         patch.object(mq, '_assemble_retry_verify_env', new=AsyncMock(return_value=None)):
        outcome = await mq._run_post_merge_verify(
            git_ops, req, tmp_path,
            timeouts={}, enospc_retries={},
            max_timeouts=2, max_enospc=1,
            merge_sha='abc123', runner=None,
            shadow_baseline_sink=sink,
        )

    assert outcome is None
    assert sink == {}


# ---------------------------------------------------------------------------
# _load_reify_attempt_sidecar — the DRIFT TRIPWIRE (task 3059, WORK item 5).
#
# READ THIS BEFORE "FIXING" A FAILURE HERE.  These tests are pinned to the
# CHECKED-IN REAL BYTES of the sidecar reify actually writes
# (tests/fixtures/reify_verify_retry/reify-verify-attempt.json, captured from a
# live warm lane on 2026-07-30).  A failure in this class means the DF/reify
# seam has DRIFTED.  The correct response is to RE-CAPTURE the fixture from a
# live lane and fix this consumer.  Do NOT edit the fixture to make a test pass
# — the shipped D2 producer was authored from its own docstring rather than the
# producer's bytes, and that is exactly the failure this task exists to undo.
#
# The prior D2 loader read a DF-invented `.reify-verify-retry/attempt0.json`
# that nothing in reify or DF has ever written.
# ---------------------------------------------------------------------------

_REIFY_FIXTURE_DIR = Path(__file__).parent / 'fixtures' / 'reify_verify_retry'
_REIFY_SIDECAR_FIXTURE = _REIFY_FIXTURE_DIR / 'reify-verify-attempt.json'


def _place_real_sidecar(merge_wt: Path, text: str | None = None) -> Path:
    """Copy the fixture bytes VERBATIM to the path reify writes under merge_wt."""
    from orchestrator.merge_queue import _REIFY_ATTEMPT_SIDECAR_RELPATH

    path = merge_wt / _REIFY_ATTEMPT_SIDECAR_RELPATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _REIFY_SIDECAR_FIXTURE.read_text() if text is None else text
    )
    return path


def test_reify_attempt_sidecar_relpath_is_reifys_default() -> None:
    """The sidecar path constant equals reify's REIFY_VERIFY_ATTEMPT_SIDECAR default.

    A bare string equality on purpose: reify's verify.sh:738 defines
    ``_ATTEMPT_SIDECAR_PATH="${REIFY_VERIFY_ATTEMPT_SIDECAR:-target/reify-verify-attempt.json}"``,
    so if that default moves this fails LOUDLY at the constant rather than
    silently degrading every retry to a full verify.
    """
    from orchestrator.merge_queue import _REIFY_ATTEMPT_SIDECAR_RELPATH

    assert _REIFY_ATTEMPT_SIDECAR_RELPATH == 'target/reify-verify-attempt.json'


def test_reify_attempt_sidecar_fixture_schema_is_exactly_three_keys() -> None:
    """The real sidecar carries EXACTLY {tree_oid, profiles, timestamp}.

    Drift in EITHER direction fails: a missing key breaks the loader's
    assumptions, an extra key means reify started publishing something DF may
    need to consume.
    """
    assert json.loads(_REIFY_SIDECAR_FIXTURE.read_text()).keys() == {
        'tree_oid',
        'profiles',
        'timestamp',
    }


def test_load_reify_attempt_sidecar_parses_real_bytes(tmp_path: Path) -> None:
    """The loader parses reify's verbatim bytes; `profiles` is a SPACE-DELIMITED STRING.

    Not a JSON list — a reader who assumed a list would get a per-character
    iteration and silently build a nonsense profile set.
    """
    from orchestrator.merge_queue import _load_reify_attempt_sidecar

    _place_real_sidecar(tmp_path)
    sidecar = _load_reify_attempt_sidecar(tmp_path)

    assert sidecar is not None
    expected_oid = json.loads(_REIFY_SIDECAR_FIXTURE.read_text())['tree_oid']
    assert sidecar.tree_oid == expected_oid
    assert sidecar.profiles == ('debug', 'release')


def test_load_reify_attempt_sidecar_absent_file_returns_none(tmp_path: Path) -> None:
    """No sidecar (reify never stamped one) -> None -> full verify."""
    from orchestrator.merge_queue import _load_reify_attempt_sidecar

    assert _load_reify_attempt_sidecar(tmp_path) is None


@pytest.mark.parametrize(
    ('label', 'text'),
    [
        ('non-JSON bytes', 'not json at all'),
        ('a JSON array', '["tree_oid", "profiles"]'),
        ('missing tree_oid', '{"profiles": "debug", "timestamp": "t"}'),
        ('empty profiles', '{"tree_oid": "abc", "profiles": "", "timestamp": "t"}'),
        (
            'whitespace-only profiles',
            '{"tree_oid": "abc", "profiles": "   ", "timestamp": "t"}',
        ),
        (
            'unknown profile name',
            '{"tree_oid": "abc", "profiles": "debug bench", "timestamp": "t"}',
        ),
    ],
)
def test_load_reify_attempt_sidecar_malformed_returns_none(
    tmp_path: Path, label: str, text: str
) -> None:
    """Every malformed/unusable shape returns None WITHOUT raising -> full verify.

    The unknown-profile case is not paranoia: DF has no
    REIFY_VERIFY_RETRY_NEXTEST_FILTER_FILE_<X> env key for a third profile, so
    it cannot satisfy reify's "set a filter file for EVERY profile named in
    `profiles`, or fall back to a full verify" obligation (verify.sh:219-230).
    Silently ignoring the unknown profile would narrow a profile that never ran.
    """
    from orchestrator.merge_queue import _load_reify_attempt_sidecar

    _place_real_sidecar(tmp_path, text=text)
    assert _load_reify_attempt_sidecar(tmp_path) is None, label


# ---------------------------------------------------------------------------
# _load_attempt0_sidecar tolerant degradation (the robustness core that keeps
# the retry path a strict no-op until reify writes a well-formed sidecar).
# Every branch must return None WITHOUT raising — a regression that let the
# loader raise (e.g. an over-narrow ``except``) would crash the merge-verify
# path.  These exercise each branch directly (the wiring tests above only cover
# the happy path indirectly).
# ---------------------------------------------------------------------------


def _write_raw_sidecar(merge_wt: Path, text: str) -> Path:
    """Write raw (possibly malformed) bytes to the attempt-0 sidecar path."""
    d = merge_wt / _SIDECAR_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    path = d / _SIDECAR_NAME
    path.write_text(text)
    return path


def _mq_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == 'orchestrator.merge_queue' and r.levelno >= logging.WARNING
    ]


def test_load_attempt0_sidecar_missing_file_returns_none(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """(a) Absent sidecar → None, and NO warning (the common pre-reify case)."""
    from orchestrator.merge_queue import _load_attempt0_sidecar

    with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
        assert _load_attempt0_sidecar(tmp_path) is None
    assert _mq_warnings(caplog) == []  # absent is silent, not a warning


def test_load_attempt0_sidecar_invalid_json_returns_none(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """(b) Present but not valid JSON → None + WARNING; never raises."""
    from orchestrator.merge_queue import _load_attempt0_sidecar

    _write_raw_sidecar(tmp_path, '{not: valid json,,,')
    with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
        assert _load_attempt0_sidecar(tmp_path) is None
    warnings = _mq_warnings(caplog)
    assert any('not valid JSON' in m for m in warnings), warnings
    assert any('full verify' in m for m in warnings), warnings


def test_load_attempt0_sidecar_missing_tree_oid_returns_none(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """(c) Valid JSON object but no 'tree_oid' → None + WARNING; never raises."""
    from orchestrator.merge_queue import _load_attempt0_sidecar

    _write_raw_sidecar(
        tmp_path, json.dumps({'debug': {'planned': [], 'verdicts': {}}})
    )
    with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
        assert _load_attempt0_sidecar(tmp_path) is None
    warnings = _mq_warnings(caplog)
    assert any('tree_oid' in m for m in warnings), warnings


def test_load_attempt0_sidecar_not_an_object_returns_none(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """(c-variant) Valid JSON that is not an object (a list) → None + WARNING."""
    from orchestrator.merge_queue import _load_attempt0_sidecar

    _write_raw_sidecar(tmp_path, json.dumps([1, 2, 3]))
    with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
        assert _load_attempt0_sidecar(tmp_path) is None
    warnings = _mq_warnings(caplog)
    assert any('not an object' in m for m in warnings), warnings


def test_load_attempt0_sidecar_malformed_subfield_returns_none(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """(d) tree_oid present but a sub-field is junk → None + WARNING; never raises.

    ``debug`` set to a non-dict scalar makes ``debug.get('planned')`` raise
    AttributeError inside the field-mapping block; the malformed-fields guard
    must catch it and degrade to full verify rather than propagate.
    """
    from orchestrator.merge_queue import _load_attempt0_sidecar

    _write_raw_sidecar(
        tmp_path, json.dumps({'tree_oid': 'abc123', 'debug': 12345})
    )
    with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
        assert _load_attempt0_sidecar(tmp_path) is None
    warnings = _mq_warnings(caplog)
    assert any('malformed' in m for m in warnings), warnings


def test_load_attempt0_sidecar_valid_returns_payload(tmp_path: Path) -> None:
    """Happy path: a well-formed sidecar parses into the _Attempt0Payload fields.

    Directly validates the JSON→dataclass field mapping (planned/verdicts per
    profile, run_all members, gui specs) that the wiring tests only exercise
    transitively through a mocked assembler.
    """
    from orchestrator.merge_queue import _Attempt0Payload, _load_attempt0_sidecar

    _write_attempt0_sidecar(tmp_path)  # tree_oid='deadbeef', minimal valid shape
    payload = _load_attempt0_sidecar(tmp_path)

    assert isinstance(payload, _Attempt0Payload)
    assert payload.tree_oid == 'deadbeef'
    assert payload.debug_planned == ['c a::x', 'c a::y']
    assert payload.debug_verdicts == {'c a::x': 'pass'}
    assert payload.release_planned == []
    assert payload.release_verdicts == {}
    assert payload.run_all_members == []
    assert payload.gui_specs == []
