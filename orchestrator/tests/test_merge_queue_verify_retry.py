"""Tests for the failed-only merge-verify retry producer (PRD verify-retry-failed-only D2).

The DF orchestrator PRODUCES the reify retry contract: per-profile nextest
filter files ({did-not-pass} exact ids) written into the shared merge_wt, plus
the brand-new REIFY_VERIFY_RETRY_* / REIFY_RUN_ALL_MEMBER_SUBSET /
REIFY_GUI_RETRY_SPECS env keys threaded through MergeVerifySpec.verify_env.
reify's verify.sh (α/β/γ) is the CONSUMER — out of scope here.

Covers:
  * ``_build_retry_verify_env`` — writes the debug/release nextest filter files
    and returns the REIFY_* env dict.
  * ``_assemble_retry_verify_env`` — INV-3 tree-OID corroboration gate.
  * ``_load_reify_attempt_sidecar`` — the drift tripwire pinned to the REAL
    bytes reify writes (task 3059).
  * ``_run_post_merge_verify`` wiring under ``req.retry_failed_only``.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

if TYPE_CHECKING:
    from orchestrator.merge_queue import MergeRequest


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
# _run_post_merge_verify wiring.  Guarded by req.retry_failed_only; the retry env
# is merged into MergeVerifySpec.verify_env inside the classified-infra-transient
# RETRY branch (task 3059) — never before the first dispatch, which would narrow
# attempt-0 itself.  Driven on the LOCAL path (runner=None) so task 2822's
# remote-green cross-check block (runs ONLY when runner is not None) stays inert
# and the wiring is isolated.
# ---------------------------------------------------------------------------


def _make_git_ops_mock() -> MagicMock:
    m = MagicMock()
    m.get_main_sha = AsyncMock(return_value='main-sha')
    m.get_free_disk_bytes = AsyncMock(return_value=100 * 1024 ** 3)
    m.cleanup_merge_worktree = AsyncMock()
    m.create_throwaway_verify_worktree = AsyncMock(return_value='/repo/_throwaway')
    m.get_head_tree_hash = AsyncMock(return_value='deadbeef')
    return m


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
