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

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


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
    req = SimpleNamespace(task_id='t-1', retry_failed_only=True)
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
    req = SimpleNamespace(task_id='t-2', retry_failed_only=True)
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
    req = SimpleNamespace(task_id='t-3', retry_failed_only=True)
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
