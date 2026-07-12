"""Tests collapsing redundant phase encodings onto the ItemLifecycle registry
(merge-queue-reliability PRD scope-4 lambda / task 2173, depends on kappa=2169).

Task kappa (task 2169) wired ``ItemLifecycle`` — the request_id-keyed
lifecycle-state registry — at every put/pop call site in
``SpeculativeMergeWorker``. This task (lambda) deletes the two redundant
phase encodings that predate it:

  step-1  RED   — vestigial single-host ``_verify_item``/``_verify_phase``/
                  ``_verify_started_at`` worker fields are gone (API-surface
                  contract; the fields are write-only, zero reads anywhere)
  step-2  GREEN — the fields and their six write sites deleted (pure
                  dead-code removal)
  step-3  RED   — snapshot()/frozen_prefix() derive phase from the
                  ItemLifecycle registry, not a stored ``InflightEntry.phase``
  step-4  GREEN — ``SpeculativeMergeWorker._entry_phase()`` added; all 5
                  readers repointed onto the registry (``InflightEntry.phase``
                  kept as a fallback source — zero existing-test churn)
  step-5  RED   — ``InflightEntry`` no longer carries a free-form phase field
  step-6  GREEN — ``InflightEntry.phase`` field deleted; existing tests
                  reconciled to drive registry state instead

This module imports orchestrator.merge_queue LOCALLY inside each test method
(not at module scope) so a not-yet-implemented symbol never breaks collection
of the rest of the file during the RED steps — mirrors
test_merge_queue_lifecycle_registry.py's / test_merge_queue_request_liveness.py's
convention.
"""

from __future__ import annotations

import asyncio
import dataclasses
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run

# ---------------------------------------------------------------------------
# Fixtures + helpers (per-file duplication convention — see
# test_merge_queue_lifecycle_registry.py; there is no shared worker conftest).
# ---------------------------------------------------------------------------


async def _setup_repo(repo: Path) -> None:
    """Initialise a bare git repository with a single commit on main."""
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create a temporary git repository with an initial commit."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_setup_repo(repo))
    return repo


@pytest.fixture
def git_config() -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
    )


@pytest.fixture
def git_ops(git_config: GitConfig, git_repo: Path) -> GitOps:
    return GitOps(git_config, git_repo)


@pytest.fixture
def config(git_repo: Path, git_config: GitConfig) -> OrchestratorConfig:
    return OrchestratorConfig(project_root=git_repo, git=git_config)


def _make_worker(git_ops: GitOps):
    """Build a bare SpeculativeMergeWorker for unit tests (no harness wiring).

    Mirrors test_merge_queue_lifecycle_registry.py's / test_merge_queue_
    invariant_integration_gate.py:212's ``_make_worker``.
    """
    from orchestrator.merge_queue import SpeculativeMergeWorker

    return SpeculativeMergeWorker(git_ops, asyncio.Queue())


def _make_request(
    task_id: str,
    branch: str,
    worktree: Path,
    config: OrchestratorConfig,
):
    """Build a MergeRequest with a fresh Future for the running event loop.

    Duplicated from test_merge_queue_lifecycle_registry.py / test_merge_queue_
    request_liveness.py (per-file duplication convention — see this file's
    module docstring).
    """
    from orchestrator.merge_types import MergeRequest

    return MergeRequest(
        task_id=task_id,
        branch=branch,
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=asyncio.get_running_loop().create_future(),
        lane='normal',
    )


# ---------------------------------------------------------------------------
# step-1 RED / step-2 GREEN: vestigial single-host observability fields
# ---------------------------------------------------------------------------


class TestVestigialVerifyFieldsRemoved:
    """The single-host ``_verify_item``/``_verify_phase``/``_verify_started_at``
    fields are write-only after epsilon (merge_queue.py:5085-5088) with ZERO
    reads anywhere in the tree — deleting them is pure dead-code removal that
    structurally eliminates the stale-phantom-snapshot-entry class described
    at merge_queue.py:6900-6903 (task lambda / task 2173 step-1).

    RED until step-2 GREEN deletes the field declarations (currently
    merge_queue.py:5089-5091) and their six write sites.
    """

    def test_vestigial_fields_absent(self, git_ops: GitOps) -> None:
        worker = _make_worker(git_ops)

        assert not hasattr(worker, '_verify_item'), (
            'worker._verify_item must be deleted — it is write-only '
            '(zero reads anywhere in the tree).'
        )
        assert not hasattr(worker, '_verify_phase'), (
            'worker._verify_phase must be deleted — it is write-only '
            '(zero reads anywhere in the tree).'
        )
        assert not hasattr(worker, '_verify_started_at'), (
            'worker._verify_started_at must be deleted — it is write-only '
            '(zero reads anywhere in the tree).'
        )


# ---------------------------------------------------------------------------
# step-3 RED / step-4 GREEN: snapshot()/frozen_prefix() derive phase from the
# ItemLifecycle registry, not the stored InflightEntry.phase field
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestPhaseDerivesFromRegistry:
    """snapshot()/frozen_prefix() derive an entry's phase from the
    ItemLifecycle registry (``self._lifecycle.current(request_id)``), not
    from the stored ``InflightEntry.phase`` field (task lambda / task 2173
    step-3).

    RED until step-4 GREEN adds ``SpeculativeMergeWorker._entry_phase()`` and
    repoints the production readers onto it: on current code, a
    registry-only transition (via ``_note_transition``, with ``entry.phase``
    left untouched) does not move snapshot()/frozen_prefix()'s reported
    state, because they read the stale ``entry.phase`` field directly.
    """

    async def test_registry_transition_moves_reported_phase(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ) -> None:
        from orchestrator.merge_queue import InflightEntry, ItemLifecycleState, RealMergeItem

        worker = _make_worker(git_ops)

        req = _make_request('lam-derive', 'lam-derive', tmp_path, config)
        item = RealMergeItem(
            request=req, merge_result=MagicMock(), merge_wt=tmp_path / 'merge_wt',
            base_sha='deadbeef', speculative=False,
        )
        rid = worker._register_item(item, initial=ItemLifecycleState.VERIFYING)
        entry = InflightEntry(
            item=item, lease=None, verify_task=MagicMock(), merge_wt=item.merge_wt,
            was_speculative=False,
        )
        worker._inflight.append(entry)

        snap = worker.snapshot()
        assert snap['entries'][0]['state'] == 'verifying'
        assert snap['verify_in_progress']['phase'] == 'verifying'
        assert rid in worker.frozen_prefix()

        # Advance the registry — InflightEntry has no phase field to touch.
        worker._note_transition(
            rid, ItemLifecycleState.VERIFYING, ItemLifecycleState.FINALIZING,
        )
        worker._note_transition(
            rid, ItemLifecycleState.FINALIZING, ItemLifecycleState.GATE_REVERIFY,
        )

        snap2 = worker.snapshot()
        assert snap2['entries'][0]['state'] == 'gate_reverify', (
            "entries[0]['state'] should follow the registry (GATE_REVERIFY); "
            f"got {snap2['entries'][0]['state']!r}."
        )
        assert snap2['verify_in_progress']['phase'] == 'gate_reverify', (
            "RED: verify_in_progress['phase'] should follow the registry "
            f"(GATE_REVERIFY); got {snap2['verify_in_progress']['phase']!r}."
        )
        assert rid in worker.frozen_prefix(), (
            'the entry must still qualify for frozen_prefix after the '
            'registry-only transition to GATE_REVERIFY.'
        )

    async def test_finalizing_head_excluded_when_registry_is_non_qualifying(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ) -> None:
        from orchestrator.merge_queue import InflightEntry, ItemLifecycleState, RealMergeItem

        worker = _make_worker(git_ops)

        req2 = _make_request('lam-fh', 'lam-fh', tmp_path, config)
        item2 = RealMergeItem(
            request=req2, merge_result=MagicMock(), merge_wt=tmp_path / 'merge_wt2',
            base_sha='deadbeef', speculative=False,
        )
        # InflightEntry has no phase field at all — exclusion must be driven
        # purely by the registry state (DISPATCHING is non-qualifying).
        entry2 = InflightEntry(
            item=item2, lease=None, verify_task=None, merge_wt=item2.merge_wt,
            was_speculative=False,
        )
        # Register the InflightEntry itself (task 2435 kappa-b: the finalize
        # head is discovered via _live_items/_finalizing_head_entry(), which
        # requires the popped InflightEntry — not the raw item — to be the
        # live object; the deleted _finalizing_head field previously made
        # this entry discoverable directly).
        rid2 = worker._register_item(entry2, initial=ItemLifecycleState.DISPATCHING)

        assert rid2 not in worker.frozen_prefix(), (
            "a finalizing-head InflightEntry registered at DISPATCHING in "
            "the registry must be excluded from frozen_prefix."
        )


class TestInflightEntryHasNoPhaseField:
    """``InflightEntry`` no longer carries a free-form ``phase`` field — every
    reader derives the phase from the ItemLifecycle registry via
    ``SpeculativeMergeWorker._entry_phase()`` (task lambda / task 2173
    step-5).

    RED until step-6 GREEN deletes ``InflightEntry.phase`` (currently a
    required field at merge_types.py:1143): on current code the field is
    still present in ``dataclasses.fields(InflightEntry)``, and constructing
    an entry without a ``phase=`` argument raises ``TypeError`` (missing
    required positional argument).
    """

    def test_phase_field_absent_from_dataclass(self, git_ops: GitOps) -> None:
        from orchestrator.merge_queue import InflightEntry

        field_names = {f.name for f in dataclasses.fields(InflightEntry)}
        assert 'phase' not in field_names, (
            'InflightEntry must no longer declare a phase field (derivation '
            f'via _entry_phase() replaces it); got fields: {field_names!r}'
        )

    @pytest.mark.asyncio
    async def test_derivation_works_with_no_phase_field(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ) -> None:
        from orchestrator.merge_queue import InflightEntry, ItemLifecycleState, RealMergeItem

        worker = _make_worker(git_ops)

        req = _make_request('lam-nofield', 'lam-nofield', tmp_path, config)
        item = RealMergeItem(
            request=req, merge_result=MagicMock(), merge_wt=tmp_path / 'merge_wt',
            base_sha='deadbeef', speculative=False,
        )
        rid = worker._register_item(item, initial=ItemLifecycleState.VERIFYING)
        # No `phase=` kwarg at all — proves derivation needs no stored field.
        entry = InflightEntry(
            item=item, lease=None, verify_task=MagicMock(), merge_wt=item.merge_wt,
            was_speculative=False,
        )
        worker._inflight.append(entry)

        assert worker._entry_phase(entry) == 'verifying', (
            f'expected _entry_phase to derive verifying from the registry '
            f'for {rid!r} with no phase field present on the entry.'
        )

        snap = worker.snapshot()
        assert snap['entries'][0]['state'] == 'verifying'
        assert snap['verify_in_progress']['phase'] == 'verifying'
