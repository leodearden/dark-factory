"""Unit tests for the C2 worktree-namespace classifier.

PRD: docs/prds/merge-worktree-lifecycle-integrity.md, task beta (§4 Contract
C2 — namespace invariant).

``classify_worktree_entry(name) -> Literal['task', 'merge', 'infra']`` is the
pure, positive-match complement of :data:`orchestrator.git_ops.PROTECTED_PREFIXES`
/ reify's PROTECT_GLOB: instead of a hand-maintained NEGATIVE per-name
exclusion list (the whack-a-mole the 2026-07-22 task/5326 incident proved
unmaintainable, when the crash-recovery sweep force-removed the persistent
``_merge-verify`` 21s after dispatching a verify into it), it encodes ONE
invariant keyed on the directory-name prefix:

    * ``_merge-``-prefixed        => 'merge'  (reported to the merge reaper)
    * any other ``_``/``.``-prefix => 'infra'  (left to its owner)
    * everything else              => 'task'   (task-id-shaped namespace)

These tests pin that invariant DIRECTLY, independent of any sweep/reaper
wiring — they assert the RULE, not a per-name enumeration.
"""
from __future__ import annotations

import pytest

from orchestrator.git_ops import classify_worktree_entry

# ── The invariant, expressed as data ────────────────────────────────────────
# `_merge-`-prefixed names are the merge-queue band the sweep REPORTS (never
# removes directly): the merge reaper owns their guarded disposition.
_MERGE_NAMES = [
    '_merge-verify',        # the persistent merge-verify worktree (the 5326 victim)
    '_merge-ba97f10a',      # an ephemeral _merge-<uuid> throwaway verify tree
    '_merge-0',
]

# Every OTHER `_`- or `.`-prefixed entry is infra-owned — protected by the
# same namespace rule with NO per-name enumeration.  This list is
# illustrative of the current bands (it mirrors PROTECTED_PREFIXES /
# WorktreeKind / the persistent-name constants), but the classifier must not
# depend on it: see test_invariant_* below, which assert the prefix RULE for
# arbitrary suffixes so a future band needs no new case here.
_INFRA_NAMES = [
    '_mainprobe-x',          # WorktreeKind.MAIN_PROBE ephemeral verify probe
    '_mainsweep-x',          # WorktreeKind.MAIN_SWEEP ephemeral verify sweep
    '_iact-slug',            # interactive worktree band (config.iact_prefix)
    '_offline-deep',         # PERSISTENT_OFFLINE_DEEP_WORKTREE_NAME
    '_solo-x',               # attribution-solo band
    '_substrate-gate-x',     # harness-substrate-gate band
    '.reseed-trash',         # reseed trash holding dir
    '.merge_verify_pgids',   # merge-verify holder-pgid rendezvous file/dir
    '.lane-state',           # LANE_STATE_DIRNAME durable lane records
    '.task-meta',            # TASK_META_DIRNAME durable plan/iteration artifacts
]

# Task-id-shaped names: NOT `_`/`.`-prefixed.  These are the only entries the
# crash-recovery sweep + orphan reaper may act on.
_TASK_NAMES = [
    '999',
    '2925',
    '2925.1',                # a dotted subtask id — still task-shaped (no leading dot)
]


@pytest.mark.parametrize('name', _MERGE_NAMES)
def test_merge_prefixed_names_classify_merge(name: str) -> None:
    assert classify_worktree_entry(name) == 'merge'


@pytest.mark.parametrize('name', _INFRA_NAMES)
def test_infra_prefixed_names_classify_infra(name: str) -> None:
    assert classify_worktree_entry(name) == 'infra'


@pytest.mark.parametrize('name', _TASK_NAMES)
def test_task_shaped_names_classify_task(name: str) -> None:
    assert classify_worktree_entry(name) == 'task'


# ── The RULE itself (not a per-name list) ───────────────────────────────────
# These assert the classifier is an INVARIANT over the prefix, so any current
# OR future infra band is covered with no enumeration (D7: invariant over
# per-name exclusion lists).


@pytest.mark.parametrize('suffix', ['verify', 'ba97f10a', '0', 'anything-at-all'])
def test_invariant_any_merge_prefix_is_merge(suffix: str) -> None:
    """`_merge-<anything>` => 'merge', regardless of the suffix."""
    assert classify_worktree_entry(f'_merge-{suffix}') == 'merge'


@pytest.mark.parametrize('suffix', ['brandnew', 'future-band-x', 'y', ''])
def test_invariant_any_other_underscore_prefix_is_infra(suffix: str) -> None:
    """Any `_`-prefixed non-`_merge-` name => 'infra', with no per-name case."""
    name = f'_{suffix}'
    assert not name.startswith('_merge-')
    assert classify_worktree_entry(name) == 'infra'


@pytest.mark.parametrize('suffix', ['reseed-trash', 'brandnew', 'future', 'x'])
def test_invariant_any_dot_prefix_is_infra(suffix: str) -> None:
    """Any `.`-prefixed name => 'infra', with no per-name case."""
    assert classify_worktree_entry(f'.{suffix}') == 'infra'


@pytest.mark.parametrize('name', ['999', '2925', '2925.1', 'abc', 'a-task-name'])
def test_invariant_non_prefixed_is_task(name: str) -> None:
    """A name NOT starting with `_` or `.` => 'task' (the task namespace)."""
    assert not name.startswith(('_', '.'))
    assert classify_worktree_entry(name) == 'task'
