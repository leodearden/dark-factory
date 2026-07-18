"""Tests for scripts/gc_agent_transcripts.py — the retention GC sweep over the
gzipped agent-transcript archive (task 2731, δ of
plans/agent-transcript-archival-prd.md).

step-1: the pure age-cap arm of select_prunable(task_dirs, now, max_age_days,
max_task_dirs) exercised with synthetic (Path, mtime) tuples — NO filesystem,
fixed NOW. A dir older than now - max_age_days*86400 prunes with reason 'age';
a fresh dir is kept; the exact boundary (now - mtime == max_age_days*86400) is
KEPT (strict >, not >=); max_age_days <= 0 disables the age axis.
"""
from __future__ import annotations

from pathlib import Path

from gc_agent_transcripts import select_prunable

DAY = 86_400
NOW = 1_000_000_000.0

# A count cap far larger than any dir count used in the age tests, so the
# count arm (once implemented in step-4) never perturbs a pure-age assertion.
HIGH_COUNT_CAP = 10_000


def _dir(name: str) -> Path:
    return Path("/archive") / name


# ---------------------------------------------------------------------------
# step-1: pure age-cap arm
# ---------------------------------------------------------------------------

def test_age_arm_prunes_dir_older_than_cutoff():
    cutoff = NOW - 90 * DAY
    old = _dir("old")
    fresh = _dir("fresh")
    task_dirs = [(old, cutoff - DAY), (fresh, NOW)]

    decision = select_prunable(task_dirs, NOW, max_age_days=90, max_task_dirs=HIGH_COUNT_CAP)

    assert decision.prune_paths == {old}
    assert decision.keep_paths == {fresh}
    assert decision.reasons[old] == "age"


def test_age_arm_keeps_fresh_dir():
    fresh = _dir("fresh")
    decision = select_prunable([(fresh, NOW)], NOW, max_age_days=90, max_task_dirs=HIGH_COUNT_CAP)

    assert decision.keep_paths == {fresh}
    assert decision.prune_paths == set()


def test_age_boundary_exact_is_kept_one_second_older_is_pruned():
    """now - mtime == max_age_days*86400 exactly is KEPT (strict >, not >=);
    one second older is pruned."""
    cutoff = NOW - 90 * DAY
    boundary = _dir("boundary")  # mtime == cutoff -> now - mtime == 90*DAY exactly
    just_old = _dir("just_old")  # one second older than the boundary
    task_dirs = [(boundary, cutoff), (just_old, cutoff - 1)]

    decision = select_prunable(task_dirs, NOW, max_age_days=90, max_task_dirs=HIGH_COUNT_CAP)

    assert decision.keep_paths == {boundary}
    assert decision.prune_paths == {just_old}
    assert decision.reasons[just_old] == "age"


def test_non_positive_max_age_disables_age_pruning():
    """max_age_days <= 0 imposes no age bound: even an ancient dir is kept
    (count also disabled here, so nothing prunes at all)."""
    ancient = _dir("ancient")
    task_dirs = [(ancient, NOW - 10_000 * DAY)]

    for disabled in (0, -1):
        decision = select_prunable(
            task_dirs, NOW, max_age_days=disabled, max_task_dirs=0,
        )
        assert decision.keep_paths == {ancient}
        assert decision.prune_paths == set()
