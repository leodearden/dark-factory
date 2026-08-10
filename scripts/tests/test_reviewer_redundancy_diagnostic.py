"""Tests for reviewer_redundancy_diagnostic.py — corrupt file handling."""
from __future__ import annotations

import logging

import reviewer_redundancy_diagnostic as mod
from orchestrator.evals.snapshots import eval_worktree_root

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_review_tree(tmp_path, *, corrupt_names=(), valid_names=()):
    """Create <tmp_path>/.worktrees/demo/.task/reviews/ with some files.

    corrupt_names: reviewer names written with non-JSON bytes
    valid_names:   reviewer names written with a minimal valid review JSON
    Names NOT listed in either argument are left absent (file never created).
    """
    reviews_dir = tmp_path / '.worktrees' / 'demo' / '.task' / 'reviews'
    reviews_dir.mkdir(parents=True)
    for name in corrupt_names:
        (reviews_dir / f'reviewer_{name}.json').write_bytes(b'{not valid')
    for name in valid_names:
        (reviews_dir / f'reviewer_{name}.json').write_text(
            '{"verdict": "PASS", "issues": []}'
        )
    return reviews_dir


# ---------------------------------------------------------------------------
# step-1: RED — corrupt review emits a WARNING naming the file
# ---------------------------------------------------------------------------

def test_corrupt_review_emits_warning(tmp_path, monkeypatch, caplog):
    """A present-but-corrupt reviewer JSON emits exactly one WARNING.

    The warning must name the corrupt file; the absent reviewer file must
    stay silent (no warning referencing its path).
    """
    _build_review_tree(
        tmp_path,
        corrupt_names=['robustness'],   # present + unreadable
        valid_names=['test_analyst'],   # present + valid
        # 'performance' deliberately not created → absent case
    )

    monkeypatch.setattr(mod, 'REPO', tmp_path)
    monkeypatch.setattr(mod, 'SEARCH_ROOTS', [tmp_path / '.worktrees'])

    with caplog.at_level(logging.WARNING):
        mod.main()

    warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warning_records) == 1, (
        f'Expected exactly 1 WARNING, got {len(warning_records)}: '
        f'{[r.getMessage() for r in warning_records]}'
    )
    assert 'reviewer_robustness.json' in warning_records[0].getMessage(), (
        f'Warning must name the corrupt file; got: {warning_records[0].getMessage()!r}'
    )
    # Absent file must NOT appear in any log record
    assert not any(
        'reviewer_performance.json' in r.getMessage() for r in caplog.records
    ), 'Absent file must stay silent — no warning expected for it'


# ---------------------------------------------------------------------------
# step-3: RED — corrupt review counted and printed in report
# ---------------------------------------------------------------------------

def test_corrupt_review_counted_and_reported(tmp_path, monkeypatch, capsys):
    """The final report prints 'skipped 1 unreadable review files'.

    Count is 1 (corrupt), not 2 — proves absent reviewer is NOT counted.
    Also checks that main() returns 0 and the normal summary still prints.
    """
    _build_review_tree(
        tmp_path,
        corrupt_names=['robustness'],   # present + unreadable → counted
        valid_names=['test_analyst'],   # present + valid
        # 'performance' deliberately not created → absent, must NOT be counted
    )

    monkeypatch.setattr(mod, 'REPO', tmp_path)
    monkeypatch.setattr(mod, 'SEARCH_ROOTS', [tmp_path / '.worktrees'])

    ret = mod.main()
    captured = capsys.readouterr()

    assert ret == 0, f'main() returned {ret!r}, expected 0'
    assert 'skipped 1 unreadable review files\n' in captured.out, (
        f'Expected "skipped 1 unreadable review files" in stdout; got:\n{captured.out}'
    )
    # Happy path still works: normal summary present
    assert 'PER-REVIEWER SUMMARY' in captured.out


# ---------------------------------------------------------------------------
# RED — SEARCH_ROOTS point at the relocated (sibling) eval-worktree root
# ---------------------------------------------------------------------------

def test_search_roots_use_relocated_eval_root():
    """SEARCH_ROOTS must scan eval_worktree_root(REPO), not REPO/.eval-worktrees.

    Post-2881 eval worktrees live at eval_worktree_root(REPO) — a SIBLING of
    REPO — not the in-repo REPO/.eval-worktrees. If SEARCH_ROOTS still points
    at the old in-repo path, the diagnostic silently discovers zero relocated
    worktrees. Asserts against the real module-level SEARCH_ROOTS computed
    from the hardcoded REPO.
    """
    assert eval_worktree_root(mod.REPO) in mod.SEARCH_ROOTS
    assert (mod.REPO / '.eval-worktrees') not in mod.SEARCH_ROOTS


# ---------------------------------------------------------------------------
# RED — task_label handles a relocated (out-of-REPO) eval reviews dir
# ---------------------------------------------------------------------------

def test_task_label_handles_relocated_eval_worktree(tmp_path, monkeypatch):
    """task_label must not crash on a relocated (out-of-REPO) eval reviews dir.

    Post-2881 eval worktrees live at eval_worktree_root(REPO) — a SIBLING of
    REPO — so `reviews_dir.relative_to(REPO)` raises ValueError. task_label
    must fall back to REPO.parent (the common ancestor of .worktrees and the
    relocated eval sibling) and still produce a label naming the task and run
    dirs. This is exactly the worktree the SEARCH_ROOTS fix newly discovers.
    """
    monkeypatch.setattr(mod, 'REPO', tmp_path)
    reviews = (
        tmp_path.parent / f'{tmp_path.name}-eval-worktrees'
        / 'df_task_12' / 'run-abc' / '.task' / 'reviews'
    )

    label = mod.task_label(reviews)  # must NOT raise ValueError

    assert 'df_task_12' in label
    assert 'run-abc' in label
