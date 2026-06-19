"""Tests for reviewer_redundancy_diagnostic.py — corrupt file handling."""
from __future__ import annotations

import logging

import reviewer_redundancy_diagnostic as mod


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
    assert 'skipped 1 unreadable review file\n' in captured.out, (
        f'Expected "skipped 1 unreadable review file" (singular) in stdout; got:\n{captured.out}'
    )
    # Happy path still works: normal summary present
    assert 'PER-REVIEWER SUMMARY' in captured.out
