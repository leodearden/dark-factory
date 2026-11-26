"""Tests for reviewer_redundancy_diagnostic.py — corrupt file handling."""
from __future__ import annotations

import logging

import reviewer_redundancy_diagnostic as mod


def _build_review_tree(tmp_path, *, corrupt_names=(), valid_names=()):
    """Create <tmp_path>/.worktrees/demo/.task/reviews/ with some files."""
    reviews_dir = tmp_path / '.worktrees' / 'demo' / '.task' / 'reviews'
    reviews_dir.mkdir(parents=True)
    for name in corrupt_names:
        (reviews_dir / f'reviewer_{name}.json').write_bytes(b'{not valid')
    for name in valid_names:
        (reviews_dir / f'reviewer_{name}.json').write_text(
            '{"verdict": "PASS", "issues": []}'
        )
    return reviews_dir


def test_corrupt_review_emits_warning(tmp_path, monkeypatch, caplog):
    """A present-but-corrupt reviewer JSON emits exactly one WARNING."""
    _build_review_tree(
        tmp_path,
        corrupt_names=['robustness'],
        valid_names=['test_analyst'],
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
    assert 'reviewer_robustness.json' in warning_records[0].getMessage()
    assert not any(
        'reviewer_performance.json' in r.getMessage() for r in caplog.records
    )


def test_corrupt_review_counted_and_reported(tmp_path, monkeypatch, capsys):
    """The final report prints 'skipped 1 unreadable review files'."""
    _build_review_tree(
        tmp_path,
        corrupt_names=['robustness'],
        valid_names=['test_analyst'],
    )

    monkeypatch.setattr(mod, 'REPO', tmp_path)
    monkeypatch.setattr(mod, 'SEARCH_ROOTS', [tmp_path / '.worktrees'])

    ret = mod.main()
    captured = capsys.readouterr()

    assert ret == 0
    assert 'skipped 1 unreadable review files' in captured.out
    assert 'PER-REVIEWER SUMMARY' in captured.out
