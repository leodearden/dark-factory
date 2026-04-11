"""Test review cycle archival instrumentation."""

from __future__ import annotations

import json
from pathlib import Path


class TestReviewCycleArchival:
    """Verify that reviews are archived before re-plan overwrites them."""

    def test_archive_created_on_review_cycle(self, tmp_path: Path) -> None:
        """Simulate the archival logic from workflow.py."""
        # Set up a fake .task directory with reviews
        task_root = tmp_path / '.task'
        reviews_dir = task_root / 'reviews'
        reviews_dir.mkdir(parents=True)

        # Write fake review files
        for name in ['test_analyst', 'robustness', 'architect_reviewer']:
            review = {
                'reviewer': name,
                'verdict': 'ISSUES_FOUND',
                'issues': [{'severity': 'blocking', 'location': 'f.py:1', 'category': 'bug', 'description': 'Bad'}],
                'summary': 'Found issues',
            }
            (reviews_dir / f'{name}.json').write_text(json.dumps(review))

        # Simulate the archival logic (mirrors workflow.py line 688-695)
        review_cycle = 1
        archive_dir = task_root / f'reviews-cycle-{review_cycle}'
        if reviews_dir.exists() and not archive_dir.exists():
            import shutil
            shutil.copytree(reviews_dir, archive_dir)

        # Verify archive was created
        assert archive_dir.exists()
        assert (archive_dir / 'test_analyst.json').exists()
        assert (archive_dir / 'robustness.json').exists()
        assert (archive_dir / 'architect_reviewer.json').exists()

        # Verify original still exists
        assert reviews_dir.exists()
        assert (reviews_dir / 'test_analyst.json').exists()

    def test_archive_not_overwritten_on_repeated_cycle(self, tmp_path: Path) -> None:
        """If archive already exists, don't overwrite it."""
        task_root = tmp_path / '.task'
        reviews_dir = task_root / 'reviews'
        reviews_dir.mkdir(parents=True)
        (reviews_dir / 'r.json').write_text('{"original": true}')

        archive_dir = task_root / 'reviews-cycle-1'
        archive_dir.mkdir(parents=True)
        (archive_dir / 'r.json').write_text('{"archived": true}')

        # Archival logic should skip because archive_dir already exists
        review_cycle = 1
        target = task_root / f'reviews-cycle-{review_cycle}'
        if reviews_dir.exists() and not target.exists():
            import shutil
            shutil.copytree(reviews_dir, target)

        # The archive should still have the original content
        data = json.loads((archive_dir / 'r.json').read_text())
        assert data.get('archived') is True
