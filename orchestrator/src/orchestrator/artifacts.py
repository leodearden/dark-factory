"""Manages the .task/ directory structure in each worktree."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

PLAN_SCHEMA_VERSION = 1


@dataclass
class ReviewAggregation:
    """Aggregated review results from all reviewers."""

    has_blocking_issues: bool
    blocking_issues: list[dict]
    suggestions: list[dict]
    reviews: dict[str, dict]

    def format_for_replan(self) -> str:
        """Format blocking issues for the architect to address."""
        lines = ['# Review Feedback — Blocking Issues\n']
        for issue in self.blocking_issues:
            reviewer = issue.get('reviewer', 'unknown')
            location = issue.get('location', '')
            category = issue.get('category', '')
            description = issue.get('description', '')
            fix = issue.get('suggested_fix', '')
            lines.append(f'## [{reviewer}] {category}')
            if location:
                lines.append(f'**Location:** {location}')
            lines.append(f'**Issue:** {description}')
            if fix:
                lines.append(f'**Suggested fix:** {fix}')
            lines.append('')
        return '\n'.join(lines)


class TaskArtifacts:
    """Manages .task/ directory in a worktree."""

    def __init__(self, worktree: Path):
        self.root = worktree / '.task'

    def init(
        self,
        task_id: str,
        task_title: str,
        task_description: str,
        base_commit: str | None = None,
    ) -> None:
        """Create .task/ with initial metadata.json and .gitignore.

        IMPORTANT: This method is safe to call when .task/ already exists
        (e.g. inherited from a contaminated main).  It unconditionally
        writes .task/.gitignore to ensure git ignores the contents even
        if the directory was pre-existing and tracked.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / 'reviews').mkdir(exist_ok=True)

        # ── .task/.gitignore — always written unconditionally ─────────
        # This file tells git to ignore everything inside .task/.
        # We write it every time init() is called, not just on first
        # creation, because:
        # 1. If .task/ was inherited from a contaminated main, the
        #    .gitignore may be missing or have different contents.
        # 2. An agent may have deleted or modified it.
        # 3. It's idempotent and cheap.
        #
        # This is ONE layer of defense.  See git_ops.py module docstring
        # for the full list of .task/ contamination safeguards.
        (self.root / '.gitignore').write_text(
            '# Ignore ALL orchestrator scratch files.\n'
            '# DO NOT remove or modify this file.  .task/ must never be\n'
            '# committed to any branch that will be merged to main.\n'
            '# See git_ops.py module docstring for why this matters.\n'
            '*\n'
        )

        metadata = {
            'task_id': task_id,
            'title': task_title,
            'description': task_description,
            'created_at': datetime.now(UTC).isoformat(),
        }
        if base_commit:
            metadata['base_commit'] = base_commit
        self._write_json(self.root / 'metadata.json', metadata)

    def read_base_commit(self) -> str | None:
        """Read the base commit SHA stored at init time."""
        meta_path = self.root / 'metadata.json'
        if not meta_path.exists():
            return None
        metadata = json.loads(meta_path.read_text())
        return metadata.get('base_commit')

    def write_plan(self, plan: dict) -> None:
        """Write .task/plan.json — the structured plan."""
        plan['_schema_version'] = PLAN_SCHEMA_VERSION
        self._write_json(self.root / 'plan.json', plan)

    def read_plan(self) -> dict:
        """Read current plan state."""
        plan_path = self.root / 'plan.json'
        if not plan_path.exists():
            return {}
        return json.loads(plan_path.read_text())

    def update_step_status(
        self, step_id: str, status: str, commit: str | None = None
    ) -> None:
        """Update ONLY status and commit fields in plan.json. Structure is immutable."""
        plan = self.read_plan()

        for collection in ('prerequisites', 'steps'):
            for item in plan.get(collection, []):
                if item.get('id') == step_id:
                    item['status'] = status
                    if commit is not None:
                        item['commit'] = commit
                    self._write_json(self.root / 'plan.json', plan)
                    return

        logger.warning(f'Step {step_id} not found in plan')

    def get_pending_steps(self) -> list[dict]:
        """Return all steps with status 'pending', in order."""
        plan = self.read_plan()
        pending = []
        for collection in ('prerequisites', 'steps'):
            for item in plan.get(collection, []):
                if item.get('status') == 'pending':
                    pending.append(item)
        return pending

    def get_completed_steps(self) -> list[dict]:
        """Return all steps with status 'done'."""
        plan = self.read_plan()
        completed = []
        for collection in ('prerequisites', 'steps'):
            for item in plan.get(collection, []):
                if item.get('status') == 'done':
                    completed.append(item)
        return completed

    def append_iteration_log(self, entry: dict) -> None:
        """Append to .task/iterations.jsonl — one JSON object per line."""
        entry['timestamp'] = datetime.now(UTC).isoformat()
        log_path = self.root / 'iterations.jsonl'
        with open(log_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def read_iteration_log(self) -> tuple[list[dict], list[str]]:
        """Read all iteration log entries, skipping corrupted lines.

        Returns (entries, corrupted) where corrupted contains raw lines that
        failed JSON parsing.
        """
        log_path = self.root / 'iterations.jsonl'
        if not log_path.exists():
            return [], []
        entries = []
        corrupted = []
        for line in log_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning('Skipping corrupted iteration log line: %s', line[:120])
                corrupted.append(line)
        return entries, corrupted

    def write_review(self, reviewer_name: str, review: dict) -> None:
        """Write .task/reviews/{name}.json."""
        review_path = self.root / 'reviews' / f'{reviewer_name}.json'
        self._write_json(review_path, review)

    def read_reviews(self) -> dict[str, dict]:
        """Read all reviews."""
        reviews_dir = self.root / 'reviews'
        if not reviews_dir.exists():
            return {}
        reviews = {}
        for path in reviews_dir.glob('*.json'):
            reviews[path.stem] = json.loads(path.read_text())
        return reviews

    def aggregate_reviews(self) -> ReviewAggregation:
        """Parse all reviews, separate blocking from suggestions."""
        reviews = self.read_reviews()
        blocking = []
        suggestions = []

        for reviewer_name, review in reviews.items():
            for issue in review.get('issues', []):
                enriched = {**issue, 'reviewer': reviewer_name}
                if issue.get('severity') == 'blocking':
                    blocking.append(enriched)
                else:
                    suggestions.append(enriched)

        return ReviewAggregation(
            has_blocking_issues=len(blocking) > 0,
            blocking_issues=blocking,
            suggestions=suggestions,
            reviews=reviews,
        )

    def stamp_plan_provenance(self, session_id: str) -> None:
        """Stamp _session_id and _created_at into plan.json.

        Reads current plan.json, adds provenance fields, writes back.

        Raises:
            ValueError: if plan.json is missing or does not contain a 'steps' list.
                This prevents silently creating a provenance-only stub that passes
                bool() checks but has no actual plan content.
        """
        plan = self.read_plan()
        if not plan.get('steps'):
            raise ValueError(
                'stamp_plan_provenance called before plan.json contains a valid plan '
                '(missing or empty steps). '
                'Ensure the architect has written a complete plan before stamping provenance.'
            )
        plan['_session_id'] = session_id
        plan['_created_at'] = datetime.now(UTC).isoformat()
        self._write_json(self.root / 'plan.json', plan)

    def validate_plan_owner(self, session_id: str) -> bool:
        """Return True if plan.json's _session_id matches the given session_id.

        Returns False on any read or parse error (JSONDecodeError, OSError, etc.)
        so that a corrupt or unreadable plan.json triggers the ownership-mismatch
        escalation path rather than a generic BLOCKED error.
        """
        try:
            plan_path = self.root / 'plan.json'
            data = json.loads(plan_path.read_text())
            return data.get('_session_id') == session_id
        except Exception as exc:
            logger.warning(
                'validate_plan_owner: failed to read/parse plan.json — treating as mismatch: %s',
                exc,
            )
            return False

    def lock_plan(self, session_id: str) -> bool:
        """Atomically acquire the plan lock.

        Uses O_CREAT|O_EXCL for atomic exclusive creation (POSIX).
        Returns True if the lock was acquired, False if already locked.
        """
        lock_path = self.root / 'plan.lock'
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        try:
            data = json.dumps({
                'session_id': session_id,
                'locked_at': datetime.now(UTC).isoformat(),
            })
            os.write(fd, data.encode())
        finally:
            os.close(fd)
        return True

    def is_plan_locked(self) -> bool:
        """Return True if plan.lock exists."""
        return (self.root / 'plan.lock').exists()

    def read_plan_lock(self) -> dict | None:
        """Read plan.lock contents. Returns dict or None if not locked."""
        lock_path = self.root / 'plan.lock'
        if not lock_path.exists():
            return None
        return json.loads(lock_path.read_text())

    def _write_json(self, path: Path, data: dict) -> None:
        path.write_text(json.dumps(data, indent=2) + '\n')
