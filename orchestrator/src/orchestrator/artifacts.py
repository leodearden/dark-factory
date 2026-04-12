"""Manages the .task/ directory structure in each worktree."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

PLAN_SCHEMA_VERSION = 1

# Keys the architect LLM sometimes uses instead of the required "steps".
_STEPS_ALIASES = ('tdd_steps', 'tdd_plan', 'implementation_steps')

# Key for deeply-nested phase structures (tdd_implementation_plan.phase_1_red, …).
_NESTED_PLAN_KEY = 'tdd_implementation_plan'


def _wrap_string_items(items: list, prefix: str) -> tuple[list, bool]:
    """Wrap any plain-string items in a list into canonical step dicts.

    Returns (new_list, was_modified).
    """
    modified = False
    out: list = []
    for i, item in enumerate(items):
        if isinstance(item, str):
            out.append({
                'id': f'{prefix}-{i + 1}',
                'description': item,
                'status': 'pending',
                'commit': None,
            })
            modified = True
        else:
            out.append(item)
    return out, modified


def _flatten_phases(phases: dict) -> list[dict]:
    """Flatten a nested tdd_implementation_plan dict into a flat step list."""
    steps: list[dict] = []
    step_counter = 0
    for _phase_key in sorted(phases):
        value = phases[_phase_key]
        if isinstance(value, list):
            items = value
        elif isinstance(value, (dict, str)):
            items = [value]
        else:
            logger.warning('Skipping unrecognisable phase value type %s in %s',
                           type(value).__name__, _phase_key)
            continue
        for item in items:
            step_counter += 1
            if isinstance(item, dict):
                # Ensure it has an id
                if 'id' not in item:
                    item['id'] = f'step-{step_counter}'
                if 'status' not in item:
                    item['status'] = 'pending'
                if 'commit' not in item:
                    item['commit'] = None
                steps.append(item)
            elif isinstance(item, str):
                steps.append({
                    'id': f'step-{step_counter}',
                    'description': item,
                    'status': 'pending',
                    'commit': None,
                })
            else:
                logger.warning('Skipping non-dict/str item in phase %s', _phase_key)
    return steps


def _normalize_plan(plan: dict) -> tuple[dict, bool]:
    """Detect and fix common malformed plan shapes.

    Returns (plan, was_modified).  The caller should write back to disk
    when *was_modified* is True.
    """
    modified = False

    # Rule 1-2: rename known step-key aliases → steps
    if 'steps' not in plan:
        for alias in _STEPS_ALIASES:
            if alias in plan and isinstance(plan[alias], list):
                logger.warning('Normalizing plan: renaming %r → "steps"', alias)
                plan['steps'] = plan.pop(alias)
                modified = True
                break

    # Rule 3: flatten nested tdd_implementation_plan phases
    if 'steps' not in plan and _NESTED_PLAN_KEY in plan:
        nested = plan[_NESTED_PLAN_KEY]
        if isinstance(nested, dict):
            logger.warning(
                'Normalizing plan: flattening %s phases from %r',
                len(nested), _NESTED_PLAN_KEY,
            )
            plan['steps'] = _flatten_phases(nested)
            plan.pop(_NESTED_PLAN_KEY)
            modified = True

    # Rule 4: wrap string prerequisites
    prereqs = plan.get('prerequisites')
    if isinstance(prereqs, list):
        new_prereqs, changed = _wrap_string_items(prereqs, 'pre')
        if changed:
            logger.warning('Normalizing plan: wrapping %d string prerequisites as dicts',
                           sum(1 for p in prereqs if isinstance(p, str)))
            plan['prerequisites'] = new_prereqs
            modified = True

    # Rule 5: wrap string steps
    steps = plan.get('steps')
    if isinstance(steps, list):
        new_steps, changed = _wrap_string_items(steps, 'step')
        if changed:
            logger.warning('Normalizing plan: wrapping %d string steps as dicts',
                           sum(1 for s in steps if isinstance(s, str)))
            plan['steps'] = new_steps
            modified = True

    return plan, modified


@dataclass
class ReviewAggregation:
    """Aggregated review results from all reviewers."""

    has_blocking_issues: bool
    blocking_issues: list[dict]
    suggestions: list[dict]
    reviews: dict[str, dict]
    reviewer_errors: list[str] = field(default_factory=list)

    @property
    def all_reviewers_errored(self) -> bool:
        """True when every reviewer returned ERROR (no usable reviews)."""
        return bool(self.reviewer_errors) and not self.reviews

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

    def update_base_commit(self, new_base: str) -> None:
        """Update the base_commit in metadata.json after a rebase."""
        meta_path = self.root / 'metadata.json'
        if not meta_path.exists():
            return
        metadata = json.loads(meta_path.read_text())
        metadata['base_commit'] = new_base
        self._write_json(meta_path, metadata)

    def write_plan(self, plan: dict) -> None:
        """Write .task/plan.json — the structured plan."""
        plan['_schema_version'] = PLAN_SCHEMA_VERSION
        self._write_json(self.root / 'plan.json', plan)

    def read_plan(self) -> dict:
        """Read current plan state, auto-normalizing malformed shapes."""
        plan_path = self.root / 'plan.json'
        if not plan_path.exists():
            return {}
        plan = json.loads(plan_path.read_text())
        plan, modified = _normalize_plan(plan)
        if modified:
            logger.warning(
                'Plan normalization applied — writing corrected plan to %s',
                plan_path,
            )
            self._write_json(plan_path, plan)
        return plan

    def update_step_status(
        self, step_id: str, status: str, commit: str | None = None
    ) -> None:
        """Update ONLY status and commit fields in plan.json. Structure is immutable."""
        plan = self.read_plan()

        for collection in ('prerequisites', 'steps'):
            for item in plan.get(collection, []):
                if not isinstance(item, dict):
                    continue
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
                if isinstance(item, dict) and item.get('status') == 'pending':
                    pending.append(item)
        return pending

    def get_completed_steps(self) -> list[dict]:
        """Return all steps with status 'done'."""
        plan = self.read_plan()
        completed = []
        for collection in ('prerequisites', 'steps'):
            for item in plan.get(collection, []):
                if isinstance(item, dict) and item.get('status') == 'done':
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
        """Parse all reviews, separate blocking from suggestions.

        Reviews with verdict ``ERROR`` are filtered into a separate list
        so the workflow can detect incomplete reviewer coverage.
        """
        reviews = self.read_reviews()
        blocking = []
        suggestions = []
        errors = []

        for reviewer_name, review in reviews.items():
            if review.get('verdict') == 'ERROR':
                errors.append(reviewer_name)
                continue
            for issue in review.get('issues', []):
                enriched = {**issue, 'reviewer': reviewer_name}
                if issue.get('severity') == 'blocking':
                    blocking.append(enriched)
                else:
                    suggestions.append(enriched)

        clean = {k: v for k, v in reviews.items() if k not in errors}
        return ReviewAggregation(
            has_blocking_issues=len(blocking) > 0,
            blocking_issues=blocking,
            suggestions=suggestions,
            reviews=clean,
            reviewer_errors=errors,
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

    def clear_stale_plan_lock(
        self, current_task_id: str, stale_threshold_secs: float = 600.0
    ) -> bool:
        """Remove plan.lock if it's stale. Returns True if lock was cleared."""
        lock_data = self.read_plan_lock()
        if lock_data is None:
            return False

        lock_session = lock_data.get('session_id', '')
        locked_at_str = lock_data.get('locked_at', '')

        # Self-lock from a crashed previous session of the same task — always clear.
        # Session IDs are formatted as '{task_id}-{uuid_hex[:8]}', so a lock
        # from any prior session of the same task shares the task_id prefix.
        if lock_session.startswith(current_task_id + '-'):
            logger.info(
                'Clearing self-owned plan.lock (session %s, task %s)',
                lock_session,
                current_task_id,
            )
            (self.root / 'plan.lock').unlink(missing_ok=True)
            return True

        # Age-based stale detection
        try:
            locked_at = datetime.fromisoformat(locked_at_str)
            age_secs = (datetime.now(UTC) - locked_at).total_seconds()
            if age_secs > stale_threshold_secs:
                logger.warning(
                    'Clearing stale plan.lock (session %s, age %.0fs > %.0fs threshold)',
                    lock_session,
                    age_secs,
                    stale_threshold_secs,
                )
                (self.root / 'plan.lock').unlink(missing_ok=True)
                return True
        except (ValueError, TypeError):
            # Unparseable timestamp — treat as stale
            logger.warning(
                'Clearing plan.lock with unparseable timestamp: %r', locked_at_str
            )
            (self.root / 'plan.lock').unlink(missing_ok=True)
            return True

        return False

    def _write_json(self, path: Path, data: dict) -> None:
        path.write_text(json.dumps(data, indent=2) + '\n')
