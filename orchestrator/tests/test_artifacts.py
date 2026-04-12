"""Tests for task artifacts management."""

import json
from pathlib import Path

import pytest

from orchestrator.artifacts import TaskArtifacts


@pytest.fixture
def worktree(tmp_path: Path) -> Path:
    return tmp_path / 'worktree'


@pytest.fixture
def artifacts(worktree: Path) -> TaskArtifacts:
    worktree.mkdir()
    ta = TaskArtifacts(worktree)
    ta.init('task-1', 'Test Task', 'A test task description')
    return ta


class TestInit:
    def test_creates_directory_structure(self, artifacts: TaskArtifacts):
        assert artifacts.root.exists()
        assert (artifacts.root / 'metadata.json').exists()
        assert (artifacts.root / 'reviews').is_dir()

    def test_creates_gitignore(self, artifacts: TaskArtifacts):
        gitignore = artifacts.root / '.gitignore'
        assert gitignore.exists()
        content = gitignore.read_text()
        # Must contain the wildcard that ignores all scratch files
        assert '*\n' in content
        # Must contain warning comments to dissuade agents from removing it
        assert 'DO NOT' in content

    def test_metadata_contents(self, artifacts: TaskArtifacts):
        metadata = json.loads((artifacts.root / 'metadata.json').read_text())
        assert metadata['task_id'] == 'task-1'
        assert metadata['title'] == 'Test Task'
        assert metadata['description'] == 'A test task description'
        assert 'created_at' in metadata

    def test_base_commit_stored(self, worktree: Path):
        worktree.mkdir()
        ta = TaskArtifacts(worktree)
        ta.init('task-2', 'Test', 'Desc', base_commit='abc123def456')
        assert ta.read_base_commit() == 'abc123def456'

    def test_base_commit_absent_when_not_provided(self, artifacts: TaskArtifacts):
        assert artifacts.read_base_commit() is None


class TestPlan:
    def test_write_and_read_plan(self, artifacts: TaskArtifacts):
        plan = {
            'task_id': 'task-1',
            'title': 'Test Task',
            'modules': ['backend'],
            'steps': [
                {'id': 'step-1', 'type': 'test', 'description': 'Write test', 'status': 'pending', 'commit': None},
                {'id': 'step-2', 'type': 'impl', 'description': 'Implement', 'status': 'pending', 'commit': None},
            ],
        }
        artifacts.write_plan(plan)
        read = artifacts.read_plan()
        assert read['task_id'] == 'task-1'
        assert len(read['steps']) == 2
        assert read['_schema_version'] == 1

    def test_update_step_status(self, artifacts: TaskArtifacts):
        plan = {
            'task_id': 'task-1',
            'steps': [
                {'id': 'step-1', 'status': 'pending', 'commit': None},
                {'id': 'step-2', 'status': 'pending', 'commit': None},
            ],
        }
        artifacts.write_plan(plan)

        artifacts.update_step_status('step-1', 'done', commit='abc123')

        updated = artifacts.read_plan()
        assert updated['steps'][0]['status'] == 'done'
        assert updated['steps'][0]['commit'] == 'abc123'
        assert updated['steps'][1]['status'] == 'pending'

    def test_update_preserves_structure(self, artifacts: TaskArtifacts):
        plan = {
            'task_id': 'task-1',
            'analysis': 'Some analysis',
            'steps': [
                {'id': 'step-1', 'type': 'test', 'description': 'Write test', 'status': 'pending', 'commit': None},
            ],
            'design_decisions': [{'decision': 'Use X', 'rationale': 'Because Y'}],
        }
        artifacts.write_plan(plan)
        artifacts.update_step_status('step-1', 'done', commit='sha')

        updated = artifacts.read_plan()
        assert updated['analysis'] == 'Some analysis'
        assert updated['design_decisions'] == [{'decision': 'Use X', 'rationale': 'Because Y'}]
        assert updated['steps'][0]['type'] == 'test'
        assert updated['steps'][0]['description'] == 'Write test'

    def test_get_pending_and_completed_steps(self, artifacts: TaskArtifacts):
        plan = {
            'task_id': 'task-1',
            'prerequisites': [
                {'id': 'pre-1', 'status': 'done', 'commit': 'abc'},
            ],
            'steps': [
                {'id': 'step-1', 'status': 'done', 'commit': 'def'},
                {'id': 'step-2', 'status': 'pending', 'commit': None},
                {'id': 'step-3', 'status': 'pending', 'commit': None},
            ],
        }
        artifacts.write_plan(plan)

        pending = artifacts.get_pending_steps()
        assert len(pending) == 2
        assert pending[0]['id'] == 'step-2'

        completed = artifacts.get_completed_steps()
        assert len(completed) == 2  # pre-1 + step-1


class TestIterationLog:
    def test_append_and_read(self, artifacts: TaskArtifacts):
        artifacts.append_iteration_log({
            'iteration': 1,
            'agent': 'implementer',
            'steps_completed': ['step-1'],
            'summary': 'First iteration',
        })
        artifacts.append_iteration_log({
            'iteration': 2,
            'agent': 'implementer',
            'steps_completed': ['step-2'],
            'summary': 'Second iteration',
        })

        log, corrupted = artifacts.read_iteration_log()
        assert len(log) == 2
        assert corrupted == []
        assert log[0]['iteration'] == 1
        assert log[1]['iteration'] == 2
        assert 'timestamp' in log[0]

    def test_read_skips_corrupted_lines(self, artifacts: TaskArtifacts):
        """Good/bad/good lines → 2 entries, 1 corrupted."""
        log_path = artifacts.root / 'iterations.jsonl'
        lines = [
            json.dumps({'iteration': 1, 'agent': 'implementer', 'summary': 'ok'}),
            r'{"iteration": 2, "summary": "has bad escape \!"}',
            json.dumps({'iteration': 3, 'agent': 'implementer', 'summary': 'also ok'}),
        ]
        log_path.write_text('\n'.join(lines) + '\n')

        entries, corrupted = artifacts.read_iteration_log()
        assert len(entries) == 2
        assert entries[0]['iteration'] == 1
        assert entries[1]['iteration'] == 3
        assert len(corrupted) == 1
        assert r'\!' in corrupted[0]

    def test_read_all_corrupted(self, artifacts: TaskArtifacts):
        """All bad lines → empty entries, all lines in corrupted."""
        log_path = artifacts.root / 'iterations.jsonl'
        bad_lines = [
            r'not json at all',
            r'{"truncated": true',
            r'{bad \escape}',
        ]
        log_path.write_text('\n'.join(bad_lines) + '\n')

        entries, corrupted = artifacts.read_iteration_log()
        assert entries == []
        assert len(corrupted) == 3


VALID_PLAN_WITH_STEPS = {
    'task_id': 'task-1',
    'title': 'Test Task',
    'steps': [
        {'id': 'step-1', 'type': 'test', 'description': 'Write test', 'status': 'pending'},
    ],
}


class TestPlanProvenance:
    def test_stamp_plan_provenance_adds_session_id_and_created_at(self, artifacts: TaskArtifacts):
        # Must use a valid plan with at least one step (empty steps raises ValueError)
        artifacts.write_plan(dict(VALID_PLAN_WITH_STEPS))
        artifacts.stamp_plan_provenance('session-abc123')
        updated = artifacts.read_plan()
        assert updated['_session_id'] == 'session-abc123'
        assert '_created_at' in updated

    def test_stamp_plan_provenance_preserves_existing_plan_data(self, artifacts: TaskArtifacts):
        plan = {
            'task_id': 'task-1',
            'analysis': 'Some analysis',
            'steps': [{'id': 'step-1', 'status': 'pending'}],
        }
        artifacts.write_plan(plan)
        artifacts.stamp_plan_provenance('session-abc123')
        updated = artifacts.read_plan()
        assert updated['task_id'] == 'task-1'
        assert updated['analysis'] == 'Some analysis'
        assert len(updated['steps']) == 1

    def test_validate_plan_owner_true_for_matching_session(self, artifacts: TaskArtifacts):
        artifacts.write_plan(dict(VALID_PLAN_WITH_STEPS))
        artifacts.stamp_plan_provenance('session-abc123')
        assert artifacts.validate_plan_owner('session-abc123') is True

    def test_validate_plan_owner_false_for_mismatched_session(self, artifacts: TaskArtifacts):
        artifacts.write_plan(dict(VALID_PLAN_WITH_STEPS))
        artifacts.stamp_plan_provenance('session-abc123')
        assert artifacts.validate_plan_owner('session-different') is False

    def test_validate_plan_owner_false_when_no_provenance(self, artifacts: TaskArtifacts):
        artifacts.write_plan(dict(VALID_PLAN_WITH_STEPS))
        # Not stamped — no _session_id in plan
        assert artifacts.validate_plan_owner('session-abc123') is False

    def test_stamp_plan_provenance_raises_on_missing_plan(self, artifacts: TaskArtifacts):
        """stamp_plan_provenance() must raise ValueError if plan.json does not exist."""
        # Ensure plan.json does not exist
        plan_path = artifacts.root / 'plan.json'
        assert not plan_path.exists()

        with pytest.raises(ValueError, match='valid plan'):
            artifacts.stamp_plan_provenance('session-abc123')

    def test_stamp_plan_provenance_raises_on_empty_plan(self, artifacts: TaskArtifacts):
        """stamp_plan_provenance() must raise ValueError if plan.json has no steps key."""
        # Write a plan with no 'steps' key at all
        (artifacts.root / 'plan.json').write_text('{}')

        with pytest.raises(ValueError, match='valid plan'):
            artifacts.stamp_plan_provenance('session-abc123')

    def test_stamp_plan_provenance_raises_on_provenance_only_stub(self, artifacts: TaskArtifacts):
        """stamp_plan_provenance() must raise ValueError on a provenance-only stub.

        A stub with only _session_id/_created_at (no steps) passes bool() checks but
        has no actual plan content — stamping it silently would hide data loss.
        """
        stub = {'_session_id': 'old-session', '_created_at': '2026-01-01T00:00:00+00:00'}
        (artifacts.root / 'plan.json').write_text(
            __import__('json').dumps(stub)
        )

        with pytest.raises(ValueError, match='valid plan'):
            artifacts.stamp_plan_provenance('session-abc123')

    def test_validate_plan_owner_returns_false_on_corrupt_plan_json(
        self, artifacts: TaskArtifacts
    ):
        """Corrupt (non-JSON) plan.json must return False, not raise JSONDecodeError."""
        plan_path = artifacts.root / 'plan.json'
        plan_path.write_text('this is not valid json }{}{')

        # Must return False without raising any exception
        result = artifacts.validate_plan_owner('session-abc123')
        assert result is False

    def test_validate_plan_owner_returns_false_on_unreadable_plan(
        self, artifacts: TaskArtifacts
    ):
        """Unreadable plan.json (chmod 000) must return False, not raise OSError."""
        plan_path = artifacts.root / 'plan.json'
        plan_path.write_text('{"_session_id": "session-abc123"}')
        plan_path.chmod(0o000)
        try:
            result = artifacts.validate_plan_owner('session-abc123')
            assert result is False
        finally:
            # Restore permissions so cleanup works
            plan_path.chmod(0o644)


class TestStalePlanLock:
    """Tests for stale plan.lock detection and cleanup."""

    def _create_lock(self, artifacts: TaskArtifacts, session_id: str, locked_at: str):
        lock_path = artifacts.root / 'plan.lock'
        lock_path.write_text(json.dumps({
            'session_id': session_id,
            'locked_at': locked_at,
        }))

    def test_self_lock_cleared(self, artifacts: TaskArtifacts):
        """Lock from a prior session of the same task (matching task_id prefix) is cleared."""
        self._create_lock(artifacts, 'task-1-abcd1234', '2026-04-03T12:00:00+00:00')
        result = artifacts.clear_stale_plan_lock('task-1')
        assert result is True
        assert not (artifacts.root / 'plan.lock').exists()

    def test_recent_foreign_lock_preserved(self, artifacts: TaskArtifacts):
        """Lock from a different task with recent timestamp is NOT cleared."""
        from datetime import UTC, datetime
        recent = datetime.now(UTC).isoformat()
        self._create_lock(artifacts, 'task-99-ffff0000', recent)
        result = artifacts.clear_stale_plan_lock('task-1')
        assert result is False
        assert (artifacts.root / 'plan.lock').exists()

    def test_old_foreign_lock_cleared(self, artifacts: TaskArtifacts):
        """Lock from a different task with old timestamp IS cleared."""
        self._create_lock(artifacts, 'task-99-ffff0000', '2025-01-01T00:00:00+00:00')
        result = artifacts.clear_stale_plan_lock('task-1', stale_threshold_secs=600.0)
        assert result is True
        assert not (artifacts.root / 'plan.lock').exists()

    def test_no_lock_returns_false(self, artifacts: TaskArtifacts):
        """No lock file → returns False."""
        result = artifacts.clear_stale_plan_lock('task-1')
        assert result is False

    def test_unparseable_timestamp_cleared(self, artifacts: TaskArtifacts):
        """Lock with garbage timestamp is treated as stale and cleared."""
        self._create_lock(artifacts, 'task-99-ffff0000', 'not-a-timestamp')
        result = artifacts.clear_stale_plan_lock('task-1')
        assert result is True
        assert not (artifacts.root / 'plan.lock').exists()


class TestPlanLock:
    def test_is_plan_locked_false_initially(self, artifacts: TaskArtifacts):
        assert artifacts.is_plan_locked() is False

    def test_lock_plan_creates_file_returns_true(self, artifacts: TaskArtifacts):
        result = artifacts.lock_plan('session-abc123')
        assert result is True
        assert (artifacts.root / 'plan.lock').exists()

    def test_is_plan_locked_true_after_lock(self, artifacts: TaskArtifacts):
        artifacts.lock_plan('session-abc123')
        assert artifacts.is_plan_locked() is True

    def test_lock_plan_returns_false_when_already_locked(self, artifacts: TaskArtifacts):
        first = artifacts.lock_plan('session-abc123')
        second = artifacts.lock_plan('session-different')
        assert first is True
        assert second is False

    def test_read_plan_lock_returns_session_and_timestamp(self, artifacts: TaskArtifacts):
        artifacts.lock_plan('session-abc123')
        lock_data = artifacts.read_plan_lock()
        assert lock_data is not None
        assert lock_data['session_id'] == 'session-abc123'
        assert 'locked_at' in lock_data

    def test_read_plan_lock_returns_none_when_not_locked(self, artifacts: TaskArtifacts):
        assert artifacts.read_plan_lock() is None


class TestReviews:
    def test_write_and_read_reviews(self, artifacts: TaskArtifacts):
        artifacts.write_review('test_analyst', {
            'reviewer': 'test_analyst',
            'verdict': 'PASS',
            'issues': [],
            'summary': 'Looks good',
        })
        artifacts.write_review('performance', {
            'reviewer': 'performance',
            'verdict': 'ISSUES_FOUND',
            'issues': [
                {'severity': 'suggestion', 'location': 'foo.py:10', 'category': 'perf', 'description': 'N+1'},
            ],
            'summary': 'Minor perf issue',
        })

        reviews = artifacts.read_reviews()
        assert 'test_analyst' in reviews
        assert 'performance' in reviews

    def test_aggregate_reviews_no_blocking(self, artifacts: TaskArtifacts):
        artifacts.write_review('reviewer1', {
            'reviewer': 'reviewer1',
            'verdict': 'PASS',
            'issues': [],
            'summary': 'OK',
        })
        artifacts.write_review('reviewer2', {
            'reviewer': 'reviewer2',
            'verdict': 'ISSUES_FOUND',
            'issues': [
                {'severity': 'suggestion', 'location': 'a.py:1', 'category': 'style', 'description': 'naming'},
            ],
            'summary': 'Minor',
        })

        agg = artifacts.aggregate_reviews()
        assert not agg.has_blocking_issues
        assert len(agg.suggestions) == 1

    def test_aggregate_reviews_with_blocking(self, artifacts: TaskArtifacts):
        artifacts.write_review('reviewer1', {
            'reviewer': 'reviewer1',
            'verdict': 'ISSUES_FOUND',
            'issues': [
                {'severity': 'blocking', 'location': 'b.py:5', 'category': 'bug', 'description': 'Missing null check'},
            ],
            'summary': 'Bug found',
        })

        agg = artifacts.aggregate_reviews()
        assert agg.has_blocking_issues
        assert len(agg.blocking_issues) == 1
        assert agg.blocking_issues[0]['reviewer'] == 'reviewer1'

    def test_format_for_replan(self, artifacts: TaskArtifacts):
        artifacts.write_review('reviewer1', {
            'reviewer': 'reviewer1',
            'verdict': 'ISSUES_FOUND',
            'issues': [
                {
                    'severity': 'blocking',
                    'location': 'c.py:10',
                    'category': 'missing_test',
                    'description': 'No test for empty input',
                    'suggested_fix': 'Add parametrized test',
                },
            ],
            'summary': 'Missing test',
        })

        agg = artifacts.aggregate_reviews()
        text = agg.format_for_replan()
        assert 'missing_test' in text
        assert 'No test for empty input' in text
        assert 'c.py:10' in text

    def test_aggregate_reviews_error_filtered(self, artifacts: TaskArtifacts):
        artifacts.write_review('reviewer1', {
            'reviewer': 'reviewer1',
            'verdict': 'PASS',
            'issues': [],
            'summary': 'OK',
        })
        artifacts.write_review('reviewer2', {
            'reviewer': 'reviewer2',
            'verdict': 'ERROR',
            'issues': [],
            'summary': 'Reviewer error: 401 Unauthorized',
        })

        agg = artifacts.aggregate_reviews()
        assert not agg.has_blocking_issues
        assert agg.reviewer_errors == ['reviewer2']
        assert 'reviewer1' in agg.reviews
        assert 'reviewer2' not in agg.reviews
        assert not agg.all_reviewers_errored

    def test_aggregate_reviews_all_errors(self, artifacts: TaskArtifacts):
        artifacts.write_review('reviewer1', {
            'reviewer': 'reviewer1',
            'verdict': 'ERROR',
            'issues': [],
            'summary': 'Reviewer error: 401',
        })
        artifacts.write_review('reviewer2', {
            'reviewer': 'reviewer2',
            'verdict': 'ERROR',
            'issues': [],
            'summary': 'Reviewer error: timeout',
        })

        agg = artifacts.aggregate_reviews()
        assert agg.all_reviewers_errored
        assert len(agg.reviewer_errors) == 2
        assert not agg.reviews

    def test_aggregate_reviews_error_plus_blocking(self, artifacts: TaskArtifacts):
        artifacts.write_review('reviewer1', {
            'reviewer': 'reviewer1',
            'verdict': 'ERROR',
            'issues': [],
            'summary': 'Reviewer error: 401',
        })
        artifacts.write_review('reviewer2', {
            'reviewer': 'reviewer2',
            'verdict': 'ISSUES_FOUND',
            'issues': [
                {'severity': 'blocking', 'location': 'd.py:1',
                 'category': 'bug', 'description': 'Null deref'},
            ],
            'summary': 'Bug found',
        })

        agg = artifacts.aggregate_reviews()
        assert agg.has_blocking_issues
        assert agg.reviewer_errors == ['reviewer1']
        assert len(agg.blocking_issues) == 1


# --- Prerequisites plans for validate_plan_prerequisites tests ---

VALID_PREREQS = [
    {'id': 'pre-1', 'description': 'Install deps', 'status': 'pending', 'commit': None, 'tests': []},
    {'id': 'pre-2', 'description': 'Set up config', 'status': 'done', 'commit': 'abc123', 'tests': []},
]

STRING_PREREQS_PLAN = {
    'task_id': 'task-1',
    'steps': [{'id': 'step-1', 'status': 'pending'}],
    'prerequisites': ['install deps', 'set up config'],
}

MIXED_PREREQS_PLAN = {
    'task_id': 'task-1',
    'steps': [{'id': 'step-1', 'status': 'pending'}],
    'prerequisites': [
        {'id': 'pre-1', 'description': 'Install deps', 'status': 'pending'},
        'set up config',
    ],
}

MISSING_KEYS_PREREQS_PLAN = {
    'task_id': 'task-1',
    'steps': [{'id': 'step-1', 'status': 'pending'}],
    'prerequisites': [
        {'id': 'pre-1', 'description': 'Install deps'},  # missing 'status'
    ],
}

VALID_PREREQS_PLAN = {
    'task_id': 'task-1',
    'steps': [{'id': 'step-1', 'status': 'pending'}],
    'prerequisites': VALID_PREREQS,
}

NO_PREREQS_PLAN = {
    'task_id': 'task-1',
    'steps': [{'id': 'step-1', 'status': 'pending'}],
}

EMPTY_PREREQS_PLAN = {
    'task_id': 'task-1',
    'steps': [{'id': 'step-1', 'status': 'pending'}],
    'prerequisites': [],
}


class TestValidatePlanPrerequisites:
    """Tests for validate_plan_prerequisites() method on TaskArtifacts."""

    # --- step-1: plain string list raises ValueError ---
    def test_raises_for_plain_string_prerequisites(self, artifacts: TaskArtifacts):
        """All plain-string prerequisites list must raise ValueError."""
        artifacts.write_plan(STRING_PREREQS_PLAN)
        with pytest.raises(ValueError, match='prerequisites'):
            artifacts.validate_plan_prerequisites()

    def test_error_message_includes_index_and_value(self, artifacts: TaskArtifacts):
        """Error message must include the offending index and value."""
        artifacts.write_plan(STRING_PREREQS_PLAN)
        with pytest.raises(ValueError) as exc_info:
            artifacts.validate_plan_prerequisites()
        msg = str(exc_info.value)
        assert '0' in msg  # first offending index
        assert 'install deps' in msg.lower()

    # --- step-2: mixed list (some strings, some dicts) raises ValueError ---
    def test_raises_for_mixed_prerequisites(self, artifacts: TaskArtifacts):
        """Mixed list with at least one non-dict must raise ValueError."""
        artifacts.write_plan(MIXED_PREREQS_PLAN)
        with pytest.raises(ValueError, match='prerequisites'):
            artifacts.validate_plan_prerequisites()

    def test_mixed_error_identifies_string_item_not_dict(self, artifacts: TaskArtifacts):
        """Error message pinpoints the string item (index 1), not the valid dict."""
        artifacts.write_plan(MIXED_PREREQS_PLAN)
        with pytest.raises(ValueError) as exc_info:
            artifacts.validate_plan_prerequisites()
        msg = str(exc_info.value)
        assert '1' in msg  # second item (index 1) is the string

    # --- step-3: dicts missing required keys raise ValueError ---
    def test_raises_for_dict_missing_required_keys(self, artifacts: TaskArtifacts):
        """Dict prerequisite missing 'status' must raise ValueError."""
        artifacts.write_plan(MISSING_KEYS_PREREQS_PLAN)
        with pytest.raises(ValueError, match='prerequisites'):
            artifacts.validate_plan_prerequisites()

    def test_error_message_mentions_missing_key(self, artifacts: TaskArtifacts):
        """Error message must identify which key is missing."""
        artifacts.write_plan(MISSING_KEYS_PREREQS_PLAN)
        with pytest.raises(ValueError) as exc_info:
            artifacts.validate_plan_prerequisites()
        msg = str(exc_info.value)
        assert 'status' in msg

    # --- step-4: valid / absent prerequisites pass silently ---
    def test_passes_for_valid_dict_prerequisites(self, artifacts: TaskArtifacts):
        """All required-key dicts must pass without raising."""
        artifacts.write_plan(VALID_PREREQS_PLAN)
        result = artifacts.validate_plan_prerequisites()
        assert result is None

    def test_passes_when_prerequisites_absent(self, artifacts: TaskArtifacts):
        """Plan with no 'prerequisites' key must pass without raising."""
        artifacts.write_plan(NO_PREREQS_PLAN)
        result = artifacts.validate_plan_prerequisites()
        assert result is None

    def test_passes_for_empty_prerequisites_list(self, artifacts: TaskArtifacts):
        """Empty prerequisites list must pass without raising."""
        artifacts.write_plan(EMPTY_PREREQS_PLAN)
        result = artifacts.validate_plan_prerequisites()
        assert result is None
