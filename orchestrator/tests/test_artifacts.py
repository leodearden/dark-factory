"""Tests for task artifacts management."""

import copy
import errno
import json
import os
import re
import shutil
import stat
import sys
import threading
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from _plan_model_helpers import read_model
from _review_fixtures import review_aggregation, review_issue
from _review_payloads import blocking_review, pass_review
from escalation.disposition import ReviewerDisposition
from shared import safe_io

from orchestrator.artifacts import (
    LANE_SYMLINKED_ARTIFACTS,
    PLAN_BASE_COMMIT_KEY,
    PLAN_SCHEMA_VERSION,
    PLAN_TASK_TEXT_FP_KEY,
    ArtifactWriteError,
    PlanConfirmation,
    PlanMutexNotHeld,
    PlanState,
    ReviewAggregation,
    ReviewsRootMissing,
    TaskArtifacts,
    _normalize_plan,
    iteration_entry_written_by_orchestrator,
    plan_task_text_fp,
)
from orchestrator.mcp.verdict_tools import _submit_review_verdict
from orchestrator.review_payload import NoteDisposition, ReviewPayloadError


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
        assert artifacts.lane_root.exists()
        assert artifacts.artifact_path('metadata.json').exists()
        assert artifacts.artifact_path('reviews').is_dir()

    def test_creates_verdicts_directory(self, artifacts: TaskArtifacts):
        assert artifacts.artifact_path('verdicts').is_dir()

    def test_init_writes_no_gitignore(self, artifacts: TaskArtifacts):
        """W11-ι: init() no longer writes a .gitignore — .task-meta/ lives
        outside the worktree, so there's nothing for git to ignore.
        """
        assert not (artifacts.lane_root / '.gitignore').exists()
        assert artifacts.artifact_path('metadata.json').exists()
        assert artifacts.artifact_path('reviews').is_dir()

    def test_metadata_contents(self, artifacts: TaskArtifacts):
        metadata = json.loads(artifacts.artifact_path('metadata.json').read_text())
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


class TestReadCreatedAt:
    def test_returns_created_at_after_init(self, artifacts: TaskArtifacts):
        created_at = artifacts.read_created_at()
        assert created_at is not None
        # Must be an ISO-8601 (UTC) datetime string.
        datetime.fromisoformat(created_at)

    def test_returns_none_when_metadata_absent(self, worktree: Path):
        worktree.mkdir()
        ta = TaskArtifacts(worktree)
        assert ta.read_created_at() is None

    def test_returns_none_when_created_at_key_missing(self, worktree: Path):
        worktree.mkdir()
        ta = TaskArtifacts(worktree)
        ta.lane_root.mkdir(parents=True, exist_ok=True)
        ta.artifact_path('metadata.json').write_text(json.dumps({'task_id': 'x'}))
        assert ta.read_created_at() is None


class TestPlan:
    def test_write_and_read_plan(self, artifacts: TaskArtifacts):
        plan = {
            'task_id': 'task-1',
            'title': 'Test Task',
            'files': ['backend/app.py'],
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


class TestMarkStepCommitted:
    """TaskArtifacts.mark_step_committed(step_id, sha) — pre-satisfy a plan
    step at AUTHORING time (status→'done', commit=full sha, prepend a
    '[COMMITTED <sha[:12]>]' description tag). Pure persistence, NO git — the
    git corroborate-before-acting guard lives in plan_tools._sha_exists_on_branch.
    See task 3030 / PRD plans/architect-already-complete-exits.md Contract A1.
    """

    SHA = 'a' * 40

    @staticmethod
    def _plan() -> dict:
        return {
            'task_id': 'task-1',
            'files': ['backend/app.py'],
            'prerequisites': [
                {'id': 'pre-1', 'type': 'test', 'description': 'Setup fixture',
                 'status': 'pending', 'commit': None},
            ],
            'steps': [
                {'id': 'step-1', 'type': 'test', 'description': 'Write test',
                 'status': 'pending', 'commit': None},
                {'id': 'step-2', 'type': 'impl', 'description': 'Implement',
                 'status': 'pending', 'commit': None},
            ],
        }

    def test_marks_step_done_full_sha_in_commit_truncated_in_tag(
        self, artifacts: TaskArtifacts
    ):
        artifacts.write_plan(self._plan())

        assert artifacts.mark_step_committed('step-1', self.SHA) is True

        step = artifacts.read_plan()['steps'][0]
        assert step['id'] == 'step-1'
        assert step['status'] == 'done'
        # FULL sha persisted to commit; only sha[:12] appears in the tag.
        assert step['commit'] == self.SHA
        assert step['description'].startswith(f'[COMMITTED {self.SHA[:12]}]')
        assert step['description'].endswith('Write test')
        # The untruncated 40-char sha must NOT leak into the tag text.
        assert self.SHA not in step['description']

    def test_marks_prerequisite(self, artifacts: TaskArtifacts):
        artifacts.write_plan(self._plan())

        assert artifacts.mark_step_committed('pre-1', self.SHA) is True

        prereq = artifacts.read_plan()['prerequisites'][0]
        assert prereq['status'] == 'done'
        assert prereq['commit'] == self.SHA
        assert prereq['description'].startswith(f'[COMMITTED {self.SHA[:12]}]')

    def test_unknown_id_returns_false_and_mutates_nothing(
        self, artifacts: TaskArtifacts
    ):
        artifacts.write_plan(self._plan())
        # Stabilize any one-time normalization rewrite before capturing bytes.
        artifacts.read_plan()
        before = artifacts.artifact_path('plan.json').read_text()

        assert artifacts.mark_step_committed('nope', self.SHA) is False

        after = artifacts.artifact_path('plan.json').read_text()
        assert after == before

    def test_idempotent_per_step_and_sha(self, artifacts: TaskArtifacts):
        artifacts.write_plan(self._plan())

        assert artifacts.mark_step_committed('step-1', self.SHA) is True
        assert artifacts.mark_step_committed('step-1', self.SHA) is True

        step = artifacts.read_plan()['steps'][0]
        assert step['status'] == 'done'
        # No double-prepend on a repeated (step_id, sha) call.
        assert step['description'].count('[COMMITTED ') == 1

    def test_remark_different_sha_replaces_tag_no_stacking(
        self, artifacts: TaskArtifacts
    ):
        # A re-mark against a DIFFERENT sha (e.g. an amended/squashed commit
        # during re-planning) must REPLACE the stale [COMMITTED ...] tag, not
        # stack a second one in front of it.
        artifacts.write_plan(self._plan())
        sha_a = 'a' * 40
        sha_b = 'b' * 40

        assert artifacts.mark_step_committed('step-1', sha_a) is True
        assert artifacts.mark_step_committed('step-1', sha_b) is True

        step = artifacts.read_plan()['steps'][0]
        assert step['status'] == 'done'
        # commit tracks the LATEST sha; description carries exactly the current
        # tag — the stale sha_a tag is gone, not stacked.
        assert step['commit'] == sha_b
        assert step['description'].count('[COMMITTED ') == 1
        assert step['description'].startswith(f'[COMMITTED {sha_b[:12]}]')
        assert sha_a[:12] not in step['description']
        assert step['description'].endswith('Write test')

    def test_affects_pending_and_completed_filters(
        self, artifacts: TaskArtifacts
    ):
        artifacts.write_plan(self._plan())

        artifacts.mark_step_committed('step-1', self.SHA)

        pending_ids = [s['id'] for s in artifacts.get_pending_steps()]
        completed_ids = [s['id'] for s in artifacts.get_completed_steps()]
        assert 'step-1' not in pending_ids
        assert 'step-1' in completed_ids


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
        log_path = artifacts.artifact_path('iterations.jsonl')
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
        log_path = artifacts.artifact_path('iterations.jsonl')
        bad_lines = [
            r'not json at all',
            r'{"truncated": true',
            r'{bad \escape}',
        ]
        log_path.write_text('\n'.join(bad_lines) + '\n')

        entries, corrupted = artifacts.read_iteration_log()
        assert entries == []
        assert len(corrupted) == 3

    def test_clear_iteration_log_removes_file(self, artifacts: TaskArtifacts):
        """clear_iteration_log() removes iterations.jsonl entirely — a
        subsequent read sees a clean slate, not just an empty-but-present file.

        Layer B (task 2372): called on re-dispatch onto a new fork point so a
        prior dispatch's entries never masquerade as current-branch evidence.
        """
        artifacts.append_iteration_log({
            'iteration': 1, 'agent': 'implementer', 'summary': 'stale',
        })
        log_path = artifacts.artifact_path('iterations.jsonl')
        assert log_path.exists()

        artifacts.clear_iteration_log()

        assert not log_path.exists()
        assert artifacts.read_iteration_log() == ([], [])

    def test_clear_iteration_log_missing_ok(self, artifacts: TaskArtifacts):
        """clear_iteration_log() on a never-written log must not raise."""
        log_path = artifacts.artifact_path('iterations.jsonl')
        assert not log_path.exists()

        artifacts.clear_iteration_log()  # must not raise

        assert artifacts.read_iteration_log() == ([], [])


_AGENT_AUTHORED_SHAPES = [
    pytest.param(
        {
            'iteration': 2,
            'agent': 'implementer',
            'steps_completed': ['step-2'],
            'blocker': 'escalated',
            'deviations': [],
            'next': 'resume after escalation',
        },
        id='no-source-no-timestamp-no-committed-5588',
    ),
    pytest.param(
        {
            'agent': 'implementer',
            'steps_completed': ['step-2'],
            'committed': True,
            'source': 'agent',
            'timestamp': '2026-08-06T04:25:00',
        },
        id='source-agent-committed-true-3713',
    ),
    pytest.param(
        {
            'agent': 'implementer',
            'steps_completed': ['step-2'],
            'committed': True,
            'source': 'post-escalation-resume',
        },
        id='post-escalation-resume-3668',
    ),
    pytest.param(
        {'agent': 'implementer', 'steps_completed': ['step-2'], 'source': 'implementer'},
        id='source-implementer-3023',
    ),
    pytest.param(
        {'agent': 'claude-task-3088-implementer', 'steps_completed': ['step-2']},
        id='role-named-agent-no-source-3088',
    ),
    pytest.param(
        {'agent': 'implementer', 'steps_completed': ['step-2'], 'source': ['orchestrator']},
        id='list-valued-source',
    ),
]


class TestIterationLogProvenance:
    """Only entries written through append_iteration_log count as orchestrator-written.

    Agents also append to iterations.jsonl by hand from their own shell
    (task 5726); those lines must never pass the provenance predicate.
    """

    def test_entry_without_source_is_stamped_orchestrator(self, artifacts: TaskArtifacts):
        artifacts.append_iteration_log({'agent': 'implementer', 'steps_completed': ['step-1']})

        (entry,), _ = artifacts.read_iteration_log()
        assert entry['source'] == 'orchestrator'
        assert iteration_entry_written_by_orchestrator(entry) is True

    def test_amendment_source_is_preserved(self, artifacts: TaskArtifacts):
        artifacts.append_iteration_log({
            'agent': 'implementer', 'steps_completed': ['step-1'], 'source': 'amendment',
        })

        (entry,), _ = artifacts.read_iteration_log()
        assert entry['source'] == 'amendment'
        assert iteration_entry_written_by_orchestrator(entry) is True

    def test_unknown_source_is_refused_before_writing(self, artifacts: TaskArtifacts):
        with pytest.raises(ValueError, match='iteration-log source') as exc_info:
            artifacts.append_iteration_log({'agent': 'implementer', 'source': 'agent'})

        message = str(exc_info.value)
        assert "'agent'" in message
        assert 'orchestrator' in message
        assert 'amendment' in message
        assert artifacts.read_iteration_log() == ([], [])

    @pytest.mark.parametrize('agent_entry', _AGENT_AUTHORED_SHAPES)
    def test_hand_appended_agent_entries_are_not_orchestrator_written(
        self, artifacts: TaskArtifacts, agent_entry: dict,
    ):
        with open(artifacts.artifact_path('iterations.jsonl'), 'a') as log:
            log.write(json.dumps(agent_entry) + '\n')

        (entry,), _ = artifacts.read_iteration_log()
        assert iteration_entry_written_by_orchestrator(entry) is False

    @pytest.mark.parametrize('value', ['orchestrator', ['orchestrator'], None])
    def test_predicate_is_total_over_non_dict_values(self, value: object):
        assert iteration_entry_written_by_orchestrator(value) is False


VALID_PLAN_WITH_STEPS = {
    'task_id': 'task-1',
    'title': 'Test Task',
    'steps': [
        {'id': 'step-1', 'type': 'test', 'description': 'Write test', 'status': 'pending'},
    ],
}

CONFIRMED_AT = '2026-01-01T00:00:00+00:00'
CONFIRMED_PLAN_WITH_STEPS = {**VALID_PLAN_WITH_STEPS, '_finalized_at': CONFIRMED_AT}


def _artifacts_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage() for record in caplog.records
        if record.name == 'orchestrator.artifacts' and record.levelname == 'WARNING'
    ]


class TestPlanProvenance:
    def test_stamp_plan_provenance_adds_session_id_and_created_at(self, artifacts: TaskArtifacts):
        # Must use a valid plan with at least one step (empty steps raises ValueError)
        artifacts.write_plan(dict(CONFIRMED_PLAN_WITH_STEPS))
        artifacts.stamp_plan_provenance('session-abc123')
        updated = artifacts.read_plan()
        assert updated['_session_id'] == 'session-abc123'
        assert '_created_at' in updated

    def test_stamping_an_unconfirmed_plan_confirms_it_injected_and_warns(
        self, artifacts: TaskArtifacts, caplog: pytest.LogCaptureFixture,
    ):
        artifacts.write_plan(dict(VALID_PLAN_WITH_STEPS))
        with caplog.at_level('WARNING', logger='orchestrator.artifacts'):
            artifacts.stamp_plan_provenance('session-abc123')
        plan = read_model(artifacts)
        assert plan.state is PlanState.STAMPED
        assert plan.lifecycle.confirmation is PlanConfirmation.INJECTED
        assert artifacts.read_plan()['_confirmed_by'] == 'injected'
        (warning,) = _artifacts_warnings(caplog)
        assert 'task-1' in warning
        assert 'partial' in warning

    def test_stamping_a_confirmed_plan_keeps_its_confirmation_silently_and_returns_the_plan(
        self, artifacts: TaskArtifacts, caplog: pytest.LogCaptureFixture,
    ):
        artifacts.write_plan({**CONFIRMED_PLAN_WITH_STEPS, '_confirmed_by': 'architect'})
        with caplog.at_level('WARNING', logger='orchestrator.artifacts'):
            written = artifacts.stamp_plan_provenance('session-abc123')
        plan = read_model(artifacts)
        assert written == plan
        assert plan.state is PlanState.STAMPED
        assert plan.lifecycle.confirmed_at == CONFIRMED_AT
        assert plan.lifecycle.confirmed_by is PlanConfirmation.ARCHITECT
        assert _artifacts_warnings(caplog) == []

    def test_the_returned_plan_is_the_plan_on_disk_for_an_unversioned_plan(self, artifacts: TaskArtifacts):
        artifacts.artifact_path('plan.json').write_text(json.dumps(CONFIRMED_PLAN_WITH_STEPS))
        written = artifacts.stamp_plan_provenance('session-abc123')
        assert written.lifecycle.schema_version == PLAN_SCHEMA_VERSION
        assert written == read_model(artifacts)

    def test_stamp_plan_provenance_preserves_existing_finalized_marker(self, artifacts: TaskArtifacts):
        artifacts.write_plan(dict(CONFIRMED_PLAN_WITH_STEPS))
        artifacts.stamp_plan_provenance('session-abc123')
        assert artifacts.read_plan()['_finalized_at'] == CONFIRMED_AT

    def test_stamp_plan_provenance_preserves_existing_plan_data(self, artifacts: TaskArtifacts):
        plan = {
            'task_id': 'task-1',
            'analysis': 'Some analysis',
            'steps': [{'id': 'step-1', 'status': 'pending'}],
            '_finalized_at': CONFIRMED_AT,
        }
        artifacts.write_plan(plan)
        artifacts.stamp_plan_provenance('session-abc123')
        updated = artifacts.read_plan()
        assert updated['task_id'] == 'task-1'
        assert updated['analysis'] == 'Some analysis'
        assert len(updated['steps']) == 1

    def test_stamp_plan_provenance_writes_the_task_text_fingerprint(
        self, artifacts: TaskArtifacts,
    ):
        artifacts.write_plan(dict(CONFIRMED_PLAN_WITH_STEPS))
        artifacts.stamp_plan_provenance('session-abc123', task_text_fp='tfp:new')
        updated = artifacts.read_plan()
        assert updated[PLAN_TASK_TEXT_FP_KEY] == 'tfp:new'
        assert updated['_session_id'] == 'session-abc123'
        assert '_created_at' in updated

    def test_stamp_plan_provenance_fingerprint_overwrites_a_stale_one(
        self, artifacts: TaskArtifacts,
    ):
        artifacts.write_plan({**CONFIRMED_PLAN_WITH_STEPS, PLAN_TASK_TEXT_FP_KEY: 'tfp:old'})
        artifacts.stamp_plan_provenance('session-abc123', task_text_fp='tfp:new')
        assert artifacts.read_plan()[PLAN_TASK_TEXT_FP_KEY] == 'tfp:new'

    def test_stamp_plan_provenance_without_fingerprint_keeps_the_existing_one(
        self, artifacts: TaskArtifacts,
    ):
        artifacts.write_plan(dict(CONFIRMED_PLAN_WITH_STEPS))
        artifacts.stamp_plan_provenance('planner', task_text_fp='tfp:authored')
        artifacts.stamp_plan_provenance('implementer-iteration')
        updated = artifacts.read_plan()
        assert updated[PLAN_TASK_TEXT_FP_KEY] == 'tfp:authored'
        assert updated['_session_id'] == 'implementer-iteration'

    def test_stamp_plan_provenance_without_fingerprint_adds_none(
        self, artifacts: TaskArtifacts,
    ):
        artifacts.write_plan(dict(CONFIRMED_PLAN_WITH_STEPS))
        artifacts.stamp_plan_provenance('session-abc123')
        assert PLAN_TASK_TEXT_FP_KEY not in artifacts.read_plan()

    def test_plan_task_text_fp_reads_a_stamped_fingerprint(self):
        assert plan_task_text_fp({PLAN_TASK_TEXT_FP_KEY: 'tfp:x'}) == 'tfp:x'

    @pytest.mark.parametrize('plan', [
        {}, {PLAN_TASK_TEXT_FP_KEY: ''}, {PLAN_TASK_TEXT_FP_KEY: None}, {PLAN_TASK_TEXT_FP_KEY: 7},
    ], ids=['absent', 'empty', 'none', 'int'])
    def test_plan_task_text_fp_is_none_for_an_absent_or_malformed_stamp(self, plan: dict):
        assert plan_task_text_fp(plan) is None

    def test_validate_plan_owner_true_for_matching_session(self, artifacts: TaskArtifacts):
        artifacts.write_plan(dict(CONFIRMED_PLAN_WITH_STEPS))
        artifacts.stamp_plan_provenance('session-abc123')
        assert artifacts.validate_plan_owner('session-abc123') is True

    def test_validate_plan_owner_false_for_mismatched_session(self, artifacts: TaskArtifacts):
        artifacts.write_plan(dict(CONFIRMED_PLAN_WITH_STEPS))
        artifacts.stamp_plan_provenance('session-abc123')
        assert artifacts.validate_plan_owner('session-different') is False

    def test_validate_plan_owner_false_when_no_provenance(self, artifacts: TaskArtifacts):
        artifacts.write_plan(dict(CONFIRMED_PLAN_WITH_STEPS))
        # Not stamped — no _session_id in plan
        assert artifacts.validate_plan_owner('session-abc123') is False

    def test_validate_plan_owner_true_for_revalidating_session(
        self, artifacts: TaskArtifacts,
    ):
        """Lever B's bump_revalidation_stamp preserves the original
        ``_session_id`` for audit but transfers ownership to the
        revalidating session via ``_revalidated_by_session``.  Both
        sessions must be accepted as valid owners; a third party still
        is not.
        """
        artifacts.write_plan(dict(CONFIRMED_PLAN_WITH_STEPS))
        artifacts.stamp_plan_provenance('orig-planner')
        artifacts.bump_revalidation_stamp('reval-runner')

        plan = artifacts.read_plan()
        # Sanity: original _session_id is intact, revalidation field set.
        assert plan['_session_id'] == 'orig-planner'
        assert plan['_revalidated_by_session'] == 'reval-runner'

        assert artifacts.validate_plan_owner('orig-planner') is True
        assert artifacts.validate_plan_owner('reval-runner') is True
        assert artifacts.validate_plan_owner('foreign-session') is False

    def test_stamp_plan_provenance_raises_on_missing_plan(self, artifacts: TaskArtifacts):
        """stamp_plan_provenance() must raise ValueError if plan.json does not exist."""
        # Ensure plan.json does not exist
        plan_path = artifacts.artifact_path('plan.json')
        assert not plan_path.exists()

        with pytest.raises(ValueError, match='valid plan'):
            artifacts.stamp_plan_provenance('session-abc123')

    def test_stamp_plan_provenance_raises_on_empty_plan(self, artifacts: TaskArtifacts):
        """stamp_plan_provenance() must raise ValueError if plan.json has no steps key."""
        # Write a plan with no 'steps' key at all
        artifacts.artifact_path('plan.json').write_text('{}')

        with pytest.raises(ValueError, match='valid plan'):
            artifacts.stamp_plan_provenance('session-abc123')

    def test_stamp_plan_provenance_raises_on_provenance_only_stub(self, artifacts: TaskArtifacts):
        """stamp_plan_provenance() must raise ValueError on a provenance-only stub.

        A stub with only _session_id/_created_at (no steps) passes bool() checks but
        has no actual plan content — stamping it silently would hide data loss.
        """
        stub = {'_session_id': 'old-session', '_created_at': '2026-01-01T00:00:00+00:00'}
        artifacts.artifact_path('plan.json').write_text(
            __import__('json').dumps(stub)
        )

        with pytest.raises(ValueError, match='valid plan'):
            artifacts.stamp_plan_provenance('session-abc123')

    def test_validate_plan_owner_returns_false_on_corrupt_plan_json(
        self, artifacts: TaskArtifacts
    ):
        """Corrupt (non-JSON) plan.json must return False, not raise JSONDecodeError."""
        plan_path = artifacts.artifact_path('plan.json')
        plan_path.write_text('this is not valid json }{}{')

        # Must return False without raising any exception
        result = artifacts.validate_plan_owner('session-abc123')
        assert result is False

    def test_validate_plan_owner_returns_false_on_unreadable_plan(
        self, artifacts: TaskArtifacts
    ):
        """Unreadable plan.json (chmod 000) must return False, not raise OSError."""
        plan_path = artifacts.artifact_path('plan.json')
        plan_path.write_text('{"_session_id": "session-abc123"}')
        plan_path.chmod(0o000)
        try:
            result = artifacts.validate_plan_owner('session-abc123')
            assert result is False
        finally:
            # Restore permissions so cleanup works
            plan_path.chmod(0o644)


class TestBumpRevalidationStamp:
    """Tests for ``bump_revalidation_stamp`` — Lever B's confirm_plan-equivalent."""

    def test_stamps_revalidated_at_and_session(self, artifacts: TaskArtifacts):
        artifacts.write_plan(dict(CONFIRMED_PLAN_WITH_STEPS))
        artifacts.bump_revalidation_stamp('sess-1')
        updated = artifacts.read_plan()
        assert '_revalidated_at' in updated
        assert updated['_revalidated_by_session'] == 'sess-1'

    def test_advances_the_plan_base_not_the_dispatch_base(self, worktree: Path):
        worktree.mkdir(parents=True, exist_ok=True)
        ta = TaskArtifacts(worktree)
        ta.init('task-1', 'T', 'd', base_commit='originalsha')
        ta.write_plan(dict(CONFIRMED_PLAN_WITH_STEPS))
        ta.bump_revalidation_stamp('sess-1', base_commit='newsha456')
        assert ta.read_plan()[PLAN_BASE_COMMIT_KEY] == 'newsha456'
        assert ta.read_base_commit() == 'originalsha'

    def test_leaves_base_commit_when_none(self, worktree: Path):
        worktree.mkdir(parents=True, exist_ok=True)
        ta = TaskArtifacts(worktree)
        ta.init('task-1', 'T', 'd', base_commit='originalsha')
        ta.write_plan(dict(CONFIRMED_PLAN_WITH_STEPS))
        ta.stamp_plan_provenance('s0', base_commit='planbase')
        ta.bump_revalidation_stamp('sess-1')  # base_commit=None default
        assert ta.read_base_commit() == 'originalsha'
        assert ta.read_plan()[PLAN_BASE_COMMIT_KEY] == 'planbase'

    def test_raises_on_missing_plan(self, artifacts: TaskArtifacts):
        with pytest.raises(ValueError, match='valid plan'):
            artifacts.bump_revalidation_stamp('sess-1')

    def test_raises_on_empty_steps(self, artifacts: TaskArtifacts):
        artifacts.artifact_path('plan.json').write_text('{"steps": []}')
        with pytest.raises(ValueError, match='valid plan'):
            artifacts.bump_revalidation_stamp('sess-1')

    def test_a_stamped_plan_stays_stamped(self, artifacts: TaskArtifacts):
        artifacts.write_plan(dict(CONFIRMED_PLAN_WITH_STEPS))
        artifacts.stamp_plan_provenance('planner')
        before = read_model(artifacts)
        artifacts.bump_revalidation_stamp('reval', base_commit='newbase')
        after = read_model(artifacts)
        assert after.state is PlanState.STAMPED
        assert after.lifecycle.confirmed_at == before.lifecycle.confirmed_at
        assert after.lifecycle.session_id == 'planner'


class TestSetPlanFiles:
    """Tests for ``set_plan_files`` — the orchestrator-side plan.files widen
    used by the resume-path scope-grant consumer (task 2505)."""

    def test_updates_files_preserves_other_fields_and_owner(self, artifacts: TaskArtifacts):
        plan = {
            'task_id': 'task-1',
            'title': 'Test Task',
            'files': ['a.py'],
            'analysis': 'Some analysis',
            'steps': [
                {'id': 'step-1', 'type': 'test', 'description': 'Write test', 'status': 'pending', 'commit': None},
            ],
            'design_decisions': [{'decision': 'Use X', 'rationale': 'Because Y'}],
            '_finalized_at': CONFIRMED_AT,
        }
        artifacts.write_plan(plan)
        artifacts.stamp_plan_provenance('S')

        artifacts.set_plan_files(['a.py', 'b.py'], 'S')

        updated = artifacts.read_plan()
        assert updated['files'] == ['a.py', 'b.py']
        assert updated['analysis'] == 'Some analysis'
        assert updated['design_decisions'] == [{'decision': 'Use X', 'rationale': 'Because Y'}]
        assert len(updated['steps']) == 1
        assert updated['steps'][0]['id'] == 'step-1'
        assert artifacts.validate_plan_owner('S') is True

    def test_preserves_original_session_id_when_owner_via_revalidation(
        self, artifacts: TaskArtifacts,
    ):
        plan = dict(CONFIRMED_PLAN_WITH_STEPS)
        plan['files'] = ['a.py']
        artifacts.write_plan(plan)
        artifacts.stamp_plan_provenance('orig-planner')
        artifacts.bump_revalidation_stamp('reval-runner')

        artifacts.set_plan_files(['a.py', 'b.py'], 'reval-runner')

        updated = artifacts.read_plan()
        assert updated['files'] == ['a.py', 'b.py']
        assert updated['_session_id'] == 'orig-planner'
        assert updated['_revalidated_by_session'] == 'reval-runner'
        assert artifacts.validate_plan_owner('reval-runner') is True

    def test_a_stamped_plan_stays_stamped(self, artifacts: TaskArtifacts):
        artifacts.write_plan(dict(CONFIRMED_PLAN_WITH_STEPS))
        artifacts.stamp_plan_provenance('S')
        before = read_model(artifacts)
        artifacts.set_plan_files(['a.py'], 'S')
        after = read_model(artifacts)
        assert after.state is PlanState.STAMPED
        assert after.lifecycle == before.lifecycle

    def test_an_unowned_confirmed_plan_passes_to_the_caller_with_created_at_backfilled(
        self, artifacts: TaskArtifacts,
    ):
        artifacts.write_plan(dict(CONFIRMED_PLAN_WITH_STEPS))
        artifacts.set_plan_files(['a.py'], 'scope')
        after = read_model(artifacts)
        assert after.state is PlanState.STAMPED
        assert after.lifecycle.session_id == 'scope'
        assert after.lifecycle.created_at is not None
        assert after.lifecycle.confirmed_at == CONFIRMED_AT

    def test_an_ownership_transfer_does_not_clobber_an_existing_created_at(self, artifacts: TaskArtifacts):
        artifacts.write_plan({**CONFIRMED_PLAN_WITH_STEPS, '_created_at': CONFIRMED_AT})
        artifacts.set_plan_files(['a.py'], 'scope')
        after = read_model(artifacts)
        assert after.lifecycle.session_id == 'scope'
        assert after.lifecycle.created_at == CONFIRMED_AT


class TestStalePlanLock:
    """Tests for stale plan.lock detection and cleanup."""

    def _create_lock(self, artifacts: TaskArtifacts, session_id: str, locked_at: str):
        lock_path = artifacts.artifact_path('plan.lock')
        lock_path.write_text(json.dumps({
            'session_id': session_id,
            'locked_at': locked_at,
        }))

    def test_self_lock_cleared(self, artifacts: TaskArtifacts):
        """Lock from a prior session of the same task (matching task_id prefix) is cleared."""
        self._create_lock(artifacts, 'task-1-abcd1234', '2026-04-03T12:00:00+00:00')
        result = artifacts.clear_stale_plan_lock('task-1')
        assert result is True
        assert not artifacts.artifact_path('plan.lock').exists()

    def test_recent_foreign_lock_preserved(self, artifacts: TaskArtifacts):
        """Lock from a different task with recent timestamp is NOT cleared."""
        from datetime import UTC, datetime
        recent = datetime.now(UTC).isoformat()
        self._create_lock(artifacts, 'task-99-ffff0000', recent)
        result = artifacts.clear_stale_plan_lock('task-1')
        assert result is False
        assert artifacts.artifact_path('plan.lock').exists()

    def test_old_foreign_lock_cleared(self, artifacts: TaskArtifacts):
        """Lock from a different task with old timestamp IS cleared."""
        self._create_lock(artifacts, 'task-99-ffff0000', '2025-01-01T00:00:00+00:00')
        result = artifacts.clear_stale_plan_lock('task-1', stale_threshold_secs=600.0)
        assert result is True
        assert not artifacts.artifact_path('plan.lock').exists()

    def test_no_lock_returns_false(self, artifacts: TaskArtifacts):
        """No lock file → returns False."""
        result = artifacts.clear_stale_plan_lock('task-1')
        assert result is False

    def test_unparseable_timestamp_cleared(self, artifacts: TaskArtifacts):
        """Lock with garbage timestamp is treated as stale and cleared."""
        self._create_lock(artifacts, 'task-99-ffff0000', 'not-a-timestamp')
        result = artifacts.clear_stale_plan_lock('task-1')
        assert result is True
        assert not artifacts.artifact_path('plan.lock').exists()


class TestPlanLock:
    def test_is_plan_locked_false_initially(self, artifacts: TaskArtifacts):
        assert artifacts.is_plan_locked() is False

    def test_lock_plan_creates_file_returns_true(self, artifacts: TaskArtifacts):
        result = artifacts.lock_plan('session-abc123')
        assert result is True
        assert artifacts.artifact_path('plan.lock').exists()

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

    def test_lock_plan_writes_owner_pid(self, artifacts: TaskArtifacts):
        artifacts.lock_plan('session-abc123')
        lock_data = artifacts.read_plan_lock()
        assert lock_data is not None
        assert lock_data['session_id'] == 'session-abc123'
        assert 'locked_at' in lock_data
        assert lock_data['owner_pid'] == os.getpid()

    def test_lock_plan_raises_when_owner_pid_invalid(
        self, artifacts: TaskArtifacts, monkeypatch
    ):
        """lock_plan must raise ValueError and leave no plan.lock if os.getpid() returns non-int."""
        monkeypatch.setattr('orchestrator.artifacts.os.getpid', lambda: None)
        lock_path = artifacts.artifact_path('plan.lock')

        with pytest.raises(ValueError, match=r'owner_pid.*non-null int'):
            artifacts.lock_plan('session-abc')

        # No partial plan.lock should be left behind
        assert not lock_path.exists(), 'plan.lock must be cleaned up after failed lock_plan'

        # A subsequent call (with a real pid) must succeed — no persistent poison-pill
        monkeypatch.undo()
        result = artifacts.lock_plan('session-abc')
        assert result is True
        assert lock_path.exists()


class TestPlanLockRunId:
    """``lock_plan`` persists the process-level run id (task 3563).

    plan.lock historically carried only ``session_id``/``locked_at``/
    ``owner_pid``, so ``TaskGroundTruth``'s PLAN_LOCK claimant source had no
    run_id to compose a full ``compose_claimant_run_id`` identity from.
    Recording it here makes the lock-derived identity byte-identical to the
    DB claimant stamp that the SAME incarnation writes.

    The key is written ONLY when the run id is known and non-blank, so an
    ABSENT ``run_id`` unambiguously means "unknown" — covering both legacy
    locks already on disk and harness-less (test/eval) workflows.  It must
    never be written as ``''``, which the resolver could not distinguish from
    a genuinely empty run_id component.
    """

    def test_lock_plan_persists_run_id_alongside_existing_keys(
        self, artifacts: TaskArtifacts
    ):
        assert artifacts.lock_plan('sess-x', run_id='run-abc') is True
        lock_data = artifacts.read_plan_lock()
        assert lock_data is not None
        assert lock_data['run_id'] == 'run-abc'
        # The pre-existing keys are untouched.
        assert lock_data['session_id'] == 'sess-x'
        assert lock_data['owner_pid'] == os.getpid()
        assert 'locked_at' in lock_data

    def test_legacy_positional_call_writes_no_run_id_key(self, artifacts: TaskArtifacts):
        """The un-widened call site must leave the key ABSENT, not empty."""
        assert artifacts.lock_plan('sess-x') is True
        lock_data = artifacts.read_plan_lock()
        assert lock_data is not None
        assert 'run_id' not in lock_data

    @pytest.mark.parametrize(
        'run_id',
        [None, '', '   ', '\t\n'],
        ids=['none', 'empty', 'spaces', 'whitespace'],
    )
    def test_unknown_or_blank_run_id_omits_the_key_entirely(
        self, artifacts: TaskArtifacts, run_id
    ):
        assert artifacts.lock_plan('sess-x', run_id=run_id) is True
        lock_data = artifacts.read_plan_lock()
        assert lock_data is not None
        assert 'run_id' not in lock_data, (
            'a blank/unknown run_id must be omitted, never written as a '
            'falsy value the resolver would mistake for a known component'
        )

    def test_run_id_does_not_change_acquire_semantics(self, artifacts: TaskArtifacts):
        """The return contract is untouched: True on acquire, False when held."""
        first = artifacts.lock_plan('sess-x', run_id='run-abc')
        second = artifacts.lock_plan('sess-other', run_id='run-other')
        assert first is True
        assert second is False
        # The loser must not have clobbered the winner's record.
        lock_data = artifacts.read_plan_lock()
        assert lock_data is not None
        assert lock_data['session_id'] == 'sess-x'
        assert lock_data['run_id'] == 'run-abc'


class TestAgentSession:
    """Sidecar marker for in-flight agent subprocesses (crash-recovery resume).

    Sidecar schema v2 (task 2771): the payload is a TYPED ``AgentSession``
    carrying the durable ``task_id`` lane→task binding, a ``resume_count``,
    and ``schema_version: 2``.  ``read_agent_session`` returns the typed model
    (attribute access, not dict subscript) and tolerates a pre-deploy v1
    sidecar (missing v2 keys → legacy defaults).
    """

    def test_write_read_roundtrip(self, artifacts: TaskArtifacts):
        artifacts.write_agent_session(
            'sess-abc', 'architect', '2026-05-12T10:00:00+00:00',
            task_id='t-42',
        )
        data = artifacts.read_agent_session()
        assert data is not None
        # Typed AgentSession — attribute access, not a raw dict.
        assert not isinstance(data, dict)
        assert data.session_id == 'sess-abc'
        assert data.role == 'architect'
        assert data.started_at == '2026-05-12T10:00:00+00:00'
        assert data.owner_pid == os.getpid()
        # v2 fields exposed as attributes.
        assert data.task_id == 't-42'
        assert data.resume_count == 0
        assert data.schema_version == 2

    def test_read_returns_none_when_missing(self, artifacts: TaskArtifacts):
        assert artifacts.read_agent_session() is None

    def test_clear_removes_file(self, artifacts: TaskArtifacts):
        artifacts.write_agent_session(
            'sess-x', 'implementer', 'now', task_id='t-1',
        )
        assert artifacts.artifact_path('agent_session.json').exists()
        artifacts.clear_agent_session()
        assert not artifacts.artifact_path('agent_session.json').exists()

    def test_clear_is_idempotent(self, artifacts: TaskArtifacts):
        # No file present
        artifacts.clear_agent_session()  # must not raise
        artifacts.clear_agent_session()  # still idempotent

    def test_write_persists_resume_count(self, artifacts: TaskArtifacts):
        artifacts.write_agent_session(
            'sess-r', 'implementer', 'now', task_id='t-7', resume_count=3,
        )
        data = artifacts.read_agent_session()
        assert data is not None
        assert data.resume_count == 3
        assert data.schema_version == 2

    def test_write_emits_v2_on_disk(self, artifacts: TaskArtifacts):
        """The on-disk JSON always carries schema_version:2 and the two new
        keys — the delivered-check contract shape for this task."""
        artifacts.write_agent_session(
            'sess-d', 'architect', 'now', task_id='t-9', resume_count=1,
        )
        raw = json.loads(artifacts.artifact_path('agent_session.json').read_text())
        assert raw['schema_version'] == 2
        assert 'task_id' in raw
        assert 'resume_count' in raw

    def test_read_v1_sidecar_legacy_semantics(self, artifacts: TaskArtifacts):
        """B11: a pre-deploy v1 sidecar (no v2 keys) reads with legacy
        defaults — task_id None, resume_count 0, schema_version 1 — while the
        four v1 fields are preserved verbatim."""
        v1 = {
            'session_id': 'sess-v1',
            'role': 'architect',
            'started_at': '2026-05-01T00:00:00+00:00',
            'owner_pid': 12345,
        }
        artifacts.artifact_path('agent_session.json').write_text(json.dumps(v1))
        data = artifacts.read_agent_session()
        assert data is not None
        # Legacy defaults for the absent v2 keys.
        assert data.task_id is None
        assert data.resume_count == 0
        assert data.schema_version == 1
        # v1 fields preserved.
        assert data.session_id == 'sess-v1'
        assert data.role == 'architect'
        assert data.started_at == '2026-05-01T00:00:00+00:00'
        assert data.owner_pid == 12345

    def test_read_returns_none_on_unparseable_numeric_field(
        self, artifacts: TaskArtifacts
    ):
        """Fail-safe contract holds when a well-formed JSON object carries a
        non-numeric resume_count/schema_version: from_mapping's int() coercion
        raises ValueError/TypeError, which must map to None (not propagate)
        like any other corruption."""
        # Non-numeric string resume_count → int('abc') raises ValueError.
        bad_str = {
            'session_id': 'sess-bad',
            'role': 'architect',
            'started_at': 'now',
            'owner_pid': 1,
            'resume_count': 'abc',
            'schema_version': 2,
        }
        artifacts.artifact_path('agent_session.json').write_text(json.dumps(bad_str))
        assert artifacts.read_agent_session() is None
        # Null schema_version → int(None) raises TypeError.
        bad_null = {
            'session_id': 'sess-bad',
            'role': 'architect',
            'started_at': 'now',
            'owner_pid': 1,
            'resume_count': 0,
            'schema_version': None,
        }
        artifacts.artifact_path('agent_session.json').write_text(json.dumps(bad_null))
        assert artifacts.read_agent_session() is None


class TestEmittedStepEscalations:
    """Durable cross-restart dedup for orphaned done-step escalations (task 2764).

    ``_reconcile_done_step_commits`` files a non-blocking info escalation for a
    done-step commit orphaned by a rebase that can be neither WIP-filename-matched
    nor patch-id/unique-subject-remapped.  The in-memory flood guard is
    process-local, so an orchestrator restart re-files every prior pair.  These
    emitted ``(step_id, stale_commit)`` keys are persisted to
    ``reconcile_state.json`` in the meta-root and hydrated on the next run so a
    restarted workflow files at most genuinely-new pairs.
    """

    def test_read_returns_empty_set_when_missing(self, artifacts: TaskArtifacts):
        assert artifacts.read_emitted_step_escalations() == set()

    def test_append_then_read_roundtrip(self, artifacts: TaskArtifacts):
        artifacts.append_emitted_step_escalation('step-1', 'abc123')
        assert artifacts.read_emitted_step_escalations() == {('step-1', 'abc123')}

    def test_append_same_pair_is_idempotent(self, artifacts: TaskArtifacts):
        artifacts.append_emitted_step_escalation('step-1', 'abc123')
        artifacts.append_emitted_step_escalation('step-1', 'abc123')
        assert artifacts.read_emitted_step_escalations() == {('step-1', 'abc123')}

    def test_append_same_pair_skips_redundant_write(self, artifacts: TaskArtifacts):
        # Efficiency (task 2764 amend): a redundant append for an
        # already-persisted pair short-circuits to a pure read and skips the
        # disk rewrite entirely (dedup-before-write). Deterministic spy on
        # ``_write_json`` rather than mtime (coarse fs resolution is a weak
        # regression guard).
        artifacts.append_emitted_step_escalation('step-1', 'abc123')
        with patch.object(
            artifacts, '_write_json', wraps=artifacts._write_json
        ) as spy:
            artifacts.append_emitted_step_escalation('step-1', 'abc123')
        spy.assert_not_called()
        # Content is unchanged and still readable after the skipped write.
        assert artifacts.read_emitted_step_escalations() == {('step-1', 'abc123')}

    def test_append_distinct_pairs_accumulate(self, artifacts: TaskArtifacts):
        artifacts.append_emitted_step_escalation('step-1', 'abc123')
        artifacts.append_emitted_step_escalation('step-2', 'def456')
        assert artifacts.read_emitted_step_escalations() == {
            ('step-1', 'abc123'),
            ('step-2', 'def456'),
        }

    def test_survives_restart_new_instance_same_root(
        self, artifacts: TaskArtifacts, worktree: Path
    ):
        """RESTART SURVIVAL: a fresh TaskArtifacts on the same root (as a
        restarted orchestrator constructs a new workflow + artifacts) reads back
        the pair persisted by the prior instance."""
        artifacts.append_emitted_step_escalation('step-1', 'abc123')
        # A brand-new TaskArtifacts on the SAME worktree resolves to the SAME
        # meta-root, so the durable store survives the "restart".
        restarted = TaskArtifacts(worktree)
        assert restarted.read_emitted_step_escalations() == {('step-1', 'abc123')}

    @pytest.mark.parametrize(
        'field_value',
        [
            pytest.param(None, id='null'),
            pytest.param(3, id='int'),
            pytest.param(True, id='bool'),
            pytest.param({'x': 1}, id='dict'),
            pytest.param('ab', id='string'),
        ],
    )
    def test_non_list_field_reads_as_empty_set_and_warns(
        self, artifacts: TaskArtifacts, caplog, field_value
    ):
        (artifacts.artifact_path('reconcile_state.json')).write_text(
            json.dumps({'emitted_done_step_escalations': field_value})
        )

        with caplog.at_level('WARNING'):
            assert artifacts.read_emitted_step_escalations() == set()

        assert any(
            record.name == 'orchestrator.artifacts'
            and record.levelname == 'WARNING'
            and 'reconcile_state.json' in record.getMessage()
            for record in caplog.records
        )

    def test_malformed_entries_are_skipped_with_a_warning(
        self, artifacts: TaskArtifacts, caplog
    ):
        (artifacts.artifact_path('reconcile_state.json')).write_text(
            json.dumps({
                'emitted_done_step_escalations': [
                    [1, 2],
                    ['only-one'],
                    ['step-1', 'abc123'],
                ]
            })
        )

        with caplog.at_level('WARNING'):
            assert artifacts.read_emitted_step_escalations() == {
                ('step-1', 'abc123')
            }

        warnings = [
            record.getMessage()
            for record in caplog.records
            if record.name == 'orchestrator.artifacts'
            and record.levelname == 'WARNING'
            and 'reconcile_state.json' in record.getMessage()
        ]
        assert len(warnings) == 1
        assert '2 malformed' in warnings[0]

    def test_well_formed_store_reads_without_warning(
        self, artifacts: TaskArtifacts, caplog
    ):
        artifacts.append_emitted_step_escalation('step-1', 'abc123')
        artifacts.append_emitted_step_escalation('step-1', 'abc123')

        with caplog.at_level('WARNING'):
            assert artifacts.read_emitted_step_escalations() == {
                ('step-1', 'abc123')
            }

        assert not [
            record
            for record in caplog.records
            if record.name == 'orchestrator.artifacts'
        ]

    def test_append_over_malformed_field_rewrites_valid_store(
        self, artifacts: TaskArtifacts
    ):
        store = artifacts.artifact_path('reconcile_state.json')
        store.write_text(json.dumps({'emitted_done_step_escalations': None}))

        artifacts.append_emitted_step_escalation('step-1', 'abc123')

        assert artifacts.read_emitted_step_escalations() == {('step-1', 'abc123')}
        assert json.loads(store.read_text()) == {
            'emitted_done_step_escalations': [['step-1', 'abc123']]
        }


class TestVerdicts:
    """Verdict artifacts written by the verdict-tools MCP server (task 2481)."""

    def test_write_read_roundtrip(self, artifacts: TaskArtifacts):
        envelope = {
            'role': 'judge',
            'schema_version': 1,
            'session_id': 's',
            'emitted_at': 't',
            'verdict': {'complete': True},
        }
        artifacts.write_verdict('judge', envelope)
        assert (artifacts.artifact_path('verdicts') / 'judge.json').exists()
        assert artifacts.read_verdict('judge') == envelope

    def test_read_returns_none_when_missing(self, artifacts: TaskArtifacts):
        assert artifacts.read_verdict('missing') is None

    def test_clear_removes_file(self, artifacts: TaskArtifacts):
        envelope = {
            'role': 'judge',
            'schema_version': 1,
            'session_id': 's',
            'emitted_at': 't',
            'verdict': {'complete': True},
        }
        artifacts.write_verdict('judge', envelope)
        artifacts.clear_verdict('judge')
        assert not (artifacts.artifact_path('verdicts') / 'judge.json').exists()
        assert artifacts.read_verdict('judge') is None

    def test_clear_is_idempotent(self, artifacts: TaskArtifacts):
        artifacts.clear_verdict('judge')  # must not raise
        artifacts.clear_verdict('judge')  # still idempotent

    @pytest.mark.parametrize('bad_role', ['../escape', 'a/b', '..', '.', '/etc/passwd', 'judge\n'])
    def test_write_verdict_rejects_unsafe_role(self, artifacts: TaskArtifacts, bad_role):
        with pytest.raises(ValueError, match='invalid verdict role'):
            artifacts.write_verdict(bad_role, {'role': bad_role})
        # Nothing escaped verdicts/ or landed anywhere under root.
        assert not any(artifacts.task_root.rglob('escape*'))
        assert not any(artifacts.task_root.rglob('passwd*'))

    @pytest.mark.parametrize('bad_role', ['../escape', 'a/b', '..', '.', '/etc/passwd', 'judge\n'])
    def test_read_verdict_rejects_unsafe_role(self, artifacts: TaskArtifacts, bad_role):
        with pytest.raises(ValueError, match='invalid verdict role'):
            artifacts.read_verdict(bad_role)

    @pytest.mark.parametrize('bad_role', ['../escape', 'a/b', '..', '.', '/etc/passwd', 'judge\n'])
    def test_clear_verdict_rejects_unsafe_role(self, artifacts: TaskArtifacts, bad_role):
        with pytest.raises(ValueError, match='invalid verdict role'):
            artifacts.clear_verdict(bad_role)


class TestJsonSidecarReadersFailSafe:
    """Every JSON-sidecar reader shares one fail-safe posture: a corrupt
    sidecar reads as that reader's fallback value and logs a warning."""

    @pytest.mark.parametrize(
        ('name', 'read', 'expected'),
        [
            pytest.param(
                'metadata.json',
                lambda ta: ta.read_created_at(),
                None,
                id='created_at',
            ),
            pytest.param(
                'review_state.json',
                lambda ta: ta.read_review_state(),
                {'amendment_rounds_total': 0, 'review_cycles_total': 0, 'verdicts': {}},
                id='review_state',
            ),
            pytest.param(
                'agent_session.json',
                lambda ta: ta.read_agent_session(),
                None,
                id='agent_session',
            ),
            pytest.param(
                'reconcile_state.json',
                lambda ta: ta.read_emitted_step_escalations(),
                set(),
                id='reconcile_state',
            ),
            pytest.param(
                'verdicts/judge.json',
                lambda ta: ta.read_verdict('judge'),
                None,
                id='verdict',
            ),
        ],
    )
    @pytest.mark.parametrize(
        'payload',
        [
            pytest.param(b'\xff\xfe{"x": 1}', id='undecodable'),
            pytest.param(b'{not valid json', id='malformed'),
            pytest.param(b'[1]', id='non_object'),
        ],
    )
    def test_corrupt_sidecar_reads_as_fail_safe_value(
        self, artifacts: TaskArtifacts, caplog, name, read, expected, payload
    ):
        artifacts.artifact_path(name).write_bytes(payload)

        with caplog.at_level('WARNING'):
            assert read(artifacts) == expected

        basename = Path(name).name
        assert any(
            record.name == 'orchestrator.artifacts'
            and record.levelname == 'WARNING'
            and basename in record.getMessage()
            for record in caplog.records
        )


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

    @staticmethod
    def _aggregation(suggestion_tags: list[str]) -> ReviewAggregation:
        return review_aggregation(
            [review_issue('blocker', 'blocking')],
            [review_issue(tag) for tag in suggestion_tags],
        )

    def test_format_for_escalation_inlines_every_suggestion(self):
        agg = self._aggregation(['alpha', 'beta'])
        text = agg.format_for_escalation()
        for suggestion in agg.suggestions:
            for key in ('location', 'category', 'description', 'suggested_fix'):
                assert suggestion[key] in text
        blocking_at = text.index('# Review Feedback — Blocking Issues')
        assert text.index('# Review Feedback — Suggestions') > blocking_at

    def test_format_for_escalation_begins_with_the_replan_rendering(self):
        agg = self._aggregation(['alpha', 'beta'])
        assert agg.format_for_escalation().startswith(agg.format_for_replan())

    def test_format_for_escalation_without_suggestions_equals_replan(self):
        agg = self._aggregation([])
        assert agg.format_for_escalation() == agg.format_for_replan()

    def test_format_for_replan_stays_blocking_only(self):
        agg = self._aggregation(['alpha', 'beta'])
        text = agg.format_for_replan()
        for suggestion in agg.suggestions:
            assert suggestion['description'] not in text

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

    def test_aggregate_reviews_rejects_an_off_contract_verdict(self, artifacts: TaskArtifacts):
        artifacts.write_review('reviewer1', {
            'reviewer': 'reviewer1',
            'verdict': 'NEEDS_CHANGES',
            'issues': [],
            'summary': 'not a contract verdict',
        })

        with pytest.raises(ReviewPayloadError, match='NEEDS_CHANGES'):
            artifacts.aggregate_reviews()

    def test_aggregate_reviews_keeps_errored_review_records(self, artifacts: TaskArtifacts):
        clean = {'reviewer': 'reviewer1', 'verdict': 'PASS', 'issues': [], 'summary': 'OK'}
        errored = {
            'reviewer': 'reviewer2',
            'verdict': 'ERROR',
            'issues': [],
            'summary': 'Reviewer emitted no/invalid verdict: ',
            'failure': {'kind': 'turn_cap', 'detail': 'agent hit max_turns', 'api_error_status': None},
        }
        artifacts.write_review('reviewer1', clean)
        artifacts.write_review('reviewer2', errored)

        agg = artifacts.aggregate_reviews()

        assert agg.errored_reviews == {'reviewer2': errored}
        assert agg.reviewer_errors == ['reviewer2']
        assert agg.reviews == {'reviewer1': clean}

    def test_errored_reviews_defaults_empty(self):
        agg = ReviewAggregation(
            has_blocking_issues=False, blocking_issues=[], suggestions=[], reviews={},
        )
        assert agg.errored_reviews == {}


_ADDRESSED_NOTE = {
    'escalation_id': 'esc-42-1', 'disposition': 'addressed', 'rationale': 'fixed in foo.py',
}
_WORK_NOTE = {
    'escalation_id': 'esc-42-2', 'disposition': 'work', 'rationale': 'needs its own task',
}
_TYPED_NOTES = (
    NoteDisposition('esc-42-1', ReviewerDisposition.ADDRESSED, 'fixed in foo.py'),
    NoteDisposition('esc-42-2', ReviewerDisposition.WORK, 'needs its own task'),
)


class TestReviewNoteDispositions:
    def test_mirrored_review_dispositions_reach_the_aggregation_in_order(
        self, artifacts: TaskArtifacts,
    ):
        artifacts.write_review('reviewer_comprehensive', {
            'reviewer': 'reviewer_comprehensive',
            'verdict': 'PASS',
            'issues': [],
            'summary': 'OK',
            'note_dispositions': [dict(_ADDRESSED_NOTE), dict(_WORK_NOTE)],
        })

        assert artifacts.aggregate_reviews().note_dispositions == _TYPED_NOTES

    def test_an_error_review_contributes_no_dispositions(self, artifacts: TaskArtifacts):
        artifacts.write_review('reviewer_comprehensive', {
            'reviewer': 'reviewer_comprehensive',
            'verdict': 'ERROR',
            'issues': [],
            'summary': 'Reviewer error',
            'note_dispositions': [dict(_ADDRESSED_NOTE)],
        })

        assert artifacts.aggregate_reviews().note_dispositions == ()

    def test_a_review_without_the_key_gives_none(self, artifacts: TaskArtifacts):
        artifacts.write_review('reviewer_comprehensive', {
            'reviewer': 'reviewer_comprehensive',
            'verdict': 'PASS',
            'issues': [],
            'summary': 'OK',
        })

        assert artifacts.aggregate_reviews().note_dispositions == ()

    def test_dispositions_survive_the_verdict_tool_to_mirror_round_trip(
        self, artifacts: TaskArtifacts,
    ):
        role = 'reviewer_comprehensive'
        result = _submit_review_verdict(
            artifacts, role, 'sid', role, 'PASS', [], 'OK',
            note_dispositions=[dict(_ADDRESSED_NOTE), dict(_WORK_NOTE)],
        )
        assert result['status'] == 'ok'
        envelope = artifacts.read_verdict(role)
        assert envelope is not None

        artifacts.write_review(role, envelope['verdict'])

        assert artifacts.aggregate_reviews().note_dispositions == _TYPED_NOTES

    def test_a_directly_built_aggregation_carries_no_dispositions(self):
        aggregation = ReviewAggregation(
            has_blocking_issues=False, blocking_issues=[], suggestions=[], reviews={},
        )

        assert aggregation.note_dispositions == ()


class TestBlockingDependency:
    """Round-trip and edge-case tests for the architect's
    ``.task/blocking_dependency.json`` artifact."""

    def test_read_returns_none_when_absent(self, artifacts: TaskArtifacts):
        assert artifacts.read_blocking_dependency() is None

    def test_round_trip(self, artifacts: TaskArtifacts):
        artifacts.write_blocking_dependency(
            depends_on_task_id='42',
            reason='task 50 references helpers.foo introduced by task 42',
            main_sha_at_report='abc123def456',
        )
        data = artifacts.read_blocking_dependency()
        assert data is not None
        assert data['depends_on_task_id'] == '42'
        assert data['reason'] == (
            'task 50 references helpers.foo introduced by task 42'
        )
        assert data['main_sha_at_report'] == 'abc123def456'
        # reported_at must be present and ISO-8601 parseable
        from datetime import datetime
        assert 'reported_at' in data
        datetime.fromisoformat(data['reported_at'])

    def test_write_atomic_overwrites_prior_artifact(
        self, artifacts: TaskArtifacts
    ):
        artifacts.write_blocking_dependency('1', 'first reason', 'sha1')
        artifacts.write_blocking_dependency('2', 'second reason', 'sha2')
        data = artifacts.read_blocking_dependency()
        assert data is not None
        assert data['depends_on_task_id'] == '2'
        assert data['reason'] == 'second reason'
        assert data['main_sha_at_report'] == 'sha2'

    def test_artifact_path_is_under_task_dir(self, artifacts: TaskArtifacts):
        artifacts.write_blocking_dependency('5', 'r', 'sha')
        path = artifacts.artifact_path('blocking_dependency.json')
        assert path.exists()
        # And contents must be JSON parseable.
        json.loads(path.read_text())


class TestAlreadyDone:
    """Round-trip and edge-case tests for the architect's
    ``.task/already_done.json`` artifact."""

    def test_read_returns_none_when_absent(self, artifacts: TaskArtifacts):
        assert artifacts.read_already_done() is None

    def test_round_trip(self, artifacts: TaskArtifacts):
        artifacts.write_already_done(
            commit='abc123def456',
            evidence='helpers.foo introduced by task 42 (commit abc123)',
        )
        data = artifacts.read_already_done()
        assert data is not None
        assert data['commit'] == 'abc123def456'
        assert data['evidence'] == 'helpers.foo introduced by task 42 (commit abc123)'
        from datetime import datetime
        datetime.fromisoformat(data['reported_at'])

    def test_clear_removes_artifact(self, artifacts: TaskArtifacts):
        artifacts.write_already_done('sha', 'e')
        artifacts.clear_already_done()
        assert artifacts.read_already_done() is None

    def test_clear_is_idempotent(self, artifacts: TaskArtifacts):
        # No raise even when the artifact doesn't exist.
        artifacts.clear_already_done()
        artifacts.clear_already_done()

    def test_write_overwrites_prior_artifact(self, artifacts: TaskArtifacts):
        artifacts.write_already_done('first-sha', 'first evidence')
        artifacts.write_already_done('second-sha', 'second evidence')
        data = artifacts.read_already_done()
        assert data is not None
        assert data['commit'] == 'second-sha'
        assert data['evidence'] == 'second evidence'


class TestReadyToMerge:
    """Round-trip and edge-case tests for the architect's
    ``.task/ready_to_merge.json`` artifact."""

    def test_read_returns_none_when_absent(self, artifacts: TaskArtifacts):
        assert artifacts.read_ready_to_merge() is None

    def test_round_trip(self, artifacts: TaskArtifacts):
        artifacts.write_ready_to_merge(
            commit='abc123def456',
            evidence='branch is a clean FF of main; verify PASSED; review PASS',
        )
        data = artifacts.read_ready_to_merge()
        assert data is not None
        assert data['commit'] == 'abc123def456'
        assert data['evidence'] == (
            'branch is a clean FF of main; verify PASSED; review PASS'
        )
        from datetime import datetime
        datetime.fromisoformat(data['reported_at'])

    def test_artifact_path_is_under_task_dir(self, artifacts: TaskArtifacts):
        artifacts.write_ready_to_merge('sha', 'e')
        path = artifacts.artifact_path('ready_to_merge.json')
        assert path.exists()
        json.loads(path.read_text())

    def test_clear_removes_artifact(self, artifacts: TaskArtifacts):
        artifacts.write_ready_to_merge('sha', 'e')
        artifacts.clear_ready_to_merge()
        assert artifacts.read_ready_to_merge() is None

    def test_clear_is_idempotent(self, artifacts: TaskArtifacts):
        # No raise even when the artifact doesn't exist.
        artifacts.clear_ready_to_merge()
        artifacts.clear_ready_to_merge()

    def test_write_overwrites_prior_artifact(self, artifacts: TaskArtifacts):
        artifacts.write_ready_to_merge('first-sha', 'first evidence')
        artifacts.write_ready_to_merge('second-sha', 'second evidence')
        data = artifacts.read_ready_to_merge()
        assert data is not None
        assert data['commit'] == 'second-sha'
        assert data['evidence'] == 'second evidence'


class TestUnactionableTask:
    """Round-trip and edge-case tests for the architect's
    ``.task/unactionable_task.json`` artifact."""

    def test_read_returns_none_when_absent(self, artifacts: TaskArtifacts):
        assert artifacts.read_unactionable_task() is None

    def test_round_trip(self, artifacts: TaskArtifacts):
        artifacts.write_unactionable_task(
            reason='spec contradicts already-merged refactor',
            evidence='task asks to add foo() but task 31 deleted it (commit a1b2c3)',
        )
        data = artifacts.read_unactionable_task()
        assert data is not None
        assert data['reason'] == 'spec contradicts already-merged refactor'
        assert 'task 31 deleted it' in data['evidence']
        from datetime import datetime
        datetime.fromisoformat(data['reported_at'])

    def test_clear_removes_artifact(self, artifacts: TaskArtifacts):
        artifacts.write_unactionable_task('r', 'e')
        artifacts.clear_unactionable_task()
        assert artifacts.read_unactionable_task() is None

    def test_clear_is_idempotent(self, artifacts: TaskArtifacts):
        artifacts.clear_unactionable_task()
        artifacts.clear_unactionable_task()

    def test_write_overwrites_prior_artifact(self, artifacts: TaskArtifacts):
        artifacts.write_unactionable_task('first', 'e1')
        artifacts.write_unactionable_task('second', 'e2')
        data = artifacts.read_unactionable_task()
        assert data is not None
        assert data['reason'] == 'second'
        assert data['evidence'] == 'e2'

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
        assert '[index 0]' in msg  # first offending index
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
        assert '[index 1]' in msg  # second item (index 1) is the string

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


class TestNormalizePlan:
    """Tests for _normalize_plan() — auto-fixing malformed architect output."""

    def test_tdd_steps_renamed(self):
        plan = {
            'task_id': '1',
            'tdd_steps': [
                {'id': 'step-1', 'description': 'Write test', 'status': 'pending'},
            ],
        }
        result, modified = _normalize_plan(plan)
        assert modified is True
        assert 'tdd_steps' not in result
        assert len(result['steps']) == 1
        assert result['steps'][0]['id'] == 'step-1'

    def test_tdd_plan_renamed(self):
        plan = {
            'task_id': '2',
            'tdd_plan': [
                {'id': 'step-1', 'description': 'Impl', 'status': 'pending'},
            ],
        }
        result, modified = _normalize_plan(plan)
        assert modified is True
        assert 'tdd_plan' not in result
        assert result['steps'][0]['id'] == 'step-1'

    def test_implementation_steps_renamed(self):
        plan = {
            'task_id': '3',
            'implementation_steps': [
                {'id': 'step-1', 'description': 'Do thing', 'status': 'pending'},
            ],
        }
        result, modified = _normalize_plan(plan)
        assert modified is True
        assert result['steps'][0]['id'] == 'step-1'

    def test_nested_phases_flattened(self):
        plan = {
            'task_id': '4',
            'tdd_implementation_plan': {
                'phase_0_verification': [
                    'Verify config is clean',
                    'Verify imports',
                ],
                'phase_1_red': {
                    'description': 'Write failing test for X',
                },
                'phase_2_green': [
                    {'id': 'impl-1', 'description': 'Implement X'},
                ],
            },
        }
        result, modified = _normalize_plan(plan)
        assert modified is True
        assert 'tdd_implementation_plan' not in result
        steps = result['steps']
        # 2 strings from phase_0, 1 dict from phase_1, 1 dict from phase_2
        assert len(steps) == 4
        # All items are dicts with id and status
        for s in steps:
            assert isinstance(s, dict)
            assert 'id' in s
            assert 'status' in s

    def test_string_prerequisites_left_as_is(self):
        """String prerequisites are NOT normalized (Rule 4 removed).

        validate_plan_prerequisites() now rejects them with a clear error
        instead of silently coercing.
        """
        plan = {
            'task_id': '5',
            'steps': [{'id': 'step-1', 'status': 'pending'}],
            'prerequisites': [
                'Confirm no ruff D-rules',
                'Confirm docstring is clean',
            ],
        }
        result, modified = _normalize_plan(plan)
        assert modified is False
        prereqs = result['prerequisites']
        assert len(prereqs) == 2
        assert prereqs[0] == 'Confirm no ruff D-rules'
        assert prereqs[1] == 'Confirm docstring is clean'

    def test_string_steps_wrapped(self):
        plan = {
            'task_id': '6',
            'steps': ['Write test', 'Implement feature'],
        }
        result, modified = _normalize_plan(plan)
        assert modified is True
        assert result['steps'][0]['description'] == 'Write test'
        assert result['steps'][0]['id'] == 'step-1'
        assert result['steps'][1]['description'] == 'Implement feature'

    def test_valid_plan_unchanged(self):
        plan = {
            'task_id': '7',
            'steps': [
                {'id': 'step-1', 'status': 'pending', 'commit': None},
            ],
            'prerequisites': [
                {'id': 'pre-1', 'status': 'done', 'commit': 'abc'},
            ],
        }
        result, modified = _normalize_plan(plan)
        assert modified is False
        assert result == plan

    def test_string_prerequisites_not_normalized(self):
        """String prerequisites are NOT wrapped by _normalize_plan (Rule 4 removed).

        String prerequisites are now rejected by validate_plan_prerequisites()
        rather than silently coerced into dicts.
        """
        plan = {
            'task_id': '8',
            'steps': [{'id': 'step-1', 'status': 'pending'}],
            'prerequisites': [
                {'id': 'pre-1', 'description': 'Already a dict', 'status': 'done'},
                'A string prerequisite',
            ],
        }
        result, modified = _normalize_plan(plan)
        assert modified is False
        prereqs = result['prerequisites']
        assert prereqs[0]['id'] == 'pre-1'
        assert prereqs[1] == 'A string prerequisite'

    def test_empty_plan_unchanged(self):
        plan: dict = {}
        result, modified = _normalize_plan(plan)
        assert modified is False
        assert result == {}

    def test_alias_not_applied_when_steps_already_exists(self):
        """tdd_steps is NOT renamed if steps already exists."""
        plan = {
            'task_id': '9',
            'steps': [{'id': 'step-1', 'status': 'pending'}],
            'tdd_steps': [{'id': 'extra-1', 'status': 'pending'}],
        }
        result, modified = _normalize_plan(plan)
        # tdd_steps is left alone since steps is present
        assert 'tdd_steps' in result
        assert len(result['steps']) == 1


class TestPlanNormalizationIntegration:
    """Tests that read_plan() triggers normalization and writes back."""

    def test_read_plan_writes_back_when_normalized(self, artifacts: TaskArtifacts):
        # Write a malformed plan directly (bypass write_plan to skip schema version)
        raw = {'task_id': 'task-1', 'tdd_steps': [{'id': 'step-1', 'status': 'pending'}]}
        artifacts.artifact_path('plan.json').write_text(json.dumps(raw))

        plan = artifacts.read_plan()
        assert 'steps' in plan
        assert 'tdd_steps' not in plan

        # Re-read from disk — should now be normalized
        raw_on_disk = json.loads(artifacts.artifact_path('plan.json').read_text())
        assert 'steps' in raw_on_disk
        assert 'tdd_steps' not in raw_on_disk

    def test_peek_plan_normalizes_without_writing_back(self, artifacts: TaskArtifacts):
        plan_path = artifacts.artifact_path('plan.json')
        raw = {'task_id': 'task-1', 'tdd_steps': [{'id': 'step-1', 'status': 'pending'}]}
        plan_path.write_text(json.dumps(raw))
        bytes_before = plan_path.read_bytes()
        listing_before = sorted(p.name for p in plan_path.parent.iterdir())

        plan = artifacts.peek_plan()

        assert 'tdd_steps' not in plan
        assert [step['id'] for step in plan['steps']] == ['step-1']
        assert plan_path.read_bytes() == bytes_before
        assert sorted(p.name for p in plan_path.parent.iterdir()) == listing_before

    def test_only_the_on_disk_repair_warns(self, artifacts: TaskArtifacts, caplog):
        plan_path = artifacts.artifact_path('plan.json')
        plan_path.write_text(json.dumps({'tdd_steps': ['write the test']}))

        def artifact_warnings() -> list[str]:
            return [
                record.getMessage()
                for record in caplog.records
                if record.name == 'orchestrator.artifacts'
                and record.levelname == 'WARNING'
            ]

        with caplog.at_level('WARNING'):
            artifacts.peek_plan()
            artifacts.peek_plan()
            assert artifact_warnings() == []

            artifacts.read_plan()
            [repair_warning] = artifact_warnings()

        assert str(plan_path) in repair_warning

    def test_read_plan_no_write_when_valid(self, artifacts: TaskArtifacts):
        valid = {
            'task_id': 'task-1',
            'steps': [{'id': 'step-1', 'status': 'pending'}],
        }
        artifacts.artifact_path('plan.json').write_text(json.dumps(valid))
        import os
        mtime_before = os.path.getmtime(artifacts.artifact_path('plan.json'))

        artifacts.read_plan()

        mtime_after = os.path.getmtime(artifacts.artifact_path('plan.json'))
        assert mtime_before == mtime_after


class TestStringItemResilience:
    """Verify iteration methods don't crash when prerequisites/steps contain strings."""

    def test_update_step_status_survives_string_items(self, artifacts: TaskArtifacts):
        plan = {
            'task_id': 'task-1',
            'prerequisites': ['A string prerequisite'],
            'steps': [{'id': 'step-1', 'status': 'pending', 'commit': None}],
        }
        # Write raw to bypass normalization in write_plan
        artifacts.artifact_path('plan.json').write_text(json.dumps(plan))

        # Should not crash — string items are skipped
        artifacts.update_step_status('step-1', 'done', commit='abc')

    def test_get_pending_steps_survives_string_items(self, artifacts: TaskArtifacts):
        plan = {
            'task_id': 'task-1',
            'prerequisites': ['String prereq'],
            'steps': [
                'String step',
                {'id': 'step-1', 'status': 'pending', 'commit': None},
            ],
        }
        artifacts.artifact_path('plan.json').write_text(json.dumps(plan))

        # read_plan normalizes steps (Rule 5) but not prerequisites (Rule 4 removed);
        # string prerequisites are skipped by the isinstance guard in get_pending_steps
        pending = artifacts.get_pending_steps()
        assert len(pending) >= 1

    def test_get_completed_steps_survives_string_items(self, artifacts: TaskArtifacts):
        plan = {
            'task_id': 'task-1',
            'prerequisites': ['String prereq'],
            'steps': [
                {'id': 'step-1', 'status': 'done', 'commit': 'abc'},
            ],
        }
        artifacts.artifact_path('plan.json').write_text(json.dumps(plan))

        completed = artifacts.get_completed_steps()
        assert len(completed) == 1
        assert completed[0]['id'] == 'step-1'


class TestWorktreeMissingTolerance:
    """Best-effort writes survive a deleted worktree without raising."""

    def test_write_verdict_skips_when_root_gone(self, tmp_path: Path):
        worktree = tmp_path / 'gone-verdict'
        worktree.mkdir()
        ta = TaskArtifacts(worktree)
        ta.init('task-v', 'gone', 'desc')
        import shutil
        shutil.rmtree(worktree)
        assert not ta.task_root.is_dir()

        # Must not raise
        ta.write_verdict('judge', {
            'role': 'judge', 'schema_version': 1, 'session_id': 's',
            'emitted_at': 't', 'verdict': {'complete': True},
        })
        # And nothing was created
        assert not ta.task_root.exists()
        assert ta.read_verdict('judge') is None

    def test_write_json_skips_when_root_gone(self, tmp_path: Path):
        worktree = tmp_path / 'also-gone'
        worktree.mkdir()
        ta = TaskArtifacts(worktree)
        ta.init('task-y', 'gone', 'desc')
        import shutil
        shutil.rmtree(worktree)

        # _write_json into an artifact whose root is gone — should swallow
        ta._write_json('reviews/foo.json', {'a': 1})  # no exception
        assert not ta.task_root.exists()
        assert not worktree.exists()

    @pytest.mark.parametrize('write', [
        pytest.param(lambda ta: ta.set_review_counters(review_cycles_total=1),
                     id='set_review_counters'),
        pytest.param(lambda ta: ta.record_review_verdict('tree', 'PASS', False),
                     id='record_review_verdict'),
        pytest.param(lambda ta: ta.write_plan({'task_id': 't', 'title': 'p', 'steps': []}),
                     id='write_plan'),
        pytest.param(lambda ta: ta.write_blocking_dependency('7', 'r', 'sha'),
                     id='write_blocking_dependency'),
    ])
    def test_default_path_writers_leave_a_vanished_root_gone(self, tmp_path: Path, write):
        worktree = tmp_path / 'gone-default'
        worktree.mkdir()
        ta = TaskArtifacts(worktree)
        ta.init('task-d', 'gone', 'desc')
        shutil.rmtree(worktree)

        write(ta)

        assert not worktree.exists()

    def test_write_review_still_raises_for_genuine_io_errors(
        self, tmp_path: Path, monkeypatch
    ):
        worktree = tmp_path / 'wt'
        worktree.mkdir()
        ta = TaskArtifacts(worktree)
        ta.init('task-z', 'present', 'desc')
        # Root exists, so the FileNotFoundError tolerance in ``_write_json``
        # must NOT engage and a genuine I/O failure must surface.
        #
        # The patch deliberately targets ``Path.mkdir``: ``_write_json``
        # creates the artifact's own directory (here reviews/) with
        # ``Path.mkdir`` before the atomic write, which is why patching it
        # still exercises the seam.  Do NOT re-narrow this to ``write_text``:
        # the atomic writer writes through an ``os.fdopen`` handle and never
        # calls it, so such a patch would stop exercising anything.
        import pathlib

        def _raise_permission(self, *args, **kwargs):
            raise PermissionError('denied')

        monkeypatch.setattr(pathlib.Path, 'mkdir', _raise_permission)
        with pytest.raises(PermissionError):
            ta.write_review('rev', {'verdict': 'PASS', 'issues': []})


class TestMetaRootDerivation:
    """`meta_root_for` is the single owner of the `.task-meta/<name>` path
    shape (PRD `.task-meta path-derivation contract`) — no other module
    joins '.task' by hand.
    """

    def test_meta_root_for_derives_task_meta_path(self):
        assert TaskArtifacts.meta_root_for(Path('/pool/.worktrees'), '_lane-0') == (
            Path('/pool/.worktrees/.task-meta/_lane-0')
        )

    def test_meta_root_for_different_base_and_name(self):
        assert TaskArtifacts.meta_root_for(Path('/other/base'), 'task-42') == (
            Path('/other/base/.task-meta/task-42')
        )


class TestTaskRootFor:
    """`task_root_for` owns the task-lifetime `.task-meta/task-<id>` shape
    (PRD plans/task-meta-task-keying-prd.md §Path-derivation contract), and
    refuses an id that could leave `.task-meta/` or name a non-dir.
    """

    BASE = Path('/pool/.worktrees')

    @pytest.mark.parametrize('task_id', ['3155', 'df_task_12', 'a.b-c_9'])
    def test_accepted_id_derives_task_prefixed_dir(self, task_id: str):
        assert TaskArtifacts.task_root_for(self.BASE, task_id) == (
            self.BASE / '.task-meta' / f'task-{task_id}'
        )

    @pytest.mark.parametrize('task_id', ['', '../x', 'a/b', 'a b', '3155\n', 3155])
    def test_rejected_id_raises_naming_value_and_pattern(self, task_id):
        with pytest.raises(ValueError) as excinfo:
            TaskArtifacts.task_root_for(self.BASE, task_id)
        message = str(excinfo.value)
        assert repr(task_id) in message
        assert '^[A-Za-z0-9._-]+$' in message

    def test_meta_root_for_lane_shape_is_unchanged(self):
        assert TaskArtifacts.meta_root_for(self.BASE, '_lane-3') == (
            self.BASE / '.task-meta' / '_lane-3'
        )


class TestReadLaneTaskId:
    """`read_lane_task_id` resolves lane -> task off `metadata.json` ONLY, and
    a missing, undecodable, corrupt or malformed file is a miss, never a raise.
    """

    @pytest.fixture
    def meta_root(self, tmp_path: Path) -> Path:
        root = tmp_path / '.task-meta' / '_lane-0'
        root.mkdir(parents=True)
        return root

    def test_returns_task_id_from_metadata_json(self, meta_root: Path):
        (meta_root / 'metadata.json').write_text(json.dumps({
            'task_id': '5069',
            'title': 'T',
            'created_at': '2026-09-02T00:00:00+00:00',
        }))
        assert TaskArtifacts.read_lane_task_id(meta_root) == '5069'

    def test_missing_meta_root_returns_none(self, tmp_path: Path):
        meta_root = tmp_path / '.task-meta' / '_lane-0'
        assert not meta_root.exists()
        assert TaskArtifacts.read_lane_task_id(meta_root) is None

    def test_missing_metadata_json_returns_none(self, meta_root: Path):
        assert TaskArtifacts.read_lane_task_id(meta_root) is None

    @pytest.mark.parametrize('payload', [
        b'{not json',
        b'\xff\xfe{"task_id": "5069"}',
        b'["a", "b"]',
        b'{"title": "T"}',
        b'{"task_id": ""}',
        b'{"task_id": {"nested": 1}}',
        b'{"task_id": true}',
    ], ids=[
        'corrupt-json', 'non-utf8', 'non-dict', 'no-task-id', 'empty-task-id',
        'dict-task-id', 'bool-task-id',
    ])
    def test_unusable_metadata_returns_none(self, meta_root: Path, payload: bytes):
        (meta_root / 'metadata.json').write_bytes(payload)
        assert TaskArtifacts.read_lane_task_id(meta_root) is None

    def test_integer_task_id_is_coerced_to_str(self, meta_root: Path):
        (meta_root / 'metadata.json').write_text(json.dumps({'task_id': 5069}))
        assert TaskArtifacts.read_lane_task_id(meta_root) == '5069'

    def test_plan_json_is_not_consulted(self, meta_root: Path):
        (meta_root / 'plan.json').write_text(json.dumps({'task_id': '5069'}))
        assert TaskArtifacts.read_lane_task_id(meta_root) is None


class TestMetaRootConstructor:
    """`meta_root` is an OPTIONAL 2nd constructor arg. Supplying it activates
    the `.task-meta` relocation; writes must land there and NEVER under the
    legacy `<worktree>/.task` path.
    """

    @pytest.fixture
    def meta_root(self, worktree: Path) -> Path:
        return TaskArtifacts.meta_root_for(worktree.parent, worktree.name)

    def test_root_is_meta_root_when_supplied(self, worktree: Path, meta_root: Path):
        worktree.mkdir()
        ta = TaskArtifacts(worktree, meta_root)
        assert ta.lane_root == ta.task_root == meta_root

    def test_init_and_write_plan_land_under_meta_root_only(
        self, worktree: Path, meta_root: Path
    ):
        worktree.mkdir()
        ta = TaskArtifacts(worktree, meta_root)
        ta.init('task-1', 'Test Task', 'A test task description')
        ta.write_plan({
            'steps': [{'id': 'step-1', 'status': 'pending', 'commit': None}],
        })

        assert (meta_root / 'metadata.json').exists()
        assert (meta_root / 'reviews').is_dir()
        assert (meta_root / 'plan.json').exists()

        # Writes are new-path ONLY — the legacy `.task` dir must not appear.
        assert not (worktree / '.task').exists()


class TestReadPlanNewThenOldCompat:
    """`read_plan`/`read_base_commit` resolve the new (meta_root) path only.
    The legacy `<worktree>/.task` path is never consulted, even when it
    holds data and the new path is empty — the compat-window fallback was
    retired in W11-ι.
    """

    @pytest.fixture
    def meta_root(self, worktree: Path) -> Path:
        return TaskArtifacts.meta_root_for(worktree.parent, worktree.name)

    @pytest.fixture
    def legacy_root(self, worktree: Path) -> Path:
        return worktree / '.task'

    def test_reads_from_new_when_only_new(
        self, worktree: Path, meta_root: Path
    ):
        worktree.mkdir()
        ta = TaskArtifacts(worktree, meta_root)
        ta.init('task-1', 'Test Task', 'Desc')
        ta.write_plan({'steps': [{'id': 'new-step', 'status': 'pending', 'commit': None}]})

        read = ta.read_plan()
        assert read['steps'][0]['id'] == 'new-step'

    def test_read_ignores_legacy_when_meta_root_supplied(
        self, worktree: Path, meta_root: Path, legacy_root: Path
    ):
        """meta_root supplied (new path empty); plan.json + metadata.json
        written ONLY under the legacy path. read_plan/read_base_commit must
        NOT fall back to legacy — they see nothing.
        """
        worktree.mkdir()
        legacy_root.mkdir(parents=True)
        legacy_plan = {'steps': [{'id': 'legacy-step', 'status': 'pending', 'commit': None}]}
        (legacy_root / 'plan.json').write_text(json.dumps(legacy_plan))
        (legacy_root / 'metadata.json').write_text(json.dumps(
            {'task_id': 'x', 'base_commit': 'legacy-sha'}
        ))

        ta = TaskArtifacts(worktree, meta_root)
        assert ta.read_plan() == {}
        assert ta.read_base_commit() is None

    def test_prefers_new_when_both_exist(
        self, worktree: Path, meta_root: Path, legacy_root: Path
    ):
        worktree.mkdir()
        legacy_root.mkdir(parents=True)
        legacy_plan = {'steps': [{'id': 'legacy-step', 'status': 'pending', 'commit': None}]}
        (legacy_root / 'plan.json').write_text(json.dumps(legacy_plan))

        ta = TaskArtifacts(worktree, meta_root)
        ta.init('task-1', 'Test Task', 'Desc')
        ta.write_plan({'steps': [{'id': 'new-step', 'status': 'pending', 'commit': None}]})

        read = ta.read_plan()
        assert read['steps'][0]['id'] == 'new-step'


class TestReadReviewsMergesLegacyAndNew:
    """read_reviews reads the new (meta_root) reviews/ directory only — the
    legacy `<worktree>/.task/reviews/` directory is never consulted, even
    when it holds reviewer files and the new dir is empty or absent. The
    legacy-merge behavior this class originally covered was retired in
    W11-ι; the remaining tests pin new-path-only reads and the still-fatal
    corrupt-file contract on that path.
    """

    @pytest.fixture
    def meta_root(self, worktree: Path) -> Path:
        return TaskArtifacts.meta_root_for(worktree.parent, worktree.name)

    @pytest.fixture
    def legacy_root(self, worktree: Path) -> Path:
        return worktree / '.task'

    def test_new_wins_on_stem_collision(
        self, worktree: Path, meta_root: Path, legacy_root: Path
    ):
        """Legacy data for the same reviewer stem is simply invisible — the
        new-path write is the only one read_reviews ever sees.
        """
        worktree.mkdir()
        (legacy_root / 'reviews').mkdir(parents=True)
        (legacy_root / 'reviews' / 'reviewer-a.json').write_text(json.dumps({
            'verdict': 'OLD',
            'issues': [],
        }))

        ta = TaskArtifacts(worktree, meta_root)
        ta.init('task-1', 'Test Task', 'Desc')
        ta.write_review('reviewer-a', {'verdict': 'NEW', 'issues': []})

        assert ta.read_reviews()['reviewer-a']['verdict'] == 'NEW'

    def test_corrupt_new_review_file_is_fatal(
        self, worktree: Path, meta_root: Path,
    ):
        """A corrupt reviewer JSON file under the new path is NOT silently
        tolerated — a malformed review aborts read_reviews (and therefore
        aggregate_reviews) instead of being skipped. Unlike
        read_iteration_log (which tolerates corrupted *lines*), read_reviews
        has no per-file error handling.
        """
        worktree.mkdir()
        (meta_root / 'reviews').mkdir(parents=True)
        (meta_root / 'reviews' / 'reviewer-b.json').write_text('{not valid json')

        ta = TaskArtifacts(worktree, meta_root)

        with pytest.raises(json.JSONDecodeError):
            ta.read_reviews()


class TestReviewVerdictCache:
    """Tests for the tree-hash-keyed verdict cache (review_state.json)."""

    def test_fresh_root_returns_default_state(self, artifacts: TaskArtifacts):
        assert artifacts.read_review_state() == {
            'amendment_rounds_total': 0,
            'review_cycles_total': 0,
            'verdicts': {},
        }

    def test_record_and_get_cached_verdict(self, artifacts: TaskArtifacts):
        artifacts.record_review_verdict(
            'tree_abc', 'suggestions_only', suggestions_routed=True
        )
        rec = artifacts.get_cached_verdict('tree_abc')
        assert rec is not None
        assert rec['verdict'] == 'suggestions_only'
        assert rec['suggestions_routed'] is True
        assert isinstance(rec['ts'], str) and rec['ts']
        # ts must be an ISO-8601 datetime string.
        datetime.fromisoformat(rec['ts'])

    def test_get_cached_verdict_absent_returns_none(self, artifacts: TaskArtifacts):
        artifacts.record_review_verdict(
            'tree_abc', 'suggestions_only', suggestions_routed=True
        )
        assert artifacts.get_cached_verdict('other_tree') is None

    def test_second_tree_hash_coexists(self, artifacts: TaskArtifacts):
        artifacts.record_review_verdict('tree_abc', 'suggestions_only', True)
        artifacts.record_review_verdict('tree_def', 'PASS', False)
        rec_abc = artifacts.get_cached_verdict('tree_abc')
        rec_def = artifacts.get_cached_verdict('tree_def')
        assert rec_abc is not None and rec_abc['verdict'] == 'suggestions_only'
        assert rec_def is not None and rec_def['verdict'] == 'PASS'

    def test_same_tree_hash_overwrites_last_wins(self, artifacts: TaskArtifacts):
        artifacts.record_review_verdict('tree_abc', 'suggestions_only', True)
        artifacts.record_review_verdict('tree_abc', 'PASS', False)
        rec = artifacts.get_cached_verdict('tree_abc')
        assert rec is not None
        assert rec['verdict'] == 'PASS'
        assert rec['suggestions_routed'] is False

    def test_get_cached_verdict_on_corrupt_review_state_returns_none(
        self, artifacts: TaskArtifacts
    ):
        artifacts.artifact_path('review_state.json').write_text('{not valid json')
        assert artifacts.get_cached_verdict('tree_abc') is None

    def test_reviewer_fingerprint_round_trips(self, artifacts: TaskArtifacts):
        # The optional fingerprint (task 2749 amendment) is stored verbatim.
        artifacts.record_review_verdict(
            'tree_abc', 'suggestions_only', True, reviewer_fingerprint='FP1'
        )
        rec = artifacts.get_cached_verdict('tree_abc')
        assert rec is not None
        assert rec['reviewer_fingerprint'] == 'FP1'

    def test_reviewer_fingerprint_defaults_none(self, artifacts: TaskArtifacts):
        # Omitting the fingerprint stores an explicit None (never absent), so
        # the reader's positive-match test is well-defined.
        artifacts.record_review_verdict('tree_abc', 'PASS', False)
        rec = artifacts.get_cached_verdict('tree_abc')
        assert rec is not None
        assert rec['reviewer_fingerprint'] is None


class TestReviewCounters:
    """Tests for the persisted task-lifetime amendment/review counters."""

    def test_counters_default_zero(self, artifacts: TaskArtifacts):
        assert artifacts.get_amendment_rounds_total() == 0
        assert artifacts.get_review_cycles_total() == 0

    def test_partial_update_amendment_only(self, artifacts: TaskArtifacts):
        artifacts.set_review_counters(amendment_rounds_total=2)
        assert artifacts.get_amendment_rounds_total() == 2
        # The other counter is untouched.
        assert artifacts.get_review_cycles_total() == 0

    def test_independent_fields_read_modify_write(self, artifacts: TaskArtifacts):
        artifacts.set_review_counters(amendment_rounds_total=2)
        artifacts.set_review_counters(review_cycles_total=1)
        # Setting review_cycles must not clobber amendment_rounds.
        assert artifacts.get_amendment_rounds_total() == 2
        assert artifacts.get_review_cycles_total() == 1

    def test_counters_and_verdicts_coexist(self, artifacts: TaskArtifacts):
        artifacts.record_review_verdict('tree_abc', 'suggestions_only', True)
        artifacts.set_review_counters(
            amendment_rounds_total=3, review_cycles_total=2
        )
        # Neither write clobbers the other's keys.
        rec = artifacts.get_cached_verdict('tree_abc')
        assert rec is not None and rec['verdict'] == 'suggestions_only'
        assert artifacts.get_amendment_rounds_total() == 3
        assert artifacts.get_review_cycles_total() == 2

    def test_clear_review_counters_resets_both(self, artifacts: TaskArtifacts):
        # The operator/resume-path reset hook (task 2749 amendment) zeroes
        # both counters so a human re-pend can make fresh replan progress.
        artifacts.set_review_counters(
            amendment_rounds_total=3, review_cycles_total=2
        )
        artifacts.clear_review_counters()
        assert artifacts.get_amendment_rounds_total() == 0
        assert artifacts.get_review_cycles_total() == 0

    def test_clear_review_counters_preserves_verdicts(
        self, artifacts: TaskArtifacts
    ):
        # Clearing counters must NOT drop the verdict cache — an unchanged
        # tree under an unchanged config legitimately still skips.
        artifacts.record_review_verdict(
            'tree_abc', 'suggestions_only', True, reviewer_fingerprint='FP1'
        )
        artifacts.set_review_counters(review_cycles_total=2)
        artifacts.clear_review_counters()
        rec = artifacts.get_cached_verdict('tree_abc')
        assert rec is not None and rec['verdict'] == 'suggestions_only'
        assert rec['reviewer_fingerprint'] == 'FP1'


_LANE_ARTIFACT_WRITERS: dict[str, Callable[[TaskArtifacts], None]] = {
    'plan.json': lambda a: a.write_plan({'steps': [{'id': 'step-1', 'status': 'pending', 'commit': None}]}),
    'iterations.jsonl': lambda a: a.append_iteration_log({'iteration': 1}),
}
"""Plants each `LANE_SYMLINKED_ARTIFACTS` name through its public writer."""


class TestEnsureLaneSymlinks:
    """`ensure_lane_symlinks()` single-sources each name in
    `LANE_SYMLINKED_ARTIFACTS`: the lane copy `<worktree>/.task/<name>` becomes
    an ABSOLUTE symlink INTO the durable `artifact_path(<name>)`. Any reader or
    appender of the lane path then transparently reaches the durable file —
    the esc-5205-9 stale-trap (a regular-file lane copy that diverges from the
    durable one) is impossible because the lane path IS the durable file.
    No-op in legacy `meta_root=None` mode, where the lane path already IS the
    durable copy.
    """

    @pytest.fixture
    def meta_root(self, worktree: Path) -> Path:
        return TaskArtifacts.meta_root_for(worktree.parent, worktree.name)

    @pytest.fixture
    def ta(self, worktree: Path, meta_root: Path) -> TaskArtifacts:
        worktree.mkdir()
        ta = TaskArtifacts(worktree, meta_root)
        ta.init('task-1', 'Test Task', 'Desc')
        return ta

    @pytest.mark.parametrize('name', LANE_SYMLINKED_ARTIFACTS)
    def test_creates_absolute_symlink_to_meta_root_artifact(
        self, ta: TaskArtifacts, worktree: Path, meta_root: Path, name: str
    ):
        # meta_root mode: init() does NOT create <worktree>/.task (see
        # TestMetaRootConstructor.test_init_and_write_plan_land_under_meta_root_only),
        # so the method itself must mkdir the lane .task/ before symlinking.
        assert not (worktree / '.task').exists()

        ta.ensure_lane_symlinks()

        lane_link = worktree / '.task' / name
        assert lane_link.is_symlink()
        assert os.path.isabs(os.readlink(lane_link))
        assert lane_link.resolve() == (meta_root / name).resolve()

    def test_read_through_returns_meta_root_content(
        self, ta: TaskArtifacts, worktree: Path, meta_root: Path
    ):
        # write_plan lands at meta_root/plan.json; reading the lane symlink
        # returns the same content.
        ta.ensure_lane_symlinks()
        ta.write_plan(
            {'steps': [{'id': 'step-1', 'status': 'pending', 'commit': None}]}
        )

        lane_plan = worktree / '.task' / 'plan.json'
        read = json.loads(lane_plan.read_text())
        assert read['steps'][0]['id'] == 'step-1'
        assert read == json.loads((meta_root / 'plan.json').read_text())

    @pytest.mark.parametrize('name', LANE_SYMLINKED_ARTIFACTS)
    def test_replaces_preplanted_regular_file(
        self, ta: TaskArtifacts, worktree: Path, meta_root: Path, name: str
    ):
        # The esc-5205-9 stale-trap: a pre-existing regular-FILE lane copy is
        # REPLACED by the symlink.
        lane_link = worktree / '.task' / name
        lane_link.parent.mkdir(parents=True, exist_ok=True)
        lane_link.write_text('{"stale": "trap"}')
        assert lane_link.is_file() and not lane_link.is_symlink()

        ta.ensure_lane_symlinks()

        assert lane_link.is_symlink()
        assert lane_link.resolve() == (meta_root / name).resolve()

    @pytest.mark.parametrize('name', LANE_SYMLINKED_ARTIFACTS)
    def test_replaces_broken_symlink(
        self, ta: TaskArtifacts, worktree: Path, meta_root: Path, name: str
    ):
        # A dangling symlink is retargeted at the meta-root, not left
        # pointing at its absent target.
        lane_task = worktree / '.task'
        lane_task.mkdir(parents=True, exist_ok=True)
        lane_link = lane_task / name
        os.symlink(lane_task / 'gone.json', lane_link)
        assert lane_link.is_symlink() and not lane_link.exists()

        ta.ensure_lane_symlinks()

        assert lane_link.is_symlink()
        assert os.path.isabs(os.readlink(lane_link))
        assert lane_link.resolve() == (meta_root / name).resolve()

    @pytest.mark.parametrize('name', LANE_SYMLINKED_ARTIFACTS)
    def test_replaces_stray_directory(
        self, ta: TaskArtifacts, worktree: Path, meta_root: Path, name: str
    ):
        # A stray NON-EMPTY directory is removed and replaced (a plain rmdir
        # would raise on it).
        lane_link = worktree / '.task' / name
        lane_link.mkdir(parents=True)
        (lane_link / 'junk.txt').write_text('stray')
        assert lane_link.is_dir() and not lane_link.is_symlink()

        ta.ensure_lane_symlinks()

        assert lane_link.is_symlink()
        assert lane_link.resolve() == (meta_root / name).resolve()

    @pytest.mark.parametrize('name', LANE_SYMLINKED_ARTIFACTS)
    def test_idempotent(
        self, ta: TaskArtifacts, worktree: Path, meta_root: Path, name: str
    ):
        ta.ensure_lane_symlinks()
        ta.ensure_lane_symlinks()

        lane_link = worktree / '.task' / name
        assert lane_link.is_symlink()
        assert lane_link.resolve() == (meta_root / name).resolve()

    @pytest.mark.parametrize('name', LANE_SYMLINKED_ARTIFACTS)
    def test_noop_in_legacy_meta_root_none_mode(
        self, artifacts: TaskArtifacts, worktree: Path, name: str
    ):
        # Legacy meta_root=None mode (the `artifacts` fixture builds
        # TaskArtifacts(worktree), rooted at worktree/.task): a pre-existing
        # real file stays a regular FILE — no self-referential symlink.
        assert artifacts.lane_root == artifacts.task_root == worktree / '.task'
        _LANE_ARTIFACT_WRITERS[name](artifacts)

        artifacts.ensure_lane_symlinks()

        lane_file = worktree / '.task' / name
        assert lane_file.is_file()
        assert not lane_file.is_symlink()


_LANE_ARTIFACT_NAMES = (
    'metadata.json', 'agent_session.json', 'interactive.json',
    'blocking_dependency.json', 'already_done.json', 'ready_to_merge.json',
    'false_premise.json', 'unactionable_task.json',
)
_TASK_ARTIFACT_NAMES = (
    'plan.json', 'plan.lock', 'iterations.jsonl', 'reviews', 'verdicts',
    'review_state.json', 'reconcile_state.json',
)
_PLAN = {'steps': [{'id': 'step-1', 'status': 'pending', 'commit': None}]}
_VERDICT = {
    'role': 'judge', 'schema_version': 1, 'session_id': 's',
    'emitted_at': 't', 'verdict': {'complete': True},
}


class _DualRootLayout:
    """One lane root and one task root for worktree ``wt-x`` / task ``42``."""

    @pytest.fixture
    def wt(self, tmp_path: Path) -> Path:
        wt = tmp_path / 'wt-x'
        wt.mkdir()
        return wt

    @pytest.fixture
    def lane(self, wt: Path) -> Path:
        return TaskArtifacts.meta_root_for(wt.parent, wt.name)

    @pytest.fixture
    def task(self, wt: Path) -> Path:
        return TaskArtifacts.task_root_for(wt.parent, '42')

    @pytest.fixture
    def arts(self, wt: Path, lane: Path, task: Path) -> TaskArtifacts:
        arts = TaskArtifacts(wt, lane, task)
        arts.init('42', 'Dual', 'desc')
        return arts


def _names(root: Path) -> set[str]:
    return {p.name for p in root.iterdir()}


def _write_every_task_artifact(arts: TaskArtifacts) -> None:
    arts.write_plan(copy.deepcopy(_PLAN))
    arts.lock_plan('42-abc')
    arts.append_iteration_log({'iteration': 1})
    arts.write_review('r', {'verdict': 'PASS', 'issues': []})
    arts.write_verdict('judge', _VERDICT)
    arts.set_review_counters(amendment_rounds_total=1)
    arts.record_review_verdict('tree', 'PASS', False)
    arts.append_emitted_step_escalation('step-1', 'deadbeef')


def _write_every_lane_artifact(arts: TaskArtifacts) -> None:
    arts.write_agent_session('42-abc', 'implementer', 't', task_id='42')
    arts.write_blocking_dependency('7', 'needs 7', 'sha')
    arts.write_already_done('sha', 'ev')
    arts.write_ready_to_merge('sha', 'ev')
    arts.write_false_premise('c', 'p', 'e', 'r')
    arts.write_unactionable_task('r', 'e')


class TestDualRootRouting(_DualRootLayout):
    """``TaskArtifacts(wt, lane, task)`` routes each artifact to the root the
    frozen table names — task-lifetime state under ``task-<id>``, per-dispatch
    exits under the lane — with no fallback between them (PRD
    plans/task-meta-task-keying-prd.md decision 7).
    """

    def test_init_creates_both_roots_and_splits_them(self, arts, lane, task):
        assert lane.is_dir() and task.is_dir()
        assert (lane / 'metadata.json').is_file()
        assert (task / 'reviews').is_dir() and (task / 'verdicts').is_dir()
        assert not (lane / 'reviews').exists()
        assert not (lane / 'verdicts').exists()

    @pytest.mark.parametrize(('write', 'relpath'), [
        (lambda a: a.write_plan(copy.deepcopy(_PLAN)), 'plan.json'),
        (lambda a: a.lock_plan('42-abc'), 'plan.lock'),
        (lambda a: a.append_iteration_log({'iteration': 1}), 'iterations.jsonl'),
        (lambda a: a.write_review('r', {'verdict': 'PASS'}), 'reviews/r.json'),
        (lambda a: a.write_verdict('judge', _VERDICT), 'verdicts/judge.json'),
        (lambda a: a.set_review_counters(review_cycles_total=2), 'review_state.json'),
        (lambda a: a.record_review_verdict('tree', 'PASS', False), 'review_state.json'),
        (lambda a: a.append_emitted_step_escalation('s', 'c'), 'reconcile_state.json'),
    ])
    def test_task_keyed_writer_lands_under_task_only(self, arts, lane, task, write, relpath):
        write(arts)
        assert (task / relpath).is_file()
        assert not (lane / relpath).exists()

    @pytest.mark.parametrize(('write', 'name'), [
        (lambda a: None, 'metadata.json'),
        (lambda a: a.write_agent_session('s', 'implementer', 't'), 'agent_session.json'),
        (lambda a: a.write_blocking_dependency('7', 'r', 'sha'), 'blocking_dependency.json'),
        (lambda a: a.write_already_done('sha', 'ev'), 'already_done.json'),
        (lambda a: a.write_ready_to_merge('sha', 'ev'), 'ready_to_merge.json'),
        (lambda a: a.write_false_premise('c', 'p', 'e', 'r'), 'false_premise.json'),
        (lambda a: a.write_unactionable_task('r', 'e'), 'unactionable_task.json'),
    ])
    def test_lane_keyed_writer_lands_under_lane_only(self, arts, lane, task, write, name):
        write(arts)
        assert (lane / name).is_file()
        assert not (task / name).exists()

    def test_lane_holds_only_lane_keyed_names_after_every_writer(self, arts, lane):
        _write_every_task_artifact(arts)
        _write_every_lane_artifact(arts)
        assert _names(lane) <= set(_LANE_ARTIFACT_NAMES)
        assert 'plan.json.mutex' not in _names(lane)

    def test_plan_planted_in_lane_is_invisible(self, arts, lane):
        (lane / 'plan.json').write_text(json.dumps(_PLAN))
        assert arts.read_plan() == {}

    def test_lane_exit_planted_in_task_is_invisible(self, arts, task):
        (task / 'blocking_dependency.json').write_text(json.dumps({'depends_on_task_id': '7'}))
        assert arts.read_blocking_dependency() is None

    def test_artifact_path_resolves_through_the_table(self, arts, lane, task):
        assert arts.artifact_path('plan.json') == task / 'plan.json'
        assert arts.artifact_path('metadata.json') == lane / 'metadata.json'
        assert arts.artifact_path('reviews/merge.json') == task / 'reviews' / 'merge.json'

    def test_artifact_path_refuses_unknown_name(self, arts):
        with pytest.raises(ValueError, match='nope.json'):
            arts.artifact_path('nope.json')

    def test_family_accessors_route_to_task(self, arts, task):
        assert arts.review_archive_dir('amend-1') == task / 'reviews-amend-1'
        assert arts.review_archive_dir('cycle-2') == task / 'reviews-cycle-2'
        assert arts.review_archives_root == task
        assert arts.zero_output_evidence_path(3) == task / 'zero_output_evidence-iter3.json'

    def test_root_properties(self, arts, lane, task):
        assert arts.lane_root == lane
        assert arts.task_root == task

    def test_for_task_derives_both_roots(self, wt):
        arts = TaskArtifacts.for_task(wt, '42')
        assert arts.lane_root == TaskArtifacts.meta_root_for(wt.parent, wt.name)
        assert arts.task_root == TaskArtifacts.task_root_for(wt.parent, '42')

    @pytest.mark.parametrize('single_root', ['legacy', 'meta'])
    def test_single_root_modes_resolve_everything_under_one_root(self, wt, lane, single_root):
        if single_root == 'legacy':
            arts, root = TaskArtifacts(wt), wt / '.task'
        else:
            arts, root = TaskArtifacts(wt, lane), lane
        assert arts.lane_root == arts.task_root == root
        for name in (*_LANE_ARTIFACT_NAMES, *_TASK_ARTIFACT_NAMES):
            assert arts.artifact_path(name) == root / name
        assert arts.review_archive_dir('amend-1') == root / 'reviews-amend-1'
        assert arts.review_archives_root == root
        assert arts.zero_output_evidence_path(1) == root / 'zero_output_evidence-iter1.json'

    def test_lane_plan_symlink_targets_the_task_plan(self, arts, wt, task):
        arts.ensure_lane_symlinks()
        lane_plan = wt / '.task' / 'plan.json'
        assert lane_plan.is_symlink()
        assert lane_plan.resolve() == (task / 'plan.json').resolve()

        arts.write_plan(copy.deepcopy(_PLAN))
        assert json.loads(lane_plan.read_text()) == arts.read_plan()

        TaskArtifacts(wt).update_step_status('step-1', 'done', 'abc123')
        assert lane_plan.is_symlink()
        stored = json.loads((task / 'plan.json').read_text())
        assert stored['steps'][0]['status'] == 'done'

    def test_lane_iterations_symlink_targets_the_task_log(self, arts, wt, task):
        arts.ensure_lane_symlinks()
        lane_log = wt / '.task' / 'iterations.jsonl'
        task_log = task / 'iterations.jsonl'
        assert lane_log.is_symlink()
        assert lane_log.resolve() == task_log.resolve()

        with open(lane_log, 'a') as f:
            f.write(json.dumps({'iteration': 0, 'source': 'agent'}) + '\n')
        assert task_log.is_file()
        entries, _ = arts.read_iteration_log()
        assert entries == [{'iteration': 0, 'source': 'agent'}]

        arts.append_iteration_log({'iteration': 1})
        assert [json.loads(line)['iteration'] for line in lane_log.read_text().splitlines()] == [0, 1]

        arts.clear_iteration_log()
        assert lane_log.is_symlink()
        assert not task_log.exists()
        with open(lane_log, 'a') as f:
            f.write(json.dumps({'iteration': 2, 'source': 'agent'}) + '\n')
        assert task_log.is_file()

    def test_task_root_gone_refuses_only_the_review_write(self, arts, lane, task):
        shutil.rmtree(task)

        with pytest.raises(ReviewsRootMissing):
            arts.write_review('r', {'verdict': 'PASS'})
        arts.write_verdict('judge', _VERDICT)
        with arts.plan_mutex():
            pass
        arts.set_review_counters(review_cycles_total=1)
        arts.write_plan(copy.deepcopy(_PLAN))
        assert not task.exists()

        arts.write_blocking_dependency('7', 'r', 'sha')
        assert arts.read_blocking_dependency()['depends_on_task_id'] == '7'
        assert (lane / 'blocking_dependency.json').is_file()

    def test_plan_mutex_sits_beside_the_task_plan(self, arts, lane, task):
        arts.write_plan(copy.deepcopy(_PLAN))
        assert (task / 'plan.json.mutex').exists()
        assert not (lane / 'plan.json.mutex').exists()

    def test_lane_symlink_view_shares_the_task_plan_mutex(self, arts, wt):
        arts.write_plan(copy.deepcopy(_PLAN))
        arts.ensure_lane_symlinks()
        with arts.plan_mutex():
            TaskArtifacts(wt).write_plan(copy.deepcopy(_PLAN), mutex_held=True)

    def test_unheld_mutex_claim_names_the_task_plan(self, arts, task):
        with pytest.raises(PlanMutexNotHeld, match=re.escape(str(task / 'plan.json'))):
            arts.write_plan(copy.deepcopy(_PLAN), mutex_held=True)


class TestClearReviews(_DualRootLayout):
    """``clear_reviews()`` empties reviews/ so the next aggregation holds only
    reviews written after it (task 3158; PRD plans/task-meta-task-keying-prd.md
    boundary B7)."""

    def test_removes_every_review_and_keeps_the_dir(self, artifacts: TaskArtifacts):
        artifacts.write_review('reviewer_comprehensive', pass_review('reviewer_comprehensive'))
        artifacts.write_review('merge', blocking_review('merge'))
        reviews_dir = artifacts.artifact_path('reviews')
        (reviews_dir / 'notes.txt').write_text('stray\n')

        artifacts.clear_reviews()

        assert reviews_dir.is_dir()
        assert list(reviews_dir.iterdir()) == []
        assert artifacts.read_reviews() == {}

    def test_a_retired_reviewers_blocking_file_is_not_aggregated_after_clear(
        self, artifacts: TaskArtifacts,
    ):
        artifacts.write_review('reviewer_retired', blocking_review('reviewer_retired'))

        artifacts.clear_reviews()
        artifacts.write_review('reviewer_comprehensive', pass_review('reviewer_comprehensive'))

        agg = artifacts.aggregate_reviews()
        assert agg.has_blocking_issues is False
        assert set(agg.reviews) == {'reviewer_comprehensive'}
        assert agg.reviewer_errors == []

    def test_leaves_archives_verdicts_review_state_and_plan_alone(
        self, artifacts: TaskArtifacts,
    ):
        archive = artifacts.review_archive_dir('amend-1')
        archive.mkdir()
        (archive / 'reviewer_comprehensive.json').write_text(
            json.dumps(blocking_review('reviewer_comprehensive'))
        )
        artifacts.write_verdict('judge', _VERDICT)
        artifacts.set_review_counters(review_cycles_total=2)
        artifacts.write_plan(copy.deepcopy(_PLAN))
        artifacts.write_review('reviewer_comprehensive', pass_review('reviewer_comprehensive'))

        artifacts.clear_reviews()

        assert (archive / 'reviewer_comprehensive.json').is_file()
        assert artifacts.read_verdict('judge') == _VERDICT
        assert artifacts.get_review_cycles_total() == 2
        assert artifacts.read_plan()['steps'] == _PLAN['steps']

    def test_is_idempotent(self, artifacts: TaskArtifacts):
        artifacts.write_review('reviewer_comprehensive', pass_review('reviewer_comprehensive'))

        artifacts.clear_reviews()
        artifacts.clear_reviews()

        assert artifacts.read_reviews() == {}

    def test_root_gone_is_a_no_op_that_recreates_nothing(self, artifacts: TaskArtifacts):
        shutil.rmtree(artifacts.task_root)

        artifacts.clear_reviews()

        assert not artifacts.task_root.exists()

    def test_dual_root_empties_the_task_root_reviews(self, arts, task):
        arts.write_review('reviewer_comprehensive', pass_review('reviewer_comprehensive'))
        assert (task / 'reviews' / 'reviewer_comprehensive.json').is_file()

        arts.clear_reviews()

        assert (task / 'reviews').is_dir()
        assert list((task / 'reviews').iterdir()) == []


class TestWriteReviewRefusesVanishedRoot:
    """``write_review`` raises instead of skipping when its root is gone, and
    never recreates that root: reviews/ is the review gate's only input, so a
    skipped write would read as a PASS (task 3158)."""

    def test_raises_when_root_gone(self, tmp_path: Path):
        worktree = tmp_path / 'gone'
        worktree.mkdir()
        ta = TaskArtifacts(worktree)
        ta.init('task-x', 'gone', 'desc')
        shutil.rmtree(worktree)
        assert not ta.task_root.is_dir()

        with pytest.raises(ReviewsRootMissing) as excinfo:
            ta.write_review('reviewer-x', pass_review('reviewer-x'))

        assert isinstance(excinfo.value, FileNotFoundError)
        assert 'reviewer-x' in str(excinfo.value)
        assert str(ta.task_root) in str(excinfo.value)
        assert not ta.task_root.exists()

    def test_a_root_vanishing_mid_write_raises_and_stays_gone(
        self, artifacts: TaskArtifacts, monkeypatch: pytest.MonkeyPatch,
    ):
        real_write = safe_io.atomic_write_text

        def root_vanishes_then_write(path, text, **kwargs):
            shutil.rmtree(artifacts.task_root)
            return real_write(path, text, **kwargs)

        monkeypatch.setattr(safe_io, 'atomic_write_text', root_vanishes_then_write)

        with pytest.raises(ReviewsRootMissing):
            artifacts.write_review('reviewer_comprehensive', pass_review('reviewer_comprehensive'))

        assert not artifacts.task_root.exists()

    def test_creates_a_missing_reviews_dir_under_a_live_root(self, artifacts: TaskArtifacts):
        shutil.rmtree(artifacts.artifact_path('reviews'))

        artifacts.write_review('reviewer_comprehensive', pass_review('reviewer_comprehensive'))

        assert set(artifacts.read_reviews()) == {'reviewer_comprehensive'}


class TestReadTitle:
    """``read_title()`` — metadata.json's title, else plan.json's, else None."""

    @pytest.fixture
    def arts(self, worktree: Path) -> TaskArtifacts:
        worktree.mkdir()
        arts = TaskArtifacts(worktree)
        arts.artifact_path('metadata.json').parent.mkdir(parents=True)
        return arts

    def _seed(self, arts: TaskArtifacts, name: str, payload) -> None:
        text = payload if isinstance(payload, str) else json.dumps(payload)
        arts.artifact_path(name).write_text(text)

    def test_metadata_title_wins(self, arts):
        self._seed(arts, 'metadata.json', {'title': 'From metadata'})
        self._seed(arts, 'plan.json', {'title': 'From plan'})
        assert arts.read_title() == 'From metadata'

    def test_falls_back_to_plan_title(self, arts):
        self._seed(arts, 'metadata.json', {'task_id': '1'})
        self._seed(arts, 'plan.json', {'title': 'From plan'})
        assert arts.read_title() == 'From plan'

    def test_absent_is_none(self, arts):
        assert arts.read_title() is None

    @pytest.mark.parametrize('payload', ['{not json', {'title': '   '}, {'title': 7}, ['title']])
    def test_unusable_title_is_none(self, arts, payload):
        self._seed(arts, 'metadata.json', payload)
        assert arts.read_title() is None


# ---------------------------------------------------------------------------
# task 3957 step-1 — the atomicity contract of ``TaskArtifacts._write_json``,
# stated as an executable property rather than as an assertion about code.
# ---------------------------------------------------------------------------


class TestWriteJsonIsAtomic:
    """A concurrent reader never observes a partial `.task-meta` artifact.

    ``_write_json`` is the single write seam behind ~13 whole-file JSON
    artifacts (metadata, plan, review_state, reviews, verdicts,
    agent_session, reconcile_state, the report artifacts).  The contract it
    owns: a reader racing a write sees either the complete OLD document or
    the complete NEW one — never a truncated file, and never a missing one.

    The only convincing evidence for that is to race it, so these tests do.
    The harness is ported from the plan-tools suite that proved the same
    property for `_atomic_write_plan`; it is asserted here at the layer that
    now owns the guarantee, and for a NON-plan artifact too, so the contract
    is visibly `_write_json`'s rather than plan.json's alone.
    """

    READERS = 4
    READS_PER_READER = 300
    WRITES = 50

    @staticmethod
    def _race(readers, reads_per_reader, read_once, write_all):
        """Run *readers* threads calling *read_once* while *write_all* writes.

        Returns the list of values *read_once* produced.  Readers are barrier
        -synchronised with the writer so the reads genuinely overlap the
        writes rather than all landing before the first one.
        """
        observations: list = []
        errors: list = []
        start = threading.Barrier(readers + 1)
        done = threading.Event()

        def reader():
            start.wait(timeout=30)
            for _ in range(reads_per_reader):
                try:
                    observations.append(read_once())
                except Exception as exc:  # noqa: BLE001 — recorded, then asserted on
                    errors.append(repr(exc))
                if done.is_set():
                    break

        threads = [
            threading.Thread(target=reader, name=f'reader-{i}', daemon=True)
            for i in range(readers)
        ]
        for thread in threads:
            thread.start()

        start.wait(timeout=30)
        try:
            write_all()
        finally:
            done.set()

        for thread in threads:
            thread.join(timeout=60)
            assert not thread.is_alive(), f'{thread.name} did not finish'

        return observations, errors

    def test_readers_only_ever_observe_a_complete_document(
        self, artifacts: TaskArtifacts
    ):
        plan_path = artifacts.artifact_path('plan.json')

        # Two documents of DELIBERATELY different sizes: a torn read of the
        # larger one cannot be mistaken for a clean read of the smaller.
        doc_a = {'task_id': 'task-1', 'title': 'A' * 4000, 'steps': []}
        doc_b = {'task_id': 'task-1', 'title': 'B', 'analysis': 'B' * 40000,
                 'steps': []}
        # What ``write_plan`` actually lands: the input plus its schema stamp.
        expected = []
        for doc in (doc_a, doc_b):
            stamped = copy.deepcopy(doc)
            stamped['_schema_version'] = PLAN_SCHEMA_VERSION
            expected.append(stamped)

        artifacts.write_plan(copy.deepcopy(doc_a))

        def write_all():
            for i in range(self.WRITES):
                artifacts.write_plan(copy.deepcopy(doc_b if i % 2 else doc_a))

        observations, errors = self._race(
            self.READERS,
            self.READS_PER_READER,
            lambda: json.loads(plan_path.read_text()),
            write_all,
        )

        assert errors == [], (
            f'a reader observed an unparseable plan.json: {errors[:3]}'
        )
        unexpected = [o for o in observations if o not in expected]
        assert unexpected == [], (
            'a reader observed a document that is neither the pre-write nor '
            f'the post-write plan: {[sorted(u) for u in unexpected[:2]]}'
        )
        assert observations, 'the readers observed nothing at all'

    # A watcher thread can only sample the unlink→create window of a
    # hypothetical unlink-then-create writer while it actually holds the GIL,
    # and that window is a microsecond of bytecode.  Every one of these three
    # constants was set by MUTATION, not by taste: with
    # ``path.unlink(missing_ok=True)`` inserted before the write, one watcher
    # at the stock 5 ms switch interval never caught it, and three watchers at
    # 1 µs over 50 writes caught it only intermittently (1 of 3 runs).  At 500
    # writes the same mutant dies on every run with four orders of magnitude
    # of margin (~20k misses).  Do not lower ABSENCE_WRITES, raise the switch
    # interval, or drop to one watcher: each silently returns this test to
    # proving nothing at all.
    WATCHERS = 3
    ABSENCE_WRITES = 500
    SWITCH_INTERVAL = 1e-6

    def test_the_artifact_path_is_never_absent_mid_write(
        self, artifacts: TaskArtifacts
    ):
        """The swap is a replace-in-place — never unlink-then-create.

        Green against the superseded ``path.write_text`` too, which truncates
        in place and so also never removes the name; this is a forward-looking
        guard on the tmp+rename implementation, where ``os.replace`` is what
        keeps the promise and a naive ``unlink`` + create would break it.
        """
        plan_path = artifacts.artifact_path('plan.json')
        artifacts.write_plan({'task_id': 'task-1', 'steps': []})

        misses: list = []
        stop = threading.Event()

        def watcher():
            while not stop.is_set():
                if not plan_path.exists():
                    misses.append(1)

        threads = [
            threading.Thread(target=watcher, name=f'watcher-{i}', daemon=True)
            for i in range(self.WATCHERS)
        ]
        previous_interval = sys.getswitchinterval()
        sys.setswitchinterval(self.SWITCH_INTERVAL)
        for thread in threads:
            thread.start()
        try:
            for i in range(self.ABSENCE_WRITES):
                artifacts.write_plan({'task_id': 'task-1', 'title': f't{i}',
                                      'steps': []})
        finally:
            stop.set()
            for thread in threads:
                thread.join(timeout=30)
            sys.setswitchinterval(previous_interval)

        for thread in threads:
            assert not thread.is_alive(), f'{thread.name} did not finish'
        assert misses == [], 'plan.json vanished during an atomic write'

    def test_a_non_plan_artifact_is_equally_atomic(
        self, artifacts: TaskArtifacts
    ):
        """The contract belongs to `_write_json`, not to plan.json.

        Driven through ``write_verdict`` so nothing about it is
        plan-specific: if only ``write_plan`` were made atomic, this fails.
        """
        verdict_path = artifacts.artifact_path('verdicts') / 'judge.json'

        def envelope(payload: str) -> dict:
            return {
                'role': 'judge',
                'schema_version': 1,
                'session_id': 's',
                'emitted_at': 't',
                'verdict': {'complete': True, 'notes': payload},
            }

        small = envelope('n' * 4000)
        large = envelope('N' * 40000)
        expected = [small, large]

        artifacts.write_verdict('judge', copy.deepcopy(small))

        def write_all():
            for i in range(self.WRITES):
                artifacts.write_verdict(
                    'judge', copy.deepcopy(large if i % 2 else small)
                )

        observations, errors = self._race(
            self.READERS,
            self.READS_PER_READER,
            lambda: json.loads(verdict_path.read_text()),
            write_all,
        )

        assert errors == [], (
            f'a reader observed an unparseable verdicts/judge.json: {errors[:3]}'
        )
        unexpected = [o for o in observations if o not in expected]
        assert unexpected == [], (
            'a reader observed a verdict envelope that is neither the '
            f'pre-write nor the post-write one: {[sorted(u) for u in unexpected[:2]]}'
        )
        assert observations, 'the readers observed nothing at all'


class TestWriteJsonPreservesMode:
    """An artifact's permission bits survive the atomic replace.

    The superseded ``Path.write_text`` TRUNCATED an existing file, so its
    mode was never touched.  tmp+rename does not inherit that for free — the
    destination takes the TEMP's bits — so without explicit preservation the
    first write after the migration would silently re-mode every
    `.task-meta` artifact: an invisible, permanent mutation performed as a
    side effect of a routine write.  This is the semantic
    ``plan_tools._target_file_mode`` existed for, asserted at the layer that
    now owns it.
    """

    @pytest.mark.parametrize('mode', [0o600, 0o640, 0o664])
    def test_an_existing_targets_mode_survives_the_write(
        self, artifacts: TaskArtifacts, mode: int
    ):
        target = artifacts.artifact_path('plan.json')
        target.write_text('{}')
        os.chmod(target, mode)

        artifacts.write_plan({'task_id': 'task-1', 'title': 'kept', 'steps': []})

        assert stat.S_IMODE(target.stat().st_mode) == mode, (
            f'expected {mode:#o}, got '
            f'{stat.S_IMODE(target.stat().st_mode):#o} — the replace re-moded '
            'a pre-existing artifact'
        )
        # A no-op write would satisfy the mode assertion vacuously.
        assert json.loads(target.read_text())['title'] == 'kept'

    def test_a_new_targets_mode_matches_what_write_text_would_produce(
        self, artifacts: TaskArtifacts, tmp_path: Path
    ):
        """A not-yet-existing artifact gets the umask-derived mode.

        The expectation is derived from a LIVE reference file created in this
        same test rather than hard-coded to 0o644, so the assertion holds
        under any umask the suite happens to run with.
        """
        reference = tmp_path / 'reference.json'
        reference.write_text('{}')

        target = artifacts.artifact_path('plan.json')
        assert not target.exists()

        artifacts.write_plan({'task_id': 'task-1', 'title': 'fresh', 'steps': []})

        assert stat.S_IMODE(target.stat().st_mode) == stat.S_IMODE(
            reference.stat().st_mode
        ), (
            'a newly created artifact must land with the same bits '
            'Path.write_text would have given it'
        )
        assert json.loads(target.read_text())['title'] == 'fresh'

    def test_a_non_missing_stat_failure_propagates(
        self, artifacts: TaskArtifacts, monkeypatch
    ):
        """Only the missing-file case is benign.

        The mode lookup must swallow FileNotFoundError ONLY — an EACCES /
        ELOOP / ENAMETOOLONG on the target's ``stat()`` is a genuine failure
        and must surface, not be misread as "this file is new, use the
        umask".  Ports ``test_target_file_mode_propagates_a_non_missing_file_os_error``
        from the plan-tools suite, whose whole point this is.
        """
        target = artifacts.artifact_path('plan.json')
        real_stat = Path.stat

        # Scoped to *target*, falling back to the real stat() for every other
        # path: a blanket raise would break unrelated machinery (pytest's own
        # bookkeeping calls Path.stat via Path.exists).
        def boom(self, *args, **kwargs):
            if self == target:
                raise OSError(errno.EACCES, 'Permission denied')
            return real_stat(self, *args, **kwargs)

        monkeypatch.setattr(Path, 'stat', boom)

        with pytest.raises(OSError) as excinfo:
            artifacts.write_plan({'task_id': 'task-1', 'steps': []})

        assert excinfo.value.errno == errno.EACCES


class TestWriteJsonFollowsSymlinks:
    """A write through the lane symlink reaches the REAL file, never eats it.

    `ensure_lane_symlinks` makes `<worktree>/.task/plan.json` an absolute
    symlink onto the meta-root plan, and `_artifacts_from_args` still supports
    the legacy `meta_root=None` shape rooted at `<worktree>/.task`
    (mcp_lifecycle's `meta_root` parameter defaults to None) — so
    `artifact_path('plan.json')` can BE that symlink.

    `Path.write_text` followed the link.  A bare `os.replace` onto the link
    path would swap the LINK for a regular file, silently re-forking the lane
    and meta-root copies: the esc-5205-9 stale-plan divergence the symlink
    exists to make impossible.  So `_write_json` must resolve first — which
    also puts the temp on the same filesystem as the real file, making the
    replace an intra-filesystem rename rather than an EXDEV error.
    """

    @pytest.fixture
    def lane_and_meta(self, worktree: Path) -> tuple[TaskArtifacts, Path, Path]:
        """A legacy-mode `TaskArtifacts` whose plan path IS the lane symlink.

        Returns ``(legacy_artifacts, lane_plan, real_plan)``.
        """
        worktree.mkdir()
        meta_root = TaskArtifacts.meta_root_for(worktree.parent, worktree.name)
        meta = TaskArtifacts(worktree, meta_root)
        meta.init('task-1', 'Test Task', 'Desc')
        meta.ensure_lane_symlinks()

        lane_plan = worktree / '.task' / 'plan.json'
        assert lane_plan.is_symlink()

        # The legacy view is rooted at <worktree>/.task, so write_plan's
        # target is the symlink itself.
        legacy = TaskArtifacts(worktree)
        assert legacy.lane_root == legacy.task_root == worktree / '.task'
        return legacy, lane_plan, meta_root / 'plan.json'

    def test_symlink_survives_and_the_real_file_receives_the_write(
        self, lane_and_meta
    ):
        legacy, lane_plan, real_plan = lane_and_meta
        link_target_before = os.readlink(lane_plan)

        legacy.write_plan({'task_id': 'task-1', 'title': 'written', 'steps': []})

        assert lane_plan.is_symlink(), 'the lane symlink was replaced by a file'
        assert os.readlink(lane_plan) == link_target_before
        assert json.loads(real_plan.read_text())['title'] == 'written'
        # And the lane path still resolves to the same single source of truth.
        assert json.loads(lane_plan.read_text())['title'] == 'written'

    def test_the_replace_targets_the_real_file_not_the_link(
        self, lane_and_meta, monkeypatch
    ):
        """The mechanism, not a side effect.

        Resolving first is what keeps the rename intra-filesystem (the lane
        and the meta root need not share a filesystem) AND what stops the
        replace from eating the link.  Spying on `os.replace` proves the temp
        was created beside the RESOLVED file rather than beside the symlink.
        """
        legacy, lane_plan, real_plan = lane_and_meta
        seen: list[tuple[Path, Path]] = []
        real_replace = safe_io.os.replace

        def spy(src, dst, *args, **kwargs):
            seen.append((Path(src), Path(dst)))
            return real_replace(src, dst, *args, **kwargs)

        monkeypatch.setattr(safe_io.os, 'replace', spy)

        legacy.write_plan({'task_id': 'task-1', 'steps': []})

        assert seen, 'no atomic replace was performed at all'
        src, dst = seen[-1]
        assert dst == real_plan, (
            f'the replace targeted {dst} — it must target the resolved file, '
            'not the lane symlink'
        )
        assert src.parent == real_plan.parent, (
            f'the temp was created in {src.parent}, not beside the real file'
        )

    @staticmethod
    def _residue(directory: Path) -> list[str]:
        """Whatever in *directory* looks like an unswept atomic-write temp.

        ``safe_io._create_temp`` names its temp ``.<dest>.<uuid4hex>.tmp``, so
        a ``.tmp`` suffix is the shape to look for; the same arm also catches
        the fixed ``<dest>.tmp`` name a hand-rolled writer would leave behind.

        A residue PREDICATE rather than an exact directory listing on
        purpose.  An exact listing couples this test to the complete set of
        artifacts ``init()`` happens to create today, so adding any new
        init-time artifact (a base_commit sidecar, a lock file, …) would fail
        a test whose message points at temp residue when nothing about
        residue changed.
        """
        return sorted(
            entry.name for entry in directory.iterdir()
            if entry.name.endswith('.tmp')
        )

    def test_no_temp_residue_in_either_directory(self, lane_and_meta):
        legacy, lane_plan, real_plan = lane_and_meta

        legacy.write_plan({'task_id': 'task-1', 'steps': []})

        assert self._residue(lane_plan.parent) == []
        assert self._residue(real_plan.parent) == []
        # Not vacuous: the swept-clean directories are the ones the write
        # actually went through and landed in.
        assert {'plan.json'} <= {p.name for p in lane_plan.parent.iterdir()}
        assert {'plan.json'} <= {p.name for p in real_plan.parent.iterdir()}
        assert json.loads(real_plan.read_text())['task_id'] == 'task-1'

    def test_dangling_symlink_fails_loudly_without_materialising_a_file(
        self, worktree: Path
    ):
        """A broken link must NOT quietly become a stray regular file.

        Materialising one at the link path IS the divergence, not a recovery
        from it: the lane would hold a plan the meta root has never seen.
        """
        lane_task = worktree / '.task'
        lane_task.mkdir(parents=True)
        lane_plan = lane_task / 'plan.json'
        missing = worktree.parent / 'gone' / 'plan.json'
        os.symlink(missing, lane_plan)
        assert lane_plan.is_symlink() and not lane_plan.exists()

        legacy = TaskArtifacts(worktree)
        with pytest.raises(ArtifactWriteError) as excinfo:
            legacy.write_plan({'task_id': 'task-1', 'steps': []})

        message = str(excinfo.value)
        assert str(lane_plan) in message, 'the ORIGINAL path must be named'
        assert str(missing) in message, 'the RESOLVED path must be named too'
        assert lane_plan.is_symlink()
        assert not lane_plan.exists(), 'a stray regular file was materialised'
        assert {p.name for p in lane_task.iterdir()} == {'plan.json'}

    def test_an_ordinary_path_with_a_missing_parent_is_still_created(
        self, artifacts: TaskArtifacts
    ):
        """The dangling check must be gated on `is_symlink()`.

        For an ordinary path a missing parent is NOT an error — `_write_json`
        creates the artifact's own directory with `Path.mkdir`, which is how
        `reviews/` and `verdicts/` come into being under a live root.  An
        unconditional parent check would break artifact creation outright.
        """
        import shutil
        shutil.rmtree(artifacts.artifact_path('verdicts'))

        artifacts.write_verdict('judge', {
            'role': 'judge', 'schema_version': 1, 'session_id': 's',
            'emitted_at': 't', 'verdict': {'complete': True},
        })

        assert artifacts.read_verdict('judge') is not None


# ---------------------------------------------------------------------------
# task 3957 amendment — the DELEGATION contract of ``_write_json``.
#
# Everything else in this group asserts an observable property (torn reads,
# mode, symlink write-through).  The kwarg below has no observable property
# to assert against on a live filesystem: dropping ``fsync=True`` leaves the
# whole orchestrator suite green while silently retiring the durability half
# of the docstring's contract.  So it is pinned directly.
# ---------------------------------------------------------------------------


class TestWriteJsonDelegatesToSharedSafeIo:
    """`_write_json` calls the blessed writer, with the kwargs it promises.

    ``fsync=True`` is a deliberate override of ``shared.safe_io``'s own
    ``fsync=False`` default (see ``_write_json``'s docstring for the measured
    ~5 ms/write it costs and the three reasons the artifacts pay it).  A
    deliberate override needs a pin, or the next reader who measures the cost
    silently deletes the kwarg and nothing anywhere goes red.
    """

    @staticmethod
    def _spy(monkeypatch) -> list[dict]:
        """Record every ``atomic_write_text`` call, then perform it for real.

        Recording without performing would make the surrounding write a no-op
        and let a `_write_json` that never wrote at all pass.
        """
        calls: list[dict] = []
        real = safe_io.atomic_write_text

        def spy(path, text, **kwargs):
            # ``kwargs`` verbatim under a nested key, so a DROPPED kwarg reads
            # back as None and trips the assertion below rather than dying
            # with a bare KeyError at the exact seam this test explains.
            calls.append({'path': Path(path), 'text': text, 'kwargs': kwargs})
            return real(path, text, **kwargs)

        monkeypatch.setattr(safe_io, 'atomic_write_text', spy)
        return calls

    def test_the_plan_write_is_fsynced(
        self, artifacts: TaskArtifacts, monkeypatch
    ):
        calls = self._spy(monkeypatch)

        artifacts.write_plan({'task_id': 'task-1', 'title': 'p', 'steps': []})

        assert len(calls) == 1, (
            f'expected exactly one delegated write, saw {len(calls)} — '
            '_write_json must not re-implement or double-write'
        )
        call = calls[0]
        assert call['path'] == artifacts.artifact_path('plan.json')
        assert call['kwargs'].get('fsync') is True, (
            "fsync=True is _write_json's stated durability contract, and "
            'safe_io defaults it to False — so dropping or flipping the '
            f'kwarg silently retires the guarantee.  Saw: {call["kwargs"]}'
        )
        # The write really happened — the spy is not standing in for it.
        assert json.loads(artifacts.artifact_path('plan.json').read_text())['title'] == 'p'

    def test_a_nested_artifact_delegates_on_the_same_terms(
        self, artifacts: TaskArtifacts, monkeypatch
    ):
        """The contract belongs to the seam, not to ``write_plan``."""
        import shutil
        shutil.rmtree(artifacts.artifact_path('verdicts'))
        calls = self._spy(monkeypatch)

        artifacts.write_verdict('judge', {
            'role': 'judge', 'schema_version': 1, 'session_id': 's',
            'emitted_at': 't', 'verdict': {'complete': True},
        })

        assert [c['kwargs'].get('fsync') for c in calls] == [True]
        assert calls[0]['path'] == artifacts.artifact_path('verdicts') / 'judge.json'
        assert artifacts.read_verdict('judge') is not None
