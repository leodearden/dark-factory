"""Tests for the verdict-tools MCP server."""

from __future__ import annotations

import pytest

from orchestrator.agents.roles import ALL_REVIEWERS
from orchestrator.artifacts import TaskArtifacts
from orchestrator.mcp.verdict_tools import (
    _SINGLETON_ROLE_TOOLS,
    _artifacts_from_args,
    _submit_completion_verdict,
    _submit_merge_disposition,
    _submit_review_verdict,
    _submit_triage,
    create_server,
)


@pytest.fixture()
def artifacts(tmp_path):
    """TaskArtifacts pointing at a temporary worktree."""
    a = TaskArtifacts(tmp_path)
    a.init('test-1', 'Test task', 'A test')
    return a


# ---------------------------------------------------------------------------
# Standalone _impl tests (independent of the MCP tool/pydantic boundary)
# ---------------------------------------------------------------------------


class TestSubmitReviewVerdict:
    def test_writes_envelope_and_returns_ok(self, artifacts):
        result = _submit_review_verdict(
            artifacts,
            role='test_analyst',
            session_id='sess-1',
            reviewer='test_analyst',
            verdict='PASS',
            issues=[],
            summary='ok',
        )
        assert result == {'status': 'ok', 'role': 'test_analyst'}

        envelope = artifacts.read_verdict('test_analyst')
        assert envelope is not None
        assert envelope['role'] == 'test_analyst'
        assert envelope['schema_version'] == 1
        assert envelope['session_id'] == 'sess-1'
        assert isinstance(envelope['emitted_at'], str) and envelope['emitted_at']
        assert envelope['verdict'] == {
            'reviewer': 'test_analyst',
            'verdict': 'PASS',
            'issues': [],
            'summary': 'ok',
        }

    def test_invalid_verdict_returns_error_and_writes_nothing(self, artifacts):
        result = _submit_review_verdict(
            artifacts,
            role='test_analyst',
            session_id='s',
            reviewer='test_analyst',
            verdict='GARBAGE',
            issues=[],
            summary='x',
        )
        assert result['status'] == 'error'
        assert 'PASS' in result['message']
        assert 'ISSUES_FOUND' in result['message']

        assert artifacts.read_verdict('test_analyst') is None

    def test_reviewer_role_mismatch_returns_error_and_writes_nothing(self, artifacts):
        """A reviewer cannot misname its payload onto a sibling's role —
        ``reviewer`` must match the authoritative ``--verdict-role``.
        """
        result = _submit_review_verdict(
            artifacts,
            role='test_analyst',
            session_id='s',
            reviewer='security',
            verdict='PASS',
            issues=[],
            summary='x',
        )
        assert result['status'] == 'error'
        assert 'test_analyst' in result['message']
        assert 'security' in result['message']

        assert artifacts.read_verdict('test_analyst') is None
        assert artifacts.read_verdict('security') is None


class TestSubmitCompletionVerdict:
    def test_writes_envelope_and_returns_ok(self, artifacts):
        result = _submit_completion_verdict(
            artifacts,
            role='judge',
            session_id='s',
            complete=True,
            reasoning='done',
            uncovered_plan_steps=[],
            substantive_work=True,
        )
        assert result == {'status': 'ok', 'role': 'judge'}

        envelope = artifacts.read_verdict('judge')
        assert envelope is not None
        assert envelope['role'] == 'judge'
        assert envelope['schema_version'] == 1
        assert envelope['verdict'] == {
            'complete': True,
            'reasoning': 'done',
            'uncovered_plan_steps': [],
            'substantive_work': True,
        }


class TestSubmitTriage:
    def test_writes_envelope_and_returns_ok(self, artifacts):
        accepted = [{
            'index': 0,
            'suggestion': 'x',
            'reason': 'r',
            'files': ['a.py'],
            'proposed_task_title': 't',
        }]
        proposed_task_groups = [{
            'title': 'g',
            'description': 'd',
            'accepted_indices': [0],
        }]
        result = _submit_triage(
            artifacts,
            role='triage',
            session_id='s',
            accepted=accepted,
            skipped=[],
            proposed_task_groups=proposed_task_groups,
        )
        assert result == {'status': 'ok', 'role': 'triage'}

        envelope = artifacts.read_verdict('triage')
        assert envelope is not None
        assert envelope['role'] == 'triage'
        assert envelope['verdict'] == {
            'accepted': accepted,
            'skipped': [],
            'proposed_task_groups': proposed_task_groups,
        }


class TestSubmitMergeDisposition:
    def test_writes_envelope_not_blocked(self, artifacts):
        result = _submit_merge_disposition(
            artifacts,
            role='merger',
            session_id='s',
            blocked=False,
            reason='',
        )
        assert result == {'status': 'ok', 'role': 'merger'}

        envelope = artifacts.read_verdict('merger')
        assert envelope is not None
        assert envelope['role'] == 'merger'
        assert envelope['verdict'] == {'blocked': False, 'reason': ''}

    def test_writes_envelope_blocked_with_reason(self, artifacts):
        result = _submit_merge_disposition(
            artifacts,
            role='merger',
            session_id='s',
            blocked=True,
            reason='conflict',
        )
        assert result == {'status': 'ok', 'role': 'merger'}

        envelope = artifacts.read_verdict('merger')
        assert envelope is not None
        assert envelope['role'] == 'merger'
        assert envelope['verdict'] == {'blocked': True, 'reason': 'conflict'}


# ---------------------------------------------------------------------------
# create_server tool-selection + async MCP tool.run boundary tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCreateServer:
    """``create_server(artifacts, role)`` registers EXACTLY ONE tool,
    selected by role: judge/triage/merger are singleton roles; any other
    role string is treated as a reviewer-panel member and gets
    ``submit_review_verdict``.
    """

    @pytest.mark.parametrize(
        ('role', 'expected_tool'),
        [
            ('judge', 'submit_completion_verdict'),
            ('triage', 'submit_triage'),
            ('merger', 'submit_merge_disposition'),
            ('test_analyst', 'submit_review_verdict'),
        ],
    )
    async def test_registers_exactly_one_tool_for_role(
        self, artifacts, role, expected_tool
    ):
        server = create_server(artifacts, role)
        tools = await server.list_tools()
        assert sorted(tool.name for tool in tools) == [expected_tool]

    async def test_merger_tool_run_writes_envelope_via_wrapper(self, artifacts):
        server = create_server(artifacts, 'merger')
        tool = await server.get_tool('submit_merge_disposition')
        assert tool is not None

        await tool.run({'blocked': True, 'reason': 'x'})

        envelope = artifacts.read_verdict('merger')
        assert envelope is not None
        assert envelope['role'] == 'merger'
        assert envelope['verdict'] == {'blocked': True, 'reason': 'x'}


class TestSingletonRoleToolsDisjointFromReviewerNames:
    """``create_server`` decides tool selection purely from
    ``_SINGLETON_ROLE_TOOLS`` vs. "anything else" — it does not import
    roles.py, by design (see the comment above ``_SINGLETON_ROLE_TOOLS``).
    That decoupling means a future rename in roles.py could silently give a
    reviewer-panel member one of the three reserved names, and
    ``create_server`` would register the singleton tool for it instead of
    ``submit_review_verdict`` — the ``assert role not in _SINGLETON_ROLE_TOOLS``
    in create_server's ``else`` branch would then simply never fire for that
    role. This test is the guard: it fails loudly at test time instead.
    """

    def test_reviewer_names_disjoint_from_singleton_roles(self):
        reviewer_names = {reviewer.name for reviewer in ALL_REVIEWERS}
        collisions = reviewer_names & _SINGLETON_ROLE_TOOLS
        assert not collisions, (
            f'reviewer-panel name(s) {collisions} collide with '
            f'_SINGLETON_ROLE_TOOLS {set(_SINGLETON_ROLE_TOOLS)}; '
            'create_server would silently misroute submit_review_verdict '
            'for them'
        )


# ---------------------------------------------------------------------------
# _artifacts_from_args parity tests (mirrors plan_tools.TestArtifactsFromArgs)
# ---------------------------------------------------------------------------


class TestArtifactsFromArgs:
    """``_artifacts_from_args`` parses ``--worktree``/``--meta-root`` into a
    ``TaskArtifacts`` — the same construction shape as plan_tools' own
    ``_artifacts_from_args`` — plus the verdict-specific ``--verdict-role``
    and ``--session-id`` flags.
    """

    def test_meta_root_flag_produces_relocated_root(self, tmp_path):
        wt = tmp_path / 'wt'
        wt.mkdir()
        mr = tmp_path / 'base' / '.task-meta' / 'wt'
        mr.mkdir(parents=True)

        arts, role, sid = _artifacts_from_args([
            '--worktree', str(wt),
            '--meta-root', str(mr),
            '--verdict-role', 'judge',
            '--session-id', 'sess-9',
        ])

        assert arts.root == mr
        assert arts.worktree == wt
        assert role == 'judge'
        assert sid == 'sess-9'

    def test_omitted_meta_root_falls_back_to_legacy_task_dir(self, tmp_path):
        wt = tmp_path / 'wt'
        wt.mkdir()
        (wt / '.task').mkdir()

        arts, role, sid = _artifacts_from_args([
            '--worktree', str(wt),
            '--verdict-role', 'merger',
        ])

        assert arts.root == wt / '.task'
        assert arts.worktree == wt
        assert role == 'merger'

    def test_omitted_session_id_defaults_empty(self, tmp_path):
        wt = tmp_path / 'wt'
        wt.mkdir()
        (wt / '.task').mkdir()

        arts, role, sid = _artifacts_from_args([
            '--worktree', str(wt),
            '--verdict-role', 'merger',
        ])

        assert sid == ''

    def test_invalid_verdict_role_exits_at_startup(self, tmp_path):
        """An unsafe ``--verdict-role`` must abort server startup (fail
        fast) rather than construct a ``TaskArtifacts`` that would only
        raise ``ValueError`` deep inside the first tool call.
        """
        wt = tmp_path / 'wt'
        wt.mkdir()
        (wt / '.task').mkdir()

        with pytest.raises(SystemExit):
            _artifacts_from_args([
                '--worktree', str(wt),
                '--verdict-role', '../escape',
            ])
