"""Tests for the verdict-tools MCP server."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from _review_issue_fixtures import contract_issue, contract_issue_without
from fastmcp import Client, FastMCP
from shared.mcp_markup_middleware import OUTCOMES as MARKUP_OUTCOMES

from orchestrator.agents.roles import ALL_REVIEWERS
from orchestrator.artifacts import TaskArtifacts
from orchestrator.mcp import markup_journal, verdict_tools
from orchestrator.mcp.verdict_tools import (
    _SINGLETON_ROLE_TOOLS,
    IssueIntakeOutcome,
    _artifacts_from_args,
    _submit_completion_verdict,
    _submit_merge_disposition,
    _submit_review_verdict,
    _submit_triage,
    create_server,
)
from orchestrator.review_issue_contract import IssueField


@pytest.fixture()
def artifacts(tmp_path):
    """TaskArtifacts pointing at a temporary worktree."""
    a = TaskArtifacts(tmp_path)
    a.init('test-1', 'Test task', 'A test')
    return a


_NO_ACTION_NOTE = {
    'escalation_id': 'esc-42-1', 'disposition': 'no_action', 'rationale': 'pure observation',
}


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

    def test_the_sync_seam_applies_the_issue_intake(self, artifacts):
        """Test doubles (_workflow_helpers.AgentStub) write verdicts through
        this seam rather than the MCP tool, so it must run the intake itself.
        """
        result = _submit_review_verdict(
            artifacts,
            role='test_analyst',
            session_id='s',
            reviewer='test_analyst',
            verdict='ISSUES_FOUND',
            issues=[contract_issue(severity='high')],
            summary='x',
        )

        assert result['status'] == 'ok'
        assert result['issue_contract']['outcome'] == IssueIntakeOutcome.NORMALISED
        envelope = artifacts.read_verdict('test_analyst')
        assert envelope is not None
        assert envelope['verdict']['issues'] == [contract_issue(severity='suggestion')]

    def test_note_dispositions_are_written_into_the_payload(self, artifacts):
        result = _submit_review_verdict(
            artifacts,
            role='test_analyst',
            session_id='s',
            reviewer='test_analyst',
            verdict='PASS',
            issues=[],
            summary='ok',
            note_dispositions=[dict(_NO_ACTION_NOTE)],
        )
        assert result == {'status': 'ok', 'role': 'test_analyst'}

        envelope = artifacts.read_verdict('test_analyst')
        assert envelope is not None
        assert envelope['verdict']['note_dispositions'] == [_NO_ACTION_NOTE]

    @pytest.mark.parametrize(
        ('note_dispositions', 'named', 'located'),
        [
            pytest.param(
                [
                    dict(_NO_ACTION_NOTE),
                    {**_NO_ACTION_NOTE, 'escalation_id': 'esc-42-2', 'disposition': 'fixed'},
                ],
                "'fixed'",
                'note_dispositions entry 1 ',
                id='unknown-disposition',
            ),
            pytest.param(
                [dict(_NO_ACTION_NOTE), {**_NO_ACTION_NOTE, 'disposition': 'work'}],
                "'esc-42-1'",
                'note_dispositions entries 0 and 1 ',
                id='duplicated-id',
            ),
        ],
    )
    def test_malformed_note_dispositions_return_error_and_write_nothing(
        self, artifacts, note_dispositions, named, located,
    ):
        result = _submit_review_verdict(
            artifacts,
            role='test_analyst',
            session_id='s',
            reviewer='test_analyst',
            verdict='PASS',
            issues=[],
            summary='x',
            note_dispositions=note_dispositions,
        )
        assert result['status'] == 'error'
        assert named in result['message']
        assert located in result['message']

        assert artifacts.read_verdict('test_analyst') is None


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

    async def test_reviewer_tool_run_writes_note_dispositions(self, artifacts):
        server = create_server(artifacts, 'test_analyst')
        tool = await server.get_tool('submit_review_verdict')
        assert tool is not None

        await tool.run({
            'reviewer': 'test_analyst',
            'verdict': 'PASS',
            'issues': [],
            'summary': 's',
            'note_dispositions': [dict(_NO_ACTION_NOTE)],
        })

        envelope = artifacts.read_verdict('test_analyst')
        assert envelope is not None
        assert envelope['verdict']['note_dispositions'] == [_NO_ACTION_NOTE]

    async def test_reviewer_tool_note_dispositions_is_optional(self, artifacts):
        server = create_server(artifacts, 'test_analyst')
        tool = await server.get_tool('submit_review_verdict')
        assert tool is not None

        assert 'note_dispositions' in tool.parameters['properties']
        assert 'note_dispositions' not in tool.parameters['required']


def _journal_lines(root: Path) -> list[dict[str, Any]]:
    """Every verdict-tools fact journalled under *root*; an absent file is no lines."""
    path = markup_journal.journal_path(root, verdict_tools.MARKUP_SINK_SPEC.server_label)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


async def _call_review_tool(
    artifacts: TaskArtifacts,
    issues: list[Any],
    *,
    journal_root: Path | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Call the registered reviewer tool through an MCP client, as an agent does."""
    artifacts.write_plan({
        'task_id': 'test-1', 'title': 'T', 'analysis': 'A',
        'prerequisites': [], 'steps': [],
    })
    server = create_server(
        artifacts, 'test_analyst', resolve_project_root=lambda _wt: journal_root,
    )
    arguments = {
        'reviewer': 'test_analyst',
        'verdict': 'ISSUES_FOUND',
        'issues': issues,
        'summary': 'x',
        **overrides,
    }
    async with Client(server) as client:
        result = await client.call_tool('submit_review_verdict', arguments)
    return result.data


@pytest.mark.asyncio
class TestSubmitReviewVerdictIssueIntake:
    """The reviewer tool checks each issue against the reviewer issue contract
    (orchestrator/review_issue_contract.py): off-contract fields are
    normalised and reported back; only an unusable issue refuses the call.
    """

    async def test_off_contract_fields_are_normalised_and_reported(self, artifacts):
        high = contract_issue(severity='high', title='kept')
        file_and_line = contract_issue_without(
            IssueField.LOCATION, file='pkg/mod.py', line=12,
        )

        result = await _call_review_tool(artifacts, [high, file_and_line])

        assert result['status'] == 'ok'
        assert result['role'] == 'test_analyst'
        report = result['issue_contract']
        assert report['outcome'] == IssueIntakeOutcome.NORMALISED
        assert [d.split(':')[0] for d in report['deviations']] == [
            'issues[0].severity', 'issues[1].location',
        ]
        assert 'submit_review_verdict' in result['message']
        assert 'again' in result['message']

        envelope = artifacts.read_verdict('test_analyst')
        assert envelope is not None
        assert envelope['verdict']['issues'] == [
            {**high, 'severity': 'suggestion'},
            {**file_and_line, 'location': 'pkg/mod.py:12'},
        ]

    async def test_a_description_taken_from_the_finding_text_lands(self, artifacts):
        issue = contract_issue_without(
            IssueField.DESCRIPTION, title='a heading', detail='the body',
        )

        result = await _call_review_tool(artifacts, [issue])

        assert result['status'] == 'ok'
        envelope = artifacts.read_verdict('test_analyst')
        assert envelope is not None
        assert envelope['verdict']['issues'] == [{**issue, 'description': 'the body'}]

    async def test_issue_without_description_is_refused_and_nothing_written(self, artifacts):
        result = await _call_review_tool(
            artifacts, [contract_issue(), contract_issue_without(IssueField.DESCRIPTION)],
        )

        assert result['status'] == 'error'
        assert 'issues[1].description' in result['message']
        assert 'nothing was written' in result['message'].lower()
        assert result['issue_contract']['outcome'] == IssueIntakeOutcome.REFUSED
        assert artifacts.read_verdict('test_analyst') is None

    async def test_issue_without_severity_is_refused_naming_it(self, artifacts):
        result = await _call_review_tool(
            artifacts, [contract_issue_without(IssueField.SEVERITY)],
        )

        assert result['status'] == 'error'
        assert 'issues[0].severity' in result['message']
        assert result['issue_contract']['outcome'] == IssueIntakeOutcome.REFUSED
        assert artifacts.read_verdict('test_analyst') is None

    async def test_off_vocabulary_severity_alone_still_lands_the_verdict(self, artifacts):
        """INV-6: a repairable issue must not strand the review gate."""
        result = await _call_review_tool(artifacts, [contract_issue(severity='major')])

        assert result['status'] == 'ok'
        assert artifacts.read_verdict('test_analyst') is not None

    async def test_on_contract_issues_add_no_keys_to_the_ack(self, artifacts):
        issues = [contract_issue(), contract_issue(severity='suggestion')]

        result = await _call_review_tool(artifacts, issues)

        assert result == {'status': 'ok', 'role': 'test_analyst'}
        envelope = artifacts.read_verdict('test_analyst')
        assert envelope is not None
        assert envelope['verdict']['issues'] == issues

    @pytest.mark.parametrize(
        ('overrides', 'named'),
        [
            ({'reviewer': 'security'}, ('security', 'test_analyst')),
            ({'verdict': 'GARBAGE'}, ('PASS', 'ISSUES_FOUND')),
        ],
    )
    async def test_identity_and_verdict_checks_run_before_the_issue_check(
        self, artifacts, overrides, named,
    ):
        result = await _call_review_tool(
            artifacts, [contract_issue_without(IssueField.DESCRIPTION)], **overrides,
        )

        assert result['status'] == 'error'
        for name in named:
            assert name in result['message']
        assert 'issues[' not in result['message']
        assert 'issue_contract' not in result
        assert artifacts.read_verdict('test_analyst') is None


@pytest.mark.asyncio
class TestIssueContractJournal:
    """The registered tool journals every off-contract intake as one
    ``review_issue_off_contract`` fact in the verdict-tools fact journal,
    the durable record the markup guard's facts already land in.
    """

    async def test_a_normalised_intake_is_journalled_once(self, artifacts, tmp_path):
        result = await _call_review_tool(
            artifacts, [contract_issue(severity='high')], journal_root=tmp_path,
        )

        assert result['status'] == 'ok'
        [line] = _journal_lines(tmp_path)
        assert line['fact'] == verdict_tools.REVIEW_ISSUE_OFF_CONTRACT_FACT
        assert (line['tool'], line['param'], line['outcome'], line['role']) == (
            'submit_review_verdict', 'issues', IssueIntakeOutcome.NORMALISED, 'test_analyst',
        )
        assert line['subject_task_id'] == 'test-1'
        assert any(d.startswith('issues[0].severity') for d in line['deviations'])

    async def test_a_refused_intake_is_journalled_once(self, artifacts, tmp_path):
        result = await _call_review_tool(
            artifacts, [contract_issue_without(IssueField.DESCRIPTION)], journal_root=tmp_path,
        )

        assert result['status'] == 'error'
        [line] = _journal_lines(tmp_path)
        assert line['outcome'] == IssueIntakeOutcome.REFUSED
        assert any(d.startswith('issues[0].description') for d in line['deviations'])

    async def test_a_clean_intake_writes_no_journal_line(self, artifacts, tmp_path):
        result = await _call_review_tool(artifacts, [contract_issue()], journal_root=tmp_path)

        assert result == {'status': 'ok', 'role': 'test_analyst'}
        assert not markup_journal.journal_path(
            tmp_path, verdict_tools.MARKUP_SINK_SPEC.server_label,
        ).exists()

    async def test_an_unresolvable_journal_root_does_not_break_the_call(self, artifacts):
        result = await _call_review_tool(artifacts, [contract_issue(severity='high')])

        assert result['status'] == 'ok'
        assert artifacts.read_verdict('test_analyst') is not None


def test_intake_outcomes_tell_a_floored_journal_line_from_a_markup_fact():
    """``fact`` is not a journal floor key, but ``outcome`` is."""
    assert 'outcome' in markup_journal.MARKUP_JOURNAL_FLOOR_KEYS
    assert not {str(outcome) for outcome in IssueIntakeOutcome} & set(MARKUP_OUTCOMES)


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

        assert arts.lane_root == arts.task_root == mr
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

        assert arts.lane_root == arts.task_root == wt / '.task'
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


def _dual_root_argv(tmp_path, task_root, role):
    wt = tmp_path / 'wt'
    wt.mkdir()
    lane = tmp_path / '.task-meta' / 'wt'
    lane.mkdir(parents=True)
    return [
        '--worktree', str(wt), '--meta-root', str(lane),
        '--task-root', str(task_root), '--verdict-role', role,
    ]


class TestArtifactsFromArgsTaskRoot:
    """``--task-root`` (task 3155) points the verdicts at the task-lifetime
    root, where the workflow reads them."""

    @pytest.mark.asyncio
    async def test_submitted_verdict_lands_in_task_root(self, tmp_path):
        task = tmp_path / '.task-meta' / 'task-3155'
        (task / 'verdicts').mkdir(parents=True)
        arts, role, _ = _artifacts_from_args(_dual_root_argv(tmp_path, task, 'merger'))
        assert arts.task_root == task.resolve()

        tool = await create_server(arts, role).get_tool('submit_merge_disposition')
        assert tool is not None
        await tool.run({'blocked': False, 'reason': ''})

        assert (task / 'verdicts' / 'merger.json').is_file()
        assert not (arts.lane_root / 'verdicts').exists()

    def test_missing_task_root_exits_naming_it(self, tmp_path, capsys):
        task = tmp_path / '.task-meta' / 'task-absent'

        with pytest.raises(SystemExit) as exc:
            _artifacts_from_args(_dual_root_argv(tmp_path, task, 'judge'))

        assert exc.value.code == 1
        assert str(task.resolve()) in capsys.readouterr().err


# ---------------------------------------------------------------------------
# main() suppresses the FastMCP startup banner + PyPI update-check (task 2942).
#
# Same root cause as plan_tools.main(): FastMCP 3.x's banner rendering fires a
# synchronous PyPI update-check (check_for_newer_version -> httpx.get) on the
# stdio startup path BEFORE serving, which can delay the MCP initialize
# handshake past the CLI's MCP_TIMEOUT and get this server silently dropped
# for reviewer/judge/merger verdict submission. This pins that main() passes
# show_banner=False. Mirrors test_plan_tools_server.TestMainSuppressesBanner.
# ---------------------------------------------------------------------------


class TestMainSuppressesBanner:
    """verdict_tools.main() calls server.run(transport='stdio', show_banner=False)."""

    def test_main_runs_stdio_with_banner_disabled(self, tmp_path, monkeypatch):
        wt = tmp_path / 'wt'
        wt.mkdir()
        mr = tmp_path / 'base' / '.task-meta' / 'wt'
        mr.mkdir(parents=True)

        monkeypatch.setattr(
            sys, 'argv',
            [
                'verdict_tools',
                '--worktree', str(wt),
                '--meta-root', str(mr),
                '--verdict-role', 'judge',
            ],
        )

        captured: dict[str, Any] = {}

        def recorder(self, *args, **kwargs):
            captured['args'] = args
            captured['kwargs'] = kwargs
            return None

        # Class-level patch: create_server builds a real FastMCP instance, so
        # patching the class method intercepts its .run() without serving.
        monkeypatch.setattr(FastMCP, 'run', recorder)

        verdict_tools.main()

        kwargs = captured.get('kwargs', {})
        args = captured.get('args', ())
        transport = kwargs.get('transport', args[0] if args else None)
        assert transport == 'stdio'
        assert kwargs.get('show_banner') is False
