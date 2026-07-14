"""Smoke tests for the _workflow_helpers module.

Verifies that FakeScheduler, FakeMcp, and FakeBriefing behave as expected.
Import-contract verification for _make_resolving_steward and
_make_status_setting_steward relies on indirect coverage from
test_workflow_e2e.py and test_workflow_status_on_resume.py.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fake_scheduler_smoke() -> None:
    """FakeScheduler tracks status changes correctly."""
    from _workflow_helpers import FakeScheduler  # noqa: PLC0415

    sched = FakeScheduler()
    await sched.set_task_status('x', 'pending')
    await sched.set_task_status('x', 'done')
    assert await sched.get_status('x') == 'done'
    assert sched.statuses == {'x': ['pending', 'done']}


def test_fake_mcp_smoke() -> None:
    """FakeMcp returns the expected URL and empty MCP config."""
    from _workflow_helpers import FakeMcp  # noqa: PLC0415

    mcp = FakeMcp()
    assert mcp.url == 'http://localhost:9999'
    assert mcp.mcp_config_json() == {'mcpServers': {}}


@pytest.mark.asyncio
async def test_fake_briefing_smoke() -> None:
    """FakeBriefing returns a non-empty string for each of the 7 prompt-builder methods.

    Asserts shape (non-empty str) and that each method threads its key input
    argument into the output.  Literal wording tokens (e.g. 'Plan task:') are
    not checked — only the caller-supplied argument value must appear in the
    result, guarding against a builder silently dropping its argument.
    """
    from _workflow_helpers import FakeBriefing  # noqa: PLC0415

    briefing = FakeBriefing()

    # build_implementer_prompt returns a fixed string ('Implement the plan')
    # without interpolating its arguments, so only a shape check applies here.
    impl_prompt = await briefing.build_implementer_prompt({'title': 't'}, [])
    assert isinstance(impl_prompt, str) and impl_prompt, 'implementer prompt must be non-empty'

    arch_prompt = await briefing.build_architect_prompt({'title': 'my task'})
    assert isinstance(arch_prompt, str) and arch_prompt, 'architect prompt must be non-empty'
    assert 'my task' in arch_prompt, 'architect prompt must include the task title'

    debug_prompt = await briefing.build_debugger_prompt('test failure msg', {})
    assert isinstance(debug_prompt, str) and debug_prompt, 'debugger prompt must be non-empty'
    assert 'test failure msg' in debug_prompt, 'debugger prompt must include the failure text'

    review_prompt = await briefing.build_reviewer_prompt('comprehensive', 'some diff')
    assert isinstance(review_prompt, str) and review_prompt, 'reviewer prompt must be non-empty'
    assert 'comprehensive' in review_prompt, 'reviewer prompt must include the reviewer_type'

    judge_prompt = await briefing.build_completion_judge_prompt(
        {'steps': [1, 2]}, [], 'some diff', task_id='42'
    )
    assert isinstance(judge_prompt, str) and judge_prompt, 'completion-judge prompt must be non-empty'
    assert '42' in judge_prompt, 'completion-judge prompt must include the task_id'

    merge_prompt = await briefing.build_merger_prompt('conflict text', 'intent text')
    assert isinstance(merge_prompt, str) and merge_prompt, 'merger prompt must be non-empty'
    assert 'conflict text' in merge_prompt, 'merger prompt must include the conflict text'

    resume_prompt = await briefing.build_resume_prompt({}, {}, 'the summary', 'the resolution')
    assert isinstance(resume_prompt, str) and resume_prompt, 'resume prompt must be non-empty'
    assert 'the resolution' in resume_prompt, 'resume prompt must include the resolution'


# ---------------------------------------------------------------------------
# Group A: merge-provenance factories (_Fixture, _make, _bind_landed_row)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_merge_provenance():
    """MergeProvenance._outbox is a process-global — never leak a bound outbox."""
    from orchestrator.landed_outbox import MergeProvenance  # noqa: PLC0415

    MergeProvenance._outbox = None
    yield
    MergeProvenance._outbox = None


def test_merge_provenance_factories_smoke(tmp_path) -> None:
    """_make builds a real TaskWorkflow+TaskArtifacts fixture; _bind_landed_row is lookup-able."""
    from _workflow_helpers import _bind_landed_row, _Fixture, _make  # noqa: PLC0415

    from orchestrator.artifacts import TaskArtifacts  # noqa: PLC0415
    from orchestrator.landed_outbox import MergeProvenance  # noqa: PLC0415
    from orchestrator.workflow import TaskWorkflow  # noqa: PLC0415

    fixture = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'pr')
    assert isinstance(fixture, _Fixture)
    assert isinstance(fixture.wf, TaskWorkflow)
    assert isinstance(fixture.artifacts, TaskArtifacts)

    _bind_landed_row(tmp_path, task_id='7', advanced_sha='deadbeef')
    row = MergeProvenance.lookup('7')
    assert row is not None
    assert row.advanced_sha == 'deadbeef'


def test_merge_provenance_factories_identity() -> None:
    """Anti-duplication guard: the producer re-exports the SAME objects as the shared module."""
    import test_workflow_merge_provenance as mp  # noqa: PLC0415
    from _workflow_helpers import _bind_landed_row, _Fixture, _make  # noqa: PLC0415

    assert mp._make is _make
    assert mp._bind_landed_row is _bind_landed_row
    assert mp._Fixture is _Fixture


# ---------------------------------------------------------------------------
# Group B: warm-lane workflow factory (_make_warmlane_workflow)
# ---------------------------------------------------------------------------


def test_warmlane_workflow_factory_smoke(tmp_path) -> None:
    """_make_warmlane_workflow builds a TaskWorkflow with worktree deliberately unset."""
    from _workflow_helpers import _make_warmlane_workflow  # noqa: PLC0415

    from orchestrator.workflow import TaskWorkflow  # noqa: PLC0415

    wf = _make_warmlane_workflow(tmp_path=tmp_path)
    assert isinstance(wf, TaskWorkflow)
    assert wf.worktree is None


def test_warmlane_workflow_factory_identity() -> None:
    """Anti-duplication guard: the producer re-imports the SAME object under its old local name."""
    import test_workflow_warm_lane_requeue as wr  # noqa: PLC0415
    from _workflow_helpers import _make_warmlane_workflow  # noqa: PLC0415

    assert wr._make_workflow is _make_warmlane_workflow


# ---------------------------------------------------------------------------
# Group C: harness factories (_build_harness, _init_git_repo)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_init_git_repo_factory_smoke(tmp_path) -> None:
    """_init_git_repo seeds a real git repo with an initial README commit."""
    from _workflow_helpers import _init_git_repo  # noqa: PLC0415

    await _init_git_repo(tmp_path)
    assert (tmp_path / '.git').is_dir()
    assert (tmp_path / 'README.md').exists()


def test_harness_factories_identity() -> None:
    """Anti-duplication guard: the producer re-exports the SAME objects as the shared module."""
    import test_harness_warm_lane_wiring as hw  # noqa: PLC0415
    from _workflow_helpers import _build_harness, _init_git_repo  # noqa: PLC0415

    assert hw._build_harness is _build_harness
    assert hw._init_git_repo is _init_git_repo


# ---------------------------------------------------------------------------
# Group D: e2e factories (PLAN, _make_review, AgentStub, _build_workflow,
# _build_workflow_with_escalation, _init_repo, _derive_meta_root_like_production)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_e2e_factories_smoke(tmp_path) -> None:
    """Behavioral smoke: PLAN/_make_review/AgentStub/_init_repo real invocation checks."""
    from _workflow_helpers import PLAN, AgentStub, _init_repo, _make_review  # noqa: PLC0415

    assert PLAN['task_id'] == '42'
    assert len(PLAN['steps']) == 2
    assert PLAN['_finalized_at']

    assert _make_review('reviewer_comprehensive')['verdict'] == 'PASS'

    assert AgentStub()._detect_role('You are a TDD architect') == 'architect'
    assert AgentStub()._detect_role('You are the completion judge') == 'judge'

    await _init_repo(tmp_path)
    assert (tmp_path / 'lib.py').exists()
    assert (tmp_path / '.git').is_dir()


def test_e2e_factories_identity() -> None:
    """Anti-duplication guard: the producer re-exports the SAME objects as the shared module.

    Does NOT `from _workflow_helpers import _derive_meta_root_like_production` here —
    that would pull the autouse-fixture NAME into this module's namespace and
    activate it in every test in this file. Instead the fixture is referenced via
    module-attribute access (`_workflow_helpers._derive_meta_root_like_production`),
    which is inert.
    """
    import _workflow_helpers  # noqa: PLC0415
    import test_workflow_e2e as e2e  # noqa: PLC0415
    from _workflow_helpers import (  # noqa: PLC0415
        PLAN,
        AgentStub,
        _build_workflow,
        _build_workflow_with_escalation,
        _init_repo,
        _make_review,
    )

    assert e2e.AgentStub is AgentStub
    assert e2e.PLAN is PLAN
    assert e2e._make_review is _make_review
    assert e2e._build_workflow is _build_workflow
    assert e2e._build_workflow_with_escalation is _build_workflow_with_escalation
    assert e2e._init_repo is _init_repo
    assert e2e._derive_meta_root_like_production is _workflow_helpers._derive_meta_root_like_production
