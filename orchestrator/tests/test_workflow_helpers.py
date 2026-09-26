"""Smoke tests for the _workflow_helpers module, plus its helpers' contract suites.

Verifies that FakeScheduler, FakeMcp, and FakeBriefing behave as expected.
Import-contract verification for _make_resolving_steward and
_make_status_setting_steward relies on indirect coverage from
test_workflow_e2e.py and test_workflow_status_on_resume.py.

Beyond those smoke checks this file also owns the behavioural CONTRACT tests
for helpers _workflow_helpers defines — today `TestSameModuleSiblings`, which
pins `same_module_siblings` across every plausible lock_depth (relocated here
from the consumer suite test_workflow_status_on_resume.py by task 3903 so the
contract sits beside the helper it guards).
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
    impl_prompt = await briefing.build_implementer_prompt({'title': 't'})
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


# ---------------------------------------------------------------------------
# Group E: transcript-archival harness (task 4384): ENC, _config,
# _make_git_ops, _make_transcript_workflow, _config_dir, _write_transcript,
# _archive_root, _archived —
# promoted out of three divergent copies (test_transcript_archive_producer_hook.py,
# test_transcript_archive_backstop.py, test_transcript_archival_boundary_gate.py).
#
# The anti-duplication identity guard that pairs with the smoke test below lives
# in test_workflow_helpers_transcript_identity.py, NOT here: its reads of the
# consumers' `_`-prefixed helper names are counted by the merge-lane ratchet,
# which this file is enrolled in and that one deliberately is not. That module's
# docstring carries the full reasoning.
# ---------------------------------------------------------------------------


def test_transcript_archival_factories_smoke(tmp_path) -> None:
    """The transcript-archival factories build the paths and objects the suites assert on.

    Only the cheap, pure-path contracts are exercised. `_make_transcript_workflow` is
    deliberately NOT driven here: it needs a real git repo plus
    `create_worktree`, and all three consumer suites already drive it
    end-to-end, so duplicating that cost buys nothing.

    Both path contracts are anchored to PRODUCTION, not to a re-spelling of
    the helper bodies: the config dir against the real ``TaskConfigDir``
    constructor, the archive root against ``TranscriptArchiveConfig().root``.
    A literal-vs-literal assertion here would restate the one-line helpers and
    could not detect the only drift that matters — the harness diverging from
    the code it stands in for. ``ENC`` is deliberately NOT pinned: its value is
    arbitrary (the archiver mirrors whatever ``projects/`` subdir name it
    finds), so an equality check against its own literal would assert nothing.
    """
    from _workflow_helpers import (  # noqa: PLC0415
        ENC,
        _archive_root,
        _archived,
        _config,
        _config_dir,
        _make_git_ops,
        _write_transcript,
    )
    from shared.config_dir import TaskConfigDir  # noqa: PLC0415

    from orchestrator.config import TranscriptArchiveConfig  # noqa: PLC0415
    from orchestrator.git_ops import GitOps  # noqa: PLC0415

    repo = tmp_path / 'repo'
    wt = tmp_path / 'wt'

    # The per-task Claude config dir, cross-checked against the REAL
    # constructor production uses (workflow.py builds exactly
    # ``TaskConfigDir(task_id, base_dir=worktree / '.task')``). This is what
    # goes red if the naming template behind CONFIG_DIR_PREFIX ever moves.
    assert _config_dir(wt, '7') == TaskConfigDir('7', base_dir=wt / '.task').path

    src = _write_transcript(wt, '7', 'sess-A', b'x')
    assert src.exists()
    assert src == _config_dir(wt, '7') / 'projects' / ENC / 'sess-A.jsonl'
    assert src.read_bytes() == b'x'

    # The archive root, composed from the config default production resolves
    # against project_root (git_ops.py / harness.py both do
    # ``project_root / transcript_archive.root``) rather than from hardcoded
    # path segments.
    assert _archive_root(repo) == repo / TranscriptArchiveConfig().root
    # ...and _archived hangs off that root, so the two cannot drift apart.
    assert _archived(repo, '7', 'sess-A') == (
        _archive_root(repo) / '7' / ENC / 'sess-A.jsonl'
    )

    assert _config(repo).project_root == repo
    assert _config(repo).transcript_archive.enabled is True
    # Overrides reach OrchestratorConfig.
    disabled = _config(repo, transcript_archive={'enabled': False})
    assert disabled.transcript_archive.enabled is False

    ops = _make_git_ops(repo)
    assert isinstance(ops, GitOps)
    assert ops.project_root == repo
    # No transcript_archive => the teardown backstop is inert...
    assert ops.transcript_archive is None
    # ...and the passthrough kwarg reaches GitOps.__init__ and arms it.
    armed = _make_git_ops(repo, transcript_archive=TranscriptArchiveConfig())
    assert armed.transcript_archive is not None
    assert armed.transcript_archive.enabled is True


@pytest.mark.asyncio
async def test_init_transcript_repo_smoke(tmp_path) -> None:
    """_init_transcript_repo seeds a real, committed git repo with the trivial greet stub.

    A THIRD seeder beside _init_git_repo (README.md) and _init_repo (lib.py +
    test_lib.py with a working greet); see its docstring for why it is not a
    merge of them. The seed contents are what the transcript suites' worktrees
    are branched from, so they are pinned here.
    """
    from _workflow_helpers import _init_transcript_repo  # noqa: PLC0415

    from orchestrator.git_ops import _run  # noqa: PLC0415

    await _init_transcript_repo(tmp_path)

    assert (tmp_path / '.git').is_dir()
    assert (tmp_path / 'lib.py').exists()
    assert (tmp_path / 'lib.py').read_text() == 'def greet(name): return name\n'

    # A committed, non-empty repo on branch main carrying the single
    # "Initial commit" — an EMPTY repo has no HEAD, so create_worktree
    # (which every consumer suite calls) would fail against it.
    rc, branch, _ = await _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=tmp_path)
    assert rc == 0
    assert branch.strip() == 'main'

    rc, log, _ = await _run(['git', 'log', '--format=%s'], cwd=tmp_path)
    assert rc == 0
    assert log.split() == ['Initial', 'commit']


# ---------------------------------------------------------------------------
# Contract: the same-module-sibling constructor is depth-invariant (task 3866,
# relocated here from test_workflow_status_on_resume.py by task 3903 so the
# contract test sits beside the helper it pins).
# ---------------------------------------------------------------------------


class TestSameModuleSiblings:
    """Pin `same_module_siblings`'s contract across every plausible lock_depth.

    See `same_module_siblings`' own docstring in `_workflow_helpers` for WHY
    the pair is derived from the depth rather than written as a literal; that
    rationale is kept in exactly one place so the copies cannot drift.

    Depths cover the operational value (12), the package-bundled default
    (`defaults.yaml:7` = 4), the pydantic Field default (2), the smallest
    depth that can honour the contract (1), and a deeper future setting (20);
    `test_rejects_depths_below_one` pins the boundary just under it.
    """

    @pytest.mark.parametrize('lock_depth', [1, 2, 3, 4, 12, 20])
    def test_siblings_are_distinct_files_in_one_module_at_any_depth(self, lock_depth: int):
        from _workflow_helpers import same_module_siblings  # noqa: PLC0415

        from orchestrator.scheduler import files_to_modules  # noqa: PLC0415

        f1, f2 = same_module_siblings(lock_depth)

        assert f1 != f2, (
            f'same_module_siblings({lock_depth}) returned identical paths {f1!r}; '
            '"same module, DIFFERENT file" is unconstructible if the two paths '
            'are the same string.'
        )

        modules = files_to_modules([f1, f2], lock_depth)
        assert len(modules) == 1, (
            f'Expected {f1!r} and {f2!r} to collapse to ONE module at '
            f'lock_depth={lock_depth}; got {sorted(modules)!r}.'
        )

        assert files_to_modules([f1], lock_depth) == modules, (
            f'Adding sibling {f2!r} must not widen the module set of {f1!r} at '
            f'lock_depth={lock_depth} — that equality is the exact precondition '
            'the same-module-widen tests assert.'
        )

    @pytest.mark.parametrize('lock_depth', [0, -1])
    def test_rejects_depths_below_one(self, lock_depth: int):
        """Below depth 1 the guarantee is UNSATISFIABLE, so it must raise.

        `OrchestratorConfig.lock_depth` carries no `ge=1` constraint, so a
        caller forwarding `config.lock_depth` can genuinely reach this input.
        """
        from _workflow_helpers import (  # noqa: PLC0415
            _unguarded_sibling_pair,
            same_module_siblings,
        )

        from orchestrator.scheduler import files_to_modules  # noqa: PLC0415

        with pytest.raises(ValueError, match='lock_depth >= 1'):
            same_module_siblings(lock_depth)

        # The boundary is real, not arbitrary — and the reason it must RAISE
        # rather than return is that the unguarded construction fails
        # SILENTLY.  Both paths truncate to '' (parts[:0]), files_to_modules
        # DROPS them, and the same-module precondition the callers assert
        # degenerates to [] == [] — passing VACUOUSLY while the workflow under
        # test holds no module lock at all.
        #
        # Driven through the REAL construction (`_unguarded_sibling_pair`, the
        # body `same_module_siblings` runs once past its guard) rather than a
        # literal pair: the assertions below hold for ANY empty-prefix pair, so
        # a hand-written copy would keep passing — and stop demonstrating what
        # the guarded constructor would actually have produced — if the path
        # scheme ever changed.
        f1, f2 = _unguarded_sibling_pair(lock_depth)
        assert files_to_modules([f1, f2], lock_depth) == [], (
            f'expected the unguarded pair at lock_depth={lock_depth} to yield NO '
            f'modules; got {files_to_modules([f1, f2], lock_depth)!r}. If this '
            'changed, re-derive the guard\'s rationale from the new behaviour.'
        )
        assert files_to_modules([f1], lock_depth) == files_to_modules(
            [f1, f2], lock_depth,
        ), (
            'the same-module precondition is expected to pass VACUOUSLY here — '
            'that vacuity is precisely why same_module_siblings refuses this '
            'depth instead of returning a pair'
        )
