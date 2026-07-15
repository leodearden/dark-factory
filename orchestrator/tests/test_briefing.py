"""Regression tests for ``BriefingAssembler._format_task``.

Locks in the post-normalization invariant that ``task['metadata']`` is
always a dict by the time ``_format_task`` reads it. The boundary
normalizer in :meth:`Scheduler.get_tasks` is responsible for the coerce;
this test confirms ``_format_task`` builds the prompt without raising
``AttributeError`` when handed the wire shape downstream consumers
should now see.

Also covers :meth:`BriefingAssembler.build_plan_tightening_prompt`,
introduced for the plan-files-not-touched architect-narrowing retry.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from orchestrator.agents.briefing import BriefingAssembler
from orchestrator.config import GitConfig, OrchestratorConfig


@pytest.fixture
def briefing(tmp_path: Path) -> BriefingAssembler:
    config = OrchestratorConfig(
        project_root=tmp_path,
        git=GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
        ),
    )
    return BriefingAssembler(config)


@pytest.fixture
def anti_anchor_task() -> dict:
    """Shared task fixture with metadata.files for anti-anchor tests.

    Used by both TestFormatTaskIncludeFiles (unit) and
    TestAntiAnchorFirstDerivation (integration) to keep the sample task
    definition in one place.
    """
    return {
        'id': '1835',
        'title': 'Anti-anchor title',
        'description': 'Derive files independently',
        'metadata': {'files': ['orchestrator', 'shared']},
    }


@pytest.fixture
def task_with_proposal() -> dict:
    """Task fixture carrying a dry_run_proposals entry.

    Shared across the prior-proposal test classes (C-A1 anti-anchor, the
    retry/resume golden tests, and the staleness-guard tests) so the sample
    proposal shape lives in one place.
    """
    return {
        'id': '9001',
        'title': 'Task with a prior block-time proposal',
        'description': 'A task that was previously blocked and investigated.',
        'metadata': {
            'dry_run_proposals': [
                {
                    'proposal_text': 'Root cause is a stale cache entry; fix by invalidating on write.',
                    'risk_label': 'medium',
                    'files_referenced': ['orchestrator/x.py'],
                    'timestamp': '2026-07-13T10:00:00+00:00',
                    'investigated_at': '2026-07-13T10:00:00+00:00',
                },
            ],
        },
    }


class TestFormatTaskMetadataInvariant:
    def test_dict_metadata_with_files(self, briefing: BriefingAssembler):
        task = {
            'id': '424',
            'title': 'Trivial DRY consolidation',
            'description': 'Move three helpers',
            'metadata': {'files': ['orchestrator', 'shared']},
        }
        out = briefing._format_task(task)
        assert '**ID:** 424' in out
        assert '**Files:** orchestrator, shared' in out

    def test_empty_dict_metadata(self, briefing: BriefingAssembler):
        task = {
            'id': '1',
            'title': 'No files',
            'metadata': {},
        }
        out = briefing._format_task(task)
        assert '**ID:** 1' in out
        assert 'Files' not in out

    def test_metadata_absent(self, briefing: BriefingAssembler):
        task = {'id': '2', 'title': 'No metadata key'}
        out = briefing._format_task(task)
        assert '**ID:** 2' in out
        assert 'Modules' not in out

    def test_metadata_dict_without_modules_key(self, briefing: BriefingAssembler):
        task = {
            'id': '3',
            'title': 'Other metadata only',
            'metadata': {'files': ['a.py']},
        }
        out = briefing._format_task(task)
        assert '**ID:** 3' in out
        assert 'Modules' not in out


class TestFormatTaskIncludeFiles:
    """Unit tests for the include_files parameter on _format_task.

    Locks in the anti-anchor mechanic introduced for C-A1: when
    include_files=False the Files line must be absent so the first architect
    derivation cannot rubber-stamp the queue-time metadata.files guess.
    """

    def test_include_files_false_omits_files_line(
        self, briefing: BriefingAssembler, anti_anchor_task: dict,
    ):
        """include_files=False must hide the Files line."""
        out = briefing._format_task(anti_anchor_task, include_files=False)
        assert '**Files:**' not in out

    def test_include_files_false_keeps_description(
        self, briefing: BriefingAssembler, anti_anchor_task: dict,
    ):
        """include_files=False must not suppress Description or other fields."""
        out = briefing._format_task(anti_anchor_task, include_files=False)
        assert '**Description:**' in out
        assert '**Title:**' in out

    def test_include_files_default_still_shows_files(
        self, briefing: BriefingAssembler, anti_anchor_task: dict,
    ):
        """Default call (no flag) must still render the Files line."""
        out = briefing._format_task(anti_anchor_task)
        assert '**Files:** orchestrator, shared' in out


@pytest.mark.asyncio
class TestBuildPlanTighteningPrompt:
    """Architect narrowing pass after the plan-files-not-touched gate.

    Prohibition assertions use positive-directive form per
    feedback_test_assert_negative_directives.md — the prompt itself
    contains the token names it forbids, so a bare ``not in`` would
    self-conflict.
    """

    async def _build(
        self,
        briefing: BriefingAssembler,
        files: list[str] | None = None,
        not_touched: list[str] | None = None,
    ) -> str:
        task = {
            'id': '2656',
            'title': 'Test task',
            'description': 'Demo',
        }
        plan = {'files': files if files is not None else ['a.py', 'b.py', 'c.py']}
        return await briefing.build_plan_tightening_prompt(
            task,
            plan,
            not_touched if not_touched is not None else ['a.py', 'b.py'],
            worktree=None,
            context='',
        )

    async def test_mentions_both_valid_actions(self, briefing: BriefingAssembler):
        prompt = await self._build(briefing)
        assert 'update_plan_metadata' in prompt
        assert 'confirm_plan' in prompt

    async def test_lists_not_touched_entries(self, briefing: BriefingAssembler):
        prompt = await self._build(
            briefing,
            files=['x.py', 'y.py', 'z.py'],
            not_touched=['x.py', 'y.py'],
        )
        assert 'x.py' in prompt
        assert 'y.py' in prompt
        # Current plan files should also be visible.
        assert 'z.py' in prompt

    async def test_forbids_creation_and_step_edits(
        self, briefing: BriefingAssembler,
    ):
        prompt = await self._build(briefing)
        # Positive-directive assertions covering the forbidden tools.
        # The prompt mentions create_plan/add_plan_step/replace_plan_step
        # while listing them as off-limits — assert by section header
        # rather than 'not in prompt' to avoid the self-conflict pattern.
        assert 'Forbidden' in prompt or 'Do not call' in prompt
        assert 'create_plan' in prompt
        assert 'add_plan_step' in prompt
        assert 'replace_plan_step' in prompt

    async def test_states_new_file_addition_will_be_rejected(
        self, briefing: BriefingAssembler,
    ):
        prompt = await self._build(briefing)
        assert 'must NOT add new files' in prompt

    async def test_includes_agent_identity(self, briefing: BriefingAssembler):
        prompt = await self._build(briefing)
        assert 'claude-task-2656-architect' in prompt


@pytest.mark.asyncio
class TestBuildPlanCompletionPrompt:
    """Completion pass that resumes a partial plan left by an interrupted run."""

    async def _build(self, briefing: BriefingAssembler) -> str:
        task = {'id': '3822', 'title': 'Add recon closure', 'description': 'Demo'}
        partial = {
            'task_id': '3822',
            'title': 'Add recon closure',
            'files': ['recon.py'],
            'analysis': 'partial analysis',
            'prerequisites': [],
            'steps': [
                {'id': 'step-1', 'type': 'test', 'description': 'test a',
                 'status': 'pending', 'commit': None},
                {'id': 'step-2', 'type': 'impl', 'description': 'impl a',
                 'status': 'pending', 'commit': None},
            ],
        }
        return await briefing.build_plan_completion_prompt(
            task, partial, worktree=None, context='',
        )

    async def test_shows_partial_and_requires_finalize(
        self, briefing: BriefingAssembler,
    ):
        prompt = await self._build(briefing)
        # The existing steps are shown so the architect can build on them.
        assert 'step-1' in prompt
        assert 'step-2' in prompt
        # Must instruct finalizing the completed plan.
        assert 'confirm_plan' in prompt
        # Frames the work as finishing, not starting over.
        assert 'Completion' in prompt or 'finish' in prompt.lower()

    async def test_offers_restart_and_rejection_escape_hatches(
        self, briefing: BriefingAssembler,
    ):
        prompt = await self._build(briefing)
        assert 'create_plan' in prompt
        assert 'report_unactionable_task' in prompt

    async def test_includes_agent_identity(self, briefing: BriefingAssembler):
        prompt = await self._build(briefing)
        assert 'claude-task-3822-architect' in prompt


@pytest.mark.asyncio
class TestAntiAnchorFirstDerivation:
    """C-A1 / C-A2 integration tests.

    C-A1: build_architect_prompt (first/fresh derivation) must EXCLUDE the
          **Files:** label so the architect cannot rubber-stamp metadata.files.
    C-A2: build_revalidation_prompt (revalidation pass) must STILL INCLUDE
          the **Files:** label — the revalidation path is explicitly exempt.
    """

    async def test_architect_prompt_excludes_files(
        self, briefing: BriefingAssembler, anti_anchor_task: dict,
    ):
        """C-A1: build_architect_prompt must not expose metadata.files."""
        prompt = await briefing.build_architect_prompt(
            anti_anchor_task, worktree=None, context='',
        )
        assert '**Files:**' not in prompt

    async def test_architect_prompt_keeps_description(
        self, briefing: BriefingAssembler, anti_anchor_task: dict,
    ):
        """C-A1: description and title must survive the files suppression."""
        prompt = await briefing.build_architect_prompt(
            anti_anchor_task, worktree=None, context='',
        )
        assert 'Derive files independently' in prompt
        assert 'Anti-anchor title' in prompt

    async def test_revalidation_prompt_includes_files(
        self, briefing: BriefingAssembler, anti_anchor_task: dict,
    ):
        """C-A2 exemption: build_revalidation_prompt must still show Files."""
        prompt = await briefing.build_revalidation_prompt(
            anti_anchor_task,
            existing_plan={'files': ['orchestrator'], 'steps': []},
            changed_files=[],
            worktree=None,
            context='',
        )
        assert '**Files:** orchestrator, shared' in prompt


@pytest.mark.asyncio
class TestPriorProposalAntiAnchor:
    """C-A1 for prior dry-run block-time proposals.

    First-dispatch (``build_architect_prompt``, default call) must NOT
    surface a prior block-time investigation — this is the same
    anti-anchoring concern as ``TestAntiAnchorFirstDerivation`` above, applied
    to ``metadata.dry_run_proposals`` instead of ``metadata.files``. The
    re-plan path opts in explicitly via ``include_prior_proposals=True``.
    """

    async def test_first_dispatch_excludes_proposal(
        self, briefing: BriefingAssembler, task_with_proposal: dict,
    ):
        prompt = await briefing.build_architect_prompt(
            task_with_proposal, worktree=None, context='',
        )
        assert 'Root cause is a stale cache entry' not in prompt
        assert 'prior block-time investigation' not in prompt

    async def test_replan_includes_proposal(
        self, briefing: BriefingAssembler, task_with_proposal: dict,
    ):
        prompt = await briefing.build_architect_prompt(
            task_with_proposal,
            worktree=None,
            context='',
            include_prior_proposals=True,
        )
        assert 'Root cause is a stale cache entry' in prompt
        assert 'prior block-time investigation' in prompt


@pytest.mark.asyncio
class TestRevalidationPriorProposal:
    """Retry golden test.

    ``build_revalidation_prompt`` is itself a retry path (blast-radius
    requeue), so it always surfaces the prior proposal — no
    ``include_prior_proposals`` gate needed, unlike ``build_architect_prompt``.
    """

    async def test_includes_proposal_with_provenance(
        self, briefing: BriefingAssembler, task_with_proposal: dict,
    ):
        prompt = await briefing.build_revalidation_prompt(
            task_with_proposal,
            existing_plan={'files': ['orchestrator'], 'steps': []},
            changed_files=[],
            worktree=None,
            context='',
        )
        assert 'prior block-time investigation' in prompt
        assert (
            'Root cause is a stale cache entry; fix by invalidating on write.'
            in prompt
        )
        assert 'medium' in prompt
        assert 'orchestrator/x.py' in prompt
        assert '2026-07-13T10:00:00+00:00' in prompt


@pytest.mark.asyncio
class TestResumePriorProposal:
    """Resume golden test.

    ``build_resume_prompt`` runs after an escalation resolution — inherently
    a retry/resume path — so it always surfaces the prior proposal. It has
    no ``context`` override parameter, so ``_get_memory_context`` is patched
    to avoid a real fused-memory HTTP call (mirrors test_briefing_judge.py).
    """

    async def test_includes_proposal_with_provenance(
        self, briefing: BriefingAssembler, task_with_proposal: dict,
    ):
        with patch.object(
            BriefingAssembler, '_get_memory_context', return_value='# Context\n\n_stub_',
        ):
            prompt = await briefing.build_resume_prompt(
                task_with_proposal,
                plan={},
                escalation_summary='the issue',
                resolution='the fix',
                worktree=None,
            )
        assert 'prior block-time investigation' in prompt
        assert (
            'Root cause is a stale cache entry; fix by invalidating on write.'
            in prompt
        )


class TestPriorProposalStaleness:
    """Staleness guard + robustness for ``_format_prior_proposal``.

    The guard omits a persisted proposal when it predates the task's last
    confirmed block transition (``metadata.last_blocked_at``) — a re-block
    without a fresh investigation must not resurface stale analysis. Absent
    or unparseable timestamps fail OPEN (include) so analysis is never
    silently lost to a formatting edge case.
    """

    # -- (a) stale: proposal predates last_blocked_at -----------------------

    def test_stale_proposal_omitted_by_helper(self, briefing: BriefingAssembler):
        task = {
            'id': '9010',
            'metadata': {
                'dry_run_proposals': [
                    {
                        'proposal_text': 'stale analysis text',
                        'risk_label': 'low',
                        'files_referenced': [],
                        'timestamp': '2026-07-13T10:00:00+00:00',
                    },
                ],
                # Later than the proposal's timestamp — a re-block without a
                # fresh investigation.
                'last_blocked_at': '2026-07-14T00:00:00+00:00',
            },
        }
        assert briefing._format_prior_proposal(task) == ''

    @pytest.mark.asyncio
    async def test_stale_proposal_omitted_from_revalidation_prompt(
        self, briefing: BriefingAssembler,
    ):
        task = {
            'id': '9011',
            'title': 'Stale proposal task',
            'description': 'Demo',
            'metadata': {
                'dry_run_proposals': [
                    {
                        'proposal_text': 'stale analysis text',
                        'risk_label': 'low',
                        'files_referenced': [],
                        'timestamp': '2026-07-13T10:00:00+00:00',
                    },
                ],
                'last_blocked_at': '2026-07-14T00:00:00+00:00',
            },
        }
        prompt = await briefing.build_revalidation_prompt(
            task,
            existing_plan={'files': [], 'steps': []},
            changed_files=[],
            worktree=None,
            context='',
        )
        assert 'stale analysis text' not in prompt

    # -- (b) fresh: proposal postdates last_blocked_at -----------------------

    def test_fresh_proposal_included(self, briefing: BriefingAssembler):
        task = {
            'id': '9012',
            'metadata': {
                'dry_run_proposals': [
                    {
                        'proposal_text': 'fresh analysis text',
                        'risk_label': 'low',
                        'files_referenced': [],
                        'timestamp': '2026-07-15T00:00:00+00:00',
                    },
                ],
                'last_blocked_at': '2026-07-14T00:00:00+00:00',
            },
        }
        out = briefing._format_prior_proposal(task)
        assert 'fresh analysis text' in out

    # -- (c) last_blocked_at absent: fail open --------------------------------

    def test_missing_last_blocked_at_fails_open(
        self, briefing: BriefingAssembler, task_with_proposal: dict,
    ):
        assert 'last_blocked_at' not in task_with_proposal['metadata']
        out = briefing._format_prior_proposal(task_with_proposal)
        assert 'Root cause is a stale cache entry' in out

    # -- (d) unparseable timestamps: fail open --------------------------------

    def test_unparseable_proposal_timestamp_fails_open(
        self, briefing: BriefingAssembler,
    ):
        task = {
            'id': '9013',
            'metadata': {
                'dry_run_proposals': [
                    {
                        'proposal_text': 'garbled proposal timestamp analysis',
                        'risk_label': 'low',
                        'files_referenced': [],
                        'timestamp': 'not-a-timestamp',
                    },
                ],
                'last_blocked_at': '2026-07-14T00:00:00+00:00',
            },
        }
        out = briefing._format_prior_proposal(task)
        assert 'garbled proposal timestamp analysis' in out

    def test_unparseable_last_blocked_at_fails_open(
        self, briefing: BriefingAssembler,
    ):
        task = {
            'id': '9014',
            'metadata': {
                'dry_run_proposals': [
                    {
                        'proposal_text': 'garbled block-stamp analysis',
                        'risk_label': 'low',
                        'files_referenced': [],
                        'timestamp': '2026-07-13T10:00:00+00:00',
                    },
                ],
                'last_blocked_at': 'not-a-timestamp',
            },
        }
        out = briefing._format_prior_proposal(task)
        assert 'garbled block-stamp analysis' in out

    # -- (e) empty / non-dict latest proposal: no content, no crash ----------

    def test_empty_proposals_list_returns_empty(self, briefing: BriefingAssembler):
        task = {'id': '9015', 'metadata': {'dry_run_proposals': []}}
        assert briefing._format_prior_proposal(task) == ''

    def test_non_dict_latest_proposal_returns_empty(
        self, briefing: BriefingAssembler,
    ):
        task = {'id': '9016', 'metadata': {'dry_run_proposals': ['not-a-dict']}}
        assert briefing._format_prior_proposal(task) == ''

    @pytest.mark.asyncio
    async def test_integration_prompt_builds_without_error_for_malformed_proposal(
        self, briefing: BriefingAssembler,
    ):
        task = {
            'id': '9017',
            'title': 'Malformed proposal task',
            'description': 'Demo',
            'metadata': {'dry_run_proposals': ['not-a-dict']},
        }
        prompt = await briefing.build_revalidation_prompt(
            task,
            existing_plan={'files': [], 'steps': []},
            changed_files=[],
            worktree=None,
            context='',
        )
        assert isinstance(prompt, str)
        assert 'prior block-time investigation' not in prompt
