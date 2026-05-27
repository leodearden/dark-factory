"""Non-fixture test helpers for orchestrator workflow tests.

Lives outside conftest.py to avoid the `sys.modules['conftest']` collision
that arises when root-level pytest loads multiple subprojects' conftests in
the same process under --import-mode=importlib. See task 977 for convention
rationale.
"""

from __future__ import annotations

from escalation.queue import EscalationQueue


class FakeMcp:
    """Minimal McpLifecycle stand-in."""

    @property
    def url(self) -> str:
        return 'http://localhost:9999'

    def mcp_config_json(self, escalation_url: str | None = None) -> dict:
        return {'mcpServers': {}}


class FakeScheduler:
    """Scheduler that tracks status changes without HTTP calls."""

    def __init__(self):
        self.statuses: dict[str, list[str]] = {}
        self.provenance: dict[str, dict] = {}
        self.reopen_reasons: dict[str, str] = {}
        # Optional task data store for tests that exercise the bypass-detection
        # path, which calls get_task to read metadata.done_provenance.
        self.task_data: dict[str, dict] = {}

    async def set_task_status(
        self,
        task_id: str,
        status: str,
        *,
        done_provenance: dict | None = None,
        reopen_reason: str | None = None,
    ) -> None:
        self.statuses.setdefault(task_id, []).append(status)
        if done_provenance is not None:
            self.provenance[task_id] = done_provenance
        if reopen_reason is not None:
            self.reopen_reasons[task_id] = reopen_reason

    async def mark_done(
        self,
        task_id: str,
        *,
        kind: str,
        sha: str,
        note: str | None = None,
    ) -> None:
        """Mirror Scheduler.mark_done: forward to set_task_status with provenance."""
        provenance: dict = {'kind': kind, 'commit': sha}
        if note is not None:
            provenance['note'] = note
        await self.set_task_status(
            task_id, 'done', done_provenance=provenance,
        )

    async def handle_blast_radius_expansion(
        self, task_id: str, current: list[str], needed: list[str]
    ) -> bool:
        return True

    async def get_status(self, task_id: str) -> str | None:
        history = self.statuses.get(task_id)
        return history[-1] if history else None

    async def get_task(self, task_id: str) -> dict | None:
        return self.task_data.get(task_id)

    async def update_task(
        self, task_id: str, metadata: str | dict, *, append: bool = False,
    ) -> bool:
        return True

    async def dispatch_tool(
        self, name: str, arguments: dict, *, timeout: float = 15
    ) -> dict:
        return {}

    def release(self, task_id: str) -> None:
        pass


class FakeBriefing:
    """BriefingAssembler that returns canned prompts."""

    async def build_architect_prompt(self, task: dict, worktree=None, context: str | None = None) -> str:
        return f'Plan task: {task.get("title", "")}'

    async def build_plan_completion_prompt(
        self, task: dict, partial_plan: dict, worktree=None,
        context: str | None = None,
    ) -> str:
        return f'Complete partial plan: {task.get("title", "")}'

    async def build_implementer_prompt(
        self, plan: dict, iteration_log: list, context: str | None = None,
        rebase_notice: dict | None = None, task_id: str | None = None,
    ) -> str:
        return 'Implement the plan'

    async def build_debugger_prompt(
        self, failures: str, plan: dict, context: str | None = None,
        task_id: str | None = None,
    ) -> str:
        return f'Fix: {failures[:100]}'

    async def build_reviewer_prompt(
        self, reviewer_type: str, diff: str, context: str | None = None
    ) -> str:
        return f'Review ({reviewer_type}): {diff[:100]}'

    async def build_completion_judge_prompt(
        self,
        plan: dict,
        iteration_log: list,
        diff: str,
        task_id: str | None = None,
        context: str | None = None,
    ) -> str:
        return f'Judge task {task_id}: plan has {len(plan.get("steps", []))} steps'

    async def build_merger_prompt(
        self, conflicts: str, task_intent: str, context: str | None = None
    ) -> str:
        return f'Merge: {conflicts[:100]}'

    async def build_resume_prompt(
        self,
        task: dict,
        plan: dict,
        escalation_summary: str,
        resolution: str,
        worktree=None,
    ) -> str:
        return f'Resume: {resolution[:100]}'


def _make_resolving_steward(queue: EscalationQueue, task_id: str) -> type:
    """Return a steward class that resolves all pending L0 escalations in start().

    Used by TestMarkBlockedFalseDoneGuard tests to drive _mark_blocked through
    the post-steward `if not remaining:` branch without a full workflow.run() cycle.
    """

    class _FakeSteward:
        def __init__(self, wt_path, cfg_dir):  # noqa: ARG002
            pass

        async def start(self) -> None:
            pending = queue.get_by_task(task_id, status='pending', level=0)
            assert pending, 'expected at least one pending L0 escalation to resolve'
            for esc in pending:
                queue.resolve(
                    esc.id, 'Resolved by FakeSteward', resolved_by='fake-steward',
                )

        async def stop(self) -> None:
            pass

    return _FakeSteward


def _make_status_setting_steward(
    queue: EscalationQueue, scheduler, task_id: str, final_status: str,
) -> type:
    """FakeSteward that resolves L0s and then sets task status to final_status.

    Simulates a steward that marked the task deferred/blocked after inspecting
    the escalation — the workflow must NOT overwrite that decision with
    pending on its auto-requeue path.
    """

    class _FakeSteward:
        def __init__(self, wt_path, cfg_dir):  # noqa: ARG002
            pass

        async def start(self) -> None:
            pending = queue.get_by_task(task_id, status='pending', level=0)
            for esc in pending:
                queue.resolve(esc.id, 'Resolved', resolved_by='fake-steward')
            await scheduler.set_task_status(task_id, final_status)

        async def stop(self) -> None:
            pass

    return _FakeSteward
