"""Periodic deep review checkpoints — integration verification + architectural review."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from shared.cli_invoke import invoke_with_cap_retry

from orchestrator.agents.invoke import invoke_agent
from orchestrator.agents.roles import DEEP_REVIEWER
from orchestrator.config import OrchestratorConfig
from orchestrator.verify import VerifyResult, run_full_verification

if TYPE_CHECKING:
    from escalation.queue import EscalationQueue
    from shared.cost_store import CostStore

    from orchestrator.mcp_lifecycle import McpLifecycle
    from orchestrator.usage_gate import UsageGate

logger = logging.getLogger(__name__)


@dataclass
class ReviewReport:
    review_id: str
    mode: str  # 'focused' or 'full'
    changed_modules: list[str]
    phase1: VerifyResult
    findings_count: int = 0
    tasks_created: list[str] = field(default_factory=list)
    escalated_findings: int = 0
    cost_usd: float = 0.0
    duration_ms: int = 0


class ReviewCheckpoint:
    """Coordinates periodic deep review checkpoints during an orchestrator run.

    Tracks successful merges, triggers focused reviews every N merges, and
    optionally runs a full review after all tasks complete.
    """

    def __init__(
        self,
        config: OrchestratorConfig,
        mcp: McpLifecycle,
        usage_gate: UsageGate | None,
    ):
        self.config = config
        self.mcp = mcp
        self.usage_gate = usage_gate
        self.cost_store: CostStore | None = None
        self.escalation_queue: EscalationQueue | None = None
        self.run_id: str = ''

        self._merge_count = 0
        self._last_review_at_merge = 0
        self._merged_modules: list[str] = []
        self._reports: list[ReviewReport] = []

    def record_merge(self, modules: list[str]) -> None:
        """Record a successful task merge. Called by the harness after a DONE result."""
        self._merge_count += 1
        # Deduplicate as we accumulate
        for m in modules:
            if m not in self._merged_modules:
                self._merged_modules.append(m)
        logger.debug(
            'Review checkpoint: merge %d recorded, %d modules accumulated',
            self._merge_count, len(self._merged_modules),
        )

    def should_trigger(self) -> bool:
        """Return True when enough merges have accumulated since the last review."""
        return (
            self._merge_count - self._last_review_at_merge
            >= self.config.review.interval
        )

    async def run_focused(self) -> ReviewReport:
        """Run a focused review scoped to modules that merged since the last checkpoint."""
        modules = list(self._merged_modules)
        report = await self._run_review('focused', modules)
        # Reset accumulators
        self._last_review_at_merge = self._merge_count
        self._merged_modules.clear()
        self._reports.append(report)
        return report

    async def run_full(self) -> ReviewReport:
        """Run a full review of the entire project."""
        report = await self._run_review('full', [])
        self._last_review_at_merge = self._merge_count
        self._merged_modules.clear()
        self._reports.append(report)
        return report

    async def _run_review(self, mode: str, modules: list[str]) -> ReviewReport:
        """Shared implementation for focused and full reviews."""
        # Guard: reject pytest tmp_path fixtures to prevent junk escalations
        project_str = str(self.config.project_root)
        if '/tmp/pytest-' in project_str or '/tmp/pytest' in project_str:
            raise ValueError(
                f'Refusing to run review against pytest fixture path: {project_str}'
            )

        review_id = datetime.now(UTC).strftime('%Y%m%dT%H%M%S')
        start = time.monotonic()

        logger.info(
            'Review checkpoint [%s] starting — mode=%s, modules=%s',
            review_id, mode, modules or 'all',
        )

        # Phase 1: mechanical verification
        logger.info('Review [%s]: Phase 1 — running full verification', review_id)
        phase1 = await run_full_verification(self.config.project_root, self.config)
        logger.info('Review [%s]: Phase 1 complete — %s', review_id, phase1.summary)

        # Load review briefing
        briefing_content = self._load_briefing()

        # Phase 2+3: architectural coherence + triage (Opus agent)
        prompt = self._build_prompt(mode, phase1, modules, briefing_content, review_id)

        # Build MCP config for fused-memory + escalation access
        escalation_url = None
        esc = self.config.escalation
        escalation_url = f'http://{esc.host}:{esc.port}/mcp'
        mcp_config = self.mcp.mcp_config_json(escalation_url=escalation_url)

        logger.info('Review [%s]: Phase 2+3 — invoking deep reviewer agent', review_id)
        started_at = datetime.now(UTC).isoformat()
        result = await invoke_with_cap_retry(
            usage_gate=self.usage_gate,
            label=f'Review checkpoint [{review_id}]',
            invoke_fn=invoke_agent,
            prompt=prompt,
            system_prompt=DEEP_REVIEWER.system_prompt,
            cwd=self.config.project_root,
            model=getattr(self.config.models, 'deep_reviewer', 'opus'),
            max_turns=getattr(self.config.max_turns, 'deep_reviewer', 100),
            max_budget_usd=getattr(self.config.budgets, 'deep_reviewer', 15.0),
            allowed_tools=DEEP_REVIEWER.allowed_tools or None,
            disallowed_tools=DEEP_REVIEWER.disallowed_tools or None,
            mcp_config=mcp_config,
            effort=getattr(self.config.effort, 'deep_reviewer', 'max'),
            backend=getattr(self.config.backends, 'deep_reviewer', 'claude'),
        )
        completed_at = datetime.now(UTC).isoformat()

        elapsed_ms = int((time.monotonic() - start) * 1000)

        if self.cost_store:
            try:
                model_name = getattr(self.config.models, 'deep_reviewer', 'opus')
                await self.cost_store.save_invocation(
                    run_id=self.run_id,
                    task_id=None,
                    project_id=self.config.fused_memory.project_id,
                    account_name=result.account_name,
                    model=model_name,
                    role='deep_reviewer',
                    cost_usd=result.cost_usd,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    cache_read_tokens=result.cache_read_tokens,
                    cache_create_tokens=result.cache_create_tokens,
                    duration_ms=elapsed_ms,
                    capped=False,
                    started_at=started_at,
                    completed_at=completed_at,
                )
            except Exception:
                logger.warning('Failed to save review invocation cost', exc_info=True)

        # Parse agent output for task creation info
        tasks_created: list[str] = []
        findings_count = 0
        escalated = 0

        parsed = result.structured_output or self._try_parse_json(result.output)

        if parsed:
            findings = parsed.get('findings', [])
            findings_count = len(findings)
            tasks_created = [
                f.get('task_id', '') for f in findings
                if f.get('triage') == 'create_task' and f.get('task_id')
            ]
            escalated = sum(
                1 for f in findings if f.get('triage') == 'escalate'
            )

        # Promote any L0 escalations emitted by the reviewer to L1.  The
        # reviewer runs outside a task workflow, so the synthetic
        # ``review-<id>`` task_id it uses has no steward that could pick
        # these up; left alone they sit pending until the next orchestrator
        # restart auto-dismisses them unread.
        self._promote_reviewer_escalations(review_id)

        report = ReviewReport(
            review_id=review_id,
            mode=mode,
            changed_modules=modules,
            phase1=phase1,
            findings_count=findings_count,
            tasks_created=tasks_created,
            escalated_findings=escalated,
            cost_usd=result.cost_usd,
            duration_ms=elapsed_ms,
        )

        self._save_report(report, result.output)

        logger.info(
            'Review [%s] complete: %d findings, %d tasks created, '
            '%d escalated, cost=$%.2f',
            review_id, findings_count, len(tasks_created),
            escalated, result.cost_usd,
        )

        return report

    def _promote_reviewer_escalations(self, review_id: str) -> int:
        """Promote pending L0 escalations from this review run to level 1.

        The deep reviewer runs outside a task workflow, so any escalations
        it emits land against the synthetic ``review-<review_id>`` task_id
        which has no steward.  Post-hoc, we promote them directly to L1
        (human-visible) and dismiss the L0.  Returns the number promoted.
        """
        if self.escalation_queue is None:
            return 0

        from escalation.models import Escalation

        synthetic_task_id = f'review-{review_id}'
        pending = self.escalation_queue.get_by_task(
            synthetic_task_id, status='pending', level=0,
        )
        if not pending:
            return 0

        promoted = 0
        for esc in pending:
            reesc = Escalation(
                id=self.escalation_queue.make_id(synthetic_task_id),
                task_id=synthetic_task_id,
                agent_role='review-checkpoint',
                severity=esc.severity,
                category=esc.category,
                summary=esc.summary,
                detail=esc.detail,
                suggested_action='manual_intervention',
                worktree=esc.worktree,
                workflow_state=esc.workflow_state,
                level=1,
            )
            self.escalation_queue.submit(reesc)
            self.escalation_queue.resolve(
                esc.id,
                'Auto-promoted to level 1 — review checkpoint has no steward',
                dismiss=True,
                resolved_by='review-checkpoint',
            )
            promoted += 1

        logger.info(
            'Review [%s]: promoted %d L0 escalation(s) to L1',
            review_id, promoted,
        )
        return promoted

    def _load_briefing(self) -> str:
        """Read the review briefing file, or return a warning if absent."""
        briefing_path = self.config.project_root / self.config.review.briefing_path
        if not briefing_path.exists():
            logger.warning(
                'No review briefing at %s — review will rely on code inspection only',
                briefing_path,
            )
            return (
                'No review briefing found. Proceed with best-effort code inspection. '
                'Focus on obvious integration issues, stubs, and cross-module consistency.'
            )
        try:
            return briefing_path.read_text()
        except OSError as e:
            logger.warning('Failed to read review briefing: %s', e)
            return f'Failed to read review briefing: {e}'

    def _build_prompt(
        self,
        mode: str,
        phase1: VerifyResult,
        modules: list[str],
        briefing_content: str,
        review_id: str,
    ) -> str:
        """Assemble the user prompt for the deep review agent."""
        project_root = str(self.config.project_root)
        project_id = self.config.fused_memory.project_id

        scope_block = ''
        if mode == 'focused' and modules:
            mod_list = '\n'.join(f'- `{m}`' for m in modules)
            scope_block = f"""\
## Scope: Focused Review

The following modules have been modified since the last review checkpoint.
Focus your analysis on interactions **between** these modules and the rest of
the codebase. Pay special attention to wiring at module boundaries.

{mod_list}
"""
        else:
            scope_block = """\
## Scope: Full Review

Review the entire codebase. Trace all critical paths end-to-end.
"""

        phase1_block = f"""\
## Phase 1 Results (Mechanical Verification)

These tests/lint/typecheck results were run automatically. Use them as input —
do NOT re-run these checks yourself.

**Status:** {'PASSED' if phase1.passed else 'FAILED'}
**Summary:** {phase1.summary}
"""
        if not phase1.passed:
            if phase1.test_output:
                phase1_block += f"""
### Test Output (last 3000 chars)
```
{phase1.test_output[-3000:]}
```
"""
            if phase1.lint_output:
                phase1_block += f"""
### Lint Output (last 2000 chars)
```
{phase1.lint_output[-2000:]}
```
"""
            if phase1.type_output:
                phase1_block += f"""
### Type Check Output (last 2000 chars)
```
{phase1.type_output[-2000:]}
```
"""

        return f"""\
## Agent Identity

- **agent_id:** `claude-review-{review_id}`
- **project_id:** `{project_id}`
- **project_root:** `{project_root}`

{scope_block}

{phase1_block}

## Review Briefing

The following is the project's review briefing — it describes purpose, key scenarios,
conventions, known gaps, and stability concerns. Use it to contextualise your review.
Respect `known_gaps` — do not flag intentionally deferred work. Enforce `conventions` —
violations are always bugs. Pay special attention to `stability_concerns`.

```yaml
{briefing_content}
```

## Your Task

1. **Search project memory** for recent decisions and known issues before starting:
   - `search(query="recent decisions and known issues", project_id="{project_id}")`

2. **Read code** — trace critical paths, audit stubs, check cross-module consistency.

3. **Triage each finding** and act:
   - Clear-cut issues → `add_task(title=..., description=..., priority=..., metadata={{"source": "review-cycle", "review_id": "{review_id}", "modules": ["path/to/module", ...]}}, project_root="{project_root}")`
   - Ambiguous/architectural → `escalate_info(category=..., summary=...)`
   - Known/accepted → dismiss (don't report)

4. **Reflect on your findings** — write separate memories for each insight worth preserving:
   - **Patterns and surprises** — recurring issues, unexpected gaps, systemic weaknesses
     (`category="observations_and_summaries"`)
   - **Discovered conventions** — implicit rules you noticed in the code that aren't documented
     (`category="preferences_and_norms"`)
   - **Architectural insights** — structural observations about how modules interact, where
     coupling is tight or loose, where the design is fragile
     (`category="decisions_and_rationale"`)
   - Use `add_memory(content=..., category=..., project_id="{project_id}", agent_id="claude-review-{review_id}")`
   - Write each insight as its own memory — don't batch into one blob
   - Skip anything obvious from the code itself; focus on what a future agent couldn't easily rediscover

5. **Output** structured JSON at the end of your response:

```json
{{
  "findings": [
    {{
      "severity": "critical|important|minor",
      "category": "stub|wiring|consistency|test_gap|dead_code|stability",
      "location": "path/to/file.py:line",
      "description": "what's wrong",
      "triage": "create_task|escalate|dismiss",
      "task_id": "id-if-created"
    }}
  ],
  "summary": "one paragraph overview"
}}
```
"""

    def _save_report(self, report: ReviewReport, agent_output: str) -> None:
        """Persist review report as JSON."""
        reports_dir = self.config.project_root / self.config.review.reports_dir
        reports_dir.mkdir(parents=True, exist_ok=True)

        data = {
            'review_id': report.review_id,
            'mode': report.mode,
            'changed_modules': report.changed_modules,
            'phase1_passed': report.phase1.passed,
            'phase1_summary': report.phase1.summary,
            'findings_count': report.findings_count,
            'tasks_created': report.tasks_created,
            'escalated_findings': report.escalated_findings,
            'cost_usd': report.cost_usd,
            'duration_ms': report.duration_ms,
            'agent_output': agent_output[:10000],  # truncate for storage
        }

        path = reports_dir / f'review-{report.review_id}.json'
        path.write_text(json.dumps(data, indent=2))
        logger.info('Review report saved: %s', path)

    @staticmethod
    def _try_parse_json(text: str) -> dict | None:
        """Try to extract a JSON object from agent output."""
        # Look for the last JSON block in the output
        last_brace = text.rfind('}')
        if last_brace == -1:
            return None
        # Walk backwards to find the matching opening brace
        depth = 0
        for i in range(last_brace, -1, -1):
            if text[i] == '}':
                depth += 1
            elif text[i] == '{':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[i:last_brace + 1])
                    except json.JSONDecodeError:
                        return None
        return None
