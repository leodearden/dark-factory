"""Async LLM-as-judge that evaluates reconciliation run quality."""

import json
import logging
import os
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.models.reconciliation import (
    JudgeVerdict,
    StageReport,
    VerdictAction,
    VerdictSeverity,
)
from fused_memory.reconciliation.journal import ReconciliationJournal
from fused_memory.reconciliation.prompts.judge import JUDGE_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


UnhaltCallback = Callable[[str], Awaitable[None] | None]


class Judge:
    """Async LLM reviewer that evaluates reconciliation run quality."""

    def __init__(
        self,
        config: ReconciliationConfig,
        journal: ReconciliationJournal,
        usage_gate=None,
        *,
        on_unhalt_cb: UnhaltCallback | None = None,
    ):
        self.config = config
        self.journal = journal
        self._halted_projects: set[str] = set()
        self._halt_cooldown_until: dict[str, datetime] = {}
        self._unhalt_grace_remaining: dict[str, int] = {}
        self._usage_gate = usage_gate
        self._on_unhalt_cb = on_unhalt_cb

    async def initialize(self) -> None:
        """Rehydrate halt state from the journal.

        Without this, a service restart would silently clear all halts —
        exactly the self-latching-halt workaround that Apr 2026 debug turned
        up. Persisting means a genuine halt survives restarts and can only be
        cleared via unhalt_reconciliation.
        """
        try:
            rows = await self.journal.get_halt_states()
        except Exception:
            logger.warning('Judge.initialize: get_halt_states failed', exc_info=True)
            return
        for row in rows:
            pid = row['project_id']
            if row['unhalted_at'] is None:
                self._halted_projects.add(pid)
                if row['cooldown_until'] is not None:
                    self._halt_cooldown_until[pid] = row['cooldown_until']
            if (row['unhalt_grace_remaining'] or 0) > 0:
                self._unhalt_grace_remaining[pid] = row['unhalt_grace_remaining']
        if self._halted_projects or self._unhalt_grace_remaining:
            logger.info(
                f'Judge: rehydrated halt state — halted={sorted(self._halted_projects)} '
                f'grace={dict(self._unhalt_grace_remaining)}'
            )

    async def review_run(self, run_id: str) -> JudgeVerdict | None:
        """Review a completed reconciliation run asynchronously."""
        try:
            run = await self.journal.get_run(run_id)
            if run is None:
                logger.warning(f'Judge: run {run_id} not found')
                return None

            if run.project_id in self._halted_projects:
                logger.warning(f'Judge: project {run.project_id} is halted, skipping review')
                return None

            entries = await self.journal.get_entries(run_id)
            combined_actions = await self.journal.get_run_actions_combined(run_id)
            recent_verdicts = await self.journal.get_recent_verdicts(
                run.project_id, limit=10
            )
            # Trend detection uses a time-windowed query so old moderates
            # outside the window cannot influence the halt decision. Separate
            # from ``recent_verdicts`` (which feeds the prompt) so the judge
            # still sees short-term history regardless of window length.
            trend_window_hours = float(self.config.halt_trend_window_hours)
            trend_verdicts = await self.journal.get_recent_verdicts(
                run.project_id,
                limit=50,
                since=datetime.now(UTC) - timedelta(hours=trend_window_hours),
            )

            prompt = self._build_review_prompt(run, entries, recent_verdicts, combined_actions)
            response_text = await self._call_llm(prompt)
            verdict = self._parse_verdict(response_text, run_id)

            # Act on verdict — must happen BEFORE persisting so the DB receives the
            # final action_taken value ('rollback', 'halt', or 'none')
            if verdict.severity == 'moderate':
                verdict.action_taken = VerdictAction.rollback
                logger.warning(f'Judge: moderate issues in run {run_id}, recommending rollback')
            elif verdict.severity == 'serious':
                if self.config.halt_on_judge_serious:
                    verdict.action_taken = VerdictAction.halt
                    await self._apply_halt(
                        run.project_id,
                        reason=f'Serious verdict in run {run_id}',
                    )
                    logger.error(
                        f'Judge: SERIOUS issues in run {run_id}, halting project {run.project_id}'
                    )

            await self.journal.add_verdict(verdict)

            logger.info(
                'reconciliation.judge_verdict',
                extra={
                    'run_id': run_id,
                    'severity': verdict.severity,
                    'findings_count': len(verdict.findings),
                    'action_taken': verdict.action_taken,
                },
            )

            # Check error trends using the windowed verdicts + the fresh one
            await self._check_error_trends(run.project_id, trend_verdicts + [verdict])

            return verdict

        except Exception as e:
            logger.error(f'Judge review failed for run {run_id}: {e}')
            return None

    def _build_review_prompt(self, run, entries, recent_verdicts,
                             combined_actions: list[dict] | None = None) -> str:
        # recent_verdicts is intentionally NOT fed into the LLM prompt.
        # _check_error_trends handles systemic-pattern detection in code; exposing the
        # history here caused a feedback loop where the LLM generated 'systemic_trend'
        # findings whenever prior verdicts were non-ok, keeping the streak alive.
        del recent_verdicts

        entry_summaries = []
        for e in entries:
            entry_summaries.append({
                'id': e.id,
                'operation': e.operation,
                'target_system': e.target_system,
                'reasoning': e.reasoning,
                'has_before': e.before_state is not None,
                'has_after': e.after_state is not None,
            })

        stage_reports = {}
        for stage_id, report in run.stage_reports.items():
            if isinstance(report, StageReport):
                stage_reports[stage_id] = {
                    'items_flagged': len(report.items_flagged),
                    'stats': report.stats,
                    'llm_calls': report.llm_calls,
                    'tokens_used': report.tokens_used,
                }
            else:
                # Plain dict entries (e.g. _error injected by harness on failure)
                stage_reports[stage_id] = report

        # Summarize MCP actions
        actions = combined_actions or []
        action_summaries = [
            {
                'operation': a.get('operation', a.get('action_type', '?')),
                'target': a.get('target', a.get('target_system', '?')),
                'source': a.get('source', '?'),
                'created_at': a.get('created_at', '?'),
            }
            for a in actions
        ]

        return f"""## Reconciliation Run Review

### Run Metadata
- Run ID: {run.id}
- Project: {run.project_id}
- Type: {run.run_type}
- Trigger: {run.trigger_reason}
- Events processed: {run.events_processed}
- Status: {run.status}

### Stage Reports
{json.dumps(stage_reports, indent=2, default=str)}

### MCP Actions ({len(action_summaries)} total)
{json.dumps(action_summaries, indent=2, default=str)}

### Journal Entries ({len(entries)} total)
{json.dumps(entry_summaries, indent=2, default=str)}

Review this run and provide your verdict as JSON.
"""

    async def _call_llm(self, prompt: str) -> str:
        """Single LLM call (not an agent loop)."""
        if self.config.judge_llm_provider == 'claude_cli':
            return await self._call_judge_cli(prompt)
        elif self.config.judge_llm_provider == 'anthropic':
            import anthropic

            client = anthropic.AsyncAnthropic()
            response = await client.messages.create(
                model=self.config.judge_llm_model,
                max_tokens=4096,
                system=JUDGE_SYSTEM_PROMPT,
                messages=[{'role': 'user', 'content': prompt}],
            )
            text_blocks = [b for b in response.content if b.type == 'text']
            return text_blocks[0].text if text_blocks else ''
        else:
            from openai import AsyncOpenAI

            client = AsyncOpenAI()
            response = await client.chat.completions.create(
                model=self.config.judge_llm_model,
                messages=[
                    {'role': 'system', 'content': JUDGE_SYSTEM_PROMPT},
                    {'role': 'user', 'content': prompt},
                ],
            )
            return response.choices[0].message.content or ''

    async def _call_judge_cli(self, prompt: str) -> str:
        """Single-shot Claude CLI call for judge evaluation.

        Includes: stdin isolation, stdout+stderr error reading, and usage-cap
        retry with account failover when a UsageGate is attached.
        """
        import asyncio as _asyncio

        while True:
            # Gate: get OAuth token (or None)
            oauth_token = None
            if self._usage_gate:
                oauth_token = await self._usage_gate.before_invoke()

            cmd = [
                'claude', '--print', '--output-format', 'json',
                '--model', self.config.judge_llm_model,
                '--system-prompt', JUDGE_SYSTEM_PROMPT,
                '--permission-mode', 'bypassPermissions',
                '--', prompt,
            ]

            # Build env: strip ANTHROPIC_API_KEY, inject OAuth token
            env = {k: v for k, v in os.environ.items() if k != 'ANTHROPIC_API_KEY'}
            if oauth_token:
                env['CLAUDE_CODE_OAUTH_TOKEN'] = oauth_token

            try:
                try:
                    proc = await _asyncio.create_subprocess_exec(
                        *cmd,
                        stdin=_asyncio.subprocess.DEVNULL,
                        stdout=_asyncio.subprocess.PIPE,
                        stderr=_asyncio.subprocess.PIPE,
                        env=env,
                    )
                except FileNotFoundError as exc:
                    raise RuntimeError(
                        'Claude CLI not found. Install with: npm install -g @anthropic-ai/claude-code'
                    ) from exc

                try:
                    stdout, stderr = await _asyncio.wait_for(proc.communicate(), timeout=600)
                except TimeoutError as exc:
                    proc.kill()
                    raise RuntimeError('Claude CLI timed out after 600 seconds') from exc

                stdout_text = stdout.decode()
                stderr_text = stderr.decode()

                # Check for cap hit before processing results
                if self._usage_gate and self._usage_gate.detect_cap_hit(
                    stderr_text, stdout_text, 'claude', oauth_token=oauth_token,
                ):
                    logger.warning('Usage cap hit during judge CLI call, retrying')
                    continue

                if proc.returncode != 0:
                    error_detail = stderr_text[-500:] if stderr_text.strip() else stdout_text[-500:]
                    raise RuntimeError(
                        f'Claude CLI exited with code {proc.returncode}: {error_detail}'
                    )

                if not stdout_text.strip():
                    if self._usage_gate:
                        self._usage_gate.confirm_account_ok(oauth_token)
                        self._usage_gate.on_agent_complete(0.0)
                    return ''

                result = json.loads(stdout_text)
                if self._usage_gate:
                    cost_usd = float(result.get('cost_usd', result.get('total_cost_usd', 0.0)))
                    self._usage_gate.confirm_account_ok(oauth_token)
                    self._usage_gate.on_agent_complete(cost_usd)
                return result.get('result', '')

            except BaseException:
                if self._usage_gate is not None:
                    try:
                        self._usage_gate.release_probe_slot(oauth_token)
                    except Exception:
                        logger.warning('release_probe_slot failed', exc_info=True)
                raise

    def _parse_verdict(self, response_text: str, run_id: str) -> JudgeVerdict:
        """Parse judge response into JudgeVerdict."""
        try:
            # Extract JSON from response (might be wrapped in markdown)
            text = response_text.strip()
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0].strip()
            elif '```' in text:
                text = text.split('```')[1].split('```')[0].strip()

            data = json.loads(text)
            return JudgeVerdict(
                run_id=run_id,
                reviewed_at=datetime.now(UTC),
                severity=data.get('severity', VerdictSeverity.ok),
                findings=data.get('findings', []),
                action_taken=VerdictAction.none,
            )
        except (json.JSONDecodeError, IndexError, KeyError) as e:
            logger.warning(f'Failed to parse judge response: {e}')
            return JudgeVerdict(
                run_id=run_id,
                reviewed_at=datetime.now(UTC),
                severity=VerdictSeverity.minor,
                findings=[{
                    'issue': 'Judge response could not be parsed',
                    'severity': 'minor',
                    'recommendation': 'Manual review recommended',
                }],
                action_taken=VerdictAction.none,
            )

    async def _check_error_trends(
        self, project_id: str, verdicts: list[JudgeVerdict],
    ) -> None:
        """Detect clustered, recent, and consecutive non-ok verdicts.

        Three gates must all hold:

        1. **Time window**: at least ``halt_trend_moderate_count`` non-ok verdicts
           within the last ``halt_trend_window_hours``. Old moderates from days
           ago cannot hold a project halted forever.
        2. **Consecutive most-recent**: the first ``halt_trend_consecutive_required``
           verdicts (newest-first) must all be non-ok. One ok/minor breaks the
           streak — this is what allows a stuck halt to clear naturally once
           the pipeline produces a clean verdict.
        3. **No post-unhalt grace, no active cooldown**: operator intervention
           gets breathing room before the detector re-fires.
        """
        if not self.config.halt_on_judge_serious:
            return

        if self._unhalt_grace_remaining.get(project_id, 0) > 0:
            logger.debug(
                f'Judge: skipping trend check for {project_id} — '
                f'post-unhalt grace has {self._unhalt_grace_remaining[project_id]} '
                f'cycles remaining'
            )
            return

        now = datetime.now(UTC)
        cooldown_until = self._halt_cooldown_until.get(project_id)
        if cooldown_until and cooldown_until > now:
            logger.debug(
                f'Judge: halt cooldown for {project_id} active until '
                f'{cooldown_until.isoformat()}'
            )
            return

        window_hours = float(self.config.halt_trend_window_hours)
        window_start = now - timedelta(hours=window_hours)
        windowed = sorted(
            (v for v in verdicts if v.reviewed_at >= window_start),
            key=lambda v: v.reviewed_at,
            reverse=True,
        )

        consecutive_required = max(1, self.config.halt_trend_consecutive_required)
        if len(windowed) < consecutive_required:
            return
        if not all(
            v.severity in ('moderate', 'serious')
            for v in windowed[:consecutive_required]
        ):
            return

        non_ok_in_window = [
            v for v in windowed if v.severity in ('moderate', 'serious')
        ]
        if len(non_ok_in_window) < self.config.halt_trend_moderate_count:
            return

        reason = (
            f'Trend: {len(non_ok_in_window)} non-ok verdicts in last '
            f'{window_hours:g}h; {consecutive_required} consecutive most-recent'
        )
        await self._apply_halt(project_id, reason=reason)
        logger.error(f'Judge: error trend detected for project {project_id} — {reason}')

    async def _apply_halt(self, project_id: str, *, reason: str) -> None:
        """Common path for trend-halt and serious-verdict-halt."""
        now = datetime.now(UTC)
        cooldown_until = now + timedelta(seconds=float(self.config.halt_cooldown_seconds))
        self._halted_projects.add(project_id)
        self._halt_cooldown_until[project_id] = cooldown_until
        self._unhalt_grace_remaining.pop(project_id, None)
        try:
            await self.journal.set_halt(
                project_id,
                halted_at=now,
                cooldown_until=cooldown_until,
                reason=reason,
            )
        except Exception:
            logger.warning('Judge._apply_halt: journal.set_halt failed', exc_info=True)

    def is_halted(self, project_id: str) -> bool:
        return project_id in self._halted_projects

    def unhalt_grace_remaining(self, project_id: str) -> int:
        return self._unhalt_grace_remaining.get(project_id, 0)

    async def consume_grace_cycle(self, project_id: str) -> int:
        """Decrement post-unhalt grace counter by one cycle.

        Returns the remaining count AFTER the decrement (0 if already expired).
        Callers (the harness) invoke this at the start of a cycle so grace is
        measured in cycles actually attempted, not real time.
        """
        if self._unhalt_grace_remaining.get(project_id, 0) <= 0:
            return 0
        try:
            remaining = await self.journal.decrement_unhalt_grace(project_id)
        except Exception:
            logger.warning(
                'Judge.consume_grace_cycle: journal decrement failed', exc_info=True,
            )
            remaining = max(0, self._unhalt_grace_remaining[project_id] - 1)
        if remaining <= 0:
            self._unhalt_grace_remaining.pop(project_id, None)
        else:
            self._unhalt_grace_remaining[project_id] = remaining
        return remaining

    async def unhalt(self, project_id: str) -> None:
        """Clear the halt and seed the post-unhalt grace counter.

        Grace is measured in cycles (decremented by the harness) so a manual
        unhalt gets a fixed number of fresh cycles to accumulate clean verdicts
        before the trend detector re-engages. Without this, historical moderates
        immediately re-halt on the next cycle.
        """
        was_halted = project_id in self._halted_projects
        self._halted_projects.discard(project_id)
        self._halt_cooldown_until.pop(project_id, None)

        grace = max(int(self.config.halt_grace_cycles), 0)
        if grace > 0:
            self._unhalt_grace_remaining[project_id] = grace
        else:
            self._unhalt_grace_remaining.pop(project_id, None)

        try:
            await self.journal.clear_halt(
                project_id,
                unhalted_at=datetime.now(UTC),
                grace_cycles=grace,
            )
        except Exception:
            logger.warning('Judge.unhalt: journal.clear_halt failed', exc_info=True)

        if was_halted and self._on_unhalt_cb is not None:
            try:
                result = self._on_unhalt_cb(project_id)
                if hasattr(result, '__await__'):
                    await result  # type: ignore[misc]
            except Exception:
                logger.warning('Judge.unhalt: on_unhalt_cb raised', exc_info=True)
