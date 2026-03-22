"""Async LLM-as-judge that evaluates reconciliation run quality."""

import json
import logging
import os
from datetime import UTC, datetime

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.models.reconciliation import JudgeVerdict, StageReport, VerdictAction, VerdictSeverity
from fused_memory.reconciliation.journal import ReconciliationJournal
from fused_memory.reconciliation.prompts.judge import JUDGE_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class Judge:
    """Async LLM reviewer that evaluates reconciliation run quality."""

    def __init__(self, config: ReconciliationConfig, journal: ReconciliationJournal,
                 usage_gate=None):
        self.config = config
        self.journal = journal
        self._halted_projects: set[str] = set()
        self._usage_gate = usage_gate

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

            prompt = self._build_review_prompt(run, entries, recent_verdicts, combined_actions)
            response_text = await self._call_llm(prompt)
            verdict = self._parse_verdict(response_text, run_id)

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

            # Act on verdict
            if verdict.severity == 'moderate':
                verdict.action_taken = VerdictAction.rollback
                logger.warning(f'Judge: moderate issues in run {run_id}, recommending rollback')
            elif verdict.severity == 'serious':
                if self.config.halt_on_judge_serious:
                    self._halted_projects.add(run.project_id)
                    verdict.action_taken = VerdictAction.halt
                    logger.error(
                        f'Judge: SERIOUS issues in run {run_id}, halting project {run.project_id}'
                    )

            # Check error trends
            all_verdicts = recent_verdicts + [verdict]
            await self._check_error_trends(run.project_id, all_verdicts)

            return verdict

        except Exception as e:
            logger.error(f'Judge review failed for run {run_id}: {e}')
            return None

    def _build_review_prompt(self, run, entries, recent_verdicts,
                             combined_actions: list[dict] | None = None) -> str:
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

        verdict_history = []
        for v in recent_verdicts:
            verdict_history.append({
                'run_id': v.run_id,
                'severity': v.severity,
                'findings_count': len(v.findings),
                'reviewed_at': v.reviewed_at.isoformat(),
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

### Recent Judge Verdicts (trend context)
{json.dumps(verdict_history, indent=2, default=str)}

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
                return ''

            result = json.loads(stdout_text)
            return result.get('result', '')

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

    async def _check_error_trends(self, project_id: str, verdicts: list[JudgeVerdict]) -> None:
        """Detect rising error rates across recent runs."""
        recent = verdicts[-10:]
        non_ok = [v for v in recent if v.severity != 'ok']
        if len(non_ok) >= 5 and self.config.halt_on_judge_serious:
                self._halted_projects.add(project_id)
                logger.error(
                    f'Judge: error trend detected for project {project_id} '
                    f'({len(non_ok)}/10 non-ok verdicts), halting'
                )

    def is_halted(self, project_id: str) -> bool:
        return project_id in self._halted_projects

    def unhalt(self, project_id: str) -> None:
        self._halted_projects.discard(project_id)
