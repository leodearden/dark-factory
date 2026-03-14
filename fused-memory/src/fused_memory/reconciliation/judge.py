"""Async LLM-as-judge that evaluates reconciliation run quality."""

import json
import logging
from datetime import datetime, timezone

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.models.reconciliation import JudgeVerdict
from fused_memory.reconciliation.journal import ReconciliationJournal
from fused_memory.reconciliation.prompts.judge import JUDGE_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class Judge:
    """Async LLM reviewer that evaluates reconciliation run quality."""

    def __init__(self, config: ReconciliationConfig, journal: ReconciliationJournal):
        self.config = config
        self.journal = journal
        self._halted_projects: set[str] = set()

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
            recent_verdicts = await self.journal.get_recent_verdicts(
                run.project_id, limit=10
            )

            prompt = self._build_review_prompt(run, entries, recent_verdicts)
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
                verdict.action_taken = 'rollback'
                logger.warning(f'Judge: moderate issues in run {run_id}, recommending rollback')
            elif verdict.severity == 'serious':
                if self.config.halt_on_judge_serious:
                    self._halted_projects.add(run.project_id)
                    verdict.action_taken = 'halt'
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

    def _build_review_prompt(self, run, entries, recent_verdicts) -> str:
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
            stage_reports[stage_id] = {
                'actions_taken': len(report.actions_taken),
                'items_flagged': len(report.items_flagged),
                'stats': report.stats,
                'llm_calls': report.llm_calls,
                'tokens_used': report.tokens_used,
            }

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

### Journal Entries ({len(entries)} total)
{json.dumps(entry_summaries, indent=2, default=str)}

### Recent Judge Verdicts (trend context)
{json.dumps(verdict_history, indent=2, default=str)}

Review this run and provide your verdict as JSON.
"""

    async def _call_llm(self, prompt: str) -> str:
        """Single LLM call (not an agent loop)."""
        import anthropic

        if self.config.judge_llm_provider == 'anthropic':
            client = anthropic.AsyncAnthropic()
            response = await client.messages.create(
                model=self.config.judge_llm_model,
                max_tokens=4096,
                system=JUDGE_SYSTEM_PROMPT,
                messages=[{'role': 'user', 'content': prompt}],
            )
            return response.content[0].text if response.content else ''
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
                reviewed_at=datetime.now(timezone.utc),
                severity=data.get('severity', 'ok'),
                findings=data.get('findings', []),
                action_taken='none',
            )
        except (json.JSONDecodeError, IndexError, KeyError) as e:
            logger.warning(f'Failed to parse judge response: {e}')
            return JudgeVerdict(
                run_id=run_id,
                reviewed_at=datetime.now(timezone.utc),
                severity='minor',
                findings=[{
                    'issue': 'Judge response could not be parsed',
                    'severity': 'minor',
                    'recommendation': 'Manual review recommended',
                }],
                action_taken='none',
            )

    async def _check_error_trends(self, project_id: str, verdicts: list[JudgeVerdict]) -> None:
        """Detect rising error rates across recent runs."""
        recent = verdicts[-10:]
        non_ok = [v for v in recent if v.severity != 'ok']
        if len(non_ok) >= 5:
            if self.config.halt_on_judge_serious:
                self._halted_projects.add(project_id)
                logger.error(
                    f'Judge: error trend detected for project {project_id} '
                    f'({len(non_ok)}/10 non-ok verdicts), halting'
                )

    def is_halted(self, project_id: str) -> bool:
        return project_id in self._halted_projects

    def unhalt(self, project_id: str) -> None:
        self._halted_projects.discard(project_id)
