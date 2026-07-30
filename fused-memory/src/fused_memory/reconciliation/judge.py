"""Async LLM-as-judge that evaluates reconciliation run quality."""

import json
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, NoReturn

from pydantic import ValidationError
from shared.cli_invoke import (
    AgentFailureKind,
    AllAccountsCappedException,
    build_failure_message,
    classify_agent_failure,
    invoke_with_cap_retry,
)

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.models.reconciliation import (
    JudgeVerdict,
    StageReport,
    VerdictAction,
    VerdictSeverity,
)
from fused_memory.reconciliation import _RECONCILIATION_STAGE_CAP_WAIT_SANITY_SECS
from fused_memory.reconciliation.journal import ReconciliationJournal
from fused_memory.reconciliation.prompts.judge import JUDGE_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


# The judge's verdict contract, carried as ``--json-schema`` rather than as prose
# in JUDGE_SYSTEM_PROMPT.  This placement is load-bearing, not cosmetic:
# ``build_claude_argv`` emits ``--json-schema`` UNCONDITIONALLY
# (cli_invoke.py:1552-1553) but DROPS ``--system-prompt-file`` on the resume path
# (:1501-1503).  A cap-retry resume (:1270-1272) therefore re-invokes the CLI
# with the schema still attached but the system prompt gone — so a prose-only
# output contract silently evaporates exactly when the run is already degraded,
# which is how a 2026-07-25 cap-storm resume produced conversational prose that
# _parse_verdict fabricated into a phantom severity=serious project halt.
#
# Both severity enums are DERIVED from VerdictSeverity so the schema and the
# pydantic field that validates the payload cannot drift.  ``summary`` is
# included because JUDGE_SYSTEM_PROMPT asks for it (JudgeVerdict has no such
# field and drops it, exactly as today) — omitting it here while still requesting
# it in prose would make prompt and schema contradict each other.
# ``additionalProperties: false`` is deliberately NOT set (no sibling schema sets
# it): rejecting a usable verdict over one extra key is strictly worse than
# accepting it.  Sibling convention — the schema lives in the module that invokes
# the CLI (task_curator.py:139, cli_stage_runner.py:95, agent_loop.py:27).
_VERDICT_SEVERITY_VALUES = [s.value for s in VerdictSeverity]

JUDGE_VERDICT_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'required': ['severity', 'findings'],
    'properties': {
        'severity': {'type': 'string', 'enum': _VERDICT_SEVERITY_VALUES},
        'findings': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'entry_id': {'type': ['string', 'null']},
                    'issue': {'type': 'string'},
                    'severity': {'type': 'string', 'enum': _VERDICT_SEVERITY_VALUES},
                    'recommendation': {'type': 'string'},
                },
            },
        },
        'summary': {'type': 'string'},
    },
}

# max_turns for the judge CLI invocation.  NOT 1: ``max_turns=1`` is incompatible
# with ``--json-schema`` because the schema mechanism burns a tool-use turn, so
# the CLI returns ``error_max_turns`` even when the structured payload is already
# attached (task_curator.py:2366-2372).  3 is the floor both migrated siblings
# use (curator ``ge=3``, path_scope_adjudicator ``ge=3``): schema tool-use +
# optional reasoning + final response.  We deliberately do NOT rely on
# cli_invoke's ``schema_salvaged`` boundary fallback (:1794-1797), which would
# make every judge run report an internal ``is_error`` and spend its turn on an
# error path.  Cost/duration exposure stays bounded by
# ``judge_cli_timeout_seconds`` (validated ≤ stage_timeout_seconds) and
# cli_invoke's ``max_budget_usd`` default.
_JUDGE_CLI_MAX_TURNS = 3


UnhaltCallback = Callable[[str], Awaitable[None] | None]
"""Fired by :meth:`Judge.unhalt` after the halt state is cleared.

The harness's implementation (``ReconciliationHarness._on_judge_unhalt``)
clears the per-process escalation dedupe sentinel AND closes the
``esc-reconciliation-halt-*`` escalation the halt opened (task 2998), so this
callback is not merely bookkeeping — skipping it leaves a pending escalation
for a halt that no longer exists.
"""


class JudgeInfraError(Exception):
    """Judge invocation produced NO usable verdict (task 2947 ask a).

    Two distinct situations, one taxonomy — in both, nothing was reviewed, so
    there is no verdict to act on:

    1. **The CLI failed** (task 2947): api_error, timeout, model-not-found,
       max-turns, ended-awaiting-background, cap-wait exhaustion, or an
       otherwise-unknown non-success.
    2. **The CLI SUCCEEDED but carried no verdict** (the 2947 residual this task
       closes): exit 0, ``is_error=false``, non-zero output tokens, yet
       ``structured_output`` was missing, empty, wrong-typed, unparseable, or
       semantically invalid.  This is the 2026-07-25 cap-resume shape — the
       resume dropped ``--system-prompt-file`` (cli_invoke.py:1501-1503) and the
       model answered with conversational prose.  Codes:
       ``cli_output_empty`` (missing/empty payload) and
       ``cli_output_unparseable`` (wrong-typed, unparseable, or invalid payload).

    Distinct from a genuinely-malformed *reachable* judge response on the
    ``anthropic``/``openai`` text path (which stays fail-closed serious via
    ``_parse_verdict``) and from a benign exit-0/empty-stdout run (which stays
    the benign ``''`` contract).

    The distinction matters because an infra failure carries NO verdict: prior
    to this taxonomy a cap-storm CLI failure was fabricated into a phantom
    ``severity=serious`` halt whose reason ("Serious verdict in run X") lied
    about a review that never happened. ``review_run`` catches this exception,
    applies bounded backoff, and only halts (truthfully, "judge-unreachable")
    after ``judge_infra_max_consecutive_failures`` consecutive occurrences.
    Carries a human-readable message (e.g. the ``build_failure_message`` dump)
    and, for case 2, a machine-readable ``code``.
    """

    def __init__(self, message: str, *, code: str | None = None):
        super().__init__(message)
        self.code = code


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
        # Per-project count of CONSECUTIVE judge transport/infra failures
        # (JudgeInfraError). Incremented on each infra failure, reset on any
        # successful transport or once a judge-unreachable halt fires. Bounds
        # the phantom-halt blast radius (task 2947 ask a).
        self._consecutive_infra_failures: dict[str, int] = {}
        # Last halt reason per project (task 2947 ask b), so the harness can
        # thread the REAL reason into the on_judge_halt escalation instead of a
        # hardcoded generic string. Set in _apply_halt, cleared in unhalt.
        self._halt_reason: dict[str, str] = {}
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
            try:
                response_text = await self._call_llm(prompt)
            except JudgeInfraError as err:
                # Transport/infra failure (cap-storm CLI failure, timeout,
                # api_error, cap-wait exhaustion, …) carries NO verdict. Do NOT
                # fabricate a phantom severity=serious halt (the defect this
                # fixes) — apply bounded backoff and only halt (truthfully) after
                # judge_infra_max_consecutive_failures in a row. This except is
                # deliberately INSIDE the broad try so its return short-circuits
                # before _parse_verdict / add_verdict.
                return await self._handle_infra_failure(run.project_id, run_id, err)

            # Successful transport (even a malformed-content verdict, which stays
            # fail-closed serious below) clears the per-project infra streak so a
            # recovered judge starts fresh.  For the claude_cli provider a success
            # that carried NO usable verdict is not "successful transport" for this
            # purpose — it raises JudgeInfraError above and is diverted before this
            # line, so it correctly keeps counting toward the streak.
            self._consecutive_infra_failures.pop(run.project_id, None)

            verdict = self._parse_verdict(response_text, run_id)

            # Act on verdict — must happen BEFORE persisting so the DB receives the
            # final action_taken value ('rollback', 'halt', or 'none')
            if verdict.severity == 'moderate':
                verdict.action_taken = VerdictAction.rollback
                logger.warning(f'Judge: moderate issues in run {run_id}, recommending rollback')
            elif verdict.severity == 'serious':
                if self.config.halt_on_judge_serious:
                    verdict.action_taken = VerdictAction.halt
                    # A serious verdict fabricated by _parse_verdict for
                    # genuinely-malformed REACHABLE content (success, non-empty,
                    # non-JSON) carries the code='unparseable_judge_response'
                    # marker. It still halts (fail-closed preserved) but with a
                    # truthful reason instead of the lying 'Serious verdict'
                    # string. A genuine content severity=serious verdict has no
                    # such marker and keeps the already-truthful reason. (Infra
                    # failures never reach here — they are diverted to
                    # _handle_infra_failure before _parse_verdict.)
                    #
                    # For the claude_cli provider this fabricated-serious branch
                    # is now UNREACHABLE: its verdict arrives as a validated
                    # structured payload, and a success carrying no usable
                    # verdict is diverted to _handle_infra_failure BEFORE
                    # _parse_verdict (raised in _call_judge_cli as
                    # cli_output_empty / cli_output_unparseable). The branch
                    # survives for the anthropic/openai text fallback, which has
                    # no --json-schema mechanism and so can still return prose
                    # that must be treated fail-closed.
                    is_parse_failure = any(
                        f.get('code') == 'unparseable_judge_response'
                        for f in verdict.findings
                    )
                    if is_parse_failure:
                        halt_reason = (
                            f'Unparseable judge response in run {run_id} '
                            f'(judge output present but not valid JSON)'
                        )
                    else:
                        halt_reason = f'Serious verdict in run {run_id}'
                    await self._apply_halt(run.project_id, reason=halt_reason)
                    logger.error(
                        f'Judge: SERIOUS issues in run {run_id}, halting project '
                        f'{run.project_id} — {halt_reason}'
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

    async def _handle_infra_failure(
        self, project_id: str, run_id: str, err: JudgeInfraError,
    ) -> None:
        """Bounded-backoff handling for a judge transport/infra failure (task 2947).

        A JudgeInfraError means the judge CLI was unreachable/broken for this run
        — there is NO verdict to trust. Rather than fabricating a phantom
        severity=serious halt whose "Serious verdict in run X" reason lies about a
        review that never happened (the defect this fixes), count consecutive
        infra failures per project:

        - below ``config.judge_infra_max_consecutive_failures``: log a structured
          warning and return None — no verdict is stamped and no halt fires, so a
          transient outage (e.g. an account-cap storm) is given room to recover.
        - at/above the threshold: reset the counter (so a post-halt recovery
          starts clean) and, when ``halt_on_judge_serious`` is set, apply a
          TRUTHFUL 'judge-unreachable' halt — which routes the existing
          infra_issue escalation via the harness. Gated on the same master switch
          as the serious-verdict halt.

        Always returns None (review_run propagates it as the run's result). The
        infra streak is reset elsewhere on any successful transport (see
        review_run), so only genuinely-consecutive failures reach the threshold.
        """
        count = self._consecutive_infra_failures.get(project_id, 0) + 1
        self._consecutive_infra_failures[project_id] = count
        threshold = self.config.judge_infra_max_consecutive_failures

        if count < threshold:
            logger.warning(
                'reconciliation.judge_infra_failure',
                extra={
                    'run_id': run_id,
                    'project_id': project_id,
                    'consecutive_failures': count,
                    'threshold': threshold,
                    'error': str(err),
                    # None for a CLI transport failure; 'cli_output_empty' /
                    # 'cli_output_unparseable' for a success that carried no
                    # usable verdict. Lets a cap-storm-with-no-content occurrence
                    # be told apart from a CLI failure in the logs — the
                    # distinction that took a hand-dug transcript to establish on
                    # 2026-07-25.
                    'code': getattr(err, 'code', None),
                },
            )
            return None

        # Threshold reached: reset the streak so a later recovery starts clean,
        # then (if enabled) halt truthfully.
        self._consecutive_infra_failures.pop(project_id, None)
        if self.config.halt_on_judge_serious:
            reason = (
                f'judge-unreachable halt: reconciliation judge outage '
                f'(cap-storm/CLI failure) — project halted after {count} '
                f'consecutive infra failures; last error: {err}'
            )
            await self._apply_halt(project_id, reason=reason)
            logger.error(
                f'Judge: judge-unreachable — halting project {project_id} after '
                f'{count} consecutive infra failures; last error: {err}'
            )
        else:
            logger.error(
                f'Judge: {count} consecutive judge infra failures for project '
                f'{project_id}, but halt_on_judge_serious is disabled — not '
                f'halting; last error: {err}'
            )
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

    async def _call_llm(self, prompt: str) -> dict | str:
        """Single LLM call (not an agent loop).

        Returns a ``dict`` for the ``claude_cli`` provider (the schema-validated
        ``structured_output`` payload) and a ``str`` for the ``anthropic`` /
        ``openai`` providers, which call their SDKs directly and therefore have no
        ``--json-schema`` mechanism — prose is all they can return.
        ``_parse_verdict`` dispatches on the type.
        """
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

    async def _call_judge_cli(self, prompt: str) -> dict | str:
        """Delegate to shared.cli_invoke.invoke_with_cap_retry for judge evaluation.

        Returns the schema-validated verdict payload (a ``dict``) on success.  The
        ``str`` return is now ONLY ever ``''`` — the benign exit-0/empty-stdout
        legacy contract — and never model prose: the text channel is no longer
        treated as the verdict.

        The verdict contract rides ``output_schema=JUDGE_VERDICT_SCHEMA`` rather
        than prose in the system prompt, so it survives every cap-retry resume:
        ``build_claude_argv`` emits ``--json-schema`` unconditionally
        (cli_invoke.py:1552-1553) while it drops ``--system-prompt-file`` on the
        resume path (:1501-1503).

        ``disallowed_tools=['*']`` is still passed VERBATIM.  cli_invoke expands
        the wildcard into ``_REAL_BUILTIN_TOOLS_DENYLIST`` when a schema is
        present (:1533-1536) — a list that omits the synthetic
        ``StructuredOutput`` tool the schema is delivered through — so "no real
        file/bash/web/MCP tool access" is preserved while the schema tool gets
        through.  Pre-expanding here would duplicate a list documented as needing
        to stay in sync with the CLI's built-ins and would skip future central
        fixes.

        InvokeSlot owns probe-slot release and terminate_process_group on timeout.
        """
        try:
            result = await invoke_with_cap_retry(
                usage_gate=self._usage_gate,
                label=f'Reconciliation judge ({self.config.judge_llm_model})',
                prompt=prompt,
                system_prompt=JUDGE_SYSTEM_PROMPT,
                model=self.config.judge_llm_model,
                disallowed_tools=['*'],
                output_schema=JUDGE_VERDICT_SCHEMA,
                # See _JUDGE_CLI_MAX_TURNS: 1 is incompatible with --json-schema
                # (the schema mechanism burns a tool-use turn — see
                # task_curator.py:2366-2372).
                max_turns=_JUDGE_CLI_MAX_TURNS,
                permission_mode='bypassPermissions',
                timeout_seconds=float(self.config.judge_cli_timeout_seconds),
                # Task 1989 (sweep verdict): kept at explore_codebase_root, NOT
                # switched to a neutral cwd. JUDGE_SYSTEM_PROMPT rates factual
                # grounding/proportionality ("were codebase verifications used
                # when appropriate?"), which plausibly draws on the auto-loaded
                # CLAUDE.md's project architecture context.
                cwd=Path(self.config.explore_codebase_root),
                cap_wait_sanity_secs=_RECONCILIATION_STAGE_CAP_WAIT_SANITY_SECS,
            )
        except AllAccountsCappedException as e:
            # Cap-wait exhaustion (all accounts capped past the 1800s recon
            # sanity bound) is a transport/infra failure, not a verdict. Re-raise
            # as JudgeInfraError so review_run counts/backs-off/escalates it
            # identically to other transport failures, instead of letting it
            # propagate opaquely into review_run's broad except (which would
            # swallow it as a generic failure with no infra-failure accounting).
            raise JudgeInfraError(f'Claude CLI judge cap-wait exhausted: {e}') from e

        if result.success:
            # The verdict lives in the schema-validated payload, NOT in the text
            # channel: after a cap-retry resume the model can (and on 2026-07-25
            # did) emit conversational prose there, which the old
            # ``result.output.strip()`` return handed to _parse_verdict to be
            # fabricated into a phantom severity=serious halt.
            structured = result.structured_output
            if isinstance(structured, str):
                # The CLI sometimes delivers the payload as a JSON string
                # (agent_loop.py:388-392 handles the same shape).
                try:
                    structured = json.loads(structured)
                except json.JSONDecodeError as e:
                    self._raise_unusable_output(
                        result, 'cli_output_unparseable',
                        f'structured_output was an unparseable JSON string: {e}',
                    )

            # A success carrying no usable verdict is INFRA, not content: nothing
            # was reviewed, so there is nothing to be fail-closed ABOUT.  Never
            # guess a severity here — a fabricated verdict is what halts a project
            # (cf. path_scope_adjudicator.py:415-441, same "not a dict → fail
            # loudly, don't normalise silently" stance, different destination).
            if not structured:
                self._raise_unusable_output(
                    result, 'cli_output_empty',
                    'structured_output was empty/missing',
                )
            if not isinstance(structured, dict):
                self._raise_unusable_output(
                    result, 'cli_output_unparseable',
                    f'structured_output was {type(structured).__name__}, not a dict',
                )
            severity = structured.get('severity')
            if severity not in {s.value for s in VerdictSeverity}:
                self._raise_unusable_output(
                    result, 'cli_output_unparseable',
                    f'structured_output severity {severity!r} is not a VerdictSeverity',
                )
            if not isinstance(structured.get('findings', []), list):
                # JudgeVerdict construction would raise a pydantic
                # ValidationError that review_run's broad except would swallow
                # into a silent None — no verdict, no halt, no signal.
                self._raise_unusable_output(
                    result, 'cli_output_unparseable',
                    'structured_output findings is not a list',
                )
            return structured

        # Not successful: distinguish a genuine benign empty verdict from a
        # transport/infra failure via the shared failure taxonomy
        # (classify_agent_failure), so infra failures never reach
        # _parse_verdict's fabricated severity=serious path (the phantom-halt
        # defect this task fixes).
        #
        # EMPTY_OUTPUT (genuine exit-0/empty-stdout) preserves the legacy
        # "empty stdout = valid empty verdict" semantics: return '' so
        # _parse_verdict's empty-text short-circuit yields a benign minor
        # verdict.  Crucially, classify_agent_failure orders API_ERROR (rule 6)
        # ABOVE EMPTY_OUTPUT (rule 7): a cap-storm run whose stdout is empty but
        # whose api_error_status is set classifies as API_ERROR, so it is
        # correctly treated as infra rather than a benign empty verdict.
        #
        # Every other failure kind (API_ERROR / TIMED_OUT / MODEL_NOT_FOUND /
        # MAX_TURNS / ENDED_AWAITING_BACKGROUND / UNKNOWN) is a transport/infra
        # failure: raise JudgeInfraError (replacing the previous generic
        # RuntimeError) so review_run can branch on it explicitly.
        if classify_agent_failure(result).kind == AgentFailureKind.EMPTY_OUTPUT:
            return ''

        raise JudgeInfraError(build_failure_message('Claude CLI judge', result))

    def _raise_unusable_output(self, result, code: str, detail: str) -> NoReturn:
        """Log the forensics for a no-verdict CLI success, then raise JudgeInfraError.

        The structured warning is the record that made the 2026-07-25 RCA possible
        (it took a hand-dug CLI transcript to establish that the halting run had
        exited 0 with 727 output tokens and no payload).  ``schema_tool_denied``
        is included deliberately: if a future CLI change starts blocking the
        synthetic ``StructuredOutput`` tool that ``--json-schema`` rides on, that
        deny-list regression becomes visible HERE instead of silently starving
        every judge run of its verdict.  The message embeds the code and an output
        prefix so ``_handle_infra_failure``'s ``str(err)`` stays diagnostic.
        """
        logger.warning(
            'reconciliation.judge_cli_output_unusable',
            extra={
                'code': code,
                'detail': detail,
                'subtype': result.subtype,
                'output_tokens': result.output_tokens,
                'session_id': result.session_id,
                'schema_salvaged': result.schema_salvaged,
                'schema_tool_denied': result.schema_tool_denied,
                'output_prefix': result.output[:200],
            },
        )
        raise JudgeInfraError(
            f'Claude CLI judge reported success but produced no usable verdict '
            f'[{code}]: {detail}; output prefix: {result.output[:200]!r}',
            code=code,
        )

    def _verdict_from_payload(self, data: dict, run_id: str) -> JudgeVerdict:
        """Build a JudgeVerdict from an already-structured verdict payload.

        The single JudgeVerdict constructor shared by the structured
        (``claude_cli``) path and the text-parsing (``anthropic`` / ``openai``)
        fallback, so the two cannot drift on field defaults.  Keys the model has
        no field for (e.g. ``summary``, which JUDGE_SYSTEM_PROMPT asks for) are
        dropped here, exactly as they always were.
        """
        return JudgeVerdict(
            run_id=run_id,
            reviewed_at=datetime.now(UTC),
            severity=data.get('severity', VerdictSeverity.ok),
            findings=data.get('findings', []),
            action_taken=VerdictAction.none,
        )

    def _parse_verdict(self, response: str | dict, run_id: str) -> JudgeVerdict:
        """Turn a judge response into a JudgeVerdict.

        A ``dict`` is an already-structured, already-validated payload from the
        ``claude_cli`` provider (``_call_judge_cli`` reads it off
        ``AgentResult.structured_output``) — it goes straight to
        ``_verdict_from_payload`` with no text parsing at all.

        The ``str`` path below is the fallback for the ``anthropic`` / ``openai``
        providers, which call their SDKs directly and so have no ``--json-schema``
        mechanism; prose is the only contract they have.  For ``claude_cli`` that
        path is dead: a success with no usable payload is classified as infra in
        ``_call_judge_cli`` and never reaches here.
        """
        if isinstance(response, dict):
            return self._verdict_from_payload(response, run_id)

        response_text = response
        if not response_text.strip():
            # Empty output is a documented benign case, NOT a parse failure:
            # _call_judge_cli returns '' for exit-0/empty-stdout CLI runs
            # (legacy "empty stdout = valid empty verdict" semantics — see
            # the comment there) and the anthropic branch returns '' when
            # the response has no text blocks. Neither indicates a systemic
            # judge/CLI outage, so it must not be conflated with the loud
            # severity=serious path below (reserved for genuinely
            # unparseable non-empty responses) — doing so would turn a
            # quiet, successful run into a project halt when
            # halt_on_judge_serious=True. Stay at severity=minor, matching
            # the non-halting behavior this replaced for the empty case.
            return JudgeVerdict(
                run_id=run_id,
                reviewed_at=datetime.now(UTC),
                severity=VerdictSeverity.minor,
                findings=[{
                    'issue': 'Judge returned empty output',
                    'severity': 'minor',
                    'recommendation': 'Manual review recommended',
                }],
                action_taken=VerdictAction.none,
            )
        try:
            # Extract JSON from response (might be wrapped in markdown)
            text = response_text.strip()
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0].strip()
            elif '```' in text:
                text = text.split('```')[1].split('```')[0].strip()

            data = json.loads(text)
            return self._verdict_from_payload(data, run_id)
        except (json.JSONDecodeError, IndexError, KeyError, ValidationError) as e:
            logger.error(f'Failed to parse judge response: {e}')
            # severity=serious is deliberate: it routes through the existing
            # halt path (review_run() halts on config.halt_on_judge_serious;
            # _check_error_trends treats 'serious' as non-ok) so a systemic
            # judge/CLI outage surfaces loudly instead of hiding behind a
            # fabricated fail-soft 'minor' verdict. This intentionally halts
            # on a SINGLE unparseable response — review_run() acts on
            # severity=serious immediately, it does not wait for
            # _check_error_trends' multi-occurrence/consecutive window. A
            # lone parse failure is treated as sufficient grounds to halt
            # (rather than requiring a trend) because an unparseable judge
            # response means the review pipeline itself may be broken, so
            # subsequent verdicts can't be trusted to detect a trend either;
            # see test_review_run_unparseable_response_halts_when_enabled /
            # ..._no_halt_when_disabled in test_judge.py for the covered
            # behavior at both settings of halt_on_judge_serious.
            #
            # ValidationError is in the caught tuple so a SEMANTICALLY-invalid
            # payload (severity outside VerdictSeverity, wrong-typed findings)
            # cannot silently vanish: without it, JudgeVerdict's pydantic error
            # escapes to review_run's broad `except Exception` and the run
            # returns None with no verdict, no halt and no signal. The
            # claude_cli provider never reaches here — its equivalent condition
            # is classified as cli_output_unparseable infra in _call_judge_cli.
            return JudgeVerdict(
                run_id=run_id,
                reviewed_at=datetime.now(UTC),
                severity=VerdictSeverity.serious,
                findings=[{
                    'issue': f'Judge response could not be parsed: {e}',
                    'severity': 'serious',
                    # Machine-readable marker (task 2947 ask b): lets review_run
                    # pick the truthful 'Unparseable judge response' halt reason
                    # via a structured field rather than fragile substring
                    # matching. Kept on this SINGLE finding so len(findings)==1
                    # (the existing loud-not-fabricated test) still holds.
                    'code': 'unparseable_judge_response',
                    'recommendation': (
                        'This verdict was not a real review — the judge output was '
                        'unparseable. Investigate the judge LLM/CLI output before '
                        'trusting subsequent verdicts.'
                    ),
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
        # Remember the reason in-memory so the harness can thread the REAL reason
        # into the on_judge_halt escalation (task 2947 ask b).
        self._halt_reason[project_id] = reason
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

    def cooldown_expired(self, project_id: str) -> bool:
        """True iff ``project_id`` is halted AND its recorded halt cooldown has
        passed (``cooldown_until <= now``).

        The harness uses this to decide auto-resume-after-cooldown when
        ``auto_unhalt_after_cooldown`` is enabled (task 2920 deliverable c).
        Returns False if the project is not halted, if no ``cooldown_until`` was
        recorded, or if the cooldown is still in the future. Uses
        ``datetime.now(UTC)`` directly, mirroring ``_check_error_trends`` /
        ``_apply_halt`` (no injected clock).
        """
        if project_id not in self._halted_projects:
            return False
        cooldown_until = self._halt_cooldown_until.get(project_id)
        if cooldown_until is None:
            return False
        return cooldown_until <= datetime.now(UTC)

    def halt_reason(self, project_id: str) -> str | None:
        """The reason the project was last halted, or None if not halted.

        Threaded by the harness into the on_judge_halt escalation so an infra
        ('judge-unreachable …') or unparseable-content halt surfaces with its
        real reason instead of a hardcoded generic string (task 2947 ask b).
        """
        return self._halt_reason.get(project_id)

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

        Fires ``on_unhalt_cb`` when the project WAS halted; that callback also
        closes the halt's escalation record (task 2998) — see
        :data:`UnhaltCallback`. Awaitable results are awaited, and callback
        failures are logged rather than raised, so a close failure never blocks
        the unhalt itself.
        """
        was_halted = project_id in self._halted_projects
        self._halted_projects.discard(project_id)
        self._halt_cooldown_until.pop(project_id, None)
        self._halt_reason.pop(project_id, None)

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
