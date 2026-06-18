"""LLM Stage-2 adjudicator for path-scope-guard heuristic hits.

When the Stage-1 heuristic regex fires (a registered project prefix appears
in the task's title/description), this adjudicator runs a cheap, bounded LLM
call to distinguish a real misroute (the task genuinely modifies another
project's files) from an incidental false positive (the prefix appears only as
an example or reference in the task text).

FAIL-SAFE ASYMMETRY (the hard constraint):
  Only a CLEAN LLM success with structured verdict=='allow' downgrades the
  heuristic hit to "permit creation".  Every other outcome — reject, uncertain,
  empty output, timeout, malformed output, breaker open, any exception — falls
  back to the existing Stage-1 behaviour: reject + scope_violation escalation.
  This is the OPPOSITE of the curator's degrade-to-create contract.

AdjudicationVerdict.should_allow_creation == True ← ONLY when verdict=='allow'
  and not failed.  Every other state: should_allow_creation == False.

Reuses:
- ``shared.cli_invoke.invoke_with_cap_retry`` (same call shape as TaskCurator)
- ``AgentResult`` / ``is_zero_output_timeout`` / ``AllAccountsCappedException``
- Consecutive-ZOT circuit breaker (mirrors task_curator.py lines 505-543)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from shared.cli_invoke import (
    AgentResult,  # noqa: F401 (re-exported for tests)
    invoke_with_cap_retry,
    is_zero_output_timeout,
)

if TYPE_CHECKING:
    from shared.usage_gate import UsageGate

    from fused_memory.config.schema import PathScopeAdjudicatorConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JSON schema for the structured LLM output
# ---------------------------------------------------------------------------

ADJUDICATION_SCHEMA: dict = {
    'type': 'object',
    'required': ['verdict', 'reason'],
    'additionalProperties': False,
    'properties': {
        'verdict': {
            'type': 'string',
            'enum': ['allow', 'reject', 'uncertain'],
            'description': (
                'allow = confident the mention is incidental (not a real misroute); '
                'reject = confident this is a genuine misroute; '
                'uncertain = cannot determine confidently.'
            ),
        },
        'reason': {
            'type': 'string',
            'description': 'Short explanation (1-3 sentences) of why you chose this verdict.',
        },
    },
}

# ---------------------------------------------------------------------------
# System prompt — encodes the asymmetric fail-safe bias in the prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a task-routing classifier for a multi-project software factory.

Your job is to decide whether a task creation request was correctly filed in
its current project, or whether it should be routed elsewhere.

The path-scope guard has flagged this request because one or more registered
project path prefixes appear in the task's text.  Your job is to determine
whether that match is:

  allow   — the prefix appears ONLY as incidental text (example, reference,
             documentation, commentary, or explanation) and the task is NOT
             asking to modify files in the other project.  Choose this ONLY
             when you are CONFIDENT.

  reject  — the task is genuinely asking to modify, create, or delete files
             under the matched prefix (a real misroute into the wrong project).

  uncertain — you cannot determine confidently.  Use this when the task text
              is ambiguous, when there is not enough information, or whenever
              you would otherwise guess.

IMPORTANT: when in doubt, choose 'uncertain', not 'allow'.  The system is
fail-safe: 'allow' only downgrades the heuristic when you are truly confident;
'uncertain' and 'reject' both preserve the existing guard.

Respond ONLY with the JSON object specified in the output schema.
Do not add explanatory text outside the JSON.
""".strip()

# Cap-wait sanity for the adjudicator.  The adjudication runs INLINE in
# submit_task (before the ticket is persisted), so a stalled cap-wait stalls
# the client-facing create call.  We keep this tight:
#   timeout_seconds (default 60s) + cap_wait (60s) = ~120s worst-case latency
# on a capped account before the call fails safe back to the heuristic reject.
# The curator uses a longer cap-wait because it runs out-of-band; here speed
# of failure matters more than exhausting wait-for-cap credit.
_ADJUDICATOR_CAP_WAIT_SANITY_SECS: float = 60.0


# ---------------------------------------------------------------------------
# Verdict dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdjudicationVerdict:
    """Structured verdict returned by :meth:`PathScopeAdjudicator.adjudicate`.

    ``should_allow_creation`` is the primary decision gate for callers:
    - ``True`` ONLY when ``verdict == 'allow'`` AND ``failed == False``.
    - ``False`` for reject / uncertain / any failure mode.
    """

    verdict: Literal['allow', 'reject', 'uncertain'] = 'uncertain'
    reason: str = ''
    failed: bool = False
    llm_used: bool = False

    @property
    def should_allow_creation(self) -> bool:
        """Return True iff the adjudicator is confident the heuristic hit was a false positive."""
        return self.verdict == 'allow' and not self.failed


_VALID_VERDICTS = frozenset({'allow', 'reject', 'uncertain'})

# Prompt input caps — description is user-supplied and unbounded; truncate to
# keep the adjudication prompt tiny (matching the design intent) and prevent an
# oversized prompt from exhausting the per-call budget or pushing toward the
# timeout before the LLM can respond.
_MAX_TITLE_CHARS: int = 512
_MAX_DESCRIPTION_CHARS: int = 4096


# ---------------------------------------------------------------------------
# Main adjudicator
# ---------------------------------------------------------------------------


class PathScopeAdjudicator:
    """LLM Stage-2 gate: runs ONLY on a Stage-1 heuristic HIT.

    Constructed once at server start-up (like :class:`ScopeViolationEscalator`)
    and injected into :class:`TaskInterceptor` via its ``__init__``.

    When ``config.enabled`` is False (or no adjudicator is wired in the
    interceptor) the behaviour is identical to the current Stage-1-only guard.

    The adjudicator never raises — any failure returns a fail-safe verdict.
    """

    def __init__(
        self,
        config: PathScopeAdjudicatorConfig,
        usage_gate: UsageGate | None = None,
        cwd: Path | None = None,
    ) -> None:
        self._config = config
        self._usage_gate = usage_gate
        self._cwd = cwd

        # Consecutive-ZOT circuit breaker (mirrors TaskCurator lines 505-543).
        self._consecutive_zero_output_timeouts: int = 0
        self._zero_output_breaker_open_until: float | None = None

    # ------------------------------------------------------------------
    # Circuit-breaker helpers (mirror task_curator.py 516-543)
    # ------------------------------------------------------------------

    def _breaker_open(self, now: float) -> bool:
        """Return True while the ZOT breaker is tripped and cooldown has not elapsed."""
        return (
            self._zero_output_breaker_open_until is not None
            and now < self._zero_output_breaker_open_until
        )

    def _record_zero_output_timeout(self, now: float) -> None:
        """Increment the consecutive-ZOT counter; open the breaker when threshold is reached."""
        self._consecutive_zero_output_timeouts += 1
        if (
            self._consecutive_zero_output_timeouts
            >= self._config.zero_output_breaker_threshold
        ):
            self._zero_output_breaker_open_until = (
                now + self._config.zero_output_breaker_cooldown_seconds
            )
            logger.warning(
                'path_scope_adjudicator: zero-output-breaker OPEN after %d consecutive ZOTs; '
                'short-circuiting adjudicator LLM calls for %.0fs',
                self._consecutive_zero_output_timeouts,
                self._config.zero_output_breaker_cooldown_seconds,
            )

    def _reset_breaker(self) -> None:
        """Reset the breaker on any real LLM success."""
        self._consecutive_zero_output_timeouts = 0
        self._zero_output_breaker_open_until = None

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_user_prompt(
        *,
        title: str,
        description: str,
        matched_paths: tuple[str, ...],
        project_id: str,
        suggested_project: str | None,
    ) -> str:
        paths_str = ', '.join(matched_paths) if matched_paths else '<none>'
        target = suggested_project or '<unknown>'

        # Truncate user-supplied text to keep the prompt bounded.  Very large
        # descriptions can inflate tokens, latency, and cost, or push toward the
        # timeout before a verdict is returned.
        title_text = title[:_MAX_TITLE_CHARS]
        if len(title) > _MAX_TITLE_CHARS:
            title_text += '…'

        desc_raw = description or '<no description provided>'
        desc_text = desc_raw[:_MAX_DESCRIPTION_CHARS]
        if len(desc_raw) > _MAX_DESCRIPTION_CHARS:
            desc_text += '…'

        return (
            f'Task filed in project: {project_id!r}\n'
            f'Flagged matched path prefixes: {paths_str}\n'
            f'Suggested correct project: {target!r}\n'
            f'\n'
            f'Task title:\n{title_text}\n'
            f'\n'
            f'Task description:\n{desc_text}\n'
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def adjudicate(
        self,
        *,
        title: str,
        description: str,
        matched_paths: tuple[str, ...],
        project_id: str,
        suggested_project: str | None,
        project_root: str,
    ) -> AdjudicationVerdict:
        """Run the LLM adjudication and return a structured verdict.

        NEVER raises — any failure returns a fail-safe verdict
        (should_allow_creation=False).  The caller MUST check
        ``verdict.should_allow_creation`` and reject+escalate on False.

        Args:
            title: Task title as submitted.
            description: Task description as submitted.
            matched_paths: Path prefixes matched by the Stage-1 heuristic.
            project_id: The project the task was filed in.
            suggested_project: The project the guard believes it belongs to.
            project_root: The filing project's root directory.
        """
        now = time.monotonic()

        # --- Circuit-breaker gate ---
        if self._breaker_open(now):
            logger.debug(
                'path_scope_adjudicator: breaker open; failing safe for project %r',
                project_id,
            )
            return AdjudicationVerdict(
                verdict='uncertain',
                reason='breaker-open: consecutive zero-output-timeout threshold reached',
                failed=True,
                llm_used=False,
            )

        cwd = self._cwd or Path(project_root)
        user_prompt = self._build_user_prompt(
            title=title,
            description=description,
            matched_paths=matched_paths,
            project_id=project_id,
            suggested_project=suggested_project,
        )

        try:
            agent_result: AgentResult = await invoke_with_cap_retry(
                usage_gate=self._usage_gate,
                label=f'path-scope-adjudicator[{project_id}]',
                project_id=project_id,
                prompt=user_prompt,
                system_prompt=_SYSTEM_PROMPT,
                cwd=cwd,
                model=self._config.model,
                max_turns=self._config.max_turns,
                max_budget_usd=self._config.max_budget_usd,
                disallowed_tools=['*'],
                output_schema=ADJUDICATION_SCHEMA,
                permission_mode='bypassPermissions',
                timeout_seconds=self._config.timeout_seconds,
                cap_wait_sanity_secs=_ADJUDICATOR_CAP_WAIT_SANITY_SECS,
            )
        except Exception as exc:
            logger.warning(
                'path_scope_adjudicator: invoke_with_cap_retry raised %s: %s; '
                'failing safe for project %r',
                type(exc).__name__, exc, project_id,
            )
            return AdjudicationVerdict(
                verdict='uncertain',
                reason=f'llm-exception: {type(exc).__name__}: {exc}',
                failed=True,
                llm_used=False,
            )

        # --- Handle LLM failure ---
        if not agent_result.success:
            if is_zero_output_timeout(agent_result):
                # Capture a fresh timestamp so the cooldown window is measured
                # from when the failure was observed, not from before the
                # (potentially 60s) hung LLM call started — mirrors task_curator.py:935-938.
                self._record_zero_output_timeout(time.monotonic())
            reason = (
                f'llm-failed: timed_out={agent_result.timed_out} '
                f'subtype={agent_result.subtype!r}'
            )
            logger.warning(
                'path_scope_adjudicator: LLM call failed for project %r — %s; failing safe',
                project_id, reason,
            )
            return AdjudicationVerdict(
                verdict='uncertain',
                reason=reason,
                failed=True,
                llm_used=True,
            )

        # --- Success: reset breaker and parse output ---
        self._reset_breaker()

        structured = agent_result.structured_output
        if not isinstance(structured, dict):
            logger.warning(
                'path_scope_adjudicator: structured_output is not a dict (%r); '
                'failing safe for project %r',
                type(structured).__name__, project_id,
            )
            return AdjudicationVerdict(
                verdict='uncertain',
                reason='malformed-output: structured_output is not a dict',
                failed=True,
                llm_used=True,
            )

        raw_verdict = structured.get('verdict', '')
        reason = str(structured.get('reason', ''))
        if raw_verdict not in _VALID_VERDICTS:
            logger.warning(
                'path_scope_adjudicator: unknown verdict %r; normalising to "uncertain"',
                raw_verdict,
            )
            return AdjudicationVerdict(
                verdict='uncertain',
                reason=f'malformed-output: unknown verdict {raw_verdict!r}; reason={reason}',
                failed=True,
                llm_used=True,
            )

        verdict = AdjudicationVerdict(
            verdict=raw_verdict,  # type: ignore[arg-type]
            reason=reason,
            failed=False,
            llm_used=True,
        )
        logger.debug(
            'path_scope_adjudicator: verdict=%r for project %r (matched=%s): %s',
            verdict.verdict, project_id,
            ', '.join(matched_paths), reason[:200],
        )
        return verdict
