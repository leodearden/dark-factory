"""The zero-work failure shape: an agent run that failed before doing any work.

Such a run never reached a model turn and billed nothing, so dispatching it
again cannot double-bill. That retry is the only thing standing between a
sub-second transport glitch and a terminally blocked task (task 3143, the
2026-07-28 ~16:31Z incident cluster: turns=0, $0.00, nothing on disk).

Deliberately EXCLUDED failure kinds, because a second dispatch buys nothing:

- TIMED_OUT: burned the full wall clock; the next attempt would too.
- MAX_TURNS: the run saturated its budget doing real work.
- API_ERROR: provider-side; the scheduler's transient requeue lane owns the
  pacing, and an immediate redispatch multiplies load on a degraded provider.
- MODEL_NOT_FOUND: deterministic; the model does not exist for this account.

The transcript is authoritative. On the two included arms the CLI's result
JSON is never parsed, so ``turns`` and ``cost_usd`` are 0 by construction and
cannot testify that nothing ran. ``transcript_turns`` can: a run SIGKILLed
from outside (OOM killer, so ``timed_out`` is False) after productive turns
arrives with the same kind but a positive transcript count, and redispatching
it would double-bill a full run. ``None`` (transcript never written or never
read) counts as zero work.
"""

from __future__ import annotations

from shared.cli_invoke import AgentFailureKind, AgentResult, classify_agent_failure

ZERO_WORK_FAILURE_KINDS: frozenset[AgentFailureKind] = frozenset({
    AgentFailureKind.EMPTY_OUTPUT,
    AgentFailureKind.CLI_INPUT_REJECTED,
})


def is_zero_work_failure(result: AgentResult) -> bool:
    """Return True when *result* failed before any model turn ran."""
    return (
        classify_agent_failure(result).kind in ZERO_WORK_FAILURE_KINDS
        and not (result.transcript_turns or 0)
    )
