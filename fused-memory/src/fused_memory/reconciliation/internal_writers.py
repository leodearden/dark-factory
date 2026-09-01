"""House-operated memory-writer registry (task 3138, PRD §9 leaf μ).

Answers exactly one question: **is this ``agent_id`` a recognised house
writer?**  Two consumers share the answer:

* ``flag_dedup.filter_style_only_authorship_flags`` — the deterministic gate
  that drops a possible-injection / fabricated-content flag whose cited
  entries turn out to have been written by our own agents.
* ``prompts/stage1.STAGE1_SYSTEM_PROMPT`` — which interpolates
  ``INTERNAL_WRITER_POPULATION_NOTE`` so the agent-facing policy states the
  population rather than hand-transcribing it (INV-5).

**Incident this closes.** In reify esc-5564-1, Stage 1 flagged its *own*
earlier consolidator output (agent_id ``recon-stage-memory_consolidator``) as
"possibly injected/fabricated" because the imperative writing style looked
foreign to it.  Writing style is not evidence of authorship; the stored
``agent_id`` is.  The heuristic that produced that flag is emergent LLM
behaviour with no committed anchor, so the fix has to author the guardrail on
both sides — a prompt policy up front and this registry-backed code gate as
the backstop.

**Why a new registry rather than reusing a near-neighbour.**  Three existing
allowlists answer different questions and each would give a wrong answer here:

* ``shared.task_transitions.derive_actor_class`` maps every *unrecognised*
  agent_id to ``ActorClass.HUMAN`` as its safe-open default, so
  ``claude-task-3138-architect`` and ``attacker-xyz`` classify identically.
  It structurally cannot separate known-house from unknown — which is the
  entire gate.
* ``stats_verifier._KNOWN_STAGE_AGENT_IDS`` covers only the three exact
  recon-stage ids, missing ``claude-task-*`` / ``orchestrator-task-*`` /
  ``claude-interactive``.
* ``standing_decision_writer.HUMAN_AUTHORED_AGENT_ID_PREFIXES`` is
  ``('claude-interactive',)`` and *deliberately* excludes agent/stage ids to
  preserve an under-suppression bias (PRD decision 10); widening it would
  silently change standing-decision authorization.

This is not INV-5 lockstep duplication: "is this a recognised house writer?"
can evolve independently of the authority taxonomy without the two needing to
agree byte-for-byte.

**Adding a prefix here widens what the gate refuses to flag.**  Every entry
must therefore be a house-operated writer family — an agent_id space this
deployment itself mints.  A prefix that could be claimed by an outside writer
would let injected content clear the gate by naming itself accordingly.
"""

from __future__ import annotations

from typing import Any

# An ``agent_id`` counts as house-operated iff it starts with one of these.
# Prefixes (not exact ids) because every family below is open-ended: the
# recon stages, per-task agents, and orchestrator workers all mint ids with a
# variable suffix.  Sourced from the CLAUDE.md write-tagging convention plus
# the runtime ids observed on reconciliation writes.
INTERNAL_WRITER_AGENT_ID_PREFIXES: tuple[str, ...] = (
    # Reconciliation stage writers — the esc-5564-1 population itself
    # (``recon-stage-memory_consolidator``, ``recon-stage-task_knowledge_sync``,
    # …).
    'recon-stage-',
    # Defensive: the ``reconciliation-stage-N`` spelling used in CLAUDE.md's
    # write-tagging examples and in older memory rows, kept so a
    # doc-convention write is not read as foreign.
    'reconciliation-stage',
    # Orchestrator-dispatched per-task agents (``claude-task-3138-architect``,
    # ``claude-task-7``, …).
    'claude-task-',
    # Orchestrator worker writes tagged by task rather than by role.
    'orchestrator-task-',
    # The documented interactive/operator agent id.
    'claude-interactive',
)


def is_internal_writer(agent_id: Any) -> bool:
    """True iff *agent_id* is a string naming a recognised house writer.

    Fail direction is **positive-recognition-only**: this predicate gates the
    DROP of a security-shaped flag, so anything not positively recognised —
    a foreign id, ``None``, ``''``, or a non-string — returns False and leaves
    the flag standing.  Unknown authorship is unknown, never internal.
    """
    if not isinstance(agent_id, str) or not agent_id:
        return False
    return any(agent_id.startswith(prefix) for prefix in INTERNAL_WRITER_AGENT_ID_PREFIXES)


# Agent-facing rendering of the population above, built by joining the tuple so
# the Stage-1 prompt interpolates the registry instead of restating it — adding
# a family below can never leave the prompt stale (INV-5).  Mirrors the
# ``standing_decision_writer._ARM1_REMEDY`` render-from-constant idiom.
INTERNAL_WRITER_POPULATION_NOTE: str = (
    'an agent_id starting with any of: '
    + ', '.join(f'`{prefix}`' for prefix in INTERNAL_WRITER_AGENT_ID_PREFIXES)
)
