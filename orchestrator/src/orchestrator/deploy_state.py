"""Orchestrator-domain deploy state machine (task 2239, PRD §5.2 DS-1..4).

The schema (``DeployPhase``, ``DeployState``, ``VerifyBaseline``) lives
shared-visible in ``shared.deploy_state`` — see that module's docstring for
why (the fused-memory process must import the SAME class to populate its
own ``shared.task_metadata`` sub-model registry, and it cannot import
``orchestrator``). This module re-exports that schema — importing it
triggers the shared registration in the orchestrator process — and adds
the orchestrator-only state machine: the ``_LEGAL`` transition table and
DS-2's ``enforce_transition``.
"""

from __future__ import annotations

from shared.deploy_state import DeployPhase, DeployState, VerifyBaseline

__all__ = [
    'DeployPhase',
    'DeployState',
    'VerifyBaseline',
    'is_legal_transition',
]

# Only legal (True) edges are stored; any pair absent from this table is
# illegal by construction (is_legal_transition defaults to False). Per the
# PRD §5.2 chart (scheduled→ran→{verified|failed|escalated}→done) plus the
# DeterministicRunner field-combo presets (CLAUDE.md "Field-combo presets").
_LEGAL: dict[tuple[DeployPhase, DeployPhase], bool] = {
    # scheduled -> ran: the deploy script was launched (before_done_ran_at).
    (DeployPhase.SCHEDULED, DeployPhase.RAN): True,
    # ran -> verified: fresh-PID verify confirmed the deploy took effect.
    (DeployPhase.RAN, DeployPhase.VERIFIED): True,
    # ran -> failed: the deploy script itself failed (non-zero exit/timeout).
    (DeployPhase.RAN, DeployPhase.FAILED): True,
    # ran -> escalated: act-then-ask (always_escalates=True) files its
    # born-at-L2 gate right after the action runs, without a verify leg.
    (DeployPhase.RAN, DeployPhase.ESCALATED): True,
    # verified -> done: auto-deploy's happy path (before_done, not
    # always_escalates) — no human gate required.
    (DeployPhase.VERIFIED, DeployPhase.DONE): True,
    # verified -> escalated: act-then-ask still gates on human resolution
    # even after a successful verify.
    (DeployPhase.VERIFIED, DeployPhase.ESCALATED): True,
    # failed -> escalated: a deploy failure always routes to a born-at-L2
    # escalation (DS-2 loudness — never silently dropped).
    (DeployPhase.FAILED, DeployPhase.ESCALATED): True,
    # escalated -> done: the human resolved the gate/failure escalation.
    (DeployPhase.ESCALATED, DeployPhase.DONE): True,
}


def is_legal_transition(old: DeployPhase, new: DeployPhase) -> bool:
    """Whether ``old -> new`` is a legal deploy-phase transition."""
    return _LEGAL.get((old, new), False)
