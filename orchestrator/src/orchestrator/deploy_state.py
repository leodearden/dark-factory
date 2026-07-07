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

from collections.abc import Callable

from shared.deploy_state import DeployPhase, DeployState, VerifyBaseline

__all__ = [
    'DeployPhase',
    'DeployState',
    'VerifyBaseline',
    'IllegalDeployTransition',
    'TransitionEscalationSink',
    'enforce_transition',
    'is_legal_transition',
]

# Legal edges are membership in this set; any pair absent is illegal by
# construction (is_legal_transition defaults to False). Per the PRD §5.2
# chart (scheduled→ran→{verified|failed|escalated}→done) plus the
# DeterministicRunner field-combo presets (CLAUDE.md "Field-combo presets").
#
# plans/harness-supervision-prd.md §5.2 sketches this as
# `_LEGAL: dict[(old, new), bool]` — illustrative shorthand for the same
# lookup table. `_LEGAL` itself is not exported (absent from __all__);
# is_legal_transition is the only public contract, so the table is a
# frozenset of legal pairs rather than a dict whose value is `True` on
# every entry by construction (there is no meaningful False value to store —
# illegality is absence).
_LEGAL: frozenset[tuple[DeployPhase, DeployPhase]] = frozenset((
    # scheduled -> ran: the deploy script was launched (before_done_ran_at).
    (DeployPhase.SCHEDULED, DeployPhase.RAN),
    # ran -> verified: fresh-PID verify confirmed the deploy took effect.
    (DeployPhase.RAN, DeployPhase.VERIFIED),
    # ran -> failed: the deploy script itself failed (non-zero exit/timeout).
    (DeployPhase.RAN, DeployPhase.FAILED),
    # ran -> escalated: act-then-ask (always_escalates=True) files its
    # born-at-L2 gate right after the action runs, without a verify leg.
    (DeployPhase.RAN, DeployPhase.ESCALATED),
    # verified -> done: auto-deploy's happy path (before_done, not
    # always_escalates) — no human gate required.
    (DeployPhase.VERIFIED, DeployPhase.DONE),
    # verified -> escalated: act-then-ask still gates on human resolution
    # even after a successful verify.
    (DeployPhase.VERIFIED, DeployPhase.ESCALATED),
    # failed -> escalated: a deploy failure always routes to a born-at-L2
    # escalation (DS-2 loudness — never silently dropped).
    (DeployPhase.FAILED, DeployPhase.ESCALATED),
    # escalated -> done: the human resolved the gate/failure escalation.
    (DeployPhase.ESCALATED, DeployPhase.DONE),
))


def is_legal_transition(old: DeployPhase, new: DeployPhase) -> bool:
    """Whether ``old -> new`` is a legal deploy-phase transition."""
    return (old, new) in _LEGAL


class IllegalDeployTransition(RuntimeError):
    """Raised by :func:`enforce_transition` on an illegal deploy-phase edge."""


# ζ wires the concrete sink — a born-at-L2 orchestrator-deterministic-sentinel
# EscalationQueue-backed callable (mirroring deterministic_runner's Escalation
# filing pattern). Required (no default) so DS-2 loudness cannot be bypassed.
TransitionEscalationSink = Callable[[str, DeployPhase, DeployPhase], None]


def enforce_transition(
    old: DeployPhase | None,
    new: DeployPhase,
    *,
    task_id: str,
    escalation_sink: TransitionEscalationSink,
) -> None:
    """Enforce DS-2: an illegal deploy-phase edge files a loud escalation, then raises.

    ``old is None`` is an initial write, not a transition, and is always
    legal. Otherwise an illegal ``old -> new`` edge files via
    ``escalation_sink`` FIRST (crash-safe file-before-raise ordering — the
    loud signal is emitted even if the subsequent exception propagation is
    interrupted) and THEN raises :class:`IllegalDeployTransition`.

    The raise is unconditional (``finally``, not a bare follow-on
    statement): if ``escalation_sink`` itself raises, the caller must still
    see :class:`IllegalDeployTransition` rather than the sink's exception —
    a transient escalation-delivery failure must not change the observed
    error type. The sink's exception is preserved as ``__context__`` on the
    raised :class:`IllegalDeployTransition` (standard implicit chaining),
    so it remains visible in the traceback rather than being swallowed.
    """
    if old is None or is_legal_transition(old, new):
        return
    try:
        escalation_sink(task_id, old, new)
    finally:
        raise IllegalDeployTransition(f'illegal deploy transition {old} -> {new} for task {task_id}')
