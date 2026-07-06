"""Table B — escalation-resolution-intent authority.

Implements ``plans/task-status-authority-prd.md`` contract C5 / decisions
D1, D2 (finding 10.0). This is the single authority mapping a resolve_issue
``(action, level, category)`` triple to the ``TaskEffect`` it produces —
consulted by ``escalation.server.resolve_issue`` BEFORE any record mutation
so an illegal action is rejected loudly (typed error, no write) instead of
silently.

Behaviour-preserving mapping (D2): the five ``TaskEffect`` values below are
transcribed verbatim from the current source of truth,
``orchestrator/src/orchestrator/harness.py:8259-8325``
(``_ACTION_TARGETS`` / ``_on_escalation_resolved``):
``restart`` -> ``pending``, ``park`` -> ``blocked`` (version-a: keeps the L2
escalation OPEN; NOT ``deferred``), ``abandon`` -> ``cancelled``, ``resume``
-> ``pending`` (orphan/cascade re-pend), ``close_only`` -> no status change.

G6 archive verification: a scan of the live ``data/escalations`` archive (636
records, 632 resolved/dismissed) found ``resolution_action`` values of
*exactly* ``{close_only, resume, restart, abandon, park}`` and nothing else,
occurring across escalation levels 0/1/2 and a wide, open category
vocabulary (including categories like ``milestone_gate`` that are not even
members of ``escalation.server.CATEGORIES``, which is inert/unvalidated).
``park`` at level 2 is semantically required (version-a) yet is absent from
the archive, proving archive-absence does not imply illegality. Conclusion:
the only combinations safe to reject are those whose *action* itself is
outside the five valid actions — legality is therefore NOT narrowed by
level or category today. ``ACTION_EFFECTS`` reflects this: every entry is
keyed on the wildcard sentinel :data:`ANY` for both the level-class and
category slots, leaving room for a future, more specific override (e.g. an
infra-hold interaction, ω4) without changing today's behaviour or this
module's public contract.

This module is intentionally pure: it defines the table and the
:func:`effect_for` lookup only. It does NOT rewire
``orchestrator.harness`` to consume it (ω3) and does NOT add an
infra-hold override (ω4) — both are separate, out-of-scope tasks. It also
does NOT import ``shared`` — the ``target_status`` values below are plain
lowercase string literals kept in parity with ``shared.task_statuses.
TaskStatus`` by test (see ``tests/test_action_effects.py``), not by import,
so this module adds no new production dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class _Wildcard:
    """Sentinel type for the :data:`ANY` wildcard — matches by identity only."""

    __slots__ = ()

    def __repr__(self) -> str:
        return 'ANY'


# Wildcard sentinel for the level-class / category slots of an ACTION_EFFECTS
# key. A single instance; equality and hashing are the default (identity)
# ones inherited from `object`, which is exactly what a sentinel needs.
ANY: Any = _Wildcard()

# ---------------------------------------------------------------------------
# Workflow disposition vocabulary (closed) — describes the live-workflow
# effect each of the five resolve_issue actions has today. Values are
# internal identifiers (not persisted), so their exact spelling is not a
# compatibility concern.
# ---------------------------------------------------------------------------
WORKFLOW_RESUME: str = 'resume_from_pause'
WORKFLOW_RESTART: str = 'restart_from_scratch'
WORKFLOW_PARK: str = 'park_kill_block_keep_l2_open'
WORKFLOW_ABANDON: str = 'abandon_kill_cancel'
WORKFLOW_NONE: str = 'no_effect'

WORKFLOW_DISPOSITIONS: frozenset[str] = frozenset(
    {
        WORKFLOW_RESUME,
        WORKFLOW_RESTART,
        WORKFLOW_PARK,
        WORKFLOW_ABANDON,
        WORKFLOW_NONE,
    }
)


@dataclass(frozen=True)
class TaskEffect:
    """The task-status effect and workflow disposition of a resolve_issue action.

    ``target_status`` is a plain lowercase status string (kept in parity
    with ``shared.task_statuses.TaskStatus`` by test, not by import) or
    ``None`` when the action has no task-status effect (``close_only``).
    """

    target_status: str | None
    workflow_disposition: str


# ---------------------------------------------------------------------------
# Table B. Keyed on (action, level_class, category) per contract C5's
# declared shape, but every entry today is wildcarded on level_class and
# category (see module docstring / G6 verification) — current semantics are
# action-only. Exactly the 5 actions resolve_issue's RESOLVE_ACTIONS enum
# accepts; any other action is illegal (effect_for returns None).
# ---------------------------------------------------------------------------
ACTION_EFFECTS: dict[tuple[str, object, object], TaskEffect] = {
    ('resume', ANY, ANY): TaskEffect('pending', WORKFLOW_RESUME),
    ('restart', ANY, ANY): TaskEffect('pending', WORKFLOW_RESTART),
    ('park', ANY, ANY): TaskEffect('blocked', WORKFLOW_PARK),
    ('abandon', ANY, ANY): TaskEffect('cancelled', WORKFLOW_ABANDON),
    ('close_only', ANY, ANY): TaskEffect(None, WORKFLOW_NONE),
}


def level_class(level: int) -> int:
    """Return the level-class used as the Table B lookup key for *level*.

    Faithful and non-lossy today (the identity function): ``ACTION_EFFECTS``
    wildcards the level-class slot for every entry (see module docstring),
    so no information is currently thrown away by classifying. A future
    task (ω3/ω4) may bucket levels more coarsely or add a more specific
    override without changing this function's contract.
    """
    return level


def effect_for(action: str, level: int, category: str) -> TaskEffect | None:
    """Look up the :class:`TaskEffect` for *(action, level, category)*.

    Does a most-specific -> wildcard fallback lookup: first tries the exact
    ``(action, level_class(level), category)`` key, then falls back to the
    wildcarded ``(action, ANY, ANY)`` key. Returns ``None`` when *action*
    itself is not a recognised resolve_issue action — today, level and
    category never narrow legality (see the G6 verification in the module
    docstring), so ``None`` is returned if and only if *action* is illegal.
    """
    lc = level_class(level)
    specific = ACTION_EFFECTS.get((action, lc, category))
    if specific is not None:
        return specific
    return ACTION_EFFECTS.get((action, ANY, ANY))
