"""Server-side identity-derived escalation authority policy.

PRD ``plans/task-status-authority-prd.md`` contract C8 / decision D7
(findings 10.1/10.2). Closes the general-case hole left by 2041's
header-opt-in capability guard: without this module, ANY caller other than
the patched auto-watcher could resolve/promote at any level simply by
omitting the X-Escalation-Levels header (default-open).

These tables constrain ONLY identified callers whose identity is a key of
``ROLE_LEVEL_ALLOWLIST`` / a member of ``PROMOTE_ALLOWED``. A header-less
connection (no X-Escalation-Identity) — the esc-2087-2 human-channel
guarantee — is NEVER narrowed by this module and keeps full authority.
Policy is DATA at module scope (mirrors the existing RESOLVE_ACTIONS /
_DISMISS_ACTIONS / _HARNESS_SENTINEL_ROLE_PREFIXES convention in
``escalation.server``), not an ``if resolved_by == watcher`` fork.

Layer direction: ``escalation`` is the lower fleet-wide package and must NOT
module-level import ``orchestrator`` (the existing orchestrator imports in
``escalation.server`` are deliberately lazy/in-function, optional). So the
canonical watcher identity string below is DUPLICATED — not imported — from
``orchestrator.harness._WATCHER_ESCALATION_HEADERS['X-Escalation-Identity']``,
and pinned in lockstep by a cross-layer TEST import in
``tests/test_authority.py``. If either string changes, update both sides
together.

Task 2630 adds a second, narrower policy on top of the above: an
above-ceiling ``close_only`` carve-out for the auto-watcher at L2 (see
:func:`l2_auto_close_class` below). The ceiling above remains the PRIMARY
gate — ``escalation.server.resolve_issue`` only consults the carve-out when
the ceiling would otherwise deny the call (``rec.level not in effective``),
so a non-``close_only`` action, a non-watcher identity, or a level other
than 2 never reaches it. The carve-out enforces STRUCTURAL evidence
presence — case-insensitive regex markers required in the resolution text —
not semantic verification: the escalation server cannot itself run a git
merge-base, probe live infra, or call ``get_task``, so it can only require
that the watcher QUOTED the proof, trusting the read-only watcher to have
done the check. The denylist (``L2_AUTO_CLOSE_DENY_CATEGORIES`` /
``L2_AUTO_CLOSE_DENY_ROLES``) is applied BEFORE the allowlist, guaranteeing
the born-at-L2 human gates (design_concern / milestone_gate categories, the
``orchestrator-deterministic`` sentinel role) are never auto-closable even
when a class predicate would otherwise match.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Must equal orchestrator.harness._WATCHER_ESCALATION_HEADERS['X-Escalation-Identity'].
# Duplicated (not imported) to preserve the escalation -> orchestrator layer
# direction; pinned in lockstep by tests/test_authority.py.
_WATCHER_AUTO_IDENTITY = 'orchestrator-escalation-watcher-auto'

# Identity -> the set of escalation levels that identity may resolve/park.
# A present X-Escalation-Levels header may only NARROW within this ceiling,
# never widen it (see escalation.server resolve_issue). Identities absent
# from this mapping fall back to the 2041 header-opt-in behaviour.
ROLE_LEVEL_ALLOWLIST: dict[str, frozenset[int]] = {
    _WATCHER_AUTO_IDENTITY: frozenset({0, 1}),
}

# Identities permitted to mint a new L2 via promote_to_l2. Header-less
# (absent identity) callers are unaffected — this only gates identified
# callers not in this set.
PROMOTE_ALLOWED: frozenset[str] = frozenset({_WATCHER_AUTO_IDENTITY})


# ---------------------------------------------------------------------------
# L2 auto-close carve-out (task 2630) — a narrow, evidence-gated exception
# ABOVE the {0,1} ceiling above, scoped to the auto-watcher identity only.
# ---------------------------------------------------------------------------

# The only resolve_issue action ever carved out above the ceiling. Every
# other action (resume/restart/park/abandon) stays level_forbidden at L2.
L2_AUTO_CLOSE_ACTION: str = 'close_only'

# Categories that are NEVER auto-closable regardless of class match — the
# born-at-L2 human-judgment categories (design decisions, milestone
# predicate gates). Checked BEFORE the allowlist (denylist-first).
L2_AUTO_CLOSE_DENY_CATEGORIES: frozenset[str] = frozenset({'design_concern', 'milestone_gate'})

# Agent roles that are NEVER auto-closable regardless of category/evidence —
# the born-at-L2 human-gate sentinel role (deterministic always_escalates /
# operator / acceptance / milestone predicate gates all file under this role
# per CLAUDE.md's "Deterministic task kind" section). Checked BEFORE the
# allowlist (denylist-first).
L2_AUTO_CLOSE_DENY_ROLES: frozenset[str] = frozenset({'orchestrator-deterministic'})


@dataclass(frozen=True)
class _L2CloseClass:
    """One allowlisted auto-close class: a record-shape predicate plus a
    structural-evidence requirement.

    ``categories`` / ``agent_roles`` are ``None`` when the class does not
    constrain that dimension (category-agnostic / role-agnostic); otherwise
    a frozenset the record's ``category`` / ``agent_role`` must belong to.

    ``evidence`` is a tuple of requirement GROUPS: every group must have at
    least one matching alternative present in the resolution text (AND
    across groups), and each group is itself a tuple of case-insensitive
    regex alternatives (OR within a group).
    """

    name: str
    categories: frozenset[str] | None
    agent_roles: frozenset[str] | None
    evidence: tuple[tuple[str, ...], ...]

    def matches_record(self, category: str, agent_role: str) -> bool:
        """Return True if *category* / *agent_role* satisfy this class's shape predicate."""
        if self.categories is not None and category not in self.categories:
            return False
        return self.agent_roles is None or agent_role in self.agent_roles

    def evidence_present(self, resolution: str) -> bool:
        """Return True if every evidence group has >=1 alternative matching *resolution*."""
        return all(
            any(re.search(pattern, resolution, re.IGNORECASE) for pattern in group)
            for group in self.evidence
        )


# Ordered MOST-SPECIFIC-FIRST: a record can satisfy more than one class's
# shape predicate (e.g. an infra_issue filed by orchestrator-main-sweep
# satisfies both (a)'s and (b)'s predicates) — first-match-wins picks the
# sharpest attribution when its stricter evidence is present, falling back
# to the more general class otherwise.
#
#   (a) superseded_main_sweep — mirrors harness._close_superseded_main_sweep_escalations:
#       requires BOTH a newer sweep-escalation id (``esc-...``) AND proof the
#       failing tip is an ancestor of a now-green main.
#   (b) self_cleared_infra — mirrors harness._revalidate_open_deterministic_escalation:
#       requires a quoted live-probe key=value liveness token.
#   (c) stale_task_scoped — mirrors harness._revalidate_open_l2: category-agnostic,
#       requires a live get_task status citation showing the subject task went
#       terminal / re-scoped / re-dispatched.
L2_AUTO_CLOSE_ALLOWLIST: tuple[_L2CloseClass, ...] = (
    _L2CloseClass(
        name='superseded_main_sweep',
        categories=None,
        agent_roles=frozenset({'orchestrator-main-sweep'}),
        evidence=(
            (r'esc-\S+',),
            (r'is[- ]ancestor', r'merge-base', r'ancestor of', r'verifies clean', r'gate re-run'),
        ),
    ),
    _L2CloseClass(
        name='self_cleared_infra',
        categories=frozenset({'infra_issue'}),
        agent_roles=None,
        evidence=(
            (r'[\w.]+\s*=\s*\S+',),
        ),
    ),
    _L2CloseClass(
        name='stale_task_scoped',
        categories=None,
        agent_roles=None,
        evidence=(
            (
                r'status\s*[=:]\s*(done|cancelled)',
                r'\bis\s+(done|cancelled)\b',
                r're-?scoped',
                r're-?dispatched',
            ),
        ),
    ),
)


def l2_auto_close_class(
    *,
    identity: str | None,
    level: int,
    action: str,
    category: str,
    agent_role: str,
    resolution: str | None,
) -> str | None:
    """Return the allowlisted auto-close class name matched by this call, or ``None``.

    Re-gates on ``identity``/``level``/``action`` even though
    ``escalation.server.resolve_issue`` only consults this function inside
    its existing ceiling-forbidden branch: a header-less human connection
    (``identity is None``) or any identity/level/action combination outside
    the narrow watcher/L2/close_only triple returns ``None`` immediately, so
    such callers are unaffected byte-for-byte by this carve-out.

    The denylist (``L2_AUTO_CLOSE_DENY_CATEGORIES`` / ``L2_AUTO_CLOSE_DENY_ROLES``)
    is applied BEFORE the allowlist, so a born-at-L2 human gate is never
    auto-closable even when a class predicate would otherwise match (e.g. an
    ``orchestrator-deterministic``-filed ``infra_issue`` is blocked by the
    role denylist despite matching class (b)'s category).

    Returns the matched class's ``name`` (for the caller's audit trail /
    rotation-digest line) or ``None`` when no class matches — the caller
    (``resolve_issue``) treats ``None`` as "stay forbidden".
    """
    if identity != _WATCHER_AUTO_IDENTITY or level != 2 or action != L2_AUTO_CLOSE_ACTION:
        return None
    if category in L2_AUTO_CLOSE_DENY_CATEGORIES or agent_role in L2_AUTO_CLOSE_DENY_ROLES:
        return None
    text = resolution or ''
    for klass in L2_AUTO_CLOSE_ALLOWLIST:
        if klass.matches_record(category, agent_role) and klass.evidence_present(text):
            return klass.name
    return None
