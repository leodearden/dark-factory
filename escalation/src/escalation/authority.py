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
"""

from __future__ import annotations

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
