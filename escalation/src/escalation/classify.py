"""Escalation resolution classification helpers.

Single site (INV-5) for two related but distinct classification concerns
described in plans/escalation-lifecycle-dashboard-prd.md Contract Seam 1:

- ``classify_resolver_tier`` — maps a ``resolved_by`` attribution string to a
  coarse resolver *tier* (human / cascade / auto-watcher / steward /
  reaper-sweep / unknown / other-auto). Used both for analytics grouping and
  to derive the per-path benign default (see ``default_resolution_class_for_resolver``).
- ``effective_benign`` — the stamp-first-with-proxy-fallback predicate that
  turns a resolved/dismissed ``Escalation`` record into a
  ``(class, provenance)`` pair, so aggregators can report the stamped-vs-
  inferred split.

Both helpers are pure functions with no I/O — callers (queue.py, server.py,
and future dashboard aggregators) import them rather than re-deriving the
same resolver membership or benign/actionable logic independently.
"""

from __future__ import annotations

# Resolver→tier classification table (plans/escalation-lifecycle-dashboard-prd.md
# Contract Seam 1). Exact-membership frozensets for the tiers whose
# resolved_by values are fixed literals; 'cascade' and 'steward' are matched
# by prefix/suffix instead since they're parameterised by escalation id / task id.
_HUMAN_RESOLVERS: frozenset[str] = frozenset({'interactive', 'escalation-watcher'})
_AUTO_WATCHER_RESOLVERS: frozenset[str] = frozenset(
    {'escalation-watcher-auto', 'orchestrator-escalation-watcher-auto'}
)
_REAPER_SWEEP_RESOLVERS: frozenset[str] = frozenset({
    'harness-orphan-reaper',
    'auto-dismissed',
    'harness-escalation-revalidation-sweep',
    'orchestrator-starvation-watchdog',
})

_CASCADE_PREFIX = 'l2-cascade:'
_STEWARD_PREFIX = 'claude-task-'
_STEWARD_SUFFIX = '-steward'


def classify_resolver_tier(resolved_by: str | None) -> str:
    """Classify a ``resolved_by`` attribution string into a resolver tier.

    Order matters: exact human/auto-watcher/reaper-sweep membership checks
    run before the prefix/suffix checks and the ``other-auto`` fallthrough,
    so a literal match always wins. Unknown non-None values fall to
    ``'other-auto'`` (never silently dropped — INV-4: growth in this bucket
    is surfaced as its own chart segment).
    """
    if resolved_by is None:
        return 'unknown'
    if resolved_by in _HUMAN_RESOLVERS:
        return 'human'
    if resolved_by.startswith(_CASCADE_PREFIX):
        return 'cascade'
    if resolved_by in _AUTO_WATCHER_RESOLVERS:
        return 'auto-watcher'
    if resolved_by.startswith(_STEWARD_PREFIX) and resolved_by.endswith(_STEWARD_SUFFIX):
        return 'steward'
    if resolved_by in _REAPER_SWEEP_RESOLVERS:
        return 'reaper-sweep'
    return 'other-auto'
