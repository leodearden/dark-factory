"""The identity the startup stale-L0 strand sweep files its level-1 record under.

The record's task_id is STALE_L0_STRAND_SENTINEL, its root_cause
STALE_L0_STRAND_ROOT_CAUSE and its agent_role STALE_L0_STRAND_ROLE. The
auto-watcher's Q6 detector-rule table in skills/escalation-watcher-auto/SKILL.md
keys a rule on STALE_L0_STRAND_SENTINEL.
"""

__all__ = ['STALE_L0_STRAND_ROLE', 'STALE_L0_STRAND_ROOT_CAUSE', 'STALE_L0_STRAND_SENTINEL']

STALE_L0_STRAND_SENTINEL = '__stale_l0_strand_sweep__'
STALE_L0_STRAND_ROOT_CAUSE = 'stale_l0_strand_sweep'
STALE_L0_STRAND_ROLE = 'harness-stale-l0-strand-sweep'
