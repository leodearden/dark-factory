"""Shared queued-branch-name normalizer — canonical_queued_branch_name.

Provides :func:`canonical_queued_branch_name`, the single source of truth
for "prepend the queue's branch prefix onto a branch name unless it is
already present". Relocated out of orchestrator/git_ops.py so escalation's
server.py — which cannot import orchestrator — can consume it statically
(escalation already imports ``shared`` directly, e.g. ``shared.timestamps``).
"""

from __future__ import annotations

__all__ = ['canonical_queued_branch_name']


def canonical_queued_branch_name(branch: str, branch_prefix: str) -> str:
    """Return *branch* with *branch_prefix* prepended, unless already present.

    Shared pure-string shape-normalizer: "prepend ``branch_prefix`` unless
    ``branch`` already starts with it".  This is the deterministic-NAME
    sibling of :meth:`GitOps.resolve_queued_branch_ref` — that method does
    git I/O and returns ``None`` when no ref resolves, which is unusable at
    call sites that need a full-ref name even when the ref is absent (e.g. a
    drop-log message, or a marker search over a deleted branch).  Use this
    helper there instead.

    Single source of truth consumed by ``recover_pending_merges``
    (merge_queue_store.py) and merge_status's git-authority tier
    (escalation/server.py) so the "is this branch name already prefixed?"
    rule is not duplicated with divergent fidelity across sites.
    """
    return branch if branch.startswith(branch_prefix) else f'{branch_prefix}{branch}'
