"""Eval config profile — the documented divergence surface from production.

PRD eval-framework-revival §β, Contract C1, Invariant P1, Boundary test B1.

``EVAL_PROFILE`` is the single source of truth for how an eval run's
``OrchestratorConfig`` is allowed to differ from a live production config
(``load_config()``). Every field NOT listed here is inherited verbatim from
the base config passed in — so a new production default can never silently
change eval behaviour; it either already matches production or is a
deliberate, documented profile entry here.

Divergences, and why (see plans/eval-framework-revival-prd.md for the drift
inventory this closes):
  - ``rebase_before_verify`` / ``inter_iteration_rebase`` (D3): production
    defaults both True (rebase onto main before/between verify attempts).
    Eval worktrees are fixed-target fixtures pinned to a
    ``task['pre_task_commit']`` snapshot — rebasing onto *live* main mid-eval
    would silently change the fixture out from under the run.
  - ``unblock_auto.enabled`` (D4): production defaults True (an autonomous
    dry-run investigation on block/timeout). Left on, every blocked/timeout
    eval spawns an unmetered ~$5 Sonnet dry-run — pure eval-budget waste with
    no eval-facing benefit.
  - ``auto_eval_enabled``: production defaults True (auto-redo via the full
    architect path on an optimistic-path block). An eval run must never
    re-trigger itself.
  - ``simple_task_enabled``: production defaults True (routes trivial tasks
    through a single-agent fast path). Eval fixtures must route through the
    full architect+implementer path deterministically, regardless of how a
    given fixture's description happens to read.

NOTE for a future task (ε, D8): the fused-memory endpoint override (pointing
eval writes at a null/recording sink instead of the real dark_factory memory
store) belongs here too — add a ``fused_memory.url``-style entry to
EVAL_PROFILE when that task lands. Until then, the parity tripwire
(test_eval_profile.py) asserts against ``set(EVAL_PROFILE)``, so adding a key
here auto-extends the tripwire with no separate test edit.
"""

from __future__ import annotations

from typing import Any

from orchestrator.config import OrchestratorConfig

# The ONLY documented divergences from a fresh load_config() — dotted leaf
# paths matching orchestrator.config._iter_leaves's output 1:1 (e.g.
# 'unblock_auto.enabled'), so the parity tripwire can assert
# `_changed_leaf_paths(...) == set(EVAL_PROFILE)` verbatim.
EVAL_PROFILE: dict[str, bool] = {
    'rebase_before_verify': False,     # D3 — no mid-eval rebase onto live main
    'inter_iteration_rebase': False,   # D3 — same
    'unblock_auto.enabled': False,     # D4 — no unmetered $5 dry-run per block
    'auto_eval_enabled': False,        # eval never re-triggers itself
    'simple_task_enabled': False,      # fixtures route through the full path deterministically
}


def resolve_eval_profile_update(base: OrchestratorConfig) -> dict[str, Any]:
    """Resolve EVAL_PROFILE into a ``base.model_copy(update=...)``-ready dict.

    Flat (undotted) keys pass straight through. A dotted key is split on its
    first ``.``, grouped by its head, and applied as
    ``update[head] = getattr(base, head).model_copy(update={leaf: value})`` —
    a real nested-submodel copy that preserves every sibling leaf from
    *base*. This is deliberate: pydantic v2's ``model_copy(update={'a.b': v})``
    does NOT descend into nested models — it would inject a stray,
    unvalidated top-level ``__dict__`` key literally named ``'a.b'`` and
    leave the real nested field untouched (silently leaving
    ``unblock_auto.enabled`` True — D4 unfixed).
    """
    update: dict[str, Any] = {}
    submodel_updates: dict[str, dict[str, Any]] = {}

    for path, value in EVAL_PROFILE.items():
        if '.' in path:
            head, leaf = path.split('.', 1)
            submodel_updates.setdefault(head, {})[leaf] = value
        else:
            update[path] = value

    for head, leaf_updates in submodel_updates.items():
        submodel = getattr(base, head)
        update[head] = submodel.model_copy(update=leaf_updates)

    return update
