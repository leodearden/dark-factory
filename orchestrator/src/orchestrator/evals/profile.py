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
  - ``fused_memory.url`` (ε, D8): production defaults to the real dark_factory
    memory endpoint (``http://localhost:8002``). Every orchestrator-side eval
    memory write (``TaskWorkflow._write_completion_to_memory`` /
    ``_write_decisions_to_memory`` / ``_write_suggestions_to_memory``) POSTs
    raw httpx to ``{self.mcp.url}/mcp/``, where
    ``self.mcp = _EvalMcpStub(orch_config.fused_memory.url)`` — so an
    un-isolated green eval run would write observations/decisions/suggestions
    into PRODUCTION dark_factory memory. This one profiled leaf neutralizes
    every such write at once: ``http://127.0.0.1:1`` is a non-routable null
    sentinel that fast-fails with an immediate ECONNREFUSED (no slow
    DNS/timeout) and can never be production. This is the safe DEFAULT — the
    CLI/default eval path picks it up with zero extra wiring. A caller that
    wants to CAPTURE the intended writes (Boundary test B5, integration gate ι,
    or an operator) starts a ``RecordingMemorySink`` and passes its ``.url`` as
    ``build_eval_orch_config(..., memory_endpoint=...)`` /
    ``run_eval(..., memory_endpoint=...)`` (runner.py), which layers over this
    null-sentinel default so the real writes are recorded and the production
    write-count delta stays 0.
"""

from __future__ import annotations

from typing import Any

from orchestrator.config import OrchestratorConfig, _iter_leaves

# The ONLY documented divergences from a fresh load_config() — dotted leaf
# paths matching orchestrator.config._iter_leaves's output 1:1 (e.g.
# 'unblock_auto.enabled'), so the parity tripwire can assert
# `_changed_leaf_paths(...) == set(EVAL_PROFILE)` verbatim.
EVAL_PROFILE: dict[str, bool | str] = {
    'rebase_before_verify': False,     # D3 — no mid-eval rebase onto live main
    'inter_iteration_rebase': False,   # D3 — same
    'unblock_auto.enabled': False,     # D4 — no unmetered $5 dry-run per block
    'auto_eval_enabled': False,        # eval never re-triggers itself
    'simple_task_enabled': False,      # fixtures route through the full path deterministically
    'fused_memory.url': 'http://127.0.0.1:1',  # D8 — non-routable null sentinel; immediate ECONNREFUSED, never production
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


def apply_eval_profile(base: OrchestratorConfig) -> OrchestratorConfig:
    """Return *base* with EVAL_PROFILE's divergences applied, everything else inherited.

    This is the profile-application unit the parity tripwire (Invariant P1,
    Boundary test B1) measures: the changed-leaf set between
    ``apply_eval_profile(base)`` and *base* must equal ``set(EVAL_PROFILE)``
    exactly (see test_eval_profile.py). Every field not named in
    EVAL_PROFILE is inherited from *base* verbatim — the D5 fix.
    """
    return base.model_copy(update=resolve_eval_profile_update(base))


def eval_profile_divergence(
    base: OrchestratorConfig,
) -> dict[str, tuple[Any, Any]]:
    """Report every leaf where the eval config diverges from *base*.

    Returns ``{dotted_leaf_path: (base_value, eval_value)}`` for every leaf
    that ``apply_eval_profile(base)`` changes relative to *base*, keyed by the
    same one-level dotted paths ``config._iter_leaves`` yields (the enumerator
    ``diff_config`` and the β parity test also use) — so on a code-default
    base the report's key set matches ``EVAL_PROFILE`` 1:1 (Invariant P1,
    Boundary test B1).

    This is the first-class, eval-side introspection the Phase-1 integration
    gate (ι) — and operators / the μ scoring driver — assert on: a
    machine-checkable answer to "exactly how does an eval run's config differ
    from production, and is that set precisely the documented EVAL_PROFILE?".
    A leaf present here but absent from ``EVAL_PROFILE`` is an undocumented
    divergence (the D5 leak class); a documented leaf absent here is a no-op
    profile entry. The gate (B1) trips on exactly those two conditions.
    """
    evaled = apply_eval_profile(base)
    base_leaves = dict(_iter_leaves(base))
    eval_leaves = dict(_iter_leaves(evaled))
    return {
        path: (base_value, eval_leaves[path])
        for path, base_value in base_leaves.items()
        if base_value != eval_leaves[path]
    }
