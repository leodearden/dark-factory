"""Unit + parity-tripwire tests for the eval config profile.

PRD eval-framework-revival §β, Contract C1, Invariant P1, Boundary test B1.

``build_eval_orch_config`` (evals/runner.py) used to build ``OrchestratorConfig``
BY CONSTRUCTOR — every field not explicitly passed silently took its pydantic
default instead of the live production value (D5, the root-cause drift). That
silent-default path is exactly how D3 (``rebase_before_verify`` /
``inter_iteration_rebase`` defaulting True, rebasing eval worktrees onto live
main mid-eval) and D4 (``unblock_auto.enabled`` defaulting True, an unmetered
~$5 Sonnet dry-run per blocked/timeout eval) leaked in.

The fix derives the eval config from the live base via
``base.model_copy(update=...)`` so any field NOT in the documented
``EVAL_PROFILE`` is inherited from production — silent divergence becomes
structurally impossible. The parity tripwire (B1, P1) asserts the changed-leaf
set between the profile-resolved config and its base equals ``set(EVAL_PROFILE)``
exactly, so the *next* production field to land can no longer silently change
eval behavior.
"""

from __future__ import annotations

import pytest

from orchestrator import config
from orchestrator.config import OrchestratorConfig
from orchestrator.evals.configs import EvalConfig
from orchestrator.evals.profile import (
    EVAL_PROFILE,
    apply_eval_profile,
    resolve_eval_profile_update,
)
from orchestrator.evals.runner import build_eval_orch_config
from orchestrator.fm_retry import FM_NULL_SENTINEL_URL, is_fm_null_sentinel


def _changed_leaf_paths(a: OrchestratorConfig, b: OrchestratorConfig) -> set[str]:
    """Return the set of dotted leaf paths where *a* and *b* differ.

    Reuses ``config._iter_leaves`` (the same one-level-nested dotted-path
    enumerator ``diff_config`` uses), so its output (e.g.
    ``'unblock_auto.enabled'``) matches ``EVAL_PROFILE`` keys 1:1.
    """
    a_leaves = dict(config._iter_leaves(a))
    b_leaves = dict(config._iter_leaves(b))
    return {path for path, a_val in a_leaves.items() if a_val != b_leaves[path]}


def test_eval_profile_documented_keys():
    """EVAL_PROFILE is exactly the 6 documented C1 leaves — no more, no less."""
    assert EVAL_PROFILE == {
        'rebase_before_verify': False,
        'inter_iteration_rebase': False,
        'unblock_auto.enabled': False,
        'auto_eval_enabled': False,
        'simple_task_enabled': False,
        'fused_memory.url': 'http://127.0.0.1:1',
    }


def test_eval_profile_sentinel_single_sourced_from_fm_retry():
    """The D8 fused_memory.url sentinel is the shared fm_retry constant (task 2880).

    The eval profile neutralizes FM writes with a non-routable sentinel; the
    McpSession transport (mcp_lifecycle._retry_backoffs) must recognize exactly
    that same string to fail fast on it instead of spinning the ~120s
    fm-restart window. Single-sourcing the string in fm_retry — with
    evals/profile.py importing it DOWN — makes profile<->transport drift
    structurally impossible.
    """
    # (a) Value agreement — a durable drift guard even if identity were lost.
    assert EVAL_PROFILE['fused_memory.url'] == FM_NULL_SENTINEL_URL
    # (b) The transport predicate recognizes exactly the profile's sentinel —
    # the live cross-module contract that makes the fail-fast actually fire.
    assert is_fm_null_sentinel(EVAL_PROFILE['fused_memory.url']) is True
    # (c) Identity — proves profile.py references the shared constant object
    # rather than holding a second copy of the literal (CPython does not intern
    # URL-shaped literals, so a distinct copy would fail `is` while passing ==).
    assert EVAL_PROFILE['fused_memory.url'] is FM_NULL_SENTINEL_URL


def test_resolve_eval_profile_update_maps_dotted_to_submodel_copy(tmp_path):
    """Dotted leaves group under their submodel head via model_copy, not a stray key.

    pydantic v2 ``model_copy(update={'unblock_auto.enabled': False})`` would
    inject a stray, unvalidated top-level ``__dict__`` key literally named
    ``'unblock_auto.enabled'`` and NOT update the nested field — silently
    leaving ``unblock_auto.enabled`` True (D4 unfixed). Assert the resolver
    instead groups the dotted key by its head and produces a real
    nested-submodel copy.
    """
    base = OrchestratorConfig(project_root=tmp_path)

    update = resolve_eval_profile_update(base)

    # The 4 flat (undotted) keys pass straight through into the top-level
    # model_copy update dict.
    assert update['rebase_before_verify'] is False
    assert update['inter_iteration_rebase'] is False
    assert update['auto_eval_enabled'] is False
    assert update['simple_task_enabled'] is False

    # No stray dotted key ever reaches the update dict.
    assert 'unblock_auto.enabled' not in update

    # 'unblock_auto' is a real nested-submodel copy: the targeted leaf flips...
    resolved_unblock_auto = update['unblock_auto']
    assert resolved_unblock_auto.enabled is False
    # ...but sibling leaves are preserved from base, proving a targeted
    # model_copy of base.unblock_auto rather than a fresh/default
    # UnblockAutoConfig() replacing it wholesale.
    assert resolved_unblock_auto.budget_usd == base.unblock_auto.budget_usd
    assert resolved_unblock_auto.model == base.unblock_auto.model


@pytest.mark.usefixtures('code_default_config')
def test_apply_eval_profile_parity(tmp_path):
    """The changed-leaf set between apply_eval_profile(base) and base equals EVAL_PROFILE exactly.

    Invariant P1 / Boundary test B1 — the D5 root-cause guard. Exactness holds
    by identity: model_copy(update=X) changes exactly X's leaves and every
    profile leaf diverges from its base value (the 5 bool True -> False flips
    plus the D8 fused_memory.url production -> null-sentinel change), so the
    diff is precisely the 6 documented EVAL_PROFILE keys — never more (an
    undocumented leak) or fewer (a no-op profile entry).
    """
    base = OrchestratorConfig(project_root=tmp_path)

    # Loud premise guard: every EVAL_PROFILE path must DIFFER from its profile
    # value on a fresh code-default base, or applying it below wouldn't
    # actually change anything and the parity assertion would pass vacuously.
    # Generalized from the old "is True" form (which only held for the bool
    # flips) so the string-valued fused_memory.url entry is guarded too.
    for path in EVAL_PROFILE:
        assert config._get_leaf(base, path) != EVAL_PROFILE[path], (
            f'premise violated: {path!r} already equals its EVAL_PROFILE value '
            f'on a fresh code-default OrchestratorConfig — the parity assertion '
            f'below would be vacuous for this leaf'
        )

    resolved = apply_eval_profile(base)

    assert _changed_leaf_paths(resolved, base) == set(EVAL_PROFILE)


def test_parity_tripwire_trips_on_undocumented_divergence(tmp_path):
    """A synthetic non-profile divergence trips the tripwire — the RED signal B1 guards.

    Proves the tripwire is actually sensitive to a leaked production field:
    hand-inject a change to a field EVAL_PROFILE never touches
    (``max_amendment_rounds``) and confirm the changed-leaf set no longer
    equals ``set(EVAL_PROFILE)`` — with the offending field named in the diff.
    """
    base = OrchestratorConfig(project_root=tmp_path)
    leaked = apply_eval_profile(base).model_copy(
        update={'max_amendment_rounds': base.max_amendment_rounds + 1},
    )

    changed = _changed_leaf_paths(leaked, base)

    assert changed != set(EVAL_PROFILE)
    assert 'max_amendment_rounds' in changed


def test_build_eval_orch_config_applies_profile_and_inherits_base(tmp_path):
    """build_eval_orch_config routes through the profile and derives from base end-to-end.

    D3/D4: the 5 profile fields land False on the built config. D5: a
    base-only field the constructor build never passed through
    (``max_amendment_rounds``) is nonetheless inherited from *base* — proof
    this derives from base rather than falling back to a pydantic default.
    Per-run overrides (candidate model, sandbox-off) still apply on top.
    """
    base = OrchestratorConfig(project_root=tmp_path, max_amendment_rounds=7)
    cfg = EvalConfig(name='t', backend='claude', model='sonnet', effort='high')
    task = {'id': 't', 'project_root': str(tmp_path)}

    result = build_eval_orch_config(cfg, task, base)

    # D3/D4 — profile applied end-to-end.
    assert result.rebase_before_verify is False
    assert result.inter_iteration_rebase is False
    assert result.auto_eval_enabled is False
    assert result.simple_task_enabled is False
    assert result.unblock_auto.enabled is False

    # D5 — inherited from base, not the pydantic default (1) the old
    # constructor build silently fell back to.
    assert result.max_amendment_rounds == 7

    # Per-run overrides still applied on top of the profile-resolved base.
    assert result.models.implementer == 'sonnet'
    assert result.sandbox.enabled is False
