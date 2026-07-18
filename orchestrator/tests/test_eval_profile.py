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
    """EVAL_PROFILE is exactly the 5 documented C1 leaves — no more, no less."""
    assert EVAL_PROFILE == {
        'rebase_before_verify': False,
        'inter_iteration_rebase': False,
        'unblock_auto.enabled': False,
        'auto_eval_enabled': False,
        'simple_task_enabled': False,
    }


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
    profile leaf flips True -> False, so the diff is precisely the 5
    documented EVAL_PROFILE keys — never more (an undocumented leak) or fewer
    (a no-op profile entry).
    """
    base = OrchestratorConfig(project_root=tmp_path)

    # Loud premise guard: every EVAL_PROFILE path must read True on a fresh
    # code-default base, or flipping it to False below wouldn't actually
    # change anything and the parity assertion would pass vacuously.
    for path in EVAL_PROFILE:
        assert config._get_leaf(base, path) is True, (
            f'premise violated: {path!r} is not True on a fresh code-default '
            f'OrchestratorConfig — the parity assertion below would be vacuous'
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
