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

from orchestrator.config import OrchestratorConfig
from orchestrator.evals.profile import EVAL_PROFILE, resolve_eval_profile_update


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
