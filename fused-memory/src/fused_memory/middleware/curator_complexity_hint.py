"""Apply a curated create's complexity hint to the task's metadata under routing experiment 2.

Every create the curator judged as one records the curator's hint: never a
fail-open create, a fall-through, or a drop or combine pool state vetoed into a
create. A declarable task (hint "simple", metadata.files the submit gate
accepts, no complexity set by the filer) also records the candidate_key its arm
is drawn from, and on the treated arm declares the SIMPLE_TASK fast path. Contract:
plans/routing-experiment-substrate-prd.md §"Experiment 2: the curator hint (ε)".
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final, TypeGuard

from shared.routing_split import split_bucket
from shared.task_complexity import SIMPLE_COMPLEXITY, is_simple_complexity
from shared.task_metadata_wire import coerce_task_metadata

from fused_memory.middleware.candidate_key import compute_candidate_key
from fused_memory.middleware.curator_types import CuratorDecision
from fused_memory.middleware.lock_charter_guard import extract_files
from fused_memory.middleware.simple_task_files_guard import simple_task_files_error

__all__ = [
    'ARM_KEY_METADATA_KEY',
    'COMPLEXITY_HINT_METADATA_KEY',
    'EXPERIMENT_BUCKETS',
    'EXPERIMENT_SALT',
    'TREATED_BUCKET',
    'is_treated',
    'with_complexity_hint',
]

COMPLEXITY_HINT_METADATA_KEY: Final = 'x_curator_complexity_hint'
ARM_KEY_METADATA_KEY: Final = 'x_routing_exp2_arm_key'
EXPERIMENT_SALT: Final = 'routing-exp-2'
EXPERIMENT_BUCKETS: Final = 2
TREATED_BUCKET: Final = 1


def is_treated(candidate_key: str) -> bool:
    """True iff *candidate_key* falls in routing experiment 2's treated arm."""
    return split_bucket(EXPERIMENT_SALT, candidate_key, EXPERIMENT_BUCKETS) == TREATED_BUCKET


def with_complexity_hint(
    metadata: Any, *, title: str | None, decision: CuratorDecision | None,
) -> Any:
    """*metadata* stamped with *decision*'s complexity hint; anything else is returned as is.

    Only a create the curator judged as one is stamped, and an unreadable wire
    value is never rewritten. The input is never mutated.
    """
    if not _judged_as_create(decision):
        return metadata
    readable = coerce_task_metadata(metadata)
    if readable is None:
        return metadata
    stamped = {**readable, COMPLEXITY_HINT_METADATA_KEY: decision.complexity_hint}
    if not _declarable(readable, decision.complexity_hint):
        return stamped
    key = compute_candidate_key(title, extract_files(stamped))
    if key is None:
        return stamped
    arm_keyed = {**stamped, ARM_KEY_METADATA_KEY: key}
    return {**arm_keyed, 'complexity': SIMPLE_COMPLEXITY} if is_treated(key) else arm_keyed


def _judged_as_create(decision: CuratorDecision | None) -> TypeGuard[CuratorDecision]:
    """True iff a usable create verdict came from the curator choosing create, not a veto or a registry."""
    return (
        decision is not None
        and decision.is_usable_create_verdict()
        and decision.action == 'create'
        and decision.vetoed_action is None
    )


def _declarable(readable: Mapping[str, Any], hint: str | None) -> bool:
    """True iff a simple declaration could be added without overriding the filer or the gate."""
    return (
        is_simple_complexity(hint)
        and not str(readable.get('complexity') or '').strip()
        and simple_task_files_error({**readable, 'complexity': SIMPLE_COMPLEXITY}) is None
    )
