"""The task id a test must use to land in a routing rule's split arm.

Read off the loaded config, so the experiment salt and bucket have one home
(``orchestrator/src/orchestrator/defaults.yaml``) and a test agrees with the
resolver by construction (both go through
``shared/src/shared/routing_split.py::split_bucket``).
"""

from __future__ import annotations

from shared.routing_split import split_bucket

from orchestrator.config import OrchestratorConfig

_CANDIDATE_TASK_IDS = range(1, 1001)


def task_id_in_rule_arm(cfg: OrchestratorConfig, rule_id: str) -> str:
    rule = next((r for r in cfg.routing.rules if r.id == rule_id), None)
    if rule is None:
        present = [r.id for r in cfg.routing.rules]
        raise LookupError(f'no routing rule {rule_id!r}; rules present: {present}')
    split = rule.match.split
    if split is None:
        return '1'
    for i in _CANDIDATE_TASK_IDS:
        if split_bucket(split.salt, str(i), split.buckets) == split.bucket:
            return str(i)
    raise LookupError(
        f'no task id in {_CANDIDATE_TASK_IDS} falls in arm {split.bucket} of rule {rule_id!r}'
    )
