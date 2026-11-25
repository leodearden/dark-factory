"""Where a recon or link-adjudicator LLM call lands in the CostStore ``invocations`` ledger.

A LedgerRole names the CALL SITE that spent the money. It is not a route role
(task 9699's config rows that choose model and effort), and not the reserve
WorkClass role. The curator family (task_curator.py, path_scope_adjudicator.py,
pack_judge.py) spells its own role strings in its own modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fused_memory.models.reconciliation import StageId

if TYPE_CHECKING:
    from shared.cli_invoke import AgentResult
    from shared.cost_store import CostStore

INVOCATIONS_LEDGER_FILENAME = 'curator_events.db'


def invocations_ledger_path(data_dir: Path) -> Path:
    """The fused-memory process's one ``invocations`` ledger file under *data_dir*.

    The curator-era name is kept: the dashboard and the audit queries read it.
    """
    return data_dir / INVOCATIONS_LEDGER_FILENAME


class LedgerRole(StrEnum):
    """The ``invocations.role`` value audit queries group recon spend by."""

    RECON_STAGE_MEMORY_CONSOLIDATOR = 'recon_stage_memory_consolidator'
    RECON_STAGE_TASK_KNOWLEDGE_SYNC = 'recon_stage_task_knowledge_sync'
    RECON_STAGE_INTEGRITY_CHECK = 'recon_stage_integrity_check'
    RECON_JUDGE = 'recon_judge'
    RECON_VERIFY = 'recon_verify'
    LINK_ADJUDICATOR = 'link_adjudicator'

    @classmethod
    def for_stage(cls, stage: StageId) -> LedgerRole:
        return _STAGE_ROLES[stage]


_STAGE_ROLES: dict[StageId, LedgerRole] = {
    StageId.memory_consolidator: LedgerRole.RECON_STAGE_MEMORY_CONSOLIDATOR,
    StageId.task_knowledge_sync: LedgerRole.RECON_STAGE_TASK_KNOWLEDGE_SYNC,
    StageId.integrity_check: LedgerRole.RECON_STAGE_INTEGRITY_CHECK,
}


@dataclass(frozen=True)
class InvocationLedger:
    """The store one LLM call's row lands in, and the keys it is written under.

    A call that writes no row is given no ledger (None), never one without a store.
    """

    cost_store: CostStore
    role: LedgerRole
    run_id: str
    project_id: str

    @classmethod
    def or_none(
        cls, cost_store: CostStore | None, role: LedgerRole, run_id: str, project_id: str,
    ) -> InvocationLedger | None:
        """The ledger over *cost_store*; None, so no row is written, when there is no store."""
        if cost_store is None:
            return None
        return cls(cost_store, role, run_id, project_id)


def ledger_invoke_kwargs(ledger: InvocationLedger | None) -> dict[str, Any]:
    """The ``invoke_with_cap_retry`` kwargs that write *ledger*'s row; none for no ledger."""
    if ledger is None:
        return {}
    return {
        'cost_store': ledger.cost_store,
        'role': ledger.role.value,
        'run_id': ledger.run_id,
        'project_id': ledger.project_id,
    }


def input_output_tokens(result: AgentResult) -> int:
    """A call's ``tokens_used``: input plus output tokens.

    Cache reads and cache creations are counted apart, never in this sum.
    """
    return (result.input_tokens or 0) + (result.output_tokens or 0)
