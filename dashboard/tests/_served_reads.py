"""The suite's one reset for the module-global stores dashboard routes serve from.

Those stores outlive a request, so a body rendered at one installed clock can
be served again under another. A test that pins a route's clock forgets them
before and after through :func:`forget_served_reads`; a new store a route
reads joins it here.
"""

from __future__ import annotations

from dashboard.api.escalations import _analytics_memo_clear
from dashboard.app import _memory_evals_cache_clear
from dashboard.data import escalation_corpus
from dashboard.data.active_tasks import reset_task_snapshot_state
from dashboard.data.prd_signal import clear_prd_documents
from dashboard.data.scheduler import clear_scheduler_state
from dashboard.data.task_activity import clear_activity_state
from dashboard.data.task_lineage import clear_lineage_state


def forget_escalation_corpus() -> None:
    """Forget the corpus walk and the analytics derived from it, through their own hooks."""
    escalation_corpus._corpus_cache_clear()
    _analytics_memo_clear()


def forget_served_reads() -> None:
    """Every access store a route reads, forgotten through its own hook."""
    reset_task_snapshot_state()
    clear_lineage_state()
    clear_scheduler_state()
    clear_activity_state()
    clear_prd_documents()
    forget_escalation_corpus()
    _memory_evals_cache_clear()
