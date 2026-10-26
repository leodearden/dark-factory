"""Stage 3's deterministic detector for dangling cross-run mem0 citations.

A settled run's findings survive only in the journal's ``stage_reports`` blob,
and a memory one of them cites may since have been deleted by a later
consolidation or supersession. This scan reads the project's recently settled
runs back out of the journal, confirms each concrete mem0 citation with the same
three-way point read the repair tool corroborates with, and returns one
``DanglingCitation`` per confirmed absence for Stage 1 to repair through
``reconciliation/citation_repair.py::repair_memory_citation``. The stage3 prompt
section ``Dangling Cross-Run Citations — Detected For You`` tells Stage 3 the
scan runs. A ``wrong_memory`` citation still resolves, so it is out of reach.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any, NamedTuple

from fused_memory.models.reconciliation import StageReport
from fused_memory.reconciliation.citation_repair import SUPPORTED_STORE
from fused_memory.reconciliation.citation_verifier import (
    is_concrete_memory_id,
    make_memory_resolver,
)
from fused_memory.reconciliation.dangling_citation import DanglingCitation
from fused_memory.reconciliation.journal import (
    REPAIRABLE_RUN_STATUS_VALUES,
    flagged_findings_by_id,
)
from fused_memory.utils.validation import is_full_uuid

logger = logging.getLogger(__name__)

CROSS_RUN_CITATION_SCAN_LOOKBACK_RUNS = 50
MAX_CROSS_RUN_CITATION_IDS_PER_SCAN = 100
MAX_CONSECUTIVE_CITATION_READ_ERRORS = 3
MAX_DANGLING_CITATION_FINDINGS_PER_SCAN = 20

_FOUND = 'found'
_MISSING = 'missing'
_ERROR = 'error'


@dataclass(frozen=True)
class CrossRunCitationScan:
    dangling: tuple[DanglingCitation, ...]
    runs_scanned: int
    citations_verified: int
    dangling_total: int
    verification_errors: int
    unchecked: int
    scan_failed: bool

    @property
    def deferred(self) -> int:
        return self.dangling_total - len(self.dangling)

    def as_stats(self, prefix: str) -> dict[str, int]:
        return {
            f'{prefix}_cross_run_runs_scanned': self.runs_scanned,
            f'{prefix}_cross_run_citations_verified': self.citations_verified,
            f'{prefix}_cross_run_dangling_citations': self.dangling_total,
            f'{prefix}_cross_run_dangling_citations_deferred': self.deferred,
            f'{prefix}_cross_run_citation_verification_errors': self.verification_errors,
            f'{prefix}_cross_run_citations_unchecked': self.unchecked,
            f'{prefix}_cross_run_citation_scan_failed': int(self.scan_failed),
        }


_FAILED_SCAN = CrossRunCitationScan(
    dangling=(),
    runs_scanned=0,
    citations_verified=0,
    dangling_total=0,
    verification_errors=0,
    unchecked=0,
    scan_failed=True,
)


class _Citation(NamedTuple):
    run_id: str
    finding_id: str
    memory_id: str


async def scan_cross_run_citations(
    journal: Any,
    memory_service: Any,
    *,
    project_id: str,
    lookback: int = CROSS_RUN_CITATION_SCAN_LOOKBACK_RUNS,
    max_ids: int = MAX_CROSS_RUN_CITATION_IDS_PER_SCAN,
    max_consecutive_errors: int = MAX_CONSECUTIVE_CITATION_READ_ERRORS,
    max_findings: int = MAX_DANGLING_CITATION_FINDINGS_PER_SCAN,
) -> CrossRunCitationScan:
    runs = await _settled_runs(journal, project_id, lookback)
    if runs is None:
        return _FAILED_SCAN
    citations = [citation for run in runs for citation in _mem0_citations(run)]
    outcomes = await _outcome_per_citation(
        citations,
        make_memory_resolver(memory_service, project_id),
        max_ids=max_ids,
        max_consecutive_errors=max_consecutive_errors,
    )
    dangling = tuple(
        record
        for c, outcome in zip(citations, outcomes, strict=True)
        if outcome == _MISSING and (record := _addressable_record(c)) is not None
    )
    counts = Counter(outcomes)
    return CrossRunCitationScan(
        dangling=dangling[:max_findings],
        runs_scanned=len(runs),
        citations_verified=counts[_FOUND],
        dangling_total=len(dangling),
        verification_errors=counts[_ERROR],
        unchecked=counts[None],
        scan_failed=False,
    )


async def _settled_runs(journal: Any, project_id: str, lookback: int) -> list[Any] | None:
    try:
        runs = await journal.get_recent_runs(project_id, limit=lookback)
    except Exception as exc:
        logger.warning(
            'cross-run citation scan: journal read failed for project %s: %r',
            project_id, exc,
        )
        return None
    if not isinstance(runs, list):
        logger.warning(
            'cross-run citation scan: get_recent_runs returned %s, not a list',
            type(runs).__name__,
        )
        return None
    return [
        run for run in runs
        if str(run.status) in REPAIRABLE_RUN_STATUS_VALUES and is_full_uuid(run.id)
    ]


def _mem0_citations(run: Any) -> Iterator[_Citation]:
    seen: set[tuple[str, str]] = set()
    for report in run.stage_reports.values():
        if not isinstance(report, StageReport):
            continue
        for finding_id, finding in flagged_findings_by_id(report.items_flagged).items():
            for memory_id in _concrete_mem0_ids(finding):
                key = (finding_id, memory_id.lower())
                if key not in seen:
                    seen.add(key)
                    yield _Citation(run.id, finding_id, memory_id)


def _concrete_mem0_ids(finding: dict[str, Any]) -> Iterator[str]:
    cited = finding.get('cited_memories')
    if not isinstance(cited, list):
        return
    for entry in cited:
        if not isinstance(entry, dict) or entry.get('store') != SUPPORTED_STORE:
            continue
        memory_id = entry.get('memory_id')
        if isinstance(memory_id, str) and is_concrete_memory_id(memory_id):
            yield memory_id


def _addressable_record(citation: _Citation) -> DanglingCitation | None:
    try:
        return DanglingCitation(
            citation.run_id, citation.finding_id, citation.memory_id, SUPPORTED_STORE,
        )
    except ValueError as exc:
        logger.warning('cross-run citation scan: skipping unaddressable %r: %s', citation, exc)
        return None


async def _outcome_per_citation(
    citations: list[_Citation],
    resolve: Callable[[Any], Awaitable[tuple[str, str | None]]],
    *,
    max_ids: int,
    max_consecutive_errors: int,
) -> list[str | None]:
    spelling_by_key: dict[str, str] = {}
    for citation in citations:
        spelling_by_key.setdefault(citation.memory_id.lower(), citation.memory_id)
    resolved = await _resolve_distinct_ids(
        spelling_by_key.values(),
        resolve,
        max_ids=max_ids,
        max_consecutive_errors=max_consecutive_errors,
    )
    outcome_by_key = dict(zip(spelling_by_key, resolved, strict=True))
    return [outcome_by_key[c.memory_id.lower()] for c in citations]


async def _resolve_distinct_ids(
    memory_ids: Iterable[str],
    resolve: Callable[[Any], Awaitable[tuple[str, str | None]]],
    *,
    max_ids: int,
    max_consecutive_errors: int,
) -> list[str | None]:
    """Resolve each id in order, answering ``None`` for every id left unchecked.

    Resolution stops after *max_ids* ids, or once *max_consecutive_errors*
    reads in a row raised: each read on a degraded store waits out its own
    backend timeout on Stage 3's critical path.
    """
    outcomes: list[str | None] = []
    consecutive_errors = 0
    for memory_id in memory_ids:
        if len(outcomes) >= max_ids or consecutive_errors >= max_consecutive_errors:
            outcomes.append(None)
            continue
        outcome, _detail = await resolve(memory_id)
        consecutive_errors = consecutive_errors + 1 if outcome == _ERROR else 0
        if consecutive_errors == max_consecutive_errors:
            logger.warning(
                'cross-run citation scan: %d consecutive mem0 reads failed; '
                'leaving the remaining ids unchecked',
                consecutive_errors,
            )
        outcomes.append(outcome)
    return outcomes
