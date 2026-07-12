"""Adjudication log for the reviewer_trial corpus (task 2495 / PRD D-6).

Records, per mined diff, the frontier-model-proposed ground-truth labels
(``mining.propose_labels_frontier``) plus whether the diff falls in a
documented, stratified human spot-check subset. The log is the artifact
that satisfies "an adjudication log shows frontier-proposed labels + human
spot-check on a documented subset" — spot-check entries start out
``spot_check_status='pending'`` for operator sign-off (see
``corpus/README.md`` for the protocol) and are never fabricated by an
agent.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any


@dataclass
class AdjudicationEntry:
    """One diff's adjudication record: frontier proposal + spot-check flag."""

    diff_id: str
    frontier_model: str
    frontier_proposal: list[dict]  # serialized GroundTruthIssue dicts
    in_spot_check_subset: bool
    spot_check_status: str = 'pending'  # 'pending' | 'confirmed' | 'rejected'
    spot_check_reviewer: str | None = None
    notes: str = ''


@dataclass
class CoverageReport:
    """Which diff_ids have an adjudication entry, and which are missing one."""

    covered: set[str]
    missing: set[str]

    @property
    def ok(self) -> bool:
        """True when every diff_id passed to coverage() has an entry."""
        return not self.missing


def _proposal_to_dicts(frontier_proposal: Iterable[Any]) -> list[dict]:
    """Normalize a frontier proposal (GroundTruthIssue instances or plain
    dicts) into plain dicts, so AdjudicationEntry is always JSON-serializable."""
    result = []
    for issue in frontier_proposal:
        result.append(asdict(issue) if is_dataclass(issue) else dict(issue))
    return result


@dataclass
class AdjudicationLog:
    """Append-only log of adjudication entries, one per mined diff."""

    entries: list[AdjudicationEntry] = field(default_factory=list)

    def append(
        self,
        diff_id: str,
        frontier_proposal: Iterable[Any],
        in_spot_check_subset: bool,
        frontier_model: str = '',
        notes: str = '',
    ) -> AdjudicationEntry:
        """Record an adjudication entry for *diff_id* and return it."""
        entry = AdjudicationEntry(
            diff_id=diff_id,
            frontier_model=frontier_model,
            frontier_proposal=_proposal_to_dicts(frontier_proposal),
            in_spot_check_subset=in_spot_check_subset,
            notes=notes,
        )
        self.entries.append(entry)
        return entry

    def coverage(self, all_diff_ids: Iterable[str]) -> CoverageReport:
        """Report which of *all_diff_ids* have an adjudication entry."""
        wanted = set(all_diff_ids)
        covered = {e.diff_id for e in self.entries} & wanted
        return CoverageReport(covered=covered, missing=wanted - covered)

    def spot_check_subset(self) -> list[AdjudicationEntry]:
        """Return the documented human spot-check subset (flagged entries)."""
        return [e for e in self.entries if e.in_spot_check_subset]

    def save(self, path: Path) -> None:
        """Write the log as JSONL — one AdjudicationEntry object per line."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w') as f:
            for entry in self.entries:
                f.write(json.dumps(asdict(entry)))
                f.write('\n')

    @classmethod
    def load(cls, path: Path) -> AdjudicationLog:
        """Load a JSONL adjudication log. Missing file -> empty log."""
        if not path.exists():
            return cls()
        entries = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            entries.append(AdjudicationEntry(**json.loads(line)))
        return cls(entries=entries)
