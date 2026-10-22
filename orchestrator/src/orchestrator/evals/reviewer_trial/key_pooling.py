"""Provenance for answer-key issues added by pooling reviewers' unmatched findings.

The protocol (who adjudicates, how severity is mapped, how a run is folded)
lives in ``corpus/README.md`` under "Pooling log" and "Answer-key pooling
protocol".
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PoolingRecord:
    """Where one pooled ground-truth issue came from."""

    diff_id: str
    issue_id: str
    pooling_date: str
    pooling_run: str
    provenance: dict[str, Any]


@dataclass(frozen=True)
class PoolingLog:
    """One PoolingRecord per pooled answer-key issue, stored as JSONL."""

    records: tuple[PoolingRecord, ...] = ()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w') as f:
            for record in self.records:
                f.write(json.dumps(asdict(record)))
                f.write('\n')

    @classmethod
    def load(cls, path: Path) -> PoolingLog:
        """A missing file is an empty log."""
        if not path.exists():
            return cls()
        return cls(records=tuple(
            PoolingRecord(**json.loads(line))
            for line in path.read_text().splitlines()
            if line.strip()
        ))
