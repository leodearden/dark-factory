"""Provenance for answer-key issues added by pooling reviewers' unmatched findings.

The protocol (who adjudicates, how severity is mapped, how a run is folded)
lives in ``corpus/README.md`` under "Pooling log" and "Answer-key pooling
protocol".
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from orchestrator.evals.reviewer_trial.corpus import (
    CorpusDiff,
    CorpusManifest,
    GroundTruthIssue,
)

_ISSUE_FIELDS = frozenset(f.name for f in fields(GroundTruthIssue))
_PROVENANCE_KEY = 'x_provenance'
POOLED_MUTATION_PREFIX = 'pooled_'


def is_pooled_issue(issue: GroundTruthIssue) -> bool:
    return issue.mutation_type.startswith(POOLED_MUTATION_PREFIX)


@dataclass(frozen=True)
class PoolingRecord:
    """Where one pooled ground-truth issue came from.

    *provenance* is stored as a read-only copy: mappings become
    MappingProxyType and lists become tuples.
    """

    diff_id: str
    issue_id: str
    pooling_date: str
    pooling_run: str
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'provenance', _frozen(self.provenance))

    def to_json(self) -> str:
        return json.dumps({f.name: _thawed(getattr(self, f.name)) for f in fields(self)})


@dataclass(frozen=True)
class PoolingLog:
    """One PoolingRecord per pooled answer-key issue, stored as JSONL."""

    records: tuple[PoolingRecord, ...] = ()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w') as f:
            for record in self.records:
                f.write(record.to_json())
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


@dataclass(frozen=True)
class PoolingAudit:
    """(diff_id, issue_id) pairs breaking the pooling log <-> pooled key issue bijection.

    A dangling record names no pooled key issue; an unrecorded issue is a
    pooled key issue that no record names.
    """

    duplicate_records: tuple[tuple[str, str], ...]
    dangling_records: tuple[tuple[str, str], ...]
    unrecorded_issues: tuple[tuple[str, str], ...]

    @property
    def ok(self) -> bool:
        return not (self.duplicate_records or self.dangling_records or self.unrecorded_issues)


def audit_pooling(manifest: CorpusManifest, log: PoolingLog) -> PoolingAudit:
    """Report every pooling-log record and pooled key issue that lacks its counterpart."""
    pooled = {
        (diff.diff_id, gt.id)
        for diff in manifest.diffs for gt in diff.ground_truth if is_pooled_issue(gt)
    }
    counts = Counter((record.diff_id, record.issue_id) for record in log.records)
    return PoolingAudit(
        duplicate_records=tuple(sorted(pair for pair, n in counts.items() if n > 1)),
        dangling_records=tuple(sorted(pair for pair in counts if pair not in pooled)),
        unrecorded_issues=tuple(sorted(pair for pair in pooled if pair not in counts)),
    )


def fold_pooled_issues(
    manifest: CorpusManifest,
    log: PoolingLog,
    pooled: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    pooling_date: str,
    pooling_run: str,
) -> tuple[CorpusManifest, PoolingLog]:
    """Append each diff's pooled entries after its existing key, recording their provenance.

    *pooled* maps diff_id to entries carrying the GroundTruthIssue fields plus
    ``x_provenance``; each entry's mutation_type must start with
    POOLED_MUTATION_PREFIX. Returns new values; the inputs are left untouched.
    """
    unknown = sorted(set(pooled) - {diff.diff_id for diff in manifest.diffs})
    if unknown:
        raise ValueError(f'pooled entries name diff_ids absent from the manifest: {unknown}')
    new_diffs: list[CorpusDiff] = []
    new_records: list[PoolingRecord] = []
    for diff in manifest.diffs:
        split = [_split_entry(diff.diff_id, entry) for entry in pooled.get(diff.diff_id, ())]
        issues = [issue for issue, _ in split]
        _reject_repeated_ids(diff, issues)
        new_diffs.append(replace(diff, ground_truth=[*diff.ground_truth, *issues]))
        new_records.extend(
            PoolingRecord(diff.diff_id, issue.id, pooling_date, pooling_run, provenance)
            for issue, provenance in split
        )
    return (
        replace(manifest, diffs=new_diffs),
        PoolingLog(records=(*log.records, *new_records)),
    )


def _split_entry(
    diff_id: str, entry: Mapping[str, Any],
) -> tuple[GroundTruthIssue, Mapping[str, Any]]:
    where = f'pooled entry {entry.get("id")!r} for diff {diff_id!r}'
    if _PROVENANCE_KEY not in entry:
        raise ValueError(f'{where} lacks {_PROVENANCE_KEY!r}')
    unexpected = sorted(set(entry) - _ISSUE_FIELDS - {_PROVENANCE_KEY})
    if unexpected:
        raise ValueError(f'{where} has unexpected keys {unexpected}')
    missing = sorted(_ISSUE_FIELDS - set(entry))
    if missing:
        raise ValueError(f'{where} is missing issue fields {missing}')
    issue = GroundTruthIssue(**{name: entry[name] for name in _ISSUE_FIELDS})
    if not is_pooled_issue(issue):
        raise ValueError(
            f'{where} has mutation_type {issue.mutation_type!r}, '
            f'which lacks the pooled prefix {POOLED_MUTATION_PREFIX!r}'
        )
    return issue, entry[_PROVENANCE_KEY]


def _reject_repeated_ids(diff: CorpusDiff, issues: Sequence[GroundTruthIssue]) -> None:
    counts = Counter(gt.id for gt in [*diff.ground_truth, *issues])
    repeated = sorted({issue.id for issue in issues if counts[issue.id] > 1})
    if repeated:
        raise ValueError(
            f'pooled issue ids {repeated} for diff {diff.diff_id!r} repeat an id '
            'already in its answer key or elsewhere in the batch'
        )


def _frozen(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _frozen(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_frozen(item) for item in value)
    return value


def _thawed(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thawed(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thawed(item) for item in value]
    return value
