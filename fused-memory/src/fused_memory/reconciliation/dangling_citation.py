"""The dangling-citation handoff record Stage 3 emits and Stage 1 consumes.

Producer: ``reconciliation/cross_run_citation_scan.py::scan_cross_run_citations``.
Consumer's tool: ``reconciliation/citation_repair.py::repair_memory_citation``.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, fields
from typing import Any

from fused_memory.models.reconciliation import FindingSeverity
from fused_memory.utils.validation import is_full_uuid

logger = logging.getLogger(__name__)

DANGLING_CITATION_KEY = 'dangling_citation'
DANGLING_CROSS_RUN_CITATION_FLAG_TYPE = 'dangling_cross_run_citation'
DANGLING_CITATION_LINE_LABEL = 'Dangling cross-run citation:'

_CATEGORY = 'cross_store_inconsistency'
_UUID_FIELDS = ('target_run_id', 'memory_id')
_TEXT_FIELDS = ('finding_id', 'store')


def _require_valid(values: dict[str, Any]) -> None:
    for name in _UUID_FIELDS:
        if not is_full_uuid(values[name]):
            raise ValueError(
                f'DanglingCitation.{name} must be a full UUID, got {values[name]!r}'
            )
    for name in _TEXT_FIELDS:
        value = values[name]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f'DanglingCitation.{name} must be a non-blank str, got {value!r}'
            )


@dataclass(frozen=True)
class DanglingCitation:
    target_run_id: str
    finding_id: str
    memory_id: str
    store: str

    def __post_init__(self) -> None:
        _require_valid(self.to_dict())

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


_FIELD_NAMES = tuple(f.name for f in fields(DanglingCitation))


def build_dangling_citation_finding(record: DanglingCitation) -> dict[str, Any]:
    return {
        'description': (
            f'Finding {record.finding_id} of settled run {record.target_run_id} cites '
            f'{record.store} memory {record.memory_id}, which no longer resolves '
            '(confirmed absent by a point read).'
        ),
        'severity': FindingSeverity.minor.value,
        'category': _CATEGORY,
        'flag_type': DANGLING_CROSS_RUN_CITATION_FLAG_TYPE,
        'actionable': True,
        'suggested_action': (
            'Call mcp__recon-report__repair_memory_citation with '
            "reason='memory_not_found' and the target_run_id, finding_id, memory_id "
            f'and store named on the "{DANGLING_CITATION_LINE_LABEL}" line. Pass '
            'replacement_memory_id only for a memory confirmed to supersede the cited '
            'one; otherwise omit it to drop the citation.'
        ),
        DANGLING_CITATION_KEY: record.to_dict(),
    }


def dangling_citation_from_finding(finding: object) -> DanglingCitation | None:
    """The record carried by *finding*, or None when absent or malformed; never raises."""
    if not isinstance(finding, dict) or DANGLING_CITATION_KEY not in finding:
        return None
    raw = finding[DANGLING_CITATION_KEY]
    try:
        if not isinstance(raw, dict):
            raise ValueError(f'expected a mapping, got {type(raw).__name__}')
        values: dict[str, Any] = {name: raw.get(name) for name in _FIELD_NAMES}
        return DanglingCitation(**values)
    except ValueError as exc:
        logger.warning('Ignoring malformed %s record %r: %s', DANGLING_CITATION_KEY, raw, exc)
        return None


def dangling_citation_identity(finding: object) -> list[str]:
    record = dangling_citation_from_finding(finding)
    if record is None:
        return []
    return [record.target_run_id, record.finding_id, record.memory_id]


def render_dangling_citation_line(finding: object) -> str | None:
    record = dangling_citation_from_finding(finding)
    if record is None:
        return None
    pairs = ' '.join(f'{name}={value}' for name, value in record.to_dict().items())
    return f'{DANGLING_CITATION_LINE_LABEL} {pairs}'
