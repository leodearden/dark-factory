"""The link adjudicator: the rater brief's question, asked of a model.

Contract: ``plans/write-triage-link-healing-prd.md`` H2. The model reads the
rater brief (:data:`RATER_BRIEF_PATH`) verbatim as its instructions, and the
brief's seven words are the scale. That scale has one home in code,
``link_heal.Verdict``. This module imports it and never respells it; a test
mirrors both against the live brief.

The adjudicator is pure in (child text, parent text). A :class:`LinkPair`
carries nothing else, so ratings, kinds, the contested flag and the pair's
source cannot reach the model. Of the maintenance package, it imports only
``link_heal``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from fused_memory.maintenance.link_heal import Verdict

RATER_BRIEF_PATH = (
    Path(__file__).resolve().parents[3] / 'calibration' / 'write_triage_rater_brief.md'
)

TRUNCATION_MARKER = '…[truncated, {total} chars total]'
_MARKER_PREFIX = TRUNCATION_MARKER.split('{total}')[0]


def cap_text(text: str, field_chars: int) -> str:
    """*text* cut to *field_chars* with the brief's marker; a capped text stays as it is."""
    if len(text) <= field_chars or _MARKER_PREFIX in text[field_chars:]:
        return text
    return text[:field_chars] + TRUNCATION_MARKER.format(total=len(text))


VERDICT_OUTPUT_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'properties': {
        'verdicts': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'id': {'type': 'string'},
                    'verdict': {'type': 'string', 'enum': [verdict.value for verdict in Verdict]},
                    'reason': {'type': 'string'},
                },
                'required': ['id', 'verdict', 'reason'],
                'additionalProperties': False,
            },
        },
    },
    'required': ['verdicts'],
    'additionalProperties': False,
}


@dataclass(frozen=True)
class LinkPair:
    """One (child, parent) pair to judge. ``key`` names it to the caller, never to the model."""

    key: str
    child_text: str
    parent_text: str


@dataclass(frozen=True)
class ShardItem:
    item_id: str
    child_text: str
    parent_text: str


@dataclass(frozen=True)
class Shard:
    model: str
    items: tuple[ShardItem, ...]


@dataclass(frozen=True)
class ShardReply:
    success: bool
    structured_output: Any = None
    cost_usd: float = 0.0
    detail: str = ''


class AdjudicationFailure(StrEnum):
    CLI_FAILED = 'cli_failed'
    PARSE_FAILURE = 'parse_failure'
    OMITTED_ID = 'omitted_id'
    DUPLICATE_ID = 'duplicate_id'
    UNKNOWN_ID = 'unknown_id'
    NOT_ATTEMPTED = 'not_attempted'


@dataclass(frozen=True)
class LinkVerdict:
    """The model's verdict on one pair, or why there is none. Never both."""

    key: str
    model: str
    verdict: Verdict | None = None
    reason: str = ''
    failure: AdjudicationFailure | None = None
    detail: str = ''

    def __post_init__(self) -> None:
        if (self.verdict is None) == (self.failure is None):
            raise ValueError(
                f'{self.key}: exactly one of verdict ({self.verdict}) and '
                f'failure ({self.failure}) must be set',
            )

    @property
    def failed(self) -> bool:
        return self.failure is not None


_PROTOCOL = (
    'Answer through the structured output only. The user message holds the '
    'items, each headed ITEM <id> and followed by its CHILD and PARENT texts. '
    'Return one entry per ITEM id, using each id exactly once, with the '
    'verdict word and a one-sentence reason.'
)


def render_system_prompt(brief_text: str) -> str:
    return f'{brief_text.rstrip()}\n\n## Protocol for this call\n\n{_PROTOCOL}\n'


def render_user_prompt(shard: Shard) -> str:
    return '\n\n'.join(
        f'ITEM {item.item_id}\nCHILD: {item.child_text}\nPARENT: {item.parent_text}'
        for item in shard.items
    )
