"""The pack judge: plans/followup-packing-sweep-prd.md §4.2 and decision 9.

One curator-model call per candidate pack decides whether its members are one coherent task, and
the parser turns any verdict that breaks the pack into an audited reject. The packing sweep is
the only caller. ``task_curator.py`` supplies the shared heuristics and is imported, never edited.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from enum import StrEnum
from typing import Annotated, Any, Literal, Self, get_args

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    ValidationError,
    model_validator,
)
from shared.followup_items import FollowupItem
from shared.prompt_artifact import PromptSpec

from fused_memory.middleware.curator_types import clip_for_prompt
from fused_memory.middleware.followup_packing import CandidatePack, PackMember
from fused_memory.middleware.task_curator import _CURATOR_HEURISTICS

__all__ = [
    'CURATOR_PACK_OUTPUT_SCHEMA',
    'CURATOR_PACK_SPEC',
    'MAX_JUSTIFICATION_CHARS',
    'PACK_LEDGER_ROLE',
    'PACK_PROMPT_HARNESS_VERSION',
    'PackGroup',
    'PackVerdict',
    'VerdictDegrade',
    'VerdictKind',
    'parse_pack_verdict',
    'render_pack_member',
]

_PACK_CONTRACT = """\
You are the pack judge for the dark-factory orchestrator. You are given ONE candidate pack: two \
or more existing held follow-up tasks that a deterministic generator proposes to turn into one \
task. You decide whether they are one coherent task.

Read the combine guidance below with the pack as its subject: "combine" means "pack these \
members", and "the target" means the pack as a whole.

## Verdicts

- "pack": every member becomes one task. Emit exactly one group naming every member id.
- "split": some members form coherent tasks and the others do not. Emit one or more disjoint \
  groups, each naming at least two member ids and none naming every member. A member left out \
  of every group stays a separate task.
- "reject": the members are not one coherent task. Emit no groups, and name the rule that \
  fired in the top-level justification.

## Narrow lock, read for a pack

The pack header names its generator. For a "parent" pack (siblings filed from one parent \
task), the lock set that must stay narrow is the set of keys its members share. For a "recipe" \
pack (members of different parents that each name one mechanical recipe), the union keys are \
already within the planning budget; the question is whether every member applies the named \
recipe mechanically.

## Rewriting, not concatenating

For each group, write the title, description, details and justification as one coherent \
rewrite of its members' work, never a concatenation of their texts. Preserve every concrete \
code reference (file paths, test names, symbol names) verbatim.

## Derived server-side

Never emit survivor ids, follow-up items, files or priority: the server derives them from the \
members. The guidance's lines on unioning files_to_modify and carrying the higher priority do \
not apply to a pack.

## Output format

Return a single JSON object matching the provided schema: `verdict`, `groups` (each with \
`member_ids`, `title`, `description`, `details` and `justification`) and a pack-level \
`justification`. Name only member ids that appear in the pack.
"""

PACK_PROMPT_HARNESS_VERSION = 'curator-pack-v1'
PACK_LEDGER_ROLE = 'curator_pack'
CURATOR_PACK_SPEC = PromptSpec(
    prompt_id='curator_pack', contract=_PACK_CONTRACT, baseline_heuristics=_CURATOR_HEURISTICS,
)

VerdictKind = Literal['pack', 'split', 'reject']
MAX_JUSTIFICATION_CHARS = 500
_MIN_GROUP_MEMBERS = 2
_NO_PARENT = 'none'
_UNSTAMPED_BAND = 'unstamped'

CURATOR_PACK_OUTPUT_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'required': ['verdict', 'groups', 'justification'],
    'properties': {
        'verdict': {'type': 'string', 'enum': list(get_args(VerdictKind))},
        'groups': {
            'type': 'array',
            'items': {
                'type': 'object',
                'required': ['member_ids', 'title', 'description', 'details', 'justification'],
                'properties': {
                    'member_ids': {
                        'type': 'array', 'items': {'type': 'integer'}, 'minItems': _MIN_GROUP_MEMBERS,
                    },
                    'title': {'type': 'string'},
                    'description': {'type': 'string'},
                    'details': {'type': 'string'},
                    'justification': {'type': 'string'},
                },
            },
        },
        'justification': {'type': 'string'},
    },
}


class VerdictDegrade(StrEnum):
    """Why a verdict was turned into a reject: the parser violation, recorded with the raw text."""

    SCHEMA_INVALID = 'schema-invalid'
    UNKNOWN_MEMBER = 'unknown-member'
    OVERLAPPING_GROUPS = 'overlapping-groups'
    GROUP_TOO_SMALL = 'group-too-small'
    PACK_NOT_WHOLE = 'pack-not-whole'
    SPLIT_NOT_PROPER = 'split-not-proper'
    REJECT_WITH_GROUPS = 'reject-with-groups'


def _repeats_a_member(member_id_lists: Iterable[Sequence[int]]) -> bool:
    counts = Counter(member_id for ids in member_id_lists for member_id in ids)
    return any(n > 1 for n in counts.values())


def _distinct_sorted_group(member_ids: tuple[int, ...]) -> tuple[int, ...]:
    if _repeats_a_member([member_ids]):
        raise ValueError(f'member_ids must be distinct; got {list(member_ids)}')
    if len(member_ids) < _MIN_GROUP_MEMBERS:
        raise ValueError(f'a group names at least {_MIN_GROUP_MEMBERS} members; got {list(member_ids)}')
    return tuple(sorted(member_ids))


def _non_blank(text: str) -> str:
    if not text.strip():
        raise ValueError('must not be blank')
    return text


_Justification = Annotated[str, Field(max_length=MAX_JUSTIFICATION_CHARS)]


class PackGroup(BaseModel):
    """Members the judge would make one task, with the prose of that task."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    member_ids: Annotated[tuple[StrictInt, ...], AfterValidator(_distinct_sorted_group)]
    title: Annotated[str, AfterValidator(_non_blank)]
    description: str
    details: str
    justification: _Justification


class PackVerdict(BaseModel):
    """The judge's verdict on one pack; ``degraded`` names the violation that forced a reject."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    verdict: VerdictKind
    groups: tuple[PackGroup, ...]
    justification: _Justification
    degraded: VerdictDegrade | None = None

    @model_validator(mode='after')
    def _coherent(self) -> Self:
        broken = next((rule for rule, breaks in _SHAPE_RULES if breaks(self)), None)
        if broken is not None:
            raise ValueError(broken)
        return self

    @classmethod
    def degraded_reject(cls, reason: VerdictDegrade, justification: str = '') -> PackVerdict:
        return cls(
            verdict='reject', groups=(), justification=_clipped(justification), degraded=reason,
        )


_SHAPE_RULES: tuple[tuple[str, Callable[[PackVerdict], bool]], ...] = (
    ('a reject carries no groups', lambda v: v.verdict == 'reject' and bool(v.groups)),
    ('a pack carries exactly one group', lambda v: v.verdict == 'pack' and len(v.groups) != 1),
    ('a split carries at least one group', lambda v: v.verdict == 'split' and not v.groups),
    ('groups are disjoint', lambda v: _repeats_a_member(g.member_ids for g in v.groups)),
    ('only a reject is degraded', lambda v: v.degraded is not None and v.verdict != 'reject'),
)


def _clipped(justification: str) -> str:
    return justification[:MAX_JUSTIFICATION_CHARS]


class _RawGroup(BaseModel):
    member_ids: list[StrictInt]
    title: StrictStr
    description: StrictStr
    details: StrictStr
    justification: StrictStr


class _RawVerdict(BaseModel):
    """The schema's shape, with no bound and no pack-relative rule: what the LLM returned."""

    verdict: VerdictKind
    groups: list[_RawGroup]
    justification: StrictStr


def _whole_pack(group: _RawGroup, members: frozenset[int]) -> bool:
    return frozenset(group.member_ids) == members


_PackCheck = Callable[[_RawVerdict, frozenset[int]], bool]

_PACK_CHECKS: tuple[tuple[VerdictDegrade, _PackCheck], ...] = (
    (VerdictDegrade.UNKNOWN_MEMBER,
     lambda v, members: any(not members.issuperset(g.member_ids) for g in v.groups)),
    (VerdictDegrade.OVERLAPPING_GROUPS,
     lambda v, _members: _repeats_a_member(g.member_ids for g in v.groups)),
    (VerdictDegrade.GROUP_TOO_SMALL,
     lambda v, _members: any(len(g.member_ids) < _MIN_GROUP_MEMBERS for g in v.groups)),
    (VerdictDegrade.PACK_NOT_WHOLE,
     lambda v, members: v.verdict == 'pack'
     and not (len(v.groups) == 1 and _whole_pack(v.groups[0], members))),
    (VerdictDegrade.SPLIT_NOT_PROPER,
     lambda v, members: v.verdict == 'split'
     and (not v.groups or any(_whole_pack(g, members) for g in v.groups))),
    (VerdictDegrade.REJECT_WITH_GROUPS, lambda v, _members: v.verdict == 'reject' and bool(v.groups)),
)


def _verdict_from(raw: _RawVerdict) -> PackVerdict:
    return PackVerdict(
        verdict=raw.verdict,
        groups=tuple(
            PackGroup(
                member_ids=tuple(g.member_ids),
                title=g.title,
                description=g.description,
                details=g.details,
                justification=_clipped(g.justification),
            )
            for g in raw.groups
        ),
        justification=_clipped(raw.justification),
    )


def _shaped(raw: object) -> _RawVerdict | None:
    if not isinstance(raw, Mapping):
        return None
    try:
        return _RawVerdict.model_validate(dict(raw))
    except ValidationError:
        return None


def _raw_justification(raw: object) -> str:
    justification = raw.get('justification') if isinstance(raw, Mapping) else None
    return justification if isinstance(justification, str) else ''


def parse_pack_verdict(raw: object, pack: CandidatePack) -> PackVerdict:
    """The verdict *raw* states for *pack*, or a degraded reject naming why it cannot stand.

    Never raises: a malformed shape, or a verdict that breaks the pack, is a reject the caller
    ledgers with the raw text.
    """
    shaped = _shaped(raw)
    if shaped is None:
        return PackVerdict.degraded_reject(VerdictDegrade.SCHEMA_INVALID, _raw_justification(raw))
    members = frozenset(pack.members)
    violation = next((reason for reason, breaks in _PACK_CHECKS if breaks(shaped, members)), None)
    if violation is not None:
        return PackVerdict.degraded_reject(violation, shaped.justification)
    try:
        return _verdict_from(shaped)
    except ValidationError:
        return PackVerdict.degraded_reject(VerdictDegrade.SCHEMA_INVALID, shaped.justification)


def render_pack_member(
    m: PackMember,
    *,
    priority: str,
    description: str,
    details: str,
    items: Sequence[FollowupItem],
    desc_cap: int,
    details_cap: int,
) -> str:
    """One member as the judge reads it; the caps are the ones ``PoolEntry.render`` is given."""
    parent = _NO_PARENT if m.parent is None else str(m.parent)
    band = _UNSTAMPED_BAND if m.band is None else m.band
    lock_keys = ', '.join(m.lock_keys) if m.lock_keys else '(none)'
    return '\n'.join([
        f'[Task {m.task_id}] kind={m.kind} priority={priority} spawned_from={parent} band={band}',
        f'  title: {m.title}',
        f'  lock_keys: {lock_keys}',
        *(f'  item: {item.summary}' for item in items),
        f'  description: {clip_for_prompt(description, desc_cap)}',
        f'  details: {clip_for_prompt(details, details_cap)}',
    ])
