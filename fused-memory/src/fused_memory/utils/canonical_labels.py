"""The canonical task-label grammar: named fragments composed into six rules.

- _TASK_NODE_NAME_PATTERN: a whole own-project node name, 'task #1153'.
- _QUALIFIED_NODE_NAME_PATTERN: a whole foreign node name, 'reify:132'.
- _TASK_VOCABULARY_QUALIFIER: a lowercased qualifier that is task vocabulary,
  'sub_task'. It is deliberately not built from _TASK_WORD, so 'Task' counts
  only once its caller has lowercased it.
- _LOCAL_MENTION_PATTERN: a bare mention in prose, 'blocked on task-173'. After
  the plural task word it also reads the list that follows, 'tasks 2226 and 2227',
  and it captures a name directly before the task word, 'Reify task 3674', which
  qualifies the mention only when it names a registered project.
- _LISTED_NUMBER_PATTERN: one number inside that list.
- _QUALIFIED_REF_PATTERN: a project-qualified mention, 'see dark_factory:2500'.

Precision over recall: the pads around '#' and ':' are space or tab, never a
line break; a qualified ref in prose takes no pad at its colon at all, because
'word: N' is prose, while a whole node name's colon may be padded; the task
word and the digits are ASCII; and the right edge refuses any word character,
so a mixed digit run or a hash such as '8c7e163' yields nothing. One accepted loss: '/' lists after the plural word ('tasks 2207/2210'),
so a progress fraction there ('tasks 3/5 done') reads as two tasks; its head
number was already one before lists were read. The measured cases live in
tests/test_canonical_labels.py, and the reasons in
plans/review-all-s26-memory-service-write-path-and-graph-client-prd.md
§Rationale rehomed.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Self

from fused_memory.utils.validation import (
    PathShapedProjectIdError,
    canonicalize_project_id,
)

logger = logging.getLogger(__name__)

# Per process, not per registry: once set, a differently broken registry is silent.
_all_path_shaped_warned: bool = False

#: Each referent kind and the label its own-project node name renders; a new kind
#: also needs its own rules before anything parses or scans it.
KIND_LABELS: Mapping[str, str] = MappingProxyType({'task': 'Task'})


class _Fragment(str):
    """A named grammar fragment that keeps the parts it was joined from, so a
    rule's use of a shared fragment is checkable by identity at any depth."""

    parts: tuple[str, ...]

    def __new__(cls, *parts: str) -> Self:
        fragment = super().__new__(cls, ''.join(parts))
        fragment.parts = parts
        return fragment


@dataclass(frozen=True)
class _Rule:
    """A compiled rule and the fragments it was composed from."""

    parts: tuple[str, ...]
    regex: re.Pattern[str]


def _compile(*parts: str, anchored: bool) -> _Rule:
    body = ''.join(parts)
    return _Rule(parts=parts, regex=re.compile(rf'^{body}\Z' if anchored else body))


def _either(*branches: str) -> _Fragment:
    alternation = [part for branch in branches for part in ('|', branch)][1:]
    return _Fragment('(?:', *alternation, ')')


_TASK_STEM = _Fragment(r'[Tt][Aa][Ss][Kk]')
_PLURAL = _Fragment(r'(?P<plural>[Ss])')
_TASK_WORD = _Fragment(_TASK_STEM, _PLURAL, '?')
_PAD = _Fragment(r'[ \t]*')
_BLANKS = _Fragment(r'[ \t]+')
_WIDE_PAD = _Fragment(r'\s*')
_GAP = _Fragment(r'\s+')
_COLON = _Fragment(':')
_PADDED_COLON = _Fragment(_PAD, _COLON, _PAD)
_MARK = _Fragment(_PAD, '[#:]', _PAD)
_SEPARATOR = _either(_MARK, _GAP)
_MENTION_SEPARATOR = _either(_MARK, _GAP, r'[-/]')
_DIGITS = _Fragment(r'[0-9]+')
_NUMBER = _Fragment('(?P<number>', _DIGITS, ')')
_NUMBER_END = _Fragment(r'(?!\w)')
_QUALIFIER = _Fragment(r'(?P<qualifier>[A-Za-z][A-Za-z0-9_-]{2,})')
_CONJUNCTION = _Fragment('(?:and|or)')
_LIST_SEPARATOR = _either(
    _Fragment(_PAD, ',', _PAD, '(?:', _CONJUNCTION, _BLANKS, ')?'),
    _Fragment(_BLANKS, _CONJUNCTION, _BLANKS),
    _Fragment(_PAD, '/', _PAD),
)
_LIST_ITEM = _Fragment('(?:', _LIST_SEPARATOR, '#?', _DIGITS, _NUMBER_END, ')')
_LIST_CONTINUATION = _Fragment('(?P<continuation>', _LIST_ITEM, '*)')
_MENTION_LEFT_EDGE = _Fragment(r'(?<![\w:-])')
_REF_LEFT_EDGE = _Fragment(r'(?<![\w:/.-])')
_PROSE_QUALIFIER = _Fragment('(?:', _REF_LEFT_EDGE, _QUALIFIER, ',?', _BLANKS, ')?')

_TASK_NODE_NAME_PATTERN = _compile(
    _WIDE_PAD, _TASK_WORD, _SEPARATOR, _NUMBER, _WIDE_PAD, anchored=True
)
_QUALIFIED_NODE_NAME_PATTERN = _compile(
    _PAD, _QUALIFIER, _PADDED_COLON, _NUMBER, _PAD, anchored=True
)
_TASK_VOCABULARY_QUALIFIER = _compile(r'(sub_?)?tasks?', anchored=True)
_LOCAL_MENTION_PATTERN = _compile(
    _PROSE_QUALIFIER,
    _MENTION_LEFT_EDGE,
    _TASK_WORD,
    _MENTION_SEPARATOR,
    _NUMBER,
    _NUMBER_END,
    _LIST_CONTINUATION,
    anchored=False,
)
_LISTED_NUMBER_PATTERN = _compile(_NUMBER, _NUMBER_END, anchored=False)
_QUALIFIED_REF_PATTERN = _compile(
    _REF_LEFT_EDGE, _QUALIFIER, _COLON, _NUMBER, _NUMBER_END, anchored=False
)


@dataclass(frozen=True, kw_only=True)
class Referent:
    """One thing a canonical label refers to: a kind, a project, a number.

    ``number`` keeps its digits verbatim, so '0132' and '132' are different
    referents. An empty ``project_id`` means the own project; otherwise it is a
    canonical id naming a different project.
    """

    number: str
    kind: str = 'task'
    project_id: str = ''

    def __post_init__(self) -> None:
        if self.kind not in KIND_LABELS:
            raise ValueError(
                f'unregistered referent kind {self.kind!r}; registered kinds are '
                f'{sorted(KIND_LABELS)}. Add it to the KIND_LABELS literal in canonical_labels.py.'
            )

    @property
    def node_name(self) -> str:
        """'<project_id>:<number>' for a foreign referent, else e.g. 'Task 132'."""
        if self.project_id:
            return f'{self.project_id}:{self.number}'
        return f'{KIND_LABELS[self.kind]} {self.number}'


@dataclass(frozen=True)
class LabelScan:
    """The referents one body of content names, in first-seen order, de-duplicated.

    Frozen with tuple fields, so no consumer can add a referent the scan refused.
    ``refs`` are safe to act on. ``ambiguous`` holds every referent whose number
    the content claims both bare and foreign-qualified; a consumer must not act
    on those.
    """

    refs: tuple[Referent, ...] = ()
    ambiguous: tuple[Referent, ...] = ()


def is_task_vocabulary_qualifier(project_id: str) -> bool:
    """Is the canonicalized *project_id* task vocabulary ('task', 'sub_tasks')
    rather than a project ('taskmaster')?"""
    return _TASK_VOCABULARY_QUALIFIER.regex.match(project_id) is not None


def _project_of(qualifier: str) -> str | None:
    try:
        project_id = canonicalize_project_id(qualifier)
    except PathShapedProjectIdError:
        return None
    return None if is_task_vocabulary_qualifier(project_id) else project_id


def parse_node_name(name: str) -> Referent | None:
    """Parse a whole entity *name* as a task label, or return None.

    'task #1153' parses to the own-project ``Task 1153`` and 'Reify:132' to the
    foreign ``reify:132``. A name that only mentions a task, or whose qualifier
    is path-shaped or task vocabulary, returns None.
    """
    local = _TASK_NODE_NAME_PATTERN.regex.match(name)
    if local is not None:
        return Referent(number=local.group('number'))
    qualified = _QUALIFIED_NODE_NAME_PATTERN.regex.match(name)
    if qualified is None:
        return None
    project_id = _project_of(qualified.group('qualifier'))
    if project_id is None:
        return None
    return Referent(project_id=project_id, number=qualified.group('number'))


def _canonical_allowlist(known_project_ids: Iterable[str] | None) -> frozenset[str] | None:
    """The usable canonical registry ids, or None (permissive) when there are none.

    Path-shaped and blank entries are skipped. A supplied registry with no
    usable entry warns once per process, because the filter then stops narrowing.
    """
    global _all_path_shaped_warned
    if not known_project_ids:
        return None
    supplied = 0
    canonical: set[str] = set()
    for raw in known_project_ids:
        supplied += 1
        try:
            canonical.add(canonicalize_project_id(raw))
        except PathShapedProjectIdError:
            continue
    canonical.discard('')
    if canonical:
        return frozenset(canonical)
    if supplied and not _all_path_shaped_warned:
        _all_path_shaped_warned = True
        logger.warning(
            'cross-project ref allowlist disabled: none of the %d supplied '
            'known_project_ids is a usable project id (each was path-shaped or blank), '
            'so the filter runs in PERMISSIVE mode. Likely cause: the registry was wired '
            'with project ROOTS where project IDs were expected. Logged once per process: '
            're-check the registry rather than reading a silent log as an all-clear.',
            supplied,
        )
    return None


def _mentioned_numbers(mention: re.Match[str]) -> Iterator[tuple[int, str]]:
    """Each number a mention names, with its offset: the head, then the list
    that follows a plural task word."""
    yield mention.start(), mention.group('number')
    if mention.group('plural') is None:
        return
    for listed in _LISTED_NUMBER_PATTERN.regex.finditer(
        mention.string, mention.start('continuation'), mention.end('continuation')
    ):
        yield listed.start(), listed.group('number')


def _prose_project(mention: re.Match[str], allowlist: frozenset[str] | None) -> str | None:
    """The registered project a name directly before the task word names, or None."""
    qualifier = mention.group('qualifier')
    if qualifier is None or allowlist is None:
        return None
    project_id = _project_of(qualifier)
    return project_id if project_id in allowlist else None


def scan_content(
    content: str,
    *,
    group_id: str,
    known_project_ids: Iterable[str] | None = None,
) -> LabelScan:
    """Scan the verbatim *content* for every task referent it mentions.

    A ref qualified with *group_id*'s own project is reclassified as an
    own-project referent. When *known_project_ids* (any iterable, consumed
    once; a mapping's keys work) has a usable entry, foreign refs outside it
    are dropped, and a registered project's name directly before the task word
    ('Reify task 3674', 'in reify, task 9727') qualifies that mention, its list
    included; without a usable registry that name is ignored. Narrowing never
    mints a referent from an unregistered qualifier, and a dropped candidate
    keeps the contest it creates with a bare mention of its number. A
    path-shaped *group_id* yields an empty scan. Accepted blind spots: bare digits, task titles, non-ASCII
    digits or lookalike letters, a qualified ref spaced around its colon
    ('reify: 132') or hard-wrapped across a line break, and a list after the
    singular task word.
    """
    if not content:
        return LabelScan()
    try:
        local_project = canonicalize_project_id(group_id)
    except PathShapedProjectIdError:
        return LabelScan()
    allowlist = _canonical_allowlist(known_project_ids)
    found: list[tuple[int, Referent]] = []
    bare: set[tuple[str, str]] = set()
    foreign: set[tuple[str, str]] = set()

    for mention in _LOCAL_MENTION_PATTERN.regex.finditer(content):
        prose_project = _prose_project(mention, allowlist)
        for offset, number in _mentioned_numbers(mention):
            if prose_project is None or prose_project == local_project:
                referent = Referent(number=number)
                if prose_project is None:
                    bare.add((referent.kind, referent.number))
            else:
                referent = Referent(project_id=prose_project, number=number)
                foreign.add((referent.kind, referent.number))
            found.append((offset, referent))

    for match in _QUALIFIED_REF_PATTERN.regex.finditer(content):
        project_id = _project_of(match.group('qualifier'))
        if project_id is None:
            continue
        if project_id == local_project:
            found.append((match.start(), Referent(number=match.group('number'))))
            continue
        referent = Referent(project_id=project_id, number=match.group('number'))
        foreign.add((referent.kind, referent.number))
        if allowlist is None or project_id in allowlist:
            found.append((match.start(), referent))

    found.sort(key=lambda item: item[0])
    deduped = list(dict.fromkeys(referent for _offset, referent in found))
    contested = bare & foreign
    return LabelScan(
        refs=tuple(r for r in deduped if (r.kind, r.number) not in contested),
        ambiguous=tuple(r for r in deduped if (r.kind, r.number) in contested),
    )
