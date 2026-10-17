"""The ruling-overreach guard's class vocabulary and judge core.

Which writes the guard judges, the ruling-shape classifiers recorded beside
every judgement, the label vocabulary, and the pure judge that labels an
episode's minted edges with one LLM call, per
docs/ruling-overreach-guard-design-2026-10-01/design.md §4 and §8. Activation
is the structured category alone; the other four classifiers are recorded,
never acted on. Nothing here does I/O except through the LLM it is handed.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

from graphiti_core.prompts.models import Message
from pydantic import BaseModel, Field

from fused_memory.models.enums import MemoryCategory

GUARDED_CATEGORIES: frozenset[MemoryCategory] = frozenset({MemoryCategory.decisions_and_rationale})


def is_guarded_category(category: MemoryCategory | None) -> bool:
    return category in GUARDED_CATEGORIES


HEAD_CHARS = 200
_RULING_LEXEME = re.compile(r'\b(ruling|ruled)\b', re.IGNORECASE)
_DECISION_LEXEME = re.compile(r'\b(ruling|ruled|decision|decisions|decided)\b', re.IGNORECASE)
_DECISION_ANCHOR = re.compile(r'\besc-\d+-\d+\b|\([^)]*\b\d{4}-\d{2}-\d{2}')
_RULING_HEADER = re.compile(r'\s*RULING\b', re.IGNORECASE)

EpisodePredicate = Callable[[MemoryCategory | None, str], bool]


def _category_decisions(category: MemoryCategory | None, content: str) -> bool:
    return is_guarded_category(category)


def _header_ruling_paren(category: MemoryCategory | None, content: str) -> bool:
    return content.lstrip().startswith('RULING (')


def _header_ruling(category: MemoryCategory | None, content: str) -> bool:
    return _RULING_HEADER.match(content) is not None


def _ruling_lexeme_head(category: MemoryCategory | None, content: str) -> bool:
    return _RULING_LEXEME.search(content[:HEAD_CHARS]) is not None


def _decision_anchor_head(category: MemoryCategory | None, content: str) -> bool:
    head = content[:HEAD_CHARS]
    return bool(_DECISION_LEXEME.search(head) and _DECISION_ANCHOR.search(head))


EPISODE_CLASSIFIERS: Mapping[str, EpisodePredicate] = MappingProxyType({
    'category_decisions': _category_decisions,
    'header_ruling_paren': _header_ruling_paren,
    'header_ruling': _header_ruling,
    'ruling_lexeme_head': _ruling_lexeme_head,
    'decision_anchor_head': _decision_anchor_head,
})
CLASSIFIER_NAMES: tuple[str, ...] = tuple(EPISODE_CLASSIFIERS)


def matching_classifiers(category: MemoryCategory | None, content: str) -> tuple[str, ...]:
    return tuple(name for name, holds in EPISODE_CLASSIFIERS.items() if holds(category, content))


LABELS: tuple[str, ...] = (
    'holding', 'bookkeeping', 'context', 'overreach', 'misbound', 'unjudgeable',
)
"""The pre-registered vocabulary; definitions in design.md §2.6."""

OVERREACH = 'overreach'


# --------------------------------------------------------------------------- #
# The judge (design.md §2.6, §2.10 and §6)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class JudgeEdge:
    uuid: str
    fact: str
    source_name: str
    target_name: str


@dataclass(frozen=True)
class EdgeJudgement:
    edge_uuid: str
    label: str
    rationale: str


class JudgeReplyError(ValueError):
    """A judge reply that must not be recorded: it does not label exactly the offered edges."""


def minted_edges(episode_uuid: str, rows: Iterable[Mapping[str, Any]]) -> tuple[JudgeEdge, ...]:
    """The rows *episode_uuid* minted: ``episodes[0]`` is the minting episode (design.md §2.5)."""
    return tuple(
        JudgeEdge(
            uuid=row['uuid'], fact=row['fact'],
            source_name=row['source_name'], target_name=row['target_name'],
        )
        for row in rows
        if (episodes := row['episodes']) and episodes[0] == episode_uuid
    )


JUDGE_SYSTEM_PROMPT = """\
You audit the facts a knowledge-graph extractor minted from one decision record.
Judge each fact against the episode text alone, all of it, not only its opening.
Give every fact exactly one label, deciding in this order:
1. Content. Does the fact assert anything the record did not decide or state? \
If yes, the label is overreach, whatever the fact is bound to.
2. Binding. The fact is faithful, but is one endpoint the wrong entity, so that \
reading it off that node attributes it to the wrong subject? If yes, the label is misbound.
3. Kind. Otherwise the fact is faithful and correctly bound: label it holding, \
bookkeeping or context.
4. Use unjudgeable only when the episode text cannot decide.

Labels:
- holding: states what the record decided (the adopted option, rule or disposition) \
within the scope the record gives it. A rejection that is itself the decision \
("X was chosen over Y") is a holding.
- bookkeeping: who, when, which escalation, task or commit, or a status. True as \
stated, and not itself a decision.
- context: faithfully restates grounds, measurements, background or history, with \
the record's own qualification intact. A rejected option restated WITH its \
rejection is context.
- overreach: asserts more than the record decided, in any one of these forms:
  (i) it generalises the holding beyond its stated scope;
  (ii) it states an implication or consequence the record did not adopt, including \
a vague relational verb ("impacts", "affects", "implies") linking two things the \
record does not link;
  (iii) it presents a weighed, rejected, proposed or counter-signal option as \
adopted or true; a proposal stated as current fact counts;
  (iv) it drops a qualifier the record attaches where that qualifier changes the \
holding's scope (who, which, when, under what condition). Dropping an incidental \
detail that does not change scope is not overreach.
  A garbled fact is overreach (ii): an inverted causal link, a misattributed actor, \
or a quantity assigned to the wrong thing asserts something the record did not state.
- misbound: faithful, but attached to the wrong endpoint, for example a node the \
episode never names for that fact. Keep it separate: it is never overreach.
- unjudgeable: cannot be decided from the episode text, for example the fact rests \
on a referent the episode never names.

Every rationale must quote, or point at, the episode span it rests on.
Return exactly one judgement per fact, keyed by the fact's E-number."""


def _ordinal(index: int) -> str:
    return f'E{index}'


def build_judge_prompt(episode_body: str, edges: Sequence[JudgeEdge]) -> str:
    facts = [
        f'{_ordinal(i)}: {edge.fact} [source: {edge.source_name} -> target: {edge.target_name}]'
        for i, edge in enumerate(edges, start=1)
    ]
    return '\n'.join(['EPISODE:', episode_body, '', 'FACTS:', *facts])


def _prompt_version() -> str:
    probe = build_judge_prompt('<episode>', [JudgeEdge('', '<fact>', '<source>', '<target>')])
    digest = hashlib.sha256(f'{JUDGE_SYSTEM_PROMPT}\x00{probe}'.encode()).hexdigest()
    return f'overreach-judge-v1-{digest[:12]}'


PROMPT_VERSION = _prompt_version()
"""Derived from the rubric and the rendered user template, so it cannot drift from either."""


class _JudgedEdge(BaseModel):
    edge: str = Field(description="The fact's E-number, for example E1.")
    label: str = Field(json_schema_extra={'enum': list(LABELS)})
    rationale: str = Field(description='Quote, or point at, the episode span the label rests on.')


class _JudgeReply(BaseModel):
    judgements: list[_JudgedEdge]


_REPLY_FIELDS = ('edge', 'label', 'rationale')


def _reply_items(reply: Any) -> list[Mapping[str, str]]:
    if not isinstance(reply, Mapping) or not isinstance(reply.get('judgements'), list):
        raise JudgeReplyError("a judge reply is {'judgements': [...]}")
    items = reply['judgements']
    for item in items:
        if not isinstance(item, Mapping) or not all(
            isinstance(item.get(name), str) for name in _REPLY_FIELDS
        ):
            raise JudgeReplyError(f'a judgement must carry string fields {_REPLY_FIELDS}: {item!r}')
    return items


def _check_items(items: list[Mapping[str, str]], offered: Mapping[str, JudgeEdge]) -> None:
    for item in items:
        if item['label'] not in LABELS:
            raise JudgeReplyError(f"{item['edge']} has label {item['label']!r}, outside {LABELS}")
        if not item['rationale'].strip():
            raise JudgeReplyError(f"{item['edge']} has a blank rationale")
    keys = Counter(item['edge'] for item in items)
    if unknown := sorted(keys.keys() - offered.keys()):
        raise JudgeReplyError(f'judgements for edges never offered: {unknown}')
    if duplicated := sorted(key for key, count in keys.items() if count > 1):
        raise JudgeReplyError(f'edges judged more than once: {duplicated}')
    if unjudged := [key for key in offered if key not in keys]:
        raise JudgeReplyError(f'offered edges left unjudged: {unjudged}')


def parse_judge_reply(reply: Any, edges: Sequence[JudgeEdge]) -> tuple[EdgeJudgement, ...]:
    """Strictly map a decoded reply back to *edges*' uuids, in edge order."""
    offered = {_ordinal(i): edge for i, edge in enumerate(edges, start=1)}
    items = _reply_items(reply)
    _check_items(items, offered)
    by_key = {item['edge']: item for item in items}
    return tuple(
        EdgeJudgement(edge_uuid=edge.uuid, label=by_key[key]['label'], rationale=by_key[key]['rationale'])
        for key, edge in offered.items()
    )


class StructuredLLM(Protocol):
    """The slice of graphiti_core's ``LLMClient`` the judge calls."""

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        *,
        prompt_name: str | None = None,
    ) -> dict[str, Any]: ...


async def judge_edges(
    llm: StructuredLLM, episode_body: str, edges: Sequence[JudgeEdge],
) -> tuple[EdgeJudgement, ...]:
    """Label every one of *edges* with ONE LLM call; no edges means no call.

    Catches nothing. The caller owns fail-open: services/overreach_shadow.py
    records nothing for a failed judgement and never fails the write, and
    scripts/eval_overreach_judge.py counts an unjudged edge against recall.
    """
    if not edges:
        return ()
    messages = [
        Message(role='system', content=JUDGE_SYSTEM_PROMPT),
        Message(role='user', content=build_judge_prompt(episode_body, edges)),
    ]
    reply = await llm.generate_response(
        messages, response_model=_JudgeReply, prompt_name='overreach_judge',
    )
    return parse_judge_reply(reply, edges)
