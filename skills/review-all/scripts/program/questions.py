"""The program's open design questions as the ledger records them, and which still await a ruling."""
from __future__ import annotations

import types
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import state


@dataclass(frozen=True)
class RecordedQuestion:
    ref: state.ItemRef
    question: str
    options: tuple[str, ...]
    recommendation: str
    needs_human: bool
    lead_rulable: bool
    source: state.QuestionSource
    return_name: str


def question_entry(stream: str, question: state.OpenQuestion, return_name: str) -> dict[str, Any]:
    """The fields of a ``question`` ledger line for one open question of *stream*'s return."""
    return {
        'stream': stream, 'id': question.id, 'question': question.question,
        'options': list(question.options), 'recommendation': question.recommendation,
        'needs_human': question.needs_human, 'lead_rulable': question.lead_rulable,
        'source': str(question.source), 'return': return_name,
    }


def _recorded(line: Mapping[str, Any]) -> RecordedQuestion:
    return RecordedQuestion(
        ref=state.ItemRef(line['stream'], line['id']), question=line['question'],
        options=tuple(line['options']), recommendation=line['recommendation'],
        needs_human=line['needs_human'], lead_rulable=line.get('lead_rulable', False),
        source=state.QuestionSource(line['source']), return_name=line['return'],
    )


def recorded_questions(
    ledger: Iterable[Mapping[str, Any]],
) -> Mapping[state.ItemRef, RecordedQuestion]:
    """The latest ``question`` line of each item, in the order the items were first recorded."""
    recorded: dict[state.ItemRef, RecordedQuestion] = {}
    for line in ledger:
        if line['op'] == state.LedgerOp.QUESTION:
            question = _recorded(line)
            recorded[question.ref] = question
    return types.MappingProxyType(recorded)


def _latest_answering(rulings: Iterable[Mapping[str, Any]]) -> dict[state.ItemRef, Mapping[str, Any]]:
    latest: dict[state.ItemRef, Mapping[str, Any]] = {}
    for ruling in rulings:
        for text in ruling.get('answers', ()):
            latest[state.ItemRef.parse(text)] = ruling
    return latest


def pending_items(
    recorded: Iterable[state.ItemRef], rulings: Iterable[Mapping[str, Any]],
) -> tuple[state.ItemRef, ...]:
    """The recorded items no ruling answers, or whose latest answering ruling is a re-ask."""
    latest = _latest_answering(rulings)
    return tuple(
        ref for ref in recorded if ref not in latest or latest[ref].get('reask', False)
    )
