"""The ruling-overreach guard's class vocabulary and judge core.

Which writes the guard judges, the ruling-shape classifiers recorded beside
every judgement, and the label vocabulary, per
docs/ruling-overreach-guard-design-2026-10-01/design.md §4 and §8. Activation
is the structured category alone; the other four classifiers are recorded,
never acted on.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from types import MappingProxyType

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
