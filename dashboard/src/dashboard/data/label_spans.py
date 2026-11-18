"""Encode a label row drawn from a union row as index spans over that union,
and decode it back.

On the wire a span is a half-open ``[start, stop)`` pair of ints indexing the
union, and a row is the list of its spans in the row's own order: expanding
each span over the union and concatenating them rebuilds the row exactly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeGuard


def label_spans(labels: Sequence[str], union: Sequence[str]) -> list[list[int]]:
    index_of = {label: index for index, label in enumerate(union)}
    spans: list[list[int]] = []
    for label in labels:
        index = _union_index(index_of, label)
        if spans and spans[-1][1] == index:
            spans[-1][1] = index + 1
        else:
            spans.append([index, index + 1])
    return spans


def labels_from_spans(spans: Sequence[Sequence[int]], union: Sequence[str]) -> list[str]:
    labels: list[str] = []
    for span in spans:
        start, stop = _checked_span(span, len(union))
        labels.extend(union[start:stop])
    return labels


def _union_index(index_of: dict[str, int], label: str) -> int:
    if label not in index_of:
        raise ValueError(f'label {label!r} is not in the union row')
    return index_of[label]


def _checked_span(span: object, union_length: int) -> tuple[int, int]:
    if not _is_int_pair(span):
        raise ValueError(
            f'label span {span!r} is not a [start, stop) pair of ints '
            f'(union has {union_length} labels)')
    start, stop = span[0], span[1]
    if not 0 <= start < stop <= union_length:
        raise ValueError(
            f'label span {span!r} is outside 0 <= start < stop <= '
            f'union length {union_length}')
    return start, stop


def _is_int_pair(span: object) -> TypeGuard[Sequence[int]]:
    return (
        isinstance(span, (list, tuple))
        and len(span) == 2
        and all(type(bound) is int for bound in span)
    )
