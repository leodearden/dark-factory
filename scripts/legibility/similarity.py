#!/usr/bin/env python3
"""The census similarity stage (plans/census-incremental-prd.md L7, D3, D4):
an embedding index over a codebook's open entries and pending candidates,
which screens a novel cluster or attaches a pending candidate that is near an
entry, and clusters the rest of the pending backlog into a ranked, capped
adjudication queue."""
from __future__ import annotations

import enum
import math
import operator
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from legibility import codebook, prescreen

Vector = tuple[float, ...]

_ID_SUFFIX = re.compile(r"-(\d+)$")


def record_text(title: str, cause: str | None) -> str:
    """What a record is embedded as: its title, and its cause on one line."""
    collapsed = " ".join((cause or "").split())
    return f"{title} — {collapsed}" if collapsed else title


class Kind(enum.StrEnum):
    ENTRY = "entry"
    PENDING = "pending"
    NOVEL = "novel"


@dataclass(frozen=True)
class Item:
    """One indexed record. ``record_id`` is None for this run's novel
    clusters, which are not yet in the codebook."""

    kind: Kind
    record_id: str | None
    title: str
    text: str
    sessions: tuple[str, ...]
    first_seen: str | None

    @property
    def key(self) -> str:
        return self.record_id if self.record_id is not None else f"{self.kind}:{self.title}"


def _item(kind: Kind, record: Mapping[str, Any], cause: str | None, *, record_id: str | None) -> Item:
    return Item(
        kind=kind,
        record_id=record_id,
        title=record["title"],
        text=record_text(record["title"], cause),
        sessions=tuple(dict.fromkeys(prescreen.sessions_of(record))),
        first_seen=record.get("first_seen"),
    )


def items_of(cb: Mapping[str, Any], novel_clusters: Iterable[Mapping[str, Any]] = ()) -> tuple[Item, ...]:
    """Every entry that is not fixed or retired, every pending candidate, and
    every one of *novel_clusters*, in that order."""
    return (
        *(
            _item(Kind.ENTRY, entry, codebook.entry_field(cb, entry, "cause"), record_id=entry["id"])
            for entry in cb.get("entries") or []
            if not prescreen.is_fixed(entry)
        ),
        *(
            _item(Kind.PENDING, candidate, candidate.get("cause"), record_id=candidate["id"])
            for candidate in cb.get("candidates") or []
            if candidate.get("disposition") == "pending"
        ),
        *(_item(Kind.NOVEL, cluster, cluster.get("cause"), record_id=None) for cluster in novel_clusters),
    )


def _unit(vector: Sequence[float]) -> Vector:
    norm = math.sqrt(math.fsum(x * x for x in vector))
    return tuple(x / norm for x in vector)


def _dot(a: Vector, b: Vector) -> float:
    return sum(map(operator.mul, a, b))


@dataclass(frozen=True)
class Nearest:
    entry: Item
    score: float


@dataclass(frozen=True)
class Index:
    """Unit vectors by ``Item.key``, so a dot product is a cosine."""

    entries: tuple[Item, ...]
    unit: Mapping[str, Vector]

    @classmethod
    def of(cls, items: Sequence[Item], vectors: Mapping[str, Sequence[float]]) -> Index:
        return cls(
            entries=tuple(item for item in items if item.kind is Kind.ENTRY),
            unit=MappingProxyType({item.key: _unit(vectors[item.key]) for item in items}),
        )

    def similarity(self, a: Item, b: Item) -> float:
        return _dot(self.unit[a.key], self.unit[b.key])

    def nearest_entry(self, item: Item) -> Nearest | None:
        nearest = None
        for entry in self.entries:
            score = self.similarity(item, entry)
            if nearest is None or score > nearest.score:
                nearest = Nearest(entry, score)
        return nearest

    def clusters(self, items: Sequence[Item], threshold: float) -> tuple[tuple[Item, ...], ...]:
        """Single-link clusters of *items*: two items share a cluster when a
        chain of pairs at or above *threshold* joins them."""
        parent = list(range(len(items)))

        def root(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if self.similarity(items[i], items[j]) >= threshold:
                    low, high = sorted((root(i), root(j)))
                    parent[high] = low
        groups: dict[int, list[Item]] = {}
        for i, item in enumerate(items):
            groups.setdefault(root(i), []).append(item)
        return tuple(tuple(group) for group in groups.values())


def _newness(item: Item) -> tuple[str, int]:
    suffix = _ID_SUFFIX.search(item.record_id or "")
    return item.first_seen or "", int(suffix.group(1)) if suffix else -1


@dataclass(frozen=True)
class QueuedCluster:
    """Pending candidates adjudicated together: the *representative*, the
    newest member, is verified for all of them."""

    representative: Item
    members: tuple[Item, ...]
    sessions: int
    newest: str | None

    @property
    def others(self) -> tuple[Item, ...]:
        return tuple(member for member in self.members if member != self.representative)


def _queued(members: Sequence[Item]) -> QueuedCluster:
    representative = max(members, key=_newness)
    return QueuedCluster(
        representative=representative,
        members=tuple(members),
        sessions=len({session for member in members for session in member.sessions}),
        newest=representative.first_seen,
    )


def rank(clusters: Iterable[Sequence[Item]]) -> tuple[QueuedCluster, ...]:
    """The queue order: most distinct sessions first, then newest, then the
    representative's id."""
    by_id = sorted((_queued(members) for members in clusters), key=lambda q: q.representative.record_id or "")
    return tuple(sorted(by_id, key=lambda q: (q.sessions, _newness(q.representative)), reverse=True))
