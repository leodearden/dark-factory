#!/usr/bin/env python3
"""The census similarity stage (plans/census-incremental-prd.md L7, D3, D4):
an embedding index over a codebook's open entries and pending candidates,
which screens a novel cluster or attaches a pending candidate that is near an
entry, and clusters the rest of the pending backlog into a ranked, capped
adjudication queue."""
from __future__ import annotations

import argparse
import copy
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

from legibility import census_findings, codebook, config, embeddings, entry_join, report_text

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
    clusters, which are not yet in the codebook; ``key`` is the record id
    for every other item."""

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
        sessions=tuple(dict.fromkeys(codebook.sessions_of(record))),
        first_seen=record.get("first_seen"),
    )


def items_of(cb: Mapping[str, Any], novel_clusters: Iterable[Mapping[str, Any]] = ()) -> tuple[Item, ...]:
    """Every entry that is not fixed or retired, every pending candidate, and
    every one of *novel_clusters*, in that order."""
    return (
        *(
            _item(Kind.ENTRY, entry, codebook.entry_field(cb, entry, "cause"), record_id=entry["id"])
            for entry in cb.get("entries") or []
            if not codebook.is_fixed(entry)
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


if sys.version_info >= (3, 12):
    _dot = math.sumprod
else:
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
        chain of pairs at or above *threshold* joins them. Every pair is
        compared, so the cost is quadratic in ``len(items)``."""
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


_SCORE_PLACES = 4


@dataclass(frozen=True)
class Attach:
    """*item* at cosine *score* to entry *entry_id*, at or above the attach
    threshold."""

    item: Item
    entry_id: str
    score: float


@dataclass(frozen=True)
class View:
    """What one run's similarity stage decided, built once before verify.
    With *skipped* set, nothing was decided and *requests* counts what the
    failed attempt spent."""

    settings: config.Adjudication
    skipped: str | None = None
    screened: tuple[Attach, ...] = ()
    attaches: tuple[Attach, ...] = ()
    queue: tuple[QueuedCluster, ...] = ()
    depth: tuple[int, int] | None = None
    requests: int = 0
    embedded: int = 0
    cached: int = 0
    pruned: int = 0

    @classmethod
    def skipped_by(cls, reason: str, requests: int, settings: config.Adjudication) -> View:
        return cls(settings=settings, skipped=reason, requests=requests)

    @property
    def model(self) -> str:
        return self.settings.embedding_model

    def screens(self, cluster: Mapping[str, Any]) -> bool:
        """Whether novel *cluster* is screened onto an entry, so no verify
        call is spent on it."""
        return any(attach.item.title == cluster["title"] for attach in self.screened)

    def select(self, limit: int) -> tuple[QueuedCluster, ...]:
        return self.queue[:max(limit, 0)]


def _attaches(index: Index, items: Iterable[Item], threshold: float) -> tuple[Attach, ...]:
    return tuple(
        Attach(item, nearest.entry.key, round(nearest.score, _SCORE_PLACES))
        for item in items
        if (nearest := index.nearest_entry(item)) is not None and nearest.score >= threshold
    )


def build_view(
    cb: Mapping[str, Any],
    novel_clusters: Sequence[Mapping[str, Any]],
    *,
    embedder: embeddings.Embedder,
    settings: config.Adjudication,
) -> View:
    """Embed *cb*'s open entries and pending candidates and this run's
    *novel_clusters*, then screen, attach and queue. Never raises for an
    embedding or cache failure: the whole stage is skipped instead."""
    items = items_of(cb, novel_clusters)
    try:
        embedded = embeddings.vectors_for(
            [embeddings.Text(item.key, item.record_id, item.text) for item in items],
            embedder=embedder,
            model=settings.embedding_model,
        )
    except (embeddings.EmbeddingFailed, embeddings.VectorCacheError) as exc:
        return View.skipped_by(str(exc), exc.requests_made, settings)
    index = Index.of(items, embedded.vectors)
    pending = [item for item in items if item.kind is Kind.PENDING]
    attaches = _attaches(index, pending, settings.attach_min_similarity)
    attached = {attach.item.key for attach in attaches}
    rest = [item for item in pending if item.key not in attached]
    queue = rank(index.clusters(rest, settings.cluster_min_similarity))
    return View(
        settings=settings,
        screened=_attaches(
            index, (item for item in items if item.kind is Kind.NOVEL), settings.attach_min_similarity,
        ),
        attaches=attaches,
        queue=queue,
        depth=(len(rest), len(queue)),
        requests=embedded.requests,
        embedded=embedded.embedded,
        cached=embedded.cached,
        pruned=embedded.pruned,
    )


# ---------------------------------------------------------------------------
# The codebook transforms: similarity attaches, and queue members following
# their representative
# ---------------------------------------------------------------------------

class Outcome(enum.StrEnum):
    VERIFIED = "verified"
    REJECTED = "rejected"
    UNRESOLVED = "unresolved"
    INCONCLUSIVE = "inconclusive"


def similarity_attached_by(score: float, run_id: str) -> dict[str, Any]:
    return {"method": "similarity", "score": score, "run_id": run_id}


@dataclass(frozen=True)
class AppliedAttach:
    """A pending candidate joined onto an entry by similarity; *novel* when
    it is this run's screened novel cluster."""

    candidate_id: str
    title: str
    entry_id: str
    score: float
    novel: bool


def attach(
    cb: dict, view: View, *, run_id: str,
) -> tuple[dict, tuple[AppliedAttach, ...], tuple[Attach, ...]]:
    """A new codebook in which every attach of *view* joins its entry: a
    backlog candidate by its id, a screened novel cluster by the pending
    candidate this run's merge gave its title. A screened cluster the merge
    gave no pending candidate is returned unattached rather than raised: it
    was neither verified nor attached, as census.py::DroppedVerdict records
    for a verdict with no pending candidate."""
    result = cb
    applied: list[AppliedAttach] = []
    unattached: list[Attach] = []
    for found, novel in (*((found, False) for found in view.attaches), *((found, True) for found in view.screened)):
        candidate_id = codebook.pending_titled(result, found.item.title) if novel else found.item.key
        if candidate_id is None:
            unattached.append(found)
            continue
        attached_by = similarity_attached_by(found.score, run_id)
        result = entry_join.join_entry(
            result, candidate_id, found.entry_id,
            candidate_fields={"attached_by": attached_by}, sighting_fields={"attached_by": attached_by},
        )
        applied.append(AppliedAttach(candidate_id, found.item.title, found.entry_id, found.score, novel))
    return result, tuple(applied), tuple(unattached)


@dataclass(frozen=True)
class Adjudicated:
    """How one offered queue cluster resolved, and the members that followed
    its representative."""

    representative: str
    outcome: Outcome
    members: tuple[str, ...]


def _candidate(cb: Mapping[str, Any], candidate_id: str) -> dict:
    for candidate in cb.get("candidates") or []:
        if candidate.get("id") == candidate_id:
            return candidate
    raise ValueError(f"similarity adjudication: no candidate {candidate_id!r}")


def _rejected_with(cb: dict, member_id: str, representative: Mapping[str, Any]) -> dict:
    result = copy.deepcopy(cb)
    member = _candidate(result, member_id)
    if member.get("disposition") != "pending":
        raise ValueError(
            f"similarity adjudication: member {member_id!r} is {member.get('disposition')!r}, not pending",
        )
    identity = census_findings.FindingIdentity.from_record(representative)
    if identity is not None:
        member.update(census_findings.stamped(member, identity))
    member.update(disposition="rejected", adjudicated_with=representative["id"])
    codebook.assert_no_deletion(cb, result)
    return result


def adjudicate_members(
    cb: dict, offered: Sequence[QueuedCluster],
) -> tuple[dict, tuple[Adjudicated, ...]]:
    """A new codebook in which each *offered* cluster's other members follow
    the disposition its representative now has: onto its entry when it was
    promoted, rejected with it when it was rejected, untouched while it is
    still pending."""
    result = cb
    outcomes: list[Adjudicated] = []
    for queued in offered:
        representative = _candidate(result, queued.representative.key)
        others = tuple(member.key for member in queued.others)
        if representative.get("disposition") == "promoted":
            for member_id in others:
                result = entry_join.join_entry(
                    result, member_id, representative["promoted_to"],
                    candidate_fields={"adjudicated_with": representative["id"]}, sighting_fields={},
                )
            outcomes.append(Adjudicated(representative["id"], Outcome.VERIFIED, others))
        elif representative.get("disposition") == "rejected":
            for member_id in others:
                result = _rejected_with(result, member_id, representative)
            outcomes.append(Adjudicated(representative["id"], Outcome.REJECTED, others))
        else:
            outcomes.append(Adjudicated(representative["id"], Outcome.UNRESOLVED, ()))
    return result, tuple(outcomes)


# ---------------------------------------------------------------------------
# The run's ## Adjudication section
# ---------------------------------------------------------------------------

_CHANGED = frozenset({Outcome.VERIFIED, Outcome.REJECTED})


@dataclass(frozen=True)
class AdjudicationReport:
    """What this run's similarity stage did: the View it built, the attaches
    applied, the queue clusters offered to verify and how each resolved, why
    the queue call was inconclusive when it was, and whether the verifier
    rejected every cluster offered, the signature of a systemic failure."""

    view: View
    attached: tuple[AppliedAttach, ...]
    unattached: tuple[Attach, ...]
    offered: tuple[QueuedCluster, ...]
    adjudicated: tuple[Adjudicated, ...]
    inconclusive: str | None
    mass_rejected: bool

    def _outcome(self, queued: QueuedCluster) -> Outcome:
        if self.inconclusive is not None:
            return Outcome.INCONCLUSIVE
        return next(
            (item.outcome for item in self.adjudicated if item.representative == queued.representative.key),
            Outcome.UNRESOLVED,
        )

    def counts(self) -> dict[str, int]:
        changed = [queued for queued in self.offered if self._outcome(queued) in _CHANGED]
        return {
            "attached": len(self.attached),
            "adjudicated": sum(len(queued.members) for queued in changed),
        }

    def to_record(self) -> dict[str, Any]:
        view = self.view
        return {
            "model": view.model,
            "thresholds": {
                "attach_min_similarity": view.settings.attach_min_similarity,
                "cluster_min_similarity": view.settings.cluster_min_similarity,
            },
            "max_clusters_per_run": view.settings.max_clusters_per_run,
            "skipped": view.skipped,
            "embedding": {
                "requests": view.requests, "embedded": view.embedded, "cached": view.cached, "pruned": view.pruned,
            },
            "queue_depth": None if view.depth is None else {"candidates": view.depth[0], "clusters": view.depth[1]},
            "processed": [
                {
                    "representative": queued.representative.key,
                    "title": queued.representative.title,
                    "members": [member.key for member in queued.members],
                    "sessions": queued.sessions,
                    "outcome": self._outcome(queued).value,
                }
                for queued in self.offered
            ],
            "attached": [
                {
                    "candidate_id": item.candidate_id,
                    "title": item.title,
                    "entry_id": item.entry_id,
                    "score": float(item.score),
                    "novel": item.novel,
                }
                for item in self.attached
            ],
            "unattached": [
                {"title": item.item.title, "entry_id": item.entry_id, "score": float(item.score)}
                for item in self.unattached
            ],
            "inconclusive": self.inconclusive,
            "mass_rejection": {"offered": len(self.offered)} if self.mass_rejected else None,
        }


def _attached_line(item: Mapping[str, Any]) -> str:
    novel = ", this run's novel cluster" if item["novel"] else ""
    return (
        f"- attached by similarity {item['score']:.4f}: {report_text.quoted_title(item['title'])} "
        f"({item['candidate_id']}{novel}) → {item['entry_id']}"
    )


def _unattached_line(item: Mapping[str, Any]) -> str:
    return (
        f"- not attached, no pending candidate after the merge: {report_text.quoted_title(item['title'])} "
        f"(similarity {item['score']:.4f} → {item['entry_id']}); not verified this run"
    )


def _mass_rejection_line(mass_rejection: Mapping[str, Any]) -> str:
    return (
        f"- **ALL {mass_rejection['offered']} queue cluster(s) were REJECTED and none survived**, "
        "and every member was rejected with its representative. Suspect a SYSTEMIC verifier failure "
        "(model unreachable, tool access denied, or unparseable verdicts) rather than unfounded claims; "
        "check the run's per-cluster 'verify failed' warnings."
    )


def _processed_line(item: Mapping[str, Any]) -> str:
    return (
        f"- {item['outcome']}: {report_text.quoted_title(item['title'])} ({item['representative']}) for "
        f"{len(item['members'])} candidate(s) across {item['sessions']} session(s)"
    )


def report_lines(record: Mapping[str, Any]) -> list[str]:
    """The report's ``## Adjudication`` section, rendered from the run
    record's ``adjudication`` value (:meth:`AdjudicationReport.to_record`)."""
    thresholds, embedding = record["thresholds"], record["embedding"]
    header = [
        "",
        "## Adjudication",
        "",
        f"- model {record['model']}; attach at cosine >= {thresholds['attach_min_similarity']}, "
        f"cluster at >= {thresholds['cluster_min_similarity']}; at most "
        f"{record['max_clusters_per_run']} queue cluster(s) per run",
        f"- embeddings: {embedding['requests']} request(s), {embedding['embedded']} embedded, "
        f"{embedding['cached']} cached, {embedding['pruned']} pruned",
    ]
    if record["skipped"] is not None:
        return [
            *header,
            f"- skipped: {record['skipped']}; no similarity attach and no queue adjudication happened this run",
        ]
    depth, processed = record["queue_depth"], record["processed"]
    return [
        *header,
        *([_mass_rejection_line(record["mass_rejection"])] if record["mass_rejection"] else []),
        f"- queue depth: {depth['candidates']} pending candidate(s) in {depth['clusters']} cluster(s)",
        f"- processed this run: {len(processed)} cluster(s), "
        f"{sum(len(item['members']) for item in processed)} candidate(s)",
        *([f"- queue inconclusive: {record['inconclusive']}"] if record["inconclusive"] else []),
        *(_processed_line(item) for item in processed),
        *(_attached_line(item) for item in record["attached"]),
        *(_unattached_line(item) for item in record["unattached"]),
    ]


# ---------------------------------------------------------------------------
# The queue CLI: the stage's signal by hand, spending embeddings only
# ---------------------------------------------------------------------------

def _cluster_line(queued: QueuedCluster) -> str:
    return (
        f"- {queued.representative.key}: {report_text.quoted_title(queued.representative.title)}, "
        f"{len(queued.members)} member(s) across {queued.sessions} session(s)"
    )


def queue_lines(view: View, top: int) -> list[str]:
    """What the queue CLI prints for *view*: its top *top* clusters."""
    if view.skipped is not None:
        return [f"similarity stage skipped: {view.skipped}"]
    settings = view.settings
    candidates, clusters = view.depth or (0, 0)
    listed = view.select(top)
    return [
        f"model {view.model}; attach at cosine >= {settings.attach_min_similarity}, "
        f"cluster at >= {settings.cluster_min_similarity}",
        f"embeddings: {view.requests} request(s), {view.embedded} embedded, {view.cached} cached, "
        f"{view.pruned} pruned",
        f"attachable now: {len(view.attaches)} pending candidate(s) near an open entry",
        f"queue depth: {candidates} pending candidate(s) in {clusters} cluster(s)",
        f"top {len(listed)} cluster(s):",
        *(_cluster_line(queued) for queued in listed),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="similarity.py",
        description="The census similarity stage over a project's codebook, read-only.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    queue = commands.add_parser(
        "queue", help="print the adjudication queue's depth, the attachable count and the top clusters",
    )
    queue.add_argument("--project-root", required=True, type=Path)
    queue.add_argument(
        "--config", type=Path, default=None, help="default: <project-root>/docs/legibility/legibility.yaml",
    )
    queue.add_argument("--top", type=int, default=10, help="how many clusters to list (default 10)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Embed the codebook's open entries and pending candidates and print the
    queue; 1 when the stage is skipped. Writes only the vector cache."""
    args = build_parser().parse_args(argv)
    root = args.project_root
    cfg = config.load_config(args.config or root / "docs" / "legibility" / "legibility.yaml")
    cb = codebook.load(root / "docs" / "legibility" / "confusion-codebook.yaml")
    view = build_view(
        cb,
        (),
        embedder=embeddings.Embedder(embeddings.openai_embed, embeddings.cache_path(cfg.project_id)),
        settings=cfg.census.adjudication,
    )
    print("\n".join(queue_lines(view, args.top)))
    return 1 if view.skipped is not None else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
