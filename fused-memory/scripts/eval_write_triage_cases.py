#!/usr/bin/env python3
"""What one run of the judge eval asks, bundled into the frozen plan the runner consumes.

One case per labelled record: what is submitted, the slate it is shown and
the outcomes that count as correct for it, collected with the records each
slate names in an :class:`EvalPlan`. Two constructions: SEEDED, built here
from the curator fixture alone, and RETRIEVED, which pairs
``eval_write_triage_retrieval.py``'s ``Slate`` values with the labels. It
imports only ``eval_write_triage_scoring.py``.
"""
from __future__ import annotations

import importlib.util
import logging
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fused_memory.server.write_triage import OUTCOME_JUDGE

logger = logging.getLogger(__name__)

_SCRIPTS = Path(__file__).resolve().parent


def _load_script(path: Path, mod_name: str) -> types.ModuleType:
    """Load a ``scripts/`` sibling by path, cached in ``sys.modules``."""
    import sys  # noqa: PLC0415

    cached = sys.modules.get(mod_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


# Pyright cannot follow the by-path loader, so it alone resolves `_scoring` through
# the scripts/ extraPath; at runtime the module is always the by-path load.
if TYPE_CHECKING:
    import eval_write_triage_scoring as _scoring
else:
    _scoring = _load_script(
        _SCRIPTS / 'eval_write_triage_scoring.py', 'eval_write_triage_scoring',
    )


_NO_ALIASES: Mapping[str, str] = types.MappingProxyType({})


# ---------------------------------------------------------------------------
# Case construction
# ---------------------------------------------------------------------------

def _distractor_pool(records: Sequence[Mapping[str, Any]], cluster_id: Any) -> list[str]:
    """Every record outside *cluster_id*, ordered by memory_id.

    Sorted rather than shuffled: this script's output is a COMMITTED artifact,
    and a seeded shuffle would make it reproducible only by someone who also
    knew the seed, an unseeded one by nobody at all.
    """
    return sorted(
        str(r['memory_id']) for r in records if r['cluster_id'] != cluster_id
    )


def _rotated(pool: list[str], offset: int, count: int) -> list[str]:
    """*count* entries from *pool*, starting at *offset*, wrapping around.

    The rotation is what stops all N cases being shown the same slate. Taking
    a plain ``pool[:count]`` would be equally deterministic and would measure
    one arbitrary handful of clusters over and over instead of the corpus.
    """
    if count <= 0:
        return []
    if not pool or len(pool) < count:
        # LOUD rather than silent: a short pool narrows every slate, and the
        # report's own width provenance is measured downstream rather than
        # asserted, so the two together are what stop a 1-wide smoke run being
        # read as the 5-wide measurement the operator is deciding on.
        logger.warning(
            'distractor pool holds %d record(s), %d requested — slate narrowed. '
            'Cross-cluster distractors come only from OTHER clusters, so a '
            'single-cluster run (typically --limit slicing one cluster) leaves '
            'nothing to draw from.',
            len(pool), count,
        )
    take = min(count, len(pool))
    if take <= 0:
        return []
    start = offset % len(pool)
    return [pool[(start + i) % len(pool)] for i in range(take)]


def _case(
    record: Mapping[str, Any],
    *,
    candidates: list[str],
    expected_class: str,
    acceptable: frozenset[str],
    band_winner_id: str | None,
    band: str,
    similarity: float | None,
    canonical_present: bool | None,
    canonical_alias_id: str | None,
) -> dict[str, Any]:
    """One judge case: what is submitted, what it is shown, how it routed.

    Every routing field is explicit because the two slate modes disagree about
    all of them, and a defaulted one would let a retrieved case silently
    publish the seeded construction's answer.
    """
    return {
        'memory_id': str(record['memory_id']),
        'content': record['content'],
        'category': record.get('category'),
        'cluster_id': str(record['cluster_id']),
        'candidates': candidates,
        'expected_class': expected_class,
        'acceptable_outcomes': acceptable,
        'band_winner_id': band_winner_id,
        'band': band,
        'similarity': similarity,
        'canonical_present': canonical_present,
        'canonical_alias_id': canonical_alias_id,
    }


def _seeded_case(
    record: Mapping[str, Any],
    slate: list[str],
    expected_class: str,
    acceptable: frozenset[str],
    canonical_alias_id: str | None,
) -> dict[str, Any]:
    """A case whose slate is CONSTRUCTED rather than retrieved.

    The lead record is the band winner and the band is the middle one — the
    only band that reaches a judge — because that is what
    `eval_write_triage_live_judge.py::build_judge_fn` synthesizes. Similarity
    and canonical liveness are unmeasured here, and `None` says so rather than
    claiming a figure.
    """
    return _case(
        record,
        candidates=slate,
        expected_class=expected_class,
        acceptable=acceptable,
        band_winner_id=slate[0] if slate else None,
        band=OUTCOME_JUDGE,
        similarity=None,
        canonical_present=None,
        canonical_alias_id=canonical_alias_id,
    )


def build_judge_cases(
    records: Sequence[Mapping[str, Any]],
    *,
    distractors: int,
    aliases: Mapping[str, str] = _NO_ALIASES,
) -> list[dict[str, Any]]:
    """One judge call per case, with the ground truth its label implies.

    Two kinds of case come out of here.

    LABELLED — one per NON-canonical record. Its slate is the record's own
    cluster canonical plus *distractors* cross-cluster records, so the attach
    target is always present: a judge shown a slate without it is answering
    about a different memory than the one an attach would touch. Canonical
    records produce no labelled case, because they ARE the target.

    CONTROL (``eval_write_triage_scoring.py::CLASS_DISTRACTOR``) — one per
    CLUSTER, capped there on cost, since every case is a paid LLM call and one
    control per cluster already answers the question the control asks. Its
    slate is ``distractors + 1``
    records drawn only from OTHER clusters, so it is exactly as wide as a
    labelled slate and nothing on it is a correct attach.

    Ordering is deterministic and independent of the input order: records are
    sorted by ``memory_id`` before anything is drawn, so the committed report
    is reproducible from the fixture alone.

    An unrecognised label raises
    ``eval_write_triage_scoring.py::UnknownLabelError``.
    """
    ordered = sorted(records, key=lambda r: str(r['memory_id']))
    cases: list[dict[str, Any]] = []

    for index, record in enumerate(ordered):
        label = str(record['label'])
        if label == _scoring.LABEL_CANONICAL:
            continue
        acceptable = _scoring.acceptable_outcomes_for(label)
        canonical_id = str(record['cluster_id'])
        pool = _distractor_pool(ordered, record['cluster_id'])
        slate = [canonical_id, *_rotated(pool, index, distractors)]
        cases.append(_seeded_case(
            record, slate, label, acceptable, aliases.get(canonical_id),
        ))

    # The control set, appended after the labelled ones so a `--limit` smoke
    # run covers the classes the labels actually measure first.
    seen_clusters: set[Any] = set()
    for index, record in enumerate(ordered):
        if str(record['label']) == _scoring.LABEL_CANONICAL:
            continue
        if record['cluster_id'] in seen_clusters:
            continue
        seen_clusters.add(record['cluster_id'])
        pool = _distractor_pool(ordered, record['cluster_id'])
        slate = _rotated(pool, index, distractors + 1)
        cases.append(_seeded_case(
            record, slate, _scoring.CLASS_DISTRACTOR, _scoring.DISTRACTOR_ACCEPTABLE,
            aliases.get(str(record['cluster_id'])),
        ))

    return cases


# ---------------------------------------------------------------------------
# The plan: what will be judged, and what its slates name
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvalPlan:
    """The cases of one run, the records their slates name, and how they were built.

    ``candidate_records`` is positional with ``cases``: entry *i* is case *i*'s
    slate resolved to records, in slate order, because a judge handed a bare id
    has no text to compare and is answering noise. Rows are never pooled across
    cases: a retrieved row carries ITS query's cosine, which the shipped
    selector re-sorts every slate by, so a borrowed row would hand one case
    another query's order.

    ``fixture_by_id`` is the curator fixture by memory id: the ground truth an
    attach target's cluster and label are read from, wherever it was retrieved.

    ``provenance`` is the plan's OWN disclosure — the fields that describe how
    the slates were obtained. It is merged UNDER the caller's, so nothing here
    can overwrite what the operator asked for.

    Checked at construction, before anything is spent or written: each case's
    records are exactly its slate, and a case routed to the judge shows it at
    least one candidate.
    """

    cases: tuple[dict[str, Any], ...]
    candidate_records: tuple[tuple[Mapping[str, Any], ...], ...]
    fixture_by_id: Mapping[str, Mapping[str, Any]]
    record_count: int
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        shown = [[str(r['memory_id']) for r in records] for records in self.candidate_records]
        unresolved = [
            f"{case['memory_id']} ({case['expected_class']})"
            for index, case in enumerate(self.cases)
            if index >= len(shown) or shown[index] != case['candidates']
        ]
        if unresolved or len(shown) != len(self.cases):
            raise ValueError(
                f'{len(self.cases)} case(s), {len(shown)} resolved slate(s); not shown '
                f'exactly their own slate, in order: {", ".join(unresolved) or "none"}',
            )
        unseen = [
            f"{case['memory_id']} ({case['expected_class']})"
            for case in self.cases
            if case['band'] == OUTCOME_JUDGE and not case['candidates']
        ]
        if unseen:
            raise ValueError(
                f'{len(unseen)} judge-band case(s) have an EMPTY slate: '
                f'{", ".join(unseen)}. judge_write answers those `stored` with no '
                'provider call, so the eval would count a verdict nobody gave. A '
                'single-cluster run leaves the distractor control nothing to draw '
                'from: widen --limit or the fixture.',
            )


def seeded_plan(
    records: Sequence[Mapping[str, Any]],
    *,
    distractors: int,
    aliases: Mapping[str, str] = _NO_ALIASES,
) -> EvalPlan:
    """Today's construction: the cluster canonical seeded at slate position 0.

    Each slate is resolved against the fixture HERE, so a dangling id raises
    ``KeyError`` before anything is spent.
    """
    cases = tuple(build_judge_cases(records, distractors=distractors, aliases=aliases))
    fixture = {str(r['memory_id']): r for r in records}
    return EvalPlan(
        cases=cases,
        candidate_records=tuple(
            tuple(fixture[cid] for cid in case['candidates']) for case in cases
        ),
        fixture_by_id=fixture,
        record_count=len(records),
        provenance={
            'slate_mode': _scoring.SLATE_SEEDED,
            'distractor_count_requested': distractors,
        },
    )


def plan_from_slates(
    records: Sequence[Mapping[str, Any]],
    slates: Sequence[Any],
    *,
    provenance: Mapping[str, Any],
    aliases: Mapping[str, str] = _NO_ALIASES,
    fixture: Sequence[Mapping[str, Any]] | None = None,
) -> EvalPlan:
    """Pair production-shaped slates with the curator labels they answer for.

    One case per NON-canonical record, positional against *slates*. No control
    class is constructed: a retrieved slate that carries no correct attach
    target is the ordinary case here rather than something to build, and
    ``distractor_count`` is ``None``.

    Attach targets are described from *fixture*, the whole curator corpus
    (default: *records*), so a ``--limit`` run describes a target exactly as
    the full run does.

    The expected class stays the fixture's label. The slate, the band winner
    and the band come from the retrieval — including for a record whose
    canonical is no longer in the corpus, which is kept in the population and
    flagged rather than dropped.
    """
    labelled = [r for r in records if str(r['label']) != _scoring.LABEL_CANONICAL]
    if len(labelled) != len(slates):
        raise ValueError(
            f'{len(labelled)} labelled record(s) but {len(slates)} slate(s): a '
            f'positional mismatch would score a case against another record\'s '
            f'retrieval',
        )
    cases: list[dict[str, Any]] = []
    for record, slate in zip(labelled, slates, strict=True):
        cases.append(_case(
            record,
            candidates=[str(c['memory_id']) for c in slate.candidates],
            expected_class=str(record['label']),
            acceptable=_scoring.acceptable_outcomes_for(str(record['label'])),
            band_winner_id=slate.attach_target_id,
            band=slate.band,
            similarity=slate.similarity,
            canonical_present=slate.canonical_present,
            canonical_alias_id=aliases.get(str(record['cluster_id'])),
        ))
    return EvalPlan(
        cases=tuple(cases),
        candidate_records=tuple(tuple(slate.candidates) for slate in slates),
        fixture_by_id={str(r['memory_id']): r for r in (records if fixture is None else fixture)},
        record_count=len(records),
        provenance={'slate_mode': _scoring.SLATE_RETRIEVED, 'distractor_count': None, **dict(provenance)},
    )
