#!/usr/bin/env python3
"""The rated soak corpus as replay cases, routed the way triage is deployed today (task 7519).

A *case* is one rated soak write: the (write, parent) text pair its raters
saw, and the facts deployed routing reads. :func:`build_cases` joins the
committed soak items to the gitignored texts, cosines and recovered caller
metadata. Each case's :class:`Route` follows the order of
``fused_memory/server/write_triage.py::triage_write``: supersession and the
caller force-stores come before retrieval, a near-verbatim write is restated
without a judge call, and everything else is judged. :func:`routed_outcome`
then applies the attach guard to the outcome, and :func:`item_row` turns it
into a ``score_write_triage_reader.py`` item row.

A case's :class:`CaseFeatures` carry no text, so they can be committed. A
committed arm's outcomes can then be re-scored from git alone.

Usage
-----
Run from ``fused-memory/``; the texts and soak bundle exist only in the main
checkout's gitignored ``data/``::

    uv run python scripts/write_triage_m3_replay_cases.py build \\
        --items calibration/write_triage_v2_soak_items.jsonl \\
        --texts ../../../data/write-triage-v21-rebase-2026-10-21/texts.json \\
        --key ../../../data/write-triage-v21-rebase-2026-10-21/key.json \\
        --hoist ../../../data/write-triage-soak-2026-10-19/correctness/hoist.json \\
        --adjudications ../../../data/write-triage-soak-2026-10-19/correctness/adjudications.jsonl \\
        --attaches ../../../data/write-triage-soak-2026-10-19/correctness/attaches_soak.json \\
        --cases-out ../data/write-triage-m3-levers/cases.jsonl \\
        --features-out calibration/write_triage_m3_replay_cases.jsonl
"""
from __future__ import annotations

import argparse
import enum
import json
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fused_memory.server.near_verbatim import is_near_verbatim
from fused_memory.server.write_triage import (
    ATTACH_OWNED_KEYS,
    OUTCOME_JUDGE,
    OUTCOME_RESTATED,
    OUTCOME_STORED,
    TRIAGE_OUTCOMES,
    AttachGuard,
    declares_attach_keys,
    declares_supersession,
    is_orchestrator_completion_write,
)

CURRENT_BASIS = 'current'


class Route(enum.StrEnum):
    """Where deployed triage sends a write before any judge verdict."""

    SUPERSESSION = 'supersession'
    FORCE_STORED = 'force_stored'
    NEAR_VERBATIM = 'near_verbatim'
    JUDGE = 'judge'


_ROUTE_BANDS: Mapping[Route, str] = {
    Route.SUPERSESSION: OUTCOME_STORED,
    Route.FORCE_STORED: OUTCOME_STORED,
    Route.NEAR_VERBATIM: OUTCOME_RESTATED,
    Route.JUDGE: OUTCOME_JUDGE,
}


@dataclass(frozen=True)
class CaseFeatures:
    """Everything about a case except its texts.

    ``cos_parent`` and ``source_cosine`` are None when the soak could not
    measure them. ``source_cosine`` is the matched record's cosine: the
    earlier sibling's on a via-sibling attach, the parent's otherwise.
    """

    entry_id: str
    target_id: str
    stratum: str
    project: str
    soak_outcome: str
    soak_band: str | None
    via_sibling: bool | None
    cross_category: bool | None
    parent_edited_after_attach: bool | None
    supersedes_target: bool
    route: Route
    cos_parent: float | None
    source_cosine: float | None
    metadata_recovered: bool
    child_chars: int
    parent_chars: int

    @property
    def cosine_recovered(self) -> bool:
        return self.cos_parent is not None

    def to_row(self) -> dict[str, Any]:
        return {**asdict(self), 'route': str(self.route), 'cosine_recovered': self.cosine_recovered}

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> CaseFeatures:
        fields = {key: value for key, value in row.items() if key != 'cosine_recovered'}
        return cls(**{**fields, 'route': Route(row['route'])})


@dataclass(frozen=True)
class ReplayCase:
    """One rated write: its features and the exact text pair its raters saw."""

    features: CaseFeatures
    child_text: str
    parent_text: str

    def to_row(self) -> dict[str, Any]:
        return {**self.features.to_row(), 'child_text': self.child_text, 'parent_text': self.parent_text}

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> ReplayCase:
        features = {key: value for key, value in row.items() if key not in ('child_text', 'parent_text')}
        return cls(CaseFeatures.from_row(features), row['child_text'], row['parent_text'])


def caller_metadata(stored_meta: Mapping[str, Any]) -> dict[str, Any]:
    """The metadata an attached write's caller sent, from the record triage stored.

    Exact for a write triage attached while ``declares_attach_keys`` was
    deployed (since 2026-08-25, before the soak): a caller who set any
    attach-owned key was force-stored, so on an attached record each of
    those keys is the attach's own.
    """
    return {key: value for key, value in stored_meta.items() if key not in ATTACH_OWNED_KEYS}


def build_cases(
    items: Iterable[Mapping[str, Any]],
    texts: Iterable[Mapping[str, Any]],
    cosines: Mapping[str, Mapping[str, Any] | None],
    metadata: Mapping[str, Mapping[str, Any]],
) -> tuple[ReplayCase, ...]:
    """Join each soak item to its current text pair, cosines and caller metadata.

    *cosines* maps an entry id to its ``hoist.json`` row, or to None when the
    soak could not measure it. *metadata* maps an entry id to its caller
    metadata, and an entry it lacks is routed as if nothing force-stores it.
    An item without a current text row, or absent from *cosines*, is refused.
    """
    item_rows = list(items)
    current = {
        (row['entry_id'], row['target_id']): row for row in texts if row['basis'] == CURRENT_BASIS
    }
    untexted = [item for item in item_rows if (item['entry_id'], item['target_id']) not in current]
    uncosined = [item for item in item_rows if item['entry_id'] not in cosines]
    if untexted or uncosined:
        raise ValueError(
            f'items without a current text row: {_pairs(untexted)};'
            f' items absent from the cosines: {_pairs(uncosined)}'
        )
    cases = [
        _case(
            item, current[(item['entry_id'], item['target_id'])],
            cosines[item['entry_id']], metadata.get(item['entry_id']),
        )
        for item in item_rows
    ]
    return tuple(sorted(cases, key=lambda case: case.features.entry_id))


def _pairs(items: Sequence[Mapping[str, Any]]) -> list[str]:
    return [f'{item["entry_id"]}/{item["target_id"]}' for item in items]


def _case(
    item: Mapping[str, Any],
    text: Mapping[str, Any],
    cosine: Mapping[str, Any] | None,
    metadata: Mapping[str, Any] | None,
) -> ReplayCase:
    child, parent = text['child_text'], text['parent_text']
    cos_parent = None if cosine is None else cosine['cos_parent']
    source = cosine['max_cos_earlier_sibling'] if cosine and item['via_sibling'] else cos_parent
    features = CaseFeatures(
        entry_id=item['entry_id'],
        target_id=item['target_id'],
        stratum=item['stratum'],
        project=item['project'],
        soak_outcome=item['outcome'],
        soak_band=item['band'],
        via_sibling=item['via_sibling'],
        cross_category=item['cross_category'],
        parent_edited_after_attach=item['parent_edited_after_attach'],
        supersedes_target=item['supersedes_target'],
        route=_route(item['supersedes_target'], metadata, child, parent),
        cos_parent=cos_parent,
        source_cosine=source,
        metadata_recovered=metadata is not None,
        child_chars=len(child),
        parent_chars=len(parent),
    )
    return ReplayCase(features, child, parent)


def _route(
    supersedes_target: bool, metadata: Mapping[str, Any] | None, child: str, parent: str,
) -> Route:
    if supersedes_target:
        return Route.SUPERSESSION
    caller = dict(metadata or {})
    if (
        declares_attach_keys(caller)
        or is_orchestrator_completion_write(caller)
        or declares_supersession(caller)
    ):
        return Route.FORCE_STORED
    if is_near_verbatim(child, parent):
        return Route.NEAR_VERBATIM
    return Route.JUDGE


def routed_outcome(features: CaseFeatures, verdict_outcome: str | None, guard: AttachGuard) -> str:
    """The outcome deployed triage gives *features* when the judge returns *verdict_outcome*.

    *verdict_outcome* is read only on the judge route. An attach the guard
    refuses is stored. A case whose cosine the soak could not measure keeps
    its attach, since no refusal can be shown for it.
    """
    if features.route in (Route.SUPERSESSION, Route.FORCE_STORED):
        return OUTCOME_STORED
    if features.route is Route.NEAR_VERBATIM:
        outcome = OUTCOME_RESTATED
    elif verdict_outcome not in TRIAGE_OUTCOMES:
        raise ValueError(
            f'case {features.entry_id}: verdict outcome {verdict_outcome!r} is not one of'
            f' {sorted(TRIAGE_OUTCOMES)}'
        )
    else:
        outcome = verdict_outcome
    if outcome == OUTCOME_STORED or not features.cosine_recovered:
        return outcome
    refusals = guard.refusals(
        source_cosine=features.source_cosine,
        target_cosine=features.cos_parent,
        hoisted=bool(features.via_sibling),
        judged=features.route is Route.JUDGE,
    )
    return OUTCOME_STORED if refusals else outcome


def item_row(features: CaseFeatures, outcome: str) -> dict[str, Any]:
    """The ``score_write_triage_reader.py`` item row for *features* given *outcome*.

    A stored write lists its rated target as the earlier record it might
    have joined; only a supersession marks it.
    """
    stored = outcome == OUTCOME_STORED
    marked = stored and features.route is Route.SUPERSESSION
    return {
        'entry_id': features.entry_id,
        'outcome': outcome,
        'target_id': None if stored else features.target_id,
        'earlier_ids': [features.target_id] if stored else [],
        'superseded_marks': [features.target_id] if marked else [],
        'supersedes_target': False if stored else features.supersedes_target,
        'band': _ROUTE_BANDS[features.route],
        'project': features.project,
        'stratum': features.stratum,
        'via_sibling': features.via_sibling,
        'cross_category': features.cross_category,
        'parent_edited_after_attach': features.parent_edited_after_attach,
    }


def read_cases(path: Path) -> tuple[ReplayCase, ...]:
    return tuple(ReplayCase.from_row(row) for row in _read_jsonl(path))


def read_features(path: Path) -> tuple[CaseFeatures, ...]:
    return tuple(CaseFeatures.from_row(row) for row in _read_jsonl(path))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))


def _intent_ids(adjudications: Path, key: Path) -> dict[str, str]:
    """Each rated child memory id's journal intent id, from the soak and extension rounds."""
    by_memory = {row['child_memory_id']: row['child_intent_id'] for row in _read_jsonl(adjudications)}
    extension = json.loads(key.read_text())['extension']
    by_memory.update({row['child_memory_id']: intent for intent, row in extension.items()})
    return by_memory


def _build(args: argparse.Namespace) -> int:
    items = _read_jsonl(args.items)
    intents = _intent_ids(args.adjudications, args.key)
    hoist = json.loads(args.hoist.read_text())
    attaches = {row['intent_id']: row for row in json.loads(args.attaches.read_text())}
    unmapped = sorted(item['entry_id'] for item in items if item['entry_id'] not in intents)
    if unmapped:
        raise SystemExit(f'items with no intent id: {unmapped}')
    cosines = {item['entry_id']: hoist.get(intents[item['entry_id']]) for item in items}
    metadata = {
        item['entry_id']: caller_metadata(attaches[intents[item['entry_id']]]['meta'])
        for item in items if intents[item['entry_id']] in attaches
    }
    cases = build_cases(items, json.loads(args.texts.read_text()), cosines, metadata)
    _write_jsonl(args.cases_out, (case.to_row() for case in cases))
    _write_jsonl(args.features_out, (case.features.to_row() for case in cases))
    print(json.dumps({
        'cases': len(cases),
        'by_route': dict(sorted(Counter(str(case.features.route) for case in cases).items())),
        'metadata_unrecovered': sum(not case.features.metadata_recovered for case in cases),
        'cosine_unrecovered': sum(not case.features.cosine_recovered for case in cases),
    }, indent=2))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    commands = parser.add_subparsers(dest='command', required=True)
    build = commands.add_parser('build', help='write the cases (with texts) and their features')
    for flag in ('items', 'texts', 'key', 'hoist', 'adjudications', 'attaches', 'cases-out',
                 'features-out'):
        build.add_argument(f'--{flag}', type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return _build(args)


if __name__ == '__main__':
    sys.exit(main())
