#!/usr/bin/env python3
"""Build the rubric v2 baseline over the 2026-10-19 soak's 553 adjudicated attaches.

One-shot. It reads the gitignored soak bundle
(``data/write-triage-soak-2026-10-19/correctness/`` in the main checkout)
read-only, and never writes inside it. The v1 ratings there already fix some
of rubric v2's binaries; :func:`v1_determined` is the one statement of which
ones. Items whose correctness the v1 labels leave open are rated blind under
``calibration/write_triage_rater_brief_v2.md``.

Subcommands, run from ``fused-memory/``:

- ``shards``: select the rating set (:func:`needs_rating`) plus a few fully
  determined anchors per shard, and write the primary shards, a seeded
  second-rater overlap, attach-time shards for parents edited after attach,
  ``key.json`` and ``rater_prompt.md`` into ``--scratch``.
- ``third``: after the first ratings, write a shard holding exactly the pairs
  whose decisive binaries are tied, for a third rater.
- ``assemble``: map the ratings back through the key and write the committed
  corpus: ``write_triage_v2_soak_{items.jsonl,verdicts.jsonl,strata.json,derivation.json}``.

Rater output lives in ``<scratch>/ratings/<shard>.jsonl``, one line per item.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sqlite3
import sys
import types
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from fused_memory.server.grouped_read import _digest
from fused_memory.server.write_triage import (
    OUTCOME_AMENDED,
    OUTCOME_CONTESTED,
    OUTCOME_JUDGE,
    OUTCOME_RESTATED,
)

_SCRIPTS = Path(__file__).resolve().parent


def _load_script(path: Path, mod_name: str) -> types.ModuleType:
    """Load a ``scripts/`` sibling by path, cached in ``sys.modules``."""
    cached = sys.modules.get(mod_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


if TYPE_CHECKING:
    import write_triage_reader_rubric as _rubric
else:
    _rubric = _load_script(_SCRIPTS / 'write_triage_reader_rubric.py', 'write_triage_reader_rubric')

Binary = _rubric.Binary
Answer = _rubric.Answer
Basis = _rubric.Basis
Role = _rubric.Role

BRIEF_PATH = _SCRIPTS.parent / 'calibration' / 'write_triage_rater_brief_v2.md'
TEXT_CAP = 4000
PRIMARY_SHARD_SIZE = 45
ANCHORS_PER_SHARD = 5
SECOND_OVERLAP = 80
SECOND_SHARD_SIZE = 40
RATER_MODELS: Mapping[str, str] = MappingProxyType({
    Role.PRIMARY: 'opus', Role.SECOND: 'sonnet', Role.THIRD: 'opus',
})
DERIVED_RATER = 'v1-soak-2026-10-19'
DERIVED_BATCH = 'soak-2026-10-19-adjudications'

STRATUM_BY_OUTCOME: Mapping[str, str] = MappingProxyType({
    OUTCOME_RESTATED: 'restated-sample',
    OUTCOME_AMENDED: 'amended-sample',
    OUTCOME_CONTESTED: 'contested-census',
})
STRATA: Mapping[str, int] = MappingProxyType({
    'restated-sample': 125, 'amended-sample': 1224, 'contested-census': 473,
})
"""Each soak stratum's population: 40 of 125 restated and 40 of 1,224 amended were
sampled, and all 473 contested attaches were rated (``correctness/analyze.py``)."""

_BAND_BY_HIGH: Mapping[bool | None, str | None] = MappingProxyType({
    True: OUTCOME_RESTATED, False: OUTCOME_JUDGE, None: None,
})
_HOME_YES = frozenset({'SAME', 'EXTENDS', 'SUBSUMED'})
_HOME_NO = frozenset({'RELATED', 'UNRELATED'})
_CONFLICT_NO = frozenset({'SAME', 'SUBSUMED', 'UNRELATED'})
_V1_WORDS = _HOME_YES | _HOME_NO | {'CORRECTS', 'UNCLEAR'}
_SCOPE_BY_DISPUTE: Mapping[str, StrEnum] = MappingProxyType({
    'genuine_correction': _rubric.Scope.CENTRAL, 'one_detail': _rubric.Scope.SECONDARY,
})
_FOLLOW_UPS: Mapping[Binary, Binary] = MappingProxyType({
    Binary.SCOPE: Binary.CONFLICT, Binary.DIGEST: Binary.LOSS,
})
_SUMMARY_KEYS = ('derived', 'rated', 'n_compared', 'agree')


def v1_determined(row: Mapping[str, Any]) -> dict[_rubric.Binary, StrEnum]:
    """The v2 answers a soak adjudication's v1 labels fix; every other binary stays open.

    HOME is fixed by SAME/EXTENDS/SUBSUMED (YES) and RELATED/UNRELATED (NO);
    CORRECTS covers "same claim or subject", so it cannot fix HOME (ruling 5a).
    LOSS is fixed only by ``new_detail`` false, since v1 never asked whether a
    reader acting on the parent would need the detail. CONFLICT is fixed by
    SAME/SUBSUMED/UNRELATED (NO) and by CORRECTS when its dispute question
    was answered: action-changing genuine_correction or one_detail is YES,
    with SCOPE CENTRAL or SECONDARY; not action-changing, or noise, is NO.
    EXTENDS and RELATED leave CONFLICT open, because v1 asked the dispute
    question only of CORRECTS. UNCLEAR fixes nothing.
    """
    word = row['rater_verdict']
    if word not in _V1_WORDS:
        raise ValueError(f'adjudication {row.get("child_memory_id")!r}: unknown v1 word {word!r}')
    if word == 'UNCLEAR':
        return {}
    determined: dict[_rubric.Binary, StrEnum] = {}
    if word in _HOME_YES or word in _HOME_NO:
        determined[Binary.HOME] = Answer.YES if word in _HOME_YES else Answer.NO
    if row.get('new_detail') is False:
        determined[Binary.LOSS] = Answer.NO
    if word in _CONFLICT_NO:
        determined[Binary.CONFLICT] = Answer.NO
    elif word == 'CORRECTS':
        determined.update(_corrects_determined(row))
    return determined


def _corrects_determined(row: Mapping[str, Any]) -> dict[_rubric.Binary, StrEnum]:
    scope = _SCOPE_BY_DISPUTE.get(row.get('dispute') or '')
    if row.get('action_changing') is True and scope is not None:
        return {Binary.CONFLICT: Answer.YES, Binary.SCOPE: scope}
    if row.get('action_changing') is False or row.get('dispute') == 'noise':
        return {Binary.CONFLICT: Answer.NO}
    return {}


def item_row(
    adjudication: Mapping[str, Any],
    *,
    high_band: bool | None,
    parent_category: str | None,
    child_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """The scorer's item row for one soak attach (``score_write_triage_reader.py::ReaderItem``)."""
    parent_id = adjudication['parent_id']
    return {
        'entry_id': adjudication['child_memory_id'],
        'outcome': adjudication['outcome'],
        'target_id': parent_id,
        'earlier_ids': [],
        'band': _BAND_BY_HIGH[high_band],
        'project': adjudication['project'],
        'stratum': STRATUM_BY_OUTCOME[adjudication['outcome']],
        'via_sibling': adjudication.get('likely_via_sibling'),
        'cross_category': (
            None if parent_category is None else adjudication['category'] != parent_category
        ),
        'parent_edited_after_attach': adjudication.get('parent_edited_after_attach'),
        'supersedes_target': parent_id in (child_metadata.get('supersedes') or []),
        'superseded_marks': [],
    }


def _derived_row(entry_id: str, target_id: str, answers: Mapping[Any, StrEnum]) -> dict[str, Any]:
    return {
        'entry_id': entry_id, 'target_id': target_id, 'basis': str(Basis.CURRENT),
        'rater': DERIVED_RATER, 'role': str(Role.DERIVED), 'batch': DERIVED_BATCH,
        **{str(binary): str(answer) for binary, answer in answers.items()},
    }


def _current_derivation(
    item: Mapping[str, Any], judgements: Mapping[_rubric.PairKey, _rubric.Judgement],
) -> _rubric.Derivation:
    key = _rubric.PairKey(item['entry_id'], item['target_id'], Basis.CURRENT)
    return _rubric.derive(
        item['outcome'], {item['target_id']: judgements.get(key)},
        supersedes_target=item['supersedes_target'],
        superseded_marks=frozenset(item['superseded_marks']),
        in_m1_band=item['band'] == _rubric.M1_BAND,
    )


def needs_rating(item: Mapping[str, Any], determined: Mapping[_rubric.Binary, StrEnum]) -> bool:
    """Whether the v1 labels alone leave a binary the derivation needs unrated."""
    rows = [_derived_row(item['entry_id'], item['target_id'], determined)] if determined else []
    derivation = _current_derivation(item, _rubric.resolve_judgements(rows))
    return isinstance(derivation, _rubric.Incomplete)


def decisive_ties(
    items: Iterable[Mapping[str, Any]], verdicts: Iterable[Mapping[str, Any]],
) -> list[tuple[str, str]]:
    """The (entry, target) pairs whose current-basis derivation is blocked by a tie."""
    judgements = _rubric.resolve_judgements(verdicts)
    tied: list[tuple[str, str]] = []
    for item in items:
        derivation = _current_derivation(item, judgements)
        if isinstance(derivation, _rubric.Incomplete):
            kinds = {gap.kind for gap in derivation.missing}
            if kinds != {_rubric.GapKind.TIED}:
                raise ValueError(
                    f'item {item["entry_id"]!r} has unrated decisive binaries'
                    f' {[str(gap.binary) for gap in derivation.missing]}; rate its shard first'
                )
            tied.append((item['entry_id'], item['target_id']))
    return sorted(tied)


def attach_time_text(
    history_db: Path, parent_id: str, child_created_at: str, current_text: str,
) -> str | None:
    """The parent's text when the child was written, from mem0's history table.

    The old text of the earliest UPDATE after the child; the current text when
    the parent was not edited after it; None when that UPDATE kept no old text.
    """
    child_at = datetime.fromisoformat(child_created_at)
    uri = f'{Path(history_db).resolve().as_uri()}?mode=ro'
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        connection.execute('PRAGMA temp_store=MEMORY')
        edits = connection.execute(
            "SELECT old_memory, updated_at FROM history WHERE memory_id = ? AND event = 'UPDATE'",
            (parent_id,),
        ).fetchall()
    later = [
        (edited_at, old) for old, at in edits
        if at and (edited_at := datetime.fromisoformat(at)) > child_at
    ]
    if not later:
        return current_text
    return min(later, key=lambda edit: edit[0])[1]


@dataclass(frozen=True)
class ShardEntry:
    """One pair to rate: the write, the record it is read against, which text, and both texts."""

    entry_id: str
    target_id: str
    basis: str
    project: str
    child_text: str
    parent_text: str


@dataclass(frozen=True)
class Shard:
    """A shard's text, and the key from each opaque item id back to its pair and basis."""

    text: str
    key: Mapping[str, Mapping[str, str]]


def shard(entries: Sequence[ShardEntry], *, name: str) -> Shard:
    """Render *entries*, in order, in the soak's shard layout plus the reader's child digest."""
    lines: list[str] = []
    key: dict[str, Mapping[str, str]] = {}
    for index, entry in enumerate(entries):
        item_id = f'{name}-{index:03d}'
        key[item_id] = MappingProxyType({
            'entry_id': entry.entry_id, 'target_id': entry.target_id, 'basis': entry.basis,
        })
        lines += [
            f'=== ITEM id={item_id} | project={entry.project} | basis={entry.basis}',
            '--- PARENT:', _capped(entry.parent_text),
            '--- CHILD:', _capped(entry.child_text),
            '--- CHILD DIGEST:', _digest(entry.child_text),
            '',
        ]
    return Shard(text='\n'.join(lines) + '\n', key=MappingProxyType(key))


def _capped(text: str) -> str:
    if len(text) <= TEXT_CAP:
        return text
    return text[:TEXT_CAP] + f'…[truncated, {len(text)} chars total]'


@dataclass(frozen=True)
class Assembly:
    """The committed verdict corpus, and how the rated answers compare with the v1 labels."""

    verdicts: list[dict[str, Any]]
    derivation: dict[str, dict[str, int]]


def assemble(
    determined: Mapping[tuple[str, str], Mapping[_rubric.Binary, StrEnum]],
    rated_rows: Iterable[Mapping[str, Any]],
) -> Assembly:
    """One family of votes per binary: v1's where it fixes the binary, the raters' elsewhere.

    Each pair gets one ``derived`` row carrying its v1-fixed binaries. A rated
    current-basis row keeps only the binaries v1 left open (a follow-up stays
    only with the binary it follows); its answers on v1-fixed binaries are
    counted against v1 in the derivation summary and never reach the corpus.
    An attach-time row keeps every answer.
    """
    verdicts = [
        _derived_row(entry_id, target_id, answers)
        for (entry_id, target_id), answers in sorted(determined.items()) if answers
    ]
    summary = {binary: Counter[str]() for binary in Binary}
    for answers in determined.values():
        for binary in answers:
            summary[Binary(binary)]['derived'] += 1
    for row in rated_rows:
        if row['basis'] == Basis.ATTACH_TIME:
            verdicts.append(dict(row))
            continue
        fixed = determined.get((row['entry_id'], row['target_id']), {})
        kept = _open_answers(row, fixed)
        for binary in Binary:
            if binary in kept:
                summary[binary]['rated'] += 1
            elif binary in row and binary in fixed and not _rubric.is_abstain(row[binary]):
                summary[binary]['n_compared'] += 1
                summary[binary]['agree'] += row[binary] == fixed[binary]
        if kept:
            verdicts.append(_kept_row(row, kept))
    return Assembly(
        verdicts=sorted(verdicts, key=_verdict_order),
        derivation={
            str(binary): {key: counts[key] for key in _SUMMARY_KEYS}
            for binary, counts in summary.items()
        },
    )


def _open_answers(row: Mapping[str, Any], fixed: Mapping[Any, StrEnum]) -> dict[str, str]:
    kept = {
        str(binary): row[binary] for binary in Binary
        if binary in row and binary not in fixed and binary not in _FOLLOW_UPS
    }
    for follow_up, asked_after in _FOLLOW_UPS.items():
        if follow_up in row and asked_after in kept:
            kept[str(follow_up)] = row[follow_up]
    return kept


def _kept_row(row: Mapping[str, Any], kept: Mapping[str, str]) -> dict[str, Any]:
    identity = {key: row[key] for key in ('entry_id', 'target_id', 'basis', 'rater', 'role', 'batch')}
    if 'reason' in row:
        identity['reason'] = row['reason']
    if any(map(_rubric.is_abstain, kept.values())) and 'unclear_reason' in row:
        identity['unclear_reason'] = row['unclear_reason']
    return {**identity, **kept}


def _verdict_order(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (row['entry_id'], row['target_id'], row['basis'], row['rater'])


# --------------------------------------------------------------------------
# The bundle, the scratch directory and the CLI
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class _SoakItem:
    item: Mapping[str, Any]
    determined: Mapping[_rubric.Binary, StrEnum]
    child_text: str
    parent_text: str
    child_created_at: str

    @property
    def pair(self) -> tuple[str, str]:
        return (self.item['entry_id'], self.item['target_id'])

    def entry(self, basis: str, parent_text: str | None = None) -> ShardEntry:
        return ShardEntry(
            entry_id=self.item['entry_id'], target_id=self.item['target_id'], basis=basis,
            project=self.item['project'], child_text=self.child_text,
            parent_text=self.parent_text if parent_text is None else parent_text,
        )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _load_bundle(bundle: Path) -> list[_SoakItem]:
    adjudications = [
        json.loads(line) for line in (bundle / 'adjudications.jsonl').read_text().splitlines()
        if line.strip()
    ]
    texts = {row['child_memory_id']: row for row in _read_json(bundle / 'items_all.json')}
    hoist = _read_json(bundle / 'hoist.json')
    metadata = {row['intent_id']: row.get('meta') or {} for row in _read_json(bundle / 'attaches_soak.json')}
    soak = []
    for adjudication in adjudications:
        text = texts[adjudication['child_memory_id']]
        intent = adjudication['child_intent_id']
        soak.append(_SoakItem(
            item=item_row(
                adjudication, high_band=hoist.get(intent, {}).get('high_band'),
                parent_category=text.get('parent_category'),
                child_metadata=metadata.get(intent, {}),
            ),
            determined=v1_determined(adjudication),
            child_text=text.get('child_text') or '',
            parent_text=text.get('parent_text') or '',
            child_created_at=adjudication['created_at'],
        ))
    return sorted(soak, key=lambda soaked: soaked.item['entry_id'])


def _refuse_inside(parser: argparse.ArgumentParser, scratch: Path, bundle: Path) -> None:
    scratch, bundle = scratch.resolve(), bundle.resolve()
    if scratch == bundle or bundle in scratch.parents:
        parser.error(f'--scratch {scratch} is inside the read-only --bundle {bundle}')


class _Scratch:
    """The scratch directory: shards, the key, the rater prompt and the ratings."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.key_path = root / 'key.json'

    def load_key(self) -> dict[str, Any]:
        if not self.key_path.exists():
            return {'shards': {}, 'items': {}, 'attach_time_unrecoverable': []}
        return _read_json(self.key_path)

    def write_shard(self, key: dict[str, Any], name: str, role: str, entries: Sequence[ShardEntry],
                    anchors: frozenset[tuple[str, str]] = frozenset()) -> None:
        rendered = shard(entries, name=name)
        (self.root / 'shards').mkdir(parents=True, exist_ok=True)
        (self.root / 'shards' / f'{name}.txt').write_text(rendered.text)
        key['shards'][name] = {'role': role, 'rater': f'{RATER_MODELS[role]}-{role}',
                               'items': sorted(rendered.key)}
        for item_id, pair in rendered.key.items():
            key['items'][item_id] = {
                **pair, 'anchor': (pair['entry_id'], pair['target_id']) in anchors,
            }

    def save_key(self, key: Mapping[str, Any]) -> None:
        self.key_path.write_text(json.dumps(key, indent=2, sort_keys=True) + '\n')

    def ratings(self, key: Mapping[str, Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for name, spec in sorted(key['shards'].items()):
            rows += self._shard_ratings(name, spec, key['items'])
        return rows

    def _shard_ratings(
        self, name: str, spec: Mapping[str, Any], items: Mapping[str, Mapping[str, str]],
    ) -> list[dict[str, Any]]:
        path = self.root / 'ratings' / f'{name}.jsonl'
        if not path.exists():
            raise ValueError(f'{path}: shard {name} has no ratings yet')
        rows, seen = [], Counter[str]()
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            answer = json.loads(line)
            item_id = answer.pop('id', None)
            if item_id not in spec['items']:
                raise ValueError(f'{path}:{number}: id {item_id!r} is not an item of shard {name}')
            seen[item_id] += 1
            row = {**items[item_id], 'rater': spec['rater'], 'role': spec['role'], 'batch': name,
                   **answer}
            row.pop('anchor', None)
            try:
                _rubric.resolve_judgements([row])
            except ValueError as exc:
                raise ValueError(f'{path}:{number}: {exc}') from exc
            rows.append(row)
        wrong = sorted(item for item in spec['items'] if seen[item] != 1)
        if wrong:
            raise ValueError(f'{path}: items without exactly one line: {wrong}')
        return rows


def _shards_command(args: argparse.Namespace) -> dict[str, Any]:
    soak = _load_bundle(args.bundle)
    scratch = _Scratch(args.scratch)
    key: dict[str, Any] = {'shards': {}, 'items': {}, 'attach_time_unrecoverable': []}
    rng = random.Random(args.seed)
    needed = [soaked for soaked in soak if needs_rating(soaked.item, soaked.determined)]
    settled = [soaked for soaked in soak if not needs_rating(soaked.item, soaked.determined)]
    rng.shuffle(needed)
    count = math.ceil(len(needed) / PRIMARY_SHARD_SIZE)
    anchors = rng.sample(settled, min(len(settled), ANCHORS_PER_SHARD * count))
    anchor_pairs = frozenset(soaked.pair for soaked in anchors)
    for index in range(count):
        members = needed[index::count] + anchors[index::count]
        rng.shuffle(members)
        scratch.write_shard(key, f'primary_{index:02d}', Role.PRIMARY,
                            [soaked.entry(Basis.CURRENT) for soaked in members], anchor_pairs)
    overlap = rng.sample(needed, min(SECOND_OVERLAP, len(needed)))
    for index in range(math.ceil(len(overlap) / SECOND_SHARD_SIZE)):
        members = overlap[index * SECOND_SHARD_SIZE:(index + 1) * SECOND_SHARD_SIZE]
        scratch.write_shard(key, f'second_{index:02d}', Role.SECOND,
                            [soaked.entry(Basis.CURRENT) for soaked in members])
    attach_time = []
    for soaked in soak:
        if not soaked.item['parent_edited_after_attach']:
            continue
        text = attach_time_text(args.history_db, soaked.item['target_id'],
                                soaked.child_created_at, soaked.parent_text)
        if text is None:
            key['attach_time_unrecoverable'].append(soaked.item['entry_id'])
        else:
            attach_time.append(soaked.entry(Basis.ATTACH_TIME, text))
    if attach_time:
        scratch.write_shard(key, 'attach_00', Role.PRIMARY, attach_time)
    scratch.save_key(key)
    (args.scratch / 'rater_prompt.md').write_text(_rater_prompt())
    return {
        'items': len(soak), 'needs_rating': len(needed), 'anchors': len(anchors),
        'second_overlap': len(overlap), 'attach_time': len(attach_time),
        'attach_time_unrecoverable': key['attach_time_unrecoverable'],
        'shards': {name: len(spec['items']) for name, spec in key['shards'].items()},
    }


def _third_command(args: argparse.Namespace) -> dict[str, Any]:
    soak = _load_bundle(args.bundle)
    scratch = _Scratch(args.scratch)
    key = scratch.load_key()
    assembly = assemble({soaked.pair: soaked.determined for soaked in soak}, scratch.ratings(key))
    tied = set(decisive_ties([soaked.item for soaked in soak], assembly.verdicts))
    name = f'third_{sum(spec["role"] == Role.THIRD for spec in key["shards"].values()):02d}'
    entries = [soaked.entry(Basis.CURRENT) for soaked in soak if soaked.pair in tied]
    if entries:
        scratch.write_shard(key, name, Role.THIRD, entries)
        scratch.save_key(key)
    return {'tied_pairs': len(entries), 'shard': name if entries else None}


def _assemble_command(args: argparse.Namespace) -> dict[str, Any]:
    soak = _load_bundle(args.bundle)
    scratch = _Scratch(args.scratch)
    key = scratch.load_key()
    assembly = assemble({soaked.pair: soaked.determined for soaked in soak}, scratch.ratings(key))
    out: Path = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / 'write_triage_v2_soak_items.jsonl', [soaked.item for soaked in soak])
    _write_jsonl(out / 'write_triage_v2_soak_verdicts.jsonl', assembly.verdicts)
    _write_json(out / 'write_triage_v2_soak_strata.json', dict(STRATA))
    _write_json(out / 'write_triage_v2_soak_derivation.json', {
        'by_binary': assembly.derivation,
        **rating_volume(list(key['items'].values())),
        'attach_time_unrecoverable': sorted(key['attach_time_unrecoverable']),
        'shards': {name: {'rater': spec['rater'], 'items': len(spec['items'])}
                   for name, spec in sorted(key['shards'].items())},
    })
    return {'items': len(soak), 'verdicts': len(assembly.verdicts)}


def rating_volume(key_items: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """The distinct (entry, target) pairs the shards put before raters, by what they were for.

    ``rated_current_pairs`` are the current-basis pairs whose v1 labels left a
    decisive binary open, ``anchor_pairs`` the current-basis anchors v1 had
    already settled, and ``attach_time_pairs`` the pairs re-read against the
    parent's text at attach time. *key_items* are the scratch key's item entries.
    """
    def pairs(basis: str, *, anchor: bool) -> int:
        return len({
            (item['entry_id'], item['target_id']) for item in key_items
            if item['basis'] == basis and item['anchor'] is anchor
        })

    return {
        'rated_current_pairs': pairs(Basis.CURRENT, anchor=False),
        'anchor_pairs': pairs(Basis.CURRENT, anchor=True),
        'attach_time_pairs': pairs(Basis.ATTACH_TIME, anchor=False),
    }


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def _rater_prompt() -> str:
    return '\n'.join([
        '# Rater task (read one shard file, write one JSONL file)',
        '',
        'You are a blind rater. Read ONLY the shard file you are given and write ONLY the'
        ' output file you are given. Do not open any other file (no key file, no repository'
        ' file, no store), and do not call any memory or task tool.',
        '',
        '## The rubric (verbatim, fused-memory/calibration/write_triage_rater_brief_v2.md)',
        '',
        BRIEF_PATH.read_text(),
        '## Output',
        '',
        'Each shard item starts with `=== ITEM id=<id> | project=<project> | basis=<basis>`'
        ' and shows PARENT, CHILD and CHILD DIGEST. Write one JSON object per line, in item'
        ' order, to the output file you are given:'
        ' `{"id": "<item id>", "home": "YES|NO|UNCLEAR", "loss": "YES|NO|UNCLEAR",'
        ' "conflict": "YES|NO|UNCLEAR", "scope": "CENTRAL|SECONDARY|UNCLEAR",'
        ' "digest": "YES|NO|UNCLEAR", "unclear_reason": "...", "reason": "<one sentence'
        ' naming both claims>"}`. Give `scope` only when `conflict` is YES, `digest` only'
        ' when `loss` is YES, and `unclear_reason` only with an UNCLEAR answer.',
        '',
        'Append to the file after every batch of about 10 items, so partial progress'
        ' survives. Every item in the shard gets exactly one line. When done, reply with'
        ' only the number of lines written.',
        '',
    ])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('shards', 'third', 'assemble'):
        command = commands.add_parser(name)
        command.add_argument('--bundle', type=Path, required=True, metavar='DIR',
                             help="the soak bundle's correctness/ directory (read-only)")
        command.add_argument('--scratch', type=Path, required=True, metavar='DIR')
    commands.choices['shards'].add_argument('--history-db', type=Path, required=True,
                                            metavar='PATH', help="mem0's history.db")
    commands.choices['shards'].add_argument('--seed', type=int, default=7157)
    commands.choices['assemble'].add_argument('--out-dir', type=Path, required=True, metavar='DIR')
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _refuse_inside(parser, args.scratch, args.bundle)
    if args.command == 'shards':
        args.history_db = args.history_db.expanduser()
        summary = _shards_command(args)
    elif args.command == 'third':
        summary = _third_command(args)
    else:
        summary = _assemble_command(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
