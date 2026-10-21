#!/usr/bin/env python3
"""Build the rubric v2 baseline over the 2026-10-19 soak's 553 adjudicated attaches.

One-shot. It reads the gitignored soak bundle
(``data/write-triage-soak-2026-10-19/correctness/`` in the main checkout)
read-only, and never writes inside it. The v1 ratings there already fix some
of rubric v2's binaries; :func:`v1_determined` is the one statement of which
ones. Items whose correctness the v1 labels leave open are rated blind under
``calibration/write_triage_rater_brief_v2.md``.

Subcommands, run from ``fused-memory/``:

- ``shards``: write the primary shards over every item, a seeded
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
import json
import math
import random
import sqlite3
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import closing
from dataclasses import asdict, dataclass
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
from fused_memory.utils.script_loader import load_script

_SCRIPTS = Path(__file__).resolve().parent


if TYPE_CHECKING:
    import write_triage_reader_rubric as _rubric
else:
    _rubric = load_script(_SCRIPTS / 'write_triage_reader_rubric.py', 'write_triage_reader_rubric')

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
_HOME_YES = frozenset({'SAME', 'SUBSUMED'})
_HOME_NO = frozenset({'RELATED', 'UNRELATED'})
_CONFLICT_NO = frozenset({'SAME', 'SUBSUMED', 'UNRELATED'})
_V1_WORDS = _HOME_YES | _HOME_NO | {'EXTENDS', 'CORRECTS', 'UNCLEAR'}
_SCOPE_BY_DISPUTE: Mapping[str, StrEnum] = MappingProxyType({
    'genuine_correction': _rubric.Scope.CENTRAL, 'one_detail': _rubric.Scope.SECONDARY,
})
DERIVATION_MIN_COMPARED = 20
DERIVATION_MIN_AGREEMENT = 0.90
_CHECKED_ROLES = frozenset({Role.PRIMARY, Role.SECOND})


@dataclass(frozen=True)
class V1Row:
    """A v1 label pattern, the rubric v2.1 binary it may fix, and the answer it fixes.

    ``fixed`` gives the answer for a soak adjudication, or None when the row
    does not apply to it. Rationale: ``calibration/write_triage_reader_metrics.md``.
    """

    name: str
    binary: Binary
    fixed: Callable[[Mapping[str, Any]], StrEnum | None]


def _word_row(name: str, binary: Binary, words: frozenset[str], answer: StrEnum) -> V1Row:
    return V1Row(name, binary, lambda row: answer if row['rater_verdict'] in words else None)


def _action_changing_scope(row: Mapping[str, Any]) -> StrEnum | None:
    if row['rater_verdict'] != 'CORRECTS' or row.get('action_changing') is not True:
        return None
    return _SCOPE_BY_DISPUTE.get(row.get('dispute') or '')


def _action_changing_conflict(row: Mapping[str, Any]) -> StrEnum | None:
    return None if _action_changing_scope(row) is None else Answer.YES


def _noise_conflict(row: Mapping[str, Any]) -> StrEnum | None:
    return Answer.NO if row['rater_verdict'] == 'CORRECTS' and row.get('dispute') == 'noise' else None


def _not_action_changing_conflict(row: Mapping[str, Any]) -> StrEnum | None:
    corrects = row['rater_verdict'] == 'CORRECTS' and row.get('dispute') != 'noise'
    return Answer.NO if corrects and row.get('action_changing') is False else None


def _no_new_detail_loss(row: Mapping[str, Any]) -> StrEnum | None:
    known = row['rater_verdict'] != 'UNCLEAR'
    return Answer.NO if known and row.get('new_detail') is False else None


V1_ROWS: tuple[V1Row, ...] = (
    _word_row('related_unrelated_home_no', Binary.HOME, _HOME_NO, Answer.NO),
    _word_row('same_subsumed_home_yes', Binary.HOME, _HOME_YES, Answer.YES),
    _word_row('same_subsumed_unrelated_conflict_no', Binary.CONFLICT, _CONFLICT_NO, Answer.NO),
    V1Row('corrects_action_changing_conflict_yes', Binary.CONFLICT, _action_changing_conflict),
    V1Row('corrects_action_changing_scope', Binary.SCOPE, _action_changing_scope),
    V1Row('corrects_noise_conflict_no', Binary.CONFLICT, _noise_conflict),
    V1Row('no_new_detail_loss_no', Binary.LOSS, _no_new_detail_loss),
)
"""The candidate rows: each fixes its binary only while :func:`derivation_check` holds it."""

RETIRED_V1_ROWS: tuple[V1Row, ...] = (
    _word_row('extends_home_yes', Binary.HOME, frozenset({'EXTENDS'}), Answer.YES),
    V1Row('corrects_not_action_changing_conflict_no', Binary.CONFLICT,
          _not_action_changing_conflict),
)
"""The rows ruling R7 retired: measured by :func:`derivation_check`, never applied."""

_CANDIDATE_NAMES = frozenset(row.name for row in V1_ROWS)


def v1_determined(
    row: Mapping[str, Any], held: frozenset[str] = _CANDIDATE_NAMES,
) -> dict[_rubric.Binary, StrEnum]:
    """The answers the *held* candidate rows fix for a soak adjudication; every other binary stays open.

    A follow-up whose ``FOLLOW_UPS`` requirements this same determination does
    not meet is pruned, so LOSS stays only beside HOME YES and CONFLICT NO.
    """
    word = row['rater_verdict']
    if word not in _V1_WORDS:
        raise ValueError(f'adjudication {row.get("child_memory_id")!r}: unknown v1 word {word!r}')
    determined = {
        v1.binary: answer for v1 in V1_ROWS
        if v1.name in held and (answer := v1.fixed(row)) is not None
    }
    return _without_unmet_follow_ups(determined)


def _without_unmet_follow_ups(
    determined: dict[_rubric.Binary, StrEnum],
) -> dict[_rubric.Binary, StrEnum]:
    kept = dict(determined)
    while unmet := [
        follow_up for follow_up in kept
        if any(kept.get(binary) != needed
               for binary, needed in _rubric.FOLLOW_UPS.get(follow_up, {}).items())
    ]:
        for follow_up in unmet:
            del kept[follow_up]
    return kept


@dataclass(frozen=True)
class RowCheck:
    """How the primary and second raters' current-basis answers compare with one v1 row."""

    binary: Binary
    n_items: int
    n_compared: int
    agree: int
    held: bool
    retired: bool


def derivation_check(
    adjudications: Mapping[tuple[str, str], Mapping[str, Any] | None],
    rated_rows: Iterable[Mapping[str, Any]],
) -> Mapping[str, RowCheck]:
    """Each candidate and retired v1 row, measured against the raters' answers on its binary.

    *adjudications* maps each (entry, target) pair to its soak adjudication, or
    None for an item v1 never labelled. Only decided primary and second answers
    on the current basis are compared: third raters see only ties. A candidate
    row is held at :data:`DERIVATION_MIN_COMPARED` compared answers and
    :data:`DERIVATION_MIN_AGREEMENT` agreement.
    """
    rated: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rated_rows:
        if row['basis'] == Basis.CURRENT and row['role'] in _CHECKED_ROLES:
            rated[row['entry_id'], row['target_id']].append(row)
    rows = [(v1, False) for v1 in V1_ROWS] + [(v1, True) for v1 in RETIRED_V1_ROWS]
    checks: dict[str, RowCheck] = {}
    for v1, retired in rows:
        fixed = {
            pair: answer for pair, adjudication in adjudications.items()
            if adjudication is not None and (answer := v1.fixed(adjudication)) is not None
        }
        compared = [
            answer == row[v1.binary]
            for pair, answer in fixed.items() for row in rated.get(pair, ())
            if v1.binary in row and not _rubric.is_abstain(row[v1.binary])
        ]
        agree = sum(compared)
        checks[v1.name] = RowCheck(
            binary=v1.binary, n_items=len(fixed), n_compared=len(compared), agree=agree,
            held=not retired and _holds(len(compared), agree), retired=retired,
        )
    return MappingProxyType(checks)


def _holds(n_compared: int, agree: int) -> bool:
    return n_compared >= DERIVATION_MIN_COMPARED and agree >= DERIVATION_MIN_AGREEMENT * n_compared


def held_rows(check: Mapping[str, RowCheck]) -> frozenset[str]:
    """The names of the rows *check* holds."""
    return frozenset(name for name, row_check in check.items() if row_check.held)


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


def validate_rating(answer: Mapping[str, Any], child_text: str, parent_text: str) -> list[str]:
    """Each piece of *answer*'s evidence not copied from the texts its rater saw; [] when none.

    Quotes and spans are matched by ``quote_found``. A rater never gives DIGEST:
    assembly computes it from the loss spans (R3).
    """
    problems = []
    if Binary.DIGEST in answer:
        problems.append(f'digest {answer[Binary.DIGEST]!r} is given; digest is computed from loss_spans')
    quote = answer.get('conflict_quote')
    if isinstance(quote, Mapping):
        for side, text, shown_as in (('parent', parent_text, 'PARENT'), ('child', child_text, 'CHILD')):
            if not _found(quote.get(side), text):
                problems.append(f'conflict_quote.{side} {quote.get(side)!r} is not in {shown_as}')
    for span in answer.get('loss_spans') or ():
        if not _found(span, child_text):
            problems.append(f'loss_spans {span!r} is not in CHILD')
    return problems


def _found(quote: object, text: str) -> bool:
    return isinstance(quote, str) and _rubric.quote_found(quote, text)


def effective_ratings(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Each rater's highest-``round`` line on each pair and basis, without its ``round``.

    A re-ask round supersedes the rater's earlier line; two lines of one round are refused.
    """
    latest: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    for row in rows:
        identity = (row['entry_id'], row['target_id'], row['basis'], row['rater'])
        earlier = latest.get(identity)
        if earlier is not None and earlier['round'] == row['round']:
            raise ValueError(f'rater line {identity} appears twice in round {row["round"]}')
        if earlier is None or row['round'] > earlier['round']:
            latest[identity] = row
    return [{key: value for key, value in row.items() if key != 'round'} for row in latest.values()]


@dataclass(frozen=True)
class Assembly:
    """The committed verdict corpus, and the per-row check that decided which v1 rows it holds."""

    verdicts: list[dict[str, Any]]
    derivation: dict[str, Any]


def assemble(
    adjudications: Mapping[tuple[str, str], Mapping[str, Any] | None],
    rated_rows: Iterable[Mapping[str, Any]],
    child_texts: Mapping[str, str],
) -> Assembly:
    """Derived rows from the held v1 rows, and every rater answer kept whole.

    *adjudications* maps each (entry, target) pair to its soak adjudication, or
    None for an extension item; *child_texts* maps each entry to its full CHILD.
    A current-basis rater answer on a binary its pair's derived row fixes is
    marked ``non_consumed``, and every LOSS YES row gets DIGEST from ``digest_visible``.
    """
    rated = list(rated_rows)
    check = derivation_check(adjudications, rated)
    held = held_rows(check)
    fixed = {
        pair: determined for pair, adjudication in adjudications.items()
        if adjudication is not None and (determined := v1_determined(adjudication, held))
    }
    verdicts = [_derived_row(entry_id, target_id, answers)
                for (entry_id, target_id), answers in fixed.items()]
    verdicts += [_assembled_rater_row(row, fixed, child_texts) for row in rated]
    return Assembly(
        verdicts=sorted(verdicts, key=_verdict_order),
        derivation={
            'check': {
                'min_compared': DERIVATION_MIN_COMPARED,
                'min_agreement': DERIVATION_MIN_AGREEMENT,
                'roles': sorted(map(str, _CHECKED_ROLES)),
            },
            'rows': {name: {**asdict(row_check), 'binary': str(row_check.binary)}
                     for name, row_check in check.items()},
            'held_rows': sorted(held),
        },
    )


def _assembled_rater_row(
    row: Mapping[str, Any],
    fixed: Mapping[tuple[str, str], Mapping[_rubric.Binary, StrEnum]],
    child_texts: Mapping[str, str],
) -> dict[str, Any]:
    assembled = dict(row)
    if row['basis'] == Basis.CURRENT:
        derived = fixed.get((row['entry_id'], row['target_id']), {})
        if non_consumed := sorted(str(binary) for binary in derived if binary in row):
            assembled['non_consumed'] = non_consumed
    if row.get(Binary.LOSS) == Answer.YES:
        visible = _rubric.digest_visible(row['loss_spans'], child_texts[row['entry_id']])
        assembled[str(Binary.DIGEST)] = str(Answer.YES if visible else Answer.NO)
    return assembled


def _verdict_order(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (row['entry_id'], row['target_id'], row['basis'], row['rater'])


# --------------------------------------------------------------------------
# The bundle, the scratch directory and the CLI
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class _SoakItem:
    item: Mapping[str, Any]
    adjudication: Mapping[str, Any] | None
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
            adjudication=adjudication,
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
    needed = list(soak)
    settled: list[_SoakItem] = []
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


def _assembled(soak: Sequence[_SoakItem], rated_rows: Iterable[Mapping[str, Any]]) -> Assembly:
    return assemble(
        {soaked.pair: soaked.adjudication for soaked in soak}, rated_rows,
        {soaked.item['entry_id']: soaked.child_text for soaked in soak},
    )


def _third_command(args: argparse.Namespace) -> dict[str, Any]:
    soak = _load_bundle(args.bundle)
    scratch = _Scratch(args.scratch)
    key = scratch.load_key()
    assembly = _assembled(soak, scratch.ratings(key))
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
    assembly = _assembled(soak, scratch.ratings(key))
    out: Path = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / 'write_triage_v2_soak_items.jsonl', [soaked.item for soaked in soak])
    _write_jsonl(out / 'write_triage_v2_soak_verdicts.jsonl', assembly.verdicts)
    _write_json(out / 'write_triage_v2_soak_strata.json', dict(STRATA))
    _write_json(out / 'write_triage_v2_soak_derivation.json', {
        **assembly.derivation,
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
