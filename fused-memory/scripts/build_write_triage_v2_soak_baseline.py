#!/usr/bin/env python3
"""Build the rubric v2.1 baseline over the 2026-10-19 soak's attaches.

One-shot. It reads the gitignored soak bundle
(``data/write-triage-soak-2026-10-19/correctness/`` in the main checkout),
the write journal and mem0's history, all read-only, and never writes inside
the bundle. Every item is rated blind by two raters under
``calibration/write_triage_rater_brief_v2.md``; a v1 label fixes a binary only
through a row :func:`derivation_check` holds.

Subcommands, run from ``fused-memory/``:

- ``shards``: extend the restated and amended samples (R8), then write a
  primary and a second shard set over every pair, attach-time pairs in their
  own shards, plus ``key.json``, ``texts.json`` and ``rater_prompt.md`` into
  ``--scratch``. A re-run shards only what is not yet keyed.
- ``reask``: write a shard per rater holding the lines that break the
  rubric's row contract or whose quotes or spans are not in the texts that
  rater saw. The rater's answer in the later round supersedes the earlier line.
- ``third``: write a shard holding exactly the pairs a decisive tie or an open
  follow-up blocks, for a new third rater each round.
- ``assemble``: map the effective ratings back through the key and write the
  committed corpus:
  ``write_triage_v2_soak_{items.jsonl,verdicts.jsonl,strata.json,derivation.json}``.
  ``third`` and ``assemble`` refuse while any line awaits a re-ask or any pair
  lacks a primary or second answer.

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
SHARD_SIZE = 40
SAMPLE_TOTAL = 100
"""How many restated and how many amended attaches the extended sample holds (R8)."""
RATER_MODELS: Mapping[str, str] = MappingProxyType({
    Role.PRIMARY: 'opus', Role.SECOND: 'opus', Role.THIRD: 'opus',
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
    attach: Mapping[str, Any],
    *,
    child_memory_id: str,
    high_band: bool | None,
    via_sibling: bool | None,
    parent_category: str | None,
    parent_edited_after_attach: bool,
) -> dict[str, Any]:
    """The scorer's item row (``score_write_triage_reader.py::ReaderItem``) for one soak attach record."""
    meta = attach['meta']
    parent_id = meta['parent_id']
    return {
        'entry_id': child_memory_id,
        'outcome': attach['outcome'],
        'target_id': parent_id,
        'earlier_ids': [],
        'band': _BAND_BY_HIGH[high_band],
        'project': attach['project'],
        'stratum': STRATUM_BY_OUTCOME[attach['outcome']],
        'via_sibling': via_sibling,
        'cross_category': None if parent_category is None else attach['category'] != parent_category,
        'parent_edited_after_attach': parent_edited_after_attach,
        'supersedes_target': parent_id in (meta.get('supersedes') or []),
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


_THIRD_ROUND_GAPS = frozenset({_rubric.GapKind.TIED, _rubric.GapKind.OPEN})
"""The gaps a third rater answering the whole item can fill: a tie, or an open follow-up."""


def third_round_pairs(
    items: Iterable[Mapping[str, Any]], verdicts: Iterable[Mapping[str, Any]],
) -> list[tuple[str, str]]:
    """The (entry, target) pairs whose current-basis derivation only a third rater can unblock.

    Each is blocked by a decisive tie or an open follow-up alone. A pair also
    blocked by an unrated decisive binary is refused.
    """
    judgements = _rubric.resolve_judgements(verdicts)
    blocked: list[tuple[str, str]] = []
    for item in items:
        derivation = _current_derivation(item, judgements)
        if isinstance(derivation, _rubric.Incomplete):
            unrated = [str(gap.binary) for gap in derivation.missing
                       if gap.kind not in _THIRD_ROUND_GAPS]
            if unrated:
                raise ValueError(
                    f'item {item["entry_id"]!r} has unrated decisive binaries {unrated};'
                    ' a third rater fills only ties and open follow-ups'
                )
            blocked.append((item['entry_id'], item['target_id']))
    return sorted(blocked)


def attach_time_text(
    history_db: Path, parent_id: str, child_created_at: str, current_text: str,
) -> str | None:
    """The parent's text when the child was written, from mem0's history table.

    The old text of the earliest UPDATE after the child; the current text when
    the parent was not edited after it; None when that UPDATE kept no old text.
    """
    child_at = datetime.fromisoformat(child_created_at)
    with _read_only(history_db) as connection:
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


def _read_only(database: Path) -> closing[sqlite3.Connection]:
    connection = sqlite3.connect(f'{Path(database).resolve().as_uri()}?mode=ro', uri=True)
    connection.execute('PRAGMA temp_store=MEMORY')
    return closing(connection)


def extend_sample(
    population: Iterable[str], sampled: Iterable[str], total: int, *, seed: int | str,
) -> list[str]:
    """The ids to add so that *total* of *population* are sampled, in draw order.

    A seeded permutation of the unsampled ids, cut to length, so a larger
    *total* extends a smaller one's draw. Sampled ids outside *population* do not count.
    """
    members = frozenset(population)
    if total > len(members):
        raise ValueError(f'a sample of {total} exceeds the population of {len(members)}')
    already = members & frozenset(sampled)
    unsampled = sorted(members - already)
    return random.Random(seed).sample(unsampled, len(unsampled))[:max(0, total - len(already))]


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


def _entry_order(entry: ShardEntry) -> tuple[str, str, str]:
    return (entry.entry_id, entry.target_id, entry.basis)


def plan_shards(
    entries: Iterable[ShardEntry], *, role: str, prefix: str, seed: int | str, first_index: int,
) -> list[tuple[str, list[ShardEntry]]]:
    """*entries* in a seeded per-role order, dealt into named shards of at most :data:`SHARD_SIZE`.

    Shards are named ``<prefix><role>_<NN>``, numbered from *first_index*.
    """
    ordered = sorted(entries, key=_entry_order)
    random.Random(f'{seed}:{prefix}{role}').shuffle(ordered)
    count = math.ceil(len(ordered) / SHARD_SIZE)
    return [(f'{prefix}{role}_{first_index + index:02d}', ordered[index::count])
            for index in range(count)]


def shard(entries: Sequence[ShardEntry], *, name: str) -> Shard:
    """Render *entries*, in order, in the soak's shard layout."""
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


def rating_problems(row: Mapping[str, Any], child_text: str, parent_text: str) -> list[str]:
    """Why an effective rater line must be asked again; [] when it need not.

    A line breaking the rubric's row contract is named for that alone, since
    :func:`validate_rating` presumes a line that keeps it; any other line gets
    :func:`validate_rating`'s evidence problems.
    """
    violation = _contract_violation(row)
    if violation is not None:
        return [violation]
    return validate_rating(row, child_text, parent_text)


def _contract_violation(row: Mapping[str, Any]) -> str | None:
    """The violation the rubric's ``resolve_judgements`` refuses *row* alone for; None if none."""
    try:
        _rubric.resolve_judgements([row])
    except ValueError as exc:
        return str(exc)
    return None


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
    verdicts.sort(key=_verdict_order)
    return Assembly(
        verdicts=verdicts,
        derivation={
            'check': {
                'min_compared': DERIVATION_MIN_COMPARED,
                'min_agreement': DERIVATION_MIN_AGREEMENT,
                'roles': sorted(map(str, _CHECKED_ROLES)),
            },
            'rows': {name: {**asdict(row_check), 'binary': str(row_check.binary)}
                     for name, row_check in check.items()},
            'held_rows': sorted(held),
            'unreached_follow_ups': _unreached_follow_ups(verdicts),
        },
    )


def _unreached_follow_ups(verdicts: Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Each (pair key, follow-up) the rubric resolves UNCLEAR because a derived row left it unasked."""
    unreached = sorted(
        (key, binary)
        for key, judgement in _rubric.resolve_judgements(verdicts).items()
        for binary in judgement.unreached_follow_ups
    )
    return [
        {'entry_id': key.entry_id, 'target_id': key.target_id, 'basis': str(key.basis),
         'binary': str(binary)}
        for key, binary in unreached
    ]


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


class _Unrecoverable(ValueError):
    """A drawn attach whose child id or current parent text no read-only source holds."""


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


@dataclass(frozen=True)
class _SoakBundle:
    """The read-only soak bundle: the triage attaches, their parents and hoist flags, and the v1 adjudications.

    Every map is keyed by the attach's intent id, except *parents*, keyed by parent id.
    """

    attaches: Mapping[str, Mapping[str, Any]]
    adjudications: Mapping[str, Mapping[str, Any]]
    parents: Mapping[str, Sequence[Mapping[str, Any]]]
    hoist: Mapping[str, Mapping[str, Any]]
    edited: frozenset[str]

    @classmethod
    def read(cls, directory: Path) -> _SoakBundle:
        adjudications = [
            json.loads(line) for line in (directory / 'adjudications.jsonl').read_text().splitlines()
            if line.strip()
        ]
        return cls(
            attaches={row['intent_id']: row for row in _read_json(directory / 'attaches_soak.json')
                      if row.get('triage_sig')},
            adjudications={row['child_intent_id']: row for row in adjudications},
            parents=_read_json(directory / 'parents.json'),
            hoist=_read_json(directory / 'hoist.json'),
            edited=frozenset(_read_json(directory / 'parent_edited_after_attach.json')),
        )

    def population(self, outcome: str) -> frozenset[str]:
        return frozenset(intent for intent, row in self.attaches.items() if row['outcome'] == outcome)

    def item(self, intent: str, child_memory_id: str) -> _SoakItem:
        """The one way a soak item is built, adjudicated or extension, against its parent's current text."""
        attach = self.attaches[intent]
        parent_id = attach['meta']['parent_id']
        hits = self.parents.get(parent_id) or []
        payload = hits[0]['payload'] if hits else {}
        if not payload.get('data'):
            raise _Unrecoverable(f'the bundle holds no current text for parent {parent_id}')
        hoisted = self.hoist.get(intent, {})
        return _SoakItem(
            item=item_row(
                attach, child_memory_id=child_memory_id, high_band=hoisted.get('high_band'),
                via_sibling=hoisted.get('likely_via_sibling'),
                parent_category=payload.get('category'),
                parent_edited_after_attach=intent in self.edited,
            ),
            adjudication=self.adjudications.get(intent),
            child_text=attach['content'],
            parent_text=payload['data'],
            child_created_at=attach['created_at'],
        )

    def keyed_items(self, key: Mapping[str, Any]) -> list[_SoakItem]:
        """Every adjudicated item and every recoverable extension item *key* records, by entry id."""
        unrecoverable = {row['intent_id'] for row in key['unrecoverable']}
        children = {intent: row['child_memory_id'] for intent, row in self.adjudications.items()}
        children |= {
            intent: record['child_memory_id'] for intent, record in key['extension'].items()
            if intent not in unrecoverable
        }
        soak = [self.item(intent, child_memory_id) for intent, child_memory_id in children.items()]
        return sorted(soak, key=lambda soaked: soaked.item['entry_id'])


def _journal_child_id(journal: sqlite3.Connection, write_op_id: str) -> str:
    row = journal.execute('SELECT result_summary FROM write_ops WHERE id = ?',
                          (write_op_id,)).fetchone()
    if row is None:
        raise _Unrecoverable(f'the write journal holds no write op {write_op_id}')
    memory_ids = json.loads(row[0] or '{}').get('memory_ids') or []
    if len(memory_ids) != 1:
        raise _Unrecoverable(f'write op {write_op_id} names {len(memory_ids)} memory ids, not one')
    return memory_ids[0]


def _draw_extension(
    bundle: _SoakBundle, totals: Mapping[str, int], seed: int, journal_db: Path,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    """Each drawn intent id's outcome and child id, and the drawn attaches no source rebuilds.

    An unrecoverable attach is recorded with its reason, never replaced by another draw.
    """
    extension: dict[str, dict[str, Any]] = {}
    unrecoverable: list[dict[str, str]] = []
    with _read_only(journal_db) as journal:
        for outcome, total in totals.items():
            drawn = extend_sample(bundle.population(outcome), bundle.adjudications, total,
                                  seed=f'{seed}:{outcome}')
            for intent in drawn:
                child_memory_id: str | None = None
                try:
                    child_memory_id = _journal_child_id(journal, bundle.attaches[intent]['write_op_id'])
                    bundle.item(intent, child_memory_id)
                except _Unrecoverable as exc:
                    unrecoverable.append({'intent_id': intent, 'outcome': outcome, 'reason': str(exc)})
                extension[intent] = {'outcome': outcome, 'child_memory_id': child_memory_id}
    return extension, sorted(unrecoverable, key=lambda row: row['intent_id'])


def _rating_entries(soak: Sequence[_SoakItem], history_db: Path) -> tuple[list[ShardEntry], list[str]]:
    """Each item's current pair, its attach-time pair when its parent was edited after attach,
    and the entries whose attach-time text history does not keep."""
    entries = [soaked.entry(Basis.CURRENT) for soaked in soak]
    unrecoverable = []
    for soaked in soak:
        if not soaked.item['parent_edited_after_attach']:
            continue
        text = attach_time_text(history_db, soaked.item['target_id'],
                                soaked.child_created_at, soaked.parent_text)
        if text is None:
            unrecoverable.append(soaked.item['entry_id'])
        else:
            entries.append(soaked.entry(Basis.ATTACH_TIME, text))
    return entries, sorted(unrecoverable)


def _refuse_inside(parser: argparse.ArgumentParser, scratch: Path, bundle: Path) -> None:
    scratch, bundle = scratch.resolve(), bundle.resolve()
    if scratch == bundle or bundle in scratch.parents:
        parser.error(f'--scratch {scratch} is inside the read-only --bundle {bundle}')


def _keyed_order(item: Mapping[str, str]) -> tuple[str, str, str]:
    """A key item's pair and basis, ordered as :func:`_entry_order` orders a shard entry."""
    return (item['entry_id'], item['target_id'], item['basis'])


_SHARD_PREFIX: Mapping[str, str] = MappingProxyType({Basis.CURRENT: '', Basis.ATTACH_TIME: 'attach_'})
_DOUBLE_RATED = (Role.PRIMARY, Role.SECOND)


class _Scratch:
    """The scratch directory: shards, the key, the texts raters saw, the rater prompt and the ratings."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.key_path = root / 'key.json'

    def load_key(self) -> dict[str, Any]:
        if not self.key_path.exists():
            return {'shards': {}, 'items': {}, 'extension': {}, 'unrecoverable': [],
                    'attach_time_unrecoverable': []}
        return _read_json(self.key_path)

    def add_shards(self, key: dict[str, Any], entries: Sequence[ShardEntry], *, seed: int) -> list[str]:
        """Shard each entry not yet keyed under each double-rating role; the new shards' names."""
        keyed = {
            (spec['role'], _keyed_order(key['items'][item]))
            for spec in key['shards'].values() for item in spec['items']
        }
        added = []
        for basis, prefix in _SHARD_PREFIX.items():
            for role in _DOUBLE_RATED:
                pending = [entry for entry in entries
                           if entry.basis == basis and (role, _entry_order(entry)) not in keyed]
                first = sum(name.startswith(f'{prefix}{role}_') for name in key['shards'])
                for name, members in plan_shards(pending, role=role, prefix=prefix, seed=seed,
                                                 first_index=first):
                    self.write_shard(key, name, members, role=role,
                                     rater=f'{RATER_MODELS[role]}-{role}', round=0)
                    added.append(name)
        return added

    def write_shard(self, key: dict[str, Any], name: str, entries: Sequence[ShardEntry], *,
                    role: str, rater: str, round: int) -> None:
        rendered = shard(entries, name=name)
        (self.root / 'shards').mkdir(parents=True, exist_ok=True)
        (self.root / 'shards' / f'{name}.txt').write_text(rendered.text)
        key['shards'][name] = {'role': role, 'rater': rater, 'round': round,
                               'items': sorted(rendered.key)}
        key['items'].update({item_id: dict(pair) for item_id, pair in rendered.key.items()})

    def save_key(self, key: Mapping[str, Any]) -> None:
        _write_json(self.key_path, key)

    def save_texts(self, entries: Iterable[ShardEntry]) -> None:
        _write_json(self.root / 'texts.json', [
            {'entry_id': entry.entry_id, 'target_id': entry.target_id, 'basis': entry.basis,
             'child_text': entry.child_text, 'parent_text': entry.parent_text}
            for entry in sorted(entries, key=_entry_order)
        ])

    def entries(self, soak: Iterable[_SoakItem]) -> dict[tuple[str, str, str], ShardEntry]:
        """Each keyed pair and basis as its raters saw it, from ``texts.json``."""
        projects = {soaked.item['entry_id']: soaked.item['project'] for soaked in soak}
        return {
            _keyed_order(record): ShardEntry(**record, project=projects[record['entry_id']])
            for record in _read_json(self.root / 'texts.json')
        }

    def ratings(self, key: Mapping[str, Any], entries: Mapping[tuple[str, str, str], ShardEntry],
                ) -> _Rated:
        """Every rater's effective lines, each checked by :func:`rating_problems`."""
        lines: list[dict[str, Any]] = []
        for name, spec in sorted(key['shards'].items()):
            lines += self._shard_ratings(name, spec, key['items'])
        rows = effective_ratings(lines)
        item_ids = {
            (name, *_keyed_order(key['items'][item_id])): item_id
            for name, spec in key['shards'].items() for item_id in spec['items']
        }
        reask = []
        for row in rows:
            entry = entries[_keyed_order(row)]
            if problems := rating_problems(row, entry.child_text, entry.parent_text):
                item_id = item_ids[row['batch'], *_keyed_order(row)]
                reask.append(_Reask(row, item_id, tuple(problems)))
        return _Rated(rows=rows, reasked=len(lines) - len(rows), reask=reask)

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
            rows.append({**row, 'round': spec['round']})
        wrong = sorted(item for item in spec['items'] if seen[item] != 1)
        if wrong:
            raise ValueError(f'{path}: items without exactly one line: {wrong}')
        return rows


@dataclass(frozen=True)
class _Reask:
    """An effective rater line to ask again, and why."""

    row: Mapping[str, Any]
    item_id: str
    problems: tuple[str, ...]

    def __str__(self) -> str:
        return f'{self.row["batch"]} item {self.item_id}: {"; ".join(self.problems)}'


@dataclass(frozen=True)
class _Rated:
    """The effective rater lines, how many earlier-round lines they superseded, and those to ask again."""

    rows: list[dict[str, Any]]
    reasked: int
    reask: list[_Reask]


def _shards_command(args: argparse.Namespace) -> dict[str, Any]:
    bundle = _SoakBundle.read(args.bundle)
    scratch = _Scratch(args.scratch)
    key = scratch.load_key()
    totals = {OUTCOME_RESTATED: args.restated_total, OUTCOME_AMENDED: args.amended_total}
    extension, unrecoverable = _draw_extension(bundle, totals, args.seed, args.journal_db)
    dropped = sorted(set(key['extension']) - set(extension))
    if dropped:
        raise ValueError(f'this extension drops intent ids already keyed: {dropped};'
                         ' re-run with totals at least as large as before')
    key['extension'], key['unrecoverable'] = extension, unrecoverable
    soak = bundle.keyed_items(key)
    entries, key['attach_time_unrecoverable'] = _rating_entries(soak, args.history_db)
    added = scratch.add_shards(key, entries, seed=args.seed)
    scratch.save_key(key)
    scratch.save_texts(entries)
    (args.scratch / 'rater_prompt.md').write_text(_rater_prompt())
    return {
        'items': len(soak),
        'extension': Counter(record['outcome'] for record in extension.values()),
        'unrecoverable': len(unrecoverable),
        'attach_time': sum(entry.basis == Basis.ATTACH_TIME for entry in entries),
        'attach_time_unrecoverable': key['attach_time_unrecoverable'],
        'new_shards': {name: len(key['shards'][name]['items']) for name in added},
    }


@dataclass(frozen=True)
class _Corpus:
    """The scratch state of a fully rated soak, and its assembled verdict corpus."""

    scratch: _Scratch
    key: dict[str, Any]
    bundle: _SoakBundle
    soak: list[_SoakItem]
    entries: Mapping[tuple[str, str, str], ShardEntry]
    rated: _Rated
    assembly: Assembly


def _corpus(args: argparse.Namespace) -> _Corpus:
    """The assembled corpus, refused while any line awaits a re-ask or any pair lacks a rater."""
    bundle = _SoakBundle.read(args.bundle)
    scratch = _Scratch(args.scratch)
    key = scratch.load_key()
    soak = bundle.keyed_items(key)
    entries = scratch.entries(soak)
    rated = scratch.ratings(key, entries)
    if rated.reask:
        raise ValueError('rater lines break the row contract or quote evidence not in the texts'
                         ' shown; run reask:\n' + '\n'.join(map(str, rated.reask)))
    rated_roles = {(*_keyed_order(row), row['role']) for row in rated.rows}
    lacking = sorted(
        (*keyed, role) for keyed in entries for role in _DOUBLE_RATED
        if (*keyed, role) not in rated_roles
    )
    if lacking:
        raise ValueError(f'pairs and bases without an effective primary or second line: {lacking}')
    assembly = assemble(
        {soaked.pair: soaked.adjudication for soaked in soak}, rated.rows,
        {soaked.item['entry_id']: soaked.child_text for soaked in soak},
    )
    return _Corpus(scratch, key, bundle, soak, entries, rated, assembly)


def _reask_command(args: argparse.Namespace) -> dict[str, Any]:
    scratch = _Scratch(args.scratch)
    key = scratch.load_key()
    entries = scratch.entries(_SoakBundle.read(args.bundle).keyed_items(key))
    reask_by_rater: defaultdict[str, list[_Reask]] = defaultdict(list)
    for reask in scratch.ratings(key, entries).reask:
        reask_by_rater[reask.row['rater']].append(reask)
    last_round = max(spec['round'] for spec in key['shards'].values())
    written = {}
    for round_, (rater, lines) in enumerate(sorted(reask_by_rater.items()), start=last_round + 1):
        role = lines[0].row['role']
        name = f'reask_{round_:02d}_{role}'
        scratch.write_shard(key, name, [entries[_keyed_order(line.row)] for line in lines],
                            role=role, rater=rater, round=round_)
        written[name] = len(lines)
    if written:
        scratch.save_key(key)
    return {'lines': sum(written.values()), 'shards': written}


def _third_command(args: argparse.Namespace) -> dict[str, Any]:
    corpus = _corpus(args)
    blocked = set(third_round_pairs([soaked.item for soaked in corpus.soak],
                                    corpus.assembly.verdicts))
    index = sum(spec['role'] == Role.THIRD for spec in corpus.key['shards'].values())
    name = f'third_{index:02d}'
    entries = [entry for (entry_id, target_id, basis), entry in sorted(corpus.entries.items())
               if basis == Basis.CURRENT and (entry_id, target_id) in blocked]
    if entries:
        corpus.scratch.write_shard(corpus.key, name, entries, role=Role.THIRD,
                                   rater=f'{RATER_MODELS[Role.THIRD]}-third-{index:02d}', round=0)
        corpus.scratch.save_key(corpus.key)
    return {'pairs': len(entries), 'shard': name if entries else None}


def _assemble_command(args: argparse.Namespace) -> dict[str, Any]:
    corpus = _corpus(args)
    key = corpus.key
    out: Path = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / 'write_triage_v2_soak_items.jsonl', [soaked.item for soaked in corpus.soak])
    _write_jsonl(out / 'write_triage_v2_soak_verdicts.jsonl', corpus.assembly.verdicts)
    _write_json(out / 'write_triage_v2_soak_strata.json', dict(STRATA))
    _write_json(out / 'write_triage_v2_soak_derivation.json', {
        **corpus.assembly.derivation,
        'rating_volume': rating_volume(key),
        'sample': _sample(corpus.bundle, key),
        'unrecoverable': key['unrecoverable'],
        'attach_time_unrecoverable': sorted(key['attach_time_unrecoverable']),
        'reasked': corpus.rated.reasked,
        'shards': {name: {'rater': spec['rater'], 'role': spec['role'], 'round': spec['round'],
                          'items': len(spec['items'])}
                   for name, spec in sorted(key['shards'].items())},
    })
    return {'items': len(corpus.soak), 'verdicts': len(corpus.assembly.verdicts)}


def rating_volume(key: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    """The distinct (entry, target) pairs the scratch *key*'s shards put before each role, by basis."""
    pairs: defaultdict[str, defaultdict[str, set[tuple[str, str]]]] = defaultdict(
        lambda: defaultdict(set))
    for spec in key['shards'].values():
        for item_id in spec['items']:
            item = key['items'][item_id]
            pairs[item['basis']][spec['role']].add((item['entry_id'], item['target_id']))
    return {basis: {role: len(rated) for role, rated in sorted(by_role.items())}
            for basis, by_role in sorted(pairs.items())}


_SAMPLE_COUNTS = ('adjudicated', 'extension', 'unrecoverable')


def _sample(bundle: _SoakBundle, key: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    """Per stratum: the adjudicated items, the recovered extension items and the unrecoverable draws."""
    counts = {stratum: Counter[str]() for stratum in STRATA}
    for adjudication in bundle.adjudications.values():
        counts[STRATUM_BY_OUTCOME[adjudication['outcome']]]['adjudicated'] += 1
    unrecoverable = {row['intent_id'] for row in key['unrecoverable']}
    for intent, record in key['extension'].items():
        kind = 'unrecoverable' if intent in unrecoverable else 'extension'
        counts[STRATUM_BY_OUTCOME[record['outcome']]][kind] += 1
    return {stratum: {kind: counted[kind] for kind in _SAMPLE_COUNTS}
            for stratum, counted in counts.items()}


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
        ' and shows PARENT and CHILD. Write one JSON object per line, in item order, to the'
        ' output file you are given:'
        ' `{"id": "<item id>", "home": "YES|NO|UNCLEAR", "conflict": "YES|NO|UNCLEAR",'
        ' "scope": "CENTRAL|SECONDARY|UNCLEAR", "action_changing": "YES|NO|UNCLEAR",'
        ' "conflict_quote": {"parent": "<words from PARENT>", "child": "<words from CHILD>"},'
        ' "loss": "YES|NO|UNCLEAR", "loss_spans": ["<words from CHILD>", ...],'
        ' "unclear_reason": "...", "reason": "<one sentence naming both claims>"}`.',
        '',
        '- Give `scope`, `action_changing` and `conflict_quote` when, and only when,'
        ' `conflict` is YES.',
        '- Give `loss` when, and only when, `home` is YES and `conflict` is NO; give'
        ' `loss_spans` when, and only when, `loss` is YES.',
        '- Give `unclear_reason` with any UNCLEAR answer.',
        '- Copy `conflict_quote.parent` verbatim from PARENT, and `conflict_quote.child` and'
        ' every `loss_spans` entry verbatim from CHILD. A line that breaks these rules, or'
        ' whose quote or span is not found in the text shown, is sent back to you to answer'
        ' again.',
        '',
        'Append to the file after every batch of about 10 items, so partial progress'
        ' survives. Every item in the shard gets exactly one line. When done, reply with'
        ' only the number of lines written.',
        '',
    ])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('shards', 'reask', 'third', 'assemble'):
        command = commands.add_parser(name)
        command.add_argument('--bundle', type=Path, required=True, metavar='DIR',
                             help="the soak bundle's correctness/ directory (read-only)")
        command.add_argument('--scratch', type=Path, required=True, metavar='DIR')
    shards = commands.choices['shards']
    shards.add_argument('--history-db', type=Path, required=True, metavar='PATH',
                        help="mem0's history.db (read-only)")
    shards.add_argument('--journal-db', type=Path, required=True, metavar='PATH',
                        help="the reconciliation write journal (read-only)")
    shards.add_argument('--seed', type=int, default=7174)
    shards.add_argument('--restated-total', type=int, default=SAMPLE_TOTAL, metavar='N',
                        help='how many restated attaches the extended sample holds')
    shards.add_argument('--amended-total', type=int, default=SAMPLE_TOTAL, metavar='N',
                        help='how many amended attaches the extended sample holds')
    commands.choices['assemble'].add_argument('--out-dir', type=Path, required=True, metavar='DIR')
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _refuse_inside(parser, args.scratch, args.bundle)
    if args.command == 'shards':
        args.history_db = args.history_db.expanduser()
        args.journal_db = args.journal_db.expanduser()
        summary = _shards_command(args)
    elif args.command == 'reask':
        summary = _reask_command(args)
    elif args.command == 'third':
        summary = _third_command(args)
    else:
        summary = _assemble_command(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
