#!/usr/bin/env python3
"""The blind, double-rated rating rounds of rubric v2.1.

The protocol is ``calibration/write_triage_reader_metrics.md`` §4. Each pair
to rate is a :class:`ShardEntry`. It is dealt into one primary and one second
shard of opaque item ids, and each shard is read by a blind Opus rater given
the brief verbatim. A line that breaks the rubric's row contract, or quotes
evidence not in the texts its rater saw, is asked again in a later round,
which supersedes it. A pair that a decisive tie or an open follow-up blocks
goes to a third round, under a new rater.

A :class:`RatingScratch` directory holds the rounds:

- ``key.json``: each shard's role, rater, round and item ids, and each item
  id's pair and basis. It is the only map from an item back to its pair.
- ``texts.json``: each pair and basis as its raters saw it.
- ``shards/<name>.txt``: what a rater reads, re-renderable from the key and the texts.
- ``ratings/<name>.jsonl``: the rater's answers, one line per item.

Callers choose which pairs to rate and what the effective lines mean; this
module imports only the rubric.
"""
from __future__ import annotations

import json
import math
import random
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

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
RATER_MODELS: Mapping[str, str] = MappingProxyType({
    Role.PRIMARY: 'opus', Role.SECOND: 'opus', Role.THIRD: 'opus',
})
DOUBLE_RATED = (Role.PRIMARY, Role.SECOND)

Triple = tuple[str, str, str]
"""A pair and basis: (entry id, target id, basis)."""


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


def _entry_order(entry: ShardEntry) -> Triple:
    return (entry.entry_id, entry.target_id, entry.basis)


def _keyed_order(item: Mapping[str, str]) -> Triple:
    """A key item's or a row's pair and basis, ordered as :func:`_entry_order` orders an entry."""
    return (item['entry_id'], item['target_id'], item['basis'])


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
    """Render *entries*, in order, each text capped at :data:`TEXT_CAP`."""
    lines: list[str] = []
    key: dict[str, Mapping[str, str]] = {}
    for index, entry in enumerate(entries):
        item_id = f'{name}-{index:03d}'
        key[item_id] = MappingProxyType({
            'entry_id': entry.entry_id, 'target_id': entry.target_id, 'basis': entry.basis,
        })
        lines += [
            f'=== ITEM id={item_id} | project={entry.project} | basis={entry.basis}',
            '--- PARENT:', capped(entry.parent_text),
            '--- CHILD:', capped(entry.child_text),
            '',
        ]
    return Shard(text='\n'.join(lines) + '\n', key=MappingProxyType(key))


_CUT = '…[truncated, '
_CUT_END = ' chars total]'


def capped(text: str) -> str:
    """*text* cut at :data:`TEXT_CAP` and marked with its full length.

    A text this function already cut is returned unchanged, so a text stored
    capped still shows a rater its original length.
    """
    if len(text) <= TEXT_CAP or _is_cut(text[TEXT_CAP:]):
        return text
    return f'{text[:TEXT_CAP]}{_CUT}{len(text)}{_CUT_END}'


def _is_cut(tail: str) -> bool:
    count = tail.removeprefix(_CUT).removesuffix(_CUT_END)
    return tail.startswith(_CUT) and tail.endswith(_CUT_END) and count.isdigit()


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


def with_digest(row: Mapping[str, Any], child_text: str) -> dict[str, Any]:
    """*row*, plus DIGEST from ``digest_visible`` of its loss spans when it answers LOSS YES (R3)."""
    if row.get(Binary.LOSS) != Answer.YES:
        return dict(row)
    visible = _rubric.digest_visible(row['loss_spans'], child_text)
    return {**row, str(Binary.DIGEST): str(Answer.YES if visible else Answer.NO)}


_THIRD_ROUND_GAPS = frozenset({_rubric.GapKind.TIED, _rubric.GapKind.OPEN})
"""The gaps a third rater answering the whole item can fill: a tie, or an open follow-up."""


def third_round_blockers(derivations: Mapping[str, _rubric.Derivation]) -> list[tuple[str, str]]:
    """The sorted (entry, target) pairs whose derivation only a third rater can unblock.

    *derivations* maps each entry to its derivation. An Incomplete one blocks
    on the target of each of its gaps, provided every gap is a decisive tie or
    an open follow-up; an unrated gap is refused, naming the entry.
    """
    blocked: set[tuple[str, str]] = set()
    for entry_id, derivation in derivations.items():
        if not isinstance(derivation, _rubric.Incomplete):
            continue
        unrated = [str(gap.binary) for gap in derivation.missing if gap.kind not in _THIRD_ROUND_GAPS]
        if unrated:
            raise ValueError(
                f'entry {entry_id!r} has unrated decisive binaries {unrated};'
                ' a third rater fills only ties and open follow-ups'
            )
        blocked.update((entry_id, gap.target_id) for gap in derivation.missing)
    return sorted(blocked)


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


def read_only(database: Path) -> closing[sqlite3.Connection]:
    """A read-only connection to *database*, with temp storage kept in memory."""
    connection = sqlite3.connect(f'{Path(database).resolve().as_uri()}?mode=ro', uri=True)
    connection.execute('PRAGMA temp_store=MEMORY')
    return closing(connection)


class MemoryHistory:
    """The UPDATE events of mem0's history table, read once from the read-only database."""

    def __init__(
        self, updates: Mapping[str, Sequence[tuple[datetime, str | None]]],
        covers: frozenset[str] | None,
    ) -> None:
        self._updates = updates
        self._covers = covers

    @classmethod
    def read(cls, history_db: Path, memory_ids: Iterable[str] | None = None) -> MemoryHistory:
        """Every memory's UPDATE events, or only those of *memory_ids*, which it then alone answers for."""
        covers = None if memory_ids is None else frozenset(memory_ids)
        updates: defaultdict[str, list[tuple[datetime, str | None]]] = defaultdict(list)
        with read_only(history_db) as connection:
            rows = connection.execute(
                "SELECT memory_id, old_memory, updated_at FROM history WHERE event = 'UPDATE'"
            ).fetchall()
        for memory_id, old, at in rows:
            if at and (covers is None or memory_id in covers):
                updates[memory_id].append((datetime.fromisoformat(at), old))
        return cls(updates, covers)

    def text_at(self, memory_id: str, instant: datetime, current_text: str) -> str | None:
        """A memory's text at *instant*.

        The old text of the earliest UPDATE after *instant*; *current_text* when
        the memory was not edited after it; None when that UPDATE kept no old text.
        """
        later = self._edits_after(memory_id, instant)
        if not later:
            return current_text
        return min(later, key=lambda edit: edit[0])[1]

    def edited_after(self, memory_id: str, instant: datetime) -> bool:
        """Whether an UPDATE of the memory follows *instant*."""
        return bool(self._edits_after(memory_id, instant))

    def _edits_after(self, memory_id: str, instant: datetime) -> list[tuple[datetime, str | None]]:
        if self._covers is not None and memory_id not in self._covers:
            raise ValueError(f'this history was read for {len(self._covers)} memories, not {memory_id!r}')
        return [(at, old) for at, old in self._updates.get(memory_id, ()) if at > instant]


def text_at(history_db: Path, memory_id: str, instant: datetime, current_text: str) -> str | None:
    """:meth:`MemoryHistory.text_at` for one memory, read from *history_db*."""
    return MemoryHistory.read(history_db, [memory_id]).text_at(memory_id, instant, current_text)


_SHARD_PREFIX: Mapping[str, str] = MappingProxyType({Basis.CURRENT: '', Basis.ATTACH_TIME: 'attach_'})


@dataclass(frozen=True)
class Reask:
    """An effective rater line to ask again, and why."""

    row: Mapping[str, Any]
    item_id: str
    problems: tuple[str, ...]

    def __str__(self) -> str:
        return f'{self.row["batch"]} item {self.item_id}: {"; ".join(self.problems)}'


@dataclass(frozen=True)
class Rated:
    """The effective rater lines, how many earlier-round lines they superseded, and those to ask again."""

    rows: list[dict[str, Any]]
    reasked: int
    reask: list[Reask]


class RatingScratch:
    """The scratch directory of one set of rating rounds; see the module docstring for its layout."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.key_path = root / 'key.json'

    def load_key(self) -> dict[str, Any]:
        if not self.key_path.exists():
            return {'shards': {}, 'items': {}}
        return json.loads(self.key_path.read_text())

    def save_key(self, key: Mapping[str, Any]) -> None:
        _write_json(self.key_path, key)

    def add_shards(self, key: dict[str, Any], entries: Sequence[ShardEntry], *, seed: int) -> list[str]:
        """Shard each entry not yet keyed under each double-rating role; the new shards' names.

        Attach-time entries get shards of their own, prefixed ``attach_``.
        """
        keyed = {
            (spec['role'], _keyed_order(key['items'][item]))
            for spec in key['shards'].values() for item in spec['items']
        }
        added = []
        for basis, prefix in _SHARD_PREFIX.items():
            for role in DOUBLE_RATED:
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
        """Render *entries* as shard *name* and record it in *key*; the caller saves the key."""
        rendered = shard(entries, name=name)
        self._shard_path(name).parent.mkdir(parents=True, exist_ok=True)
        self._shard_path(name).write_text(rendered.text)
        key['shards'][name] = {'role': role, 'rater': rater, 'round': round,
                               'items': sorted(rendered.key)}
        key['items'].update({item_id: dict(pair) for item_id, pair in rendered.key.items()})

    def render_missing(self, key: Mapping[str, Any], entries: Mapping[Triple, ShardEntry]) -> list[str]:
        """Render again, byte for byte, each keyed shard whose text file is missing; their names."""
        rendered = []
        for name, spec in sorted(key['shards'].items()):
            if self._shard_path(name).exists():
                continue
            members = []
            for item_id in spec['items']:
                triple = _keyed_order(key['items'][item_id])
                if triple not in entries:
                    raise ValueError(f'shard {name}: no text for its item {item_id} {triple}')
                members.append(entries[triple])
            self._shard_path(name).parent.mkdir(parents=True, exist_ok=True)
            self._shard_path(name).write_text(shard(members, name=name).text)
            rendered.append(name)
        return rendered

    def save_texts(self, entries: Iterable[ShardEntry]) -> None:
        _write_json(self.root / 'texts.json', [
            {'entry_id': entry.entry_id, 'target_id': entry.target_id, 'basis': entry.basis,
             'child_text': entry.child_text, 'parent_text': entry.parent_text}
            for entry in sorted(entries, key=_entry_order)
        ])

    def entries(self, projects: Mapping[str, str]) -> dict[Triple, ShardEntry]:
        """Each pair and basis as its raters saw it, from ``texts.json``; *projects* maps entry to project."""
        return {
            _keyed_order(record): ShardEntry(**record, project=projects[record['entry_id']])
            for record in json.loads((self.root / 'texts.json').read_text())
        }

    def ratings(self, key: Mapping[str, Any], entries: Mapping[Triple, ShardEntry]) -> Rated:
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
                reask.append(Reask(row, item_id, tuple(problems)))
        return Rated(rows=rows, reasked=len(lines) - len(rows), reask=reask)

    def _shard_path(self, name: str) -> Path:
        return self.root / 'shards' / f'{name}.txt'

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


def reask_round(
    scratch: RatingScratch, key: dict[str, Any], entries: Mapping[Triple, ShardEntry],
) -> dict[str, Any]:
    """Shard every line to ask again, one shard per rater in a new round each; saves the key.

    Returns the lines written and each new shard's size.
    """
    reask_by_rater: defaultdict[str, list[Reask]] = defaultdict(list)
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


def third_round(
    scratch: RatingScratch, key: dict[str, Any], entries: Mapping[Triple, ShardEntry],
    blocked: Iterable[tuple[str, str]],
) -> dict[str, Any]:
    """Shard the current-basis texts of the *blocked* (entry, target) pairs for a new third rater.

    Saves the key when a shard is written. Returns the pairs sharded and the shard's name, or None.
    """
    pairs = set(blocked)
    index = sum(spec['role'] == Role.THIRD for spec in key['shards'].values())
    name = f'third_{index:02d}'
    members = [entry for (entry_id, target_id, basis), entry in sorted(entries.items())
               if basis == Basis.CURRENT and (entry_id, target_id) in pairs]
    if members:
        scratch.write_shard(key, name, members, role=Role.THIRD,
                            rater=f'{RATER_MODELS[Role.THIRD]}-third-{index:02d}', round=0)
        scratch.save_key(key)
    return {'pairs': len(members), 'shard': name if members else None}


def refuse_unfinished(rated: Rated, entries: Iterable[Triple]) -> None:
    """Refuse while any line awaits a re-ask, or any pair and basis lacks a primary or second line."""
    if rated.reask:
        raise ValueError('rater lines break the row contract or quote evidence not in the texts'
                         ' shown; run reask:\n' + '\n'.join(map(str, rated.reask)))
    rated_roles = {(*_keyed_order(row), row['role']) for row in rated.rows}
    lacking = sorted(
        (*keyed, role) for keyed in entries for role in DOUBLE_RATED
        if (*keyed, role) not in rated_roles
    )
    if lacking:
        raise ValueError(f'pairs and bases without an effective primary or second line: {lacking}')


def answer_contract() -> str:
    """The brief verbatim and the answer it asks for: what every rater, file or API, is given."""
    return '\n'.join([
        '## The rubric (verbatim, fused-memory/calibration/write_triage_rater_brief_v2.md)',
        '',
        BRIEF_PATH.read_text(),
        '## Output',
        '',
        'Each item starts with `=== ITEM id=<id> | project=<project> | basis=<basis>`'
        ' and shows PARENT and CHILD. Answer each item with one JSON object:'
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
        ' every `loss_spans` entry verbatim from CHILD. An answer that breaks these rules,'
        ' or whose quote or span is not found in the text shown, is asked again.',
        '',
    ])


def rater_prompt() -> str:
    """The prompt a blind rater agent gets for one shard file and one output file."""
    return '\n'.join([
        '# Rater task (read one shard file, write one JSONL file)',
        '',
        'You are a blind rater. Read ONLY the shard file you are given and write ONLY the'
        ' output file you are given. Do not open any other file (no key file, no repository'
        ' file, no store), and do not call any memory or task tool.',
        '',
        answer_contract(),
        '## Writing the output file',
        '',
        'Write one JSON object per line, in item order, to the output file you are given.'
        ' Append to the file after every batch of about 10 items, so partial progress'
        ' survives. Every item in the shard gets exactly one line. When done, reply with'
        ' only the number of lines written.',
        '',
    ])


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
