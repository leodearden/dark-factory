#!/usr/bin/env python3
"""Build the fully labelled rubric v2.1 write-triage eval set.

Every (write, target) pair production's judge would be shown at width
:data:`EVAL_WIDTH` is rated blind by two Opus raters under the round
protocol of ``write_triage_rating_rounds.py``. The pairs come from two sources:

- offline: the write-time draw of the frozen 2026-10-05 population
  (``run_write_triage_population_arms.py::draw_run_set``), less the
  reconciliation markers and the writes declaring attach keys, which
  production never judges;
- live: the soak's judge band, rebuilt by ``rebuild_write_triage_live_slates.py``.

A pair is fully labelled when its :data:`FULL_LABEL_OUTCOME` derivation is
not Incomplete; ``calibration/write_triage_eval_set_v2.md`` says why. Every
pair is rated on the ``current`` basis: the text the judge was shown.

Subcommands, run from ``fused-memory/``. Each takes ``--snapshot`` (read-only,
the main checkout's frozen population), ``--live-slates`` and ``--scratch``:

- ``shards``: key every pair not yet keyed into one primary and one second
  shard, re-render any shard text missing from the scratch, and write
  ``texts.json``, ``rater_prompt.md`` and the scratch's ``.gitignore``.
- ``reask``: shard every rater line that breaks the row contract or quotes
  evidence not in the texts shown; the later round supersedes it.
- ``third``: shard the pairs a decisive tie or an open follow-up blocks, for a
  new third rater.
- ``assemble``: write ``write_triage_eval_set_v2_{writes.jsonl,verdicts.jsonl,summary.json}``
  into ``--out-dir``, refusing while any pair is not fully labelled.

``split --eval-dir DIR`` freezes the assembled set's held-out split
(:func:`freeze_split`) into ``write_triage_eval_set_v2_split.json`` beside it,
and refuses to overwrite a frozen split without ``--refreeze``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fused_memory.server.write_triage import OUTCOME_RESTATED, OUTCOME_STORED
from fused_memory.utils.script_loader import load_script

_SCRIPTS = Path(__file__).resolve().parent

if TYPE_CHECKING:
    import freeze_write_triage_population as _freeze
    import run_write_triage_population_arms as _arms
    import score_write_triage_reader as _score
    import write_triage_rating_rounds as _rounds
    import write_triage_reader_rubric as _rubric
else:
    _rubric = load_script(_SCRIPTS / 'write_triage_reader_rubric.py', 'write_triage_reader_rubric')
    _rounds = load_script(_SCRIPTS / 'write_triage_rating_rounds.py', 'write_triage_rating_rounds')
    _score = load_script(_SCRIPTS / 'score_write_triage_reader.py', 'score_write_triage_reader')
    _freeze = load_script(_SCRIPTS / 'freeze_write_triage_population.py',
                          'freeze_write_triage_population')
    _arms = load_script(_SCRIPTS / 'run_write_triage_population_arms.py',
                        'run_write_triage_population_arms')

Basis = _rubric.Basis

FULL_LABEL_OUTCOME = OUTCOME_RESTATED
"""The outcome whose derivation of a pair reads every binary a judge outcome reads (see the eval-set doc)."""
EVAL_WIDTH = 5
OFFLINE_MAX_WRITES = 658
OFFLINE_STRATUM = 'offline-write-time'
SOURCE_OFFLINE = 'offline'
SOURCE_LIVE = 'live'
SOAK_BUNDLE_NAME = 'write-triage-soak-2026-10-19'
"""The read-only soak bundle, beside the snapshot's directory in the main checkout's data/."""

_CALIBRATION = _SCRIPTS.parent / 'calibration'
OUT_STEM = 'write_triage_eval_set_v2'
DEFAULT_LIVE_SLATES = _CALIBRATION / f'{OUT_STEM}_live_slates.jsonl'
DEFAULT_SCRATCH = _CALIBRATION / f'{OUT_STEM}_rounds'
_SCRATCH_IGNORED = ('shards/', 'texts.json', 'rater_prompt.md')
"""The scratch files re-rendered from the inputs; the key and the ratings are committed."""

HELD_OUT_PROJECT = 'solar_challenge_platform'
TEST_TARGET_WRITES = 300
SPLIT_SEED = '7160'
SPLITS = ('test', 'dev', 'train')
SPLIT_RULE = (
    f'test: every live write of {HELD_OUT_PROJECT}, and every other live write created at or'
    ' after the UTC midnight inside the live window whose test split is closest to'
    f' {TEST_TARGET_WRITES} writes, a tie going to the later midnight. tuning: every other'
    f" write, ordered by sha256('{SPLIT_SEED}:<entry_id>'); dev is its first half, rounded"
    ' down, and train the rest. Verdicts are never read to decide membership.'
)


@dataclass(frozen=True)
class EvalWrite:
    """One write's committed row, its text, and each target's text as the judge was shown it."""

    row: Mapping[str, Any]
    child_text: str
    target_texts: Mapping[str, str]

    def entries(self) -> list[_rounds.ShardEntry]:
        """One current-basis rating entry per target, in rank order."""
        return [
            _rounds.ShardEntry(
                entry_id=self.row['entry_id'], target_id=target['target_id'],
                basis=str(Basis.CURRENT), project=self.row['project'],
                child_text=self.child_text, parent_text=self.target_texts[target['target_id']],
            )
            for target in self.row['targets']
        ]


def offline_writes(snapshot: Mapping[str, Any], snapshot_sha256: str) -> list[EvalWrite]:
    """The judged writes of the offline write-time draw, each with its top targets."""
    run = _arms.draw_run_set(snapshot, snapshot_sha256, max_writes=OFFLINE_MAX_WRITES,
                             slates=_arms.Slates.WRITE_TIME)
    return [
        _eval_write(write, source=SOURCE_OFFLINE, stratum=OFFLINE_STRATUM,
                    snapshot_targets=snapshot['targets'], edited=lambda _target: None,
                    production=(OUTCOME_STORED, None))
        for write in run.writes if _judged(write)
    ]


def _judged(write: Mapping[str, Any]) -> bool:
    """Whether production would triage *write*: not a reconciliation marker, no attach keys."""
    return not write['recon_marker'] and not write['declares_attach_keys']


def offline_population(snapshot: Mapping[str, Any], snapshot_sha256: str) -> int:
    """How many of the snapshot's writes the offline draw samples from: judged, judge band at write time."""
    run = _arms.draw_run_set(snapshot, snapshot_sha256, max_writes=None,
                             slates=_arms.Slates.WRITE_TIME)
    return sum(_judged(write) for write in run.writes)


def live_writes(header: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> list[EvalWrite]:
    """The rebuilt live slates' writes, each with its top targets; refused unless *header* drew them all."""
    drawn = sum(header['allocation'].values())
    if drawn != len(rows):
        raise ValueError(f'the live slates header drew {drawn} writes, but {len(rows)} rows follow it')
    return [
        _eval_write(row, source=SOURCE_LIVE, stratum=f'live-{row["project_id"]}',
                    snapshot_targets=row['targets'], edited=_live_edited(row),
                    production=(row['production_outcome'], row['production_target']))
        for row in rows
    ]


def _live_edited(row: Mapping[str, Any]) -> Callable[[str], bool | None]:
    on_slate = {c['memory_id']: c['edited_after_write'] for c in row['candidates']}
    return lambda target: (on_slate[target] if target in on_slate
                           else row['targets'][target]['edited_after_write'])


def _eval_write(
    write: Mapping[str, Any], *, source: str, stratum: str, snapshot_targets: Mapping[str, Any],
    edited: Callable[[str], bool | None], production: tuple[str, str | None],
) -> EvalWrite:
    targets = _arms.top_targets(write, EVAL_WIDTH)
    texts = {target: _arms.target_text(snapshot_targets, write, target) for target in targets}
    textless = sorted(target for target, text in texts.items() if text is None)
    if textless:
        raise ValueError(f'write {write["memory_id"]}: no text for its targets {textless}')
    cosine: dict[str, float] = {}
    for candidate in write['candidates']:
        score = candidate['store_score']
        if score is not None:
            canonical = candidate['canonical_id']
            cosine[canonical] = max(score, cosine.get(canonical, score))
    row = {
        'entry_id': write['memory_id'], 'source': source, 'project': write['project_id'],
        'category': write['category'], 'created_at': write['created_at'], 'stratum': stratum,
        'production_outcome': production[0], 'production_target': production[1],
        'targets': [{'target_id': target, 'rank': rank, 'cosine': cosine[target],
                     'edited_after_write': edited(target)}
                    for rank, target in enumerate(targets, start=1)],
    }
    return EvalWrite(row=row, child_text=write['content'],
                     target_texts={target: text for target, text in texts.items() if text is not None})


Pair = tuple[str, str]


def pair_derivations(
    writes: Iterable[Mapping[str, Any]], verdicts: Iterable[Mapping[str, Any]],
) -> dict[Pair, _rubric.Derivation]:
    """Each (write, target) pair's :data:`FULL_LABEL_OUTCOME` derivation on the current basis."""
    judgements = _rubric.resolve_judgements(verdicts)
    return {
        (row['entry_id'], target['target_id']): _rubric.derive(FULL_LABEL_OUTCOME, {
            target['target_id']: judgements.get(
                _rubric.PairKey(row['entry_id'], target['target_id'], Basis.CURRENT)),
        })
        for row in writes for target in row['targets']
    }


def _third_round_pairs(derivations: Mapping[Pair, _rubric.Derivation]) -> list[Pair]:
    """The pairs only a third rater can unblock; an unrated decisive binary is refused."""
    missing: defaultdict[str, list[_rubric.Gap]] = defaultdict(list)
    for (entry_id, _), derivation in derivations.items():
        if isinstance(derivation, _rubric.Incomplete):
            missing[entry_id] += derivation.missing
    return _rounds.third_round_blockers({
        entry_id: _rubric.Incomplete(tuple(sorted(gaps))) for entry_id, gaps in missing.items()
    })


def _refuse_incomplete(derivations: Mapping[Pair, _rubric.Derivation]) -> None:
    gaps = [
        f'{gap.kind} {entry_id} {gap.target_id} {gap.binary}'
        for (entry_id, _), derivation in sorted(derivations.items())
        if isinstance(derivation, _rubric.Incomplete) for gap in derivation.missing
    ]
    if gaps:
        raise ValueError('pairs are not fully labelled; run third, or rate the shards they'
                         ' wait on:\n' + '\n'.join(gaps))


def _summary(
    inputs: _Inputs, key: Mapping[str, Any], rated: _rounds.Rated,
    verdicts: Sequence[Mapping[str, Any]], derivations: Mapping[Pair, _rubric.Derivation],
) -> dict[str, Any]:
    decided = {pair: d for pair, d in derivations.items() if isinstance(d, _rubric.Decided)}
    consumed = [
        (_rubric.PairKey(entry_id, target_id, Basis.CURRENT), binary)
        for (entry_id, target_id), derivation in decided.items()
        for _, binary in derivation.consumed
    ]
    unclear = sorted((pair, d) for pair, d in derivations.items() if isinstance(d, _rubric.Unclear))
    return {
        'full_label_outcome': FULL_LABEL_OUTCOME,
        'width': EVAL_WIDTH,
        'pairs': dict(Counter(write.row['source'] for write in inputs.writes
                              for _ in write.row['targets'])),
        'labels': {'decided': len(decided), 'unclear': len(unclear)},
        'unclear_pairs': [
            {'entry_id': entry_id, 'target_id': target_id,
             'binaries': sorted(str(binary) for _, binary in derivation.binaries)}
            for (entry_id, target_id), derivation in unclear
        ],
        'kappa': _score.kappa_over_consumed(consumed, _rubric.resolve_judgements(verdicts)),
        'rating_volume': _rounds.rating_volume(key),
        'reasked': rated.reasked,
        'shards': {name: {'rater': spec['rater'], 'role': spec['role'], 'round': spec['round'],
                          'items': len(spec['items'])}
                   for name, spec in sorted(key['shards'].items())},
        'strata': inputs.strata,
        'inputs': {'snapshot_sha256': inputs.snapshot_sha256,
                   'live_slates_sha256': inputs.live_slates_sha256},
    }


# --------------------------------------------------------------------------
# The inputs, the scratch and the CLI
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class _Inputs:
    """Every eval write from the read-only inputs, the strata they were drawn from, and the inputs' digests."""

    writes: list[EvalWrite]
    strata: dict[str, int]
    snapshot_sha256: str
    live_slates_sha256: str

    @classmethod
    def read(cls, snapshot_path: Path, live_slates: Path) -> _Inputs:
        snapshot, snapshot_sha256 = _freeze.load_snapshot(snapshot_path)
        body = live_slates.read_bytes()
        header, *rows = [json.loads(line) for line in body.decode().splitlines() if line.strip()]
        return cls(
            writes=offline_writes(snapshot, snapshot_sha256) + live_writes(header, rows),
            strata={OFFLINE_STRATUM: offline_population(snapshot, snapshot_sha256),
                    **{f'live-{project}': count for project, count in header['strata'].items()}},
            snapshot_sha256=snapshot_sha256,
            live_slates_sha256=hashlib.sha256(body).hexdigest(),
        )

    def entries(self) -> dict[_rounds.Triple, _rounds.ShardEntry]:
        return {(entry.entry_id, entry.target_id, entry.basis): entry
                for write in self.writes for entry in write.entries()}

    def rows(self) -> list[Mapping[str, Any]]:
        return sorted((write.row for write in self.writes), key=lambda row: row['entry_id'])


@dataclass(frozen=True)
class _Rounds:
    """The inputs, the scratch and its key, and the effective rater lines so far."""

    inputs: _Inputs
    scratch: _rounds.RatingScratch
    key: dict[str, Any]

    @classmethod
    def read(cls, args: argparse.Namespace) -> _Rounds:
        scratch = _rounds.RatingScratch(args.scratch)
        return cls(_Inputs.read(args.snapshot, args.live_slates), scratch, scratch.load_key())

    def finished(self) -> _rounds.Rated:
        """The effective Opus lines, refused while any awaits a re-ask or any pair lacks a rater.

        A cross-family line is refused too: it is compared with the Opus labels, never mixed in.
        """
        entries = self.inputs.entries()
        rated = self.scratch.ratings(self.key, entries)
        _rounds.refuse_unfinished(rated, entries)
        foreign = sorted({row['batch'] for row in rated.rows
                          if row['role'] == _rubric.Role.CROSS_FAMILY})
        if foreign:
            raise ValueError(f'cross_family rows do not belong in the Opus corpus; shards {foreign}')
        return rated


def _shards_command(args: argparse.Namespace) -> dict[str, Any]:
    rounds = _Rounds.read(args)
    entries = rounds.inputs.entries()
    added = rounds.scratch.add_shards(rounds.key, list(entries.values()), seed=args.seed)
    rounds.scratch.save_key(rounds.key)
    rounds.scratch.save_texts(entries.values())
    rendered = rounds.scratch.render_missing(rounds.key, entries)
    (args.scratch / 'rater_prompt.md').write_text(_rounds.rater_prompt())
    (args.scratch / '.gitignore').write_text(''.join(f'{name}\n' for name in _SCRATCH_IGNORED))
    return {
        'pairs': dict(Counter(write.row['source'] for write in rounds.inputs.writes
                              for _ in write.row['targets'])),
        'new_shards': {name: len(rounds.key['shards'][name]['items']) for name in added},
        'rendered_again': rendered,
    }


def _reask_command(args: argparse.Namespace) -> dict[str, Any]:
    rounds = _Rounds.read(args)
    return _rounds.reask_round(rounds.scratch, rounds.key, rounds.inputs.entries())


def _third_command(args: argparse.Namespace) -> dict[str, Any]:
    rounds = _Rounds.read(args)
    derivations = pair_derivations(rounds.inputs.rows(), rounds.finished().rows)
    return _rounds.third_round(rounds.scratch, rounds.key, rounds.inputs.entries(),
                               _third_round_pairs(derivations))


def _assemble_command(args: argparse.Namespace) -> dict[str, Any]:
    rounds = _Rounds.read(args)
    rated = rounds.finished()
    child_texts = {write.row['entry_id']: write.child_text for write in rounds.inputs.writes}
    verdicts = sorted((_rounds.with_digest(row, child_texts[row['entry_id']]) for row in rated.rows),
                      key=_verdict_order)
    rows = rounds.inputs.rows()
    derivations = pair_derivations(rows, verdicts)
    _refuse_incomplete(derivations)
    summary = _summary(rounds.inputs, rounds.key, rated, verdicts, derivations)
    out: Path = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / f'{OUT_STEM}_writes.jsonl', rows)
    _write_jsonl(out / f'{OUT_STEM}_verdicts.jsonl', verdicts)
    (out / f'{OUT_STEM}_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    return {'writes': len(rows), 'verdicts': len(verdicts), 'labels': summary['labels']}


def _verdict_order(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (row['entry_id'], row['target_id'], row['basis'], row['rater'])


def _jsonl_body(rows: Iterable[Mapping[str, Any]]) -> str:
    """The committed JSONL form: one sort_keys JSON object per line."""
    return ''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text(_jsonl_body(rows))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha256(body: str) -> str:
    return hashlib.sha256(body.encode()).hexdigest()


def freeze_split(
    writes: Iterable[Mapping[str, Any]], verdicts: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """The held-out split of the eval set by :data:`SPLIT_RULE`, with the digests it froze.

    Membership reads only entry ids, sources, projects and created_at; the
    verdicts enter only ``test_sha256`` and ``verdicts_sha256``.
    """
    rows, verdict_rows = list(writes), list(verdicts)
    cutoff = _cutoff(rows)
    test = sorted(row['entry_id'] for row in rows if _in_test(row, cutoff))
    held = set(test)
    tuning = sorted((row['entry_id'] for row in rows if row['entry_id'] not in held),
                    key=lambda entry_id: _sha256(f'{SPLIT_SEED}:{entry_id}'))
    half = len(tuning) // 2
    splits = {'test': test, 'dev': sorted(tuning[:half]), 'train': sorted(tuning[half:])}
    return {
        'rule': SPLIT_RULE,
        'held_out_project': HELD_OUT_PROJECT,
        'test_target_writes': TEST_TARGET_WRITES,
        'cutoff': cutoff.isoformat(),
        'counts': {name: _split_counts(rows, set(ids)) for name, ids in splits.items()},
        'splits': splits,
        'test_sha256': _sha256(
            _jsonl_body(row for row in rows if row['entry_id'] in held)
            + _jsonl_body(row for row in verdict_rows if row['entry_id'] in held)),
        'writes_sha256': _sha256(_jsonl_body(rows)),
        'verdicts_sha256': _sha256(_jsonl_body(verdict_rows)),
    }


def _instant(row: Mapping[str, Any]) -> datetime:
    instant = _freeze.parse_created_at(row['created_at'])
    if instant is None:
        raise ValueError(f'write {row["entry_id"]}: created_at {row["created_at"]!r} is not an aware instant')
    return instant


def _cutoff(rows: Sequence[Mapping[str, Any]]) -> datetime:
    """The UTC midnight inside the live window whose test split is nearest the target; ties go later."""
    live = [row for row in rows if row['source'] == SOURCE_LIVE]
    instants = [_instant(row) for row in live]
    if not instants:
        raise ValueError('the eval set holds no live write to split on')
    midnight = min(instants).astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    midnights = []
    while (midnight := midnight + timedelta(days=1)) <= max(instants):
        midnights.append(midnight)
    if not midnights:
        raise ValueError('no UTC midnight falls inside the live window')
    return min(midnights, key=lambda cut: (
        abs(sum(_in_test(row, cut) for row in live) - TEST_TARGET_WRITES), -cut.timestamp()))


def _in_test(row: Mapping[str, Any], cutoff: datetime) -> bool:
    return row['source'] == SOURCE_LIVE and (
        row['project'] == HELD_OUT_PROJECT or _instant(row) >= cutoff)


def _split_counts(rows: Iterable[Mapping[str, Any]], ids: set[str]) -> dict[str, dict[str, int]]:
    counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        if row['entry_id'] in ids:
            counts[row['source']][row['project']] += 1
    return {source: dict(sorted(by_project.items())) for source, by_project in sorted(counts.items())}


def verify_split(
    split: Mapping[str, Any], writes: Iterable[Mapping[str, Any]],
    verdicts: Iterable[Mapping[str, Any]],
) -> None:
    """Refuse when *writes* and *verdicts* no longer give the frozen *split*.

    A write that moved, appeared or vanished is named; a changed test write or
    test verdict is named by ``test_sha256``. Tuning verdicts may be refined.
    """
    again = freeze_split(writes, verdicts)
    moved = sorted({entry_id for name in SPLITS
                    for entry_id in set(split['splits'][name]) ^ set(again['splits'][name])})
    if moved:
        raise ValueError(f'writes moved between splits, appeared or vanished: {moved}')
    if again['test_sha256'] != split['test_sha256']:
        raise ValueError(f'test_sha256 drifted from {split["test_sha256"]} to'
                         f' {again["test_sha256"]}: a test write or verdict changed')


def tuning_writes(
    writes: Iterable[Mapping[str, Any]], split: Mapping[str, Any],
) -> Iterator[Mapping[str, Any]]:
    """The writes of the dev and train splits; a test write is never yielded."""
    tuning = set(split['splits']['dev']) | set(split['splits']['train'])
    return (row for row in writes if row['entry_id'] in tuning)


def _split_path(eval_dir: Path) -> Path:
    return eval_dir / f'{OUT_STEM}_split.json'


def _split_command(args: argparse.Namespace) -> dict[str, Any]:
    writes = _read_jsonl(args.eval_dir / f'{OUT_STEM}_writes.jsonl')
    verdicts = _read_jsonl(args.eval_dir / f'{OUT_STEM}_verdicts.jsonl')
    split = freeze_split(writes, verdicts)
    _split_path(args.eval_dir).write_text(json.dumps(split, indent=2, sort_keys=True) + '\n')
    return {'cutoff': split['cutoff'], 'counts': split['counts']}


def _refuse_inside_read_only(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Refuse a scratch or output directory inside the snapshot's directory or the soak bundle."""
    if args.command == 'split':
        if _split_path(args.eval_dir).exists() and not args.refreeze:
            parser.error(f'{_split_path(args.eval_dir)} is frozen; pass --refreeze to replace it')
        return
    snapshot_dir = args.snapshot.resolve().parent
    read_only = (snapshot_dir, snapshot_dir.parent / SOAK_BUNDLE_NAME)
    for option, path in (('--scratch', args.scratch), ('--out-dir', getattr(args, 'out_dir', None))):
        if path is None:
            continue
        resolved = path.resolve()
        for root in read_only:
            if resolved == root or root in resolved.parents:
                parser.error(f'{option} {resolved} is inside the read-only {root}')


_COMMANDS: Mapping[str, Callable[[argparse.Namespace], dict[str, Any]]] = {
    'shards': _shards_command, 'reask': _reask_command, 'third': _third_command,
    'assemble': _assemble_command, 'split': _split_command,
}
_ROUND_COMMANDS = ('shards', 'reask', 'third', 'assemble')


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in _ROUND_COMMANDS:
        command = commands.add_parser(name)
        command.add_argument('--snapshot', type=Path, required=True, metavar='PATH',
                             help="the frozen population's snapshot.json (read-only)")
        command.add_argument('--live-slates', type=Path, default=DEFAULT_LIVE_SLATES,
                             metavar='PATH')
        command.add_argument('--scratch', type=Path, default=DEFAULT_SCRATCH, metavar='DIR')
    commands.choices['shards'].add_argument('--seed', type=int, default=7160)
    commands.choices['assemble'].add_argument('--out-dir', type=Path, default=_CALIBRATION,
                                              metavar='DIR')
    split = commands.add_parser('split')
    split.add_argument('--eval-dir', type=Path, default=_CALIBRATION, metavar='DIR',
                       help='the directory holding the assembled writes and verdicts')
    split.add_argument('--refreeze', action='store_true',
                       help='replace an already frozen split')
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _refuse_inside_read_only(parser, args)
    print(json.dumps(_COMMANDS[args.command](args), indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
