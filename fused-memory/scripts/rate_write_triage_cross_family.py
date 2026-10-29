#!/usr/bin/env python3
"""Re-rate the eval set's discordant pairs, and a concordant control, with a rater of another family.

The eval set's labels come from Opus raters (``build_write_triage_eval_set.py``).
This script asks :data:`DEFAULT_MODEL` at :data:`DEFAULT_EFFORT` effort — a
different family from the raters, and not the judge's configuration — about
the pairs whose consumed primary and second answers split, plus a seeded
sample of the concordant ones, and measures how often it agrees with each.
Each pair goes alone, as ``answer_contract()`` plus that pair rendered by
``write_triage_rating_rounds.py::shard``, so it is blind and capped exactly as
the Opus raters saw it. A reply that is not JSON, breaks the row contract or
quotes text not shown is asked again up to :data:`MAX_REASKS` times, then
recorded as a failure, never invented.

Subcommands, run from ``fused-memory/``:

- ``rate``: append one record per pair to ``--out``
  (``calibration/write_triage_eval_set_v2_cross_family.jsonl``). A re-run
  skips recorded pairs, and the run stops before a call that would pass
  ``--budget-usd`` at list price. Needs ``OPENAI_API_KEY``.
- ``report``: agreement per selection and binary, and the RELATED-boundary
  block, into ``<out>.json`` and its rendering ``<out>.md``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fused_memory.utils.script_loader import load_script

_SCRIPTS = Path(__file__).resolve().parent

if TYPE_CHECKING:
    import build_write_triage_eval_set as _eval
    import score_write_triage_pairs as _pairs
    import score_write_triage_reader as _score
    import write_triage_rating_rounds as _rounds
    import write_triage_reader_rubric as _rubric
    import write_triage_stats as _stats
else:
    _rubric = load_script(_SCRIPTS / 'write_triage_reader_rubric.py', 'write_triage_reader_rubric')
    _rounds = load_script(_SCRIPTS / 'write_triage_rating_rounds.py', 'write_triage_rating_rounds')
    _stats = load_script(_SCRIPTS / 'write_triage_stats.py', 'write_triage_stats')
    _score = load_script(_SCRIPTS / 'score_write_triage_reader.py', 'score_write_triage_reader')
    _pairs = load_script(_SCRIPTS / 'score_write_triage_pairs.py', 'score_write_triage_pairs')
    _eval = load_script(_SCRIPTS / 'build_write_triage_eval_set.py', 'build_write_triage_eval_set')

Binary = _rubric.Binary
Basis = _rubric.Basis
Role = _rubric.Role
Resolved = _rubric.Resolved

DEFAULT_MODEL = 'gpt-6.1-sol'
DEFAULT_EFFORT = 'high'
DEFAULT_BUDGET_USD = 15.0
CONTROL_N = 100
MAX_REASKS = 2
ITEM_NAME = 'cross_family'
SELECTION_DISCORDANT = 'discordant'
SELECTION_CONTROL = 'control'
REPORT_BINARIES = (Binary.HOME, Binary.CONFLICT, Binary.SCOPE, Binary.ACTION_CHANGING, Binary.LOSS)
_UNDECIDED = frozenset({Resolved.UNCLEAR, Resolved.TIED, Resolved.OPEN})
_VERSUS = ((('resolved', None), ('primary', Role.PRIMARY), ('second', Role.SECOND)))
V1_GROUPS: Mapping[str, str] = {
    'RELATED': 'RELATED/UNRELATED', 'UNRELATED': 'RELATED/UNRELATED',
    'SAME': 'SAME/SUBSUMED', 'SUBSUMED': 'SAME/SUBSUMED',
    'EXTENDS': 'EXTENDS', 'CORRECTS': 'CORRECTS',
}
"""Each lambda v1 word's group in the RELATED-boundary cross-tab; UNCLEAR is left out."""
_HOME_WORDS = ('YES', 'NO', 'UNCLEAR')

_CALIBRATION = _SCRIPTS.parent / 'calibration'
_STEM = 'write_triage_eval_set_v2'

Pair = tuple[str, str]


@dataclass(frozen=True)
class Reply:
    """A model's answer text and its token usage, as the ask port returns it."""

    text: str
    usage: Mapping[str, int] | None


Ask = Callable[[str, str], Awaitable[Reply]]


def _consumed_answers(
    writes: Sequence[Mapping[str, Any]], verdicts: Sequence[Mapping[str, Any]],
) -> tuple[dict[Pair, _rubric.Derivation], dict[_rubric.Binary, list[tuple[_rubric.PairKey, str, str]]]]:
    derivations = _eval.pair_derivations(writes, verdicts)
    consumed = [
        (_rubric.PairKey(entry_id, target_id, Basis.CURRENT), binary)
        for (entry_id, target_id), derivation in derivations.items()
        if isinstance(derivation, _rubric.Decided) for _, binary in derivation.consumed
    ]
    return derivations, _score.consumed_answer_pairs(consumed, _rubric.resolve_judgements(verdicts))


def _split_on(answered: Iterable[tuple[_rubric.PairKey, str, str]]) -> set[Pair]:
    return {(key.entry_id, key.target_id) for key, primary, second in answered if primary != second}


def discordant_pairs(
    writes: Sequence[Mapping[str, Any]], verdicts: Sequence[Mapping[str, Any]],
) -> list[Pair]:
    """The decided pairs whose primary and second raters split on any consumed answer."""
    _, answered = _consumed_answers(writes, verdicts)
    return sorted(set().union(*(_split_on(rows) for rows in answered.values())))


def control_pairs(
    writes: Sequence[Mapping[str, Any]], verdicts: Sequence[Mapping[str, Any]], *,
    seed: int | str, n: int = CONTROL_N,
) -> list[Pair]:
    """A seeded sample of *n* decided pairs whose raters agreed on every consumed answer."""
    derivations, answered = _consumed_answers(writes, verdicts)
    split = set().union(*(_split_on(rows) for rows in answered.values()))
    concordant = sorted(pair for pair, derivation in derivations.items()
                        if isinstance(derivation, _rubric.Decided) and pair not in split)
    return sorted(random.Random(f'{seed}:control').sample(concordant, min(n, len(concordant))))


def item_text(entry: _rounds.ShardEntry) -> str:
    """*entry* rendered alone, as the Opus raters saw it in their shards."""
    return _rounds.shard([entry], name=ITEM_NAME).text


def _parsed(text: str, entry: _rounds.ShardEntry, *, rater: str, batch: str,
            ) -> tuple[dict[str, Any] | None, list[str]]:
    """The cross-family row a reply gives, or None and why it must be asked again."""
    try:
        answer = json.loads(text)
    except ValueError:
        return None, ['the reply is not JSON']
    if not isinstance(answer, dict):
        return None, ['the reply is not a JSON object']
    row = {**{key: value for key, value in answer.items() if key != 'id'},
           'entry_id': entry.entry_id, 'target_id': entry.target_id, 'basis': entry.basis,
           'rater': rater, 'role': str(Role.CROSS_FAMILY), 'batch': batch}
    problems = _rounds.rating_problems(row, entry.child_text, entry.parent_text)
    return (None, problems) if problems else (row, [])


@dataclass
class _Ledger:
    """The spend so far, the costliest call seen, and the calls in flight: the budget's one account."""

    spent: float
    per_call: float
    budget: float
    in_flight: int = 0
    stopped: bool = False

    def reserve(self) -> bool:
        if self.spent + (self.in_flight + 1) * self.per_call > self.budget:
            self.stopped = True
            return False
        self.in_flight += 1
        return True

    def settle(self, usd: float) -> None:
        self.in_flight -= 1
        self.spent += usd
        self.per_call = max(self.per_call, usd)


def read_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


async def rate_pairs(
    selection: Sequence[tuple[Pair, str]], entries: Mapping[Pair, _rounds.ShardEntry], ask: Ask,
    *, out: Path, model: str, effort: str, batch: str, budget_usd: float, concurrency: int = 1,
) -> dict[str, Any]:
    """Ask about each selected pair not yet recorded in *out*, appending one record per pair.

    Returns the pairs rated, whether the budget stopped the run, the spend so
    far (earlier runs included) and the pairs left.
    """
    price = _pairs.LIST_PRICES.get(model)
    if price is None:
        raise ValueError(f'{model!r} has no list price, so no budget can be kept')
    prior = read_records(out) if out.exists() else []
    done = {(record['entry_id'], record['target_id']) for record in prior}
    pending = iter([(pair, chosen) for pair, chosen in selection if pair not in done])
    ledger = _Ledger(spent=sum(record['usd'] for record in prior), budget=budget_usd,
                     per_call=max((record['usd'] / record['attempts'] for record in prior
                                   if record['attempts']), default=0.0))
    rater = f'{model}:{effort}'
    rated = 0

    async def one(pair: Pair, chosen: str) -> dict[str, Any] | None:
        entry, attempts, usd, problems = entries[pair], 0, 0.0, ['not asked']
        for _ in range(MAX_REASKS + 1):
            if not ledger.reserve():
                if attempts == 0:
                    return None
                problems = [f'the budget stopped the re-asks: {"; ".join(problems)}']
                break
            reply = await ask(_rounds.answer_contract(), item_text(entry))
            cost = price.usd((reply.usage or {}).get('input_tokens', 0),
                             (reply.usage or {}).get('output_tokens', 0))
            ledger.settle(cost)
            attempts, usd = attempts + 1, usd + cost
            row, problems = _parsed(reply.text, entry, rater=rater, batch=batch)
            if row is not None:
                return _record(pair, chosen, rater, row, None, attempts, usd)
        return _record(pair, chosen, rater, None, '; '.join(problems), attempts, usd)

    async def worker() -> None:
        nonlocal rated
        while not ledger.stopped and (job := next(pending, None)) is not None:
            record = await one(*job)
            if record is None:
                return
            with out.open('a') as handle:
                handle.write(json.dumps(record, sort_keys=True) + '\n')
            rated += 1

    out.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    left = len([pair for pair, _ in selection if pair not in done]) - rated
    return {'rated': rated, 'stopped': ledger.stopped, 'spent_usd': ledger.spent, 'remaining': left}


def _record(pair: Pair, chosen: str, rater: str, row: dict[str, Any] | None, failure: str | None,
            attempts: int, usd: float) -> dict[str, Any]:
    return {'entry_id': pair[0], 'target_id': pair[1], 'selection': chosen, 'rater': rater,
            'row': row, 'failure': failure, 'attempts': attempts, 'usd': usd}


def report(
    records: Iterable[Mapping[str, Any]], writes: Iterable[Mapping[str, Any]],
    verdicts: Iterable[Mapping[str, Any]], v1_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """How the cross-family answers agree with the Opus labels, per selection and binary."""
    records, writes, verdicts = list(records), list(writes), list(verdicts)
    judgements = _rubric.resolve_judgements(verdicts)
    _, answered = _consumed_answers(writes, verdicts)
    rows = {(r['entry_id'], r['target_id']): r['row'] for r in records if r['row'] is not None}
    chosen = {(r['entry_id'], r['target_id']): r['selection'] for r in records}

    def counts(selection: str, failed: bool) -> int:
        return sum(chosen_as == selection and (pair not in rows) == failed
                   for pair, chosen_as in chosen.items())

    return {
        'raters': sorted({row['rater'] for row in rows.values()}),
        'spend_usd': sum(record['usd'] for record in records),
        'calls': sum(record['attempts'] for record in records),
        'coverage': {
            SELECTION_DISCORDANT: {
                'selected': len(set().union(*(_split_on(r) for r in answered.values()))),
                'rated': counts(SELECTION_DISCORDANT, False),
                'failed': counts(SELECTION_DISCORDANT, True)},
            SELECTION_CONTROL: {'rated': counts(SELECTION_CONTROL, False),
                                'failed': counts(SELECTION_CONTROL, True)},
        },
        'failures': sorted(({'entry_id': r['entry_id'], 'target_id': r['target_id'],
                             'failure': r['failure']} for r in records if r['row'] is None),
                           key=lambda failure: (failure['entry_id'], failure['target_id'])),
        'agreement': {
            selection: {str(binary): _agreement(binary, {
                pair: row for pair, row in rows.items() if chosen[pair] == selection}, judgements)
                for binary in REPORT_BINARIES}
            for selection in (SELECTION_DISCORDANT, SELECTION_CONTROL)
        },
        'related_boundary': _related_boundary(rows, answered[Binary.HOME], judgements, v1_rows),
    }


def _agreement(
    binary: _rubric.Binary, rows: Mapping[Pair, Mapping[str, Any]],
    judgements: Mapping[_rubric.PairKey, _rubric.Judgement],
) -> dict[str, Any]:
    compared: dict[str, list[tuple[str, str]]] = {versus: [] for versus, _ in _VERSUS}
    for pair, row in sorted(rows.items()):
        answer = row.get(binary)
        judgement = judgements.get(_rubric.PairKey(*pair, Basis.CURRENT))
        if answer is None or _rubric.is_abstain(answer) or judgement is None:
            continue
        for versus, role in _VERSUS:
            opus = _opus_answer(judgement, binary, role)
            if opus is not None:
                compared[versus].append((str(answer), opus))
    blocks = {f'vs_{versus}': _agree(pairs) for versus, pairs in compared.items()}
    blocks['vs_resolved']['kappa'] = _stats.cohen_kappa(compared['resolved'])
    return blocks


def _opus_answer(judgement: _rubric.Judgement, binary: _rubric.Binary, role: Any) -> str | None:
    """The resolved label (*role* None) or the one *role* rater's decided answer; None when there is none."""
    if role is None:
        resolved = judgement.answer(binary)
        return None if resolved is None or resolved in _UNDECIDED else str(resolved)
    votes = [vote for vote in judgement.votes if vote.role is role]
    answer = votes[0].answers.get(binary) if len(votes) == 1 else None
    return None if answer is None or _rubric.is_abstain(answer) else str(answer)


def _agree(pairs: Sequence[tuple[str, str]]) -> dict[str, Any]:
    agree = sum(first == second for first, second in pairs)
    return {'n': len(pairs), 'agree': agree, 'rate': _stats.rate(agree, len(pairs)),
            'wilson95': _stats.wilson95(agree, len(pairs))}


def _related_boundary(
    rows: Mapping[Pair, Mapping[str, Any]], home_answers: Iterable[tuple[_rubric.PairKey, str, str]],
    judgements: Mapping[_rubric.PairKey, _rubric.Judgement], v1_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """HOME where the Opus raters split on it, and cross-family HOME by lambda's v1 word."""
    split = sorted(pair for pair in _split_on(home_answers) if pair in rows)
    homes = [(rows[pair].get(Binary.HOME), _opus_answer(
        judgements[_rubric.PairKey(*pair, Basis.CURRENT)], Binary.HOME, None)) for pair in split]
    by_word = {group: dict.fromkeys(_HOME_WORDS, 0) for group in dict.fromkeys(V1_GROUPS.values())}
    for v1 in v1_rows:
        row = rows.get((v1['entry_id'], v1['target_id']))
        group = V1_GROUPS.get(v1['verdict'])
        if row is not None and group is not None and row.get(Binary.HOME) in _HOME_WORDS:
            by_word[group][row[Binary.HOME]] += 1
    return {
        'home_split': {'n': len(homes),
                       'cross_family_yes': sum(cross == 'YES' for cross, _ in homes),
                       'agree_resolved': sum(cross == resolved for cross, resolved in homes)},
        'by_v1_word': by_word,
    }


def render_markdown(report_: Mapping[str, Any]) -> str:
    """*report_* as Markdown tables; the same report always renders to the same text."""
    coverage = report_['coverage']
    summary = '\n'.join([
        '# Cross-family re-rate of the write-triage eval set (rubric v2.1)', '',
        f'- Raters: {", ".join(report_["raters"]) or "—"}; calls {report_["calls"]};'
        f' spend {report_["spend_usd"]:.2f} USD at list price.',
        f'- Discordant pairs: {coverage[SELECTION_DISCORDANT]["selected"]} selected,'
        f' {coverage[SELECTION_DISCORDANT]["rated"]} rated,'
        f' {coverage[SELECTION_DISCORDANT]["failed"]} failed.',
        f'- Control pairs: {coverage[SELECTION_CONTROL]["rated"]} rated,'
        f' {coverage[SELECTION_CONTROL]["failed"]} failed.',
    ])
    agreement = _table(
        ['selection', 'binary', 'versus', 'n', 'agree', 'rate', 'wilson95', 'kappa'],
        [[selection, binary, versus, block['n'], block['agree'], block['rate'], block['wilson95'],
          block.get('kappa')]
         for selection in (SELECTION_DISCORDANT, SELECTION_CONTROL)
         for binary in map(str, REPORT_BINARIES)
         for versus, _ in _VERSUS
         for block in [report_['agreement'][selection][binary][f'vs_{versus}']]],
    )
    boundary = report_['related_boundary']
    split = boundary['home_split']
    related = '\n'.join([
        f'HOME on pairs whose Opus raters split on HOME: n {split["n"]},'
        f' cross-family YES {split["cross_family_yes"]}, agrees with the resolved label'
        f' {split["agree_resolved"]}.', '',
        _table(['v1 word', *_HOME_WORDS],
               [[group, *(boundary['by_v1_word'][group][word] for word in _HOME_WORDS)]
                for group in dict.fromkeys(V1_GROUPS.values())]),
    ])
    return '\n\n'.join([summary, agreement, related]) + '\n'


def _shown(value: Any) -> str:
    if value is None:
        return '—'
    return str(value)


def _table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = [f'| {" | ".join(header)} |', f'|{"---|" * len(header)}']
    lines += [f'| {" | ".join(_shown(cell) for cell in row)} |' for row in rows]
    return '\n'.join(lines)


# --------------------------------------------------------------------------
# The OpenAI port and the CLI
# --------------------------------------------------------------------------

_BACKOFF_SECONDS = 2.0
_TRANSPORT_ATTEMPTS = 5


def openai_ask(model: str, effort: str, *, timeout: float = 600.0,
               sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep) -> Ask:
    """The real port: one Responses call per ask, client built and closed per call.

    Transport failures are retried with doubling backoff, as the arms runner does.
    """
    import openai  # the network dependency stays out of the report path

    transient = (TimeoutError, openai.APIConnectionError, openai.RateLimitError,
                 openai.InternalServerError)

    sampling: dict[str, Any] = {'reasoning': {'effort': effort}}

    async def ask(instructions: str, item: str) -> Reply:
        for attempt in range(_TRANSPORT_ATTEMPTS):
            try:
                async with openai.AsyncOpenAI() as client:
                    response = await asyncio.wait_for(client.responses.create(
                        model=model, instructions=instructions, input=item,
                        text={'format': {'type': 'json_object'}}, **sampling,
                    ), timeout=timeout)
            except transient:
                if attempt + 1 == _TRANSPORT_ATTEMPTS:
                    raise
                await sleep(_BACKOFF_SECONDS * 2 ** attempt)
                continue
            usage = getattr(response, 'usage', None)
            return Reply(text=response.output_text or '', usage=None if usage is None else {
                'input_tokens': usage.input_tokens, 'output_tokens': usage.output_tokens})
        raise AssertionError('unreachable: the last transport failure re-raises')

    return ask


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _rate_command(args: argparse.Namespace) -> dict[str, Any]:
    writes, verdicts = _jsonl(args.writes), _jsonl(args.verdicts)
    projects = {row['entry_id']: row['project'] for row in writes}
    entries = {(record['entry_id'], record['target_id']): _rounds.ShardEntry(
        **record, project=projects[record['entry_id']]) for record in json.loads(args.texts.read_text())
        if record['basis'] == Basis.CURRENT}
    selection = [(pair, SELECTION_DISCORDANT) for pair in discordant_pairs(writes, verdicts)]
    selection += [(pair, SELECTION_CONTROL)
                  for pair in control_pairs(writes, verdicts, seed=args.seed, n=args.control)]
    return asyncio.run(rate_pairs(
        selection, entries, openai_ask(args.model, args.effort), out=args.out, model=args.model,
        effort=args.effort, batch=args.batch, budget_usd=args.budget_usd,
        concurrency=args.concurrency,
    ))


def _report_command(args: argparse.Namespace) -> dict[str, Any]:
    built = report(read_records(args.records), _jsonl(args.writes), _jsonl(args.verdicts),
                   _jsonl(args.v1_verdicts))
    args.out.with_suffix('.json').write_text(json.dumps(built, indent=2, sort_keys=True) + '\n')
    args.out.with_suffix('.md').write_text(render_markdown(json.loads(json.dumps(built))))
    return {'coverage': built['coverage'], 'spend_usd': built['spend_usd']}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('rate', 'report'):
        command = commands.add_parser(name)
        command.add_argument('--writes', type=Path, default=_CALIBRATION / f'{_STEM}_writes.jsonl')
        command.add_argument('--verdicts', type=Path,
                             default=_CALIBRATION / f'{_STEM}_verdicts.jsonl')
    rate = commands.choices['rate']
    rate.add_argument('--texts', type=Path, default=_CALIBRATION / f'{_STEM}_rounds' / 'texts.json',
                      help='the rounds scratch texts.json, as the Opus raters saw each pair')
    rate.add_argument('--out', type=Path, default=_CALIBRATION / f'{_STEM}_cross_family.jsonl')
    rate.add_argument('--model', default=DEFAULT_MODEL)
    rate.add_argument('--effort', default=DEFAULT_EFFORT)
    rate.add_argument('--budget-usd', type=float, default=DEFAULT_BUDGET_USD)
    rate.add_argument('--seed', type=int, default=7160)
    rate.add_argument('--control', type=int, default=CONTROL_N)
    rate.add_argument('--concurrency', type=int, default=4)
    rate.add_argument('--batch', default=f'cross-family-{datetime.now(UTC).date().isoformat()}')
    report_ = commands.choices['report']
    report_.add_argument('--records', type=Path,
                         default=_CALIBRATION / f'{_STEM}_cross_family.jsonl')
    report_.add_argument('--v1-verdicts', type=Path,
                         default=_CALIBRATION / 'write_triage_pair_verdicts.jsonl')
    report_.add_argument('--out', type=Path, default=_CALIBRATION / f'{_STEM}_cross_family',
                         help='the report path without suffix: <out>.json and <out>.md')
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    command = _rate_command if args.command == 'rate' else _report_command
    print(json.dumps(command(args), indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
