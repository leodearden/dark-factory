#!/usr/bin/env python3
"""Score link-adjudicator arms against blind majority verdicts.

The contract is plans/write-triage-link-healing-prd.md H3. An *arm* is one
model run through the link adjudicator (``maintenance/link_adjudicator.py``)
over a frozen population of rated pairs. Each arm's verdicts are scored with
plain confusion counts against the raters' majority class, so the figures are
defined here and nowhere else:

- ``misfile_recall``: arm misfile / truth misfile;
- ``false_detach_rate``: arm misfile / truth belongs (CORRECTS or agreeing);
- ``corrects_recall``: arm CORRECTS / truth CORRECTS;
- ``false_corrects_rate``: arm CORRECTS / truth agreeing;
- ``parse_failure_rate``: pairs with no verdict / all pairs.

A pair with no verdict is a miss for both recalls and neither a detach nor a
flag. A figure whose denominator is 0 is ``None``, never 0, so a gate reading
it fails closed. ι's ``score_write_triage_pairs.py::score_pairs`` is not used:
its metrics are rates over a judge's attaches, a different quantity. Only its
majority rule, ``resolve_verdicts``, is reused, because that rule has one home.

Two populations:

- hand-link mode (no ``--pairs``): the committed hand-link corpus. Each pair is
  read live by id and kept only while its texts still hash to the rated ones.
- λ mode (``--pairs``): per-rater verdict rows (task 6152's contract) resolved
  by majority, with texts from a pairs file keyed ``entry_text`` and
  ``target_text``, the keys of task 6151's committed
  ``calibration/write_triage_pairs_to_rate.jsonl``. A pairs row lacking either
  id or either text key, and a pair listed twice, are refused.

Arm ``fake`` is a deterministic stand-in, pure in the texts, that runs the real
sharding and parsing path with no CLI call. A real arm names a Claude CLI model
alias. A storm of failed shards is recorded on its arm row, never filed: this is
measurement.

Usage, from ``fused-memory/``::

    uv run python scripts/eval_link_adjudicator.py \\
        --corpus calibration/hand_link_verdicts.jsonl --arms opus,sonnet

The report goes to ``--out`` (default ``calibration/link_adjudicator_report.json``,
or ``calibration/link_adjudicator_report_triage.json`` with ``--pairs``) and to
stdout. Exit 0 when it is written; 2 when the corpus or pairs are refused.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import importlib.util
import json
import logging
import math
import sys
import types
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from shared.safe_io import atomic_write_text

from fused_memory.config.schema import FusedMemoryConfig, LinkHealConfig
from fused_memory.maintenance._utils import override_config_path
from fused_memory.maintenance.link_adjudicator import (
    RATER_BRIEF_PATH,
    ClaudeShardAsker,
    LinkPair,
    LinkVerdict,
    Shard,
    ShardAsker,
    ShardItem,
    ShardReply,
    adjudicate_links,
)
from fused_memory.maintenance.link_heal import (
    AGREEING_VERDICTS,
    MISFILE_VERDICTS,
    CorpusRow,
    KindClass,
    Verdict,
    healed_kind,
    kind_class,
    load_corpus_rows,
)
from fused_memory.maintenance.link_heal_store import (
    LinkHealStore,
    LiveRecord,
    StoreReadFailed,
    StoreUnreachable,
    ToolCaller,
    server_tool_caller,
)
from fused_memory.server.grouped_read import AMENDMENT_KIND

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


# Pyright cannot follow the by-path loader, so it alone resolves `_iota` through
# the scripts/ extraPath; at runtime the module is always the by-path load.
if TYPE_CHECKING:
    import score_write_triage_pairs as _iota
else:
    _iota = _load_script(_SCRIPTS / 'score_write_triage_pairs.py', 'score_write_triage_pairs')


class TruthClass(StrEnum):
    MISFILE = 'misfile'
    CORRECTS = 'corrects'
    AGREEING = 'agreeing'
    UNCLEAR = 'unclear'


BELONGS_CLASSES = frozenset({TruthClass.CORRECTS, TruthClass.AGREEING})


def truth_class(verdict: Verdict) -> TruthClass:
    """The class a majority *verdict* word puts its pair in."""
    if verdict in MISFILE_VERDICTS:
        return TruthClass.MISFILE
    if verdict is Verdict.CORRECTS:
        return TruthClass.CORRECTS
    if verdict in AGREEING_VERDICTS:
        return TruthClass.AGREEING
    return TruthClass.UNCLEAR


_RESOLUTION_TRUTH: Mapping[Any, TruthClass] = MappingProxyType({
    _iota.Resolution.MISFILE: TruthClass.MISFILE,
    _iota.Resolution.CORRECTS: TruthClass.CORRECTS,
    _iota.Resolution.AGREEING: TruthClass.AGREEING,
    _iota.Resolution.UNCLEAR: TruthClass.UNCLEAR,
})


class Exclusion(StrEnum):
    """Why a rated pair is not in the scored population."""

    RATED_TEXT_MISMATCH = 'rated_text_mismatch'
    CHILD_MISSING = 'child_missing'
    PARENT_MISSING = 'parent_missing'
    CHILD_TEXT_CHANGED = 'child_text_changed'
    PARENT_TEXT_CHANGED = 'parent_text_changed'
    READ_FAILED = 'read_failed'
    TIED = 'tied'
    UNRATED = 'unrated'
    NO_TEXTS = 'no_texts'


@dataclass(frozen=True)
class ScoredItem:
    """One rated pair in the population: its texts, and what the raters' majority says.

    ``majority`` is the majority's word where the corpus states one; λ mode
    knows only the class.
    """

    key: str
    child_text: str
    parent_text: str
    truth: TruthClass
    kind_at_rating: str | None = None
    majority: Verdict | None = None


@dataclass(frozen=True)
class Population:
    items: tuple[ScoredItem, ...]
    excluded: Mapping[Exclusion, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'excluded', MappingProxyType(dict(self.excluded)))

    def summary(self) -> dict[str, Any]:
        truths = Counter(item.truth for item in self.items)
        return {
            'n_pairs': len(self.items),
            'n_misfile': truths[TruthClass.MISFILE],
            'n_corrects': truths[TruthClass.CORRECTS],
            'n_belongs': sum(truths[truth] for truth in BELONGS_CLASSES),
            'n_unclear': truths[TruthClass.UNCLEAR],
            'excluded': sum(self.excluded.values()),
            'excluded_by_reason': {
                reason.value: count for reason, count in sorted(self.excluded.items()) if count
            },
        }


_WILSON_Z = 1.96


def wilson95(k: int, n: int) -> list[float] | None:
    """The Wilson score 95% interval of *k* successes in *n* trials; ``None`` for none."""
    if n == 0:
        return None
    p = k / n
    z2 = _WILSON_Z * _WILSON_Z
    denominator = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denominator
    half = _WILSON_Z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denominator
    return [max(0.0, centre - half), min(1.0, centre + half)]


def _figure(name: str, num: int, den: int) -> dict[str, Any]:
    return {
        name: None if den == 0 else num / den,
        f'{name}_num': num,
        f'{name}_den': den,
        f'{name}_ci95': wilson95(num, den),
    }


def _detaches(verdict: Verdict | None) -> bool:
    return verdict in MISFILE_VERDICTS


def _flags(verdict: Verdict | None) -> bool:
    return verdict is Verdict.CORRECTS


def _failed(verdict: Verdict | None) -> bool:
    return verdict is None


#: Each H3 figure: the truth classes it is over, and what of the arm's it counts.
_FIGURES: tuple[tuple[str, frozenset[TruthClass], Callable[[Verdict | None], bool]], ...] = (
    ('misfile_recall', frozenset({TruthClass.MISFILE}), _detaches),
    ('false_detach_rate', BELONGS_CLASSES, _detaches),
    ('corrects_recall', frozenset({TruthClass.CORRECTS}), _flags),
    ('false_corrects_rate', frozenset({TruthClass.AGREEING}), _flags),
    ('parse_failure_rate', frozenset(TruthClass), _failed),
)
FIGURE_NAMES = tuple(name for name, _truths, _counts in _FIGURES)


def _said(verdicts: Iterable[LinkVerdict]) -> dict[str, Verdict | None]:
    return {verdict.key: verdict.verdict for verdict in verdicts}


def score_arm(items: Sequence[ScoredItem], verdicts: Iterable[LinkVerdict]) -> dict[str, Any]:
    """The five H3 figures of one arm's *verdicts* on *items*; a missing verdict is a failure."""
    said = _said(verdicts)
    figures: dict[str, Any] = {}
    for name, truths, counts in _FIGURES:
        within = [item for item in items if item.truth in truths]
        figures.update(_figure(name, sum(counts(said.get(item.key)) for item in within), len(within)))
    return figures


_KIND_AGREEMENT_CLASSES = frozenset({KindClass.SIGHTING, KindClass.HALF_LINK})


def kind_agreement(items: Sequence[ScoredItem], verdicts: Iterable[LinkVerdict]) -> dict[str, Any]:
    """How often the arm heals a sighting or half-link to the kind the majority heals it to.

    The baseline is the share an arm answering "amendment" every time would get.
    """
    said = _said(verdicts)
    eligible = [
        (item, item.majority) for item in items
        if item.majority is not None and kind_class(item.kind_at_rating) in _KIND_AGREEMENT_CLASSES
    ]
    agrees = amendments = 0
    for item, majority in eligible:
        wanted = healed_kind(item.kind_at_rating, majority)
        arm = said.get(item.key)
        agrees += arm is not None and healed_kind(item.kind_at_rating, arm) == wanted
        amendments += wanted == AMENDMENT_KIND
    return {
        **_figure('kind_agreement', agrees, len(eligible)),
        **_figure('kind_agreement_always_amendment_baseline', amendments, len(eligible)),
    }


def _rank(value: float | None, *, higher_is_better: bool = False) -> float:
    if value is None:
        return math.inf
    return -value if higher_is_better else value


def _selection_key(row: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        _rank(row['misfile_recall'], higher_is_better=True),
        _rank(row['false_detach_rate']),
        _rank(row['false_corrects_rate']),
        row['cost_usd'],
    )


def select_arm(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """A copy of the best arm row: highest misfile recall, then fewest false detaches,
    then fewest false flags, then cheapest. A ``None`` figure ranks worst."""
    return copy.deepcopy(dict(min(rows, key=_selection_key)))


TRIAGE_CHILD_TEXT_KEY = 'entry_text'
TRIAGE_PARENT_TEXT_KEY = 'target_text'
_TRIAGE_PAIR_KEYS = ('entry_id', 'target_id', TRIAGE_CHILD_TEXT_KEY, TRIAGE_PARENT_TEXT_KEY)

Pair = tuple[str, str]


def _pair_texts(pair_rows: Iterable[Mapping[str, Any]]) -> dict[Pair, tuple[Any, Any]]:
    """Each pair's (child, parent) texts; a row lacking a key, or a pair listed twice, is refused."""
    texts: dict[Pair, tuple[Any, Any]] = {}
    for index, row in enumerate(pair_rows):
        missing = [key for key in _TRIAGE_PAIR_KEYS if key not in row]
        if missing:
            raise ValueError(
                f'pairs row {index} ({row.get("entry_id")}, {row.get("target_id")}) '
                f'lacks {", ".join(missing)}',
            )
        pair = (row['entry_id'], row['target_id'])
        if pair in texts:
            raise ValueError(f'pairs row {index}: pair {pair} is listed twice')
        texts[pair] = (row[TRIAGE_CHILD_TEXT_KEY], row[TRIAGE_PARENT_TEXT_KEY])
    return texts


def triage_items(
    verdict_rows: Iterable[Mapping[str, Any]], pair_rows: Iterable[Mapping[str, Any]],
) -> Population:
    """The λ population: each rated pair resolved by majority, with its texts."""
    texts = _pair_texts(pair_rows)
    resolved = _iota.resolve_verdicts(verdict_rows)
    items: list[ScoredItem] = []
    excluded: list[Exclusion] = [Exclusion.UNRATED for pair in texts if pair not in resolved]
    for (entry_id, target_id), resolution in resolved.items():
        child, parent = texts.get((entry_id, target_id), (None, None))
        if resolution is _iota.Resolution.TIED:
            excluded.append(Exclusion.TIED)
        elif not isinstance(child, str) or not isinstance(parent, str):
            excluded.append(Exclusion.NO_TEXTS)
        else:
            items.append(ScoredItem(
                key=f'{entry_id}:{target_id}',
                child_text=child,
                parent_text=parent,
                truth=_RESOLUTION_TRUTH[resolution],
            ))
    return Population(items=tuple(items), excluded=Counter(excluded))


class Mode(StrEnum):
    HAND_LINK = 'hand_link'
    TRIAGE = 'triage'


_TEXT_PROVENANCE: Mapping[Mode, Mapping[str, Any]] = MappingProxyType({
    Mode.HAND_LINK: {
        'text_keys': None,
        'text_key_basis': (
            'texts read live by id through get_memory_by_id, kept only while they '
            'hash to the corpus row'
        ),
    },
    Mode.TRIAGE: {
        'text_keys': {'child': TRIAGE_CHILD_TEXT_KEY, 'parent': TRIAGE_PARENT_TEXT_KEY},
        'text_key_basis': (
            "the keys of task 6151's committed "
            'fused-memory/calibration/write_triage_pairs_to_rate.jsonl, built by '
            'scripts/run_write_triage_population_arms.py::build_pairs_to_rate'
        ),
    },
})


def build_report(
    population: Population,
    arms: Sequence[Mapping[str, Any]],
    *,
    mode: Mode,
    corpus_sha256: str,
    pairs_sha256: str | None,
    brief_sha256: str,
    field_chars: int,
    shard_size: int,
) -> dict[str, Any]:
    return {
        'provenance': {
            'mode': mode.value,
            'corpus_sha256': corpus_sha256,
            'pairs_sha256': pairs_sha256,
            'brief_sha256': brief_sha256,
            'field_chars': field_chars,
            'shard_size': shard_size,
            **_TEXT_PROVENANCE[mode],
        },
        'population': population.summary(),
        'arms': [dict(row) for row in arms],
        'selection': select_arm(arms) if arms else None,
    }


class EvalRefused(Exception):
    """The eval refused its inputs before scoring anything."""


REFUSALS = (EvalRefused, OSError, ValueError)

EXIT_OK = 0
EXIT_REFUSED = 2

_CALIBRATION = _SCRIPTS.parent / 'calibration'
DEFAULT_OUT: Mapping[Mode, Path] = MappingProxyType({
    Mode.HAND_LINK: _CALIBRATION / 'link_adjudicator_report.json',
    Mode.TRIAGE: _CALIBRATION / 'link_adjudicator_report_triage.json',
})

FAKE_ARM = 'fake'
_FAKE_WORDS = tuple(Verdict)


class FakeShardAsker:
    """A deterministic arm, pure in the texts: each item's word is picked by the
    sha256 of its child and parent, and answered through the real reply shape."""

    async def __call__(self, shard: Shard) -> ShardReply:
        return ShardReply(success=True, structured_output={
            'verdicts': [
                {'id': item.item_id, 'verdict': _fake_word(item), 'reason': 'fake arm: text hash'}
                for item in shard.items
            ],
        })


def _fake_word(item: ShardItem) -> str:
    digest = hashlib.sha256(f'{item.child_text}\0{item.parent_text}'.encode()).digest()
    return _FAKE_WORDS[digest[0] % len(_FAKE_WORDS)].value


def default_ask_for(arm: str) -> ShardAsker:
    return FakeShardAsker() if arm == FAKE_ARM else ClaudeShardAsker()


class _CostMeter:
    """A ShardAsker that sums what each reply cost."""

    def __init__(self, ask: ShardAsker) -> None:
        self._ask = ask
        self.cost_usd = 0.0

    async def __call__(self, shard: Shard) -> ShardReply:
        reply = await self._ask(shard)
        self.cost_usd += reply.cost_usd
        return reply


def load_config(config_path: str | None) -> FusedMemoryConfig:
    """The config as its file reads now; ``$CONFIG_PATH`` when *config_path* is None."""
    with override_config_path(config_path):
        return FusedMemoryConfig()


@dataclass(frozen=True)
class EvalEnv:
    """What the eval reaches beyond its arguments. The defaults are production's."""

    config_loader: Callable[[str | None], FusedMemoryConfig] = load_config
    tool_caller_for: Callable[[str], AbstractAsyncContextManager[ToolCaller]] = (
        server_tool_caller
    )
    ask_for: Callable[[str], ShardAsker] = default_ask_for


async def _live_pair(row: CorpusRow, store: LinkHealStore) -> tuple[LiveRecord, LiveRecord] | Exclusion:
    """The rated pair's child and parent as they read now, while they still hash as rated."""
    basis = row.basis
    if not basis.rated_text_matches_live:
        return Exclusion.RATED_TEXT_MISMATCH
    try:
        child = await store.read(basis.project_id, basis.child_id)
        parent = await store.read(basis.project_id, basis.parent_id)
    except StoreReadFailed:
        return Exclusion.READ_FAILED
    if child is None:
        return Exclusion.CHILD_MISSING
    if child.text_sha256 != basis.child_sha256:
        return Exclusion.CHILD_TEXT_CHANGED
    if parent is None:
        return Exclusion.PARENT_MISSING
    if parent.text_sha256 != basis.parent_sha256:
        return Exclusion.PARENT_TEXT_CHANGED
    return child, parent


async def hand_link_population(rows: Sequence[CorpusRow], store: LinkHealStore) -> Population:
    items: list[ScoredItem] = []
    excluded: list[Exclusion] = []
    for row in rows:
        live = await _live_pair(row, store)
        if isinstance(live, Exclusion):
            excluded.append(live)
            continue
        child, parent = live
        items.append(ScoredItem(
            key=row.basis.key,
            child_text=child.text,
            parent_text=parent.text,
            truth=truth_class(row.basis.verdict),
            kind_at_rating=row.kind_at_rating,
            majority=row.basis.verdict,
        ))
    return Population(items=tuple(items), excluded=Counter(excluded))


async def _load_hand_links(
    args: argparse.Namespace, config: FusedMemoryConfig, env: EvalEnv,
) -> Population:
    rows = load_corpus_rows(args.corpus)
    server_url = args.server_url or f'http://127.0.0.1:{config.server.port}'
    async with env.tool_caller_for(server_url) as tool_caller:
        store = LinkHealStore(tool_caller)
        if rows:
            await _probe(store, rows[0].basis.project_id, server_url)
        return await hand_link_population(rows, store)


async def _probe(store: LinkHealStore, project_id: str, server_url: str) -> None:
    try:
        await store.probe(project_id)
    except StoreUnreachable as failure:
        raise EvalRefused(
            f'the fused-memory server at {server_url} did not answer a read of {project_id} '
            f'({failure.error_type}: {failure.detail})',
        ) from failure


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvalRefused(f'{path} line {line_no}: not JSON ({exc})') from exc
        if not isinstance(row, dict):
            raise EvalRefused(f'{path} line {line_no}: not a JSON object')
        rows.append(row)
    return rows


async def _load_population(
    args: argparse.Namespace, config: FusedMemoryConfig, env: EvalEnv, mode: Mode,
) -> Population:
    if mode is Mode.TRIAGE:
        return triage_items(_jsonl(args.corpus), _jsonl(args.pairs))
    return await _load_hand_links(args, config, env)


async def score_arm_run(
    arm: str,
    population: Population,
    *,
    link_heal: LinkHealConfig,
    ask: ShardAsker,
    mode: Mode,
) -> dict[str, Any]:
    """One arm's row: its figures over *population*, its cost, and any failure storm."""
    meter = _CostMeter(ask)
    storms: list[Mapping[str, Any]] = []
    verdicts = await adjudicate_links(
        [LinkPair(item.key, item.child_text, item.parent_text) for item in population.items],
        model=arm,
        shard_size=link_heal.shard_size,
        field_chars=link_heal.field_chars,
        failure_streak=link_heal.shard_failure_streak,
        ask=meter,
        on_failure_storm=storms.append,
    )
    row: dict[str, Any] = {
        'arm': arm,
        'cost_usd': meter.cost_usd,
        'failure_storm': dict(storms[0]) if storms else None,
        **score_arm(population.items, verdicts),
    }
    if mode is Mode.HAND_LINK:
        row.update(kind_agreement(population.items, verdicts))
    return row


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _arm_list(text: str) -> list[str]:
    arms = list(dict.fromkeys(part.strip() for part in text.split(',') if part.strip()))
    if not arms:
        raise argparse.ArgumentTypeError('names no arm')
    return arms


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='eval_link_adjudicator.py',
        description='Score link-adjudicator arms against blind majority verdicts (PRD H3).',
    )
    parser.add_argument('--corpus', type=Path, required=True, metavar='PATH',
                        help='the hand-link corpus, or with --pairs the per-rater verdict rows')
    parser.add_argument('--arms', type=_arm_list, required=True, metavar='ARM,ARM',
                        help=f'model aliases, or {FAKE_ARM!r}')
    parser.add_argument('--pairs', type=Path, metavar='PATH',
                        help='λ mode: the pairs file holding each rated pair\'s texts')
    parser.add_argument('--out', type=Path, metavar='PATH', help='where to write the report')
    parser.add_argument('--config', help='fused-memory config YAML (default: $CONFIG_PATH)')
    parser.add_argument('--server-url',
                        help='fused-memory server (default: http://127.0.0.1:<server.port>)')
    return parser


async def run(argv: Sequence[str], env: EvalEnv) -> int:
    args = build_parser().parse_args(argv)
    config = env.config_loader(args.config)
    mode = Mode.HAND_LINK if args.pairs is None else Mode.TRIAGE
    try:
        population = await _load_population(args, config, env, mode)
    except REFUSALS as refusal:
        print(f'eval_link_adjudicator refused: {refusal}', file=sys.stderr)
        return EXIT_REFUSED
    arms = [
        await score_arm_run(
            arm, population, link_heal=config.link_heal, ask=env.ask_for(arm), mode=mode,
        )
        for arm in args.arms
    ]
    report = build_report(
        population, arms,
        mode=mode,
        corpus_sha256=_sha256(args.corpus),
        pairs_sha256=None if args.pairs is None else _sha256(args.pairs),
        brief_sha256=_sha256(RATER_BRIEF_PATH),
        field_chars=config.link_heal.field_chars,
        shard_size=config.link_heal.shard_size,
    )
    document = json.dumps(report, indent=2, sort_keys=True) + '\n'
    atomic_write_text(args.out or DEFAULT_OUT[mode], document)
    print(document, end='')
    return EXIT_OK


def main() -> int:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    return asyncio.run(run(sys.argv[1:], EvalEnv()))


if __name__ == '__main__':
    sys.exit(main())
