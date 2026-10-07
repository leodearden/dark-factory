"""The link adjudicator: the rater brief's question, asked of a model.

Contract: ``plans/write-triage-link-healing-prd.md`` H2. The model reads the
rater brief (:data:`RATER_BRIEF_PATH`) verbatim as its instructions, and the
brief's seven words are the scale. That scale has one home in code,
``link_heal.Verdict``. This module imports it and never respells it; a test
mirrors both against the live brief.

The adjudicator is pure in (child text, parent text). A :class:`LinkPair`
carries nothing else, so ratings, kinds, the contested flag and the pair's
source cannot reach the model. Of the maintenance package, it imports only
``link_heal``.

The seam is :class:`ShardAsker`: a structured :class:`Shard` in, a
:class:`ShardReply` out. :func:`adjudicate_links` owns sharding, id validation,
parsing and the failed-shard streak, so every asker, fake or real, runs through
them. :class:`ClaudeShardAsker` is the production asker, in the shape of
``middleware/path_scope_adjudicator.py::PathScopeAdjudicator.adjudicate`` and
``reconciliation/judge.py::Judge._call_judge_cli``: the Claude CLI over OAuth,
schema-bound output, every tool denied, no MCP servers, a neutral cwd.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias

from shared.cli_invoke import AgentResult, invoke_with_cap_retry, no_mcp_servers_config
from shared.neutral_cwd import neutral_cli_cwd
from shared.storm_counter import StormCounter

from fused_memory.maintenance.link_heal import Verdict

if TYPE_CHECKING:
    from fused_memory.config.schema import LinkHealConfig

RATER_BRIEF_PATH = (
    Path(__file__).resolve().parents[3] / 'calibration' / 'write_triage_rater_brief.md'
)

TRUNCATION_MARKER = '…[truncated, {total} chars total]'
_MARKER_PREFIX = TRUNCATION_MARKER.split('{total}')[0]


def cap_text(text: str, field_chars: int) -> str:
    """*text* cut to *field_chars* with the brief's marker; a capped text stays as it is."""
    if len(text) <= field_chars or _MARKER_PREFIX in text[field_chars:]:
        return text
    return text[:field_chars] + TRUNCATION_MARKER.format(total=len(text))


VERDICT_OUTPUT_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'properties': {
        'verdicts': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'id': {'type': 'string'},
                    'verdict': {'type': 'string', 'enum': [verdict.value for verdict in Verdict]},
                    'reason': {'type': 'string'},
                },
                'required': ['id', 'verdict', 'reason'],
                'additionalProperties': False,
            },
        },
    },
    'required': ['verdicts'],
    'additionalProperties': False,
}
_VERDICT_WORDS = frozenset(verdict.value for verdict in Verdict)
_REPLY_KEYS = frozenset(VERDICT_OUTPUT_SCHEMA['required'])
_ENTRY_KEYS = frozenset(VERDICT_OUTPUT_SCHEMA['properties']['verdicts']['items']['required'])


@dataclass(frozen=True)
class LinkPair:
    """One (child, parent) pair to judge. ``key`` names it to the caller, never to the model."""

    key: str
    child_text: str
    parent_text: str


@dataclass(frozen=True)
class ShardItem:
    item_id: str
    child_text: str
    parent_text: str


@dataclass(frozen=True)
class Shard:
    model: str
    items: tuple[ShardItem, ...]


@dataclass(frozen=True)
class ShardReply:
    success: bool
    structured_output: Any = None
    cost_usd: float = 0.0
    detail: str = ''


class AdjudicationFailure(StrEnum):
    CLI_FAILED = 'cli_failed'
    PARSE_FAILURE = 'parse_failure'
    OMITTED_ID = 'omitted_id'
    DUPLICATE_ID = 'duplicate_id'
    UNKNOWN_ID = 'unknown_id'
    NOT_ATTEMPTED = 'not_attempted'


@dataclass(frozen=True)
class LinkVerdict:
    """The model's verdict on one pair, or why there is none. Never both."""

    key: str
    model: str
    verdict: Verdict | None = None
    reason: str = ''
    failure: AdjudicationFailure | None = None
    detail: str = ''

    def __post_init__(self) -> None:
        if (self.verdict is None) == (self.failure is None):
            raise ValueError(
                f'{self.key}: exactly one of verdict ({self.verdict}) and '
                f'failure ({self.failure}) must be set',
            )

    @property
    def failed(self) -> bool:
        return self.failure is not None


_PROTOCOL = (
    'Answer through the structured output only. The user message holds the '
    'items, each headed ITEM <id> and followed by its CHILD and PARENT texts. '
    'Return one entry per ITEM id, using each id exactly once, with the '
    'verdict word and a one-sentence reason.'
)


def render_system_prompt(brief_text: str) -> str:
    return f'{brief_text.rstrip()}\n\n## Protocol for this call\n\n{_PROTOCOL}\n'


def render_user_prompt(shard: Shard) -> str:
    return '\n\n'.join(
        f'ITEM {item.item_id}\nCHILD: {item.child_text}\nPARENT: {item.parent_text}'
        for item in shard.items
    )


class ShardAsker(Protocol):
    async def __call__(self, shard: Shard) -> ShardReply: ...


StormHook: TypeAlias = Callable[[Mapping[str, Any]], None]
ShardVerdicts: TypeAlias = dict[str, tuple[Verdict, str]]
ShardFailure: TypeAlias = tuple[AdjudicationFailure, str]

#: A streak is consecutive shards of one run, so its window must outlast any run.
_STREAK_WINDOW_SECONDS = 7 * 24 * 3600.0


async def adjudicate_links(
    pairs: Sequence[LinkPair],
    *,
    model: str,
    shard_size: int,
    field_chars: int,
    failure_streak: int,
    ask: ShardAsker | None = None,
    on_failure_storm: StormHook | None = None,
) -> list[LinkVerdict]:
    """One :class:`LinkVerdict` per pair, in order; a failure is counted, never defaulted.

    *ask* defaults to a :class:`ClaudeShardAsker`. *failure_streak* consecutive
    failed shards stop the run: *on_failure_storm* gets the storm summary, and
    every pair not yet asked fails as not_attempted.
    """
    _refuse_duplicate_keys(pairs)
    ask = ask if ask is not None else ClaudeShardAsker()
    chunks = _chunks(pairs, shard_size)
    streak = _ShardFailureStreak(failure_streak)
    verdicts: list[LinkVerdict] = []
    for index, chunk in enumerate(chunks):
        outcome = await _judge_chunk(ask, chunk, model=model, field_chars=field_chars)
        verdicts.extend(_link_verdicts(chunk, model, outcome))
        storm = streak.observe(outcome)
        if storm is not None:
            if on_failure_storm is not None:
                on_failure_storm({**storm, 'model': model, 'shards_total': len(chunks)})
            verdicts.extend(_not_attempted(chunks[index + 1:], model, streak.shards_failed))
            break
    return verdicts


def _refuse_duplicate_keys(pairs: Sequence[LinkPair]) -> None:
    repeated = sorted(key for key, count in Counter(pair.key for pair in pairs).items() if count > 1)
    if repeated:
        raise ValueError(f'duplicate LinkPair keys: {", ".join(repeated)}')


def _chunks(pairs: Sequence[LinkPair], size: int) -> list[tuple[LinkPair, ...]]:
    return [tuple(pairs[start:start + size]) for start in range(0, len(pairs), size)]


def _shard(chunk: Sequence[LinkPair], model: str, field_chars: int) -> Shard:
    return Shard(
        model=model,
        items=tuple(
            ShardItem(
                item_id=f'p{position}',
                child_text=cap_text(pair.child_text, field_chars),
                parent_text=cap_text(pair.parent_text, field_chars),
            )
            for position, pair in enumerate(chunk, 1)
        ),
    )


async def _judge_chunk(
    ask: ShardAsker, chunk: Sequence[LinkPair], *, model: str, field_chars: int,
) -> ShardVerdicts | ShardFailure:
    shard = _shard(chunk, model, field_chars)
    return parse_shard_reply(shard, await _ask(ask, shard))


async def _ask(ask: ShardAsker, shard: Shard) -> ShardReply:
    try:
        return await ask(shard)
    except Exception as exc:  # noqa: BLE001 — any asker fault fails its shard, never the run
        return ShardReply(success=False, detail=f'{type(exc).__name__}: {exc}')


def parse_shard_reply(shard: Shard, reply: ShardReply) -> ShardVerdicts | ShardFailure:
    """The verdict per item id, or why the shard fails whole."""
    if not reply.success:
        return AdjudicationFailure.CLI_FAILED, reply.detail or 'the CLI call failed'
    entries = _reply_entries(reply.structured_output)
    if isinstance(entries, str):
        return AdjudicationFailure.PARSE_FAILURE, entries
    return _id_mismatch(shard, entries) or {
        entry['id']: (Verdict(entry['verdict']), entry['reason']) for entry in entries
    }


def _reply_entries(payload: Any) -> list[dict[str, Any]] | str:
    """The reply's verdict entries, schema-checked, or what is wrong with them."""
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            return f'structured output is not JSON: {exc}'
    if not isinstance(payload, dict) or set(payload) != _REPLY_KEYS:
        return f'structured output is not an object of exactly {sorted(_REPLY_KEYS)}'
    entries = payload['verdicts']
    if not isinstance(entries, list):
        return 'verdicts is not a list'
    for index, entry in enumerate(entries):
        problem = _entry_problem(entry)
        if problem is not None:
            return f'verdicts[{index}]: {problem}'
    return entries


def _entry_problem(entry: Any) -> str | None:
    if not isinstance(entry, dict) or set(entry) != _ENTRY_KEYS:
        return f'not an object of exactly {sorted(_ENTRY_KEYS)}'
    if not all(isinstance(entry[key], str) for key in _ENTRY_KEYS):
        return 'a field is not a string'
    if entry['verdict'] not in _VERDICT_WORDS:
        return f'{entry["verdict"]!r} is not a verdict word'
    return None


def _id_mismatch(shard: Shard, entries: list[dict[str, Any]]) -> ShardFailure | None:
    answered = Counter(entry['id'] for entry in entries)
    expected = {item.item_id for item in shard.items}
    duplicate = sorted(item_id for item_id, count in answered.items() if count > 1)
    if duplicate:
        return AdjudicationFailure.DUPLICATE_ID, f'answered more than once: {duplicate}'
    unknown = sorted(set(answered) - expected)
    if unknown:
        return AdjudicationFailure.UNKNOWN_ID, f'not in the shard: {unknown}'
    omitted = sorted(expected - set(answered))
    if omitted:
        return AdjudicationFailure.OMITTED_ID, f'not answered: {omitted}'
    return None


def _link_verdicts(
    chunk: Sequence[LinkPair], model: str, outcome: ShardVerdicts | ShardFailure,
) -> list[LinkVerdict]:
    if isinstance(outcome, tuple):
        failure, detail = outcome
        return [LinkVerdict(pair.key, model, failure=failure, detail=detail) for pair in chunk]
    verdicts = []
    for position, pair in enumerate(chunk, 1):
        verdict, reason = outcome[f'p{position}']
        verdicts.append(LinkVerdict(pair.key, model, verdict=verdict, reason=reason))
    return verdicts


def _not_attempted(
    chunks: Sequence[Sequence[LinkPair]], model: str, shards_failed: int,
) -> list[LinkVerdict]:
    detail = f'adjudication stopped after {shards_failed} failed shards'
    return [
        LinkVerdict(pair.key, model, failure=AdjudicationFailure.NOT_ATTEMPTED, detail=detail)
        for chunk in chunks
        for pair in chunk
    ]


class _ShardFailureStreak:
    """Consecutive failed shards, on a latched StormCounter that a success replaces."""

    def __init__(self, threshold: int) -> None:
        self._threshold = threshold
        self._counter = StormCounter(fire_mode='latched')
        self.shards_failed = 0

    def observe(self, outcome: ShardVerdicts | ShardFailure) -> dict[str, Any] | None:
        """The storm summary on the shard that completes the streak, else ``None``."""
        if not isinstance(outcome, tuple):
            self._counter = StormCounter(fire_mode='latched')
            return None
        self.shards_failed += 1
        storm = self._counter.record(
            threshold=self._threshold,
            window_seconds=_STREAK_WINDOW_SECONDS,
            label=outcome[0].value,
        )
        return None if storm is None else {**storm, 'shards_failed': self.shards_failed}


#: Never 1; see reconciliation/judge.py::_JUDGE_CLI_MAX_TURNS for why.
_MAX_TURNS = 3
#: One shard is up to 40 pairs of two 4,000-character texts: ~80k input tokens.
_SHARD_TIMEOUT_SECONDS = 900.0
_SHARD_MAX_BUDGET_USD = 5.0
#: An operator-run script waits out a usage cap for at most this long per shard.
_CAP_WAIT_SANITY_SECONDS = 3600.0

Invoke: TypeAlias = Callable[..., Awaitable[AgentResult]]


class ClaudeShardAsker:
    """Asks one shard of the Claude CLI, with the rater brief read afresh as its instructions."""

    def __init__(
        self,
        *,
        brief_path: Path = RATER_BRIEF_PATH,
        invoke: Invoke = invoke_with_cap_retry,
        timeout_seconds: float = _SHARD_TIMEOUT_SECONDS,
        max_budget_usd: float = _SHARD_MAX_BUDGET_USD,
        max_turns: int = _MAX_TURNS,
    ) -> None:
        self._brief_path = brief_path
        self._invoke = invoke
        self._timeout_seconds = timeout_seconds
        self._max_budget_usd = max_budget_usd
        self._max_turns = max_turns

    async def __call__(self, shard: Shard) -> ShardReply:
        try:
            result = await self._invoke(
                usage_gate=None,
                label=f'link-adjudicator[{shard.model}]',
                prompt=render_user_prompt(shard),
                system_prompt=render_system_prompt(self._brief_path.read_text(encoding='utf-8')),
                cwd=neutral_cli_cwd(),
                model=shard.model,
                max_turns=self._max_turns,
                max_budget_usd=self._max_budget_usd,
                disallowed_tools=['*'],
                output_schema=VERDICT_OUTPUT_SCHEMA,
                mcp_config=no_mcp_servers_config(),
                strict_mcp_config=True,
                permission_mode='bypassPermissions',
                timeout_seconds=self._timeout_seconds,
                cap_wait_sanity_secs=_CAP_WAIT_SANITY_SECONDS,
            )
        except Exception as exc:  # noqa: BLE001 — a failed call is a failed shard
            return ShardReply(success=False, detail=f'{type(exc).__name__}: {exc}')
        return _shard_reply(result)


def _shard_reply(result: AgentResult) -> ShardReply:
    detail = '' if result.success else f'subtype={result.subtype!r} timed_out={result.timed_out}'
    return ShardReply(
        success=result.success,
        structured_output=result.structured_output,
        cost_usd=result.cost_usd,
        detail=detail,
    )


class LinkAdjudicator(Protocol):
    async def __call__(
        self, pairs: Sequence[LinkPair], *, on_failure_storm: StormHook | None = None,
    ) -> list[LinkVerdict]: ...


def configured_adjudicator(
    config: LinkHealConfig, *, ask: ShardAsker | None = None,
) -> LinkAdjudicator:
    """:func:`adjudicate_links` with the ``link_heal`` leaves bound, as read now."""

    async def adjudicate(
        pairs: Sequence[LinkPair], *, on_failure_storm: StormHook | None = None,
    ) -> list[LinkVerdict]:
        return await adjudicate_links(
            pairs,
            model=config.adjudicator_model,
            shard_size=config.shard_size,
            field_chars=config.field_chars,
            failure_streak=config.shard_failure_streak,
            ask=ask,
            on_failure_storm=on_failure_storm,
        )

    return adjudicate
