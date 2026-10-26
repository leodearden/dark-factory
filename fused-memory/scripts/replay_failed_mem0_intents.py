"""Replay selected 'failed' Mem0 write intents through the live add_memory MCP tool.

A write whose Mem0 leg failed is stamped 'failed' in write_journal.db's
mem0_intents table, which keeps its full content, metadata and scope. This
script replays such rows through the LIVE add_memory tool rather than calling
Mem0 directly, so write triage and the write guards catch anything restated or
rewritten since the original write. Each row that lands (or that the server
itself queues for retry) is resolved failed -> completed through a
compare-and-set, so a second run never replays it again.

Dry run by default: it prints the rows it would replay and opens no MCP
client. Pass --apply to replay.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import contextlib
import json
import sys
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import Implementation
from shared.cli_boundary import LoudArgumentParser, run_cli

from fused_memory.services.write_journal import Mem0IntentStatus, WriteJournal

MCP_URL = 'http://127.0.0.1:8002/mcp'
ADD_MEMORY_TIMEOUT = timedelta(seconds=180)
#: Rows journaled with a NULL agent_id are attributed to this client by the
#: tool's identity resolution, rather than to a default 'mcp'.
CLIENT_INFO = Implementation(name='mem0-intent-replay', version='1')

EXIT_RUN_FAILED = 1
"""Some selected row did not land, or stdout failed.

Agrees with ``shared.cli_boundary.EXIT_STDOUT_FAILED``.
"""

AddMemoryFn = Callable[[dict[str, Any]], Awaitable[Mapping[str, Any]]]


class ReplayResult(StrEnum):
    LANDED = 'landed'
    PENDING_RETRY = 'pending_retry'
    REJECTED = 'rejected'
    NOT_LANDED = 'not_landed'
    ERROR = 'error'
    NOT_FAILED = 'not_failed'
    DRY_RUN = 'dry_run'


_SUCCESSFUL_RESULTS = frozenset(
    {ReplayResult.LANDED, ReplayResult.PENDING_RETRY, ReplayResult.DRY_RUN}
)


@dataclass(frozen=True)
class Selection:
    """Which failed intents to replay: listed ids, OR a half-open created_at window."""

    intent_ids: tuple[str, ...] = ()
    created_from: datetime | None = None
    created_before: datetime | None = None
    skip_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        by_window = self.created_from is not None or self.created_before is not None
        if bool(self.intent_ids) == by_window:
            raise ValueError(
                'select by --intent-id OR by --created-from/--created-before, exactly one'
            )
        for bound in (self.created_from, self.created_before):
            if bound is not None and bound.tzinfo is None:
                raise ValueError(f'window bound {bound.isoformat()} has no UTC offset')

    def admits(self, intent: Mapping[str, Any]) -> bool:
        if intent['id'] in self.skip_ids:
            return False
        if self.intent_ids:
            return intent['id'] in self.intent_ids
        created_at = datetime.fromisoformat(intent['created_at'])
        return (self.created_from is None or created_at >= self.created_from) and (
            self.created_before is None or created_at < self.created_before
        )


@dataclass(frozen=True)
class ReplayOutcome:
    intent_id: str
    project_id: str | None
    outcome: ReplayResult
    detail: str

    def render(self) -> str:
        return '\t'.join([self.intent_id, str(self.project_id), self.outcome.value, self.detail])


async def select_intents(journal: WriteJournal, selection: Selection) -> list[dict]:
    """The failed intents *selection* admits, oldest first."""
    failed = await journal.get_mem0_intents(status=Mem0IntentStatus.FAILED)
    return [intent for intent in failed if selection.admits(intent)]


def replay_arguments(intent: Mapping[str, Any]) -> dict[str, Any]:
    """The add_memory arguments that re-issue *intent*'s original write."""
    arguments: dict[str, Any] = {
        'content': intent['content'],
        'project_id': intent['project_id'],
        'category': intent['category'],
        'metadata': json.loads(intent['metadata'] or '{}'),
    }
    for identity in ('agent_id', 'session_id'):
        if intent.get(identity) is not None:
            arguments[identity] = intent[identity]
    return arguments


def classify_ack(ack: Mapping[str, Any]) -> tuple[ReplayResult, str]:
    if 'error' in ack:
        return ReplayResult.REJECTED, f'{ack.get("error_type")}: {ack["error"]}'
    memory_ids = ack.get('memory_ids') or []
    if memory_ids:
        return ReplayResult.LANDED, f'memory_ids={list(memory_ids)}'
    if 'mem0' in (ack.get('stores_pending_retry') or []):
        return (
            ReplayResult.PENDING_RETRY,
            "the replay's own Mem0 leg is pending server-side retry",
        )
    return ReplayResult.NOT_LANDED, str(ack.get('message', ''))


async def replay(
    journal: WriteJournal,
    add_memory: AddMemoryFn,
    *,
    selection: Selection,
    apply: bool,
) -> list[ReplayOutcome]:
    """Replay (or, without *apply*, list) every failed intent *selection* admits."""
    selected = await select_intents(journal, selection)
    selected_ids = {intent['id'] for intent in selected}
    outcomes = [
        ReplayOutcome(
            intent_id, None, ReplayResult.NOT_FAILED, 'no failed intent with this id'
        )
        for intent_id in selection.intent_ids
        if intent_id not in selected_ids and intent_id not in selection.skip_ids
    ]
    for intent in selected:
        if apply:
            outcomes.append(await _replay_one(journal, add_memory, intent))
        else:
            outcomes.append(_outcome(intent, ReplayResult.DRY_RUN, _preview(intent)))
    return outcomes


async def _replay_one(
    journal: WriteJournal, add_memory: AddMemoryFn, intent: Mapping[str, Any]
) -> ReplayOutcome:
    try:
        ack = await add_memory(replay_arguments(intent))
    except Exception as exc:
        return _outcome(intent, ReplayResult.ERROR, f'{type(exc).__name__}: {exc}')
    result, detail = classify_ack(ack)
    if result in (ReplayResult.LANDED, ReplayResult.PENDING_RETRY):
        resolved = await journal.resolve_mem0_intent(
            intent['id'],
            Mem0IntentStatus.COMPLETED,
            reason=f'replayed via add_memory: {detail}',
            expected_status=Mem0IntentStatus.FAILED,
        )
        if not resolved:
            detail += '; intent NOT resolved, so do not re-run this id'
    return _outcome(intent, result, detail)


def _outcome(intent: Mapping[str, Any], result: ReplayResult, detail: str) -> ReplayOutcome:
    return ReplayOutcome(intent['id'], intent['project_id'], result, detail)


def _preview(intent: Mapping[str, Any]) -> str:
    return (
        f'{intent["category"]} agent_id={intent["agent_id"]} '
        f'created_at={intent["created_at"]} {intent["content"][:80]!r}'
    )


async def _refuse_add_memory(arguments: dict[str, Any]) -> Mapping[str, Any]:
    raise RuntimeError('a dry run never calls add_memory')


@contextlib.asynccontextmanager
async def open_mcp_add_memory(url: str) -> AsyncIterator[AddMemoryFn]:
    """An :data:`AddMemoryFn` over the live fused-memory MCP ``add_memory`` tool.

    An exception from the ``async with`` body is re-raised once the session has
    closed, so the caller receives it as itself rather than inside the
    ExceptionGroup the MCP client's anyio task groups would wrap it in.
    """
    body_error: Exception | None = None
    async with streamablehttp_client(url) as (read, write, _session_id), ClientSession(
        read, write, client_info=CLIENT_INFO
    ) as session:
        await session.initialize()

        async def add_memory(arguments: dict[str, Any]) -> Mapping[str, Any]:
            result = await session.call_tool(
                'add_memory', arguments, read_timeout_seconds=ADD_MEMORY_TIMEOUT
            )
            if result.isError:
                text = ' '.join(getattr(block, 'text', repr(block)) for block in result.content)
                return {'error': text, 'error_type': 'mcp_tool_error'}
            payload = result.structuredContent or {}
            if set(payload) == {'result'} and isinstance(payload['result'], Mapping):
                payload = payload['result']
            return payload

        try:
            yield add_memory
        except Exception as exc:
            body_error = exc
    if body_error is not None:
        raise body_error


async def _run(
    journal_dir: Path, selection: Selection, *, apply: bool, mcp_url: str
) -> list[ReplayOutcome]:
    journal = WriteJournal(journal_dir)
    await journal.initialize()
    try:
        if not apply:
            return await replay(journal, _refuse_add_memory, selection=selection, apply=False)
        async with open_mcp_add_memory(mcp_url) as add_memory:
            return await replay(journal, add_memory, selection=selection, apply=True)
    finally:
        await journal.close()


def _aware_timestamp(text: str) -> datetime:
    try:
        moment = datetime.fromisoformat(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f'not an ISO-8601 timestamp: {text!r}') from exc
    if moment.tzinfo is None:
        raise argparse.ArgumentTypeError(f'{text!r} has no UTC offset')
    return moment


def build_parser() -> LoudArgumentParser:
    parser = LoudArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--journal-dir',
        type=Path,
        required=True,
        help=(
            'directory holding write_journal.db; opening it applies the '
            "journal's idempotent schema migration"
        ),
    )
    parser.add_argument('--intent-id', action='append', default=[], dest='intent_ids')
    parser.add_argument('--created-from', type=_aware_timestamp, help='inclusive, ISO-8601')
    parser.add_argument('--created-before', type=_aware_timestamp, help='exclusive, ISO-8601')
    parser.add_argument('--skip', action='append', default=[], help='an intent id to leave alone')
    parser.add_argument('--mcp-url', default=MCP_URL)
    parser.add_argument('--apply', action='store_true', help='replay (default: dry run)')
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        selection = Selection(
            intent_ids=tuple(args.intent_ids),
            created_from=args.created_from,
            created_before=args.created_before,
            skip_ids=frozenset(args.skip),
        )
    except ValueError as exc:
        parser.error(str(exc))
    if not WriteJournal(args.journal_dir).db_path.is_file():
        parser.error(f'no write_journal.db in {args.journal_dir}')
    outcomes = asyncio.run(
        _run(args.journal_dir, selection, apply=args.apply, mcp_url=args.mcp_url)
    )
    for outcome in outcomes:
        print(outcome.render())
    counts = collections.Counter(outcome.outcome for outcome in outcomes)
    print('summary: ' + ' '.join(f'{result.value}={counts[result]}' for result in ReplayResult))
    if all(outcome.outcome in _SUCCESSFUL_RESULTS for outcome in outcomes):
        return 0
    return EXIT_RUN_FAILED


if __name__ == '__main__':
    sys.exit(run_cli(main))
