#!/usr/bin/env python3
"""Heal hand-linked mem0 records from a verdict corpus or the link adjudicator.

The contract is plans/write-triage-link-healing-prd.md H1 and H2.

Usage:
    python fused-memory/scripts/link_heal.py [--config PATH] [--server-url URL] COMMAND

    plan (--from-corpus PATH | --from-adjudicator [--project P ...]) [--out PATH]
        Decide every live link, ledger the heals, and write the plan document.
        Writes nothing to the store. --from-adjudicator asks the link
        adjudicator about the links of each --project (default: the server's
        own project) that no deterministic row decides and that it has not
        judged at their current texts, and ledgers its verdicts.
    apply [--from-adjudicator] [--approved-plan-sha SHA256]
        Heal the pending plan of the corpus (default) or of the adjudicator,
        oldest first. With the sha of the pending plan document, every pending
        heal is applied and the per-run cap is lifted. When the adjudications
        behind the pending heals pass a share ceiling, nothing is written.
    undo --run RUN_ID
        Take back every heal one apply run applied. RUN_ID may be its first 8 characters.
    status [--limit N]
        List recent runs from the ledger. Never creates the ledger.

Exit codes: 0 when the run is complete, or partial only by the cap or stale
skips; 1 when a heal, read or adjudication failed or the run stopped; 2 when
the command was refused before any run started.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import sys
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fused_memory.backends.mem0_client import Mem0Backend
from fused_memory.config.schema import FusedMemoryConfig, TaskmasterConfig
from fused_memory.maintenance._utils import override_config_path
from fused_memory.maintenance.link_adjudicator import LinkAdjudicator, configured_adjudicator
from fused_memory.maintenance.link_heal import CorpusFormatError, LinkBasis, load_corpus_bases
from fused_memory.maintenance.link_heal_executor import (
    ApprovalMismatch,
    FoldedEscapeFiler,
    RunLimits,
    RunReport,
    run_adjudicator_plan,
    run_apply,
    run_plan,
    run_undo,
)
from fused_memory.maintenance.link_heal_ledger import (
    LEDGER_FILENAME,
    LOCK_FILENAME,
    ActionRow,
    AmbiguousRun,
    LinkHealLedger,
    RunLock,
    RunLockHeld,
    RunRow,
    RunSource,
    UnknownRun,
)
from fused_memory.maintenance.link_heal_store import (
    CensusFailed,
    LinkCensus,
    LinkHealStore,
    QdrantLinkCensus,
    StoreUnreachable,
    ToolCaller,
    server_tool_caller,
)
from fused_memory.models.scope import resolve_project_id

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_REFUSED = 2

NO_WRITING_RUN = 'no link-heal run has written here'
STATUS_COUNTS = ('planned', 'applied', 'failed', 'skipped_stale', 'skipped_cap')


def load_config(config_path: str | None) -> FusedMemoryConfig:
    """The config as its file reads now; ``$CONFIG_PATH`` when *config_path* is None."""
    with override_config_path(config_path):
        return FusedMemoryConfig()


def config_ledger_dir(config: FusedMemoryConfig) -> Path:
    return Path(config.reconciliation.data_dir).resolve()


def home_project_root(config: FusedMemoryConfig) -> str:
    """The fused-memory server's own project root, absolute."""
    taskmaster = config.taskmaster or TaskmasterConfig()
    return str(Path(taskmaster.project_root).expanduser().resolve())


@contextlib.asynccontextmanager
async def qdrant_census(config: FusedMemoryConfig) -> AsyncIterator[LinkCensus]:
    backend = Mem0Backend(config)
    try:
        yield QdrantLinkCensus(backend, config.mem0.collection_prefix)
    finally:
        await backend.close()


def config_adjudicator(config: FusedMemoryConfig) -> LinkAdjudicator:
    return configured_adjudicator(config.link_heal)


@dataclass(frozen=True)
class LinkHealEnv:
    """What a command reaches beyond its arguments. The defaults are production's.

    ``home_root_for`` names the server's own project: escapes file into its queue,
    and a run with no project of its own probes it.
    """

    config_loader: Callable[[str | None], FusedMemoryConfig] = load_config
    tool_caller_for: Callable[[str], AbstractAsyncContextManager[ToolCaller]] = (
        server_tool_caller
    )
    census_for: Callable[[FusedMemoryConfig], AbstractAsyncContextManager[LinkCensus]] = (
        qdrant_census
    )
    ledger_dir_for: Callable[[FusedMemoryConfig], Path] = config_ledger_dir
    home_root_for: Callable[[FusedMemoryConfig], str] = home_project_root
    adjudicator_for: Callable[[FusedMemoryConfig], LinkAdjudicator] = config_adjudicator


class Refused(Exception):
    """A command refused before any run started."""


REFUSALS = (
    Refused, RunLockHeld, ApprovalMismatch, UnknownRun, AmbiguousRun, CorpusFormatError,
    CensusFailed,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='link_heal.py', description='Heal hand-linked mem0 records (PRD H1, H2).',
    )
    parser.add_argument('--config', help='fused-memory config YAML (default: $CONFIG_PATH)')
    parser.add_argument(
        '--server-url', help='fused-memory server (default: http://127.0.0.1:<server.port>)',
    )
    commands = parser.add_subparsers(dest='command', required=True)
    plan = commands.add_parser(
        'plan', help='ledger the heals a corpus or the adjudicator supports; no writes',
    )
    source = plan.add_mutually_exclusive_group(required=True)
    source.add_argument('--from-corpus', type=Path, metavar='PATH')
    source.add_argument('--from-adjudicator', action='store_true')
    plan.add_argument(
        '--project', action='append', dest='projects', metavar='PROJECT',
        help='with --from-adjudicator: a project to plan (repeatable; default: the home project)',
    )
    plan.add_argument('--out', type=Path, metavar='PATH', help='where to write the plan')
    apply = commands.add_parser('apply', help='heal the pending plan')
    apply.add_argument(
        '--from-adjudicator', action='store_true', help="heal the adjudicator's pending plan",
    )
    apply.add_argument('--approved-plan-sha', metavar='SHA256')
    undo = commands.add_parser('undo', help="take back an apply run's heals")
    undo.add_argument('--run', required=True, metavar='RUN_ID')
    status = commands.add_parser('status', help='list recent runs')
    status.add_argument('--limit', type=int, default=10)
    return parser


def exit_code(report: RunReport) -> int:
    """0 when complete or partial only by the cap or stale skips; 1 otherwise."""
    counts = report.counts
    failed = (
        counts.failed,
        counts.read_failed,
        counts.not_attempted,
        counts.adjudication_failed,
        counts.stopped_by,
    )
    return EXIT_FAILED if any(failed) else EXIT_OK


def report_document(report: RunReport) -> dict[str, Any]:
    return {
        'run_id': report.run_id,
        'outcome': 'complete' if report.counts.complete else 'partial',
        'plan_path': None if report.plan_path is None else str(report.plan_path),
        'plan_sha256': report.plan_sha256,
        'counts': report.counts.as_json(),
    }


def render_status(ledger_path: Path, limit: int) -> str:
    lines = [f'ledger: {ledger_path}']
    if not ledger_path.exists():
        return '\n'.join([*lines, NO_WRITING_RUN])
    ledger = LinkHealLedger(ledger_path)
    try:
        lines.extend(_status_line(run) for run in ledger.recent_runs(limit))
        if ledger.writing_run_count() == 0:
            lines.append(NO_WRITING_RUN)
    finally:
        ledger.close()
    return '\n'.join(lines)


def _status_line(run: RunRow) -> str:
    counts = run.counts or {}
    tallies = '  '.join(f'{name} {counts.get(name, 0)}' for name in STATUS_COUNTS)
    return (
        f'{run.run_id[:8]}  {run.source.value}  writes {"yes" if run.writes else "no"}  '
        f'finished {run.finished_at or "never"}  {tallies}'
    )


@dataclass(frozen=True)
class _Session:
    """What a run command works with once it holds the lock."""

    config: FusedMemoryConfig
    env: LinkHealEnv
    store: LinkHealStore
    ledger: LinkHealLedger
    ledger_dir: Path
    server_url: str

    @property
    def limits(self) -> RunLimits:
        return RunLimits.from_config(self.config.link_heal)

    @property
    def home_root(self) -> str:
        return self.env.home_root_for(self.config)

    async def probe(self, first_project: str | None) -> None:
        """Refuse unless the server reads the run's first project, else its own."""
        project_id = first_project or resolve_project_id(self.home_root)
        try:
            await self.store.probe(project_id)
        except StoreUnreachable as failure:
            raise Refused(
                f'the fused-memory server at {self.server_url} did not answer a read of '
                f'{project_id} ({failure.error_type}: {failure.detail}); no run was started',
            ) from failure


def _first_project(rows: Sequence[ActionRow]) -> str | None:
    return rows[0].planned.project_id if rows else None


def _load_corpus(path: Path) -> tuple[LinkBasis, ...]:
    try:
        return load_corpus_bases(path)
    except OSError as exc:
        raise Refused(f'cannot read the corpus {path}: {exc}') from exc


def _plan_path_for(out: Path | None, ledger_dir: Path) -> Callable[[str], Path]:
    if out is None:
        return lambda run_id: ledger_dir / f'link-heal-plan-{run_id[:8]}.json'
    if not out.parent.is_dir():
        raise Refused(f'cannot write the plan to {out}: {out.parent} is not a directory')
    return lambda _run_id: out


async def _plan(args: argparse.Namespace, session: _Session) -> RunReport:
    if args.from_adjudicator:
        return await _plan_from_adjudicator(args, session)
    if args.projects:
        raise Refused('--project applies only to --from-adjudicator')
    return await _plan_from_corpus(args, session)


async def _plan_from_adjudicator(args: argparse.Namespace, session: _Session) -> RunReport:
    plan_path_for = _plan_path_for(args.out, session.ledger_dir)
    projects = list(dict.fromkeys(args.projects or [resolve_project_id(session.home_root)]))
    await session.probe(projects[0])
    async with session.env.census_for(session.config) as census:
        return await run_adjudicator_plan(
            adjudicate=session.env.adjudicator_for(session.config),
            store=session.store,
            census=census,
            ledger=session.ledger,
            limits=session.limits,
            projects=projects,
            plan_path_for=plan_path_for,
        )


async def _plan_from_corpus(args: argparse.Namespace, session: _Session) -> RunReport:
    bases = _load_corpus(args.from_corpus)
    plan_path_for = _plan_path_for(args.out, session.ledger_dir)
    projects = list(dict.fromkeys(basis.project_id for basis in bases))
    await session.probe(projects[0] if projects else None)
    async with session.env.census_for(session.config) as census:
        report = await run_plan(
            bases,
            store=session.store,
            census=census,
            ledger=session.ledger,
            limits=session.limits,
            projects=projects,
            source=RunSource.CORPUS,
            plan_path_for=plan_path_for,
        )
    return report


async def _apply(args: argparse.Namespace, session: _Session) -> RunReport:
    source = RunSource.ADJUDICATOR if args.from_adjudicator else RunSource.CORPUS
    await session.probe(_first_project(session.ledger.pending_actions(source)))
    return await run_apply(
        store=session.store,
        ledger=session.ledger,
        limits=session.limits,
        filer=FoldedEscapeFiler(session.home_root),
        source=source,
        approved_plan_sha256=args.approved_plan_sha,
    )


async def _undo(args: argparse.Namespace, session: _Session) -> RunReport:
    target = session.ledger.resolve_run(args.run)
    await session.probe(_first_project(session.ledger.applied_actions(target.run_id)))
    return await run_undo(
        target, store=session.store, ledger=session.ledger, limits=session.limits,
    )


_RUN_COMMANDS = {'plan': _plan, 'apply': _apply, 'undo': _undo}


async def _run_command(
    args: argparse.Namespace, config: FusedMemoryConfig, env: LinkHealEnv, ledger_dir: Path,
) -> RunReport:
    server_url = args.server_url or f'http://127.0.0.1:{config.server.port}'
    ledger_dir.mkdir(parents=True, exist_ok=True)
    with RunLock(ledger_dir / LOCK_FILENAME):
        ledger = LinkHealLedger(ledger_dir / LEDGER_FILENAME)
        try:
            async with env.tool_caller_for(server_url) as tool_caller:
                session = _Session(
                    config=config,
                    env=env,
                    store=LinkHealStore(tool_caller),
                    ledger=ledger,
                    ledger_dir=ledger_dir,
                    server_url=server_url,
                )
                return await _RUN_COMMANDS[args.command](args, session)
        finally:
            ledger.close()


async def run(argv: Sequence[str], env: LinkHealEnv) -> int:
    args = build_parser().parse_args(argv)
    config = env.config_loader(args.config)
    ledger_dir = env.ledger_dir_for(config)
    if args.command == 'status':
        print(render_status(ledger_dir / LEDGER_FILENAME, args.limit))
        return EXIT_OK
    try:
        report = await _run_command(args, config, env, ledger_dir)
    except REFUSALS as refusal:
        print(f'link-heal {args.command} refused: {refusal}', file=sys.stderr)
        return EXIT_REFUSED
    print(json.dumps(report_document(report), indent=2, sort_keys=True))
    return exit_code(report)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    return asyncio.run(run(sys.argv[1:], LinkHealEnv()))


if __name__ == '__main__':
    sys.exit(main())
