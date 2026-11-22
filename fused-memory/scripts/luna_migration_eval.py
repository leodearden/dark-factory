#!/usr/bin/env python3
"""Draw, run, worksheet and score task 9945's gpt-4o-mini vs gpt-6-luna eval, one call site at a time.

A thin CLI over luna_eval_frame.py, luna_eval_sites.py and luna_eval_score.py.
The extraction site is replayed and worksheeted by
replay_overreach_extraction.py itself, over the frame.json this CLI draws;
``score`` reads that replay back through score_replay, so the extraction leg
has one implementation.

Usage (from fused-memory/, after ``set -a; . /home/leo/src/dark-factory/.env; set +a``),
with ``EVAL=../docs/luna-migration-eval-2026-11``::

    uv run python scripts/luna_migration_eval.py frame --dir $EVAL \\
        --graph dark_factory --graph reify --graph know_live --graph market_predictor \\
        --graph autopilot_video --graph pump_web_ui --graph scoping_intake \\
        --graph solar_challenge --graph solar_challenge_platform --graph stream_touch \\
        --extraction-window 2026-08-01T00:00:00Z 2026-11-18T00:00:00Z --cap 3 \\
        --fixture-window 2026-10-01T00:00:00Z 2026-11-18T00:00:00Z \\
        --resolve-n 200 --summaries-n 80 --nodes-per-episode 2 \\
        --journal /home/leo/src/dark-factory/data/reconciliation/write_journal.db \\
        --journal-window 2026-11-04T00:00:00Z 2026-11-18T00:00:00Z \\
        --classifier-per-category 50 --router-n 300 \\
        --worksheet /tmp/9945-worksheet.jsonl --summary-inputs /tmp/9945-summary-inputs.jsonl

    uv run python scripts/replay_overreach_extraction.py replay \\
        --worksheet /tmp/9945-worksheet.jsonl --baseline-report $EVAL/frame.json \\
        --arm control --arm luna-low --out $EVAL/replay.jsonl

    uv run python scripts/luna_migration_eval.py run --site resolve_edge --dir $EVAL \\
        --arm control --arm luna-low --out $EVAL/runs.jsonl
    uv run python scripts/luna_migration_eval.py run --site summaries --dir $EVAL \\
        --summary-inputs /tmp/9945-summary-inputs.jsonl \\
        --arm control --arm luna-low --out $EVAL/runs.jsonl
    uv run python scripts/luna_migration_eval.py run --site classifier --dir $EVAL \\
        --arm control --arm luna-low --out $EVAL/runs.jsonl
    uv run python scripts/luna_migration_eval.py run --site router --dir $EVAL \\
        --arm control --arm luna-low --out $EVAL/runs.jsonl

    uv run python scripts/replay_overreach_extraction.py worksheet \\
        --replay $EVAL/replay.jsonl --worksheet /tmp/9945-worksheet.jsonl \\
        --out /tmp/9945-blind-extraction.jsonl
    uv run python scripts/luna_migration_eval.py worksheet --kind resolve_gold --dir $EVAL \\
        --out /tmp/9945-resolve-gold.jsonl
    uv run python scripts/luna_migration_eval.py worksheet --kind summaries --dir $EVAL \\
        --runs $EVAL/runs.jsonl --summary-inputs /tmp/9945-summary-inputs.jsonl \\
        --out /tmp/9945-summaries.jsonl --key-out /tmp/9945-ratings-summaries-key.json
    uv run python scripts/luna_migration_eval.py worksheet --kind routing --dir $EVAL \\
        --runs $EVAL/runs.jsonl --out /tmp/9945-routing.jsonl

    uv run python scripts/luna_migration_eval.py score --dir $EVAL \\
        --calls-per-day /tmp/9945-calls-per-day.json --plan-window-usd-per-week 280 470 \\
        --out $EVAL/report.json

``frame`` writes frame.json and the four fixture files into --dir, and every
file carrying episode bodies (the extraction worksheet, the summary inputs)
only outside it. ``run`` resumes one runs.jsonl shared by every site, keyed
(site, arm, fixture), and retries failure lines. The summary key is moved
into --dir as ratings-summaries-key.json only after rating. ``score`` reads
every input from --dir by the names below, and calls-per-day as
``{site: calls}``.

Exit codes, as replay_overreach_extraction.py's: 0 ran; 1 the LLM or graph is
unavailable, or a fixture failed (re-run to resume); 2 an input is invalid,
or an arm's client cannot be built (nothing is written).
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import sys
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import httpx
from openai import OpenAIError

from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.models.enums import MemoryCategory, QueryType
from fused_memory.routing.openai_client import RoutingEndpointUnset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _graph_script_cli import (
    GRAPH_UNAVAILABLE,
    ReaderFactory,
    UnreadableInput,
    add_graph_arguments,
    live_reader_factory,
    read_json,
    write_files_atomically,
)
from audit_ruling_overreach import SampleDefinition
from audit_ruling_overreach import VerdictError as ExtractionVerdictError
from luna_eval_frame import (
    EXTRACTION_STRATA,
    FrameReadIncomplete,
    SummaryInputsMismatch,
    check_summary_inputs,
    classifier_fixtures,
    extraction_frame,
    resolve_edge_fixtures,
    router_fixtures,
    summary_frame,
)
from luna_eval_score import (
    RunFileError,
    VerdictError,
    build_report,
    cost_block,
    load_resolve_gold,
    load_routing_verdicts,
    load_summary_ratings,
    resolve_edge_decision,
    resolve_gold_worksheet,
    routing_discordant_worksheet,
    routing_label_decision,
    routing_layer_decision,
    site_block,
    summary_decision,
    summary_rating_worksheet,
)
from luna_eval_sites import (
    SITE_ARMS,
    Site,
    SiteRunner,
    UsageRecordingOpenAI,
    fixture_id,
    routing_arm,
    run_classifier,
    run_resolve_edge,
    run_router,
    run_site_fixtures,
    run_summaries,
)
from replay_overreach_extraction import (
    EXIT_INVALID,
    EXIT_OK,
    EXIT_UNAVAILABLE,
    ArmClientMismatch,
    LlmFactory,
    ReplayArm,
    ReplayFileError,
    UnmeasurableClient,
    WorksheetMismatch,
    build_arm_clients,
    build_arm_llm,
    check_arm_client,
    jsonl_text,
    read_jsonl,
)

logger = logging.getLogger('luna_migration_eval')

# --------------------------------------------------------------------------- #
# The eval directory's files
# --------------------------------------------------------------------------- #

FRAME_FILE = 'frame.json'
FIXTURE_FILES: Mapping[Site, str] = MappingProxyType({
    Site.resolve_edge: 'fixtures-resolve-edge.jsonl', Site.summaries: 'fixtures-summaries.json',
    Site.classifier: 'fixtures-classifier.jsonl', Site.router: 'fixtures-router.jsonl',
})
REPLAY_FILE = 'replay.jsonl'
RUNS_FILE = 'runs.jsonl'
EXTRACTION_VERDICTS_FILE = 'verdicts-extraction.json'
RESOLVE_GOLD_FILE = 'gold-resolve-edge.json'
SUMMARY_RATINGS_FILE = 'ratings-summaries.json'
SUMMARY_KEY_FILE = 'ratings-summaries-key.json'
ROUTING_VERDICTS_FILE = 'verdicts-routing.json'

EXTRACTION_SITE = 'extraction'
"""The replay's records, priced beside the sites' runs."""

ROUTING_SITES = (Site.classifier, Site.router)
ROUTING_TEXT: Mapping[Site, str] = MappingProxyType({Site.classifier: 'input', Site.router: 'query'})
"""The fixture field each routing site's production call reads."""
ROUTING_VOCABULARY: Mapping[Site, tuple[str, ...]] = MappingProxyType({
    Site.classifier: tuple(category.value for category in MemoryCategory),
    Site.router: tuple(query_type.value for query_type in QueryType),
})


class PrivateFileInDir(ValueError):
    """A file carrying episode bodies, asked to be written inside the committed eval directory."""


def _private(path: str | Path, eval_dir: Path) -> Path:
    resolved = Path(path)
    if resolved.resolve().is_relative_to(eval_dir.resolve()):
        raise PrivateFileInDir(f'{resolved} carries episode bodies, so it is never written inside {eval_dir}')
    return resolved


def _rows_text(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    """JSONL, or one indented JSON list for a .json file."""
    if path.suffix == '.json':
        return json.dumps(list(rows), indent=2, sort_keys=True) + '\n'
    return jsonl_text(rows)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows = read_json(path) if path.suffix == '.json' else read_jsonl(path)
    if not isinstance(rows, list):
        raise UnreadableInput(f'{path} is not a list of rows')
    return rows


def _site_records(records: Sequence[Mapping[str, Any]], site: Site) -> list[Mapping[str, Any]]:
    return [record for record in records if record.get('site') == site]


def _by_fixture_id(site: Site, fixtures: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {fixture_id(site, fixture): fixture for fixture in fixtures}


def _checked_summary_inputs(eval_dir: Path, inputs_path: str | Path) -> list[dict[str, Any]]:
    inputs = read_jsonl(inputs_path)
    check_summary_inputs(_read_rows(eval_dir / FIXTURE_FILES[Site.summaries]), inputs)
    return inputs


# --------------------------------------------------------------------------- #
# frame
# --------------------------------------------------------------------------- #

Window = tuple[str, str]


async def _journal_fixtures(args: argparse.Namespace) -> dict[Site, list[dict[str, Any]]]:
    window: Window = tuple(args.journal_window)
    return {
        Site.classifier: await classifier_fixtures(args.journal, window, args.classifier_per_category),
        Site.router: await router_fixtures(args.journal, window, args.router_n),
    }


async def _graph_draw(args: argparse.Namespace, reader_factory: ReaderFactory) -> dict[str, Any]:
    definition = SampleDefinition(
        window_start=args.extraction_window[0], window_end=args.extraction_window[1], cap=args.cap,
        strata=EXTRACTION_STRATA,
    )
    window: Window = tuple(args.fixture_window)
    async with contextlib.AsyncExitStack() as opened:
        readers = {}
        for graph in dict.fromkeys(args.graph):
            readers[graph] = reader_factory(graph)
            opened.push_async_callback(readers[graph].aclose)
        extraction = await extraction_frame(readers, definition)
        summaries = await summary_frame(readers, window, args.summaries_n, args.nodes_per_episode)
        return {
            'extraction': extraction,
            Site.resolve_edge: await resolve_edge_fixtures(readers, window, args.resolve_n),
            Site.summaries: summaries.items,
            'summary_inputs': summaries.inputs,
        }


async def _frame_command(args: argparse.Namespace, reader_factory: ReaderFactory) -> int:
    eval_dir = Path(args.dir)
    try:
        worksheet, inputs_path = _private(args.worksheet, eval_dir), _private(args.summary_inputs, eval_dir)
        journal = await _journal_fixtures(args)
    except (PrivateFileInDir, FileNotFoundError) as exc:
        logger.error('invalid inputs, nothing is written: %s', exc)
        return EXIT_INVALID
    try:
        drawn = await _graph_draw(args, reader_factory)
    except (FrameReadIncomplete, *GRAPH_UNAVAILABLE) as exc:
        logger.error('a graph cannot be read completely, nothing is written: %s', exc)
        return EXIT_UNAVAILABLE
    fixtures = {Site.resolve_edge: drawn[Site.resolve_edge], Site.summaries: drawn[Site.summaries], **journal}
    for site, rows in fixtures.items():
        logger.info('frame: %d %s fixture(s)', len(rows), site)
    logger.info('frame: %d extraction episode(s)', len(drawn['extraction'].worksheet))
    texts = {eval_dir / FRAME_FILE: json.dumps(drawn['extraction'].frame, indent=2, sort_keys=True) + '\n'}
    texts |= {eval_dir / FIXTURE_FILES[site]: _rows_text(eval_dir / FIXTURE_FILES[site], rows)
              for site, rows in fixtures.items()}
    texts |= {worksheet: jsonl_text(drawn['extraction'].worksheet), inputs_path: jsonl_text(drawn['summary_inputs'])}
    write_files_atomically(texts)
    return EXIT_OK


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Seams:
    """Everything a run reaches outside this process, each defaulting to the live one."""

    llm_factory: LlmFactory | None = None
    reader_factory: ReaderFactory | None = None
    environ: Mapping[str, str] = field(default_factory=lambda: os.environ)
    config: FusedMemoryConfig | None = None
    routing_transport: httpx.AsyncBaseTransport | None = None

    def loaded_config(self) -> FusedMemoryConfig:
        return self.config or FusedMemoryConfig()

    def graphiti_llm_factory(self) -> LlmFactory:
        if self.llm_factory is not None:
            return self.llm_factory
        config = self.loaded_config()
        return lambda llm: build_arm_llm(config, llm, self.environ)


class ArmUnavailable(LookupError):
    """An arm's client factory built nothing."""


async def _graphiti_runner(site: Site, arms: Sequence[ReplayArm], seams: Seams) -> SiteRunner:
    clients = build_arm_clients(arms, seams.graphiti_llm_factory())
    if clients is None:
        raise ArmUnavailable(f'no client for one of {[arm.name for arm in arms]}')
    for arm, llm in clients:
        await check_arm_client(llm, arm)
    by_arm = {arm.name: llm for arm, llm in clients}
    if site is Site.resolve_edge:
        return lambda arm, fixture: run_resolve_edge(by_arm[arm.name], fixture, arm)
    return lambda arm, fixture: run_summaries(by_arm[arm.name], fixture, fixture['previous'], arm)


def _routing_runner(
    site: Site, arms: Sequence[ReplayArm], seams: Seams, opened: contextlib.AsyncExitStack,
) -> SiteRunner:
    config = seams.loaded_config()
    by_arm: dict[str, tuple[FusedMemoryConfig, UsageRecordingOpenAI]] = {}
    for arm in arms:
        arm_cfg, client = routing_arm(config, arm, seams.environ, transport=seams.routing_transport)
        opened.push_async_callback(client.close)
        by_arm[arm.name] = (arm_cfg, client)
    call = run_classifier if site is Site.classifier else run_router

    def runner(arm: ReplayArm, fixture: Mapping[str, Any]) -> Awaitable[dict[str, Any]]:
        arm_cfg, client = by_arm[arm.name]
        return call(arm_cfg, client, fixture, arm)

    return runner


async def _run_command(args: argparse.Namespace, seams: Seams) -> int:
    site, eval_dir = Site(args.site), Path(args.dir)
    arms = [SITE_ARMS[name] for name in dict.fromkeys(args.arm)]
    try:
        fixtures = (
            _checked_summary_inputs(eval_dir, args.summary_inputs) if site is Site.summaries
            else _read_rows(eval_dir / FIXTURE_FILES[site])
        )
    except (ReplayFileError, UnreadableInput, SummaryInputsMismatch) as exc:
        logger.error('invalid inputs, nothing is written: %s', exc)
        return EXIT_INVALID
    async with contextlib.AsyncExitStack() as opened:
        try:
            runner = (
                _routing_runner(site, arms, seams, opened) if site in ROUTING_SITES
                else await _graphiti_runner(site, arms, seams)
            )
        except (ArmClientMismatch, UnmeasurableClient, RoutingEndpointUnset, OpenAIError) as exc:
            logger.error('an arm cannot be run on its client, nothing is written: %s', exc)
            return EXIT_INVALID
        except ArmUnavailable as exc:
            logger.error('%s; nothing is written', exc)
            return EXIT_UNAVAILABLE
        failures = await run_site_fixtures(site, fixtures, arms, runner, Path(args.out))
    if failures:
        logger.error('%d (arm, fixture) run(s) of %s failed; re-run to resume them', failures, site)
        return EXIT_UNAVAILABLE
    return EXIT_OK


# --------------------------------------------------------------------------- #
# worksheet
# --------------------------------------------------------------------------- #


def _routing_worksheet(eval_dir: Path, records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Both routing sites' discordant items, each with its own site's vocabulary."""
    rows = []
    for site in ROUTING_SITES:
        texts = {fid: fixture[ROUTING_TEXT[site]]
                 for fid, fixture in _by_fixture_id(site, _read_rows(eval_dir / FIXTURE_FILES[site])).items()}
        rows += routing_discordant_worksheet(texts, _site_records(records, site), ROUTING_VOCABULARY[site])
    return rows


def _worksheet_texts(args: argparse.Namespace) -> dict[Path, str]:
    eval_dir, out = Path(args.dir), Path(args.out)
    if args.kind == 'resolve_gold':
        fixtures = _by_fixture_id(Site.resolve_edge, _read_rows(eval_dir / FIXTURE_FILES[Site.resolve_edge]))
        return {out: jsonl_text(resolve_gold_worksheet(fixtures))}
    if args.kind == 'routing':
        return {out: jsonl_text(_routing_worksheet(eval_dir, read_jsonl(args.runs)))}
    out = _private(out, eval_dir)
    inputs = _by_fixture_id(Site.summaries, _checked_summary_inputs(eval_dir, args.summary_inputs))
    rows, key = summary_rating_worksheet(inputs, _site_records(read_jsonl(args.runs), Site.summaries))
    return {out: jsonl_text(rows), Path(args.key_out): json.dumps(key, indent=2, sort_keys=True) + '\n'}


def _worksheet_command(args: argparse.Namespace) -> int:
    try:
        texts = _worksheet_texts(args)
    except (ReplayFileError, UnreadableInput, RunFileError, SummaryInputsMismatch, PrivateFileInDir) as exc:
        logger.error('invalid inputs, nothing is written: %s', exc)
        return EXIT_INVALID
    write_files_atomically(texts)
    return EXIT_OK


# --------------------------------------------------------------------------- #
# score
# --------------------------------------------------------------------------- #


def _read_text(path: Path) -> str:
    try:
        return path.read_text()
    except OSError as exc:
        raise UnreadableInput(f'cannot read {path}: {exc}') from exc


def _calls_per_day(obj: Any) -> dict[str, float]:
    if not isinstance(obj, dict) or not all(
        isinstance(site, str) and isinstance(calls, int | float) for site, calls in obj.items()
    ):
        raise UnreadableInput('calls-per-day is {site: calls per day}')
    return {site: float(calls) for site, calls in obj.items()}


def _registered_records(eval_dir: Path, runs: Sequence[Mapping[str, Any]], site: Site) -> list[Mapping[str, Any]]:
    """*site*'s run records, once they cover exactly the fixtures its committed fixture file
    pre-registers. A run killed between fixtures leaves no failure line, so without this a
    decision would quietly be taken over the subset that ran."""
    records = _site_records(runs, site)
    registered = _by_fixture_id(site, _read_rows(eval_dir / FIXTURE_FILES[site])).keys()
    ran = {record['fixture_id'] for record in records}
    if unran := sorted(registered - ran):
        raise RunFileError(f'{site}: {len(unran)} pre-registered fixture(s) have no run line under any arm: {unran}')
    if unregistered := sorted(ran - registered):
        raise RunFileError(f'{site}: {len(unregistered)} run fixture(s) are not pre-registered: {unregistered}')
    return records


def _site_decisions(eval_dir: Path, runs: Sequence[Mapping[str, Any]]) -> dict[Site, dict[str, Any]]:
    records = {site: _registered_records(eval_dir, runs, site) for site in Site}
    resolve = _by_fixture_id(Site.resolve_edge, _read_rows(eval_dir / FIXTURE_FILES[Site.resolve_edge]))
    gold = load_resolve_gold(read_json(eval_dir / RESOLVE_GOLD_FILE), resolve_gold_worksheet(resolve))
    ratings = load_summary_ratings(
        read_json(eval_dir / SUMMARY_RATINGS_FILE), records[Site.summaries], read_json(eval_dir / SUMMARY_KEY_FILE),
    )
    routing = load_routing_verdicts(read_json(eval_dir / ROUTING_VERDICTS_FILE), _routing_worksheet(eval_dir, runs))
    return {
        Site.resolve_edge: resolve_edge_decision(records[Site.resolve_edge], gold),
        Site.summaries: summary_decision(records[Site.summaries], ratings),
        **{site: routing_label_decision(records[site], routing.get(site, {})) for site in ROUTING_SITES},
    }


def _report(args: argparse.Namespace) -> dict[str, Any]:
    eval_dir = Path(args.dir)
    runs = read_jsonl(eval_dir / RUNS_FILE)
    replayed = read_jsonl(eval_dir / REPLAY_FILE)
    decisions = _site_decisions(eval_dir, runs)
    cost = cost_block(
        [*runs, *({**record, 'site': EXTRACTION_SITE} for record in replayed)],
        _calls_per_day(read_json(args.calls_per_day)), tuple(args.plan_window_usd_per_week),
    )
    return build_report(
        frame_text=_read_text(eval_dir / FRAME_FILE), extraction_replay=replayed,
        extraction_verdicts=read_json(eval_dir / EXTRACTION_VERDICTS_FILE),
        sites={site.value: site_block(_site_records(runs, site), decision) for site, decision in decisions.items()},
        routing_layer=routing_layer_decision(decisions[Site.classifier], decisions[Site.router]), cost=cost,
    )


def _score_command(args: argparse.Namespace) -> int:
    try:
        report = _report(args)
    except (
        ReplayFileError, UnreadableInput, WorksheetMismatch, ExtractionVerdictError, RunFileError, VerdictError,
    ) as exc:
        logger.error('invalid inputs, nothing is written (%s): %s', type(exc).__name__, exc)
        return EXIT_INVALID
    write_files_atomically({Path(args.out): json.dumps(report, indent=2, sort_keys=True) + '\n'})
    return EXIT_OK


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

WORKSHEET_KINDS = ('resolve_gold', 'summaries', 'routing')

REQUIRED_WITH: Mapping[tuple[str, str], tuple[str, ...]] = MappingProxyType({
    ('run', Site.summaries.value): ('summary_inputs',),
    ('worksheet', 'summaries'): ('runs', 'summary_inputs', 'key_out'),
    ('worksheet', 'routing'): ('runs',),
})
"""(command, --site or --kind) -> the options it cannot run without."""


def _window_argument(parser: argparse.ArgumentParser, name: str, what: str) -> None:
    parser.add_argument(name, nargs=2, required=True, metavar=('START', 'END'), help=f'{what}: [START, END).')


def _frame_parser(commands: Any) -> None:
    frame = commands.add_parser('frame', help='Draw frame.json, the fixtures and the private inputs.')
    frame.add_argument('--dir', required=True, help='The committed eval directory.')
    frame.add_argument('--graph', action='append', required=True)
    add_graph_arguments(frame)
    _window_argument(frame, '--extraction-window', 'Extraction episodes created in')
    frame.add_argument('--cap', type=int, required=True, help='Extraction episodes per (graph, stratum).')
    _window_argument(frame, '--fixture-window', 'resolve_edge edges and summary episodes created in')
    frame.add_argument('--resolve-n', type=int, required=True)
    frame.add_argument('--summaries-n', type=int, required=True)
    frame.add_argument('--nodes-per-episode', type=int, required=True)
    frame.add_argument('--journal', required=True, help='The write journal, opened read-only.')
    _window_argument(frame, '--journal-window', 'Journal rows created in')
    frame.add_argument('--classifier-per-category', type=int, required=True)
    frame.add_argument('--router-n', type=int, required=True)
    frame.add_argument('--worksheet', required=True, help='The extraction worksheet (bodies), outside --dir.')
    frame.add_argument('--summary-inputs', required=True, help='The restored summary inputs (bodies), outside --dir.')


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='luna_migration_eval', description="Task 9945's per-call-site gpt-4o-mini vs gpt-6-luna eval.",
    )
    commands = parser.add_subparsers(dest='command', required=True)
    _frame_parser(commands)
    run = commands.add_parser('run', help="Run one site's fixtures under each arm.")
    run.add_argument('--site', required=True, choices=[site.value for site in Site])
    run.add_argument('--dir', required=True)
    run.add_argument('--summary-inputs', help='The frame\'s summary inputs; --site summaries runs on them.')
    run.add_argument('--arm', action='append', required=True, choices=sorted(SITE_ARMS))
    run.add_argument('--out', required=True, help='runs.jsonl; an existing file is resumed.')
    worksheet = commands.add_parser('worksheet', help='Emit one blind adjudication worksheet.')
    worksheet.add_argument('--kind', required=True, choices=WORKSHEET_KINDS)
    worksheet.add_argument('--dir', required=True)
    worksheet.add_argument('--runs', help='runs.jsonl, for --kind summaries and routing.')
    worksheet.add_argument('--summary-inputs', help='For --kind summaries.')
    worksheet.add_argument('--key-out', help="--kind summaries' A/B key, opened only after rating.")
    worksheet.add_argument('--out', required=True)
    score = commands.add_parser('score', help='Decide every site against its pre-registered bar.')
    score.add_argument('--dir', required=True)
    score.add_argument('--calls-per-day', required=True, help='JSON {site: calls per day}, for cost.')
    score.add_argument('--plan-window-usd-per-week', nargs=2, type=float, required=True, metavar=('LOW', 'HIGH'))
    score.add_argument('--out', required=True, help='report.json to write.')
    return parser


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = _build_parser()
    args = parser.parse_args(argv)
    variant = getattr(args, 'site', None) or getattr(args, 'kind', None)
    if missing := [name for name in REQUIRED_WITH.get((args.command, variant or ''), ()) if not getattr(args, name)]:
        parser.error(f"{args.command} {variant} requires {', '.join('--' + m.replace('_', '-') for m in missing)}")
    return args


async def _run(args: argparse.Namespace, seams: Seams) -> int:
    if args.command == 'frame':
        return await _frame_command(args, seams.reader_factory or live_reader_factory(args))
    if args.command == 'run':
        return await _run_command(args, seams)
    if args.command == 'worksheet':
        return _worksheet_command(args)
    return _score_command(args)


def run(
    argv: Sequence[str] | None = None, *,
    llm_factory: LlmFactory | None = None, reader_factory: ReaderFactory | None = None,
    environ: Mapping[str, str] | None = None, config: FusedMemoryConfig | None = None,
    routing_transport: httpx.AsyncBaseTransport | None = None,
) -> int:
    """Parse *argv* (default ``sys.argv[1:]``) and run one subcommand.

    *llm_factory* maps a graphiti site's arm ExtractionLlm to its client (default:
    build_arm_llm over *config* and *environ*); *reader_factory* maps a graph name to a
    ReadOnlyGraphReader (default: FalkorDB over GRAPH.RO_QUERY); *environ* supplies the
    plan proxy's variables (default: the process environment); *config* is the loaded
    FusedMemoryConfig the routing arms are built on; *routing_transport* replaces the
    routing clients' network transport.
    """
    seams = Seams(
        llm_factory=llm_factory, reader_factory=reader_factory,
        environ=os.environ if environ is None else environ, config=config, routing_transport=routing_transport,
    )
    return asyncio.run(_run(_parse(argv), seams))


def main() -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format='%(levelname)s %(name)s: %(message)s')
    return run()


if __name__ == '__main__':
    sys.exit(main())

