"""The link-heal CLI (tasks 6181, 6184, 7159; plans/write-triage-link-healing-prd.md H1, H2).

``scripts/link_heal.py`` is driven through ``await run(argv, env)``. The env
swaps in the in-process server from ``_link_heal_harness`` as the transport,
its ``FakeMem0`` as the census, a ``FakeAdjudicator`` as the link adjudicator,
and tmp directories for the ledger and the escalation queue. Config is read
from a tmp YAML by the script's own loader, at every run.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import shlex
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio
import yaml
from _fm_helpers import load_script_module
from _link_heal_harness import (
    ALL_LINK_HEAL_PREFIXES,
    COMPLETION_META,
    DF,
    FAKE_ADJUDICATOR_MODEL,
    FAKE_PROMPT_SHA256,
    REIFY,
    WITHOUT_LINK_HEAL_PREFIX,
    Answer,
    FailingCensus,
    FakeAdjudicator,
    LinkHealHarness,
    assert_store_invariants,
    build_harness,
)
from escalation.queue import EscalationQueue

from fused_memory.maintenance.link_adjudicator import AdjudicationFailure
from fused_memory.maintenance.link_heal import RunCounts, Verdict
from fused_memory.maintenance.link_heal_calibration import load_soak_population
from fused_memory.maintenance.link_heal_executor import (
    ADJUDICATOR_ANCHOR,
    CALIBRATION_ANCHOR,
    RUN_REPORT_ANCHOR,
    SHARE_STOP,
    RunReport,
    undo_command,
)
from fused_memory.maintenance.link_heal_ledger import (
    LEDGER_FILENAME,
    LOCK_FILENAME,
    LinkHealLedger,
    RunLock,
    RunRow,
    RunSource,
)
from fused_memory.maintenance.link_heal_store import (
    READ_TOOL,
    LinkCensus,
    ToolCaller,
    text_sha256,
)
from fused_memory.models.scope import resolve_project_id
from fused_memory.server.grouped_read import SIGHTING_KIND

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'link_heal.py'
REPO_ROOT = SCRIPT_PATH.parents[2]

cli = load_script_module(SCRIPT_PATH, mod_name='link_heal_cli')

NO_WRITING_RUN = 'no link-heal run has written here'
SERVER_PORT = 8765
DEFAULT_SERVER_URL = f'http://127.0.0.1:{SERVER_PORT}'
WRONG_SHA = 'deadbeef' * 8
PENDING_NOW = 'pending now'
STALE = 'stale (pending heals changed since; re-plan)'
NOTHING_PENDING = 'nothing pending'


@dataclasses.dataclass
class Workspace:
    """The tmp directories and config file one CLI test runs against."""

    root: Path

    @property
    def config_path(self) -> Path:
        return self.root / 'fused-memory.yaml'

    @property
    def ledger_dir(self) -> Path:
        return self.root / 'ledger'

    @property
    def ledger_path(self) -> Path:
        return self.ledger_dir / LEDGER_FILENAME

    @property
    def home_root(self) -> Path:
        return self.root / 'home-project'

    @property
    def data_dir(self) -> Path:
        return self.root / 'reconciliation'

    @property
    def reports_dir(self) -> Path:
        return self.root / 'reports'

    def write_config(
        self, *, max_actions_per_run: int = 25, link_heal: Mapping[str, Any] | None = None,
    ) -> None:
        self.config_path.write_text(yaml.safe_dump({
            'server': {'port': SERVER_PORT},
            'taskmaster': {'project_root': str(self.root / 'dark-factory')},
            'reconciliation': {'data_dir': str(self.data_dir)},
            'link_heal': {
                'max_actions_per_run': max_actions_per_run,
                'backlog_multiplier': 5,
                'write_failure_streak': 3,
                **(link_heal or {}),
            },
        }))

    def pending_escalations(self, anchor: str) -> list[Any]:
        """The home project's pending escalations filed under *anchor*."""
        queue = EscalationQueue(self.home_root / 'data' / 'escalations')
        return [record for record in queue.get_pending() if record.task_id == anchor]

    def runs(self) -> list[RunRow]:
        """Every run the ledger holds, newest first; none when it does not exist."""
        if not self.ledger_path.exists():
            return []
        ledger = LinkHealLedger(self.ledger_path)
        try:
            return ledger.recent_runs(100)
        finally:
            ledger.close()

    def pending(self, source: RunSource) -> int:
        ledger = LinkHealLedger(self.ledger_path)
        try:
            return len(ledger.pending_actions(source))
        finally:
            ledger.close()

    def adjudication_count(self, run_id: str) -> int:
        ledger = LinkHealLedger(self.ledger_path)
        try:
            return len(ledger.adjudication_verdicts({run_id}))
        finally:
            ledger.close()

    def calibrate_fake_adjudicator(self, *, passed: bool) -> str:
        """Ledger a calibration of the fake adjudicator's model and prompt; its run id."""
        self.ledger_dir.mkdir(parents=True, exist_ok=True)
        ledger = LinkHealLedger(self.ledger_path)
        try:
            run_id = ledger.start_run(RunSource.CALIBRATION, writes=False)
            ledger.add_calibration(
                run_id, model=FAKE_ADJUDICATOR_MODEL, prompt_sha256=FAKE_PROMPT_SHA256,
                population_sha256='0' * 64, passed=passed, report={'passed': passed},
            )
            return run_id
        finally:
            ledger.close()


@pytest.fixture
def workspace(tmp_path) -> Workspace:
    made = Workspace(tmp_path)
    made.write_config()
    return made


async def _harness(mock_config, tmp_path: Path, prefixes: list[str]):
    built = await build_harness(mock_config, tmp_path, metadata_patch_prefixes=prefixes)
    yield built
    await built.journal.close()
    assert_store_invariants(built)


@pytest_asyncio.fixture
async def harness(mock_config, tmp_path):
    async for built in _harness(mock_config, tmp_path, ALL_LINK_HEAL_PREFIXES):
        yield built


@pytest_asyncio.fixture
async def unadmitted(mock_config, tmp_path):
    """A server whose metadata allowlist does not admit ``link-heal-``."""
    async for built in _harness(mock_config, tmp_path, WITHOUT_LINK_HEAL_PREFIX):
        yield built


def _never_asked(text: str) -> Answer:
    raise AssertionError(f'the adjudicator was asked about {text!r}')


def env_for(
    harness: LinkHealHarness,
    workspace: Workspace,
    *,
    tool_caller: ToolCaller | None = None,
    server_urls: list[str] | None = None,
    census: LinkCensus | None = None,
    adjudicator: FakeAdjudicator | None = None,
) -> Any:
    """The CLI env over *harness*, with the ledger and the home project under *workspace*.

    The adjudicator defaults to one that fails any test that asks it anything.
    """
    caller = tool_caller or harness.tool_caller()
    adjudicate = adjudicator or FakeAdjudicator(_never_asked)

    def tool_caller_for(server_url: str):
        if server_urls is not None:
            server_urls.append(server_url)
        return contextlib.nullcontext(caller)

    return cli.LinkHealEnv(
        tool_caller_for=tool_caller_for,
        census_for=lambda _config: contextlib.nullcontext(census or harness.mem0),
        ledger_dir_for=lambda _config: workspace.ledger_dir,
        home_root_for=lambda _config: str(workspace.home_root),
        adjudicator_for=lambda _config: adjudicate,
    )


def _sighting_ids(index: int) -> tuple[str, str]:
    return (f'{index:08x}-c1c1-4c1c-8c1c-{index:012x}', f'{index:08x}-a1a1-4a1a-8a1a-{index:012x}')


def seed_corpus(harness: LinkHealHarness, workspace: Workspace, count: int) -> Path:
    """*count* EXTENDS-rated sightings in the store, and the corpus rating them: *count* relabels."""
    rows = []
    for index in range(count):
        child, parent = _sighting_ids(index)
        child_text, parent_text = f'cli sighting {index}', f'cli parent {index}'
        harness.seed_link(
            kind=SIGHTING_KIND, child=child, parent=parent,
            child_text=child_text, parent_text=parent_text,
        )
        rows.append({
            'item_id': f'H{index:03d}',
            'project': DF,
            'entry_id': child,
            'target_id': parent,
            'verdict': 'EXTENDS',
            'child_sha256': text_sha256(child_text),
            'parent_sha256': text_sha256(parent_text),
            'rated_text_matches_live': True,
        })
    path = workspace.root / 'corpus.jsonl'
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return path


async def run_cli(env: Any, workspace: Workspace, *argv: str) -> int:
    return await cli.run(['--config', str(workspace.config_path), *argv], env)


def printed_report(capsys) -> dict[str, Any]:
    return json.loads(capsys.readouterr().out)


def status_lines(capsys) -> list[str]:
    return capsys.readouterr().out.splitlines()


def run_line(lines: list[str], run_id: str) -> str:
    """The status line listing run *run_id* itself."""
    (line,) = [line for line in lines if line.startswith(run_id[:8])]
    return line


def latest_plan_line(lines: list[str], source: RunSource) -> str:
    (line,) = [line for line in lines if line.startswith(f'latest {source.value} plan')]
    return line


async def plan_from(env: Any, workspace: Workspace, corpus: Path, capsys) -> dict[str, Any]:
    assert await run_cli(env, workspace, 'plan', '--from-corpus', str(corpus)) == 0
    return printed_report(capsys)


async def apply(env: Any, workspace: Workspace, capsys, *argv: str) -> tuple[int, dict[str, Any]]:
    code = await run_cli(env, workspace, 'apply', *argv)
    return code, printed_report(capsys)


def refusing_tool_caller() -> ToolCaller:
    async def call(tool: str, arguments: dict[str, Any]) -> dict[str, Any] | None:
        raise httpx.ConnectError('connection refused')

    return call


class TestStatus:
    @pytest.mark.asyncio
    async def test_no_ledger_says_nothing_has_written_and_creates_none(
        self, harness, workspace, capsys,
    ):
        code = await run_cli(env_for(harness, workspace), workspace, 'status')

        out = capsys.readouterr().out
        assert code == 0
        assert NO_WRITING_RUN in out.splitlines()
        assert str(workspace.ledger_path) in out
        assert not workspace.ledger_path.exists()

    @pytest.mark.asyncio
    async def test_a_plan_run_is_listed_as_non_writing(self, harness, workspace, capsys):
        env = env_for(harness, workspace)
        planned = await plan_from(env, workspace, seed_corpus(harness, workspace, 2), capsys)

        assert await run_cli(env, workspace, 'status') == 0

        lines = status_lines(capsys)
        assert 'writes no' in run_line(lines, planned['run_id'])
        assert NO_WRITING_RUN in lines

    @pytest.mark.asyncio
    async def test_a_zero_action_apply_is_a_writing_run(self, harness, workspace, capsys):
        env = env_for(harness, workspace)
        code, applied = await apply(env, workspace, capsys)
        assert code == 0

        assert await run_cli(env, workspace, 'status') == 0

        lines = status_lines(capsys)
        applied_line = run_line(lines, applied['run_id'])
        assert 'writes yes' in applied_line
        assert 'applied 0' in applied_line
        assert NO_WRITING_RUN not in lines

    @pytest.mark.asyncio
    async def test_the_default_ledger_lives_in_the_reconciliation_data_dir(
        self, harness, workspace, capsys,
    ):
        env = dataclasses.replace(env_for(harness, workspace), ledger_dir_for=cli.config_ledger_dir)

        assert await run_cli(env, workspace, 'status') == 0

        assert str(workspace.data_dir.resolve() / LEDGER_FILENAME) in capsys.readouterr().out


class TestPlan:
    @pytest.mark.asyncio
    async def test_plan_prints_the_report_and_writes_the_approvable_document(
        self, harness, workspace, capsys,
    ):
        env = env_for(harness, workspace)

        report = await plan_from(env, workspace, seed_corpus(harness, workspace, 2), capsys)

        counts = report['counts']
        assert counts['planned_by_action'] == {'relabel': 2}
        assert (counts['links_total'], counts['examined'], counts['adjudicated']) == (2, 2, 2)
        assert counts['would_escape'] == []
        assert report['outcome'] == 'complete'
        plan_path = Path(report['plan_path'])
        assert plan_path == workspace.ledger_dir / f'link-heal-plan-{report["run_id"][:8]}.json'
        assert hashlib.sha256(plan_path.read_bytes()).hexdigest() == report['plan_sha256']
        assert harness.mem0.write_count == 0

    @pytest.mark.asyncio
    async def test_out_names_the_plan_document(self, harness, workspace, capsys):
        corpus = seed_corpus(harness, workspace, 1)
        out_path = workspace.root / 'reviewed-plan.json'

        code = await run_cli(
            env_for(harness, workspace), workspace,
            'plan', '--from-corpus', str(corpus), '--out', str(out_path),
        )

        report = printed_report(capsys)
        assert code == 0
        assert Path(report['plan_path']) == out_path
        assert hashlib.sha256(out_path.read_bytes()).hexdigest() == report['plan_sha256']

    @pytest.mark.asyncio
    async def test_a_malformed_corpus_is_refused_before_any_run(
        self, harness, workspace, capsys,
    ):
        corpus = workspace.root / 'corpus.jsonl'
        corpus.write_text('not json\n')

        code = await run_cli(
            env_for(harness, workspace), workspace, 'plan', '--from-corpus', str(corpus),
        )

        assert code == 2
        assert 'line 1' in capsys.readouterr().err
        assert workspace.runs() == []


class TestRefusedBeforeAnyRun:
    @pytest.mark.asyncio
    async def test_plan_refuses_an_unreachable_server_naming_its_url(
        self, harness, workspace, capsys,
    ):
        urls: list[str] = []
        env = env_for(harness, workspace, tool_caller=refusing_tool_caller(), server_urls=urls)

        code = await run_cli(
            env, workspace, 'plan', '--from-corpus', str(seed_corpus(harness, workspace, 1)),
        )

        assert code == 2
        assert urls == [DEFAULT_SERVER_URL]
        assert DEFAULT_SERVER_URL in capsys.readouterr().err
        assert workspace.runs() == []

    @pytest.mark.asyncio
    async def test_apply_refuses_an_unreachable_server_naming_its_url(
        self, harness, workspace, capsys,
    ):
        server_url = 'http://127.0.0.1:59999'
        urls: list[str] = []
        env = env_for(harness, workspace, tool_caller=refusing_tool_caller(), server_urls=urls)

        code = await run_cli(env, workspace, '--server-url', server_url, 'apply')

        assert code == 2
        assert urls == [server_url]
        assert server_url in capsys.readouterr().err
        assert workspace.runs() == []

    @pytest.mark.asyncio
    async def test_a_held_lock_is_refused_naming_its_holder(self, harness, workspace, capsys):
        workspace.ledger_dir.mkdir(parents=True)
        lock_path = workspace.ledger_dir / LOCK_FILENAME

        with RunLock(lock_path):
            holder = json.loads(lock_path.read_text())
            code = await run_cli(
                env_for(harness, workspace), workspace,
                'plan', '--from-corpus', str(seed_corpus(harness, workspace, 1)),
            )

        err = capsys.readouterr().err
        assert code == 2
        assert str(os.getpid()) in err
        assert holder['started_at'] in err
        assert workspace.runs() == []

    @pytest.mark.asyncio
    async def test_a_mismatched_approved_sha_is_refused(self, harness, workspace, capsys):
        env = env_for(harness, workspace)
        await plan_from(env, workspace, seed_corpus(harness, workspace, 2), capsys)

        code = await run_cli(env, workspace, 'apply', '--approved-plan-sha', WRONG_SHA)

        assert code == 2
        assert WRONG_SHA in capsys.readouterr().err
        assert [run.writes for run in workspace.runs()] == [False]
        assert harness.mem0.write_count == 0

    @pytest.mark.asyncio
    async def test_undo_of_an_unknown_run_is_refused(self, harness, workspace, capsys):
        code = await run_cli(env_for(harness, workspace), workspace, 'undo', '--run', 'abcdef12')

        assert code == 2
        assert 'abcdef12' in capsys.readouterr().err
        assert workspace.runs() == []

    @pytest.mark.asyncio
    async def test_plan_refuses_an_out_path_in_a_missing_directory(
        self, harness, workspace, capsys,
    ):
        out_path = workspace.root / 'no-such-dir' / 'plan.json'

        code = await run_cli(
            env_for(harness, workspace), workspace,
            'plan', '--from-corpus', str(seed_corpus(harness, workspace, 1)),
            '--out', str(out_path),
        )

        assert code == 2
        assert str(out_path.parent) in capsys.readouterr().err
        assert workspace.runs() == []

    @pytest.mark.asyncio
    async def test_plan_refuses_a_failed_census_naming_its_failure(
        self, harness, workspace, capsys,
    ):
        env = env_for(harness, workspace, census=FailingCensus())

        code = await run_cli(
            env, workspace, 'plan', '--from-corpus', str(seed_corpus(harness, workspace, 1)),
        )

        assert code == 2
        assert FailingCensus.DETAIL in capsys.readouterr().err
        assert workspace.runs() == []

    @pytest.mark.asyncio
    async def test_undo_of_a_plan_run_is_refused_naming_its_kind(
        self, harness, workspace, capsys,
    ):
        env = env_for(harness, workspace)
        planned = await plan_from(env, workspace, seed_corpus(harness, workspace, 2), capsys)

        code = await run_cli(env, workspace, 'undo', '--run', planned['run_id'][:8])

        err = capsys.readouterr().err
        assert code == 2
        assert planned['run_id'][:8] in err
        assert 'plan run' in err
        assert 'no heal still applied' in err
        assert [run.run_id for run in workspace.runs()] == [planned['run_id']]
        assert harness.mem0.write_count == 0

    @pytest.mark.asyncio
    async def test_undo_of_an_already_undone_apply_is_refused(
        self, harness, workspace, capsys,
    ):
        env = env_for(harness, workspace)
        await plan_from(env, workspace, seed_corpus(harness, workspace, 1), capsys)
        _code, applied = await apply(env, workspace, capsys)
        assert await run_cli(env, workspace, 'undo', '--run', applied['run_id'][:8]) == 0
        printed_report(capsys)
        runs_before = len(workspace.runs())

        code = await run_cli(env, workspace, 'undo', '--run', applied['run_id'][:8])

        err = capsys.readouterr().err
        assert code == 2
        assert 'apply run' in err
        assert 'no heal still applied' in err
        assert len(workspace.runs()) == runs_before

    @pytest.mark.asyncio
    async def test_a_refused_undo_never_probes_the_server(self, harness, workspace, capsys):
        planned = await plan_from(
            env_for(harness, workspace), workspace, seed_corpus(harness, workspace, 1), capsys,
        )
        calls: list[tuple[str, dict[str, Any]]] = []
        env = env_for(harness, workspace, tool_caller=harness.recording_tool_caller(calls))

        code = await run_cli(env, workspace, 'undo', '--run', planned['run_id'][:8])

        assert code == 2
        assert calls == []


class TestTheHomeProject:
    @pytest.mark.asyncio
    async def test_a_run_with_no_project_of_its_own_probes_the_home_project(
        self, harness, workspace, capsys,
    ):
        calls: list[tuple[str, dict[str, Any]]] = []
        env = env_for(harness, workspace, tool_caller=harness.recording_tool_caller(calls))

        code, _applied = await apply(env, workspace, capsys)

        assert code == 0
        read_projects = [arguments['project_id'] for tool, arguments in calls if tool == READ_TOOL]
        assert read_projects == [resolve_project_id(str(workspace.home_root))]


class TestApplyAndUndo:
    @pytest.mark.asyncio
    async def test_an_approved_plan_applies_every_pending_heal(self, harness, workspace, capsys):
        env = env_for(harness, workspace)
        planned = await plan_from(env, workspace, seed_corpus(harness, workspace, 3), capsys)

        code, applied = await apply(
            env, workspace, capsys, '--approved-plan-sha', planned['plan_sha256'],
        )

        assert code == 0
        assert applied['counts']['applied'] == 3
        assert applied['counts']['approved_plan_sha256'] == planned['plan_sha256']
        assert applied['outcome'] == 'complete'

    @pytest.mark.asyncio
    async def test_undo_resolves_a_run8_prefix(self, harness, workspace, capsys):
        env = env_for(harness, workspace)
        await plan_from(env, workspace, seed_corpus(harness, workspace, 2), capsys)
        _code, applied = await apply(env, workspace, capsys)

        code = await run_cli(env, workspace, 'undo', '--run', applied['run_id'][:8])

        undone = printed_report(capsys)
        assert code == 0
        assert undone['counts']['applied'] == 2
        assert undone['outcome'] == 'complete'

    @pytest.mark.asyncio
    async def test_config_is_read_afresh_at_every_run(self, harness, workspace, capsys):
        env = env_for(harness, workspace)
        await plan_from(env, workspace, seed_corpus(harness, workspace, 4), capsys)

        workspace.write_config(max_actions_per_run=1)
        _code, first = await apply(env, workspace, capsys)
        workspace.write_config(max_actions_per_run=2)
        _code, second = await apply(env, workspace, capsys)

        assert first['counts']['applied'] == 1
        assert second['counts']['applied'] == 2
        assert harness.mem0.write_count == 3


class TestTheUndoCommandOperatorsAreShown:
    RUN_ID = '0123456789abcdef0123456789abcdef'

    def test_it_names_this_script_from_the_repo_root(self):
        script, *_arguments = shlex.split(undo_command(self.RUN_ID))

        assert (REPO_ROOT / script).resolve() == SCRIPT_PATH.resolve()

    def test_this_script_parses_it_as_an_undo_of_that_run(self):
        _script, *arguments = shlex.split(undo_command(self.RUN_ID))

        args = cli.build_parser().parse_args(arguments)

        assert (args.command, args.run) == ('undo', self.RUN_ID)


class TestExitCodes:
    @pytest.mark.asyncio
    async def test_a_streak_stopped_apply_exits_1(self, unadmitted, workspace, capsys):
        env = env_for(unadmitted, workspace)
        await plan_from(env, workspace, seed_corpus(unadmitted, workspace, 5), capsys)

        code, applied = await apply(env, workspace, capsys)

        assert code == 1
        assert applied['counts']['stopped_by'] == 'write_failure_streak'
        assert applied['outcome'] == 'partial'

    @pytest.mark.asyncio
    async def test_a_capped_only_apply_exits_0_and_says_partial(self, harness, workspace, capsys):
        env = env_for(harness, workspace)
        await plan_from(env, workspace, seed_corpus(harness, workspace, 2), capsys)
        workspace.write_config(max_actions_per_run=1)

        code, applied = await apply(env, workspace, capsys)

        assert code == 0
        assert (applied['counts']['applied'], applied['counts']['skipped_cap']) == (1, 1)
        assert applied['outcome'] == 'partial'

    @pytest.mark.parametrize(('counts', 'expected'), [
        (RunCounts(), 0),
        (RunCounts(skipped_cap=3, caps_bit=('max_actions_per_run',)), 0),
        (RunCounts(skipped_stale=2), 0),
        (RunCounts(failed=1), 1),
        (RunCounts(read_failed=1), 1),
        (RunCounts(failed=3, not_attempted=4, stopped_by='write_failure_streak'), 1),
        (RunCounts(adjudication_failed=1), 1),
    ])
    def test_exit_code_maps_the_report(self, counts, expected):
        assert cli.exit_code(RunReport(run_id='0' * 32, counts=counts)) == expected


def seed_unrated_sightings(
    harness: LinkHealHarness, count: int, *, project: str = DF, first: int = 100,
) -> None:
    """*count* sightings no corpus rates, child texts ``cli sighting <n>`` from *first*."""
    for index in range(first, first + count):
        child, parent = _sighting_ids(index)
        harness.seed_link(
            kind=SIGHTING_KIND, child=child, parent=parent, project=project,
            child_text=f'cli sighting {index}', parent_text=f'cli parent {index}',
        )


def always(answer: Answer) -> FakeAdjudicator:
    return FakeAdjudicator(lambda _text: answer)


async def plan_adjudicated(env: Any, workspace: Workspace, capsys, *argv: str) -> tuple[int, dict[str, Any]]:
    code = await run_cli(env, workspace, 'plan', '--from-adjudicator', *argv)
    return code, printed_report(capsys)


class TestPlanFromAdjudicator:
    @pytest.mark.asyncio
    async def test_plan_ledgers_an_adjudicator_run_its_verdicts_and_its_document(
        self, harness, workspace, capsys,
    ):
        seed_unrated_sightings(harness, 2)
        fake = always(Verdict.EXTENDS)

        code, report = await plan_adjudicated(
            env_for(harness, workspace, adjudicator=fake), workspace, capsys, '--project', DF,
        )

        assert code == 0
        assert report['counts']['planned_by_action'] == {'relabel': 2}
        assert report['counts']['adjudication_failed'] == 0
        assert report['outcome'] == 'complete'
        (run,) = workspace.runs()
        assert (run.run_id, run.source, run.writes) == (report['run_id'], RunSource.ADJUDICATOR, False)
        assert workspace.adjudication_count(run.run_id) == 2
        document = json.loads(Path(report['plan_path']).read_text())
        assert {action['basis_source'] for action in document['actions']} == {'adjudicator'}
        assert hashlib.sha256(Path(report['plan_path']).read_bytes()).hexdigest() == report['plan_sha256']
        assert harness.mem0.write_count == 0

    @pytest.mark.parametrize(
        'source_args',
        [
            pytest.param(['--from-corpus', 'corpus.jsonl', '--from-adjudicator'], id='both'),
            pytest.param(
                ['--deterministic', '--from-corpus', 'corpus.jsonl'], id='deterministic_and_corpus',
            ),
            pytest.param(['--deterministic', '--from-adjudicator'], id='deterministic_and_adjudicator'),
            pytest.param([], id='neither'),
        ],
    )
    @pytest.mark.asyncio
    async def test_exactly_one_plan_source_is_required(self, harness, workspace, source_args):
        with pytest.raises(SystemExit) as exited:
            await run_cli(env_for(harness, workspace), workspace, 'plan', *source_args)

        assert exited.value.code == 2
        assert workspace.runs() == []

    @pytest.mark.asyncio
    async def test_project_with_a_corpus_is_refused(self, harness, workspace, capsys):
        corpus = seed_corpus(harness, workspace, 1)

        code = await run_cli(
            env_for(harness, workspace), workspace,
            'plan', '--from-corpus', str(corpus), '--project', DF,
        )

        err = capsys.readouterr().err
        assert code == 2
        assert '--project' in err
        assert '--from-adjudicator' in err
        assert '--deterministic' in err
        assert workspace.runs() == []

    @pytest.mark.asyncio
    async def test_project_is_repeatable(self, harness, workspace, capsys):
        seed_unrated_sightings(harness, 1, project=DF, first=100)
        seed_unrated_sightings(harness, 1, project=REIFY, first=200)
        fake = always(Verdict.EXTENDS)

        code, _report = await plan_adjudicated(
            env_for(harness, workspace, adjudicator=fake), workspace, capsys,
            '--project', DF, '--project', REIFY,
        )

        assert code == 0
        assert sorted(pair.child_text for pair in fake.pairs) == ['cli sighting 100', 'cli sighting 200']

    @pytest.mark.asyncio
    async def test_project_defaults_to_the_home_project(self, harness, workspace, capsys):
        home = resolve_project_id(str(workspace.home_root))
        seed_unrated_sightings(harness, 1, project=home, first=300)
        seed_unrated_sightings(harness, 1, project=DF, first=100)
        fake = always(Verdict.EXTENDS)

        code, _report = await plan_adjudicated(
            env_for(harness, workspace, adjudicator=fake), workspace, capsys,
        )

        assert code == 0
        assert [pair.key.split(':')[0] for pair in fake.pairs] == [home]

    @pytest.mark.asyncio
    async def test_status_lists_the_adjudicator_run_as_non_writing(self, harness, workspace, capsys):
        seed_unrated_sightings(harness, 1)
        env = env_for(harness, workspace, adjudicator=always(Verdict.EXTENDS))
        _code, report = await plan_adjudicated(env, workspace, capsys, '--project', DF)

        assert await run_cli(env, workspace, 'status') == 0

        assert 'adjudicator  writes no' in run_line(status_lines(capsys), report['run_id'])

    @pytest.mark.asyncio
    async def test_a_partly_failed_adjudication_exits_1(self, harness, workspace, capsys):
        seed_unrated_sightings(harness, 2)
        fake = FakeAdjudicator(
            lambda text: AdjudicationFailure.PARSE_FAILURE if text.endswith('100') else Verdict.EXTENDS,
        )

        code, report = await plan_adjudicated(
            env_for(harness, workspace, adjudicator=fake), workspace, capsys, '--project', DF,
        )

        assert code == 1
        assert report['counts']['adjudication_failed'] == 1
        assert report['outcome'] == 'partial'

    @pytest.mark.asyncio
    async def test_an_unreachable_server_is_refused_before_any_adjudication(
        self, harness, workspace, capsys,
    ):
        seed_unrated_sightings(harness, 1)
        fake = always(Verdict.EXTENDS)
        env = env_for(harness, workspace, tool_caller=refusing_tool_caller(), adjudicator=fake)

        code = await run_cli(env, workspace, 'plan', '--from-adjudicator', '--project', DF)

        assert code == 2
        assert DEFAULT_SERVER_URL in capsys.readouterr().err
        assert fake.calls == []
        assert workspace.runs() == []


class TestApplyFromAdjudicator:
    @pytest.mark.asyncio
    async def test_each_apply_drains_only_its_own_sources_heals(self, harness, workspace, capsys):
        env = env_for(harness, workspace, adjudicator=always(Verdict.EXTENDS))
        await plan_from(env, workspace, seed_corpus(harness, workspace, 1), capsys)
        seed_unrated_sightings(harness, 1)
        await plan_adjudicated(env, workspace, capsys, '--project', DF)

        code, adjudicated = await apply(env, workspace, capsys, '--from-adjudicator')

        assert code == 0
        assert adjudicated['counts']['applied'] == 1
        assert workspace.pending(RunSource.CORPUS) == 1
        assert workspace.pending(RunSource.ADJUDICATOR) == 0

        code, corpus = await apply(env, workspace, capsys)

        assert code == 0
        assert corpus['counts']['applied'] == 1
        assert workspace.pending(RunSource.CORPUS) == 0

    @pytest.mark.asyncio
    async def test_a_share_refused_apply_exits_1(self, harness, workspace, capsys):
        seed_unrated_sightings(harness, 20)
        misfiled = {f'cli sighting {index}' for index in range(100, 106)}
        fake = FakeAdjudicator(
            lambda text: Verdict.RELATED if text in misfiled else Verdict.EXTENDS,
        )
        env = env_for(harness, workspace, adjudicator=fake)
        await plan_adjudicated(env, workspace, capsys, '--project', DF)

        code, applied = await apply(env, workspace, capsys, '--from-adjudicator')

        assert code == 1
        assert applied['counts']['stopped_by'] == SHARE_STOP
        assert applied['counts']['applied'] == 0
        assert harness.mem0.write_count == 0


def seed_completion_children(
    harness: LinkHealHarness, *projects: str, first: int = 400,
) -> None:
    """One orchestrator completion record per project, each filed as an amendment of its own parent."""
    for index, project in enumerate(projects, start=first):
        child, parent = _sighting_ids(index)
        harness.seed_link(
            child=child, parent=parent, project=project,
            child_text=f'cli completion {index}', parent_text=f'cli parent {index}',
            meta={**COMPLETION_META, 'task_id': str(index)},
        )


async def plan_deterministic(
    env: Any, workspace: Workspace, capsys, *argv: str,
) -> tuple[int, dict[str, Any]]:
    code = await run_cli(env, workspace, 'plan', '--deterministic', *argv)
    return code, printed_report(capsys)


def plan_document_actions(report: dict[str, Any]) -> list[dict[str, Any]]:
    return json.loads(Path(report['plan_path']).read_text())['actions']


class TestPlanDeterministic:
    @pytest.mark.asyncio
    async def test_plans_the_completion_detaches_of_every_project_without_a_verdict(
        self, harness, workspace, capsys,
    ):
        seed_completion_children(harness, DF, REIFY)
        seed_unrated_sightings(harness, 1)

        code, report = await plan_deterministic(
            env_for(harness, workspace), workspace, capsys, '--project', DF, '--project', REIFY,
        )

        assert code == 0
        assert report['counts']['planned_by_action'] == {'detach': 2}
        assert report['counts']['unexamined'] == 1
        (run,) = workspace.runs()
        assert (run.run_id, run.source, run.writes) == (
            report['run_id'], RunSource.DETERMINISTIC, False,
        )
        actions = plan_document_actions(report)
        assert {action['basis_source'] for action in actions} == {'completion_record'}
        assert sorted(action['project_id'] for action in actions) == sorted([DF, REIFY])
        assert harness.mem0.write_count == 0

    @pytest.mark.asyncio
    async def test_project_defaults_to_the_home_project(self, harness, workspace, capsys):
        home = resolve_project_id(str(workspace.home_root))
        seed_completion_children(harness, home, DF)

        code, report = await plan_deterministic(env_for(harness, workspace), workspace, capsys)

        assert code == 0
        assert [action['project_id'] for action in plan_document_actions(report)] == [home]

    @pytest.mark.asyncio
    async def test_status_names_the_latest_deterministic_plan_as_pending_now(
        self, harness, workspace, capsys,
    ):
        env = env_for(harness, workspace)
        seed_completion_children(harness, DF)
        _code, planned = await plan_deterministic(env, workspace, capsys, '--project', DF)

        assert await run_cli(env, workspace, 'status') == 0

        line = latest_plan_line(status_lines(capsys), RunSource.DETERMINISTIC)
        assert planned['run_id'][:8] in line
        assert planned['plan_path'] in line
        assert planned['plan_sha256'] in line
        assert PENDING_NOW in line


class TestApplyDeterministic:
    @pytest.mark.asyncio
    async def test_an_approved_deterministic_apply_drains_only_its_own_heals(
        self, harness, workspace, capsys,
    ):
        env = env_for(harness, workspace)
        await plan_from(env, workspace, seed_corpus(harness, workspace, 1), capsys)
        seed_completion_children(harness, DF)
        _code, planned = await plan_deterministic(env, workspace, capsys, '--project', DF)

        code, applied = await apply(
            env, workspace, capsys, '--deterministic', '--approved-plan-sha', planned['plan_sha256'],
        )

        assert code == 0
        assert applied['counts']['applied'] == 1
        assert applied['counts']['approved_plan_sha256'] == planned['plan_sha256']
        assert workspace.pending(RunSource.DETERMINISTIC) == 0
        assert workspace.pending(RunSource.CORPUS) == 1

    @pytest.mark.asyncio
    async def test_a_completion_detach_a_corpus_plan_ledgered_first_drains_only_through_its_apply(
        self, harness, workspace, capsys,
    ):
        env = env_for(harness, workspace)
        seed_completion_children(harness, DF)
        corpus = await plan_from(env, workspace, seed_corpus(harness, workspace, 1), capsys)
        _code, deterministic = await plan_deterministic(env, workspace, capsys, '--project', DF)

        _code, drained = await apply(env, workspace, capsys, '--deterministic')
        _code, applied = await apply(
            env, workspace, capsys, '--approved-plan-sha', corpus['plan_sha256'],
        )

        assert corpus['counts']['planned_by_action'] == {'detach': 1, 'relabel': 1}
        assert deterministic['counts']['already_pending'] == 1
        assert plan_document_actions(deterministic) == []
        assert drained['counts']['applied'] == 0
        assert applied['counts']['applied'] == 2
        assert workspace.pending(RunSource.CORPUS) == 0

    @pytest.mark.asyncio
    async def test_deterministic_and_adjudicator_together_is_a_usage_error(self, harness, workspace):
        with pytest.raises(SystemExit) as exited:
            await run_cli(
                env_for(harness, workspace), workspace,
                'apply', '--deterministic', '--from-adjudicator',
            )

        assert exited.value.code == 2
        assert workspace.runs() == []


class TestStatusShowsWhatAPlanCameTo:
    @pytest.mark.asyncio
    async def test_a_plan_runs_line_shows_adjudication_failed_and_would_escape(
        self, harness, workspace, capsys,
    ):
        seed_unrated_sightings(harness, 2)
        fake = FakeAdjudicator(
            lambda text: AdjudicationFailure.PARSE_FAILURE if text.endswith('100') else Verdict.EXTENDS,
            storm={'count': 3, 'threshold': 3, 'labels': ['parse_failure'], 'model': 'fake-model'},
        )
        env = env_for(harness, workspace, adjudicator=fake)
        code, report = await plan_adjudicated(env, workspace, capsys, '--project', DF)
        assert code == 1

        assert await run_cli(env, workspace, 'status') == 0

        plan_run_line = run_line(status_lines(capsys), report['run_id'])
        assert 'adjudication_failed 1' in plan_run_line
        assert f'would_escape {ADJUDICATOR_ANCHOR}' in plan_run_line

    @pytest.mark.asyncio
    async def test_a_run_with_nothing_to_escape_says_none(self, harness, workspace, capsys):
        env = env_for(harness, workspace)
        planned = await plan_from(env, workspace, seed_corpus(harness, workspace, 1), capsys)

        assert await run_cli(env, workspace, 'status') == 0

        plan_run_line = run_line(status_lines(capsys), planned['run_id'])
        assert 'would_escape none' in plan_run_line
        assert 'adjudication_failed 0' in plan_run_line

    @pytest.mark.asyncio
    async def test_status_names_the_latest_plan_of_each_source_with_its_document_and_sha(
        self, harness, workspace, capsys,
    ):
        env = env_for(harness, workspace, adjudicator=always(Verdict.EXTENDS))
        corpus = seed_corpus(harness, workspace, 2)
        first = await plan_from(env, workspace, corpus, capsys)
        second = await plan_from(env, workspace, corpus, capsys)
        seed_unrated_sightings(harness, 1)
        _code, adjudicated = await plan_adjudicated(env, workspace, capsys, '--project', DF)

        assert await run_cli(env, workspace, 'status', '--limit', '1') == 0

        lines = status_lines(capsys)
        corpus_line = latest_plan_line(lines, RunSource.CORPUS)
        assert second['run_id'][:8] in corpus_line
        assert second['plan_path'] in corpus_line
        assert second['plan_sha256'] in corpus_line
        assert first['run_id'][:8] not in corpus_line
        adjudicator_line = latest_plan_line(lines, RunSource.ADJUDICATOR)
        assert adjudicated['run_id'][:8] in adjudicator_line
        assert adjudicated['plan_path'] in adjudicator_line
        assert adjudicated['plan_sha256'] in adjudicator_line

    @pytest.mark.asyncio
    async def test_a_latest_plan_whose_pending_heals_changed_is_marked_stale(
        self, harness, workspace, capsys,
    ):
        env = env_for(harness, workspace)
        corpus = seed_corpus(harness, workspace, 3)
        await plan_from(env, workspace, corpus, capsys)
        workspace.write_config(max_actions_per_run=1)
        await apply(env, workspace, capsys)

        assert await run_cli(env, workspace, 'status') == 0

        stale_line = latest_plan_line(status_lines(capsys), RunSource.CORPUS)
        assert STALE in stale_line
        assert PENDING_NOW not in stale_line

        replanned = await plan_from(env, workspace, corpus, capsys)
        assert await run_cli(env, workspace, 'status') == 0

        current_line = latest_plan_line(status_lines(capsys), RunSource.CORPUS)
        assert replanned['run_id'][:8] in current_line
        assert PENDING_NOW in current_line

    @pytest.mark.asyncio
    async def test_a_fully_applied_plan_says_nothing_is_pending_rather_than_stale(
        self, harness, workspace, capsys,
    ):
        env = env_for(harness, workspace)
        planned = await plan_from(env, workspace, seed_corpus(harness, workspace, 2), capsys)
        code, applied = await apply(env, workspace, capsys)
        assert (code, applied['counts']['applied']) == (0, 2)

        assert await run_cli(env, workspace, 'status') == 0

        applied_line = latest_plan_line(status_lines(capsys), RunSource.CORPUS)
        assert planned['plan_sha256'] in applied_line
        assert NOTHING_PENDING in applied_line
        assert STALE not in applied_line
        assert PENDING_NOW not in applied_line

    @pytest.mark.asyncio
    async def test_a_relative_out_path_is_recorded_absolute(
        self, harness, workspace, capsys, monkeypatch,
    ):
        env = env_for(harness, workspace)
        corpus = seed_corpus(harness, workspace, 1)
        monkeypatch.chdir(workspace.root)

        code = await run_cli(
            env, workspace, 'plan', '--from-corpus', str(corpus), '--out', 'reviewed.json',
        )

        report = printed_report(capsys)
        assert code == 0
        assert report['plan_path'] == str(workspace.root / 'reviewed.json')
        assert await run_cli(env, workspace, 'status') == 0
        assert report['plan_path'] in latest_plan_line(status_lines(capsys), RunSource.CORPUS)


SWEEP_NOW = datetime(2026, 11, 12, 2, 15, tzinfo=UTC)
NEW = '2026-11-11T12:00:00+00:00'
FIXTURES = Path(__file__).resolve().parent / 'fixtures'
RATINGS = FIXTURES / 'link_heal_calibration_ratings.jsonl'
ITEMS = FIXTURES / 'link_heal_calibration_items.json'


def sweep_env_for(harness: LinkHealHarness, workspace: Workspace, **kwargs: Any) -> Any:
    """:func:`env_for`, with the reports under *workspace* and the clock at :data:`SWEEP_NOW`."""
    return dataclasses.replace(
        env_for(harness, workspace, **kwargs),
        reports_dir_for=lambda _config: workspace.reports_dir,
        now=lambda: SWEEP_NOW,
    )


def seed_sightings_created(
    harness: LinkHealHarness, indices: range, *, created_at: str = NEW, project: str = DF,
) -> None:
    """Sightings ``cli sighting <n>`` for each n in *indices*, created at *created_at*."""
    for index in indices:
        child, parent = _sighting_ids(index)
        harness.seed_link(
            kind=SIGHTING_KIND, child=child, parent=parent, project=project,
            child_text=f'cli sighting {index}', parent_text=f'cli parent {index}',
            meta={'created_at': created_at},
        )


async def sweep(env: Any, workspace: Workspace, capsys, *argv: str) -> tuple[int, dict[str, Any]]:
    code = await run_cli(env, workspace, 'sweep', *argv)
    return code, printed_report(capsys)


def asked_texts(fake: FakeAdjudicator) -> list[str]:
    return sorted(pair.child_text for pair in fake.pairs)


class TestSweep:
    @pytest.mark.asyncio
    async def test_a_dry_run_prints_its_runs_report_and_counts(self, harness, workspace, capsys):
        env = sweep_env_for(harness, workspace)

        code, report = await sweep(env, workspace, capsys, '--dry-run')

        assert code == 0
        assert report['apply_run_id'] is None
        (run,) = workspace.runs()
        assert (run.run_id, run.source, run.writes) == (report['plan_run_id'], RunSource.SWEEP, False)
        assert Path(report['report_path']).is_file()
        assert Path(report['report_path']).parent == workspace.reports_dir
        assert report['counts']['plan']['examined'] == 0
        assert report['counts']['apply'] is None

    @pytest.mark.asyncio
    async def test_since_and_project_bound_what_is_examined(self, harness, workspace, capsys):
        seed_sightings_created(harness, range(100, 101), created_at='2026-10-20T00:00:00+00:00')
        seed_sightings_created(harness, range(101, 102), created_at='2026-10-01T00:00:00+00:00')
        seed_sightings_created(
            harness, range(200, 201), created_at='2026-10-20T00:00:00+00:00', project=REIFY,
        )
        fake = always(Verdict.EXTENDS)

        code, report = await sweep(
            sweep_env_for(harness, workspace, adjudicator=fake), workspace, capsys,
            '--since', '2026-10-11T00:00:00+00:00', '--project', DF,
        )

        assert code == 0
        assert asked_texts(fake) == ['cli sighting 100']
        assert report['counts']['plan']['examined'] == 1

    @pytest.mark.asyncio
    async def test_a_since_without_an_offset_is_refused_before_any_run(
        self, harness, workspace, capsys,
    ):
        code = await run_cli(
            sweep_env_for(harness, workspace), workspace, 'sweep', '--since', '2026-10-11',
        )

        assert code == 2
        assert '--since' in capsys.readouterr().err
        assert workspace.runs() == []

    @pytest.mark.asyncio
    async def test_without_a_project_it_sweeps_the_configured_projects(
        self, harness, workspace, capsys,
    ):
        workspace.write_config(link_heal={'projects': [REIFY]})
        seed_sightings_created(harness, range(100, 101))
        seed_sightings_created(harness, range(200, 201), project=REIFY)
        fake = always(Verdict.EXTENDS)

        code, _report = await sweep(
            sweep_env_for(harness, workspace, adjudicator=fake), workspace, capsys,
        )

        assert code == 0
        assert asked_texts(fake) == ['cli sighting 200']

    @pytest.mark.asyncio
    async def test_with_no_project_configured_it_sweeps_the_home_project(
        self, harness, workspace, capsys,
    ):
        home = resolve_project_id(str(workspace.home_root))
        seed_sightings_created(harness, range(300, 301), project=home)
        seed_sightings_created(harness, range(100, 101))
        fake = always(Verdict.EXTENDS)

        code, _report = await sweep(
            sweep_env_for(harness, workspace, adjudicator=fake), workspace, capsys,
        )

        assert code == 0
        assert [pair.key.split(':')[0] for pair in fake.pairs] == [home]

    @pytest.mark.asyncio
    async def test_a_sweep_with_adjudication_failures_exits_1(self, harness, workspace, capsys):
        seed_sightings_created(harness, range(100, 102))
        fake = FakeAdjudicator(
            lambda text: AdjudicationFailure.PARSE_FAILURE if text.endswith('100') else Verdict.EXTENDS,
        )

        code, report = await sweep(
            sweep_env_for(harness, workspace, adjudicator=fake), workspace, capsys, '--project', DF,
        )

        assert code == 1
        assert report['counts']['plan']['adjudication_failed'] == 1
        assert report['outcome'] == 'partial'

    @pytest.mark.asyncio
    async def test_its_report_is_filed_into_the_home_queue(self, harness, workspace, capsys):
        code, report = await sweep(sweep_env_for(harness, workspace), workspace, capsys)

        assert code == 0
        (filed,) = workspace.pending_escalations(RUN_REPORT_ANCHOR)
        assert report['report_path'] in filed.summary
        assert filed.id == report['escalation_id']

    @pytest.mark.asyncio
    async def test_an_unknown_sweep_action_is_refused_before_any_run(
        self, harness, workspace, capsys,
    ):
        workspace.write_config(link_heal={'sweep_actions': ['detach', 'explode']})

        code = await run_cli(sweep_env_for(harness, workspace), workspace, 'sweep')

        assert code == 2
        assert 'explode' in capsys.readouterr().err
        assert workspace.runs() == []

    @pytest.mark.asyncio
    async def test_an_unreachable_server_is_refused_before_any_adjudication(
        self, harness, workspace, capsys,
    ):
        seed_sightings_created(harness, range(100, 101))
        fake = always(Verdict.EXTENDS)
        env = sweep_env_for(harness, workspace, tool_caller=refusing_tool_caller(), adjudicator=fake)

        code = await run_cli(env, workspace, 'sweep', '--project', DF)

        assert code == 2
        assert DEFAULT_SERVER_URL in capsys.readouterr().err
        assert fake.calls == []
        assert workspace.runs() == []


def calibrated_answers(overrides: Mapping[str, Answer] | None = None) -> dict[str, Answer]:
    """Each rated pair's final verdict, keyed by its child text."""
    answers: dict[str, Answer] = {
        item.child_text: item.final for item in load_soak_population(RATINGS, ITEMS)
    }
    answers.update(overrides or {})
    return answers


async def calibrate(env: Any, workspace: Workspace, capsys, *paths: Path) -> tuple[int, dict[str, Any]]:
    ratings, items = paths or (RATINGS, ITEMS)
    code = await run_cli(env, workspace, 'calibrate', '--ratings', str(ratings), '--items', str(items))
    return code, printed_report(capsys)


class TestCalibrate:
    @pytest.mark.asyncio
    async def test_a_passing_adjudicator_exits_0_naming_its_report(
        self, harness, workspace, capsys,
    ):
        fake = FakeAdjudicator(calibrated_answers().__getitem__)

        code, report = await calibrate(sweep_env_for(harness, workspace, adjudicator=fake), workspace, capsys)

        assert code == 0
        assert report['passed'] is True
        assert Path(report['report_path']).is_file()
        assert Path(report['report_path']).parent == workspace.reports_dir
        (run,) = workspace.runs()
        assert (run.run_id, run.source) == (report['run_id'], RunSource.CALIBRATION)
        assert workspace.pending_escalations(CALIBRATION_ANCHOR) == []

    @pytest.mark.asyncio
    async def test_a_failing_adjudicator_exits_1_and_files_the_escape(
        self, harness, workspace, capsys,
    ):
        fake = always(Verdict.UNCLEAR)

        code, report = await calibrate(sweep_env_for(harness, workspace, adjudicator=fake), workspace, capsys)

        assert code == 1
        assert report['passed'] is False
        (filed,) = workspace.pending_escalations(CALIBRATION_ANCHOR)
        assert filed.id == report['escalation_id']

    @pytest.mark.asyncio
    async def test_a_missing_input_file_is_refused_before_any_run(self, harness, workspace, capsys):
        missing = workspace.root / 'no-such-ratings.jsonl'

        code = await run_cli(
            sweep_env_for(harness, workspace), workspace,
            'calibrate', '--ratings', str(missing), '--items', str(ITEMS),
        )

        assert code == 2
        assert str(missing) in capsys.readouterr().err
        assert workspace.runs() == []

    @pytest.mark.asyncio
    async def test_a_malformed_input_file_is_refused_before_any_run(
        self, harness, workspace, capsys,
    ):
        malformed = workspace.root / 'items.json'
        malformed.write_text('not json')

        code = await run_cli(
            sweep_env_for(harness, workspace), workspace,
            'calibrate', '--ratings', str(RATINGS), '--items', str(malformed),
        )

        assert code == 2
        assert 'items.json' in capsys.readouterr().err
        assert workspace.runs() == []

    @pytest.mark.asyncio
    async def test_a_passing_calibration_lets_a_writing_sweep_apply(
        self, harness, workspace, capsys,
    ):
        workspace.write_config(link_heal={'writes': True})
        seed_sightings_created(harness, range(100, 101))
        fake = FakeAdjudicator(
            calibrated_answers({'cli sighting 100': Verdict.EXTENDS}).__getitem__,
        )
        env = sweep_env_for(harness, workspace, adjudicator=fake)
        withheld_code, withheld = await sweep(env, workspace, capsys, '--project', DF)
        assert (withheld_code, withheld['apply_run_id']) == (0, None)
        assert (await calibrate(env, workspace, capsys))[0] == 0

        code, report = await sweep(
            env, workspace, capsys, '--project', DF, '--since', '2026-11-01T00:00:00+00:00',
        )

        assert code == 0
        assert report['apply_run_id'] is not None
        assert report['undo_run_id'] == report['apply_run_id']
        assert report['counts']['apply']['applied'] == 1


class TestTheSweepThroughApplyAndStatus:
    @pytest.mark.asyncio
    async def test_apply_sweep_drains_the_sweeps_pending_heals(self, harness, workspace, capsys):
        workspace.write_config(link_heal={'writes': True})
        seed_sightings_created(harness, range(100, 102))
        env = sweep_env_for(harness, workspace, adjudicator=always(Verdict.EXTENDS))
        await sweep(env, workspace, capsys, '--dry-run', '--project', DF)
        assert workspace.pending(RunSource.SWEEP) == 2
        workspace.calibrate_fake_adjudicator(passed=True)

        code, applied = await apply(env, workspace, capsys, '--sweep')

        assert code == 0
        assert applied['counts']['applied'] == 2
        assert workspace.pending(RunSource.SWEEP) == 0

    @pytest.mark.asyncio
    async def test_status_lists_the_latest_sweep_plan(self, harness, workspace, capsys):
        seed_sightings_created(harness, range(100, 101))
        env = sweep_env_for(harness, workspace, adjudicator=always(Verdict.EXTENDS))
        _code, report = await sweep(env, workspace, capsys, '--project', DF)

        assert await run_cli(env, workspace, 'status') == 0

        lines = status_lines(capsys)
        assert report['plan_run_id'][:8] in latest_plan_line(lines, RunSource.SWEEP)
        assert 'sweep  writes no' in run_line(lines, report['plan_run_id'])


class TestTheNewCommandsHoldTheLock:
    @pytest.mark.asyncio
    @pytest.mark.parametrize('argv', [
        ('sweep', '--dry-run'),
        ('calibrate', '--ratings', str(RATINGS), '--items', str(ITEMS)),
    ])
    async def test_a_held_lock_is_refused(self, harness, workspace, capsys, argv):
        workspace.ledger_dir.mkdir(parents=True)

        with RunLock(workspace.ledger_dir / LOCK_FILENAME):
            code = await run_cli(sweep_env_for(harness, workspace), workspace, *argv)

        assert code == 2
        assert str(os.getpid()) in capsys.readouterr().err
        assert workspace.runs() == []
