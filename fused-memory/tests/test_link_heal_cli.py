"""The link-heal CLI (tasks 6181, 6184; plans/write-triage-link-healing-prd.md H1, H2).

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
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio
import yaml
from _fm_helpers import load_script_module
from _link_heal_harness import (
    ALL_LINK_HEAL_PREFIXES,
    DF,
    REIFY,
    WITHOUT_LINK_HEAL_PREFIX,
    Answer,
    FailingCensus,
    FakeAdjudicator,
    LinkHealHarness,
    assert_store_invariants,
    build_harness,
)

from fused_memory.maintenance.link_adjudicator import AdjudicationFailure
from fused_memory.maintenance.link_heal import RunCounts, Verdict
from fused_memory.maintenance.link_heal_executor import SHARE_STOP, RunReport
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

cli = load_script_module(SCRIPT_PATH, mod_name='link_heal_cli')

NO_WRITING_RUN = 'no link-heal run has written here'
SERVER_PORT = 8765
DEFAULT_SERVER_URL = f'http://127.0.0.1:{SERVER_PORT}'
WRONG_SHA = 'deadbeef' * 8


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

    def write_config(self, *, max_actions_per_run: int = 25) -> None:
        self.config_path.write_text(yaml.safe_dump({
            'server': {'port': SERVER_PORT},
            'taskmaster': {'project_root': str(self.root / 'dark-factory')},
            'reconciliation': {'data_dir': str(self.data_dir)},
            'link_heal': {
                'max_actions_per_run': max_actions_per_run,
                'backlog_multiplier': 5,
                'write_failure_streak': 3,
            },
        }))

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

        lines = capsys.readouterr().out.splitlines()
        (run_line,) = [line for line in lines if planned['run_id'][:8] in line]
        assert 'writes no' in run_line
        assert NO_WRITING_RUN in lines

    @pytest.mark.asyncio
    async def test_a_zero_action_apply_is_a_writing_run(self, harness, workspace, capsys):
        env = env_for(harness, workspace)
        code, applied = await apply(env, workspace, capsys)
        assert code == 0

        assert await run_cli(env, workspace, 'status') == 0

        lines = capsys.readouterr().out.splitlines()
        (run_line,) = [line for line in lines if applied['run_id'][:8] in line]
        assert 'writes yes' in run_line
        assert 'applied 0' in run_line
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

        lines = capsys.readouterr().out.splitlines()
        (run_line,) = [line for line in lines if report['run_id'][:8] in line]
        assert 'adjudicator  writes no' in run_line

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
