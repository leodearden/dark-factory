"""The link-heal CLI (task 6181, plans/write-triage-link-healing-prd.md H1).

``scripts/link_heal.py`` is driven through ``await run(argv, env)``. The env
swaps in the in-process server from ``_link_heal_harness`` as the transport,
its ``FakeMem0`` as the census, and tmp directories for the ledger and the
escalation queue. Config is read from a tmp YAML by the script's own loader,
at every run.
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
    WITHOUT_LINK_HEAL_PREFIX,
    LinkHealHarness,
    assert_store_invariants,
    build_harness,
)

from fused_memory.maintenance.link_heal import RunCounts
from fused_memory.maintenance.link_heal_executor import RunReport
from fused_memory.maintenance.link_heal_ledger import (
    LEDGER_FILENAME,
    LOCK_FILENAME,
    LinkHealLedger,
    RunLock,
    RunRow,
)
from fused_memory.maintenance.link_heal_store import ToolCaller, text_sha256
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
    def escalation_root(self) -> Path:
        return self.root / 'escalations'

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


def env_for(
    harness: LinkHealHarness,
    workspace: Workspace,
    *,
    tool_caller: ToolCaller | None = None,
    server_urls: list[str] | None = None,
) -> Any:
    """The CLI env over *harness*, with the ledger and escalations under *workspace*."""
    caller = tool_caller or harness.tool_caller()

    def tool_caller_for(server_url: str):
        if server_urls is not None:
            server_urls.append(server_url)
        return contextlib.nullcontext(caller)

    return cli.LinkHealEnv(
        tool_caller_for=tool_caller_for,
        census_for=lambda _config: contextlib.nullcontext(harness.mem0),
        ledger_dir_for=lambda _config: workspace.ledger_dir,
        escalation_root_for=lambda _config: str(workspace.escalation_root),
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
    ])
    def test_exit_code_maps_the_report(self, counts, expected):
        assert cli.exit_code(RunReport(run_id='0' * 32, counts=counts)) == expected
