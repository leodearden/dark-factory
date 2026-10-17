"""How the offline lane runs a sub-run: one streaming launcher and the three detection seams.

Every sub-run, detection or confirm, streams its child's combined output
straight to a file in its pass directory and leaves a run record beside it
(``orchestrator/src/orchestrator/offline_lane_records.py::RunRecordStore``).
"""

from __future__ import annotations

import asyncio
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, Protocol

from shared.verify_admission import nice_prefix

from orchestrator.offline_lane_records import (
    FIRST_LINE_MAX_CHARS,
    LEGACY_INFRA_COMMAND,
    LEGACY_NUMERIC_COMMAND,
    EndedBy,
    FailureRecord,
    LanePass,
    Phase,
    RunRecord,
    RunRecordStore,
    classify_exit,
)
from orchestrator.verify import (
    _extract_failing_test_ids_from_junit,
    extract_test_outcomes_from_junit,
    parse_pytest_failures,
)
from orchestrator.verify_cmd import parse_config_command, render, with_junitxml

if TYPE_CHECKING:
    from orchestrator.config import LaneCommand
    from orchestrator.offline_lane import CommandRunner, SuiteRunner

RUN_OFFLINE_DEEP_SCRIPT = 'scripts/run-offline-deep.sh'

# reify's H9 host-exclusive runner; `--scope host-infra` takes the H8 Lane-X
# flock itself, so the lane takes none.
RUN_ALL_INFRA_SCRIPT = 'tests/infra/run_all.sh'
INFRA_SCOPE = 'host-infra'

# One H9 per-test verdict line, e.g. '  RESULT: FAIL (test_x.sh)' or
# '  RESULT: PASS (test_y.sh) [flaky: passed on serial retry]'.
INFRA_RESULT_RE = re.compile(r'^\s*RESULT:\s*(?P<verdict>PASS|FAIL|SKIP)\s*\((?P<name>[^)]+)\)')

_INFRA_OUTCOMES = {'PASS': 'passed', 'SKIP': 'skipped'}

_TAIL_CHARS = 2000

_PYTEST_OUTCOME = r'\d+ (?:failed|passed|skipped|deselected|xfailed|xpassed|warnings?|errors?|rerun)'
_PYTEST_FINAL_SUMMARY_RE = re.compile(
    rf'^(?:=+ )?(?:no tests ran|{_PYTEST_OUTCOME}(?:, {_PYTEST_OUTCOME})*)'
    r' in \d+(?:\.\d+)?s(?: \(\d+:\d{2}:\d{2}\))?(?: =+)?[ \t]*$',
    re.MULTILINE,
)


class OutputFormat(StrEnum):
    """How a sub-run's output is parsed into failures and ``session_complete``.

    PYTEST: ``session_complete`` is pytest's final summary line being present. Every generic
    command is PYTEST whatever its parse, so a non-pytest one records ``session_complete=False``.
    INFRA and OPAQUE: ``session_complete`` is the run having ended by exit, not by signal.
    """

    PYTEST = 'pytest'
    INFRA = 'infra'
    OPAQUE = 'opaque'


class Launched(NamedTuple):
    record: RunRecord
    tail: str
    text: str


def _pytest_failures(text: str) -> tuple[FailureRecord, ...]:
    return tuple(
        FailureRecord(failure.node_id, failure.kind.value, failure.first_line[:FIRST_LINE_MAX_CHARS])
        for failure in parse_pytest_failures(text)
    )


def _infra_verdicts(text: str) -> list[tuple[str, str, str]]:
    return [
        (match.group('verdict'), match.group('name'), line.strip())
        for line in text.splitlines()
        if (match := INFRA_RESULT_RE.match(line))
    ]


def _infra_failures(text: str) -> tuple[FailureRecord, ...]:
    return tuple(
        FailureRecord(name, 'failed', line[:FIRST_LINE_MAX_CHARS])
        for verdict, name, line in _infra_verdicts(text)
        if verdict == 'FAIL'
    )


def _infra_outcomes(text: str) -> dict[str, str]:
    return {
        name: _INFRA_OUTCOMES[verdict]
        for verdict, name, _ in _infra_verdicts(text)
        if verdict in _INFRA_OUTCOMES
    }


def _failures(text: str, output_format: OutputFormat) -> tuple[FailureRecord, ...]:
    match output_format:
        case OutputFormat.PYTEST:
            return _pytest_failures(text)
        case OutputFormat.INFRA:
            return _infra_failures(text)
        case OutputFormat.OPAQUE:
            return ()


def _session_complete(text: str, output_format: OutputFormat, ended_by: EndedBy) -> bool:
    if output_format is OutputFormat.PYTEST:
        return _PYTEST_FINAL_SUMMARY_RE.search(text) is not None
    return ended_by is EndedBy.EXIT


def _node_prefix(classname: str, cwd: Path) -> str | None:
    if '/' in classname:
        return None
    parts = classname.split('.')
    for end in range(len(parts), 0, -1):
        module = '/'.join(parts[:end]) + '.py'
        if (cwd / module).is_file():
            return '::'.join([module, *parts[end:]])
    return None


def _node_id_resolver(cwd: Path) -> Callable[[str], str | None]:
    """Maps one report's junit ids to pytest node ids under *cwd*, caching each classname's resolution.

    A junit id is ``<dotted classname>::<name>``; the classname's longest dotted
    prefix naming a ``.py`` file under *cwd* is the module, the rest its class
    nesting.  A bare id is a module that failed to collect.  ``None`` for an id
    whose classname resolves to no file.
    """
    prefixes: dict[str, str | None] = {}

    def node_id(junit_id: str) -> str | None:
        classname, separator, name = junit_id.partition('::')
        if classname not in prefixes:
            prefixes[classname] = _node_prefix(classname, cwd)
        prefix = prefixes[classname]
        if prefix is None or not separator:
            return prefix
        return f'{prefix}::{name}'

    return node_id


def junit_node_outcomes(junit_path: Path, cwd: Path) -> dict[str, str] | None:
    """The junit report's per-test outcomes keyed by pytest node id, resolved against *cwd*.

    A test whose classname resolves to no file has no entry.  ``None`` when no report could be read.
    """
    outcomes = extract_test_outcomes_from_junit(junit_path)
    if outcomes is None:
        return None
    resolve = _node_id_resolver(cwd)
    node_outcomes: dict[str, str] = {}
    for junit_id, verdict in outcomes.items():
        node_id = resolve(junit_id)
        if node_id is not None:
            node_outcomes[node_id] = verdict
    return node_outcomes


def junit_failing_node_ids(junit_path: Path, cwd: Path) -> frozenset[str] | None:
    """The pytest node ids the junit report names as failing, resolved against *cwd*.

    ``None`` when no report could be read, or when any failing test's classname
    resolves to no file: such a report cannot say which node ids failed.
    """
    failing = _extract_failing_test_ids_from_junit(junit_path)
    if failing is None:
        return None
    resolve = _node_id_resolver(cwd)
    node_ids: set[str] = set()
    for junit_id in failing:
        node_id = resolve(junit_id)
        if node_id is None:
            return None
        node_ids.add(node_id)
    return frozenset(node_ids)


def _outcomes(
    text: str, output_format: OutputFormat, junit_path: Path | None, cwd: Path,
) -> dict[str, str] | None:
    if junit_path is not None and junit_path.is_file():
        return junit_node_outcomes(junit_path, cwd)
    if output_format is OutputFormat.INFRA:
        return _infra_outcomes(text)
    return None


def _read_text(path: Path) -> str:
    return path.read_bytes().decode(errors='replace')


async def launch(
    argv: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    store: RunRecordStore,
    lane_pass: LanePass,
    phase: Phase,
    output_format: OutputFormat,
    junit_path: Path | None = None,
) -> Launched:
    """Run *argv* with its output streamed to *lane_pass*'s *phase* log, then record it.

    Nothing of the output is held in memory while the child runs; the read,
    the parse, the junit read and the record write happen in one hop off the event loop.
    The whole output comes back as ``text``; ``tail`` is its last ``_TAIL_CHARS`` characters.
    """
    output_path = lane_pass.output_path(phase)
    started_at = store.now()
    with output_path.open('wb') as output:
        proc = await asyncio.create_subprocess_exec(
            *argv, cwd=str(cwd), env=env, stdout=output, stderr=asyncio.subprocess.STDOUT,
        )
        await proc.wait()
    ended_at = store.now()
    returncode = proc.returncode or 0

    def record_run() -> Launched:
        text = _read_text(output_path)
        ended_by = classify_exit(returncode)
        failures = _failures(text, output_format)
        record = RunRecord(
            command=lane_pass.command,
            leg=lane_pass.leg,
            head=lane_pass.head,
            phase=phase.value,
            started_at=started_at,
            ended_at=ended_at,
            returncode=returncode,
            ended_by=ended_by.value,
            output_path=str(output_path),
            junit_path=None if junit_path is None else str(junit_path),
            failures=failures,
            ids_available=bool(failures),
            outcomes=_outcomes(text, output_format, junit_path, cwd),
            session_complete=_session_complete(text, output_format, ended_by),
            probe=None,
        )
        store.write_run(lane_pass, record)
        return Launched(record, text[-_TAIL_CHARS:], text)

    return await asyncio.to_thread(record_run)


def _lane_env() -> dict[str, str]:
    return {**os.environ, 'DF_VERIFY_ROLE': 'offline'}


class ConfirmRunner(Protocol):
    """Runs one confirm sub-run of a bound pass; a junit-producing confirm writes to ``junit_path``."""

    @property
    def junit_path(self) -> Path: ...

    async def __call__(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        output_format: OutputFormat,
        junit_path: Path | None,
    ) -> Launched: ...


@dataclass(frozen=True)
class _PassConfirmRunner:
    store: RunRecordStore
    lane_pass: LanePass
    junit_path: Path

    async def __call__(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        output_format: OutputFormat,
        junit_path: Path | None,
    ) -> Launched:
        return await launch(
            argv,
            cwd=cwd,
            env=_lane_env(),
            store=self.store,
            lane_pass=self.lane_pass,
            phase=Phase.CONFIRM,
            output_format=output_format,
            junit_path=junit_path,
        )


def confirm_runner(store: RunRecordStore, lane_pass: LanePass) -> ConfirmRunner:
    """*lane_pass*'s confirm seam: ``launch`` in the confirm phase with the lane's environment."""
    return _PassConfirmRunner(store, lane_pass, lane_pass.junit_path(Phase.CONFIRM))


def require_current_pass(store: RunRecordStore, command: str) -> LanePass:
    lane_pass = store.current_pass(command)
    if lane_pass is None:
        raise RuntimeError(f'offline-lane: no open pass for {command}')
    return lane_pass


def _with_junit(command: str, junit_path: Path) -> tuple[str, Path | None]:
    parsed = parse_config_command(command)
    rewritten = with_junitxml(parsed, str(junit_path))
    if rewritten is parsed:
        return command, None
    return render(rewritten), junit_path


def suite_runner(store: RunRecordStore) -> SuiteRunner:
    """The numeric leg's detect seam: reify's offline-deep script, which self-nices."""

    async def run_suite(wt_path: Path, head: str, threads: int) -> tuple[int, str]:
        launched = await launch(
            [RUN_OFFLINE_DEEP_SCRIPT, f'--test-threads={threads}'],
            cwd=wt_path,
            env=_lane_env(),
            store=store,
            lane_pass=require_current_pass(store, LEGACY_NUMERIC_COMMAND),
            phase=Phase.DETECT,
            output_format=OutputFormat.OPAQUE,
        )
        return launched.record.returncode, launched.tail

    return run_suite


def infra_runner(store: RunRecordStore) -> SuiteRunner:
    """The infra leg's detect seam: H9's host-infra scope, serial under its own flock."""

    async def run_infra(wt_path: Path, head: str, threads: int) -> tuple[int, str]:
        launched = await launch(
            [RUN_ALL_INFRA_SCRIPT, '--scope', INFRA_SCOPE],
            cwd=wt_path,
            env=_lane_env(),
            store=store,
            lane_pass=require_current_pass(store, LEGACY_INFRA_COMMAND),
            phase=Phase.DETECT,
            output_format=OutputFormat.INFRA,
        )
        return launched.record.returncode, launched.tail

    return run_infra


def command_runner(store: RunRecordStore) -> CommandRunner:
    """A generic command's detect seam: ``sh -c`` at idle nice/ionice in ``<wt>/<cmd.cwd>``.

    A structured pytest command also writes a junit report into the pass directory.
    """

    async def run_command(cmd: LaneCommand, wt_path: Path, head: str) -> tuple[int, str]:
        lane_pass = require_current_pass(store, cmd.name)
        command, junit_path = _with_junit(cmd.command, lane_pass.junit_path(Phase.DETECT))
        launched = await launch(
            [*nice_prefix('background'), 'sh', '-c', command],
            cwd=wt_path / cmd.cwd,
            env=_lane_env(),
            store=store,
            lane_pass=lane_pass,
            phase=Phase.DETECT,
            output_format=OutputFormat.PYTEST,
            junit_path=junit_path,
        )
        return launched.record.returncode, launched.tail

    return run_command
