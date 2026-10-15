"""How the offline lane runs a sub-run: one streaming launcher and the three detection seams.

Every sub-run, detection or confirm, streams its child's combined output
straight to a file in its pass directory and leaves a run record beside it
(``orchestrator/src/orchestrator/offline_lane_records.py::RunRecordStore``).
"""

from __future__ import annotations

import asyncio
import os
import re
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

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
from orchestrator.verify import parse_pytest_failures
from orchestrator.verify_cmd import parse_config_command, render, with_junitxml

if TYPE_CHECKING:
    from orchestrator.config import LaneCommand
    from orchestrator.offline_lane import CommandRunner, SuiteRunner

RUN_OFFLINE_DEEP_SCRIPT = 'scripts/run-offline-deep.sh'

# reify's H9 host-exclusive runner; `--scope host-infra` takes the H8 Lane-X
# flock itself, so the lane takes none.
RUN_ALL_INFRA_SCRIPT = 'tests/infra/run_all.sh'
INFRA_SCOPE = 'host-infra'

# One H9 per-test verdict line, e.g. '  RESULT: FAIL (test_x.sh)'.
INFRA_RESULT_FAIL_RE = re.compile(r'^\s*RESULT:\s*FAIL\s*\((?P<name>.+?)\)\s*$')

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
    text: str | None


def _pytest_failures(text: str) -> tuple[FailureRecord, ...]:
    return tuple(
        FailureRecord(failure.node_id, failure.kind.value, failure.first_line[:FIRST_LINE_MAX_CHARS])
        for failure in parse_pytest_failures(text)
    )


def _infra_failures(text: str) -> tuple[FailureRecord, ...]:
    return tuple(
        FailureRecord(match.group('name'), 'failed', line.strip()[:FIRST_LINE_MAX_CHARS])
        for line in text.splitlines()
        if (match := INFRA_RESULT_FAIL_RE.match(line))
    )


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
    keep_text: bool = False,
) -> Launched:
    """Run *argv* with its output streamed to *lane_pass*'s *phase* log, then record it.

    Nothing of the output is held in memory while the child runs; the read,
    the parse and the record write happen in one hop off the event loop.
    The whole output comes back as ``text`` only when *keep_text* asks for it.
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
            outcomes=None,
            session_complete=_session_complete(text, output_format, ended_by),
            probe=None,
        )
        store.write_run(lane_pass, record)
        return Launched(record, text[-_TAIL_CHARS:], text if keep_text else None)

    return await asyncio.to_thread(record_run)


def _lane_env() -> dict[str, str]:
    return {**os.environ, 'DF_VERIFY_ROLE': 'offline'}


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
