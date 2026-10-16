"""The confirm of a red offline-lane pass and its typed outcome (task 6858).

Unit tests drive ``confirm`` with a fake ``ConfirmRunner``; seat C2's real-pytest
scenarios follow in this file under ``TestRealPytest…`` classes.
"""

from __future__ import annotations

import json
import logging
import shlex
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from orchestrator.config import LaneCommand
from orchestrator.offline_lane_confirm import (
    CONFIRM_TEST_TIMEOUT_SECS,
    NUMERIC_CONFIRM_IDS_EXIT,
    Confirmed,
    ConfirmMode,
    ConfirmTarget,
    Outage,
    PassOutcome,
    Unconfirmable,
    UnconfirmableReason,
    confirm,
    confirm_mode,
    legacy_confirmed_ids,
    outcome_from_json,
    outcome_to_json,
    partition,
    reproduce_line,
    round_trip_failure,
    serial_confirm_command,
)
from orchestrator.offline_lane_records import (
    LEGACY_INFRA_COMMAND,
    EndedBy,
    FailureRecord,
    Leg,
    Phase,
    ProbeResult,
    RunRecord,
)
from orchestrator.offline_lane_runs import (
    INFRA_SCOPE,
    RUN_ALL_INFRA_SCRIPT,
    RUN_OFFLINE_DEEP_SCRIPT,
    Launched,
    OutputFormat,
    junit_node_outcomes,
)
from orchestrator.verify import _serial_pytest_str
from orchestrator.verify_cmd import ToolKind, parse_config_command

T0 = datetime(2026, 10, 16, 9, 0, 0, tzinfo=UTC)
JUNIT_PATH = Path('/records/unit/pass/confirm.junit.xml')
FUSED_MEMORY_ADDOPTS = "-n auto --dist loadgroup --max-worker-restart=0 -m 'not integration'"
TASK_TEXT_EXAMPLE = (
    "pytest -m integration -p no:xdist -o 'addopts=-m '\"'\"'not integration'\"'\"'' "
    "--timeout 300 --junitxml {junit} 'tests/test_a.py::TestX::test_b[1]'"
)

_VERIFY_USAGE_DUMP = (
    "verify.sh: ERROR — unknown argument '--test-threads=1'\n"
    '\n'
    'Usage:\n'
    '  verify.sh <test|lint|typecheck|all> [options]\n'
    '\n'
    'Options:\n'
    '  --scope <all|rust|gui|infra>  Restrict verification to a subsystem.\n'
)


def _run(
    *failures: tuple[str, str],
    phase: Phase = Phase.DETECT,
    returncode: int = 1,
    ended_by: EndedBy = EndedBy.EXIT,
    outcomes: dict[str, str] | None = None,
    session_complete: bool = True,
    ids_available: bool | None = None,
) -> RunRecord:
    """A run record failing each ``(node, kind)`` of *failures*, in that order."""
    return RunRecord(
        command='unit',
        leg=Leg.GENERIC.value,
        head='abcdef0123',
        phase=phase.value,
        started_at=T0,
        ended_at=T0,
        returncode=returncode,
        ended_by=ended_by.value,
        output_path='/records/unit/pass/run.log',
        junit_path=None,
        failures=tuple(FailureRecord(node, kind, f'{kind}: {node}') for node, kind in failures),
        ids_available=bool(failures) if ids_available is None else ids_available,
        outcomes=outcomes,
        session_complete=session_complete,
        probe=None,
    )


def _rerun(*failures: tuple[str, str], **fields: Any) -> RunRecord:
    return _run(*failures, phase=Phase.CONFIRM, **{'returncode': 1 if failures else 0, **fields})


@dataclass(frozen=True)
class _Call:
    argv: list[str]
    cwd: Path
    output_format: OutputFormat
    junit_path: Path | None


@dataclass
class _FakeRunner:
    """A ``ConfirmRunner`` that records what it was asked to run and hands back *rerun*."""

    rerun: RunRecord
    text: str = ''
    junit_path: Path = JUNIT_PATH
    calls: list[_Call] = field(default_factory=list)

    async def __call__(
        self, argv: Any, *, cwd: Path, output_format: OutputFormat, junit_path: Path | None,
    ) -> Launched:
        self.calls.append(_Call(list(argv), cwd, output_format, junit_path))
        return Launched(self.rerun, self.text[-2000:], self.text)

    @property
    def only_call(self) -> _Call:
        assert len(self.calls) == 1, self.calls
        return self.calls[0]


def _lane_command(command: str = 'pytest -m integration', cwd: str = 'fused-memory') -> LaneCommand:
    return LaneCommand(name='qdrant-integration', command=command, cwd=cwd)


def _project(tmp_path: Path, addopts: str = FUSED_MEMORY_ADDOPTS) -> Path:
    """A worktree whose ``fused-memory`` directory carries a pyproject with *addopts*."""
    project = tmp_path / 'wt' / 'fused-memory'
    project.mkdir(parents=True)
    (project / 'pyproject.toml').write_text(
        f'[tool.pytest.ini_options]\naddopts = "{addopts}"\n', encoding='utf-8',
    )
    return tmp_path / 'wt'


def _generic(tmp_path: Path, command: str = 'pytest -m integration') -> ConfirmTarget:
    return ConfirmTarget.generic(_lane_command(command), _project(tmp_path))


def _narrowable(command: str) -> bool:
    parsed = parse_config_command(command)
    return parsed.tool is ToolKind.PYTEST and parsed.raw is None


_COMMANDS = [
    'pytest -m integration',
    'uv run --project shared pytest tests',
    'timeout -k 60 1800 pytest -m integration',
    'uv run --extra dev pytest tests',
    'pytest -q tests && echo done',
    'make check',
]


class TestConfirmMode:
    @pytest.mark.parametrize('command', _COMMANDS)
    def test_the_mode_follows_the_parse_of_the_command(self, command: str) -> None:
        expected = ConfirmMode.NARROWED if _narrowable(command) else ConfirmMode.INTERSECTED
        assert confirm_mode(_lane_command(command), _run(('tests/t.py::test_a', 'failed'))) is expected

    def test_a_structured_pytest_command_narrows(self) -> None:
        detect = _run(('tests/t.py::test_a', 'failed'))
        assert confirm_mode(_lane_command('pytest -m integration'), detect) is ConfirmMode.NARROWED

    @pytest.mark.parametrize('command', _COMMANDS)
    def test_a_detection_without_ids_intersects(self, command: str) -> None:
        detect = _run(('tests/t.py::test_a', 'failed'), ids_available=False)
        assert confirm_mode(_lane_command(command), detect) is ConfirmMode.INTERSECTED

    @pytest.mark.asyncio
    @pytest.mark.parametrize('command', _COMMANDS)
    async def test_confirm_runs_the_mode_the_parse_selects(self, tmp_path: Path, command: str) -> None:
        target = _generic(tmp_path, command)
        runner = _FakeRunner(_rerun(('tests/t.py::test_a', 'failed')))

        outcome = await confirm(target, _run(('tests/t.py::test_a', 'failed')), runner)

        run_cwd = target.worktree / 'fused-memory'
        call = runner.only_call
        assert (call.cwd, call.output_format) == (run_cwd, OutputFormat.PYTEST)
        assert call.argv[0] == 'nice' and call.argv[-3:-1] == ['sh', '-c']
        if _narrowable(command):
            assert outcome == Confirmed(ConfirmMode.NARROWED, ('tests/t.py::test_a',), (), ())
            assert call.junit_path == JUNIT_PATH
            assert call.argv[-1] == serial_confirm_command(
                _lane_command(command), ['tests/t.py::test_a'], invocation_dir=run_cwd, junit_path=JUNIT_PATH,
            )
        else:
            assert outcome == Confirmed(ConfirmMode.INTERSECTED, ('tests/t.py::test_a',), (), ())
            assert call.junit_path is None
            assert call.argv[-1] == (_serial_pytest_str(command, invocation_dir=run_cwd) or command)


class TestSerialConfirmCommand:
    def test_the_narrowed_command_renders_the_task_texts_example(self, tmp_path: Path) -> None:
        worktree = _project(tmp_path)
        command = serial_confirm_command(
            _lane_command(),
            ['tests/test_a.py::TestX::test_b[1]'],
            invocation_dir=worktree / 'fused-memory',
            junit_path=JUNIT_PATH,
        )
        assert command == TASK_TEXT_EXAMPLE.format(junit=JUNIT_PATH)

    def test_it_runs_serially_with_the_governing_addopts_minus_xdist(self, tmp_path: Path) -> None:
        worktree = _project(tmp_path, addopts="-n 4 -m 'not warm_lane_bash'")
        tokens = shlex.split(serial_confirm_command(
            _lane_command('pytest -m warm_lane_bash'),
            ['tests/t.py::test_a'],
            invocation_dir=worktree / 'fused-memory',
            junit_path=JUNIT_PATH,
        ))
        assert tokens[:7] == ['pytest', '-m', 'warm_lane_bash', '-p', 'no:xdist', '-o', "addopts=-m 'not warm_lane_bash'"]

    def test_it_carries_the_per_test_timeout_and_the_junit_path(self, tmp_path: Path) -> None:
        worktree = _project(tmp_path)
        tokens = shlex.split(serial_confirm_command(
            _lane_command(), ['tests/t.py::test_a'], invocation_dir=worktree / 'fused-memory', junit_path=JUNIT_PATH,
        ))
        assert tokens[tokens.index('--timeout') + 1] == str(CONFIRM_TEST_TIMEOUT_SECS)
        assert tokens[tokens.index('--junitxml') + 1] == str(JUNIT_PATH)
        assert tokens[-1] == 'tests/t.py::test_a'


class TestReproduceLine:
    def test_a_structured_command_gives_its_directory_and_the_narrowed_command(self) -> None:
        line = reproduce_line(_lane_command(), ['tests/test_a.py::TestX::test_b[1]', 'tests/t.py::test_c'])
        assert line == "cd fused-memory && pytest -m integration 'tests/test_a.py::TestX::test_b[1]' tests/t.py::test_c"

    @pytest.mark.parametrize('command', [c for c in _COMMANDS if not _narrowable(c)])
    def test_a_command_that_cannot_be_narrowed_has_none(self, command: str) -> None:
        assert reproduce_line(_lane_command(command), ['tests/t.py::test_a']) is None


class TestRoundTrip:
    def test_an_id_truncated_at_whitespace_fails(self) -> None:
        assert round_trip_failure(_lane_command(), ['tests/t.py::test_ok', 'tests/t.py::test_a[x']) == (
            'tests/t.py::test_a[x'
        )

    def test_an_id_the_parser_reads_back_as_a_flag_fails(self) -> None:
        assert round_trip_failure(_lane_command(), ['tests/t.py::test_ok', '-x']) == '-x'

    def test_whole_ids_survive(self) -> None:
        ids = ['tests/t.py::test_a[x y]', 'tests/t.py::TestK::test_b[a-[1]]', "tests/t.py::test_q['q']"]
        assert round_trip_failure(_lane_command(), ids) is None

    def test_the_first_failing_id_is_named(self) -> None:
        assert round_trip_failure(_lane_command(), ['tests/t.py::a]', 'tests/t.py::b[']) == 'tests/t.py::a]'


class TestPartition:
    """Each detection node lands in exactly one of reproduced, cleared, inconclusive."""

    @pytest.mark.parametrize(
        ('detect_kind', 'rerun_kind'), [('failed', 'failed'), ('error', 'error'), ('error', 'failed')],
    )
    def test_a_compatible_rerun_failure_reproduces(self, detect_kind: str, rerun_kind: str) -> None:
        detect = _run(('tests/t.py::test_a', detect_kind))
        rerun = _rerun(('tests/t.py::test_a', rerun_kind))
        assert partition(detect, rerun) == (('tests/t.py::test_a',), (), ())

    def test_a_confirm_error_on_a_detected_failure_is_inconclusive(self) -> None:
        detect = _run(('tests/t.py::test_a', 'failed'))
        rerun = _rerun(('tests/t.py::test_a', 'error'), outcomes={'tests/t.py::test_a': 'passed'})
        assert partition(detect, rerun) == ((), (), ('tests/t.py::test_a',))

    def test_per_test_evidence_of_a_pass_clears_even_an_incomplete_session(self) -> None:
        detect = _run(('tests/t.py::test_a', 'failed'))
        rerun = _rerun(outcomes={'tests/t.py::test_a': 'passed'}, session_complete=False)
        assert partition(detect, rerun) == ((), ('tests/t.py::test_a',), ())

    def test_per_test_evidence_of_a_skip_is_inconclusive_even_in_a_complete_session(self) -> None:
        detect = _run(('tests/t.py::test_a', 'failed'))
        rerun = _rerun(outcomes={'tests/t.py::test_a': 'skipped'}, session_complete=True)
        assert partition(detect, rerun) == ((), (), ('tests/t.py::test_a',))

    def test_a_complete_session_that_did_not_name_it_clears(self) -> None:
        detect = _run(('tests/t.py::test_a', 'failed'))
        assert partition(detect, _rerun(session_complete=True)) == ((), ('tests/t.py::test_a',), ())

    def test_an_incomplete_session_that_did_not_name_it_is_inconclusive(self) -> None:
        detect = _run(('tests/t.py::test_a', 'failed'))
        assert partition(detect, _rerun(session_complete=False)) == ((), (), ('tests/t.py::test_a',))

    def test_detection_order_is_kept_and_rerun_only_nodes_are_ignored(self) -> None:
        detect = _run(
            ('tests/t.py::test_c', 'failed'),
            ('tests/t.py::test_a', 'failed'),
            ('tests/t.py::test_b', 'failed'),
            ('tests/t.py::test_a', 'error'),
        )
        rerun = _rerun(
            ('tests/t.py::test_z', 'failed'),
            ('tests/t.py::test_b', 'failed'),
            ('tests/t.py::test_c', 'failed'),
            outcomes={'tests/t.py::test_a': 'passed'},
        )
        assert partition(detect, rerun) == (
            ('tests/t.py::test_c', 'tests/t.py::test_b'), ('tests/t.py::test_a',), (),
        )


class TestConfirmUnconfirmable:
    @pytest.mark.asyncio
    async def test_no_detect_record(self, tmp_path: Path) -> None:
        runner = _FakeRunner(_rerun())
        outcome = await confirm(_generic(tmp_path), None, runner)
        assert outcome == Unconfirmable(UnconfirmableReason.NO_DETECT_RECORD, '')
        assert runner.calls == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize('leg', list(Leg))
    async def test_a_signal_ended_detection_carries_its_returncode(self, tmp_path: Path, leg: Leg) -> None:
        target = _generic(tmp_path) if leg is Leg.GENERIC else ConfirmTarget(leg, tmp_path, None)
        runner = _FakeRunner(_rerun())
        detect = _run(('tests/t.py::test_a', 'failed'), returncode=-9, ended_by=EndedBy.SIGNAL)

        outcome = await confirm(target, detect, runner)

        assert outcome == Unconfirmable(UnconfirmableReason.DETECT_SIGNAL_ENDED, '-9')
        assert runner.calls == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize('leg', [Leg.GENERIC, Leg.INFRA])
    async def test_a_red_detection_naming_no_test_runs_no_confirm(self, tmp_path: Path, leg: Leg) -> None:
        target = _generic(tmp_path) if leg is Leg.GENERIC else ConfirmTarget.infra(tmp_path)
        runner = _FakeRunner(_rerun())

        outcome = await confirm(target, _run(returncode=2), runner)

        assert outcome == Unconfirmable(UnconfirmableReason.DETECT_RED_NAMED_NO_TEST, '2')
        assert runner.calls == []

    @pytest.mark.asyncio
    async def test_an_id_failing_the_round_trip_runs_no_confirm(self, tmp_path: Path) -> None:
        runner = _FakeRunner(_rerun())
        detect = _run(('tests/t.py::test_ok', 'failed'), ('tests/t.py::test_a[x', 'failed'))

        outcome = await confirm(_generic(tmp_path), detect, runner)

        assert outcome == Unconfirmable(UnconfirmableReason.ID_FAILED_ROUND_TRIP, 'tests/t.py::test_a[x')
        assert runner.calls == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize('leg', list(Leg))
    async def test_a_signal_ended_confirm_carries_its_returncode(self, tmp_path: Path, leg: Leg) -> None:
        target = _generic(tmp_path) if leg is Leg.GENERIC else ConfirmTarget(leg, tmp_path, None)
        runner = _FakeRunner(_rerun(('tests/t.py::test_a', 'failed'), returncode=137, ended_by=EndedBy.SIGNAL))

        outcome = await confirm(target, _run(('tests/t.py::test_a', 'failed')), runner)

        assert outcome == Unconfirmable(UnconfirmableReason.CONFIRM_SIGNAL_ENDED, '137')

    @pytest.mark.asyncio
    @pytest.mark.parametrize('command', ['pytest -m integration', 'make check'])
    async def test_a_generic_confirm_exiting_nonzero_naming_no_test(self, tmp_path: Path, command: str) -> None:
        runner = _FakeRunner(_rerun(returncode=2, session_complete=False))
        outcome = await confirm(_generic(tmp_path, command), _run(('tests/t.py::test_a', 'failed')), runner)
        assert outcome == Unconfirmable(UnconfirmableReason.CONFIRM_EXITED_NONZERO_NAMING_NO_TEST, '2')

    @pytest.mark.asyncio
    async def test_an_infra_crash_printing_no_failing_line(self, tmp_path: Path) -> None:
        runner = _FakeRunner(_rerun(returncode=1, outcomes={'test_y.sh': 'passed'}))
        outcome = await confirm(ConfirmTarget.infra(tmp_path), _run(('test_x.sh', 'failed')), runner)
        assert outcome == Unconfirmable(UnconfirmableReason.CONFIRM_EXITED_NONZERO_NAMING_NO_TEST, '1')

    @pytest.mark.asyncio
    async def test_a_numeric_exit_other_than_0_or_100(self, tmp_path: Path) -> None:
        runner = _FakeRunner(_rerun(returncode=64), text='tests::test_a\n')
        outcome = await confirm(ConfirmTarget.numeric(tmp_path), _run(returncode=1), runner)
        assert outcome == Unconfirmable(UnconfirmableReason.NUMERIC_EXIT, '64')

    @pytest.mark.asyncio
    @pytest.mark.parametrize('returncode', [NUMERIC_CONFIRM_IDS_EXIT, 64])
    async def test_numeric_usage_text_is_not_a_list_of_ids(self, tmp_path: Path, returncode: int) -> None:
        runner = _FakeRunner(_rerun(returncode=returncode), text=_VERIFY_USAGE_DUMP)
        outcome = await confirm(ConfirmTarget.numeric(tmp_path), _run(returncode=1), runner)
        assert outcome == Unconfirmable(
            UnconfirmableReason.USAGE_TEXT, "verify.sh: ERROR — unknown argument '--test-threads=1'",
        )

    @pytest.mark.parametrize(
        'line', ['Usage:', '  Options:  ', 'verify.sh: ERROR missing scope'], ids=['usage', 'options', 'error'],
    )
    @pytest.mark.asyncio
    async def test_each_usage_marker_alone_is_usage_text(self, tmp_path: Path, line: str) -> None:
        runner = _FakeRunner(_rerun(returncode=100), text=f'tests::test_a\n{line}\n')
        outcome = await confirm(ConfirmTarget.numeric(tmp_path), _run(returncode=1), runner)
        assert outcome == Unconfirmable(UnconfirmableReason.USAGE_TEXT, line.strip())

    def test_every_reason_is_covered_here(self) -> None:
        covered = {
            UnconfirmableReason.NO_DETECT_RECORD,
            UnconfirmableReason.DETECT_SIGNAL_ENDED,
            UnconfirmableReason.DETECT_RED_NAMED_NO_TEST,
            UnconfirmableReason.ID_FAILED_ROUND_TRIP,
            UnconfirmableReason.CONFIRM_SIGNAL_ENDED,
            UnconfirmableReason.CONFIRM_EXITED_NONZERO_NAMING_NO_TEST,
            UnconfirmableReason.NUMERIC_EXIT,
            UnconfirmableReason.USAGE_TEXT,
        }
        assert covered == set(UnconfirmableReason)


class TestLegacyLegs:
    @pytest.mark.asyncio
    async def test_numeric_exit_100_reproduces_the_printed_ids(self, tmp_path: Path) -> None:
        runner = _FakeRunner(_rerun(returncode=100), text='tests::test_b\n\n  tests::test_a  \n')

        outcome = await confirm(ConfirmTarget.numeric(tmp_path), _run(returncode=1), runner)

        assert outcome == Confirmed(ConfirmMode.LEG_NATIVE, ('tests::test_b', 'tests::test_a'), (), ())
        assert runner.only_call == _Call(
            [RUN_OFFLINE_DEEP_SCRIPT, '--test-threads=1', '--confirm-failed'], tmp_path, OutputFormat.OPAQUE, None,
        )

    @pytest.mark.asyncio
    async def test_numeric_exit_0_reproduces_and_names_nothing(self, tmp_path: Path) -> None:
        runner = _FakeRunner(_rerun(returncode=0), text='all green\n')
        outcome = await confirm(ConfirmTarget.numeric(tmp_path), _run(returncode=1), runner)
        assert outcome == Confirmed(ConfirmMode.LEG_NATIVE, (), (), ())

    @pytest.mark.asyncio
    async def test_infra_intersects_the_two_runs(self, tmp_path: Path) -> None:
        detect = _run(('test_x.sh', 'failed'), ('test_y.sh', 'failed'), ('test_z.sh', 'failed'))
        runner = _FakeRunner(_rerun(
            ('test_new.sh', 'failed'), ('test_x.sh', 'failed'),
            outcomes={'test_y.sh': 'passed', 'test_z.sh': 'skipped'},
        ))

        outcome = await confirm(ConfirmTarget.infra(tmp_path), detect, runner)

        assert outcome == Confirmed(ConfirmMode.INTERSECTED, ('test_x.sh',), ('test_y.sh',), ('test_z.sh',))
        assert runner.only_call == _Call(
            [RUN_ALL_INFRA_SCRIPT, '--scope', INFRA_SCOPE], tmp_path, OutputFormat.INFRA, None,
        )

    @pytest.mark.asyncio
    async def test_an_opaque_confirm_red_only_on_another_test_reproduces_nothing(self, tmp_path: Path) -> None:
        detect = _run(('tests/t.py::test_a', 'failed'))
        runner = _FakeRunner(_rerun(('tests/t.py::test_b', 'failed')))

        outcome = await confirm(_generic(tmp_path, 'make check'), detect, runner)

        assert outcome == Confirmed(ConfirmMode.INTERSECTED, (), ('tests/t.py::test_a',), ())


class TestConfirmTarget:
    def test_the_constructors_name_their_leg(self, tmp_path: Path) -> None:
        command = _lane_command()
        assert ConfirmTarget.generic(command, tmp_path) == ConfirmTarget(Leg.GENERIC, tmp_path, command)
        assert ConfirmTarget.numeric(tmp_path) == ConfirmTarget(Leg.NUMERIC, tmp_path, None)
        assert ConfirmTarget.infra(tmp_path) == ConfirmTarget(Leg.INFRA, tmp_path, None)

    @pytest.mark.parametrize(
        ('leg', 'with_command'), [(Leg.GENERIC, False), (Leg.NUMERIC, True), (Leg.INFRA, True)],
    )
    def test_a_command_goes_with_the_generic_leg_only(self, tmp_path: Path, leg: Leg, with_command: bool) -> None:
        with pytest.raises(ValueError, match='generic'):
            ConfirmTarget(leg, tmp_path, _lane_command() if with_command else None)


class TestLegacyConfirmedIds:
    def test_confirmed_projects_to_its_sorted_reproduced_ids(self) -> None:
        outcome = Confirmed(ConfirmMode.NARROWED, ('tests/t.py::b', 'tests/t.py::a'), ('tests/t.py::c',), ())
        assert legacy_confirmed_ids(outcome, command_name='unit', leg=Leg.GENERIC) == ['tests/t.py::a', 'tests/t.py::b']

    def test_a_generic_confirm_exiting_nonzero_naming_no_test_is_the_command_sentinel(self) -> None:
        outcome = Unconfirmable(UnconfirmableReason.CONFIRM_EXITED_NONZERO_NAMING_NO_TEST, '2')
        assert legacy_confirmed_ids(outcome, command_name='unit', leg=Leg.GENERIC) == ['unit::unconfirmable']

    @pytest.mark.parametrize(
        ('reason', 'leg'),
        [
            (UnconfirmableReason.CONFIRM_EXITED_NONZERO_NAMING_NO_TEST, Leg.INFRA),
            (UnconfirmableReason.DETECT_RED_NAMED_NO_TEST, Leg.GENERIC),
            (UnconfirmableReason.CONFIRM_SIGNAL_ENDED, Leg.GENERIC),
            (UnconfirmableReason.ID_FAILED_ROUND_TRIP, Leg.GENERIC),
            (UnconfirmableReason.NUMERIC_EXIT, Leg.NUMERIC),
        ],
    )
    def test_every_other_unconfirmable_confirms_nothing_with_a_warning(
        self, caplog: pytest.LogCaptureFixture, reason: UnconfirmableReason, leg: Leg,
    ) -> None:
        with caplog.at_level(logging.WARNING, logger='orchestrator.offline_lane_confirm'):
            ids = legacy_confirmed_ids(Unconfirmable(reason, 'why'), command_name=LEGACY_INFRA_COMMAND, leg=leg)
        assert ids == []
        assert [r.getMessage() for r in caplog.records] == [
            f'offline-lane: {LEGACY_INFRA_COMMAND} pass unconfirmable: {reason} why',
        ]


_PROBE = ProbeResult(result='outage', returncode=2, failed_dependency='qdrant', first_line='refused', duration=1.5)


class TestOutcomeJson:
    @pytest.mark.parametrize(
        'outcome',
        [
            Confirmed(ConfirmMode.NARROWED, ('tests/t.py::a',), ('tests/t.py::b',), ('tests/t.py::c[x y]',)),
            Confirmed(ConfirmMode.LEG_NATIVE, (), (), ()),
            Unconfirmable(UnconfirmableReason.CONFIRM_SIGNAL_ENDED, '-9'),
            Outage(_PROBE),
        ],
        ids=['confirmed', 'leg-native', 'unconfirmable', 'outage'],
    )
    def test_each_arm_round_trips_through_json_text(self, outcome: PassOutcome) -> None:
        payload = json.loads(json.dumps(outcome_to_json(outcome)))
        assert outcome_from_json(payload) == outcome

    def test_the_arm_is_named(self) -> None:
        assert outcome_to_json(Outage(_PROBE))['arm'] == 'outage'
        assert outcome_to_json(Unconfirmable(UnconfirmableReason.USAGE_TEXT, ''))['arm'] == 'unconfirmable'

    @pytest.mark.parametrize(
        'payload',
        [
            {},
            {'arm': 'nope'},
            {'arm': 'confirmed', 'mode': 'sideways', 'reproduced': [], 'cleared': [], 'inconclusive': []},
            {'arm': 'confirmed', 'mode': 'narrowed', 'reproduced': [1], 'cleared': [], 'inconclusive': []},
            {'arm': 'confirmed', 'mode': 'narrowed', 'reproduced': [], 'cleared': []},
            {'arm': 'unconfirmable', 'reason': 'bored', 'detail': ''},
            {'arm': 'unconfirmable', 'reason': 'usage_text'},
            {'arm': 'outage', 'probe': {'result': 'outage'}},
            {'arm': 'outage'},
        ],
    )
    def test_a_malformed_payload_is_a_value_error(self, payload: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            outcome_from_json(payload)


class TestJunitNodeOutcomes:
    def test_the_longest_module_prefix_on_disk_wins(self, tmp_path: Path) -> None:
        (tmp_path / 'tests').mkdir()
        (tmp_path / 'tests.py').write_text('')
        (tmp_path / 'tests' / 'test_a.py').write_text('')
        report = tmp_path / 'r.xml'
        report.write_text(
            '<testsuite><testcase classname="tests.test_a.TestX.TestY" name="test_b[1]"/>'
            '<testcase classname="tests.test_a" name="test_c"><skipped/></testcase></testsuite>'
        )
        assert junit_node_outcomes(report, tmp_path) == {
            'tests/test_a.py::TestX::TestY::test_b[1]': 'passed',
            'tests/test_a.py::test_c': 'skipped',
        }

    def test_no_report_is_none(self, tmp_path: Path) -> None:
        assert junit_node_outcomes(tmp_path / 'absent.xml', tmp_path) is None


class TestIntersectedRoundTrip:
    @pytest.mark.asyncio
    async def test_an_intersected_id_with_unbalanced_brackets_runs_no_confirm(self, tmp_path: Path) -> None:
        runner = _FakeRunner(_rerun(('tests/t.py::test_a[x', 'failed')))
        detect = _run(('tests/t.py::test_ok', 'failed'), ('tests/t.py::test_a[x', 'failed'))

        outcome = await confirm(_generic(tmp_path, 'make check'), detect, runner)

        assert outcome == Unconfirmable(UnconfirmableReason.ID_FAILED_ROUND_TRIP, 'tests/t.py::test_a[x')
        assert runner.calls == []
