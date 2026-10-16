"""The confirm of a red offline-lane pass, and the typed outcome it ends in.

A red pass ends in exactly one of ``Outage``, ``Confirmed`` or ``Unconfirmable``
(PRD ``plans/offline-lane-red-path-prd.md`` §5.2).  ``confirm`` decides between
the last two from the pass's detect record and one confirm sub-run: it narrows a
structured pytest command to the detection's failing tests and runs them
serially, and otherwise re-runs the whole command and intersects the two runs.
The numeric leg's outcome comes from its confirm alone.
"""

from __future__ import annotations

import logging
import re
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from shared.verify_admission import nice_prefix

from orchestrator.offline_lane_records import EndedBy, Leg, ProbeResult, RunRecord
from orchestrator.offline_lane_runs import (
    INFRA_SCOPE,
    RUN_ALL_INFRA_SCRIPT,
    RUN_OFFLINE_DEEP_SCRIPT,
    ConfirmRunner,
    OutputFormat,
)
from orchestrator.verify import (
    FailureKind,
    PytestFailure,
    _serial_pytest_str,
    _with_junitxml_str,
    _with_pytest_timeout_str,
    node_ids_of,
    reproduced_failures,
)
from orchestrator.verify_cmd import ToolKind, VerifyCmd, parse_config_command, render, scope_to

if TYPE_CHECKING:
    from orchestrator.config import LaneCommand

logger = logging.getLogger(__name__)

CONFIRM_TEST_TIMEOUT_SECS = 300
NUMERIC_CONFIRM_IDS_EXIT = 100

# Markers of reify verify.sh's usage()/error text (task 5308): a whole line that
# is exactly ``Usage:`` or ``Options:``, or a ``verify.sh: ERROR`` banner.
# run-offline-deep.sh merges stderr into stdout, so a malformed invocation
# floods the numeric confirm's id list with this text.
_VERIFY_USAGE_MARKER_RE = re.compile(r'^\s*(?:Usage|Options):\s*$|^\s*verify\.sh:\s*ERROR\b')

_T = TypeVar('_T')


class ConfirmMode(StrEnum):
    NARROWED = 'narrowed'
    INTERSECTED = 'intersected'
    LEG_NATIVE = 'leg_native'


class UnconfirmableReason(StrEnum):
    NO_DETECT_RECORD = 'no_detect_record'
    DETECT_SIGNAL_ENDED = 'detect_signal_ended'
    DETECT_RED_NAMED_NO_TEST = 'detect_red_named_no_test'
    ID_FAILED_ROUND_TRIP = 'id_failed_round_trip'
    CONFIRM_SIGNAL_ENDED = 'confirm_signal_ended'
    CONFIRM_EXITED_NONZERO_NAMING_NO_TEST = 'confirm_exited_nonzero_naming_no_test'
    NUMERIC_EXIT = 'numeric_exit'
    USAGE_TEXT = 'usage_text'


@dataclass(frozen=True)
class Confirmed:
    """``reproduced``, ``cleared`` and ``inconclusive`` partition the detection's failing nodes, in detection order."""

    mode: ConfirmMode
    reproduced: tuple[str, ...]
    cleared: tuple[str, ...]
    inconclusive: tuple[str, ...]


@dataclass(frozen=True)
class Unconfirmable:
    reason: UnconfirmableReason
    detail: str


@dataclass(frozen=True)
class Outage:
    probe: ProbeResult


PassOutcome = Outage | Confirmed | Unconfirmable


def outcome_to_json(outcome: PassOutcome) -> dict[str, Any]:
    match outcome:
        case Confirmed():
            return {
                'arm': 'confirmed',
                'mode': outcome.mode.value,
                'reproduced': list(outcome.reproduced),
                'cleared': list(outcome.cleared),
                'inconclusive': list(outcome.inconclusive),
            }
        case Unconfirmable():
            return {'arm': 'unconfirmable', 'reason': outcome.reason.value, 'detail': outcome.detail}
        case Outage():
            return {'arm': 'outage', 'probe': asdict(outcome.probe)}


def _field(payload: Mapping[str, Any], key: str, expected: type[_T]) -> _T:
    value = payload.get(key)
    if not isinstance(value, expected):
        raise ValueError(f'pass outcome {key!r} must be {expected.__name__}, got {value!r}')
    return value


def _nodes(payload: Mapping[str, Any], key: str) -> tuple[str, ...]:
    nodes = _field(payload, key, list)
    if not all(isinstance(node, str) for node in nodes):
        raise ValueError(f'pass outcome {key!r} must list strings')
    return tuple(nodes)


def outcome_from_json(payload: Mapping[str, Any]) -> PassOutcome:
    """The inverse of :func:`outcome_to_json`; ``ValueError`` on a malformed payload."""
    match _field(payload, 'arm', str):
        case 'confirmed':
            return Confirmed(
                ConfirmMode(_field(payload, 'mode', str)),
                _nodes(payload, 'reproduced'),
                _nodes(payload, 'cleared'),
                _nodes(payload, 'inconclusive'),
            )
        case 'unconfirmable':
            return Unconfirmable(
                UnconfirmableReason(_field(payload, 'reason', str)), _field(payload, 'detail', str),
            )
        case 'outage':
            return Outage(ProbeResult.from_json(payload.get('probe')))
        case arm:
            raise ValueError(f'unknown pass outcome arm {arm!r}')


@dataclass(frozen=True)
class ConfirmTarget:
    """What a pass confirms: a generic lane command, or one of the two legacy legs."""

    leg: Leg
    worktree: Path
    command: LaneCommand | None

    def __post_init__(self) -> None:
        if (self.leg is Leg.GENERIC) != (self.command is not None):
            raise ValueError(f'a {self.leg} confirm target takes a command exactly when it is generic')

    @classmethod
    def generic(cls, command: LaneCommand, worktree: Path) -> ConfirmTarget:
        return cls(Leg.GENERIC, worktree, command)

    @classmethod
    def numeric(cls, worktree: Path) -> ConfirmTarget:
        return cls(Leg.NUMERIC, worktree, None)

    @classmethod
    def infra(cls, worktree: Path) -> ConfirmTarget:
        return cls(Leg.INFRA, worktree, None)


def _narrowable(parsed: VerifyCmd) -> bool:
    return parsed.tool is ToolKind.PYTEST and parsed.raw is None


def _narrowed(cmd: LaneCommand, ids: Sequence[str]) -> str:
    return render(scope_to(parse_config_command(cmd.command), list(ids)))


def serial_confirm_command(
    cmd: LaneCommand, ids: Sequence[str], *, invocation_dir: Path, junit_path: Path,
) -> str:
    """The narrowed confirm of *ids*: serial, with a per-test timeout, writing junit to *junit_path*."""
    narrowed = _narrowed(cmd, ids)
    serial = _serial_pytest_str(narrowed, invocation_dir=invocation_dir) or narrowed
    timed = _with_pytest_timeout_str(serial, CONFIRM_TEST_TIMEOUT_SECS) or serial
    return _with_junitxml_str(timed, str(junit_path)) or timed


def reproduce_line(cmd: LaneCommand, ids: Sequence[str]) -> str | None:
    """The line a person runs to reproduce *ids* of *cmd*; ``None`` when *cmd* cannot be narrowed."""
    if not _narrowable(parse_config_command(cmd.command)):
        return None
    return f'cd {shlex.quote(cmd.cwd)} && {_narrowed(cmd, ids)}'


def confirm_mode(cmd: LaneCommand, detect: RunRecord) -> ConfirmMode:
    if _narrowable(parse_config_command(cmd.command)) and detect.ids_available:
        return ConfirmMode.NARROWED
    return ConfirmMode.INTERSECTED


_BRACKET_DEPTH = {'[': 1, ']': -1}


def _brackets_balance(node_id: str) -> bool:
    depth = 0
    for char in node_id:
        depth += _BRACKET_DEPTH.get(char, 0)
        if depth < 0:
            return False
    return depth == 0


def round_trip_failure(cmd: LaneCommand, ids: Sequence[str]) -> str | None:
    """The first of *ids* that does not survive narrowing *cmd* to it and re-parsing; ``None`` when all do."""
    parsed = parse_config_command(cmd.command)
    for node_id in ids:
        reparsed = parse_config_command(render(scope_to(parsed, [node_id])))
        if not _brackets_balance(node_id) or reparsed.targets != (node_id,):
            return node_id
    return None


def _as_pytest_failures(record: RunRecord) -> tuple[PytestFailure, ...]:
    return tuple(
        PytestFailure(failure.node, FailureKind(failure.kind), failure.first_line)
        for failure in record.failures
    )


class _Verdict(StrEnum):
    REPRODUCED = 'reproduced'
    CLEARED = 'cleared'
    INCONCLUSIVE = 'inconclusive'


_EVIDENCE_VERDICTS = {'passed': _Verdict.CLEARED, 'skipped': _Verdict.INCONCLUSIVE}


def _verdict(node: str, reproduced: set[str], rerun: RunRecord) -> _Verdict:
    if node in reproduced:
        return _Verdict.REPRODUCED
    if any(failure.node == node for failure in rerun.failures):
        return _Verdict.INCONCLUSIVE
    evidence = (rerun.outcomes or {}).get(node)
    if evidence in _EVIDENCE_VERDICTS:
        return _EVIDENCE_VERDICTS[evidence]
    return _Verdict.CLEARED if rerun.session_complete else _Verdict.INCONCLUSIVE


def _nodes_with(verdicts: Mapping[str, _Verdict], wanted: _Verdict) -> tuple[str, ...]:
    return tuple(node for node, verdict in verdicts.items() if verdict is wanted)


def partition(
    detect: RunRecord, rerun: RunRecord,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """``(reproduced, cleared, inconclusive)`` over *detect*'s failing nodes, in detection order.

    Reproduced: *rerun* fails the node with a compatible kind
    (``orchestrator/src/orchestrator/verify.py::reproduced_failures``).
    Inconclusive: *rerun* names it with an incompatible kind, or its per-test
    evidence is a skip, or *rerun* neither completed its session nor named it.
    Cleared: its per-test evidence is a pass, or *rerun* completed without naming it.
    """
    original = _as_pytest_failures(detect)
    reproduced = {f.node_id for f in reproduced_failures(original, _as_pytest_failures(rerun))}
    verdicts = {node: _verdict(node, reproduced, rerun) for node in node_ids_of(original)}
    return (
        _nodes_with(verdicts, _Verdict.REPRODUCED),
        _nodes_with(verdicts, _Verdict.CLEARED),
        _nodes_with(verdicts, _Verdict.INCONCLUSIVE),
    )


def _compared(mode: ConfirmMode, detect: RunRecord, rerun: RunRecord) -> Confirmed | Unconfirmable:
    if rerun.ended_by == EndedBy.SIGNAL:
        return Unconfirmable(UnconfirmableReason.CONFIRM_SIGNAL_ENDED, str(rerun.returncode))
    if rerun.returncode != 0 and not rerun.failures:
        return Unconfirmable(
            UnconfirmableReason.CONFIRM_EXITED_NONZERO_NAMING_NO_TEST, str(rerun.returncode),
        )
    reproduced, cleared, inconclusive = partition(detect, rerun)
    return Confirmed(mode, reproduced, cleared, inconclusive)


async def _confirm_generic(
    cmd: LaneCommand, worktree: Path, detect: RunRecord, runner: ConfirmRunner,
) -> Confirmed | Unconfirmable:
    run_cwd = worktree / cmd.cwd
    mode = confirm_mode(cmd, detect)
    if mode is ConfirmMode.NARROWED:
        ids = node_ids_of(_as_pytest_failures(detect))
        failing_id = round_trip_failure(cmd, ids)
        if failing_id is not None:
            return Unconfirmable(UnconfirmableReason.ID_FAILED_ROUND_TRIP, failing_id)
        junit_path = runner.junit_path
        command = serial_confirm_command(cmd, ids, invocation_dir=run_cwd, junit_path=junit_path)
    else:
        junit_path = None
        command = _serial_pytest_str(cmd.command, invocation_dir=run_cwd) or cmd.command
    launched = await runner(
        [*nice_prefix('background'), 'sh', '-c', command],
        cwd=run_cwd,
        output_format=OutputFormat.PYTEST,
        junit_path=junit_path,
    )
    return _compared(mode, detect, launched.record)


def _usage_marker(text: str) -> str | None:
    return next(
        (line.strip() for line in text.splitlines() if _VERIFY_USAGE_MARKER_RE.search(line)), None,
    )


def _printed_ids(text: str) -> tuple[str, ...]:
    return tuple(line.strip() for line in text.splitlines() if line.strip())


async def _confirm_numeric(worktree: Path, runner: ConfirmRunner) -> Confirmed | Unconfirmable:
    launched = await runner(
        [RUN_OFFLINE_DEEP_SCRIPT, '--test-threads=1', '--confirm-failed'],
        cwd=worktree,
        output_format=OutputFormat.OPAQUE,
        junit_path=None,
    )
    record = launched.record
    if record.ended_by == EndedBy.SIGNAL:
        return Unconfirmable(UnconfirmableReason.CONFIRM_SIGNAL_ENDED, str(record.returncode))
    marker = _usage_marker(launched.text)
    if marker is not None:
        return Unconfirmable(UnconfirmableReason.USAGE_TEXT, marker)
    if record.returncode == NUMERIC_CONFIRM_IDS_EXIT:
        return Confirmed(ConfirmMode.LEG_NATIVE, _printed_ids(launched.text), (), ())
    if record.returncode == 0:
        return Confirmed(ConfirmMode.LEG_NATIVE, (), (), ())
    return Unconfirmable(UnconfirmableReason.NUMERIC_EXIT, str(record.returncode))


async def _confirm_infra(
    worktree: Path, detect: RunRecord, runner: ConfirmRunner,
) -> Confirmed | Unconfirmable:
    launched = await runner(
        [RUN_ALL_INFRA_SCRIPT, '--scope', INFRA_SCOPE],
        cwd=worktree,
        output_format=OutputFormat.INFRA,
        junit_path=None,
    )
    return _compared(ConfirmMode.INTERSECTED, detect, launched.record)


async def confirm(
    target: ConfirmTarget, detect: RunRecord | None, runner: ConfirmRunner,
) -> Confirmed | Unconfirmable:
    """Confirm a red pass of *target* whose detection left *detect*, running one confirm through *runner*."""
    if detect is None:
        return Unconfirmable(UnconfirmableReason.NO_DETECT_RECORD, '')
    if detect.ended_by == EndedBy.SIGNAL:
        return Unconfirmable(UnconfirmableReason.DETECT_SIGNAL_ENDED, str(detect.returncode))
    if target.leg is not Leg.NUMERIC and not detect.failures:
        return Unconfirmable(UnconfirmableReason.DETECT_RED_NAMED_NO_TEST, str(detect.returncode))
    if target.command is not None:
        return await _confirm_generic(target.command, target.worktree, detect, runner)
    if target.leg is Leg.NUMERIC:
        return await _confirm_numeric(target.worktree, runner)
    return await _confirm_infra(target.worktree, detect, runner)


def legacy_confirmed_ids(
    outcome: Confirmed | Unconfirmable, *, command_name: str, leg: Leg,
) -> list[str]:
    """TRANSITIONAL (deleted by δ2): the id list the lane's existing red path consumes.

    Only a generic command whose confirm exited non-zero naming no test gets the
    command-level ``<command_name>::unconfirmable`` sentinel; every other
    unconfirmable pass confirms nothing, with a warning naming its reason.
    """
    if isinstance(outcome, Confirmed):
        return sorted(outcome.reproduced)
    if leg is Leg.GENERIC and outcome.reason is UnconfirmableReason.CONFIRM_EXITED_NONZERO_NAMING_NO_TEST:
        return [f'{command_name}::unconfirmable']
    logger.warning(
        'offline-lane: %s pass unconfirmable: %s %s', command_name, outcome.reason, outcome.detail,
    )
    return []
