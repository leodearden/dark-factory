"""The merge lane's halt state (PRD ``plans/merge-lane-quality-prd.md`` task θ).

``HaltState`` is driven only through its public surface: which lanes are
halted, who owns each halt, whether the operator halt is up, and how often
it asks the worker to wake. The lane tuple and the resume callback are
injected, exactly as the worker's composition root hands them over.
"""
from __future__ import annotations

import logging

import pytest

from orchestrator.merge_lane.halt import HaltState

LANES = ('high', 'normal')


class ResumeRecorder:
    """The ``on_resume`` callback, counting how often it was asked to wake."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> None:
        self.calls += 1


@pytest.fixture
def resumes() -> ResumeRecorder:
    return ResumeRecorder()


@pytest.fixture
def halt(resumes: ResumeRecorder) -> HaltState:
    return HaltState(LANES, on_resume=resumes)


def halted_lanes(halt: HaltState) -> list[str]:
    return [lane for lane in LANES if halt.is_lane_halted(lane)]


def test_a_fresh_state_halts_nothing(halt: HaltState, resumes: ResumeRecorder) -> None:
    assert halted_lanes(halt) == []
    assert halt.is_wip_halted is False
    assert halt.halt_owner_esc_id is None
    assert halt.is_operator_halted is False
    assert resumes.calls == 0


def test_halting_one_lane_records_its_owner(halt: HaltState) -> None:
    halt.halt_lane('normal', 'wip conflict', owner_esc_id='e1')

    assert halted_lanes(halt) == ['normal']
    assert halt.is_wip_halted is True
    assert halt.lane_owned_by('e1') == 'normal'
    assert halt.is_halt_owner('e1') is True
    assert halt.halt_owner_esc_id == 'e1'
    assert halt.is_operator_halted is False


def test_unhalting_a_lane_clears_halt_and_owner_and_wakes_once(
    halt: HaltState, resumes: ResumeRecorder,
) -> None:
    halt.halt_lane('normal', 'wip conflict', owner_esc_id='e1')

    halt.unhalt_lane('normal', 'resolved')

    assert halted_lanes(halt) == []
    assert halt.lane_owned_by('e1') is None
    assert halt.is_halt_owner('e1') is False
    assert resumes.calls == 1


def test_a_lane_owner_is_never_overwritten_by_a_different_one(halt: HaltState) -> None:
    halt.halt_lane('normal', 'wip conflict')
    halt.set_lane_halt_owner('normal', 'e1')
    halt.set_lane_halt_owner('normal', 'e1')

    with pytest.raises(AssertionError) as refused:
        halt.set_lane_halt_owner('normal', 'e2')

    message = str(refused.value)
    assert "'normal'" in message
    assert "'e1'" in message
    assert "'e2'" in message
    assert halt.lane_owned_by('e1') == 'normal'


def test_the_wip_halt_stops_every_lane_and_takes_one_owner(halt: HaltState) -> None:
    halt.halt_for_wip('wip conflict')

    assert halted_lanes(halt) == list(LANES)
    assert halt.halt_owner_esc_id is None

    halt.set_halt_owner('e2')

    assert halt.halt_owner_esc_id == 'e2'
    assert halt.is_halt_owner('e2') is True
    with pytest.raises(AssertionError, match="already set to 'e2'"):
        halt.set_halt_owner('e2')
    assert halt.unhalt_lanes_owned_by('e2') == list(LANES)


def test_an_owner_needs_a_halt_to_own(halt: HaltState) -> None:
    with pytest.raises(AssertionError, match='no lane is halted'):
        halt.set_halt_owner('e2')


def test_only_the_operator_halt_raises_the_operator_signal(halt: HaltState) -> None:
    halt.halt_for_wip('wip conflict')
    assert halt.is_operator_halted is False

    halt.operator_halt('operator says stop')

    assert halt.is_operator_halted is True
    assert halted_lanes(halt) == list(LANES)
    assert halt.halt_owner_esc_id is None


def test_resume_all_clears_lanes_owners_and_the_operator_halt(
    halt: HaltState, resumes: ResumeRecorder,
) -> None:
    halt.halt_lane('high', 'wip conflict', owner_esc_id='e1')
    halt.operator_halt('operator says stop')

    halt.unhalt_all_lanes('operator resumed')

    assert halted_lanes(halt) == []
    assert halt.lane_owned_by('e1') is None
    assert halt.is_operator_halted is False
    assert halt.is_wip_halted is False
    assert resumes.calls == 1


def test_resuming_by_owner_reports_exactly_the_lanes_it_resumed(
    halt: HaltState, resumes: ResumeRecorder,
) -> None:
    halt.halt_lane('normal', 'wip conflict', owner_esc_id='e1')
    halt.halt_lane('high', 'other conflict', owner_esc_id='e9')

    assert halt.unhalt_lanes_owned_by('e1') == ['normal']
    assert halted_lanes(halt) == ['high']
    assert resumes.calls == 1

    assert halt.unhalt_lanes_owned_by('e1') == []
    assert resumes.calls == 1


def test_shutdown_release_runs_every_lane_but_keeps_owners_and_stays_quiet(
    halt: HaltState, resumes: ResumeRecorder,
) -> None:
    halt.halt_lane('normal', 'wip conflict', owner_esc_id='e1')
    halt.halt_for_wip('wip conflict')

    halt.release_lanes_for_shutdown()

    assert halted_lanes(halt) == []
    assert halt.is_wip_halted is False
    assert resumes.calls == 0


def test_shutdown_release_leaves_a_lane_owner_in_place(halt: HaltState) -> None:
    halt.halt_lane('normal', 'wip conflict', owner_esc_id='e1')

    halt.release_lanes_for_shutdown()

    assert halt.lane_owned_by('e1') == 'normal'


def test_halt_log_lines_are_unchanged(
    halt: HaltState, caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger='orchestrator.merge_lane'):
        halt.halt_lane('normal', 'wip conflict')
        halt.operator_halt('operator says stop')

    warnings = [
        record.getMessage() for record in caplog.records
        if record.name == 'orchestrator.merge_lane' and record.levelno == logging.WARNING
    ]
    assert warnings == [
        "Merge queue lane 'normal' halted: wip conflict",
        'Merge queue operator-halted: operator says stop',
    ]
