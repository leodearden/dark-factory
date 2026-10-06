"""The scratch-name guard: an allow-list every arm-harness write point calls."""

from enum import Enum

import pytest

from arm_harness._fakes import PROTECTED_GRAPHS
from fused_memory.arm_harness.scratch_guard import (
    SCRATCH_NAME_PATTERN,
    GuardCheckpoint,
    ScratchGuardError,
    require_scratch_name,
)

MALFORMED_SCRATCH_NAMES = [
    'evalmem_',
    'evalmem_Bad',
    'EVALMEM_x',
    'evalmem-x',
    'xevalmem_a',
    'evalmem_a b',
    'evalmem_a\n',
    '',
]


@pytest.mark.parametrize('name', ['evalmem_ctrl_a', 'evalmem_x1', 'evalmem_a_b_c'])
def test_accepts_scratch_names_and_returns_them_unchanged(name):
    assert require_scratch_name(name, checkpoint=GuardCheckpoint.REPLAY) == name


@pytest.mark.parametrize('name', [*PROTECTED_GRAPHS, *MALFORMED_SCRATCH_NAMES, None, 3])
def test_rejects_everything_outside_the_allow_list(name):
    with pytest.raises(ScratchGuardError):
        require_scratch_name(name, checkpoint=GuardCheckpoint.TEARDOWN_GRAPH)


def test_trailing_newline_is_rejected_because_the_match_is_full():
    assert SCRATCH_NAME_PATTERN.match('evalmem_a\n') is not None
    assert SCRATCH_NAME_PATTERN.fullmatch('evalmem_a\n') is None
    with pytest.raises(ScratchGuardError):
        require_scratch_name('evalmem_a\n', checkpoint=GuardCheckpoint.ARM_SPEC)


def test_error_is_not_a_value_error_so_pydantic_propagates_it_raw():
    assert not issubclass(ScratchGuardError, ValueError)


def test_error_carries_structured_name_and_checkpoint():
    with pytest.raises(ScratchGuardError) as caught:
        require_scratch_name('dark_factory', checkpoint=GuardCheckpoint.INDEX_BUILD)

    error = caught.value
    assert error.name == 'dark_factory'
    assert error.checkpoint is GuardCheckpoint.INDEX_BUILD
    message = str(error)
    assert '^evalmem_[a-z0-9_]+$' in message
    assert repr('dark_factory') in message
    assert GuardCheckpoint.INDEX_BUILD.value in message


def test_non_string_name_is_reported_by_repr():
    with pytest.raises(ScratchGuardError) as caught:
        require_scratch_name(None, checkpoint=GuardCheckpoint.TOPOLOGY_READ)

    assert caught.value.name is None
    assert 'None' in str(caught.value)


def test_checkpoints_are_a_closed_enum():
    assert issubclass(GuardCheckpoint, Enum)
    assert {member.name for member in GuardCheckpoint} == {
        'ARM_SPEC',
        'REPLAY',
        'INDEX_BUILD',
        'INDEX_PROBE',
        'TOPOLOGY_READ',
        'TEARDOWN_GRAPH',
        'TEARDOWN_COLLECTION',
    }
