"""The incumbent's measured LLM spend: production write telemetry and the controls' unit cost."""

import re
from datetime import datetime

import pytest

from fused_memory.arm_harness.incumbent_cost import (
    LlmWriteOperation,
    LlmWriteTelemetry,
    TelemetryAccountingError,
    TelemetryWindow,
    TelemetryWindowError,
    select_llm_writes,
)

PRE_START = '2026-10-05T10:59:00.000001+00:00'
FIRST_TOKENED = '2026-10-05T11:23:27.123456+00:00'
LATER = '2026-10-06T08:00:00+00:00'
UNTIL_TEXT = '2026-10-08T00:00:00+00:00'
UNTIL = datetime.fromisoformat(UNTIL_TEXT)
_UNTOKENED = (None, None, None)


def telemetry_row(
    created_at: str,
    *,
    backend: str = 'graphiti',
    operation: str | None = 'add_memory',
    project_id: str | None = 'dark_factory',
    success: int = 1,
    tokens: tuple[int | None, int | None, int | None] = (1000, 100, 9),
) -> dict[str, object]:
    """One object shaped exactly like a ``telemetry_query.py`` output line."""
    input_tokens, output_tokens, llm_calls = tokens
    total_tokens = (
        None if input_tokens is None or output_tokens is None else input_tokens + output_tokens
    )
    return {
        'created_at': created_at,
        'operation': operation,
        'project_id': project_id,
        'backend': backend,
        'success': success,
        'duration_ms': 1234.5,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'total_tokens': total_tokens,
        'llm_calls': llm_calls,
    }


def untokened_row(created_at: str, **overrides) -> dict[str, object]:
    return telemetry_row(created_at, tokens=_UNTOKENED, **overrides)


def at(text: str) -> datetime:
    return datetime.fromisoformat(text)


def write_at(text: str, **overrides) -> LlmWriteTelemetry:
    return LlmWriteTelemetry.from_telemetry_row(telemetry_row(text, **overrides))


def test_only_graphiti_add_memory_and_add_episode_rows_are_llm_writes():
    rows = [
        untokened_row(PRE_START),
        telemetry_row(FIRST_TOKENED),
        telemetry_row(LATER, operation='add_episode'),
        untokened_row('2026-10-06T09:00:00+00:00', backend='mem0'),
        untokened_row('2026-10-06T10:00:00+00:00', operation='update_edge'),
        untokened_row(
            '2026-10-06T11:00:00+00:00',
            backend='sqlite_task_backend',
            operation='update_task',
            project_id='solar_challenge',
        ),
        untokened_row('2026-10-06T12:00:00+00:00', backend='graphiti', operation=None),
    ]

    window = select_llm_writes(rows, until=UNTIL)

    assert [(w.created_at, w.operation) for w in window.writes] == [
        (at(FIRST_TOKENED), LlmWriteOperation.ADD_MEMORY),
        (at(LATER), LlmWriteOperation.ADD_EPISODE),
    ]


def test_window_runs_from_the_first_token_bearing_write_up_to_but_excluding_until():
    rows = [
        untokened_row('2026-10-08T00:00:01+00:00'),
        telemetry_row(UNTIL_TEXT),
        telemetry_row(LATER, project_id='reify'),
        telemetry_row(FIRST_TOKENED),
        untokened_row(PRE_START),
        untokened_row('2026-10-01T00:00:00+00:00'),
    ]

    window = select_llm_writes(rows, until=UNTIL)

    assert window.start == at(FIRST_TOKENED)
    assert window.end == UNTIL
    assert [w.created_at for w in window.writes] == [at(FIRST_TOKENED), at(LATER)]
    assert [w.project_id for w in window.writes] == ['dark_factory', 'reify']


@pytest.mark.parametrize(
    'pre_start_rows',
    [
        [],
        [untokened_row(PRE_START, backend='mem0')],
        [untokened_row(PRE_START, operation='update_edge')],
    ],
)
def test_a_dump_that_does_not_reach_back_past_the_telemetry_start_is_refused(pre_start_rows):
    rows = [*pre_start_rows, telemetry_row(FIRST_TOKENED), telemetry_row(LATER)]

    with pytest.raises(TelemetryWindowError, match=re.escape(FIRST_TOKENED)):
        select_llm_writes(rows, until=UNTIL)


def test_an_untokened_llm_write_inside_the_window_is_unknown_not_zero():
    rows = [
        untokened_row(PRE_START),
        telemetry_row(FIRST_TOKENED),
        untokened_row('2026-10-06T01:00:00+00:00'),
        telemetry_row(LATER),
        untokened_row('2026-10-07T01:00:00+00:00', operation='add_episode'),
    ]

    with pytest.raises(TelemetryAccountingError) as caught:
        select_llm_writes(rows, until=UNTIL)

    message = str(caught.value)
    assert '2 ' in message
    assert '2026-10-06T01:00:00+00:00' in message
    assert 'unknown, not zero' in message


def test_a_partially_tokened_llm_write_inside_the_window_is_refused():
    rows = [
        untokened_row(PRE_START),
        telemetry_row(FIRST_TOKENED),
        telemetry_row(LATER, tokens=(1000, 100, None)),
    ]

    with pytest.raises(TelemetryAccountingError, match=re.escape(LATER)):
        select_llm_writes(rows, until=UNTIL)


def test_untokened_writes_after_until_are_outside_the_accounting():
    rows = [
        untokened_row(PRE_START),
        telemetry_row(FIRST_TOKENED),
        untokened_row('2026-10-08T03:00:00+00:00'),
    ]

    assert len(select_llm_writes(rows, until=UNTIL).writes) == 1


@pytest.mark.parametrize(
    'rows',
    [
        [untokened_row(PRE_START), untokened_row(LATER)],
        [untokened_row(PRE_START), telemetry_row(UNTIL_TEXT)],
        [],
    ],
)
def test_no_token_bearing_llm_write_before_until_is_refused(rows):
    with pytest.raises(TelemetryWindowError, match=re.escape(UNTIL_TEXT)):
        select_llm_writes(rows, until=UNTIL)


def test_a_naive_until_is_refused():
    rows = [untokened_row(PRE_START), telemetry_row(FIRST_TOKENED)]

    with pytest.raises(TelemetryWindowError, match='until'):
        select_llm_writes(rows, until=datetime(2026, 10, 8))


def test_a_naive_created_at_is_refused_at_the_boundary():
    with pytest.raises(ValueError, match=re.escape('2026-10-05T12:00:00')):
        LlmWriteTelemetry.from_telemetry_row(telemetry_row('2026-10-05T12:00:00'))


def test_a_naive_created_at_is_refused_by_selection():
    rows = [untokened_row('2026-10-05T10:00:00'), telemetry_row(FIRST_TOKENED)]

    with pytest.raises(ValueError, match=re.escape('2026-10-05T10:00:00')):
        select_llm_writes(rows, until=UNTIL)


def test_a_total_that_is_not_input_plus_output_is_refused_naming_the_row():
    row = telemetry_row(LATER) | {'total_tokens': 1}

    with pytest.raises(ValueError, match=re.escape(LATER)) as caught:
        LlmWriteTelemetry.from_telemetry_row(row)

    assert 'total_tokens' in str(caught.value)


@pytest.mark.parametrize(('raw', 'parsed'), [(1, True), (0, False)])
def test_success_parses_to_a_bool(raw, parsed):
    assert write_at(LATER, success=raw).success is parsed


def test_a_parsed_write_keeps_only_what_costing_needs():
    write = write_at(LATER, tokens=(1650, 95, 7))

    assert set(LlmWriteTelemetry.model_fields) == {
        'created_at',
        'operation',
        'project_id',
        'success',
        'input_tokens',
        'output_tokens',
        'llm_calls',
    }
    assert write.created_at == at(LATER)
    assert write.created_at.tzinfo is not None
    assert write.operation is LlmWriteOperation.ADD_MEMORY
    assert (write.input_tokens, write.output_tokens, write.llm_calls) == (1650, 95, 7)


def test_a_valid_window_constructs():
    writes = (write_at(FIRST_TOKENED), write_at(LATER))

    window = TelemetryWindow(start=at(FIRST_TOKENED), end=UNTIL, writes=writes)

    assert window.writes == writes


@pytest.mark.parametrize(
    ('start', 'end', 'stamps', 'invariant'),
    [
        (FIRST_TOKENED, UNTIL_TEXT, (LATER, FIRST_TOKENED), 'sorted'),
        (FIRST_TOKENED, LATER, (FIRST_TOKENED, LATER), 'end'),
        (FIRST_TOKENED, UNTIL_TEXT, (FIRST_TOKENED, '2026-10-09T00:00:00+00:00'), 'end'),
        (UNTIL_TEXT, FIRST_TOKENED, (), 'end'),
        (FIRST_TOKENED, UNTIL_TEXT, (), 'empty'),
        (PRE_START, UNTIL_TEXT, (FIRST_TOKENED, LATER), 'first write'),
    ],
)
def test_window_construction_enforces_its_invariant(start, end, stamps, invariant):
    with pytest.raises(ValueError, match=invariant):
        TelemetryWindow(
            start=at(start), end=at(end), writes=tuple(write_at(stamp) for stamp in stamps)
        )
