"""Tests for escalation.declared_pins — THE declared-pin predicate (task 4377).

PROTECTION AT RESOLUTION TIME, deliberately distinct from
``escalation.pins.classify_pins``'s CLASSIFICATION AT RECOVERY TIME: an OPEN
escalation record is a preservation mechanism for its subject task, so closing
a marked record is a state-changing act on that task.  ``pins.py`` answers
"does this already-open record veto recovery?" and protects nothing from being
CLOSED; this module answers "may a resolver close this record at all?".

Covers:
  step-3: the whole pure surface — the frozen ``PinDeclaration`` value object,
          ``blocking_pin_declarations`` (marker detection, input ordering,
          blank-declarer handling, the ``acknowledged=`` filter, purity), and
          ``format_refusal``'s content contract.

The GATE that consumes this predicate lives at
``escalation/server.py::resolve_issue`` and is covered in tests/test_server.py.
"""

from __future__ import annotations

import dataclasses

import pytest

from escalation.declared_pins import (
    PinDeclaration,
    blocking_pin_declarations,
    format_refusal,
    normalise_declarers,
)
from escalation.models import Escalation


def _esc(
    *,
    id: str = 'esc-3371-2',  # noqa: A002 — mirrors the Escalation attribute name
    pin_declared_by: list[str] | None = None,
    pin_declared_reason: str = '',
) -> Escalation:
    """A real ``escalation.models.Escalation`` — the only implementer."""
    return Escalation(
        id=id,
        task_id='3371',
        agent_role='implementer',
        severity='blocking',
        category='risk_identified',
        summary='s',
        level=1,
        pin_declared_by=list(pin_declared_by or []),
        pin_declared_reason=pin_declared_reason,
    )


# ---------------------------------------------------------------------------
# (a) PinDeclaration — the frozen value object
# ---------------------------------------------------------------------------


class TestPinDeclarationTypeSurface:
    """``PinDeclaration`` is a frozen value object carrying TUPLES, not lists."""

    def test_carries_the_three_documented_fields(self) -> None:
        decl = PinDeclaration(
            escalation_id='esc-3371-2',
            declared_by=('task-3546-second-deviation-notice',),
            reason='mu-gate validation specimen',
        )
        assert decl.escalation_id == 'esc-3371-2'
        assert decl.declared_by == ('task-3546-second-deviation-notice',)
        assert decl.reason == 'mu-gate validation specimen'

    def test_is_frozen(self) -> None:
        decl = PinDeclaration('esc-3371-2', ('a',), 'r')
        with pytest.raises(dataclasses.FrozenInstanceError):
            decl.escalation_id = 'esc-other'  # type: ignore[misc]

    def test_declared_by_is_a_tuple_not_a_list(self) -> None:
        """Tuples (not lists) — same rationale as ``PinReport``: genuinely
        immutable, so no consumer can mutate a bucket in place."""
        (decl,) = blocking_pin_declarations([_esc(pin_declared_by=['a', 'b'])])
        assert isinstance(decl.declared_by, tuple), (
            f'declared_by must be a tuple, got {type(decl.declared_by).__name__}'
        )


# ---------------------------------------------------------------------------
# (b)-(e) blocking_pin_declarations — marker detection
# ---------------------------------------------------------------------------


class TestBlockingPinDeclarations:
    """One entry per marked record the acknowledgement does not cover."""

    def test_empty_input_returns_empty_tuple(self) -> None:
        assert blocking_pin_declarations([]) == ()

    def test_wholly_unmarked_records_return_empty_tuple(self) -> None:
        records = [_esc(id='esc-1-1'), _esc(id='esc-1-2'), _esc(id='esc-1-3')]
        assert blocking_pin_declarations(records) == ()

    def test_returns_the_result_as_a_tuple(self) -> None:
        result = blocking_pin_declarations([_esc(pin_declared_by=['a'])])
        assert isinstance(result, tuple)

    def test_marked_record_yields_one_declaration_with_id_declarers_and_reason(self) -> None:
        records = [
            _esc(
                id='esc-3371-2',
                pin_declared_by=['task-3546-second-deviation-notice', 'esc-3914-1'],
                pin_declared_reason='mu-gate validation specimen; see task 3546',
            )
        ]

        blocked = blocking_pin_declarations(records)

        assert len(blocked) == 1
        assert blocked[0].escalation_id == 'esc-3371-2'
        assert blocked[0].declared_by == (
            'task-3546-second-deviation-notice',
            'esc-3914-1',
        )
        assert blocked[0].reason == 'mu-gate validation specimen; see task 3546'

    def test_declarers_are_reported_in_declaration_order(self) -> None:
        (decl,) = blocking_pin_declarations([_esc(pin_declared_by=['first', 'second', 'third'])])
        assert decl.declared_by == ('first', 'second', 'third')

    def test_mixed_input_returns_one_entry_per_marked_record_in_input_order(self) -> None:
        records = [
            _esc(id='esc-a-1'),
            _esc(id='esc-b-2', pin_declared_by=['gate-b']),
            _esc(id='esc-c-3'),
            _esc(id='esc-d-4', pin_declared_by=['gate-d']),
        ]

        blocked = blocking_pin_declarations(records)

        assert [d.escalation_id for d in blocked] == ['esc-b-2', 'esc-d-4']

    def test_reason_without_a_declarer_is_not_a_marker(self) -> None:
        """The declarer list is the marker; prose alone blocks nothing."""
        records = [_esc(pin_declared_by=[], pin_declared_reason='I think this matters')]
        assert blocking_pin_declarations(records) == ()

    def test_all_blank_declarers_are_treated_as_unmarked(self) -> None:
        """An all-blank marker is not a declaration."""
        records = [_esc(pin_declared_by=['', '   ', '\t\n'])]
        assert blocking_pin_declarations(records) == ()

    def test_blank_declarers_are_stripped_from_a_genuine_declaration(self) -> None:
        (decl,) = blocking_pin_declarations([_esc(pin_declared_by=['', 'real-gate', '  '])])
        assert decl.declared_by == ('real-gate',)

    def test_surrounding_whitespace_is_stripped_from_each_declarer(self) -> None:
        (decl,) = blocking_pin_declarations([_esc(pin_declared_by=['  task-3546  '])])
        assert decl.declared_by == ('task-3546',)

    def test_duplicate_declarers_are_reported_once(self) -> None:
        """The READ side normalises exactly as the WRITE side does.

        ``queue.declare_pin`` de-duplicates, so both docstrings describe
        de-duplication as a contract of the field.  A record hand-edited (or
        written by a future second writer) with duplicates must therefore not
        report them twice in ``declared_pins`` or in the refusal message.
        """
        (decl,) = blocking_pin_declarations(
            [_esc(pin_declared_by=['gate-a', '  gate-a  ', 'gate-b', 'gate-a'])]
        )
        assert decl.declared_by == ('gate-a', 'gate-b')


# ---------------------------------------------------------------------------
# normalise_declarers — THE one normalisation, shared with queue.declare_pin
# ---------------------------------------------------------------------------


class TestNormaliseDeclarers:
    """One helper, so the read side and the write side cannot drift."""

    def test_returns_a_tuple(self) -> None:
        assert normalise_declarers(['gate-a']) == ('gate-a',)

    def test_empty_input_returns_empty_tuple(self) -> None:
        assert normalise_declarers([]) == ()

    def test_strips_surrounding_whitespace(self) -> None:
        assert normalise_declarers(['  gate-a\t']) == ('gate-a',)

    def test_drops_blank_and_whitespace_only_entries(self) -> None:
        assert normalise_declarers(['', '   ', '\t\n', 'gate-a']) == ('gate-a',)

    def test_de_duplicates_preserving_first_occurrence_order(self) -> None:
        assert normalise_declarers(
            ['gate-b', 'gate-a', 'gate-b', '  gate-a  ']
        ) == ('gate-b', 'gate-a')

    def test_accepts_any_iterable(self) -> None:
        assert normalise_declarers(iter(['gate-a', 'gate-a'])) == ('gate-a',)


# ---------------------------------------------------------------------------
# (f) the acknowledged= filter
# ---------------------------------------------------------------------------


class TestAcknowledgedFilter:
    """Acknowledgement is by ID and must name every blocked record."""

    def _cluster(self) -> list[Escalation]:
        return [
            _esc(id='esc-a-1', pin_declared_by=['gate-a'], pin_declared_reason='ra'),
            _esc(id='esc-b-2', pin_declared_by=['gate-b'], pin_declared_reason='rb'),
        ]

    def test_acknowledging_every_blocked_id_returns_empty(self) -> None:
        assert blocking_pin_declarations(self._cluster(), acknowledged=['esc-a-1', 'esc-b-2']) == ()

    def test_acknowledging_a_strict_subset_returns_only_the_remainder(self) -> None:
        blocked = blocking_pin_declarations(self._cluster(), acknowledged=['esc-a-1'])

        assert [d.escalation_id for d in blocked] == ['esc-b-2']

    def test_acknowledging_an_unblocked_id_is_a_harmless_no_op(self) -> None:
        blocked = blocking_pin_declarations(
            self._cluster(), acknowledged=['esc-not-in-this-cluster']
        )

        assert [d.escalation_id for d in blocked] == ['esc-a-1', 'esc-b-2']

    def test_acknowledged_defaults_to_nothing_acknowledged(self) -> None:
        assert len(blocking_pin_declarations(self._cluster())) == 2

    def test_acknowledged_accepts_any_collection(self) -> None:
        """A set works as well as a list — the parameter is a Collection."""
        assert blocking_pin_declarations(self._cluster(), acknowledged={'esc-a-1', 'esc-b-2'}) == ()


# ---------------------------------------------------------------------------
# (g) purity — no I/O, no mutation of inputs
# ---------------------------------------------------------------------------


class TestPurity:
    """PURE by construction: the caller binds the store read and passes rows in."""

    def test_inputs_are_not_mutated(self) -> None:
        records = [
            _esc(id='esc-a-1', pin_declared_by=['  gate-a  ', ''], pin_declared_reason='ra'),
            _esc(id='esc-b-2'),
        ]

        blocking_pin_declarations(records, acknowledged=['esc-b-2'])

        assert records[0].pin_declared_by == ['  gate-a  ', ''], (
            'the raw declarer list must be left exactly as passed in'
        )
        assert records[0].pin_declared_reason == 'ra'
        assert records[1].pin_declared_by == []

    def test_returned_declarers_do_not_alias_the_input_list(self) -> None:
        record = _esc(pin_declared_by=['gate-a'])

        (decl,) = blocking_pin_declarations([record])

        assert decl.declared_by is not record.pin_declared_by
        record.pin_declared_by.append('gate-b')
        assert decl.declared_by == ('gate-a',), (
            'the declaration must be a snapshot, not a live view of the record'
        )

    def test_calling_twice_returns_equal_results(self) -> None:
        """No hidden state: the predicate is a function of its arguments alone."""
        records = [_esc(pin_declared_by=['gate-a'])]

        first = blocking_pin_declarations(records)
        second = blocking_pin_declarations(records)

        assert first == second


# ---------------------------------------------------------------------------
# (h) format_refusal — the human half of the error
# ---------------------------------------------------------------------------


class TestFormatRefusal:
    """The message alone tells a closer WHAT declared the pin and how to proceed."""

    DECLS = (
        PinDeclaration(
            escalation_id='esc-3371-2',
            declared_by=('task-3546-second-deviation-notice', 'esc-3914-1'),
            reason='mu-gate validation specimen — the evidence base',
        ),
        PinDeclaration(
            escalation_id='esc-3105-3',
            declared_by=('operator-gate-3105',),
            reason='last hold on task 3105',
        ),
    )

    def test_names_every_blocked_escalation_id(self) -> None:
        message = format_refusal(self.DECLS)
        assert 'esc-3371-2' in message
        assert 'esc-3105-3' in message

    def test_names_every_declarer_string(self) -> None:
        message = format_refusal(self.DECLS)
        assert 'task-3546-second-deviation-notice' in message
        assert 'esc-3914-1' in message
        assert 'operator-gate-3105' in message

    def test_names_every_reason(self) -> None:
        message = format_refusal(self.DECLS)
        assert 'mu-gate validation specimen — the evidence base' in message
        assert 'last hold on task 3105' in message

    def test_names_the_acknowledgement_parameter_as_the_override(self) -> None:
        assert 'acknowledge_declared_pins' in format_refusal(self.DECLS)

    def test_single_declaration_message_is_well_formed(self) -> None:
        message = format_refusal(self.DECLS[:1])
        assert 'esc-3371-2' in message
        assert 'esc-3105-3' not in message
        assert 'acknowledge_declared_pins' in message

    def test_a_declaration_with_no_reason_still_names_id_and_declarers(self) -> None:
        message = format_refusal(
            [PinDeclaration('esc-9-9', ('some-gate',), '')]
        )
        assert 'esc-9-9' in message
        assert 'some-gate' in message
