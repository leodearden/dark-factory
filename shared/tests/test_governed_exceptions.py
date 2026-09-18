"""Tests for shared.governed_exceptions — INV-12's disposition vocabulary.

PRD ``plans/inv12-exceptions-owned-or-ratified-prd.md``, Contract section
**Vocabulary** and decision **D2** ("three legal states"): ``Debt(owner)`` with
an owner that is a ``TaskRef`` or a ``TicketRef``, and ``Policy(ratified)``
naming a row of the ratification table.  ``Disposition`` is the union of the
two.

THE BOUNDARY THESE TESTS PIN.  The constructors validate **shape only**.
Whether a ``Policy`` id is actually a row in
``docs/legibility/exception-ratifications.yaml`` is D3's closed-world check,
which the register performs; whether a ``TaskRef`` names a live task is the
sweep's.  Neither belongs to a dataclass, and a test that expected either here
would be pinning the wrong module's job.

Modelled on ``shared/tests/test_merge_state.py``, including its ``_SRC``
expression: the structural guards below read the module source from the LOCAL
``shared/src`` tree rather than an installed copy.

TDD pair 1: the Disposition vocabulary (GREEN on impl step-2).
"""
from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

from shared.governed_exceptions import (
    Debt,
    Disposition,
    MalformedDisposition,
    Policy,
    TaskRef,
    TicketRef,
)

# Same src-root expression as shared/tests/conftest.py and
# test_pure_stdlib_leaves.py — read the LOCAL tree, never an installed copy.
_SRC = Path(__file__).resolve().parent.parent / 'src'
_MODULE_SOURCE = _SRC / 'shared' / 'governed_exceptions.py'

# A real ticket id, shaped as fused-memory/src/fused_memory/middleware/
# ticket_store.py::_new_ticket_id mints them: the literal 'tkt_' prefix plus
# Crockford base32 (0-9 A-Z minus I, L, O, U).  The LENGTH is deliberately not
# pinned by the validator, so this constant exercises the shape, not a count.
TICKET_ID = 'tkt_0RTCC80EM92A7WD08D6RF6ZZPY'


class TestTaskRef:
    """A task id is a positive int, and ``bool`` is not one."""

    def test_round_trips_its_id(self):
        assert TaskRef(5601).id == 5601

    def test_is_frozen(self):
        ref = TaskRef(5601)
        with pytest.raises(dataclasses.FrozenInstanceError):
            ref.id = 5602  # type: ignore[misc]

    def test_is_equal_and_hashable_by_value(self):
        assert TaskRef(5601) == TaskRef(5601)
        assert TaskRef(5601) != TaskRef(5602)
        assert len({TaskRef(5601), TaskRef(5601), TaskRef(5602)}) == 2

    @pytest.mark.parametrize('bad', [0, -1, True, '5601'])
    def test_rejects_a_bad_shape_naming_the_value(self, bad):
        """``True`` is in this table deliberately: ``bool`` is an ``int`` subclass.

        ``isinstance(True, int)`` is true, so an ``isinstance`` guard would
        admit ``TaskRef(True)`` and render it as task 1.  The check must be
        ``type(id) is int``.
        """
        with pytest.raises(MalformedDisposition) as excinfo:
            TaskRef(bad)
        assert repr(bad) in str(excinfo.value)


class TestTicketRef:
    """A ticket id is the literal ``tkt_`` prefix plus Crockford base32."""

    def test_round_trips_its_id(self):
        assert TicketRef(TICKET_ID).id == TICKET_ID

    def test_is_frozen_and_equal_by_value(self):
        assert TicketRef(TICKET_ID) == TicketRef(TICKET_ID)
        with pytest.raises(dataclasses.FrozenInstanceError):
            TicketRef(TICKET_ID).id = 'tkt_OTHER'  # type: ignore[misc]

    def test_does_not_pin_the_minted_length(self):
        """Shape is checked; the 26-character body of a minted id is not.

        ``ticket_store.py::_new_ticket_id`` happens to mint 26 characters
        today.  Pinning that here would make a future id-length change read as
        a vocabulary violation at every disposition site in the repo.
        """
        assert TicketRef('tkt_A').id == 'tkt_A'

    @pytest.mark.parametrize(
        'bad',
        [
            '5601',  # no prefix at all
            'tkt_',  # prefix with an empty body
            'tkt_lowercase',  # Crockford base32 is upper-case
            'TKT_ABC',  # the prefix is literal and lower-case
            'tkt_ABCI',  # I, L, O and U are excluded from the alphabet
            'tkt_ABC-DEF',  # punctuation is not in the alphabet
            5601,  # not a str
        ],
    )
    def test_rejects_a_bad_shape_naming_the_value(self, bad):
        with pytest.raises(MalformedDisposition) as excinfo:
            TicketRef(bad)
        assert repr(bad) in str(excinfo.value)


class TestPolicy:
    """A policy names a ratification row id, which is kebab-case."""

    def test_round_trips_its_id(self):
        assert Policy('inv12-day-one-test-doubles').ratified == 'inv12-day-one-test-doubles'

    def test_is_frozen_and_equal_by_value(self):
        assert Policy('inv12-x') == Policy('inv12-x')
        with pytest.raises(dataclasses.FrozenInstanceError):
            Policy('inv12-x').ratified = 'inv12-y'  # type: ignore[misc]

    @pytest.mark.parametrize(
        'bad',
        ['', 'Not-Kebab', 'has_underscore', 'trailing-', '-leading', 'double--dash', 17],
    )
    def test_rejects_a_bad_shape_naming_the_value(self, bad):
        with pytest.raises(MalformedDisposition) as excinfo:
            Policy(bad)
        assert repr(bad) in str(excinfo.value)

    def test_does_not_check_the_row_exists(self):
        """D3's closed world is the register's check, not the dataclass's.

        A kebab-case id that names no row in the ratification table is a
        VIOLATION the register reports — but it is well-formed, so constructing
        it must succeed.  If this ever raised, an agent could not even write
        down the policy the operator is about to ratify.
        """
        assert Policy('a-row-that-does-not-exist-yet').ratified


class TestDebt:
    """Debt carries an owner, and the owner is one of the two ref types."""

    @pytest.mark.parametrize('owner', [TaskRef(5601), TicketRef(TICKET_ID)])
    def test_constructs_from_either_ref(self, owner):
        assert Debt(owner).owner == owner

    def test_is_frozen_and_equal_by_value(self):
        assert Debt(TaskRef(5601)) == Debt(TaskRef(5601))
        with pytest.raises(dataclasses.FrozenInstanceError):
            Debt(TaskRef(5601)).owner = TaskRef(5602)  # type: ignore[misc]

    @pytest.mark.parametrize('bad', ['task 5601', 5601, Policy('x'), None])
    def test_rejects_an_owner_that_is_not_a_ref(self, bad):
        """A ``Policy`` is a sibling disposition, never an owner of debt."""
        with pytest.raises(MalformedDisposition) as excinfo:
            Debt(bad)
        assert repr(bad) in str(excinfo.value)


class TestDispositionUnion:
    """``Disposition`` is exactly ``Debt | Policy``."""

    @pytest.mark.parametrize('value', [Debt(TaskRef(5601)), Debt(TicketRef(TICKET_ID)), Policy('x')])
    def test_accepts_every_legal_state(self, value):
        assert isinstance(value, Disposition)

    @pytest.mark.parametrize('value', ['debt: task 5601', TaskRef(5601), TicketRef(TICKET_ID), None])
    def test_rejects_anything_else(self, value):
        """A bare ref is NOT a disposition — D2's states are Debt and Policy."""
        assert not isinstance(value, Disposition)


class TestModuleStructure:
    """D4: production modules gain no import-time raise, so nothing runs here."""

    def test_module_body_has_no_bare_expression_statement(self):
        """Only imports, constants, the exception and the dataclasses.

        D4's guarantee is that "a mis-edited disposition fails a test, never an
        import": every consumer of this vocabulary is reached through a bare
        ``python3`` checker or a test module, so an import-time side effect
        here would surface as an instrument failure far from its cause.  A
        module-level *call* is the shape that would do it, so the guard is that
        the body carries no bare expression statement other than the docstring
        and the attribute docstrings that follow the constants.
        """
        body = ast.parse(_MODULE_SOURCE.read_text(encoding='utf-8')).body
        offenders = [
            ast.unparse(node)
            for node in body
            if isinstance(node, ast.Expr)
            and not (isinstance(node.value, ast.Constant) and isinstance(node.value.value, str))
        ]
        assert offenders == [], (
            'shared/src/shared/governed_exceptions.py runs statement(s) at import time:\n  '
            + '\n  '.join(offenders)
            + '\nPRD D4 requires that importing this module define things and do nothing else.'
        )

    def test_malformed_disposition_is_not_a_value_error(self):
        """Design decision: no shared base, and no ValueError ancestry.

        The three faults this package raises map to different exit codes and
        different operator actions.  A broad ``except ValueError`` anywhere in a
        consumer must never be able to swallow one of them.
        """
        assert not issubclass(MalformedDisposition, ValueError)
