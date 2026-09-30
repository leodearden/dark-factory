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
TDD pair 2: the inline marker parser + INLINE_MARKER_FORMS (GREEN on impl step-4).
TDD pair 3: GovernedList + the accepting half of governed_exceptions (GREEN on impl step-6).
TDD pair 4: every rejection of governed_exceptions, incl. scenario 12 (GREEN on impl step-8).
"""
from __future__ import annotations

import ast
import dataclasses
import importlib.util
from pathlib import Path

import pytest

from shared.governed_exceptions import (
    DECLARATION_FORMS,
    INLINE_MARKER_FORMS,
    Debt,
    Disposition,
    MalformedDeclaration,
    MalformedDisposition,
    Policy,
    TaskRef,
    TicketRef,
    UndisposedException,
    governed_exceptions,
    parse_disposition_marker,
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
            'tkt_ABC\n',  # a trailing newline is not part of a ticket id
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
        [
            '',
            'Not-Kebab',
            'has_underscore',
            'trailing-',
            '-leading',
            'double--dash',
            # A TRAILING NEWLINE, which an anchored `match` would have accepted:
            # Python's `$` matches immediately before one.  The id would then
            # silently fail to match the ratification row it names, and D3's
            # closed-world check would report a row that plainly exists as absent.
            'inv12-x\n',
            17,
        ],
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


class TestParseDispositionMarkerAccepts:
    """D6's grammar, parsed out of the whole COMMENT token as tokenize yields it."""

    @pytest.mark.parametrize(
        ('comment', 'expected'),
        [
            # The three examples printed verbatim in PRD D6.
            ('# type: ignore[attr-defined]  # debt: task 5601', Debt(TaskRef(5601))),
            (f'# noqa: E402  # debt: ticket {TICKET_ID}', Debt(TicketRef(TICKET_ID))),
            (
                '# pyright: ignore[reportArgumentType]  # ratified: inv12-day-one-test-doubles',
                Policy('inv12-day-one-test-doubles'),
            ),
        ],
    )
    def test_the_prd_examples_parse_to_their_values(self, comment, expected):
        assert parse_disposition_marker(comment) == expected

    def test_a_marker_with_no_suppression_before_it_still_parses(self):
        """PINNED DELIBERATELY: ``x = 1  # debt: task 5601`` is not this function's violation.

        D6 names two distinct violations — a marker that does not parse, and a
        marker sitting on a line with no suppression.  Only the first is a
        property of the disposition grammar.  Detecting the second needs the
        scanner's kind table (``type: ignore``, ``noqa``, ``pyright: ignore``,
        ``pragma: no cover``, ``nosec``) and D8's consumer model, and putting a
        copy of that table here would give it two homes.  So this parses, and
        the scanner is what reports it.
        """
        assert parse_disposition_marker('# debt: task 5601') == Debt(TaskRef(5601))

    @pytest.mark.parametrize(
        'comment',
        [
            '# debt: task 5601',
            '  # debt: task 5601',
            '# debt: task 5601  ',
            '\t# debt: task 5601\t',
        ],
    )
    def test_is_unaffected_by_whitespace_around_the_token(self, comment):
        """A tokenize COMMENT token carries whatever spacing the author wrote."""
        original = comment
        assert parse_disposition_marker(comment) == Debt(TaskRef(5601))
        assert comment == original


class TestParseDispositionMarkerReturnsNone:
    """Absent marker is None, never a raise — the two-outcome contract's first half."""

    @pytest.mark.parametrize(
        'comment',
        [
            '# type: ignore[attr-defined]',
            '# noqa: E402',
            '',
            '# a plain comment',
            '# see task 5601',
            '# ratification pending',  # 'ratified' does not appear as a keyword
        ],
    )
    def test_no_keyword_means_no_marker(self, comment):
        assert parse_disposition_marker(comment) is None

    def test_an_uppercase_keyword_is_not_a_keyword(self):
        """PINNED CHOICE: ``# DEBT: task 5601`` returns None, it does not raise.

        The grammar D6 publishes is lower-case, and the keyword probe is
        case-sensitive, so an upper-case spelling is simply not a marker.
        Raising instead would make every comment that happens to start with
        the word DEBT an instrument failure.  The cost of this choice is that a
        shouted marker is silently not a disposition — which the scanner still
        catches, because the suppression it sits beside remains undisposed.
        """
        assert parse_disposition_marker('# DEBT: task 5601') is None


class TestParseDispositionMarkerRejects:
    """Marker present and unparseable is a raise — the contract's second half."""

    @pytest.mark.parametrize(
        'comment',
        [
            '# debt: soon',  # D6's own named case: an owner that is not a ref
            '# debt:',
            '# debt: task',
            '# debt: task abc',
            '# debt: task 0',
            '# debt: task 007',  # a decimal numeral carries no leading zeros
            '# debt: ticket 5601',
            '# debt: ticket tkt_lowercase',
            '# ratified:',
            '# ratified: Not_Kebab',
            '# debt: task 5601 and more',  # trailing text after the disposition
            '# type: ignore  # debt: task 5601 (see also 5602)',
        ],
    )
    def test_names_the_comment_and_publishes_every_accepted_form(self, comment):
        with pytest.raises(MalformedDisposition) as excinfo:
            parse_disposition_marker(comment)
        message = str(excinfo.value)
        assert repr(comment) in message
        for form in INLINE_MARKER_FORMS:
            assert form in message

    def test_two_dispositions_in_one_comment_is_a_violation(self):
        """"One disposition covers every suppression in that comment" means exactly one.

        Two markers cannot be reconciled without inventing a precedence rule
        nobody has ruled on, so the parser refuses rather than silently
        honouring the first.
        """
        comment = '# debt: task 5601  # ratified: inv12-day-one-test-doubles'
        with pytest.raises(MalformedDisposition) as excinfo:
            parse_disposition_marker(comment)
        assert repr(comment) in str(excinfo.value)


class TestInlineMarkerFormsAreTheOneGrammar:
    """The published forms and the implemented grammar cannot drift apart."""

    #: Placeholder -> a concrete legal value.  Keyed by the placeholder text as
    #: it appears in INLINE_MARKER_FORMS, so a form that grows a NEW placeholder
    #: fails ``test_every_placeholder_has_a_substitution`` below rather than
    #: silently skipping the round trip.
    SUBSTITUTIONS = {
        '<task id>': '5601',
        'tkt_<ticket id>': TICKET_ID,
        '<ratification row id>': 'inv12-day-one-test-doubles',
    }

    #: Which Disposition type each keyword must produce.
    KEYWORD_TYPES = {'debt:': Debt, 'ratified:': Policy}

    @staticmethod
    def _fill(form):
        for placeholder, value in TestInlineMarkerFormsAreTheOneGrammar.SUBSTITUTIONS.items():
            form = form.replace(placeholder, value)
        return form

    def test_is_a_non_empty_tuple_of_str(self):
        assert isinstance(INLINE_MARKER_FORMS, tuple)
        assert INLINE_MARKER_FORMS
        assert all(isinstance(form, str) for form in INLINE_MARKER_FORMS)

    @pytest.mark.parametrize('form', INLINE_MARKER_FORMS)
    def test_every_placeholder_has_a_substitution(self, form):
        filled = self._fill(form)
        assert '<' not in filled and '>' not in filled, (
            f'INLINE_MARKER_FORMS entry {form!r} carries a placeholder this test cannot '
            'fill. Add it to SUBSTITUTIONS so the round-trip below actually exercises '
            'the new form.'
        )

    @pytest.mark.parametrize('form', INLINE_MARKER_FORMS)
    def test_every_published_form_actually_parses(self, form):
        """THE ANTI-DRIFT GUARANTEE, made behavioural rather than a wording pin.

        What must not be retyped is the GRAMMAR.  Presentation legitimately
        differs between an exception message, a CLI rejection line and the
        implementer-facing prompt block, so the SPOT is a tuple of forms each
        consumer renders itself — and the way that tuple is kept honest is
        this: every form it publishes is filled in and fed to the parser, and
        the value that comes back must be the type the form's keyword promises.
        A form nobody implemented, or an implementation nobody published, turns
        this red.
        """
        filled = self._fill(form)
        keywords = [kw for kw in self.KEYWORD_TYPES if kw in form]
        assert len(keywords) == 1, f'{form!r} must carry exactly one keyword, found {keywords}'
        assert type(parse_disposition_marker(filled)) is self.KEYWORD_TYPES[keywords[0]]


# A declaration the register will really see, used throughout the declaration
# tests so the shapes below are the shapes D4 puts in a package's test tree.
LIST_ID = 'orchestrator.tests.timeout_marker_grandfathered'
RULE = 'every grandfathered timeout marker is owned or ratified'


def _keys(count, prefix='key'):
    return [f'{prefix}-{index:03d}' for index in range(count)]


class TestGovernedListAccepts:
    """The declaration shapes D4 and the PRD's scenario 15 actually produce."""

    def test_a_fully_overridden_list(self):
        declaration = governed_exceptions(
            LIST_ID,
            RULE,
            ['a', 'b'],
            dispositions={'a': Debt(TaskRef(5149)), 'b': Policy('inv12-x')},
        )
        assert declaration.list_id == LIST_ID
        assert declaration.rule == RULE
        assert declaration.keys == ('a', 'b')
        assert dict(declaration.overrides) == {'a': Debt(TaskRef(5149)), 'b': Policy('inv12-x')}
        assert declaration.default is None
        assert declaration.default_covers is None

    def test_a_defaulted_list(self):
        """61 keys, no overrides, one default that covers all of them."""
        declaration = governed_exceptions(
            LIST_ID, RULE, _keys(61), default=Debt(TaskRef(5149)), default_covers=61
        )
        assert len(declaration.keys) == 61
        assert declaration.overrides == {}

    def test_a_mixed_list_counts_only_the_keys_without_an_override(self):
        """default_covers=59, not 61: the two overridden keys are not the default's."""
        declaration = governed_exceptions(
            LIST_ID,
            RULE,
            _keys(61),
            default=Debt(TaskRef(5149)),
            default_covers=59,
            dispositions={'key-000': Policy('inv12-x'), 'key-001': Debt(TaskRef(5602))},
        )
        assert declaration.default_covers == 59
        assert len(declaration.overrides) == 2

    def test_an_empty_list(self):
        """A governed list that is currently empty is legal and disposes nothing."""
        declaration = governed_exceptions(LIST_ID, RULE, [])
        assert declaration.keys == ()
        assert declaration.overrides == {}

    def test_a_baseline_unit_is_a_declaration_with_exactly_one_key(self):
        """The PRD's convention, and it needs no special code path.

        D2 makes each machine-generated baseline ONE governed entry carrying a
        disposition of its own.  That is spelled as an ordinary declaration
        whose single key is the baseline's repo-relative path — so nothing in
        this module knows what a baseline is, and the report counts it like
        any other entry.
        """
        path = 'scripts/inline_suppression_baseline.json'
        declaration = governed_exceptions(
            'scripts.inline_suppression_baseline',
            'the grandfathered inline-suppression baseline, shrink-only',
            [path],
            dispositions={path: Policy('inv12-day-one-test-doubles')},
        )
        assert declaration.keys == (path,)
        assert declaration.disposition_for(path) == Policy('inv12-day-one-test-doubles')

    def test_keys_are_a_tuple_in_declaration_order(self):
        declaration = governed_exceptions(
            LIST_ID, RULE, ['z', 'a', 'm'], default=Policy('inv12-x'), default_covers=3
        )
        assert declaration.keys == ('z', 'a', 'm')

    def test_accepts_any_iterable_of_keys(self):
        """The Contract types ``keys`` as an Iterable, not a list."""
        declaration = governed_exceptions(
            LIST_ID, RULE, (k for k in ('a', 'b')), default=Policy('inv12-x'), default_covers=2
        )
        assert declaration.keys == ('a', 'b')


class TestGovernedListIsImmutable:
    """A declaration is a value: nothing downstream can edit one."""

    def test_is_frozen(self):
        declaration = governed_exceptions(
            LIST_ID, RULE, ['a'], dispositions={'a': Policy('inv12-x')}
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            declaration.rule = 'something else'  # type: ignore[misc]

    def test_overrides_is_read_only(self):
        declaration = governed_exceptions(
            LIST_ID, RULE, ['a'], dispositions={'a': Policy('inv12-x')}
        )
        with pytest.raises(TypeError):
            declaration.overrides['a'] = Policy('inv12-y')  # type: ignore[index]

    def test_overrides_is_not_the_caller_s_dict(self):
        """A defensive copy, so a later mutation of the caller's dict cannot reach in.

        Declarations are module-level literals in a test file; a shared mutable
        mapping would let one package's declaration silently re-disposition
        another's.
        """
        mutable: dict[str, Disposition] = {'a': Policy('inv12-x')}
        declaration = governed_exceptions(LIST_ID, RULE, ['a'], dispositions=mutable)
        mutable['a'] = Debt(TaskRef(5149))
        assert declaration.overrides['a'] == Policy('inv12-x')


class TestDispositionFor:
    """The single home of the override-else-default resolution rule."""

    def test_returns_the_override_when_one_exists(self):
        declaration = governed_exceptions(
            LIST_ID,
            RULE,
            ['a', 'b'],
            default=Debt(TaskRef(5149)),
            default_covers=1,
            dispositions={'a': Policy('inv12-x')},
        )
        assert declaration.disposition_for('a') == Policy('inv12-x')

    def test_falls_back_to_the_default(self):
        declaration = governed_exceptions(
            LIST_ID,
            RULE,
            ['a', 'b'],
            default=Debt(TaskRef(5149)),
            default_covers=1,
            dispositions={'a': Policy('inv12-x')},
        )
        assert declaration.disposition_for('b') == Debt(TaskRef(5149))

    def test_an_unknown_key_raises_naming_the_list_and_the_key(self):
        """Not a silent None: a key that is not in the declaration is a caller bug.

        Returning the default for a key nobody declared would let a report
        print a disposition for an entry that does not exist, which is exactly
        the shape of the drift D5 makes loud by construction.
        """
        declaration = governed_exceptions(LIST_ID, RULE, ['a'], dispositions={'a': Policy('x')})
        with pytest.raises(KeyError) as excinfo:
            declaration.disposition_for('nope')
        assert LIST_ID in str(excinfo.value)
        assert 'nope' in str(excinfo.value)


class TestScenario15RuntimeHalf:
    """A defaulted list reports "default, default_covers, N overrides" with no key set."""

    def test_a_report_can_render_a_defaulted_list_without_the_keys(self):
        """The runtime half of PRD scenario 15.

        The static reader sees ``default``, ``default_covers`` and the
        overrides because they are literals; only this runtime value also
        knows the key set.  The report prints the former three, so they must
        read back off the GovernedList exactly as declared — otherwise the
        report would have to enumerate 61 keys to say one sentence.
        """
        declaration = governed_exceptions(
            LIST_ID,
            RULE,
            _keys(61),
            default=Debt(TaskRef(5149)),
            default_covers=59,
            dispositions={'key-000': Policy('inv12-x'), 'key-001': Debt(TaskRef(5602))},
        )
        assert declaration.default == Debt(TaskRef(5149))
        assert declaration.default_covers == 59
        assert len(declaration.overrides) == 2


class TestDeclarationForms:
    """The declaration-side SPOT, rendered by whoever needs it."""

    def test_is_a_non_empty_tuple_of_str(self):
        assert isinstance(DECLARATION_FORMS, tuple)
        assert DECLARATION_FORMS
        assert all(isinstance(form, str) for form in DECLARATION_FORMS)


class TestUndisposedException:
    """The INV-12 violation itself: an entry nobody owns and no ruling covers."""

    @staticmethod
    def _raised(**kwargs):
        with pytest.raises(UndisposedException) as excinfo:
            governed_exceptions(LIST_ID, RULE, **kwargs)
        return excinfo.value

    def test_names_only_the_undisposed_key(self):
        error = self._raised(keys=['a', 'b'], dispositions={'a': Policy('inv12-x')})
        assert error.keys == ('b',)
        assert "'b'" in str(error)

    def test_names_every_undisposed_key(self):
        error = self._raised(keys=['a', 'b'])
        assert error.keys == ('a', 'b')

    def test_message_names_the_list_and_publishes_every_declaration_form(self):
        message = str(self._raised(keys=['a', 'b']))
        assert LIST_ID in message
        for form in DECLARATION_FORMS:
            assert form in message

    def test_carries_structured_attributes(self):
        error = self._raised(
            keys=['a', 'b', 'c'], default=Debt(TaskRef(5149)), default_covers=2
        )
        assert error.list_id == LIST_ID
        assert error.keys == ('a', 'b', 'c')
        assert error.covered == 3
        assert error.default_covers == 2

    def test_scenario_12_a_defaulted_list_gains_a_key(self):
        """BOUNDARY SCENARIO 12, and it falls out of the general count rule.

        A list declared ``default_covers=2`` now has three keys and no
        overrides.  Nothing about the two original entries changed; the third
        is simply undisposed, which is exactly what D4 makes ``default_covers``
        a COUNT for — a new entry needs an explicit override or a visible
        increment beside the disposition it is borrowing.  There is no
        special-case branch for this in the implementation, and there must not
        be one: it is the same "covered > declared" arm as a list with no
        default at all.

        WHICH key is the new one is NOT asserted, because the declaration never
        said: ``default_covers`` is a count, so all three keys borrow the default
        and nothing distinguishes the third.  The added key is among the
        borrowers on ``.keys``, and that is the whole of what is knowable here.
        """
        error = self._raised(
            keys=['a', 'b', 'c'], default=Debt(TaskRef(5149)), default_covers=2
        )
        message = str(error)
        assert 'c' in error.keys  # the added key is among the borrowers
        assert 'default_covers=2' in message  # the declared count, as the author wrote it
        assert '3' in message  # how many actually borrow
        assert '1 entr' in message  # how many of them are undisposed
        assert 'override' in message and 'increment' in message  # what to do about it

    def test_a_defaulted_list_does_not_call_every_borrower_undisposed(self):
        """The size at which the misstatement stops being invisible.

        A 61-key list whose default covers 59 has TWO undisposed entries.  Naming
        all 61 under the word "undisposed" would be false, and would hand an agent
        reading the red gate a wall of noise whose real cause is two lines — the
        failure mode the check ordering elsewhere in this module exists to avoid.
        The count is knowable and is stated; WHICH entries is not, and saying so
        is the honest report.  Pinned at 61 rather than at 3 because at 3 the
        difference between the true and the false message is invisible.
        """
        keys = _keys(61)
        error = self._raised(keys=keys, default=Debt(TaskRef(5149)), default_covers=59)
        message = str(error)
        assert error.keys == tuple(sorted(keys))  # every borrower, for a report
        assert not any(key in message for key in keys)  # but not in the prose
        assert '2 entr' in message  # the number that IS knowable
        assert 'does not record WHICH' in message  # and the honest statement
        assert len(message) < len(str(DECLARATION_FORMS)) + 400  # no 992-char wall

    def test_is_raised_when_the_declaring_module_is_collected(self, tmp_path):
        """The shape the PRD specifies, observed rather than assumed.

        D4 puts the declaration at module scope in a package's test tree, so
        the violation surfaces when pytest COLLECTS that module.  Calling the
        function directly would leave that unverified, so this writes a real
        module and imports it.
        """
        module_path = tmp_path / 'declares_a_governed_list.py'
        module_path.write_text(
            'from shared.governed_exceptions import Debt, TaskRef, governed_exceptions\n'
            '\n'
            'governed_exceptions(\n'
            "    'pkg.tests.example',\n"
            "    'every entry is owned or ratified',\n"
            "    ['a', 'b', 'c'],\n"
            '    default=Debt(TaskRef(5149)),\n'
            '    default_covers=2,\n'
            ')\n',
            encoding='utf-8',
        )
        spec = importlib.util.spec_from_file_location('declares_a_governed_list', module_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        with pytest.raises(UndisposedException) as excinfo:
            spec.loader.exec_module(module)
        assert excinfo.value.list_id == 'pkg.tests.example'

    def test_is_not_a_value_error(self):
        """A broad ``except ValueError`` must never swallow an INV-12 breach.

        The three faults map to different exit codes — 1 for this violation, 2
        for a broken instrument — so they share no base class and none
        subclasses a builtin that consumers already catch.
        """
        assert not issubclass(UndisposedException, ValueError)
        assert not issubclass(UndisposedException, MalformedDeclaration)
        assert not issubclass(MalformedDeclaration, UndisposedException)


class TestMalformedDeclaration:
    """Instrument failures: the declaration itself is broken, so nothing is judged."""

    @staticmethod
    def _message(list_id=LIST_ID, rule=RULE, keys=(), **kwargs):
        with pytest.raises(MalformedDeclaration) as excinfo:
            governed_exceptions(list_id, rule, keys, **kwargs)
        return str(excinfo.value)

    @pytest.mark.parametrize('bad_keys', [[1, 2], [1, 'a'], [None], [('a', 'b')]])
    def test_a_declared_key_that_is_not_a_str(self, bad_keys):
        """A non-str key escapes BOTH exit codes if it is not caught here.

        Every check below the key set sorts it, so a mixed-type key list raised a
        bare ``TypeError`` out of ``sorted`` — not MalformedDeclaration (exit 2)
        and not UndisposedException (exit 1), which is the whole two-code split
        this module is built around.  A list keyed by task ids or line numbers is
        a natural thing for a declaration author to pass.
        """
        message = self._message(keys=bad_keys)
        assert repr(bad_keys[0]) in message

    def test_a_duplicate_key(self):
        message = self._message(keys=['a', 'b', 'a'], dispositions={'a': Policy('x'), 'b': Policy('y')})
        assert "'a'" in message

    def test_an_override_for_a_key_that_is_not_declared(self):
        message = self._message(keys=['a'], dispositions={'a': Policy('x'), 'stray': Policy('y')})
        assert "'stray'" in message

    def test_a_default_without_a_count(self):
        message = self._message(keys=['a'], default=Policy('x'))
        assert 'default_covers' in message

    def test_a_count_without_a_default(self):
        message = self._message(keys=['a'], default_covers=1)
        assert 'default_covers' in message

    def test_a_count_that_over_claims(self):
        """A stale literal, not a violation: nothing is left undisposed.

        ``default_covers`` larger than the number of keys without an override
        means the list SHRANK and the literal was not decremented.  Every
        remaining entry still has a disposition, so this is exit 2 (fix the
        instrument), not exit 1 (fix the invariant).
        """
        message = self._message(keys=['a', 'b'], default=Policy('x'), default_covers=5)
        assert 'default_covers=5' in message
        assert '2' in message

    @pytest.mark.parametrize(
        'bad_list_id',
        [
            'nodots',
            '',
            '.leading',
            'trailing.',
            'a..b',
            # A trailing newline: well formed to the eye, and two list ids that
            # differ only by one would be indistinguishable in every report, which
            # is exactly the global uniqueness the dotted form exists to give.
            'a.b\n',
            17,
        ],
    )
    def test_a_list_id_that_is_not_dotted_and_unique(self, bad_list_id):
        message = self._message(list_id=bad_list_id, keys=['a'], dispositions={'a': Policy('x')})
        assert repr(bad_list_id) in message

    @pytest.mark.parametrize('bad_rule', ['', '   ', None])
    def test_a_blank_rule(self, bad_rule):
        """The rule is what an operator judges a NEW entry against, so it is required."""
        message = self._message(rule=bad_rule, keys=['a'], dispositions={'a': Policy('x')})
        assert repr(bad_rule) in message

    @pytest.mark.parametrize('bad', ['task 5149', TaskRef(5149), 5149, None])
    def test_an_override_value_that_is_not_a_disposition(self, bad):
        if bad is None:
            pytest.skip('None is how "no override" is spelled; it never appears as a value')
        message = self._message(keys=['a'], dispositions={'a': bad})
        assert repr(bad) in message

    @pytest.mark.parametrize('bad', ['task 5149', TaskRef(5149), 5149])
    def test_a_default_that_is_not_a_disposition(self, bad):
        message = self._message(keys=['a'], default=bad, default_covers=1)
        assert repr(bad) in message


class TestCheckOrdering:
    """Preconditions before judgements — the local fault is what gets reported."""

    def test_a_duplicate_key_is_reported_before_an_undisposed_one(self):
        """Both faults are present; the STRUCTURAL one is named.

        A duplicate key makes every count downstream of it meaningless, so
        reporting the undisposed keys first would hand back a wall of noise
        whose one real cause is a line the reader can see.  This is the
        ordering discipline ``scripts/merge_lane_metrics.py::
        check_against_baseline`` applies for the same reason.
        """
        with pytest.raises(MalformedDeclaration):
            governed_exceptions(LIST_ID, RULE, ['a', 'a', 'b'])
