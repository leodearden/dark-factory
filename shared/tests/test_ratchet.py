"""Tests for shared.ratchet — the multiset ratchet kernel.

PRD ``plans/inv12-exceptions-owned-or-ratified-prd.md``, Contract section
**Kernel**, decisions **D7** (the inline baseline is a content-keyed multiset)
and **D13** (two ratchet idioms, SPOT per idiom, this PRD owns the multiset
one), and design invariant **INV-11** ``no-silent-fail-soft``.

WHAT INV-11 BUYS HERE, and why so much of this file is about refusals.  A
ratchet that compares a PARTIAL enumeration against a baseline measures lower
than the truth, so it reads as a clean tree — or worse, as an improvement
worth writing into the baseline.  The same is true of a comparison between
two enumerations produced under different parameters.  Both are therefore
refusals, not degraded comparisons, and the refusal names what was skipped
rather than reducing it to a count.

D13 IS WHY THIS FILE DOES NOT IMPORT ``scripts/merge_lane_metrics.py``.  That
instrument is the SCALAR ratchet idiom and is owned by other tasks; its
conventions are matched here, its code is not reached for.

TDD pair 1: the Enumeration value (GREEN on impl step-10).
TDD pair 2: excess / slack / tighten + the no-add-key property (GREEN on impl step-12).
TDD pair 3: the two comparability refusals, uniform across all three (GREEN on impl step-14).
TDD pair 4: dump / load round trip + the committed-file shape (GREEN on impl step-16).
TDD pair 5: load()'s refusals, every one naming the path (GREEN on impl step-18).
TDD pair 6: a params key that is ABSENT vs. one that is present and null (GREEN on impl step-20).
"""
from __future__ import annotations

import ast
import dataclasses
import json
from collections import Counter
from pathlib import Path

import pytest

import shared.ratchet
from shared.ratchet import (
    BASELINE_README,
    SCHEMA_VERSION,
    BaselineUnusable,
    Enumeration,
    IncompleteEnumeration,
    ParamsMismatch,
    RatchetError,
    dump,
    excess,
    load,
    slack,
    tighten,
)

# Same src-root expression as shared/tests/conftest.py and
# test_pure_stdlib_leaves.py — read the LOCAL tree, never an installed copy.
_SRC = Path(__file__).resolve().parent.parent / 'src'
_MODULE_SOURCE = _SRC / 'shared' / 'ratchet.py'

# A params block in the shape D7's scanner produces: the key version it hashed
# under, and the suppression kinds it swept for.
PARAMS = {'key_version': 1, 'kinds': ('noqa', 'type: ignore')}


class TestEnumerationAccepts:
    """The value the kernel compares: a multiset, its parameters, its honesty."""

    def test_counts_and_params_read_back(self):
        enumeration = Enumeration(counts={'abc123': 2, 'def456': 1}, params=PARAMS)
        assert dict(enumeration.counts) == {'abc123': 2, 'def456': 1}
        assert dict(enumeration.params) == PARAMS

    def test_complete_defaults_to_true_and_unreadable_to_empty(self):
        """The honest default: a run that skipped nothing says so without being asked."""
        enumeration = Enumeration(counts={'abc123': 1}, params=PARAMS)
        assert enumeration.complete is True
        assert enumeration.unreadable == ()

    def test_is_frozen(self):
        enumeration = Enumeration(counts={'abc123': 1}, params=PARAMS)
        with pytest.raises(dataclasses.FrozenInstanceError):
            enumeration.complete = False  # type: ignore[misc]

    @pytest.mark.parametrize('field', ['counts', 'params'])
    def test_mappings_are_read_only(self, field):
        enumeration = Enumeration(counts={'abc123': 1}, params=PARAMS)
        with pytest.raises(TypeError):
            getattr(enumeration, field)['abc123'] = 99

    def test_counts_is_a_defensive_copy(self):
        mutable = {'abc123': 1}
        enumeration = Enumeration(counts=mutable, params=PARAMS)
        mutable['abc123'] = 99
        assert enumeration.counts['abc123'] == 1

    def test_params_is_a_defensive_copy(self):
        mutable = {'key_version': 1}
        enumeration = Enumeration(counts={}, params=mutable)
        mutable['key_version'] = 2
        assert enumeration.params['key_version'] == 1

    def test_a_list_in_params_normalises_to_a_tuple(self):
        """So a JSON round trip cannot change equality.

        JSON has one sequence type and Python has two.  If a params list stayed
        a list, an Enumeration built in memory would compare unequal to the
        same Enumeration read back off disk, and every comparison after a
        reload would refuse with a spurious params mismatch.
        """
        enumeration = Enumeration(counts={}, params={'kinds': ['noqa', 'nosec']})
        assert enumeration.params['kinds'] == ('noqa', 'nosec')

    @pytest.mark.parametrize('value', ['a string', 17, 1.5, True, None])
    def test_params_accepts_every_json_scalar(self, value):
        assert Enumeration(counts={}, params={'p': value}).params['p'] == value

    def test_an_empty_but_honestly_incomplete_enumeration_is_legal(self):
        """The shape that must REFUSE comparison later, so it has to construct now.

        A scan that could read nothing is a real outcome.  Making it
        unconstructable would leave the caller with no way to say "I measured
        nothing and I know it", which is the only alternative to silently
        reporting an empty — and therefore clean — tree.
        """
        enumeration = Enumeration(counts={}, params={}, complete=False, unreadable=('x.py',))
        assert enumeration.complete is False
        assert enumeration.unreadable == ('x.py',)


class TestEnumerationRejects:
    """A bad in-process construction is a programmer error: ValueError, named.

    The counterpart refusal for external bytes is ``BaselineUnusable``, which
    names the PATH — at that boundary the path is what an operator needs, and
    the distinction is drawn deliberately.
    """

    @staticmethod
    def _message(**kwargs):
        kwargs.setdefault('counts', {})
        kwargs.setdefault('params', {})
        with pytest.raises(ValueError) as excinfo:
            Enumeration(**kwargs)
        message = str(excinfo.value)
        assert 'Enumeration' in message
        return message

    @pytest.mark.parametrize('count', [0, -1])
    def test_a_zero_or_negative_count(self, count):
        """A multiset baseline carries no zero entries.

        Counter arithmetic drops non-positive results anyway, so a zero on disk
        would make dump and load non-canonical: the file would not round-trip
        to itself, and two runs over equal input would produce different bytes.
        """
        assert repr(count) in self._message(counts={'abc123': count})

    @pytest.mark.parametrize('count', ['2', 1.0, None, True])
    def test_a_non_int_count(self, count):
        """``True`` is here for the same reason it is in TaskRef's table."""
        assert repr(count) in self._message(counts={'abc123': count})

    @pytest.mark.parametrize('key', [17, None, ('a', 'b')])
    def test_a_non_str_key(self, key):
        """Keys are opaque identity tokens, and JSON object keys are strings."""
        assert repr(key) in self._message(counts={key: 1})

    @pytest.mark.parametrize('key', [17, None, ('a', 'b')])
    def test_a_non_str_params_key(self, key):
        """A JSON object key is a string, and this block IS a JSON object.

        Both consequences escaped the RatchetError family the base class promises
        one ``except`` clause for.  ``dump`` raised a bare ``TypeError`` out of
        ``json.dumps(sort_keys=True)`` when the keys were of mixed type; and a
        block that did dump came back with the key rewritten as its JSON spelling,
        so ``load(p) == e`` was False and the next comparison sorted a str against
        an int and raised a bare ``TypeError`` rather than ParamsMismatch.  A
        caller mapping ``except RatchetError`` to exit 2 got a traceback instead.
        """
        assert repr(key) in self._message(params={key: 'x'})

    @pytest.mark.parametrize('value', [{'a': 1}, {1, 2}, object()])
    def test_a_params_value_that_is_not_json(self, value):
        """The params block is written to and read from a JSON file verbatim."""
        assert 'p' in self._message(params={'p': value})

    @pytest.mark.parametrize('value', [float('nan'), float('inf'), float('-inf')])
    def test_a_non_finite_float_in_params(self, value):
        """The value the round-trip argument is ABOUT, and it was let through.

        ``_JSON_SCALARS`` exists because a value JSON cannot express would not
        survive the round trip and would make every post-reload comparison refuse
        with a spurious mismatch.  A non-finite float is precisely that value:
        ``json.dumps`` writes the bare token ``NaN``, which is not RFC 8259 and
        which strict readers reject in the committed file a human is told to
        review, and ``nan != nan`` refuses every later comparison FOREVER.
        """
        assert repr(value) in self._message(params={'p': value})

    def test_a_tuple_containing_a_non_finite_float(self):
        """Enforced at both depths, so the scalar rule has one reading."""
        assert 'p' in self._message(params={'p': (1.0, float('inf'))})

    def test_a_tuple_of_non_scalars_in_params(self):
        assert 'p' in self._message(params={'p': ({'a': 1},)})

    @pytest.mark.parametrize('value', [1.5, -0.0, 1e308])
    def test_a_finite_float_is_still_accepted(self, value):
        """The over-correction guard: only NON-finite floats are excluded."""
        assert Enumeration(counts={}, params={'p': value}).params['p'] == value

    def test_complete_true_with_a_non_empty_unreadable(self):
        """THE CONTRADICTION INV-11 EXISTS TO FORBID, refused rather than corrected.

        Silently flipping ``complete`` to False would repair a caller's bug
        without telling anyone, and the caller's next line probably reports the
        run as clean.  Refusing is the loud half of loud-over-silent, and an
        honest caller never trips it: the defaults are ``complete=True`` with
        an empty ``unreadable``.
        """
        assert 'x.py' in self._message(complete=True, unreadable=('x.py',))


class TestModuleStructure:
    """D4: importing the kernel defines things and does nothing else."""

    def test_module_body_has_no_bare_expression_statement(self):
        body = ast.parse(_MODULE_SOURCE.read_text(encoding='utf-8')).body
        offenders = [
            ast.unparse(node)
            for node in body
            if isinstance(node, ast.Expr)
            and not (isinstance(node.value, ast.Constant) and isinstance(node.value.value, str))
        ]
        assert offenders == [], (
            'shared/src/shared/ratchet.py runs statement(s) at import time:\n  '
            + '\n  '.join(offenders)
            + '\nPRD D4 requires that importing this module define things and do nothing else.'
        )


def comparable(current_counts, baseline_counts, params=None):
    """Two enumerations that differ only in their counts.

    Every arithmetic test below builds its pair through this helper, so no test
    accidentally exercises the comparability preconditions instead of the
    arithmetic it names.
    """
    shared_params = PARAMS if params is None else params
    return (
        Enumeration(counts=current_counts, params=shared_params),
        Enumeration(counts=baseline_counts, params=shared_params),
    )


class TestExcess:
    """The violation report: what the current run has beyond the baseline."""

    def test_reports_growth_and_new_keys(self):
        current, baseline = comparable({'a': 3, 'b': 1}, {'a': 1})
        assert excess(current, baseline) == Counter({'a': 2, 'b': 1})

    def test_equality_is_empty(self):
        current, baseline = comparable({'a': 3, 'b': 1}, {'a': 3, 'b': 1})
        assert excess(current, baseline) == Counter()

    def test_shrinkage_is_empty_not_negative(self):
        """Counter's saturating subtraction is what makes "lowering is fine" free."""
        current, baseline = comparable({'a': 1}, {'a': 3, 'b': 2})
        assert excess(current, baseline) == Counter()


class TestSlack:
    """The un-spent headroom: what the baseline still permits and nobody uses."""

    def test_reports_the_unused_multiplicity(self):
        current, baseline = comparable({'a': 1}, {'a': 3})
        assert slack(current, baseline) == Counter({'a': 2})

    def test_a_key_gone_from_current_shows_its_full_baseline_multiplicity(self):
        current, baseline = comparable({'a': 1}, {'a': 1, 'gone': 4})
        assert slack(current, baseline) == Counter({'gone': 4})

    def test_equality_is_empty(self):
        current, baseline = comparable({'a': 3}, {'a': 3})
        assert slack(current, baseline) == Counter()


class TestTighten:
    """The only baseline-producing function: the pointwise minimum."""

    def test_admits_no_new_key_and_drops_a_departed_one(self):
        current, baseline = comparable({'a': 3, 'b': 1}, {'a': 1, 'c': 5})
        assert tighten(current, baseline) == Counter({'a': 1})

    def test_is_idempotent(self):
        """The second ``--tighten`` of boundary scenario 5 changes nothing."""
        current, baseline = comparable({'a': 3, 'b': 1}, {'a': 1, 'c': 5})
        tightened = Enumeration(counts=dict(tighten(current, baseline)), params=PARAMS)
        assert tighten(current, tightened) == Counter(tightened.counts)


class TestTheKernelIsAMultisetNotASet:
    """Repeated multiplicities survive every operation.

    Mirrors ``shared/tests/test_loop_blocking_gate.py::
    test_ratchet_is_a_multiset_not_a_set``, and for the same reason: under set
    semantics a SECOND site sharing a key with an already-blessed one slips in
    silently.  D7 accepts that 67% of sites share a key with another site, so
    multiplicity is the only thing standing between the baseline and a free
    extra suppression per key.  Nobody "simplifies" this to sets.
    """

    def test_a_second_site_under_an_existing_key_is_excess(self):
        current, baseline = comparable({'a': 2}, {'a': 1})
        assert excess(current, baseline) == Counter({'a': 1})

    def test_multiplicity_survives_slack_and_tighten(self):
        current, baseline = comparable({'a': 2}, {'a': 5})
        assert slack(current, baseline) == Counter({'a': 3})
        assert tighten(current, baseline) == Counter({'a': 2})


class TestOperationsAreSideEffectFree:
    """All three return a fresh Counter and mutate neither argument."""

    @pytest.mark.parametrize('operation', [excess, slack, tighten])
    def test_arguments_are_unchanged(self, operation):
        current, baseline = comparable({'a': 3, 'b': 1}, {'a': 1, 'c': 5})
        operation(current, baseline)
        assert dict(current.counts) == {'a': 3, 'b': 1}
        assert dict(baseline.counts) == {'a': 1, 'c': 5}

    @pytest.mark.parametrize('operation', [excess, slack, tighten])
    def test_the_result_is_a_fresh_counter(self, operation):
        current, baseline = comparable({'a': 3}, {'a': 1})
        result = operation(current, baseline)
        assert isinstance(result, Counter)
        result['a'] = 99
        assert dict(current.counts) == {'a': 3}
        assert dict(baseline.counts) == {'a': 1}


class TestNoFunctionCanAddAKeyToABaseline:
    """The surface pin the capability manifest's `manual` check defers to.

    The PRD's Contract says "no function can add a key to an existing
    baseline".  That is asserted two ways here, because the property has two
    halves.  ARITHMETICALLY, :func:`tighten` is the only baseline-producing
    function and its keys are a subset of the baseline's by construction
    (``Counter.__and__`` is the pointwise minimum), so the property is
    structural rather than a check anyone could forget.  BY SURFACE, no other
    verb exists — and a future ``absorb`` / ``widen`` / ``write_baseline``
    cannot appear without turning ``test_the_public_surface_is_pinned`` red.

    :func:`excess`'s output legitimately contains keys the baseline never had.
    That is not a counter-example: excess is a VIOLATION REPORT, never a
    baseline, and a new key in it IS the finding.
    """

    PAIRS = [
        ({}, {}),
        ({'a': 1}, {}),
        ({}, {'a': 1}),
        ({'a': 1}, {'b': 1}),  # disjoint
        ({'a': 1, 'b': 2}, {'a': 1}),  # current is a superset
        ({'a': 1}, {'a': 1, 'b': 2}),  # baseline is a superset
        ({'a': 5}, {'a': 5}),  # equal
        ({'a': 9, 'b': 9, 'c': 9}, {'b': 1}),  # growth everywhere
    ]

    @pytest.mark.parametrize(('current_counts', 'baseline_counts'), PAIRS)
    def test_tighten_never_admits_a_key_the_baseline_lacks(
        self, current_counts, baseline_counts
    ):
        current, baseline = comparable(current_counts, baseline_counts)
        assert set(tighten(current, baseline)) <= set(baseline.counts)

    #: The kernel's whole intended public surface. There is no absorb, no
    #: widen, and no write-baseline verb, and there is not meant to be one.
    #: The two constants are part of the surface because a consumer asserting
    #: against the README or the schema version must cite them rather than
    #: retype them.
    EXPECTED_SURFACE = frozenset(
        {
            'BASELINE_README',
            'BaselineUnusable',
            'Enumeration',
            'IncompleteEnumeration',
            'ParamsMismatch',
            'RatchetError',
            'SCHEMA_VERSION',
            'dump',
            'excess',
            'load',
            'slack',
            'tighten',
        }
    )

    def test_the_public_surface_carries_no_unexpected_verb(self):
        """An explicit expected surface, so a new verb cannot arrive unnoticed.

        Asserted as containment rather than equality on purpose, and it is not
        the weaker pin it looks like.  Containment is exactly the property the
        manifest defers to — a future ``absorb`` / ``widen`` /
        ``write_baseline`` turns this red the moment it is exported.  The other
        half of equality, that every name above still EXISTS, is pinned by this
        module's own imports: the suite cannot collect without them.  Splitting
        it this way also means the pin holds at every stage of the module's
        construction rather than only once the last function lands.
        """
        assert set(shared.ratchet.__all__) <= self.EXPECTED_SURFACE


#: Every kernel operation, so a refusal is asserted UNIFORMLY. A future fourth
#: operation added without its precondition has to be added here too, and the
#: parametrization is what makes that visible.
OPERATIONS = [excess, slack, tighten]


class TestParamsMismatchRefusal:
    """Different parameters mean the two sides are not measurements of one thing.

    From ``scripts/merge_lane_metrics.py::_require_matching_params``: comparing
    across a parameter change reports a wall of downstream violations instead
    of the one named cause, and the one named cause is the parameter change.
    """

    @pytest.mark.parametrize('operation', OPERATIONS)
    @pytest.mark.parametrize(
        ('current_params', 'baseline_params'),
        [
            ({'key_version': 2}, {'key_version': 1}),  # a changed value
            ({'key_version': 1, 'extra': 'x'}, {'key_version': 1}),  # an added key
            ({'key_version': 1}, {'key_version': 1, 'extra': 'x'}),  # a removed key
            ({'kinds': ('noqa', 'nosec')}, {'kinds': ('nosec', 'noqa')}),  # tuple order
            # A null-valued key gained and lost.  These two rows exist because
            # ``.get``-based difference detection conflates ABSENT with NULL,
            # and the params block is the one mapping in this module where
            # None is a legal VALUE rather than a sentinel for missing: every
            # other row above differs on a str or an int, so a bare
            # ``.get(key) != .get(key)`` catches them all while letting a
            # params block that gained or lost a null compare as MATCHING.
            ({'strict': None}, {}),  # the current side gained a null-valued key
            ({}, {'strict': None}),  # the current side lost one
        ],
    )
    def test_every_operation_refuses(self, operation, current_params, baseline_params):
        current = Enumeration(counts={'a': 1}, params=current_params)
        baseline = Enumeration(counts={'a': 1}, params=baseline_params)
        with pytest.raises(ParamsMismatch):
            operation(current, baseline)

    def test_names_the_differing_keys_and_both_sides(self):
        current = Enumeration(counts={}, params={'key_version': 2})
        baseline = Enumeration(counts={}, params={'key_version': 1})
        with pytest.raises(ParamsMismatch) as excinfo:
            excess(current, baseline)
        error = excinfo.value
        assert error.differing == ('key_version',)
        assert dict(error.current_params) == {'key_version': 2}
        assert dict(error.baseline_params) == {'key_version': 1}
        assert 'key_version' in str(error)
        assert '1' in str(error) and '2' in str(error)

    @pytest.mark.parametrize('operation', OPERATIONS)
    def test_a_null_valued_key_on_both_sides_is_not_a_difference(self, operation):
        """The over-correction guard for the absent-vs-null distinction.

        A sentinel fix that made every None-valued key differ from ITSELF
        would refuse every comparison of a params block containing a null, and
        nothing else in this suite would notice: no other case here puts a
        None in params on both sides.  None is a legal params value, so a key
        holding one on both sides is two sides agreeing.
        """
        params = {'strict': None, 'key_version': 1}
        current = Enumeration(counts={'a': 2}, params=dict(params))
        baseline = Enumeration(counts={'a': 1}, params=dict(params))
        assert operation(current, baseline) == {
            excess: Counter({'a': 1}),
            slack: Counter(),
            tighten: Counter({'a': 1}),
        }[operation]

    def test_names_an_absent_side_as_absent_and_never_as_none(self):
        """``None`` and ``absent`` are different mistakes with different fixes.

        Rendering the absent side as ``None`` sends the operator hunting for a
        null they never wrote — the mistake this module already solved at the
        file boundary with ``_found``.  The present side's ``None`` must still
        render as ``None``, because that null IS what they wrote.
        """
        current = Enumeration(counts={}, params={'strict': None})
        baseline = Enumeration(counts={}, params={})
        with pytest.raises(ParamsMismatch) as excinfo:
            excess(current, baseline)
        error = excinfo.value
        assert error.differing == ('strict',)
        assert 'strict: baseline=absent current=None' in str(error)
        assert 'baseline=None' not in str(error)


class TestIncompleteEnumerationRefusal:
    """A partial scan measures LOW, so comparing it reads as a clean tree."""

    @staticmethod
    def _partial(counts):
        return Enumeration(
            counts=counts, params=PARAMS, complete=False, unreadable=('pkg/unreadable.py',)
        )

    @staticmethod
    def _whole(counts):
        return Enumeration(counts=counts, params=PARAMS)

    @pytest.mark.parametrize('operation', OPERATIONS)
    @pytest.mark.parametrize('incomplete_side', ['current', 'baseline', 'both'])
    def test_every_operation_refuses_whichever_side_is_partial(
        self, operation, incomplete_side
    ):
        current = self._partial({'a': 1}) if incomplete_side in ('current', 'both') else self._whole({'a': 1})
        baseline = self._partial({'a': 1}) if incomplete_side in ('baseline', 'both') else self._whole({'a': 1})
        with pytest.raises(IncompleteEnumeration):
            operation(current, baseline)

    def test_names_the_side_and_lists_what_it_skipped(self):
        with pytest.raises(IncompleteEnumeration) as excinfo:
            excess(self._partial({'a': 1}), self._whole({'a': 1}))
        error = excinfo.value
        assert error.side == 'current'
        assert error.unreadable == ('pkg/unreadable.py',)
        assert 'pkg/unreadable.py' in str(error)

    def test_the_gate_is_identity_not_truthiness(self):
        """An ABSENT completeness flag must refuse, not be read as "probably fine".

        INV-11's shape is ``complete is True``, and
        ``scripts/merge_lane_metrics.py::_require_complete_enumeration`` spells
        it that way for exactly this reason: under truthiness a value that is
        merely non-falsy — or a flag a future loader forgot to populate —
        compares green.  Identity refuses anything that is not literally True.
        """
        with pytest.raises(IncompleteEnumeration):
            excess(self._partial({}), self._whole({}))


class TestPreconditionsRunBeforeAnyComparison:
    """Ordering is deterministic and the same for all three operations."""

    @pytest.mark.parametrize('operation', OPERATIONS)
    def test_incompleteness_is_reported_before_a_params_mismatch(self, operation):
        """Both faults present; the named winner is asserted, not left to chance.

        The discipline of ``scripts/merge_lane_metrics.py::
        check_against_baseline`` and its
        ``test_preconditions_are_checked_before_any_comparison``: the
        preconditions run ahead of every comparison, in a fixed order, so the
        red a caller sees is reproducible rather than dependent on which check
        happened to be written first.
        """
        current = Enumeration(
            counts={'a': 1},
            params={'key_version': 2},
            complete=False,
            unreadable=('pkg/unreadable.py',),
        )
        baseline = Enumeration(counts={'a': 1}, params={'key_version': 1})
        with pytest.raises(IncompleteEnumeration):
            operation(current, baseline)


class TestTheRefusalFamilyIsOneExceptClause:
    """gamma1 maps every refusal to exit 2 with a single ``except``."""

    @pytest.mark.parametrize('error', [ParamsMismatch, IncompleteEnumeration])
    def test_is_a_ratchet_error(self, error):
        assert issubclass(error, RatchetError)


#: A value exercising everything the file format has to carry: multiplicities
#: above 1, and a params block mixing an int, a str, a bool and a tuple of str.
REPRESENTATIVE = Enumeration(
    counts={'ffee11223344': 3, 'aabb00998877': 1, '00ff11ee22dd': 2},
    params={'key_version': 1, 'algorithm': 'sha256', 'strict': True, 'kinds': ('noqa', 'nosec')},
)


class TestRoundTrip:
    """What goes to disk comes back equal — including the honesty flags."""

    def test_a_complete_enumeration_survives(self, tmp_path):
        path = tmp_path / 'baseline.json'
        dump(REPRESENTATIVE, path)
        assert load(path) == REPRESENTATIVE

    def test_an_incomplete_enumeration_survives(self, tmp_path):
        """Completeness and the NAMED list must cross the file boundary.

        If they did not, a later run would load a partial baseline as a
        complete one and compare against it — which is the whole failure INV-11
        names, merely deferred by one process boundary.
        """
        partial = Enumeration(
            counts={'aabb00998877': 1},
            params={'key_version': 1},
            complete=False,
            unreadable=('pkg/a.py', 'pkg/b.py'),
        )
        path = tmp_path / 'baseline.json'
        dump(partial, path)
        reloaded = load(path)
        assert reloaded == partial
        assert reloaded.complete is False
        assert reloaded.unreadable == ('pkg/a.py', 'pkg/b.py')

    def test_an_empty_enumeration_still_carries_its_params_and_version(self, tmp_path):
        path = tmp_path / 'baseline.json'
        dump(Enumeration(counts={}, params={'key_version': 1}), path)
        parsed = json.loads(path.read_text(encoding='utf-8'))
        assert parsed['params'] == {'key_version': 1}
        assert parsed['schema_version'] == SCHEMA_VERSION
        assert load(path).counts == {}


class TestCommittedFileShape:
    """The bytes a human opens and a merge has to combine."""

    @staticmethod
    def _dumped(tmp_path, enumeration=REPRESENTATIVE):
        path = tmp_path / 'baseline.json'
        dump(enumeration, path)
        return path.read_text(encoding='utf-8')

    def test_is_json_with_a_trailing_newline(self, tmp_path):
        text = self._dumped(tmp_path)
        assert text.endswith('\n')
        assert json.loads(text)

    def test_every_count_entry_occupies_exactly_one_line(self, tmp_path):
        """The only property that makes the baseline mergeable.

        The same shape ``orchestrator/tests/test_merge_lane_ratchet.py::
        test_every_per_path_entry_occupies_exactly_one_line`` pins, for the same
        reason: concurrent branches editing disjoint keys must land in disjoint
        hunks.  For THIS file it is sharper still — D7 says the inline
        baseline's only legal diff is deletions, and a deletion is only legible
        as one when it is a whole line.

        The value is parsed OUT of the matched line rather than merely found to
        start there, which is what proves it did not continue onto the next.
        """
        text = self._dumped(tmp_path)
        lines = text.splitlines()
        for key, value in REPRESENTATIVE.counts.items():
            prefix = json.dumps(key) + ':'
            hits = [line for line in lines if line.lstrip().startswith(prefix)]
            assert len(hits) == 1, f'{key} is not on exactly one line'
            tail = hits[0].lstrip()[len(prefix) :].strip().rstrip(',')
            assert json.loads(tail) == value, f'{key} value spans lines'

    def test_count_keys_are_emitted_in_sorted_order(self, tmp_path):
        text = self._dumped(tmp_path)
        positions = [text.index(json.dumps(key) + ':') for key in sorted(REPRESENTATIVE.counts)]
        assert positions == sorted(positions)

    def test_stable_order_is_not_insertion_order(self, tmp_path):
        """Two dumps of equal Enumerations built from differently-ordered dicts.

        Byte-identical, so re-running the scanner over unchanged input produces
        no diff at all — the property that lets "the only legal diff is
        deletions" actually hold in practice.
        """
        forwards = Enumeration(counts={'aaa': 1, 'bbb': 2, 'ccc': 3}, params={'v': 1})
        backwards = Enumeration(counts={'ccc': 3, 'bbb': 2, 'aaa': 1}, params={'v': 1})
        dump(forwards, tmp_path / 'one.json')
        dump(backwards, tmp_path / 'two.json')
        assert (tmp_path / 'one.json').read_bytes() == (tmp_path / 'two.json').read_bytes()

    def test_rendering_is_idempotent(self, tmp_path):
        """``dump(load(dump(e)))`` reproduces the same bytes.

        Regenerating a baseline from a baseline is a no-op rather than a
        churned file full of manufactured conflicts.
        """
        first = tmp_path / 'first.json'
        second = tmp_path / 'second.json'
        dump(REPRESENTATIVE, first)
        dump(load(first), second)
        assert first.read_bytes() == second.read_bytes()

    def test_the_first_key_is_the_readme(self, tmp_path):
        """The rule has to live at the file a human actually opens.

        The baseline is unreviewable by content — its keys are 12-hex digests —
        so a reviewer's only handle on it is the paragraph at the top saying
        what a legal change to it looks like.
        """
        text = self._dumped(tmp_path)
        parsed = json.loads(text)
        assert next(iter(parsed)) == '_README'
        assert parsed['_README'] == BASELINE_README

    def test_the_readme_states_the_three_things_a_reader_needs(self, tmp_path):
        readme = json.loads(self._dumped(tmp_path))['_README'].lower()
        assert 'machine-generated' in readme
        assert 'deletion' in readme
        assert 'tighten' in readme
        assert 'hand-edit' in readme or 'by hand' in readme

    def test_an_inbound_readme_cannot_survive_a_round_trip(self, tmp_path):
        """load drops it, dump re-emits the constant — so an edited one is erased.

        Without this, someone could soften the rule in the file that publishes
        it and the next regeneration would preserve their edit.
        """
        path = tmp_path / 'baseline.json'
        dump(REPRESENTATIVE, path)
        tampered = json.loads(path.read_text(encoding='utf-8'))
        tampered['_README'] = 'feel free to regenerate this whenever a test is red'
        path.write_text(json.dumps(tampered), encoding='utf-8')
        dump(load(path), path)
        assert json.loads(path.read_text(encoding='utf-8'))['_README'] == BASELINE_README

    def test_carries_the_kernel_schema_version_and_the_scanner_params(self, tmp_path):
        """Two orthogonal reasons a file is incomparable, so two fields."""
        parsed = json.loads(self._dumped(tmp_path))
        assert parsed['schema_version'] == SCHEMA_VERSION
        assert parsed['params'] == {
            'key_version': 1,
            'algorithm': 'sha256',
            'strict': True,
            'kinds': ['noqa', 'nosec'],
        }


class TestDumpIsAtomic:
    """A truncated baseline is a WIDENED ratchet, so the write is all-or-nothing."""

    def test_leaves_no_temp_residue(self, tmp_path):
        dump(REPRESENTATIVE, tmp_path / 'baseline.json')
        assert [p.name for p in tmp_path.iterdir()] == ['baseline.json']

    def test_overwriting_leaves_exactly_one_file(self, tmp_path):
        path = tmp_path / 'baseline.json'
        dump(REPRESENTATIVE, path)
        dump(Enumeration(counts={'aaa': 1}, params=dict(REPRESENTATIVE.params)), path)
        assert [p.name for p in tmp_path.iterdir()] == ['baseline.json']
        assert dict(load(path).counts) == {'aaa': 1}

    def test_creates_the_parent_directory(self, tmp_path):
        path = tmp_path / 'nested' / 'deeper' / 'baseline.json'
        dump(REPRESENTATIVE, path)
        assert load(path) == REPRESENTATIVE


class TestLoadRefuses:
    """Absent, unreadable, malformed and wrong-schema all mean one thing.

    Every case raises :class:`BaselineUnusable`, every message names the path,
    and the exception carries the path as a structured attribute — at this
    boundary the path is what an operator needs, which is why these refusals
    are not the bare ValueError ``Enumeration.__post_init__`` raises.

    ONE ENUMERATION SHAPE FAULT HAS NO CASE HERE, and the omission is measured
    rather than forgotten: a counts entry keyed by a non-str cannot cross this
    boundary at all, because JSON object keys are strings by the format's
    definition and ``json.loads`` therefore hands ``load`` str keys whatever the
    file said.  ``Enumeration.__post_init__`` remains that check's only home,
    where it catches the in-process construction that can actually produce it.
    What IS worth pinning at the file boundary is the other half of the same
    stance — that a key which merely LOOKS like something else is still an
    opaque token — which
    :meth:`TestLoadDoesNotPoliceWhatAHumanAdded.test_a_numeric_looking_key_is_still_an_opaque_token`
    does.
    """

    @staticmethod
    def _refuses(path):
        with pytest.raises(BaselineUnusable) as excinfo:
            load(path)
        error = excinfo.value
        assert str(path) in str(error)
        assert str(error.path) == str(path)
        return str(error)

    @staticmethod
    def _written(tmp_path, payload):
        path = tmp_path / 'baseline.json'
        path.write_text(json.dumps(payload), encoding='utf-8')
        return path

    def test_a_missing_baseline_is_never_an_empty_baseline_pass(self, tmp_path):
        """INV-11, in this test's own words: absent is not clean.

        An empty baseline compares clean against EVERYTHING, so a load that
        answered "no file, have an empty one" would turn a deleted or
        never-seeded baseline into a permanently green gate — the silent
        fail-soft the invariant is named for.

        Boundary scenario 11's "baseline absent" path is the scanner's to
        detect by an existence check BEFORE calling load, so it can say
        something better than "unusable".  That is a nicer message, not a
        weaker contract: load itself refuses, and this asserts it.
        """
        self._refuses(tmp_path / 'never-written.json')

    def test_a_directory(self, tmp_path):
        (tmp_path / 'baseline.json').mkdir()
        self._refuses(tmp_path / 'baseline.json')

    def test_bytes_that_are_not_utf8(self, tmp_path):
        path = tmp_path / 'baseline.json'
        path.write_bytes(b'\xff\xfe not utf-8 at all')
        self._refuses(path)

    def test_text_that_is_not_json(self, tmp_path):
        path = tmp_path / 'baseline.json'
        path.write_text('{ this is not json', encoding='utf-8')
        self._refuses(path)

    @pytest.mark.parametrize('payload', [[], 'a string', 17, None])
    def test_a_top_level_that_is_not_an_object(self, tmp_path, payload):
        self._refuses(self._written(tmp_path, payload))

    @pytest.mark.parametrize('version', [None, 'one', 1.0, 999])
    def test_a_schema_version_this_build_does_not_read(self, tmp_path, version):
        """Naming BOTH the found and the expected version.

        The ``fused-memory/scripts/census_memory_metadata.py::
        load_coverage_history`` precedent: refusing to misread it as one.
        """
        payload = {'schema_version': version, 'params': {}, 'complete': True, 'counts': {}}
        if version is None:
            del payload['schema_version']
        message = self._refuses(self._written(tmp_path, payload))
        assert str(SCHEMA_VERSION) in message
        if version is not None:
            assert repr(version) in message

    @pytest.mark.parametrize('params', [None, [], 'x', 17])
    def test_params_absent_or_not_an_object(self, tmp_path, params):
        payload = {'schema_version': SCHEMA_VERSION, 'params': params, 'complete': True, 'counts': {}}
        if params is None:
            del payload['params']
        self._refuses(self._written(tmp_path, payload))

    @pytest.mark.parametrize(
        'counts',
        [
            None,  # absent
            [],  # not an object
            {'a': 'two'},  # non-int
            {'a': 0},  # zero
            {'a': -1},  # negative
        ],
    )
    def test_every_counts_shape_fault_surfaces_as_baseline_unusable(self, tmp_path, counts):
        """Not the bare ValueError __post_init__ raises.

        At this boundary the operator is holding a file, so the refusal has to
        name the file.  A ValueError describing a count in the abstract leaves
        them hunting for which of several baselines produced it.
        """
        payload = {'schema_version': SCHEMA_VERSION, 'params': {}, 'complete': True, 'counts': counts}
        if counts is None:
            del payload['counts']
        self._refuses(self._written(tmp_path, payload))

    @pytest.mark.parametrize('complete', [None, 'true', 1, []])
    def test_complete_absent_or_not_a_bool(self, tmp_path, complete):
        payload = {'schema_version': SCHEMA_VERSION, 'params': {}, 'complete': complete, 'counts': {}}
        if complete is None:
            del payload['complete']
        self._refuses(self._written(tmp_path, payload))

    @pytest.mark.parametrize('unreadable', ['x.py', [17], {'a': 1}])
    def test_unreadable_present_but_not_a_list_of_str(self, tmp_path, unreadable):
        self._refuses(
            self._written(
                tmp_path,
                {
                    'schema_version': SCHEMA_VERSION,
                    'params': {},
                    'complete': False,
                    'counts': {},
                    'unreadable': unreadable,
                },
            )
        )

    def test_the_completeness_contradiction_survives_the_file_boundary(self, tmp_path):
        """complete=true with a non-empty unreadable is refused off disk too.

        The in-process constructor refuses it; if the file boundary did not,
        the contradiction would simply be laundered through a write.
        """
        self._refuses(
            self._written(
                tmp_path,
                {
                    'schema_version': SCHEMA_VERSION,
                    'params': {},
                    'complete': True,
                    'counts': {},
                    'unreadable': ['pkg/a.py'],
                },
            )
        )

    def test_is_a_ratchet_error(self):
        assert issubclass(BaselineUnusable, RatchetError)


class TestLoadDoesNotPoliceWhatAHumanAdded:
    """A hand-added key still loads, and that is not a hole."""

    def test_an_added_key_loads(self, tmp_path):
        """The kernel does not police the file's CONTENT, only its SHAPE.

        There is no need to: an added key cannot widen anything, because the
        only baseline-producing function is :func:`tighten` and its result is a
        subset of the baseline by construction.  A key someone added by hand
        survives exactly until the next tighten, and buys them nothing in the
        meantime beyond permitting an entry the gate would otherwise name.
        Refusing it here would be a second, weaker enforcement point for a
        property the arithmetic already guarantees.
        """
        path = tmp_path / 'baseline.json'
        dump(REPRESENTATIVE, path)
        tampered = json.loads(path.read_text(encoding='utf-8'))
        tampered['counts']['deadbeef0000'] = 7
        path.write_text(json.dumps(tampered), encoding='utf-8')

        reloaded = load(path)
        assert reloaded.counts['deadbeef0000'] == 7

        scan = Enumeration(counts=dict(REPRESENTATIVE.counts), params=dict(REPRESENTATIVE.params))
        assert 'deadbeef0000' not in tighten(scan, reloaded)

    def test_a_numeric_looking_key_is_still_an_opaque_token(self, tmp_path):
        """The kernel never INTERPRETS a key, so a digit string is just a key.

        The counterpart to the note in :class:`TestLoadRefuses`: a non-str
        counts key cannot survive JSON, so what a file boundary can actually
        get wrong is reading a str key as the thing it resembles.  A baseline
        keyed by content digests will sooner or later hold one that is all
        digits, and it must round-trip as the same key rather than as an int.
        """
        path = tmp_path / 'baseline.json'
        numeric = Enumeration(counts={'1234567890ab': 3}, params=dict(PARAMS))
        dump(numeric, path)

        reloaded = load(path)
        assert reloaded == numeric
        assert list(reloaded.counts) == ['1234567890ab']
