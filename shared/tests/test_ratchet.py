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

    @pytest.mark.parametrize('value', [{'a': 1}, {1, 2}, object()])
    def test_a_params_value_that_is_not_json(self, value):
        """The params block is written to and read from a JSON file verbatim."""
        assert 'p' in self._message(params={'p': value})

    def test_a_tuple_of_non_scalars_in_params(self):
        assert 'p' in self._message(params={'p': ({'a': 1},)})

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
