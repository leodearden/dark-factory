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
"""
from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

from shared.ratchet import Enumeration

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
