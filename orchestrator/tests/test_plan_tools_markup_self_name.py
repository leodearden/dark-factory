"""plan-tools' read-repair against the SELF-NAME closer, end to end.

One concern, split out of ``test_plan_tools_markup_repair`` by size (heuristic
14) rather than by topic drift: a field mis-closed with its OWN name-echoing
tag. 212 of the 444 corrupted live entries have exactly that shape and the
blanket ``detect`` cannot see any of them, which makes it the one dialect whose
whole path — prefilter, repair, published pattern, and the truncation the
repair accepts — has to be pinned together.

Four classes, in the order a value travels:

* ``TestSelfNameCloserIsSeenByTheReadRepair`` — the prefilter widening (4696).
* ``TestOneSemanticsPerFieldOnBothArms`` — one leak, one published pattern,
  whichever arm of ``_repair_one_field`` reports it (5283).
* ``TestQuotedSiblingTagIsNeverTruncated`` — the false positive the widening
  must NOT create.
* ``TestTheSelfNameTruncationIsAcceptedAndReconstructible`` — the accepted
  loss, and the record that makes it reversible (5283).

## Sentinel-literal hazard — every specimen is BUILT, never written verbatim

A raw envelope literal here would corrupt the tool call that edits this file,
so every specimen is assembled from ``_markup_helpers``' builders and that
module's ``assert_no_raw_sentinels`` enforces it on this file's own bytes at
import. The plan document, the artifacts fixture and the refusal-memo isolation
come from ``_plan_markup_fixtures``, shared with the suite this split from —
the memo especially, since both suites report refusals on the same locator.
"""

from __future__ import annotations

import copy
import json

from _markup_helpers import (
    INVOKE_CLOSER,
    LT,
    assert_no_raw_sentinels,
    closer,
    param_opener,
)
from _plan_markup_fixtures import (
    corrupt_plan,
    isolate_the_refusal_memo,  # noqa: F401 — autouse; pytest resolves it by name
    on_disk,
    plan_artifacts,  # noqa: F401 — a fixture, requested by test parameter name
)
from shared.toolcall_markup import detect

from orchestrator.mcp import plan_tools

assert_no_raw_sentinels(__file__)


# ---------------------------------------------------------------------------
# Task 4696 — the SELF-NAME closer, invisible to the read-repair prefilter.
# ---------------------------------------------------------------------------

#: A rationale mis-closed with its OWN tag and NOTHING else: no invoke closer,
#: no parameter-open token. 296 real plan entries have exactly this shape.
_SELF_NAME_RATIONALE_PROSE = 'Both mechanisms partition rather than race.'
_SELF_NAME_RATIONALE = _SELF_NAME_RATIONALE_PROSE + closer('rationale')

#: The same on the second-largest victim, ``add_reuse_item.how`` (129 entries).
_SELF_NAME_HOW_PROSE = 'Reuse the declared table directly.'
_SELF_NAME_HOW = _SELF_NAME_HOW_PROSE + closer('how')

#: THE 4525 SHAPE, verbatim in structure: the field's own closer, then an
#: invoke closer, then the SAME parameter re-declared. repair() refuses it —
#: the invoke closer leads the tail so no candidate parses, and ``invoke`` does
#: not qualify — so the string must be left byte-identical and merely FLAGGED.
_UNREPAIRABLE_RATIONALE = (
    'See plan_tools.py:65-74.'
    + closer('rationale')
    + '\n'
    + INVOKE_CLOSER
    + '\n'
    + param_opener('rationale')
    + 'See decision text.'
)


def _seed_self_name_plan(artifacts) -> dict:
    """Write a plan whose ONLY corruption is two self-name closers."""
    plan = corrupt_plan()
    plan['design_decisions'][0]['rationale'] = _SELF_NAME_RATIONALE
    plan['reuse'][0]['how'] = _SELF_NAME_HOW
    artifacts.write_plan(copy.deepcopy(plan))
    return plan


class TestSelfNameCloserIsSeenByTheReadRepair:
    """Epsilon's lazy read-repair was gated on a predicate that could not see it.

    ``_carries_markup`` is the cheap prefilter that decides whether the repair
    pass runs at all, and it asked the param-free ``detect``. A plan whose only
    damage is a field mis-closed with its OWN name-echoing tag therefore looked
    CLEAN: the prefilter returned False, the deep copy never happened, and the
    corruption sat on disk untouched read after read — which is exactly why 296
    ``rationale`` and 129 ``how`` specimens were still there five weeks after
    the read-repair went live.

    The repairer behind that gate was correct for them the whole time. Unlike
    the middleware boundary, this site pays NOTHING to be fully schema-aware:
    the walk already yields the ``_PlanField`` record, so ``record.field`` and
    ``record.schema_params`` are both in hand from the DECLARED table.
    """

    def test_the_specimen_is_invisible_to_the_blanket_predicate(self):
        """Otherwise this class would be re-testing an already-caught dialect."""
        for value in (_SELF_NAME_RATIONALE, _SELF_NAME_HOW):
            assert INVOKE_CLOSER not in value
            assert LT + 'parameter ' not in value
            assert detect(value) is None

    def test_carries_markup_sees_the_self_name_closers(self):
        plan = corrupt_plan()
        plan['design_decisions'][0]['rationale'] = _SELF_NAME_RATIONALE
        plan['reuse'][0]['how'] = _SELF_NAME_HOW

        assert plan_tools._carries_markup(plan) is True

    def test_a_genuinely_clean_plan_is_still_not_copied(self):
        """The prefilter's whole purpose survives the widening."""
        assert plan_tools._carries_markup(corrupt_plan()) is False

    def test_both_fields_come_back_repaired(self, plan_artifacts):
        _seed_self_name_plan(plan_artifacts)

        plan, _facts = plan_tools._read_plan_repaired(plan_artifacts)

        assert plan['design_decisions'][0]['rationale'] == _SELF_NAME_RATIONALE_PROSE
        assert plan['reuse'][0]['how'] == _SELF_NAME_HOW_PROSE

    def test_the_repair_is_persisted_to_disk(self, plan_artifacts):
        _seed_self_name_plan(plan_artifacts)

        plan_tools._read_plan_repaired(plan_artifacts)

        on_disk = json.loads((plan_artifacts.root / 'plan.json').read_text(encoding='utf-8'))
        assert on_disk['design_decisions'][0]['rationale'] == _SELF_NAME_RATIONALE_PROSE
        assert on_disk['reuse'][0]['how'] == _SELF_NAME_HOW_PROSE

    def test_the_facts_locate_each_repair_by_collection_index_and_field(
        self, plan_artifacts
    ):
        _seed_self_name_plan(plan_artifacts)

        _plan, facts = plan_tools._read_plan_repaired(plan_artifacts)

        located = {
            (f['collection'], f['index'], f['field']): f
            for f in facts
        }
        assert set(located) == {
            ('design_decisions', 0, 'rationale'),
            ('reuse', 0, 'how'),
        }
        for (_collection, _index, field), fact in located.items():
            assert fact['outcome'] == 'repaired'
            assert fact['param'] == field
            assert fact['misclose'] == closer(field)
            assert fact['recovered_params'] == []

    def test_the_4525_shape_is_flagged_UNREPAIRABLE_and_never_guessed(
        self, plan_artifacts
    ):
        """The task's own specimen: refuse, flag, and change not one byte.

        Its tail leads with an invoke closer, so no candidate parses and the
        only other candidate name — ``invoke`` — does not qualify. There is
        nothing to delete and nothing to preserve separately: the tail's
        re-declaration is of the SAME parameter, INSIDE the one string, so the
        "fabricated sibling" is not a sibling key at all. Visible damage beats
        a guessed repair.
        """
        plan = corrupt_plan()
        plan['design_decisions'][0]['rationale'] = _UNREPAIRABLE_RATIONALE
        plan_artifacts.write_plan(copy.deepcopy(plan))
        before = (plan_artifacts.root / 'plan.json').read_bytes()

        repaired, facts = plan_tools._read_plan_repaired(plan_artifacts)

        flagged = [f for f in facts if f['field'] == 'rationale']
        assert [f['outcome'] for f in flagged] == ['unrepairable']
        assert repaired['design_decisions'][0]['rationale'] == _UNREPAIRABLE_RATIONALE
        assert sorted(repaired['design_decisions'][0]) == ['decision', 'rationale']
        assert (plan_artifacts.root / 'plan.json').read_bytes() == before, (
            'an unrepairable field must leave the file BYTE-IDENTICAL — a '
            'rewrite here would mean something was guessed'
        )

    def test_the_unrepairable_fact_names_the_tag_it_actually_saw(
        self, plan_artifacts
    ):
        """Not ``None``, and not the invoke closer that merely follows it.

        The diagnostic pattern on the refusal path came from the same blind
        predicate, so before this task it named whatever fixed literal happened
        to trail the leak — PRD section 2.2's original complaint, one layer in.
        """
        plan = corrupt_plan()
        plan['design_decisions'][0]['rationale'] = _UNREPAIRABLE_RATIONALE
        plan_artifacts.write_plan(copy.deepcopy(plan))

        _repaired, facts = plan_tools._read_plan_repaired(plan_artifacts)

        flagged = [f for f in facts if f['field'] == 'rationale']
        assert flagged[0]['pattern'] == closer('rationale')


#: A ``rationale`` whose prose legitimately ENDS by quoting a SIBLING field's
#: tag pair. ``decision`` is a real sibling parameter of ``add_design_decision``
#: and its closer is NOT one of the fixed ``ENVELOPE_LITERALS``, so it is a name
#: the task-4696 widening contributed and nothing else.
_QUOTED_SIBLING_PROSE = (
    'The harness emits ' + LT + 'decision>X' + closer('decision')
)


#: ``how`` closed with its own tag, then the swallowed ``where`` in the
#: CANONICAL dialect. A genuine repair — ``where`` really is recovered — so this
#: is the arm that HAS a ``Repair`` to read a pattern off. The self-name closer
#: leads and the fixed literal trails it, which is what makes the two
#: derivations disagree.
_MIXED_HOW = (
    _SELF_NAME_HOW_PROSE
    + closer('how')
    + '\n'
    + param_opener('where')
    + 'plan_tools'
    + closer('parameter')
)

#: The SAME leak head on the SAME field, in a shape ``repair`` refuses: the
#: tail leads with an invoke closer so no candidate parses. Its only purpose is
#: to drive the OTHER arm of ``_repair_one_field`` for one field.
_UNREPAIRABLE_HOW = (
    _SELF_NAME_HOW_PROSE
    + closer('how')
    + '\n'
    + INVOKE_CLOSER
    + '\n'
    + param_opener('how')
    + 'again'
)


class TestOneSemanticsPerFieldOnBothArms:
    """``_repair_one_field``'s two arms publish ``pattern`` from two sources.

    The UNREPAIRABLE arm publishes the gate's ``detect_for(value, field,
    schema_params)`` — the value the in-file comment above the gate already
    claims "names the tag actually seen instead of whatever fixed literal
    happens to trail the leak". The REPAIRED arm publishes ``result.pattern``,
    derived from the blanket, param-free ``detect``. So one fact stream carries
    two semantics, and which one a reader gets depends on whether the repair
    happened to succeed.

    MEASURED at HEAD on ``reuse[0].how``, before the fix::

        repaired arm      '\\x3cparameter name='   (the literal TRAILING it)
        unrepairable arm  the ``how`` closer       (the HEAD of the leak)

    Both specimens below leak from the same field with the same head, so the
    disagreement is not about the input.
    """

    def test_the_two_specimens_share_a_leak_head(self):
        """The premise. Without it these rows would compare two different leaks."""
        for value in (_MIXED_HOW, _UNREPAIRABLE_HOW):
            assert value.index(closer('how')) == len(_SELF_NAME_HOW_PROSE)
        assert detect(_MIXED_HOW) == LT + 'parameter name='
        assert detect(_UNREPAIRABLE_HOW) == INVOKE_CLOSER

    def test_the_repaired_arm_names_the_head_of_the_leak(self, plan_artifacts):
        """(3) The repaired arm, which is the one that goes red today."""
        plan = corrupt_plan()
        plan['reuse'][0]['how'] = _MIXED_HOW
        # A HOLE for the recovery to fill: an authored sibling counts as
        # supplied, and repair() refuses a tail that collides with one.
        plan['reuse'][0]['where'] = ''
        plan_artifacts.write_plan(copy.deepcopy(plan))

        repaired, facts = plan_tools._read_plan_repaired(plan_artifacts)

        (fact,) = facts
        assert fact['outcome'] == 'repaired'
        assert fact['field'] == 'how'
        assert fact['pattern'] == closer('how')
        assert fact['misclose'] == closer('how')
        # The diagnostic changed; the repair did not.
        assert repaired['reuse'][0]['how'] == _SELF_NAME_HOW_PROSE
        assert repaired['reuse'][0]['where'] == 'plan_tools'
        assert fact['recovered_params'] == ['where']

    def test_both_arms_name_the_same_literal_for_the_same_field(
        self, plan_artifacts
    ):
        """One field, one semantics — whichever way the repair goes.

        This is the property the fix delivers BY CONSTRUCTION rather than by
        convention: both arms become the same expression over the same
        ``(value, param, schema_params)`` triple, so they cannot drift apart
        again without someone editing one of them on purpose.
        """
        patterns = {}
        for outcome, value, where in (
            ('repaired', _MIXED_HOW, ''),
            ('unrepairable', _UNREPAIRABLE_HOW, 'plan_tools'),
        ):
            plan = corrupt_plan()
            plan['reuse'][0]['how'] = value
            plan['reuse'][0]['where'] = where
            plan_artifacts.write_plan(copy.deepcopy(plan))

            _plan, facts = plan_tools._read_plan_repaired(plan_artifacts)

            (fact,) = facts
            assert fact['outcome'] == outcome, 'the two arms really were driven'
            patterns[outcome] = fact['pattern']

        assert patterns['repaired'] == patterns['unrepairable'] == closer('how')


class TestQuotedSiblingTagIsNeverTruncated:
    """A plan that TALKS ABOUT the markup must not be rewritten by the reader.

    The widening at ``_carries_markup`` / ``_repair_one_field`` added every
    sibling ``record.schema_params`` name to the gate, and ``repair`` accepts an
    EMPTY tail — a candidate closer at end-of-string recovers ``{}`` and still
    returns ``clean_value = value[:candidate.start()]``. Composed, a rationale
    ending in ``\x3c/decision>`` was TRUNCATED, reported ``repaired``, and
    persisted atomically by ``_read_plan_repaired``, with nothing left to
    surface the loss. Pre-4696 the blanket ``detect`` returned None and the
    value was left alone, so the loss surface was introduced by that change.

    This is not hypothetical in a repo whose plans discuss tool-call markup —
    the containment PRD itself quotes these tags. And the widening bought
    nothing measured: the PRD's 2026-08-25 census puts the CROSS-FIELD
    population at ZERO (212/212 invisible specimens are self-name).

    The fix is at the shared ``repair`` chokepoint, so the sweep's own
    sibling-key widening is closed by the same mechanism (INV-5).
    """

    def test_the_specimen_was_invisible_before_the_widening(self):
        """Otherwise this class would be pinning pre-existing behaviour."""
        assert detect(_QUOTED_SIBLING_PROSE) is None

    def test_the_value_comes_back_byte_identical(self, plan_artifacts):
        plan = corrupt_plan()
        plan['design_decisions'][0]['rationale'] = _QUOTED_SIBLING_PROSE
        plan_artifacts.write_plan(copy.deepcopy(plan))

        repaired, _facts = plan_tools._read_plan_repaired(plan_artifacts)

        assert repaired['design_decisions'][0]['rationale'] == _QUOTED_SIBLING_PROSE

    def test_no_repaired_fact_is_emitted_for_it(self, plan_artifacts):
        """``repaired`` would be an outright false report: nothing was
        recovered, so the only change would have been text DESTROYED."""
        plan = corrupt_plan()
        plan['design_decisions'][0]['rationale'] = _QUOTED_SIBLING_PROSE
        plan_artifacts.write_plan(copy.deepcopy(plan))

        _repaired, facts = plan_tools._read_plan_repaired(plan_artifacts)

        flagged = [f for f in facts if f['field'] == 'rationale']
        assert [f['outcome'] for f in flagged] == ['unrepairable']
        assert flagged[0]['recovered_params'] == []

    def test_the_file_is_never_rewritten(self, plan_artifacts):
        """The all-refusals branch of ``_read_plan_repaired`` must hold, or the
        truncation would be durable and the mtime would churn under watchers."""
        plan = corrupt_plan()
        plan['design_decisions'][0]['rationale'] = _QUOTED_SIBLING_PROSE
        plan_artifacts.write_plan(copy.deepcopy(plan))
        before = (plan_artifacts.root / 'plan.json').read_bytes()

        plan_tools._read_plan_repaired(plan_artifacts)

        assert (plan_artifacts.root / 'plan.json').read_bytes() == before

    def test_a_self_name_closer_in_the_same_plan_is_still_repaired(
        self, plan_artifacts
    ):
        """The guard is scoped to ``name != param``, so the dialect this task
        exists to fix is untouched — the fix narrows REPAIR, not DETECTION."""
        plan = corrupt_plan()
        plan['design_decisions'][0]['rationale'] = _QUOTED_SIBLING_PROSE
        plan['reuse'][0]['how'] = _SELF_NAME_HOW
        plan_artifacts.write_plan(copy.deepcopy(plan))

        repaired, _facts = plan_tools._read_plan_repaired(plan_artifacts)

        assert repaired['reuse'][0]['how'] == _SELF_NAME_HOW_PROSE
        assert repaired['design_decisions'][0]['rationale'] == _QUOTED_SIBLING_PROSE


# ---------------------------------------------------------------------------
# task 5283 — the ACCEPTED self-name truncation, pinned as a deliberate choice.
# ---------------------------------------------------------------------------

#: A rationale that ends with its OWN closer and then only WHITESPACE. At read
#: time this is indistinguishable from authored prose whose last words quote the
#: field's own tag; it is truncated regardless, because that is the shape the
#: 2026-08-25 census found 212 times and the cross-field shape zero times.
_ACCEPTED_TRUNCATION = _SELF_NAME_RATIONALE_PROSE + closer('rationale') + '\n  '

#: The same string with ONE name changed — a SIBLING parameter's closer instead
#: of the field's own. Same field, same prose, same trailing whitespace, and the
#: opposite outcome. The pair is what makes the asymmetry visible in the tests.
_REFUSED_TRUNCATION = _SELF_NAME_RATIONALE_PROSE + closer('decision') + '\n  '


class TestTheSelfNameTruncationIsAcceptedAndReconstructible:
    """Prose ending in its own closer IS truncated on disk, and that is chosen.

    ``repair``'s quotation guard accepts an empty tail when the closer echoes
    the parameter being repaired (``name == param``) and refuses it otherwise,
    so a ``rationale`` ending in its own tag loses that tag permanently while a
    ``rationale`` ending in a sibling's keeps every byte. Nothing pinned the
    accepted half, and an accepted trade-off with no test reads exactly like an
    accident — which is the whole of the reviewer's point.

    WHY IT IS ACCEPTABLE, and why the record matters more than the truncation:
    the deleted span is RECONSTRUCTIBLE from the emitted fact alone, because
    ``misclose`` names the tag that was removed. A reader who disagrees with the
    repair can put the value back from the record; the loss is therefore
    reportable rather than silent, which is the bar this surface sets.
    """

    @staticmethod
    def _read_back(artifacts, value: str):
        """Seed ``design_decisions[0].rationale`` = *value*, then read-repair.

        Returns ``(on_disk_value, facts, bytes_before)``. The value as
        PERSISTED, not merely as returned, since durability is the half of the
        contract a returned dict cannot show; the facts because on this surface
        the RECORD is the other half — every row below reads one or the other,
        and several read both.
        """
        plan = corrupt_plan()
        plan['design_decisions'][0]['rationale'] = value
        artifacts.write_plan(copy.deepcopy(plan))
        before = (artifacts.root / 'plan.json').read_bytes()

        _plan, facts = plan_tools._read_plan_repaired(artifacts)

        return (
            on_disk(artifacts)['design_decisions'][0]['rationale'],
            facts,
            before,
        )

    def test_the_field_is_truncated_to_its_prose_and_persisted_that_way(
        self, plan_artifacts
    ):
        on_disk, _facts, _before = self._read_back(
            plan_artifacts, _ACCEPTED_TRUNCATION
        )

        assert on_disk == _SELF_NAME_RATIONALE_PROSE

    def test_the_trailing_whitespace_goes_WITH_the_tag(self, plan_artifacts):
        """The full extent of the deletion, not just the visible part.

        ``clean_value`` is the slice BEFORE the closer, so everything after it
        goes too. Stating it here keeps the contract from being read as "the
        tag is stripped" when it is "the value is cut at the tag".
        """
        on_disk, _facts, _before = self._read_back(
            plan_artifacts, _ACCEPTED_TRUNCATION
        )

        assert not on_disk.endswith(('\n', ' '))
        assert _ACCEPTED_TRUNCATION[len(on_disk):] == closer('rationale') + '\n  '

    def test_the_fact_is_a_repair_that_recovered_NOTHING(self, plan_artifacts):
        """PRD boundary row B4's last-parameter shape: nothing was absorbed."""
        _on_disk_value, facts, _before = self._read_back(
            plan_artifacts, _ACCEPTED_TRUNCATION
        )

        (fact,) = facts
        assert fact['outcome'] == 'repaired'
        assert fact['recovered_params'] == []
        assert fact['declined_params'] == []

    def test_the_deleted_span_is_reconstructible_from_the_fact_alone(
        self, plan_artifacts
    ):
        """``misclose`` names the tag, so the cut is reversible from the record.

        Reversible up to the trailing whitespace, which the fact does not carry
        and which holds no authored content — every CHARACTER OF TEXT the
        truncation removed is named by ``misclose``.
        """
        on_disk, facts, _before = self._read_back(
            plan_artifacts, _ACCEPTED_TRUNCATION
        )

        (fact,) = facts
        assert fact['misclose'] == closer('rationale')
        assert on_disk + fact['misclose'] == _ACCEPTED_TRUNCATION.rstrip()

    def test_the_CROSS_FIELD_counterpart_is_refused_byte_identically(
        self, plan_artifacts
    ):
        """One name apart from the row above, and the opposite disposition.

        plan-tools repairs a value it received as a NAMED PARAMETER of a known
        tool, so the field's own closer is evidence about that parameter. A
        sibling's closer is not, and the census puts that population at zero —
        so the same empty tail is refused, reported, and left alone.
        """
        on_disk, _facts, before = self._read_back(
            plan_artifacts, _REFUSED_TRUNCATION
        )

        assert on_disk == _REFUSED_TRUNCATION
        assert (plan_artifacts.root / 'plan.json').read_bytes() == before

    def test_the_refusal_is_reported_rather_than_swallowed(self, plan_artifacts):
        """Byte-identical must not mean invisible, or the asymmetry hides."""
        _on_disk_value, facts, _before = self._read_back(
            plan_artifacts, _REFUSED_TRUNCATION
        )

        (fact,) = facts
        assert fact['outcome'] == 'unrepairable'
        assert fact['misclose'] is None
        assert fact['pattern'] == closer('decision')
