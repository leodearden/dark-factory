"""Tests for the consolidation-gate brief + closure predicate (task 3112).

Covers the two defects the task owns: the gate-filing instruction that
prescribed no end-state shape (Defect 1 — :func:`render_end_state_brief` /
:func:`render_consolidation_gate_section`), and the absence of any mechanical
refusal to close a gate task over a malformed cluster (Defect 2 —
:func:`evaluate_closure`).

Assertions pin runtime return values (the verdict dataclass, the builder's
returned dict), resolvable IDENTIFIERS that appear in the rendered text
(symbol names, metadata keys, tool names), and COMPOSITION identity
(``render_end_state_brief() in render_consolidation_gate_section()``).  They
never pin English phrasing: a bare word or a human-readable heading is not a
contract, a semantics-preserving reword breaks it for nothing, and a tighter
regex over prose is a worse pin rather than a better one.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path

import pytest

from fused_memory import memory_metadata
from fused_memory.middleware.deterministic_task_guard import deterministic_task_error
from fused_memory.middleware.execution_class_guard import execution_class_error
from fused_memory.reconciliation import consolidation_gate
from fused_memory.reconciliation.consolidation_gate import (
    GATE_METADATA_KEY,
    ClosureVerdict,
    build_consolidation_gate_task,
    evaluate_closure,
    render_end_state_brief,
)


class TestRenderEndStateBrief:
    """Defect 1's payload: the end-state shape a filed gate must carry.

    Identifier assertions only — the prose is free to change, the shape it
    prescribes is not.
    """

    def test_returns_non_empty_str(self):
        brief = render_end_state_brief()
        assert isinstance(brief, str)
        assert brief.strip()

    def test_mandates_the_option_c_shape(self):
        """N short single-claim peers sharing one topic, exactly one canonical."""
        brief = render_end_state_brief()
        assert 'metadata.topic' in brief
        assert 'metadata.canonical' in brief

    def test_points_at_the_landed_op(self):
        """The brief and the executable op cannot prescribe different end states."""
        brief = render_end_state_brief()
        assert 'consolidate_memories' in brief

    def test_public_surface_is_exported(self):
        assert 'render_end_state_brief' in consolidation_gate.__all__
        assert 'GATE_METADATA_KEY' in consolidation_gate.__all__

    def test_gate_key_is_composed_from_the_shared_experimental_prefix(self):
        """Composed, never re-spelled — the CONTESTED_METADATA_KEY precedent."""
        assert GATE_METADATA_KEY.startswith(memory_metadata.EXPERIMENTAL_KEY_PREFIX)
        assert GATE_METADATA_KEY != memory_metadata.EXPERIMENTAL_KEY_PREFIX


# --------------------------------------------------------------------------- #
# Closure-predicate fixtures.  Payload shape mirrors what
# ``MemoryService.get_memories_by_metadata`` returns: {'id', 'created_at',
# 'metadata'} (see services/topic_anchor.py::select_canonical_payload, which
# documents the same shape).
# --------------------------------------------------------------------------- #

_TOPIC = 'recon-consolidation-gate-demo'


def _member(mid, *, canonical=None, topic=_TOPIC, supersedes=None, **extra_meta):
    """One scrolled topic member."""
    meta = dict(extra_meta)
    if topic is not None:
        meta['topic'] = topic
    if canonical is not None:
        meta['canonical'] = canonical
    if supersedes is not None:
        meta['supersedes'] = supersedes
    return {'id': mid, 'created_at': '2026-08-24T00:00:00+00:00', 'metadata': meta}


def _uuid(n):
    return f'00000000-0000-4000-8000-{n:012d}'


def _closure(members, gate_block=None, **kwargs):
    """Call ``evaluate_closure`` with a complete, available scroll by default.

    The production signature deliberately requires the completeness arguments
    (a caller must not be able to forget to disclose truncation); this helper
    supplies them so the membership tests stay about membership.
    """
    kwargs.setdefault('scroll_total', len(members))
    kwargs.setdefault('scroll_truncated', False)
    kwargs.setdefault('scroll_available', True)
    return evaluate_closure(
        gate_block if gate_block is not None else {'topic': _TOPIC},
        members=members,
        **kwargs,
    )


def _codes(verdict):
    return [r['code'] for r in verdict.reasons]


def _well_formed_cluster(n):
    """The canonical Option-C end state: N peers, exactly one canonical."""
    members = [_member(_uuid(1), canonical=True)]
    members += [_member(_uuid(i)) for i in range(2, n + 1)]
    return members


class TestClosureMembership:
    """Fix (1) — the membership test, and the reason the whole task is correct.

    Under PRD §3 Option C the end state IS N same-topic peers under one
    canonical, so a live same-topic peer is a legitimate member and can never
    be a refusal on membership grounds.  The prior plan's predicate flagged
    every such peer, which made every CORRECTLY executed consolidation
    permanently uncloseable; these tests go red if that is reintroduced.
    """

    @pytest.mark.parametrize('n', [1, 2, 3, 12])
    def test_well_formed_option_c_cluster_closes(self, n):
        verdict = _closure(_well_formed_cluster(n))
        assert verdict.closed is True
        assert verdict.exit_code == 0
        assert list(verdict.reasons) == []
        assert verdict.topic == _TOPIC

    def test_peer_count_alone_never_refuses(self):
        """A surviving peer that is neither canonical nor superseded is FINE."""
        peer = _uuid(7)
        members = [
            _member(_uuid(1), canonical=True, supersedes=[_uuid(99)]),
            _member(peer),
        ]
        verdict = _closure(members)
        assert verdict.closed is True
        # The peer must not be named anywhere in the verdict — not as a defect,
        # not as a warning.  This is the exact mutual exclusion that made the
        # prior plan's predicate unshippable.
        assert peer not in repr(verdict.reasons)

    def test_zero_canonicals_refuses(self):
        verdict = _closure([_member(_uuid(1)), _member(_uuid(2))])
        assert verdict.closed is False
        assert verdict.exit_code != 0
        assert 'no_canonical' in _codes(verdict)

    def test_multiple_canonicals_refuses_naming_both(self):
        """3198's per-topic uniqueness ships warn-mode-first, so this is
        REACHABLE through ordinary writes, not theoretical."""
        members = [
            _member(_uuid(1), canonical=True),
            _member(_uuid(2), canonical=True),
            _member(_uuid(3)),
        ]
        verdict = _closure(members)
        assert verdict.closed is False
        assert 'multiple_canonicals' in _codes(verdict)
        named = [r for r in verdict.reasons if r['code'] == 'multiple_canonicals']
        assert set(named[0]['ids']) == {_uuid(1), _uuid(2)}

    def test_canonical_is_strict_bool_not_truthy(self):
        """``meta.get('canonical') is True`` — an int 1 is NOT canonical.

        Mirrors services/topic_anchor.py::select_canonical_payload's identity
        check, so the closure predicate, the write-side uniqueness rule and the
        read-side anchor cannot disagree about what 'canonical' means.
        """
        verdict = _closure([_member(_uuid(1), canonical=1), _member(_uuid(2))])
        assert verdict.closed is False
        assert 'no_canonical' in _codes(verdict)

    def test_truthy_canonical_does_not_join_the_strict_one(self):
        """One strict True plus one truthy-but-not-True is ONE canonical."""
        members = [_member(_uuid(1), canonical=True), _member(_uuid(2), canonical=1)]
        verdict = _closure(members)
        assert verdict.closed is True


class TestClosureVerdictContract:
    def test_verdict_is_a_frozen_dataclass(self):
        verdict = _closure(_well_formed_cluster(2))
        assert dataclasses.is_dataclass(verdict)
        assert dataclasses.fields(ClosureVerdict)
        with pytest.raises(dataclasses.FrozenInstanceError):
            # setattr, not a direct attribute assignment, so this stays pyright-clean
            # (a direct assignment on a frozen dataclass is reportAttributeAccessIssue).
            setattr(verdict, 'closed', False)  # noqa: B010

    def test_closed_and_exit_code_can_never_disagree(self):
        closed = _closure(_well_formed_cluster(3))
        assert closed.closed is True and closed.exit_code == 0
        refused = _closure([_member(_uuid(1))])
        assert refused.closed is False and refused.exit_code != 0

    def test_message_is_non_empty_on_both_arms(self):
        assert _closure(_well_formed_cluster(2)).message.strip()
        assert _closure([_member(_uuid(1))]).message.strip()


class TestClosureSupersedes:
    """Fix (2) — an id claimed absorbed must actually be gone."""

    def test_scalar_legacy_supersedes_is_one_member_not_36_characters(self):
        """The corpus carries 81 records with a SCALAR supersedes.

        Iterating that string character-by-character would manufacture 36
        bogus ids and refuse the gate systematically — the exact false
        refusal the prior plan would have produced.
        """
        absorbed = _uuid(50)
        scalar = _closure([_member(_uuid(1), canonical=True, supersedes=absorbed)])
        listed = _closure([_member(_uuid(1), canonical=True, supersedes=[absorbed])])
        assert scalar.closed is True
        # Same input, two spellings — identical verdict, not merely both closed.
        assert scalar == listed
        # No character-shaped id may appear anywhere in the verdict.
        for reason in scalar.reasons:
            assert all(len(i) != 1 for i in reason['ids'])

    def test_scalar_legacy_supersedes_still_detects_a_live_member(self):
        """The scalar shape must go through the SAME classification, not be
        waved through as 'unparseable so fine'."""
        absorbed = _uuid(2)
        members = [
            _member(_uuid(1), canonical=True, supersedes=absorbed),
            _member(absorbed),
        ]
        verdict = _closure(members)
        assert verdict.closed is False
        assert 'absorbed_member_still_live' in _codes(verdict)

    @pytest.mark.parametrize('value', [None, []])
    def test_nothing_absorbed_is_accepted(self, value):
        members = [_member(_uuid(1), canonical=True)]
        if value is not None:
            members[0]['metadata']['supersedes'] = value
        verdict = _closure(members)
        assert verdict.closed is True

    def test_absorbed_member_still_live_names_the_id(self):
        """The curator claimed it was deleted; the live scroll says otherwise."""
        absorbed = _uuid(3)
        members = [
            _member(_uuid(1), canonical=True, supersedes=[absorbed, _uuid(90)]),
            _member(_uuid(2)),
            _member(absorbed),
        ]
        verdict = _closure(members)
        assert verdict.closed is False
        named = [r for r in verdict.reasons if r['code'] == 'absorbed_member_still_live']
        assert named and named[0]['ids'] == [absorbed]
        # The correctly-folded member contributes nothing.
        assert _uuid(90) not in repr(verdict.reasons)

    def test_absorbed_member_comparison_is_case_insensitive(self):
        """A casing difference between the scroll row and the stored metadata
        must not manufacture a false absorbed_member_still_live, nor hide a
        real one: is_full_uuid tolerates case because casing is a rendering
        choice, not a different identifier."""
        absorbed = _uuid(4)
        members = [
            _member(_uuid(1), canonical=True, supersedes=[absorbed.upper()]),
            _member(absorbed),
        ]
        verdict = _closure(members)
        assert 'absorbed_member_still_live' in _codes(verdict)

    @pytest.mark.parametrize(
        'malformed',
        ['deadbeef', 12345, None, '00000000000040008000000000000001'],
        ids=['short_hex', 'int', 'none', 'undashed_32'],
    )
    def test_malformed_supersedes_member_is_named_never_raised(self, malformed):
        """normalize_supersedes deliberately PRESERVES malformed members (the
        census counts 3 short-hex and 8 non-string live) so a validator can
        reject them by name.  A raising predicate would permanently block
        exactly those gates."""
        members = [_member(_uuid(1), canonical=True, supersedes=[malformed])]
        verdict = _closure(members)
        assert verdict.closed is False
        named = [
            r for r in verdict.reasons if r['code'] == 'malformed_supersedes_member'
        ]
        assert named and named[0]['ids'] == [str(malformed)]

    def test_every_offender_is_collected_not_short_circuited(self):
        live = _uuid(5)
        members = [
            _member(_uuid(1), canonical=True, supersedes=[live, 'deadbeef']),
            _member(live),
        ]
        codes = _codes(_closure(members))
        assert 'absorbed_member_still_live' in codes
        assert 'malformed_supersedes_member' in codes

    def test_uses_the_shared_supersedes_parser(self, monkeypatch):
        """INV-5 single-home lock — there is never a SECOND supersedes parser.

        BEHAVIORAL, following
        ``test_targeted.py::test_targeted_uses_the_shared_supersedes_parser``:
        an identity assertion is near-tautological (a plain ``from ... import``
        always yields it) and still passes on the realistic violation, which is
        inlining ``isinstance(v, str)`` shape logic at the call site while
        leaving the now-unused import in place.  ``normalize_supersedes``' own
        docstring names this closure predicate as one of its two designated
        readers.
        """
        seen = []

        def _sentinel_parser(value):
            seen.append(value)
            return ['deadbeef']

        monkeypatch.setattr(
            consolidation_gate, 'normalize_supersedes', _sentinel_parser
        )
        # Under the REAL parser this is 'nothing absorbed' and closes; the
        # refusal can only come from the patched shared name being called.
        verdict = _closure([_member(_uuid(1), canonical=True)])
        assert seen == [None], 'the raw metadata value must reach the shared parser'
        assert 'malformed_supersedes_member' in _codes(verdict)

    def test_only_the_canonical_supersedes_is_read(self):
        """A non-canonical peer's stale supersedes is not the cluster's claim."""
        live = _uuid(6)
        members = [
            _member(_uuid(1), canonical=True),
            _member(_uuid(2), supersedes=[live]),
            _member(live),
        ]
        assert _closure(members).closed is True


def _waiver(mid, note='curator judged this entry a legitimate separate claim'):
    return {
        'id': mid,
        'note': note,
        'recorded_at': '2026-08-24T12:00:00+00:00',
        'recorded_by': 'recon-stage-2',
    }


class TestClosureScrollCompleteness:
    """Fix (4) — the false-PASS killer.

    ``get_memories_by_metadata`` is a SINGLE Qdrant scroll capped at ``limit``
    whose ``_next_offset`` is DISCARDED, and it propagates a read
    ``TimeoutError`` rather than returning ``[]``.  So a partial or absent view
    is reachable, and a predicate whose entire job is refuting a false closure
    claim must never read one as closed (INV-3).
    """

    def test_truncated_scroll_never_closes(self):
        verdict = _closure(
            _well_formed_cluster(3), scroll_truncated=True, scroll_total=3
        )
        assert verdict.closed is False
        assert 'scroll_incomplete' in _codes(verdict)

    def test_truncation_reason_discloses_how_much_was_seen(self):
        """A human must be able to see how much of the cluster was unseen."""
        verdict = _closure(
            _well_formed_cluster(3), scroll_truncated=True, scroll_total=3
        )
        reason = next(r for r in verdict.reasons if r['code'] == 'scroll_incomplete')
        assert reason['scroll_total'] == 3

    def test_unavailable_scroll_never_closes(self):
        verdict = _closure(
            _well_formed_cluster(2), scroll_available=False, scroll_total=None
        )
        assert verdict.closed is False
        assert 'scroll_unavailable' in _codes(verdict)

    def test_empty_and_unavailable_is_never_nothing_left(self):
        """The three-way distinction build_consolidation_result already draws:
        genuinely-empty and never-answered are different outcomes."""
        verdict = _closure([], scroll_available=False, scroll_total=None)
        assert verdict.closed is False
        assert 'scroll_unavailable' in _codes(verdict)
        # Nothing was SEEN, so nothing about the cluster's content may be
        # asserted — an absence-based accusation drawn on no view at all
        # would be a fabrication.
        assert 'no_canonical' not in _codes(verdict)

    def test_presence_based_defects_survive_truncation(self):
        """Seeing two canonicals PROVES two canonicals; more rows can only add.

        Truncation makes absence unprovable, not presence.
        """
        members = [
            _member(_uuid(1), canonical=True),
            _member(_uuid(2), canonical=True),
        ]
        codes = _codes(_closure(members, scroll_truncated=True, scroll_total=2))
        assert 'scroll_incomplete' in codes
        assert 'multiple_canonicals' in codes

    def test_absence_based_defects_are_suppressed_on_a_partial_view(self):
        """A canonical beyond the cap is indistinguishable from no canonical,
        so `no_canonical` on a truncated view would name a defect that may not
        exist.  The gate still refuses — on the honest reason."""
        codes = _codes(
            _closure([_member(_uuid(1))], scroll_truncated=True, scroll_total=1)
        )
        assert 'scroll_incomplete' in codes
        assert 'no_canonical' not in codes


class TestClosureAuditedEscape:
    """Fix (5) — a sanctioned, audited exit, not a rubber stamp.

    Gates routinely sit for days (the Stage-1 stale-gate threshold is 48h), so
    without an exit a late write against the topic makes a gate permanently
    uncloseable.
    """

    def _live_but_claimed_absorbed(self):
        absorbed = _uuid(3)
        return absorbed, [
            _member(_uuid(1), canonical=True, supersedes=[absorbed]),
            _member(absorbed),
        ]

    def test_audited_waiver_suppresses_the_refusal(self):
        absorbed, members = self._live_but_claimed_absorbed()
        assert _closure(members).closed is False  # baseline: refuses
        verdict = _closure(
            members,
            gate_block={
                'topic': _TOPIC,
                'considered_and_kept': [_waiver(absorbed)],
            },
        )
        assert verdict.closed is True
        assert list(verdict.reasons) == []

    def test_waiver_is_echoed_so_it_is_visible_not_silent(self):
        absorbed, members = self._live_but_claimed_absorbed()
        verdict = _closure(
            members,
            gate_block={
                'topic': _TOPIC,
                'considered_and_kept': [_waiver(absorbed)],
            },
        )
        assert [w['id'] for w in verdict.waived] == [absorbed]
        assert verdict.waived[0]['note']

    @pytest.mark.parametrize('note', ['', '   ', None])
    def test_unaudited_waiver_waives_nothing(self, note):
        """A bare id list must not be a rubber stamp."""
        absorbed, members = self._live_but_claimed_absorbed()
        entry = _waiver(absorbed, note=note)
        if note is None:
            del entry['note']
        verdict = _closure(
            members,
            gate_block={'topic': _TOPIC, 'considered_and_kept': [entry]},
        )
        assert verdict.closed is False
        codes = _codes(verdict)
        assert 'unaudited_waiver' in codes
        # ...and the refusal it tried to waive is still standing.
        assert 'absorbed_member_still_live' in codes
        assert list(verdict.waived) == []

    def test_stale_waiver_is_reported_not_silently_ignored(self):
        """A waiver naming nothing live no longer describes reality."""
        verdict = _closure(
            _well_formed_cluster(2),
            gate_block={
                'topic': _TOPIC,
                'considered_and_kept': [_waiver(_uuid(77))],
            },
        )
        assert verdict.closed is False
        stale = [r for r in verdict.reasons if r['code'] == 'stale_waiver']
        assert stale and stale[0]['ids'] == [_uuid(77)]

    def test_a_waiver_cannot_wave_through_a_cluster_shape_defect(self):
        """The escape covers "this live entry is fine", not "ignore the gate".

        Two canonicals is a shape defect to FIX, and naming one of them in
        considered_and_kept must not close the gate.
        """
        members = [
            _member(_uuid(1), canonical=True),
            _member(_uuid(2), canonical=True),
        ]
        verdict = _closure(
            members,
            gate_block={
                'topic': _TOPIC,
                'considered_and_kept': [_waiver(_uuid(2))],
            },
        )
        assert verdict.closed is False
        assert 'multiple_canonicals' in _codes(verdict)

    def test_a_waiver_cannot_wave_through_an_incomplete_view(self):
        """Completeness is unconditional: you cannot waive not having looked."""
        absorbed, members = self._live_but_claimed_absorbed()
        verdict = _closure(
            members,
            gate_block={
                'topic': _TOPIC,
                'considered_and_kept': [_waiver(absorbed)],
            },
            scroll_truncated=True,
            scroll_total=2,
        )
        assert verdict.closed is False
        assert 'scroll_incomplete' in _codes(verdict)


class TestBuildConsolidationGateTask:
    """The recon-side filing convention, modelled on PredicateContradictionTask."""

    def test_spec_is_a_frozen_dataclass(self):
        spec = build_consolidation_gate_task(topic=_TOPIC)
        assert dataclasses.is_dataclass(spec)
        with pytest.raises(dataclasses.FrozenInstanceError):
            # setattr, not a direct attribute assignment, so this stays pyright-clean
            # (a direct assignment on a frozen dataclass is reportAttributeAccessIssue).
            setattr(spec, 'title', 'x')  # noqa: B010

    def test_as_submit_task_kwargs_is_exactly_the_submit_task_key_set(self):
        """Splattable straight into submit_task, matching the sibling builder."""
        kwargs = build_consolidation_gate_task(topic=_TOPIC).as_submit_task_kwargs()
        assert set(kwargs) == {
            'title',
            'description',
            'priority',
            'task_kind',
            'metadata',
        }

    def test_metadata_carries_the_topic_as_the_working_key(self):
        meta = build_consolidation_gate_task(topic=_TOPIC).metadata
        assert meta[GATE_METADATA_KEY]['topic'] == _TOPIC

    def test_metadata_routes_through_the_pure_gate_path(self):
        """execution_class + operational_mode are what make this a human gate,
        and what the step-12 seam triggers on."""
        meta = build_consolidation_gate_task(topic=_TOPIC).metadata
        assert meta['execution_class'] == 'operational'
        assert meta['operational_mode'] == 'gate'

    def test_the_submission_satisfies_the_deterministic_pure_gate_invariant(self):
        """A deterministic task with no before_done MUST always escalate, or
        deterministic_task_guard rejects it as an ill-formed no-op."""
        spec = build_consolidation_gate_task(topic=_TOPIC)
        assert deterministic_task_error(
            spec.task_kind, spec.metadata, str(Path.cwd())
        ) is None
        assert 'before_done' not in spec.metadata

    def test_execution_class_is_accepted_for_a_recon_stage_filer(self):
        """Enforcement fires only for a 'recon-stage-*' agent_id, which is
        exactly who files these gates."""
        spec = build_consolidation_gate_task(topic=_TOPIC)
        assert (
            execution_class_error(spec.metadata, 'recon-stage-2', str(Path.cwd()))
            is None
        )

    @pytest.mark.parametrize(
        'bad', ['Not A Slug', 'trailing-', 'has_underscore', '', 'UPPER', None]
    )
    def test_a_non_slug_topic_fails_loud_naming_it(self, bad):
        """One namespace is shared with ProceduralTopicCluster.topic_id (D4);
        a gate filed against an unmatched topic could never be closed because
        the closure scroll would never find its cluster."""
        with pytest.raises(ValueError) as exc:
            build_consolidation_gate_task(topic=bad)
        assert repr(bad) in str(exc.value) or str(bad) in str(exc.value)

    def test_description_embeds_the_end_state_brief(self):
        """The filed gate and the prompt cannot disagree about the target."""
        spec = build_consolidation_gate_task(topic=_TOPIC)
        assert render_end_state_brief() in spec.description

    def test_title_and_description_are_overridable(self):
        spec = build_consolidation_gate_task(
            topic=_TOPIC, title='T', description='D'
        )
        assert spec.title == 'T' and spec.description == 'D'


class TestInertProvenance:
    """An enumeration is PROVENANCE, never the working list.

    The prior plan made enumerations unconstructible, which collided head-on
    with PRD leaf κ (task 3136), whose report framing gives gate metadata
    ``{report_run, observed_members, detector, authoritative: false}``.
    Accepting it as inert lets both compose.
    """

    def test_enumeration_is_stored_in_the_kappa_report_shape(self):
        spec = build_consolidation_gate_task(
            topic=_TOPIC,
            observed_members=[_uuid(1), _uuid(2)],
            report_run='run-abc',
            detector='topic-cluster-scan',
        )
        prov = spec.metadata[GATE_METADATA_KEY]['provenance']
        assert set(prov) == {
            'report_run',
            'observed_members',
            'detector',
            'authoritative',
        }
        assert prov['observed_members'] == [_uuid(1), _uuid(2)]
        assert prov['report_run'] == 'run-abc'
        assert prov['detector'] == 'topic-cluster-scan'

    def test_authoritative_is_forced_false_even_when_the_caller_says_true(self):
        """This is what makes 'inert' structural rather than aspirational."""
        spec = build_consolidation_gate_task(
            topic=_TOPIC, observed_members=[_uuid(1)], authoritative=True
        )
        assert spec.metadata[GATE_METADATA_KEY]['provenance']['authoritative'] is False

    def test_no_provenance_block_when_nothing_was_enumerated(self):
        spec = build_consolidation_gate_task(topic=_TOPIC)
        assert 'provenance' not in spec.metadata[GATE_METADATA_KEY]

    def test_provenance_never_grants_a_pass(self):
        """THE inertness property: same live members, identical verdict with
        and without a provenance list.

        DF gate 3036's hand-written enumeration under an invented
        `metadata.memory_ids` key was extended 7->8 by a later cycle while it
        still defined 'done'. A list that cannot change the verdict cannot do
        that.
        """
        members = [_member(_uuid(1)), _member(_uuid(2))]  # no canonical: refuses
        bare = _closure(members, gate_block={'topic': _TOPIC})
        with_prov = _closure(
            members,
            gate_block={
                'topic': _TOPIC,
                'provenance': {
                    'report_run': 'run-abc',
                    'observed_members': [_uuid(1), _uuid(2), _uuid(3)],
                    'detector': 'topic-cluster-scan',
                    'authoritative': False,
                },
            },
        )
        assert bare == with_prov

    def test_provenance_can_still_ADD_a_refusal(self):
        """A cluster member that is live but never got stamped into the topic
        is the one thing provenance may contribute — a refusal, never a pass."""
        stray = _uuid(42)
        verdict = _closure(
            _well_formed_cluster(2), unstamped_live_ids=[stray]
        )
        assert verdict.closed is False
        named = [
            r for r in verdict.reasons if r['code'] == 'unstamped_cluster_member'
        ]
        assert named and named[0]['ids'] == [stray]

    def test_an_unstamped_member_is_absence_based_so_truncation_suppresses_it(self):
        """Past the scroll cap, 'not stamped' and 'not seen' are the same."""
        codes = _codes(
            _closure(
                _well_formed_cluster(2),
                unstamped_live_ids=[_uuid(42)],
                scroll_truncated=True,
                scroll_total=2,
            )
        )
        assert 'scroll_incomplete' in codes
        assert 'unstamped_cluster_member' not in codes


# --------------------------------------------------------------------------- #
# Task 4808 — the BRIDGE: deriving `unstamped_live_ids` from inert provenance.
#
# `evaluate_closure` has always ACCEPTED `unstamped_live_ids`, and
# `build_consolidation_gate_task` has always WRITTEN
# `provenance.observed_members`, but nothing computed one from the other, so
# `unstamped_cluster_member` was unreachable outside these tests.
# --------------------------------------------------------------------------- #


def _prov(observed):
    """A gate block whose inert kappa-shaped provenance names *observed*."""
    return {
        'topic': _TOPIC,
        'provenance': {
            'report_run': 'run-abc',
            'observed_members': list(observed),
            'detector': 'topic-cluster-scan',
            'authoritative': False,
        },
    }


class TestUnstampedCandidates:
    """The PURE half of the bridge: which observed ids must be PROBED.

    `unstamped_candidates` is `observed_members` MINUS the live topic scroll
    MINUS the canonical\'s `supersedes` claim.  It answers "which ids are
    ambiguous", not "which ids are unstamped" — an id absent from the scroll
    is either absorbed-and-deleted or live-but-unstamped, and only a probe
    can tell those apart.
    """

    def test_a_stamped_observed_member_is_not_a_candidate(self):
        """The measured 2026-08-27 corpus shape: every observed member is in
        the scroll, so all four known-good gates derive ZERO candidates."""
        members = _well_formed_cluster(3)
        observed = [_uuid(1), _uuid(2), _uuid(3)]
        assert (
            consolidation_gate.unstamped_candidates(
                _prov(observed), members=members
            )
            == ()
        )

    def test_an_observed_member_absent_from_the_scroll_is_a_candidate(self):
        stray = _uuid(42)
        assert consolidation_gate.unstamped_candidates(
            _prov([_uuid(1), stray]), members=_well_formed_cluster(2)
        ) == (stray,)

    def test_the_canonicals_supersedes_claim_suppresses_a_candidate(self):
        """The DELETE arm.  Subtracting the cluster\'s own absorption claim is
        what keeps a correctly executed delete-arm consolidation closeable."""
        absorbed = _uuid(42)
        members = [
            _member(_uuid(1), canonical=True, supersedes=[absorbed]),
            _member(_uuid(2)),
        ]
        assert (
            consolidation_gate.unstamped_candidates(
                _prov([_uuid(1), absorbed]), members=members
            )
            == ()
        )

    def test_the_legacy_bare_scalar_supersedes_spelling_also_suppresses(self):
        """81 live records predate 3196\'s list migration; `normalize_supersedes`
        accepts both spellings and this derivation must not disagree with it."""
        absorbed = _uuid(42)
        members = [
            _member(_uuid(1), canonical=True, supersedes=absorbed),
            _member(_uuid(2)),
        ]
        assert (
            consolidation_gate.unstamped_candidates(
                _prov([absorbed]), members=members
            )
            == ()
        )

    def test_a_non_canonical_peers_supersedes_does_not_suppress(self):
        """Only the CANONICAL\'s claim is the cluster\'s claim — the same rule
        `consolidation_gate.py::_classify_supersedes` already states."""
        stray = _uuid(42)
        members = [
            _member(_uuid(1), canonical=True),
            _member(_uuid(2), supersedes=[stray]),
        ]
        assert consolidation_gate.unstamped_candidates(
            _prov([stray]), members=members
        ) == (stray,)

    def test_supersedes_is_only_read_when_exactly_one_canonical_exists(self):
        """With two canonicals there is no single cluster claim to trust; the
        gate is refusing on `multiple_canonicals` anyway."""
        absorbed = _uuid(42)
        members = [
            _member(_uuid(1), canonical=True, supersedes=[absorbed]),
            _member(_uuid(2), canonical=True),
        ]
        assert consolidation_gate.unstamped_candidates(
            _prov([absorbed]), members=members
        ) == (absorbed,)

    def test_a_non_uuid_observed_id_is_dropped(self):
        """It cannot be probed, so it can never be substantiated."""
        assert (
            consolidation_gate.unstamped_candidates(
                _prov(['not-a-uuid', '', 'deadbeef', None]),
                members=_well_formed_cluster(2),
            )
            == ()
        )

    def test_scroll_matching_is_case_insensitive(self):
        """Mirrors `evaluate_closure`\'s own case-folded `live_ids`, so the two
        cannot disagree about what the scroll saw."""
        members = [_member(_uuid(1).upper(), canonical=True)]
        assert (
            consolidation_gate.unstamped_candidates(
                _prov([_uuid(1)]), members=members
            )
            == ()
        )

    def test_supersedes_matching_is_case_insensitive(self):
        absorbed = _uuid(42)
        members = [_member(_uuid(1), canonical=True, supersedes=[absorbed.upper()])]
        assert (
            consolidation_gate.unstamped_candidates(
                _prov([absorbed]), members=members
            )
            == ()
        )

    def test_the_result_is_deduped_and_first_seen_ordered(self):
        """Deterministic, so refusal messages and tests are stable."""
        a, b = _uuid(42), _uuid(43)
        assert consolidation_gate.unstamped_candidates(
            _prov([b, a, b, a]), members=_well_formed_cluster(2)
        ) == (b, a)

    def test_the_original_id_spelling_is_returned_not_the_folded_one(self):
        """A refusal must name the id exactly as the gate recorded it."""
        stray = _uuid(42).upper()
        assert consolidation_gate.unstamped_candidates(
            _prov([stray]), members=_well_formed_cluster(2)
        ) == (stray,)

    @pytest.mark.parametrize(
        'block',
        [
            None,
            'not-a-mapping',
            42,
            {},
            {'topic': _TOPIC},
            {'topic': _TOPIC, 'provenance': None},
            {'topic': _TOPIC, 'provenance': 'not-a-mapping'},
            {'topic': _TOPIC, 'provenance': {}},
            {'topic': _TOPIC, 'provenance': {'observed_members': None}},
            {'topic': _TOPIC, 'provenance': {'observed_members': 42}},
            # A bare string is a Sequence but is NOT a member list; iterating
            # it would yield 36 single characters, none of them a uuid.
            {'topic': _TOPIC, 'provenance': {'observed_members': _uuid(42)}},
            {'topic': _TOPIC, 'provenance': {'observed_members': b'bytes'}},
        ],
    )
    def test_defensive_shapes_return_empty_rather_than_raising(self, block):
        """This predicate\'s job is to REPORT malformedness; one that dies on
        bad input blocks the very gates it exists to adjudicate."""
        assert (
            consolidation_gate.unstamped_candidates(
                block, members=_well_formed_cluster(2)
            )
            == ()
        )

    def test_malformed_scroll_rows_do_not_raise(self):
        assert consolidation_gate.unstamped_candidates(
            _prov([_uuid(42)]), members=['not-a-mapping', None, {}, {'id': None}]
        ) == (_uuid(42),)

    def test_it_is_exported(self):
        assert 'unstamped_candidates' in consolidation_gate.__all__

class _RecordingProbe:
    """An ``exists`` collaborator that records every call it is given.

    Hand-rolled rather than an ``AsyncMock`` so the recorded shape is the
    exact ``(memory_id, *, project_id)`` contract under test — an AsyncMock
    would accept any signature and pass a scoping slip silently.
    """

    def __init__(self, live=(), raises=None):
        self.live = {str(i).lower() for i in live}
        self.raises = raises
        self.calls = []

    async def __call__(self, memory_id, *, project_id):
        self.calls.append((memory_id, project_id))
        if self.raises is not None:
            raise self.raises
        return str(memory_id).lower() in self.live


class TestResolveUnstampedLiveIds:
    """The PROBE half of the bridge: which candidates are genuinely live.

    A candidate absent from the scroll is either absorbed-and-deleted or
    live-but-unstamped.  One point read per candidate settles it — and the
    candidate list is empty on every well-formed gate, so the common path
    issues no reads at all.
    """

    @pytest.mark.asyncio
    async def test_a_live_candidate_is_returned(self):
        """THE defect: live, but never stamped into the topic, therefore
        invisible to the topic scroll."""
        stray = _uuid(42)
        probe = _RecordingProbe(live=[stray])
        assert await consolidation_gate.resolve_unstamped_live_ids(
            _prov([stray]),
            members=_well_formed_cluster(2),
            exists=probe,
            project_id='dark_factory',
        ) == (stray,)

    @pytest.mark.asyncio
    async def test_an_absent_candidate_is_not_returned(self):
        """Deleted without being claimed in `supersedes` is still an ABSORBED
        id, not a stray — so the delete arm still closes."""
        gone = _uuid(42)
        probe = _RecordingProbe(live=[])
        assert (
            await consolidation_gate.resolve_unstamped_live_ids(
                _prov([gone]),
                members=_well_formed_cluster(2),
                exists=probe,
                project_id='dark_factory',
            )
            == ()
        )
        assert len(probe.calls) == 1

    @pytest.mark.asyncio
    async def test_each_candidate_is_probed_exactly_once_and_scoped(self):
        """A cross-project probe would judge one project\'s gate against
        another project\'s memories — the same scoping argument
        `TaskInterceptor.set_consolidation_scroll` makes for the scroll."""
        a, b = _uuid(42), _uuid(43)
        probe = _RecordingProbe(live=[a, b])
        await consolidation_gate.resolve_unstamped_live_ids(
            _prov([a, b, a]),
            members=_well_formed_cluster(2),
            exists=probe,
            project_id='dark_factory',
        )
        assert probe.calls == [(a, 'dark_factory'), (b, 'dark_factory')]

    @pytest.mark.asyncio
    async def test_suppressed_ids_are_never_probed(self):
        """The cheap subtraction runs FIRST: stamped, claimed and non-uuid ids
        cost nothing."""
        stamped, absorbed = _uuid(1), _uuid(42)
        members = [
            _member(stamped, canonical=True, supersedes=[absorbed]),
            _member(_uuid(2)),
        ]
        probe = _RecordingProbe(live=[stamped, absorbed])
        assert (
            await consolidation_gate.resolve_unstamped_live_ids(
                _prov([stamped, absorbed, 'not-a-uuid']),
                members=members,
                exists=probe,
                project_id='dark_factory',
            )
            == ()
        )
        assert probe.calls == []

    @pytest.mark.asyncio
    async def test_zero_candidates_never_awaits_the_probe(self):
        """The measured 2026-08-27 corpus shape: every observed member is
        already stamped, so all four known-good gates cost zero reads."""
        probe = _RecordingProbe()
        assert (
            await consolidation_gate.resolve_unstamped_live_ids(
                _prov([_uuid(1), _uuid(2)]),
                members=_well_formed_cluster(2),
                exists=probe,
                project_id='dark_factory',
            )
            == ()
        )
        assert probe.calls == []

    @pytest.mark.asyncio
    async def test_an_unwired_probe_is_dormant(self):
        """Without a probe we cannot tell absorbed from unstamped, and guessing
        \'unstamped\' would make every delete-arm consolidation uncloseable."""
        assert (
            await consolidation_gate.resolve_unstamped_live_ids(
                _prov([_uuid(42)]),
                members=_well_formed_cluster(2),
                exists=None,
                project_id='dark_factory',
            )
            == ()
        )

    @pytest.mark.asyncio
    async def test_a_raising_probe_propagates(self):
        """Both callers own the fail-closed policy; swallowing here would let
        an unreadable store read as \'no strays\'."""
        probe = _RecordingProbe(raises=TimeoutError('qdrant timed out'))
        with pytest.raises(TimeoutError):
            await consolidation_gate.resolve_unstamped_live_ids(
                _prov([_uuid(42)]),
                members=_well_formed_cluster(2),
                exists=probe,
                project_id='dark_factory',
            )

    @pytest.mark.asyncio
    async def test_the_result_preserves_candidate_order(self):
        a, b, c = _uuid(42), _uuid(43), _uuid(44)
        probe = _RecordingProbe(live=[a, c])
        assert await consolidation_gate.resolve_unstamped_live_ids(
            _prov([c, b, a]),
            members=_well_formed_cluster(2),
            exists=probe,
            project_id='dark_factory',
        ) == (c, a)

    def test_it_is_exported(self):
        assert 'resolve_unstamped_live_ids' in consolidation_gate.__all__

class TestHandrolledMemberEnumeration:
    """The SECOND gap: a gate filed with no `x_recon_consolidation_gate` block
    at all, for which the seam is fully dormant and `set_task_status(done)`
    closes it untouched (task 4747 was filed exactly that way).

    A refusal is NOT available here — `operational_mode == 'gate'` is a
    GENERIC human-gate marker — so the detector exists to FLAG, and its
    precision only has to be good enough for a log line.
    """

    @staticmethod
    def _meta(**extra):
        meta = {'execution_class': 'operational', 'operational_mode': 'gate'}
        meta.update(extra)
        return meta

    def test_fires_for_the_memory_ids_spelling(self):
        """Gate 3036 — the very gate this module\'s docstring already indicts
        for inventing `metadata.memory_ids`."""
        ids = [_uuid(1), _uuid(2)]
        assert consolidation_gate.handrolled_member_enumeration(
            self._meta(memory_ids=ids)
        ) == ('memory_ids', ids)

    def test_fires_for_the_related_memory_ids_spelling(self):
        """Gate 4747, before it was retro-fitted with a real block."""
        ids = [_uuid(1)]
        assert consolidation_gate.handrolled_member_enumeration(
            self._meta(related_memory_ids=ids)
        ) == ('related_memory_ids', ids)

    def test_silent_for_an_ordinary_gate_with_neither_key(self):
        """The 118-task majority. A false positive here would spam the log for
        every gate close in the fleet."""
        assert consolidation_gate.handrolled_member_enumeration(self._meta()) is None

    def test_silent_when_a_proper_block_is_present(self):
        """4747\'s post-retrofit shape: a real block WINS, even alongside a
        leftover hand-rolled key."""
        assert (
            consolidation_gate.handrolled_member_enumeration(
                self._meta(
                    memory_ids=[_uuid(1)],
                    **{GATE_METADATA_KEY: {'topic': _TOPIC}},
                )
            )
            is None
        )

    def test_silent_for_a_non_gate_carrying_the_key(self):
        assert (
            consolidation_gate.handrolled_member_enumeration(
                {'execution_class': 'operational',
                 'operational_mode': 'llm',
                 'memory_ids': [_uuid(1)]}
            )
            is None
        )

    @pytest.mark.parametrize('metadata', [None, 'not-a-mapping', 42, [], {}])
    def test_silent_for_non_dict_or_absent_metadata(self, metadata):
        assert consolidation_gate.handrolled_member_enumeration(metadata) is None

    def test_key_selection_is_deterministic_when_both_are_present(self):
        """Sorted-first, so the warning text is stable across runs."""
        key, _ = consolidation_gate.handrolled_member_enumeration(
            self._meta(memory_ids=[_uuid(1)], related_memory_ids=[_uuid(2)])
        )
        assert key == sorted(consolidation_gate.HANDROLLED_MEMBER_KEYS)[0]

    def test_an_empty_enumeration_is_not_a_hit(self):
        """An empty list enumerates nothing, so there is nothing to flag."""
        assert (
            consolidation_gate.handrolled_member_enumeration(self._meta(memory_ids=[]))
            is None
        )

    def test_the_measured_key_set_is_exported(self):
        assert set(consolidation_gate.HANDROLLED_MEMBER_KEYS) == {
            'memory_ids',
            'related_memory_ids',
        }
        assert 'HANDROLLED_MEMBER_KEYS' in consolidation_gate.__all__
        assert 'handrolled_member_enumeration' in consolidation_gate.__all__

# --------------------------------------------------------------------------- #
# Guard: the seam's import weight, and INV-5's single homes (step-15a)
# --------------------------------------------------------------------------- #


class TestImportLeafAndSingleHomes:
    """``middleware/task_interceptor.py`` imports this module, so this module's
    import weight becomes the interceptor's.

    PRD D4 records a MEASURED hard import cycle from a careless import of
    exactly this kind (``config/schema.py`` -> ``memory_metadata`` ->
    ``backends.mem0_client`` -> ``config.schema``, raising ``ImportError:
    cannot import name 'FusedMemoryConfig'``), which is why ``TOPIC_SLUG_RE``
    got its own stdlib-only leaf module with a regression test. These probe
    BOTH import orders, because a cycle is directional: whichever module the
    process reaches first is the one left half-initialised.

    Probed in a FRESH interpreter each time — this test process has already
    imported both modules at collection, so an in-process ``sys.modules`` check
    would pass vacuously (the same reasoning as
    ``test_topic_slug_namespace.py::test_module_is_import_light``).
    """

    #: Heavy modules the leaf must never pull in. ``targeted``/``harness`` are
    #: the reconciliation runtime; ``services.memory_service`` and
    #: ``server.tools`` are the store/MCP layers that import the interceptor
    #: back.
    FORBIDDEN = (
        'fused_memory.reconciliation.targeted',
        'fused_memory.reconciliation.harness',
        'fused_memory.services.memory_service',
        'fused_memory.server.tools',
    )

    @staticmethod
    def _probe(body: str):
        return subprocess.run(
            [sys.executable, '-c', body], capture_output=True, text=True, timeout=300
        )

    def test_imports_with_the_interceptor_first(self):
        """The production order: the seam imports the leaf."""
        result = self._probe(
            'import fused_memory.middleware.task_interceptor as ti; '
            'import fused_memory.reconciliation.consolidation_gate as cg; '
            'assert ti.GATE_METADATA_KEY is cg.GATE_METADATA_KEY'
        )
        assert result.returncode == 0, result.stderr

    def test_imports_with_the_gate_first(self):
        """The reversed order: a test module (or the CLI) imports the leaf
        first and the interceptor afterwards.

        A cycle is directional, so passing one order proves nothing about the
        other — the measured D4 failure only surfaced from one side.
        """
        result = self._probe(
            'import fused_memory.reconciliation.consolidation_gate as cg; '
            'import fused_memory.middleware.task_interceptor as ti; '
            'assert ti.GATE_METADATA_KEY is cg.GATE_METADATA_KEY'
        )
        assert result.returncode == 0, result.stderr

    def test_module_imports_stay_leaf(self):
        """Importing the leaf alone must not drag in the reconciliation
        runtime, the memory service or the MCP tool layer."""
        forbidden = ', '.join(repr(m) for m in self.FORBIDDEN)
        result = self._probe(
            'import sys; '
            'import fused_memory.reconciliation.consolidation_gate  # noqa: F401\n'
            f'forbidden = [{forbidden}]\n'
            'present = [m for m in forbidden if m in sys.modules]\n'
            'assert not present, present\n'
        )
        assert result.returncode == 0, result.stderr

    def test_binds_the_same_normalize_supersedes(self):
        """INV-5: ``is``, never ``==``. Equality would be satisfied by a
        re-typed copy of the parser, which is exactly the drift INV-5 forbids
        and which prose alone cannot prevent.
        """
        assert (
            consolidation_gate.normalize_supersedes
            is memory_metadata.normalize_supersedes
        )

    def test_binds_the_same_is_full_uuid(self):
        from fused_memory.utils import validation

        assert consolidation_gate.is_full_uuid is validation.is_full_uuid

    def test_binds_the_same_is_valid_topic_slug(self):
        """The single home is the stdlib-only ``topic_slug`` leaf; going
        through ``memory_metadata``'s re-export must reach the SAME object, or
        the one topic namespace (PRD D4) has quietly split in two.
        """
        import fused_memory.topic_slug as ts

        assert consolidation_gate.is_valid_topic_slug is ts.is_valid_topic_slug
        assert memory_metadata.is_valid_topic_slug is ts.is_valid_topic_slug
