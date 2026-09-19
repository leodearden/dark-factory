"""Contract guards on ``escalation.shadow_ruling`` — the shadow-mode L2 ruling codec.

Task 5374. ``docs/escalation-standing-policy.md`` proposes classes of L2 that an
adjudicating session could one day rule without waiting for the human. NONE is
adopted. This module is the measurement half: the watcher stamps what it WOULD
have ruled, takes no action, and a weekly count compares the stamped proposal
against what the human actually did.

Nothing here grants authority. These tests hold the codec, the mechanical-gate
detector, the agreement report and the CLI arm — never the prose of the policy
document (that boundary lives in
``tests/scripts/test_shadow_ruling_doc_contract.py``).

NO ``pytest.importorskip`` AND NO try/except-and-skip ANYWHERE IN THIS MODULE.
An unimportable ``escalation.authority`` must fail these guards rather than
silently pass them — a SPOT guard that skips is worse than no SPOT guard,
because the drift it exists to catch then lands green.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from escalation.authority import L2_AUTO_CLOSE_DENY_CATEGORIES, L2_AUTO_CLOSE_DENY_ROLES
from escalation.classify import classify_resolver_tier
from escalation.models import Escalation
from escalation.queue import EscalationQueue, iter_all_escalation_paths
from escalation.shadow_ruling import (
    COMPARABLE_ACTIONS,
    DETECTABLE_GATES,
    FIRST_TRANCHE_CLASSES,
    GATED_CATEGORIES,
    GATED_ROLES,
    HUMAN_FOREVER_GATES,
    REVERSIBLE_ACTIONS,
    SHADOW_RULING_MARKER,
    ShadowRuling,
    agreement_report,
    main,
    mechanically_gated,
    parse_shadow_ruling,
)

# A well-formed ruling used wherever the test's subject is something OTHER than
# the field values themselves.
_A_RULING = ShadowRuling(
    ruling_class='risk_identified_branch_behind_main',
    proposed_action='close_only',
    evidence='git merge-base --is-ancestor task/5374 main -> rc=0 (branch is not behind)',
    confidence=0.9,
)

# The freshness contract a real triage_note satisfies (see the watcher skill's
# "Reading a triage-ack annotation"): a named world-facing predicate plus the
# probe used to check it. The marker line must COMPOSE with this, not replace
# it — that composition is the whole reason the payload is a LINE.
_FRESHNESS_NOTE = (
    'task-5374 branch tip not behind main | probe: git merge-base --is-ancestor '
    'main task/5374 -> rc=0'
)


class TestSlugVocabularies:
    """The three slug sets the policy document enumerates and the codec validates."""

    def test_first_tranche_classes_are_the_four_shadowed_candidates(self):
        assert isinstance(FIRST_TRANCHE_CLASSES, frozenset)
        assert all(isinstance(s, str) for s in FIRST_TRANCHE_CLASSES)
        assert len(FIRST_TRANCHE_CLASSES) == 4, (
            f'expected exactly the four first-tranche candidate classes, got '
            f'{sorted(FIRST_TRANCHE_CLASSES)}'
        )

    def test_reversible_actions_are_the_five_ratified_skeleton_actions(self):
        assert isinstance(REVERSIBLE_ACTIONS, frozenset)
        assert all(isinstance(s, str) for s in REVERSIBLE_ACTIONS)
        assert len(REVERSIBLE_ACTIONS) == 5, (
            f'expected exactly the five reversible actions the ratified skeleton '
            f'names, got {sorted(REVERSIBLE_ACTIONS)}'
        )

    def test_the_two_c1_resolution_actions_are_in_the_reversible_list(self):
        """`close_only` and `resume` are the only two that are also C1
        ``resolution_action`` values — the ones the weekly count can compare.
        The other three are task-side and are reported `not_comparable`."""
        assert {'close_only', 'resume'} <= REVERSIBLE_ACTIONS

    def test_human_forever_gates_are_the_seven_gates(self):
        assert isinstance(HUMAN_FOREVER_GATES, frozenset)
        assert all(isinstance(s, str) for s in HUMAN_FOREVER_GATES)
        assert len(HUMAN_FOREVER_GATES) == 7, (
            f'expected exactly the seven human-forever gates, got '
            f'{sorted(HUMAN_FOREVER_GATES)}'
        )

    def test_the_vocabularies_do_not_overlap(self):
        """Three orthogonal dimensions — a class, an action and a gate are
        different kinds of thing, so a slug shared between two of them would
        make a payload ambiguous to a reader."""
        assert not FIRST_TRANCHE_CLASSES & REVERSIBLE_ACTIONS
        assert not FIRST_TRANCHE_CLASSES & HUMAN_FOREVER_GATES
        assert not REVERSIBLE_ACTIONS & HUMAN_FOREVER_GATES


class TestShadowRulingShape:
    """The payload is exactly the four things the task names, and it is frozen."""

    def test_fields_are_exactly_the_four_the_task_names(self):
        names = tuple(f.name for f in dataclasses.fields(ShadowRuling))
        assert names == ('ruling_class', 'proposed_action', 'evidence', 'confidence'), (
            f'expected the four fields class/proposed_action/evidence/confidence, got {names}'
        )

    def test_is_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            _A_RULING.confidence = 0.1  # type: ignore[misc]

    def test_rejects_unknown_ruling_class_naming_the_legal_values(self):
        with pytest.raises(ValueError) as exc:
            ShadowRuling(
                ruling_class='not_a_class', proposed_action='close_only',
                evidence='e', confidence=0.5,
            )
        message = str(exc.value)
        assert 'not_a_class' in message
        for slug in FIRST_TRANCHE_CLASSES:
            assert slug in message, f'rejection must name the legal values; {slug!r} missing'

    def test_rejects_unknown_proposed_action_naming_the_legal_values(self):
        with pytest.raises(ValueError) as exc:
            ShadowRuling(
                ruling_class=sorted(FIRST_TRANCHE_CLASSES)[0], proposed_action='restart',
                evidence='e', confidence=0.5,
            )
        message = str(exc.value)
        assert 'restart' in message
        for slug in REVERSIBLE_ACTIONS:
            assert slug in message, f'rejection must name the legal values; {slug!r} missing'

    @pytest.mark.parametrize('confidence', [-0.01, 1.01, 2.0, -1.0])
    def test_rejects_confidence_outside_the_unit_interval(self, confidence: float):
        with pytest.raises(ValueError, match='confidence'):
            ShadowRuling(
                ruling_class=sorted(FIRST_TRANCHE_CLASSES)[0], proposed_action='close_only',
                evidence='e', confidence=confidence,
            )

    @pytest.mark.parametrize('confidence', [0.0, 0.5, 1.0])
    def test_accepts_the_closed_unit_interval_endpoints(self, confidence: float):
        ShadowRuling(
            ruling_class=sorted(FIRST_TRANCHE_CLASSES)[0], proposed_action='close_only',
            evidence='e', confidence=confidence,
        )


class TestNoteLineRendering:
    """``to_note_line`` renders ONE line: the marker, then a JSON object."""

    def test_renders_a_single_line_starting_with_the_marker(self):
        line = _A_RULING.to_note_line()
        assert '\n' not in line, 'the payload must be ONE line so it composes with the note'
        assert line.startswith(SHADOW_RULING_MARKER)

    def test_the_remainder_is_a_json_object_with_the_four_keys(self):
        line = _A_RULING.to_note_line()
        payload = json.loads(line[len(SHADOW_RULING_MARKER):])
        assert isinstance(payload, dict)
        assert set(payload) == {'class', 'proposed_action', 'evidence', 'confidence'}
        assert payload['class'] == _A_RULING.ruling_class
        assert payload['proposed_action'] == _A_RULING.proposed_action
        assert payload['evidence'] == _A_RULING.evidence
        assert payload['confidence'] == _A_RULING.confidence

    def test_a_multiline_evidence_string_still_renders_one_line(self):
        """Evidence is operator-supplied text and may carry newlines; JSON
        escaping keeps the marker a single line regardless."""
        ruling = dataclasses.replace(_A_RULING, evidence='line one\nline two')
        line = ruling.to_note_line()
        assert '\n' not in line
        assert parse_shadow_ruling(line) == ruling


class TestParseRoundTrip:
    def test_round_trips(self):
        assert parse_shadow_ruling(_A_RULING.to_note_line()) == _A_RULING

    def test_parses_when_embedded_in_a_multi_line_freshness_note(self):
        """The composition property the whole design depends on: a real note
        keeps its predicate/probe text and gains the marker on its OWN line."""
        note = f'{_FRESHNESS_NOTE}\n{_A_RULING.to_note_line()}'
        assert parse_shadow_ruling(note) == _A_RULING

    def test_parses_when_the_marker_line_is_not_last(self):
        note = f'{_A_RULING.to_note_line()}\n{_FRESHNESS_NOTE}'
        assert parse_shadow_ruling(note) == _A_RULING

    def test_returns_the_ruling_not_the_prose(self):
        note = f'{_FRESHNESS_NOTE}\n{_A_RULING.to_note_line()}'
        parsed = parse_shadow_ruling(note)
        assert parsed is not None
        assert _FRESHNESS_NOTE not in parsed.evidence


class TestParseReturnsNoneForAbsentOrUnusable:
    """``None`` means "no usable shadow ruling here", NEVER an exception: the
    weekly count sweeps thousands of records that carry no stamp at all."""

    @pytest.mark.parametrize(
        'note',
        [
            pytest.param('', id='empty'),
            pytest.param(_FRESHNESS_NOTE, id='freshness-note-only'),
            pytest.param('   \n\n  ', id='whitespace'),
            pytest.param(f'{SHADOW_RULING_MARKER} not json at all', id='not-json'),
            pytest.param(f'{SHADOW_RULING_MARKER} [1, 2, 3]', id='json-array'),
            pytest.param(f'{SHADOW_RULING_MARKER} "a string"', id='json-scalar-string'),
            pytest.param(f'{SHADOW_RULING_MARKER} 7', id='json-scalar-number'),
            pytest.param(f'{SHADOW_RULING_MARKER} null', id='json-null'),
            pytest.param(f'{SHADOW_RULING_MARKER} {{}}', id='empty-object'),
        ],
    )
    def test_returns_none(self, note: str):
        assert parse_shadow_ruling(note) is None

    def test_returns_none_for_an_out_of_vocabulary_class(self):
        payload = json.dumps({
            'class': 'invented_class', 'proposed_action': 'close_only',
            'evidence': 'e', 'confidence': 0.5,
        })
        assert parse_shadow_ruling(f'{SHADOW_RULING_MARKER} {payload}') is None

    def test_returns_none_for_an_out_of_vocabulary_proposed_action(self):
        payload = json.dumps({
            'class': sorted(FIRST_TRANCHE_CLASSES)[0], 'proposed_action': 'abandon',
            'evidence': 'e', 'confidence': 0.5,
        })
        assert parse_shadow_ruling(f'{SHADOW_RULING_MARKER} {payload}') is None

    def test_returns_none_for_a_payload_missing_a_key(self):
        payload = json.dumps({'class': sorted(FIRST_TRANCHE_CLASSES)[0]})
        assert parse_shadow_ruling(f'{SHADOW_RULING_MARKER} {payload}') is None

    def test_returns_none_for_a_payload_with_an_extra_key(self):
        """An unrecognised key means the writer believed something this codec
        does not implement — refuse rather than silently drop it."""
        payload = json.dumps({
            'class': sorted(FIRST_TRANCHE_CLASSES)[0], 'proposed_action': 'close_only',
            'evidence': 'e', 'confidence': 0.5, 'authority_granted': True,
        })
        assert parse_shadow_ruling(f'{SHADOW_RULING_MARKER} {payload}') is None

    def test_a_marker_mentioned_mid_line_is_not_a_stamp(self):
        """Only a line that BEGINS with the marker is a payload, so prose that
        merely mentions the token — e.g. this skill's own guidance — cannot be
        mistaken for one."""
        note = f'see {SHADOW_RULING_MARKER} {_A_RULING.to_note_line()}'
        assert parse_shadow_ruling(note) is None


def test_escalation_carries_the_triage_fields_the_codec_rides_in():
    """The stamp has no field of its own: it rides inside ``triage_note``,
    which is why the model needs no migration (design decision 1)."""
    record = Escalation(
        id='esc-5374-1', task_id='5374', agent_role='task-agent',
        severity='info', category='risk_identified', summary='s',
    )
    assert record.triage_note == ''
    assert record.triaged_by is None
    assert record.triaged_at is None


# ---------------------------------------------------------------------------
# Mechanical gate detection (step-3)
# ---------------------------------------------------------------------------


def _record(**overrides: object) -> Escalation:
    """A real ``Escalation`` — never a dict. The detector reads dataclass
    attributes, so a dict stand-in would pass while the production call site
    raised."""
    fields: dict[str, object] = {
        'id': 'esc-5374-1',
        'task_id': '5374',
        'agent_role': 'claude-task-5374-implementer',
        'severity': 'info',
        'category': 'risk_identified',
        'summary': 'an ordinary L2',
        'level': 2,
    }
    fields.update(overrides)
    return Escalation(**fields)  # type: ignore[arg-type]


class TestMechanicallyGated:
    """The detectable subset of the human-forever gate list, read off
    ``escalation.authority`` rather than restated here."""

    def test_flags_the_milestone_gate_category(self):
        assert mechanically_gated(_record(category='milestone_gate')) == 'milestone_gate'

    @pytest.mark.parametrize('category', sorted(L2_AUTO_CLOSE_DENY_CATEGORIES))
    def test_flags_every_denied_category(self, category: str):
        """authority.py's own docstring calls all of these the born-at-L2 human
        gates, so every member flags — not only the one named `milestone_gate`."""
        assert mechanically_gated(_record(category=category)) == 'milestone_gate'

    @pytest.mark.parametrize('role', sorted(L2_AUTO_CLOSE_DENY_ROLES))
    def test_flags_every_denied_role(self, role: str):
        assert mechanically_gated(_record(agent_role=role)) == 'deterministic_runner_filing'

    def test_returns_none_for_an_ordinary_risk_identified_l2(self):
        assert mechanically_gated(_record()) is None

    def test_role_is_checked_even_when_the_category_is_benign(self):
        """Defence in depth, the property authority.py already relies on: the
        two checks are independent, so neither benign half can mask the other."""
        record = _record(category='risk_identified', agent_role='orchestrator-deterministic')
        assert record.category not in L2_AUTO_CLOSE_DENY_CATEGORIES
        assert mechanically_gated(record) == 'deterministic_runner_filing'

    def test_category_is_checked_even_when_the_role_is_benign(self):
        record = _record(category='milestone_gate', agent_role='claude-task-1-implementer')
        assert record.agent_role not in L2_AUTO_CLOSE_DENY_ROLES
        assert mechanically_gated(record) == 'milestone_gate'


class TestMechanicalGateSpot:
    """The detector must not hold a SECOND copy of the denylists."""

    def test_the_detector_reads_the_imported_category_denylist(self):
        assert GATED_CATEGORIES is L2_AUTO_CLOSE_DENY_CATEGORIES, (
            'the detector must consult the imported frozenset itself, so a member '
            'added in escalation/src/escalation/authority.py propagates here '
            'instead of drifting'
        )

    def test_the_detector_reads_the_imported_role_denylist(self):
        assert GATED_ROLES is L2_AUTO_CLOSE_DENY_ROLES, (
            'the detector must consult the imported frozenset itself, so a member '
            'added in escalation/src/escalation/authority.py propagates here '
            'instead of drifting'
        )

    def test_every_slug_the_detector_can_return_is_a_human_forever_gate(self):
        assert DETECTABLE_GATES <= HUMAN_FOREVER_GATES

    def test_the_detectable_gates_are_a_proper_subset(self):
        """Five of the seven gates are semantic and have NO record-level signal.
        A detector that ever covered all seven would be claiming a completeness
        it cannot have — so the properness is the guard, not a coincidence."""
        assert DETECTABLE_GATES < HUMAN_FOREVER_GATES
        assert len(HUMAN_FOREVER_GATES - DETECTABLE_GATES) == 5

    def test_the_observable_slugs_match_the_declared_detectable_set(self):
        """Non-vacuity: DETECTABLE_GATES is not a hand-written label — every
        slug in it is really reachable, and nothing reachable is missing."""
        observed = {
            slug
            for slug in (
                [mechanically_gated(_record(category=c)) for c in L2_AUTO_CLOSE_DENY_CATEGORIES]
                + [mechanically_gated(_record(agent_role=r)) for r in L2_AUTO_CLOSE_DENY_ROLES]
            )
            if slug is not None
        }
        assert observed == DETECTABLE_GATES


# ---------------------------------------------------------------------------
# Archive round-trip and the weekly agreement report (step-5)
# ---------------------------------------------------------------------------

_MICROSECOND = timedelta(microseconds=1)
_BRANCH_BEHIND = 'risk_identified_branch_behind_main'
_VETO_STREAK = 'risk_identified_recovery_veto_streak'
_SEMANTIC_COLLISION = 'design_concern_semantic_collision'


def _ruling(ruling_class: str = _BRANCH_BEHIND, action: str = 'close_only') -> ShadowRuling:
    return ShadowRuling(
        ruling_class=ruling_class, proposed_action=action,
        evidence='probe output quoted verbatim', confidence=0.8,
    )


def _stamped_note(ruling: ShadowRuling) -> str:
    """A real note: the freshness predicate/probe PLUS the marker on its own
    line. Never the marker alone — that would delete the contract the note
    already carries."""
    return f'{_FRESHNESS_NOTE}\n{ruling.to_note_line()}'


class _Fixture:
    """Builds records through the REAL queue: submit -> stamp_triage -> resolve.

    Hand-written archive JSON would make every assertion below a statement about
    this file's beliefs rather than about production code paths — and the
    load-bearing premise of the whole measurement is precisely that the real
    paths preserve the stamp into the archive.
    """

    def __init__(self, tmp_path: Path, seq: int = 0):
        self.queue = EscalationQueue(tmp_path / 'escalations')
        self._seq = seq

    def submit(
        self, *, category: str = 'risk_identified', agent_role: str = 'claude-task-1-implementer',
        resolution_action: str | None = None,
    ) -> Escalation:
        self._seq += 1
        record = Escalation(
            id=f'esc-5374-{self._seq}', task_id='5374', agent_role=agent_role,
            severity='info', category=category, summary='shadow measurement subject',
            level=2, resolution_action=resolution_action,
        )
        self.queue.submit(record)
        return record

    def stamp(self, record: Escalation, ruling: ShadowRuling, *, by: str) -> None:
        stamped = self.queue.stamp_triage(
            record.id, triaged_by=by, triage_note=_stamped_note(ruling),
        )
        assert stamped is not None, 'stamp_triage refused a pending record'

    def resolve(self, record: Escalation, *, by: str, dismiss: bool = True) -> Escalation:
        resolved = self.queue.resolve(
            record.id, 'ruled', dismiss=dismiss, resolved_by=by,
        )
        assert resolved is not None
        return resolved

    def stamped_and_resolved(
        self, ruling: ShadowRuling, *, observed_action: str | None, stamped_by: str = 'watcher-a',
        resolved_by: str = 'interactive', category: str = 'risk_identified',
        agent_role: str = 'claude-task-1-implementer',
    ) -> Escalation:
        record = self.submit(
            category=category, agent_role=agent_role, resolution_action=observed_action,
        )
        self.stamp(record, ruling, by=stamped_by)
        return self.resolve(record, by=resolved_by, dismiss=observed_action == 'close_only')

    def report(self, *, since: datetime | None = None, until: datetime | None = None):
        return agreement_report(
            self.queue.queue_dir,
            since=since or datetime(2000, 1, 1, tzinfo=UTC),
            until=until or datetime(2100, 1, 1, tzinfo=UTC),
        )


class TestStampSurvivesIntoTheArchive:
    """THE LOAD-BEARING PREMISE, asserted end to end on the real queue.

    Everything downstream — the weekly count, the adoption threshold, the whole
    measurement half of task 5374 — is worthless if `resolve` loses the stamp.
    """

    def test_archived_record_still_carries_the_marker_beside_the_outcome(self, tmp_path: Path):
        fixture = _Fixture(tmp_path)
        ruling = _ruling()
        record = fixture.stamped_and_resolved(
            ruling, observed_action='close_only', stamped_by='watcher-a',
            resolved_by='interactive',
        )

        archived_paths = [
            p for p in iter_all_escalation_paths(fixture.queue.queue_dir)
            if p.stem == record.id
        ]
        assert archived_paths, 'the resolved record vanished from the root+archive sweep'
        assert 'archive' in archived_paths[0].parts, 'expected the record to have been archived'

        reloaded = Escalation.from_json(archived_paths[0].read_text())
        assert parse_shadow_ruling(reloaded.triage_note) == ruling
        assert _FRESHNESS_NOTE in reloaded.triage_note, (
            'the marker must COMPOSE with the freshness note, not replace it'
        )
        assert reloaded.triaged_by == 'watcher-a'
        assert reloaded.resolved_by == 'interactive'
        assert reloaded.resolved_at is not None
        assert reloaded.resolution_action == 'close_only'
        assert reloaded.status == 'dismissed', 'close_only dismisses rather than resolves'


class TestAgreementBuckets:
    def test_matching_action_is_agreed(self, tmp_path: Path):
        fixture = _Fixture(tmp_path)
        fixture.stamped_and_resolved(_ruling(action='close_only'), observed_action='close_only')

        klass = fixture.report().for_class(_BRANCH_BEHIND)
        assert klass is not None
        assert (klass.agreed, klass.diverged, klass.not_comparable) == (1, 0, 0)
        assert klass.comparable == 1
        assert klass.total == 1
        assert klass.agreement_rate == 1.0

    def test_differing_action_is_diverged(self, tmp_path: Path):
        fixture = _Fixture(tmp_path)
        fixture.stamped_and_resolved(_ruling(action='close_only'), observed_action='resume')

        klass = fixture.report().for_class(_BRANCH_BEHIND)
        assert klass is not None
        assert (klass.agreed, klass.diverged, klass.not_comparable) == (0, 1, 0)
        assert klass.agreement_rate == 0.0

    @pytest.mark.parametrize('task_side', ['add_dependency', 'update_task_amendment', 'file_task'])
    def test_task_side_proposals_are_not_comparable(self, tmp_path: Path, task_side: str):
        """Three of the five reversible actions leave `resolution_action` unset.
        Folding them into `diverged` would wrongly revert a class; folding them
        into `agreed` would inflate it; dropping them would shrink the
        denominator the threshold is read off without saying so."""
        fixture = _Fixture(tmp_path)
        fixture.stamped_and_resolved(_ruling(action=task_side), observed_action=None)

        klass = fixture.report().for_class(_BRANCH_BEHIND)
        assert klass is not None
        assert (klass.agreed, klass.diverged, klass.not_comparable) == (0, 0, 1)
        assert klass.comparable == 0
        assert klass.total == 1

    def test_the_comparable_denominator_is_reported_so_the_threshold_is_decidable(
        self, tmp_path: Path,
    ):
        """"95% or better over at least 10 items" must be decidable from the
        output alone — which needs the denominator, not only the rate."""
        fixture = _Fixture(tmp_path)
        for _ in range(3):
            fixture.stamped_and_resolved(_ruling(), observed_action='close_only')
        fixture.stamped_and_resolved(_ruling(action='file_task'), observed_action=None)

        klass = fixture.report().for_class(_BRANCH_BEHIND)
        assert klass is not None
        assert klass.comparable == 3
        assert klass.total == 4
        assert klass.agreement_rate == 1.0

    def test_agreement_rate_is_none_when_nothing_is_comparable(self, tmp_path: Path):
        """A class whose proposals are all task-side is visibly NOT YET
        MEASURABLE rather than falsely 0% or falsely 100%."""
        fixture = _Fixture(tmp_path)
        fixture.stamped_and_resolved(_ruling(action='add_dependency'), observed_action=None)

        klass = fixture.report().for_class(_BRANCH_BEHIND)
        assert klass is not None
        assert klass.agreement_rate is None

    def test_classes_are_counted_separately(self, tmp_path: Path):
        fixture = _Fixture(tmp_path)
        fixture.stamped_and_resolved(_ruling(_BRANCH_BEHIND), observed_action='close_only')
        fixture.stamped_and_resolved(
            _ruling(_VETO_STREAK, action='resume'), observed_action='close_only',
        )

        report = fixture.report()
        branch = report.for_class(_BRANCH_BEHIND)
        veto = report.for_class(_VETO_STREAK)
        assert branch is not None and veto is not None
        assert branch.agreed == 1 and branch.diverged == 0
        assert veto.agreed == 0 and veto.diverged == 1

    def test_close_only_dismissed_records_are_counted(self, tmp_path: Path):
        """`close_only` resolves with dismiss=True, so status is 'dismissed',
        not 'resolved'. It is the MAJORITY outcome — a counter that only looked
        at 'resolved' would be vacuous."""
        fixture = _Fixture(tmp_path)
        record = fixture.stamped_and_resolved(_ruling(), observed_action='close_only')
        assert record.status == 'dismissed'

        klass = fixture.report().for_class(_BRANCH_BEHIND)
        assert klass is not None and klass.agreed == 1


class TestSelfResolvedBucket:
    """The bucket task 5361 made necessary.

    `escalation/src/escalation/classify.py::_HUMAN_RESOLVERS` contains
    `escalation-watcher`, so a watcher that stamps a proposal and then closes
    the record itself is read back in the SAME tier as a Leo ruling. Without
    this bucket the session inflates its own class toward its own adoption
    threshold, silently.
    """

    def test_watcher_is_classified_human_which_is_why_the_bucket_exists(self):
        """The fact that makes the bucket load-bearing rather than defensive.
        Pinned against the live function so that if the tier table ever stops
        calling the watcher human, this fails and a reader is sent to re-judge
        whether the bucket is still needed — instead of the justification
        quietly rotting in a comment."""
        assert classify_resolver_tier('escalation-watcher') == 'human'

    def test_a_5361_shaped_self_close_is_self_resolved_not_agreed(self, tmp_path: Path):
        """The exact shape: stamp as `escalation-watcher`, then close as
        `escalation-watcher` with action='close_only'."""
        fixture = _Fixture(tmp_path)
        fixture.stamped_and_resolved(
            _ruling(_SEMANTIC_COLLISION, action='close_only'), observed_action='close_only',
            stamped_by='escalation-watcher', resolved_by='escalation-watcher',
        )

        report = fixture.report()
        assert report.self_resolved == 1
        klass = report.for_class(_SEMANTIC_COLLISION)
        assert klass is None or (klass.agreed, klass.diverged, klass.not_comparable) == (0, 0, 0), (
            'a self-close must not be counted as an agreement even though its '
            'observed action matches the stamped proposal'
        )

    def test_a_self_close_does_not_move_the_rate(self, tmp_path: Path):
        fixture = _Fixture(tmp_path)
        fixture.stamped_and_resolved(
            _ruling(_SEMANTIC_COLLISION, action='close_only'), observed_action='resume',
            stamped_by='watcher-a', resolved_by='interactive',
        )
        before = fixture.report().for_class(_SEMANTIC_COLLISION)
        assert before is not None

        fixture.stamped_and_resolved(
            _ruling(_SEMANTIC_COLLISION, action='close_only'), observed_action='close_only',
            stamped_by='escalation-watcher', resolved_by='escalation-watcher',
        )
        after_report = fixture.report()
        after = after_report.for_class(_SEMANTIC_COLLISION)
        assert after is not None
        assert after.agreement_rate == before.agreement_rate
        assert (after.agreed, after.diverged, after.total) == (
            before.agreed, before.diverged, before.total,
        )
        assert after_report.self_resolved == 1

    def test_an_independent_adjudicator_stays_comparable(self, tmp_path: Path):
        """The NEGATIVE, so the bucket cannot over-capture: stamped by the AUTO
        watcher and resolved by the interactive watcher is the legitimate
        independent-adjudicator case."""
        fixture = _Fixture(tmp_path)
        fixture.stamped_and_resolved(
            _ruling(), observed_action='close_only',
            stamped_by='orchestrator-escalation-watcher-auto', resolved_by='escalation-watcher',
        )

        report = fixture.report()
        assert report.self_resolved == 0
        klass = report.for_class(_BRANCH_BEHIND)
        assert klass is not None and klass.agreed == 1

    def test_two_none_attributions_are_not_self_agreement(self, tmp_path: Path):
        """`None == None` must not read as a session agreeing with itself."""
        fixture = _Fixture(tmp_path)
        record = fixture.submit(resolution_action='close_only')
        stamped = fixture.queue.stamp_triage(
            record.id, triaged_by=None, triage_note=_stamped_note(_ruling()),
        )
        assert stamped is not None and stamped.triaged_by is None
        resolved = fixture.queue.resolve(record.id, 'ruled', dismiss=True, resolved_by=None)
        assert resolved is not None and resolved.resolved_by is None

        report = fixture.report()
        assert report.self_resolved == 0


class TestExclusionsAndWindow:
    def test_pending_records_are_unresolved_never_diverged(self, tmp_path: Path):
        fixture = _Fixture(tmp_path)
        record = fixture.submit(resolution_action=None)
        fixture.stamp(record, _ruling(), by='watcher-a')

        report = fixture.report()
        assert report.unresolved == 1
        assert report.for_class(_BRANCH_BEHIND) is None, (
            'a still-open record must not appear in any outcome bucket'
        )

    def test_a_gated_stamp_is_reported_not_averaged_away(self, tmp_path: Path):
        """A stamp the watcher should never have produced — made visible rather
        than folded into a rate."""
        fixture = _Fixture(tmp_path)
        fixture.stamped_and_resolved(
            _ruling(), observed_action='close_only', category='milestone_gate',
        )

        report = fixture.report()
        assert report.gated_stamps == 1
        assert report.for_class(_BRANCH_BEHIND) is None

    def test_a_gated_stamp_by_role_is_also_reported(self, tmp_path: Path):
        fixture = _Fixture(tmp_path)
        fixture.stamped_and_resolved(
            _ruling(), observed_action='close_only', agent_role='orchestrator-deterministic',
        )
        assert fixture.report().gated_stamps == 1

    def test_gating_wins_over_self_resolution(self, tmp_path: Path):
        """Order is part of the contract: gated -> self_resolved -> ... so a
        gated stamp is never ALSO counted somewhere that reads as a sample."""
        fixture = _Fixture(tmp_path)
        fixture.stamped_and_resolved(
            _ruling(), observed_action='close_only', category='milestone_gate',
            stamped_by='escalation-watcher', resolved_by='escalation-watcher',
        )

        report = fixture.report()
        assert report.gated_stamps == 1
        assert report.self_resolved == 0

    def test_a_record_resolved_inside_the_window_is_counted_on_both_edges(self, tmp_path: Path):
        fixture = _Fixture(tmp_path)
        record = fixture.stamped_and_resolved(_ruling(), observed_action='close_only')
        assert record.resolved_at is not None
        at = datetime.fromisoformat(record.resolved_at)

        klass = fixture.report(since=at, until=at).for_class(_BRANCH_BEHIND)
        assert klass is not None and klass.agreed == 1, 'both window edges are inclusive'

    def test_a_record_resolved_before_the_window_is_excluded(self, tmp_path: Path):
        fixture = _Fixture(tmp_path)
        record = fixture.stamped_and_resolved(_ruling(), observed_action='close_only')
        assert record.resolved_at is not None
        at = datetime.fromisoformat(record.resolved_at)

        report = fixture.report(since=at + _MICROSECOND, until=at + timedelta(days=1))
        assert report.for_class(_BRANCH_BEHIND) is None
        assert report.classes == ()

    def test_a_record_resolved_after_the_window_is_excluded(self, tmp_path: Path):
        fixture = _Fixture(tmp_path)
        record = fixture.stamped_and_resolved(_ruling(), observed_action='close_only')
        assert record.resolved_at is not None
        at = datetime.fromisoformat(record.resolved_at)

        report = fixture.report(since=at - timedelta(days=1), until=at - _MICROSECOND)
        assert report.for_class(_BRANCH_BEHIND) is None

    def test_a_cascade_resolver_is_reported_in_its_tier_not_as_a_human_agreement(
        self, tmp_path: Path,
    ):
        fixture = _Fixture(tmp_path)
        fixture.stamped_and_resolved(
            _ruling(), observed_action='close_only', resolved_by='l2-cascade:esc-9-1',
        )

        report = fixture.report()
        assert report.for_class(_BRANCH_BEHIND) is None, (
            'a cascade close is not a human ruling and must not inflate the rate'
        )
        assert report.resolver_tiers.get('cascade') == 1

    def test_a_sweep_resolver_is_reported_in_its_tier(self, tmp_path: Path):
        fixture = _Fixture(tmp_path)
        fixture.stamped_and_resolved(
            _ruling(), observed_action='close_only', resolved_by='auto-dismissed',
        )

        report = fixture.report()
        assert report.for_class(_BRANCH_BEHIND) is None
        assert report.resolver_tiers.get('reaper-sweep') == 1

    def test_human_resolutions_are_reported_in_the_human_tier(self, tmp_path: Path):
        fixture = _Fixture(tmp_path)
        fixture.stamped_and_resolved(_ruling(), observed_action='close_only')
        assert fixture.report().resolver_tiers.get('human') == 1


class TestSweepRobustness:
    def test_a_missing_queue_dir_yields_an_empty_report_without_raising(self, tmp_path: Path):
        report = agreement_report(
            tmp_path / 'does-not-exist',
            since=datetime(2000, 1, 1, tzinfo=UTC), until=datetime(2100, 1, 1, tzinfo=UTC),
        )
        assert report.classes == ()
        assert (report.gated_stamps, report.self_resolved, report.unresolved) == (0, 0, 0)

    def test_an_empty_queue_dir_yields_an_empty_report(self, tmp_path: Path):
        fixture = _Fixture(tmp_path)
        fixture.queue.queue_dir.mkdir(parents=True, exist_ok=True)
        assert fixture.report().classes == ()

    def test_a_record_with_no_marker_is_skipped_silently(self, tmp_path: Path):
        fixture = _Fixture(tmp_path)
        record = fixture.submit(resolution_action='close_only')
        fixture.queue.stamp_triage(record.id, triaged_by='watcher-a', triage_note=_FRESHNESS_NOTE)
        fixture.resolve(record, by='interactive')

        report = fixture.report()
        assert report.classes == ()
        assert (report.gated_stamps, report.self_resolved, report.unresolved) == (0, 0, 0)

    def test_an_unstamped_record_is_skipped_silently(self, tmp_path: Path):
        fixture = _Fixture(tmp_path)
        record = fixture.submit(resolution_action='close_only')
        fixture.resolve(record, by='interactive')
        assert fixture.report().classes == ()

    def test_the_report_and_its_class_rows_are_frozen(self, tmp_path: Path):
        fixture = _Fixture(tmp_path)
        fixture.stamped_and_resolved(_ruling(), observed_action='close_only')
        report = fixture.report()
        klass = report.for_class(_BRANCH_BEHIND)
        assert klass is not None

        with pytest.raises(dataclasses.FrozenInstanceError):
            report.gated_stamps = 5  # type: ignore[misc]
        with pytest.raises(dataclasses.FrozenInstanceError):
            klass.agreed = 5  # type: ignore[misc]


def test_comparable_actions_stay_in_lockstep_with_the_c1_vocabulary():
    """COMPARABLE_ACTIONS is a local copy, so it needs a pin.

    `escalation.server` is not importable from `escalation.shadow_ruling`
    without inverting the layer direction (a pure archive reader would drag in
    fastmcp), so the identity is held here by a cross-module TEST import — the
    convention `escalation/src/escalation/authority.py` already uses for the
    watcher identity string. Add a C1 action to the reversible list and this
    fails rather than silently reporting the new action as `not_comparable`.
    """
    from escalation.server import RESOLVE_ACTIONS

    assert REVERSIBLE_ACTIONS & frozenset(RESOLVE_ACTIONS) == COMPARABLE_ACTIONS


# ---------------------------------------------------------------------------
# The CLI arm the skill quotes (step-7)
# ---------------------------------------------------------------------------


def _row_for(out: str, ruling_class: str) -> str:
    rows = [line for line in out.splitlines() if ruling_class in line]
    assert len(rows) == 1, f'expected exactly one row for {ruling_class!r}, got {rows}'
    return rows[0]


class TestCliTable:
    def test_prints_a_row_per_class_with_counts_and_rate(self, tmp_path: Path, capsys):
        fixture = _Fixture(tmp_path)
        for _ in range(3):
            fixture.stamped_and_resolved(_ruling(), observed_action='close_only')
        fixture.stamped_and_resolved(_ruling(action='resume'), observed_action='close_only')
        fixture.stamped_and_resolved(_ruling(action='file_task'), observed_action=None)

        code = main([
            '--queue-dir', str(fixture.queue.queue_dir),
            '--since', '2000-01-01T00:00:00+00:00',
            '--until', '2100-01-01T00:00:00+00:00',
        ])

        assert code == 0
        row = _row_for(capsys.readouterr().out, _BRANCH_BEHIND)
        for field in ('3', '1', '4', '75'):
            assert field in row, f'expected {field!r} in the class row: {row!r}'

    def test_prints_the_three_excluded_buckets(self, tmp_path: Path, capsys):
        fixture = _Fixture(tmp_path)
        fixture.stamped_and_resolved(_ruling(), observed_action='close_only')
        fixture.stamped_and_resolved(
            _ruling(), observed_action='close_only', category='milestone_gate',
        )
        fixture.stamped_and_resolved(
            _ruling(), observed_action='close_only',
            stamped_by='escalation-watcher', resolved_by='escalation-watcher',
        )
        pending = fixture.submit()
        fixture.stamp(pending, _ruling(), by='watcher-a')

        assert main(['--queue-dir', str(fixture.queue.queue_dir)]) == 0
        out = capsys.readouterr().out
        assert 'gated_stamps=1' in out
        assert 'self_resolved=1' in out
        assert 'unresolved=1' in out

    def test_self_resolved_is_printed_even_when_zero(self, tmp_path: Path, capsys):
        """A reader deciding whether a class cleared "95% over at least 10
        items" must see, from the pasted output alone, how many of its records
        were thrown out for self-agreement. A class whose stamps are mostly
        self_resolved is NOT MEASURABLE YET and must not read like a class with
        a small sample."""
        fixture = _Fixture(tmp_path)
        fixture.stamped_and_resolved(_ruling(), observed_action='close_only')

        assert main(['--queue-dir', str(fixture.queue.queue_dir)]) == 0
        assert 'self_resolved=0' in capsys.readouterr().out

    def test_self_resolved_is_not_folded_into_unresolved_or_the_class_row(
        self, tmp_path: Path, capsys,
    ):
        fixture = _Fixture(tmp_path)
        fixture.stamped_and_resolved(_ruling(), observed_action='close_only')
        for _ in range(2):
            fixture.stamped_and_resolved(
                _ruling(), observed_action='close_only',
                stamped_by='escalation-watcher', resolved_by='escalation-watcher',
            )

        assert main(['--queue-dir', str(fixture.queue.queue_dir)]) == 0
        out = capsys.readouterr().out
        assert 'self_resolved=2' in out
        assert 'unresolved=0' in out
        row = _row_for(out, _BRANCH_BEHIND)
        assert '2' not in row.replace(_BRANCH_BEHIND, ''), (
            f'the two self-resolved records leaked into the class row: {row!r}'
        )

    def test_an_unmeasurable_rate_is_not_printed_as_a_number(self, tmp_path: Path, capsys):
        fixture = _Fixture(tmp_path)
        fixture.stamped_and_resolved(_ruling(action='file_task'), observed_action=None)

        assert main(['--queue-dir', str(fixture.queue.queue_dir)]) == 0
        assert 'n/a' in _row_for(capsys.readouterr().out, _BRANCH_BEHIND)


class TestCliWindow:
    def test_defaults_to_the_trailing_seven_days(self, tmp_path: Path, capsys):
        fixture = _Fixture(tmp_path)
        fixture.stamped_and_resolved(_ruling(), observed_action='close_only')

        assert main(['--queue-dir', str(fixture.queue.queue_dir)]) == 0
        out = capsys.readouterr().out
        assert _BRANCH_BEHIND in out, 'a record resolved just now falls in the trailing week'

    def test_echoes_the_resolved_window_so_a_pasted_report_is_self_describing(
        self, tmp_path: Path, capsys,
    ):
        fixture = _Fixture(tmp_path)
        fixture.queue.queue_dir.mkdir(parents=True, exist_ok=True)

        before = datetime.now(UTC)
        assert main(['--queue-dir', str(fixture.queue.queue_dir)]) == 0
        after = datetime.now(UTC)

        out = capsys.readouterr().out
        windows = [
            datetime.fromisoformat(token)
            for token in out.replace(',', ' ').split()
            if token.count('-') >= 2 and 'T' in token
        ]
        assert len(windows) == 2, f'expected the window echoed as two timestamps: {out!r}'
        since, until = windows
        assert before - timedelta(days=7, seconds=5) <= since <= after - timedelta(days=7)
        assert before <= until <= after

    def test_an_explicit_window_is_honoured(self, tmp_path: Path, capsys):
        fixture = _Fixture(tmp_path)
        record = fixture.stamped_and_resolved(_ruling(), observed_action='close_only')
        assert record.resolved_at is not None
        at = datetime.fromisoformat(record.resolved_at)

        assert main([
            '--queue-dir', str(fixture.queue.queue_dir),
            '--since', (at - timedelta(days=1)).isoformat(),
            '--until', (at - _MICROSECOND).isoformat(),
        ]) == 0
        assert _BRANCH_BEHIND not in capsys.readouterr().out


class TestCliDegradesLoudly:
    def test_a_missing_queue_dir_is_a_nonzero_exit_on_stderr(self, tmp_path: Path, capsys):
        """An empty report and a misconfigured path must not look identical: an
        all-zero table reads like a real measurement."""
        code = main(['--queue-dir', str(tmp_path / 'nope')])

        assert code != 0
        captured = capsys.readouterr()
        assert 'nope' in captured.err
        assert 'gated_stamps' not in captured.out, (
            'a misconfigured path must not print a table at all'
        )

    def test_an_empty_window_says_so_rather_than_printing_nothing(
        self, tmp_path: Path, capsys,
    ):
        fixture = _Fixture(tmp_path)
        fixture.queue.queue_dir.mkdir(parents=True, exist_ok=True)

        assert main(['--queue-dir', str(fixture.queue.queue_dir)]) == 0
        assert 'no shadow rulings in window' in capsys.readouterr().out

    # -- the window arguments (step-13) ------------------------------------
    #
    # EVERY case below drives a stamped AND resolved record. The naive-argument
    # crash is LATENT ON AN EMPTY ARCHIVE: the naive-vs-aware comparison lives
    # in the window filter, which is only reached once a parsed shadow stamp
    # exists, so a case that omitted the record would pass vacuously against
    # the unfixed code and prove nothing. Measured before these were written:
    # with a record present, a naive `--since` raised
    # `TypeError: can't compare offset-naive and offset-aware datetimes`.
    #
    # The aware path's own regression guard is TestCliWindow above —
    # `test_an_explicit_window_is_honoured` and
    # `test_defaults_to_the_trailing_seven_days` — which must stay green rather
    # than be restated here.

    def test_a_naive_since_is_coerced_and_the_record_is_still_counted(
        self, tmp_path: Path, capsys,
    ):
        """The operator-facing case, in the operator's own spelling: a bare
        `--since 2026-09-01` is valid ISO-8601 and simply carries no offset."""
        fixture = _Fixture(tmp_path)
        fixture.stamped_and_resolved(_ruling(), observed_action='close_only')

        code = main(['--queue-dir', str(fixture.queue.queue_dir), '--since', '2026-09-01'])

        assert code == 0
        assert _BRANCH_BEHIND in capsys.readouterr().out, (
            'a record resolved inside the coerced window was not counted'
        )

    def test_a_naive_until_is_coerced_too(self, tmp_path: Path, capsys):
        """The same defect, reached through the other flag: with `--since` left
        to default, `since = until - 7 days` inherits the naiveness, so a fix
        exercised only through `--since` would leave half of it live.

        The bound is derived from the record rather than written as a literal
        future date, which would silently stop covering anything once passed.
        """
        fixture = _Fixture(tmp_path)
        record = fixture.stamped_and_resolved(_ruling(), observed_action='close_only')
        assert record.resolved_at is not None
        at = datetime.fromisoformat(record.resolved_at)
        naive_until = (at + timedelta(days=1)).replace(tzinfo=None).isoformat()

        code = main(['--queue-dir', str(fixture.queue.queue_dir), '--until', naive_until])

        assert code == 0
        assert _BRANCH_BEHIND in capsys.readouterr().out, (
            'the default 7-day window below a coerced --until dropped the record'
        )

    def test_a_coerced_window_is_echoed_as_aware_utc(self, tmp_path: Path, capsys):
        """Which INSTANT a naive argument was read as is a decision the report
        must show, not one the reader has to assume — the self-describing
        property TestCliWindow requires, extended to cover the coercion."""
        fixture = _Fixture(tmp_path)
        fixture.stamped_and_resolved(_ruling(), observed_action='close_only')

        assert main([
            '--queue-dir', str(fixture.queue.queue_dir), '--since', '2026-09-01',
        ]) == 0
        window = next(
            line for line in capsys.readouterr().out.splitlines() if 'window' in line
        )
        assert '2026-09-01T00:00:00+00:00' in window, (
            f'the coerced window must echo the instant actually used: {window!r}'
        )

    def test_a_naive_window_still_excludes_an_out_of_range_record(
        self, tmp_path: Path, capsys,
    ):
        """Coercion must repair the comparison, not defeat the filter."""
        fixture = _Fixture(tmp_path)
        record = fixture.stamped_and_resolved(_ruling(), observed_action='close_only')
        assert record.resolved_at is not None
        at = datetime.fromisoformat(record.resolved_at)
        naive = (at - _MICROSECOND).replace(tzinfo=None).isoformat()

        assert main(['--queue-dir', str(fixture.queue.queue_dir), '--until', naive]) == 0
        assert _BRANCH_BEHIND not in capsys.readouterr().out

    @pytest.mark.parametrize('flag', ['--since', '--until'])
    def test_a_malformed_window_value_names_the_flag_and_prints_no_table(
        self, tmp_path: Path, capsys, flag: str,
    ):
        """Same discipline as the `--queue-dir` case above: a misconfiguration
        must never be mistaken for an all-zero measurement. Asserted on the
        RETURNED int, not `pytest.raises(SystemExit)` — the guard belongs after
        `parse_args`, beside the `--queue-dir` branch, not in an argparse
        `type=` callback that raises from inside it."""
        fixture = _Fixture(tmp_path)
        fixture.stamped_and_resolved(_ruling(), observed_action='close_only')

        code = main(['--queue-dir', str(fixture.queue.queue_dir), flag, 'yesterday'])

        assert code == 2
        captured = capsys.readouterr()
        assert flag in captured.err
        assert 'yesterday' in captured.err
        assert 'gated_stamps' not in captured.out, (
            'a misconfigured window must not print a table at all'
        )


class TestCliJson:
    def test_json_parses_back_to_the_same_counts_as_the_table(self, tmp_path: Path, capsys):
        fixture = _Fixture(tmp_path)
        fixture.stamped_and_resolved(_ruling(), observed_action='close_only')
        fixture.stamped_and_resolved(_ruling(action='resume'), observed_action='close_only')
        fixture.stamped_and_resolved(
            _ruling(), observed_action='close_only', category='milestone_gate',
        )
        fixture.stamped_and_resolved(
            _ruling(), observed_action='close_only',
            stamped_by='escalation-watcher', resolved_by='escalation-watcher',
        )

        assert main(['--queue-dir', str(fixture.queue.queue_dir), '--json']) == 0
        payload = json.loads(capsys.readouterr().out)

        assert payload['gated_stamps'] == 1
        assert payload['self_resolved'] == 1
        assert payload['unresolved'] == 0
        row = next(c for c in payload['classes'] if c['class'] == _BRANCH_BEHIND)
        assert (row['agreed'], row['diverged'], row['not_comparable']) == (1, 1, 0)
        assert row['comparable'] == 2
        assert row['agreement_rate'] == 0.5
        assert payload['since'] and payload['until']

    def test_json_reports_an_unmeasurable_rate_as_null(self, tmp_path: Path, capsys):
        fixture = _Fixture(tmp_path)
        fixture.stamped_and_resolved(_ruling(action='file_task'), observed_action=None)

        assert main(['--queue-dir', str(fixture.queue.queue_dir), '--json']) == 0
        payload = json.loads(capsys.readouterr().out)
        row = next(c for c in payload['classes'] if c['class'] == _BRANCH_BEHIND)
        assert row['agreement_rate'] is None

    def test_json_is_emitted_for_an_empty_window_too(self, tmp_path: Path, capsys):
        fixture = _Fixture(tmp_path)
        fixture.queue.queue_dir.mkdir(parents=True, exist_ok=True)

        assert main(['--queue-dir', str(fixture.queue.queue_dir), '--json']) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload['classes'] == []
        assert payload['self_resolved'] == 0
