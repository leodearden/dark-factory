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

import pytest

from escalation.authority import L2_AUTO_CLOSE_DENY_CATEGORIES, L2_AUTO_CLOSE_DENY_ROLES
from escalation.models import Escalation
from escalation.shadow_ruling import (
    DETECTABLE_GATES,
    FIRST_TRANCHE_CLASSES,
    GATED_CATEGORIES,
    GATED_ROLES,
    HUMAN_FOREVER_GATES,
    REVERSIBLE_ACTIONS,
    SHADOW_RULING_MARKER,
    ShadowRuling,
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
