"""The lane-resolution contract, driven as pure functions (task 4888).

``escalation/src/escalation/merge_lane_resolution.py`` owns the whole lane
decision for the MCP submit path: validate a caller-supplied lane, apply the
precedence ``argument > metadata.merge_lane > 'normal'``, and report WHICH
source won.  It is pure — no I/O, no awaits, no server capture — so this
module needs no server, no queue, no registry and no fake worker to pin the
rule the server then merely wires up.

Every test drives the PUBLIC interface only.  The server-side tests in
``test_merge_request_lane.py`` assert the WIRING (which lane reaches the
enqueued ``MergeRequest``); neither reaches into internals to check the
other's job.

The asymmetry these tests pin is the point of the module: an INHERITED
metadata value normalises silently (a lane resolution must never fail a merge
submission), while a CALLER-SUPPLIED one is rejected loudly (silently
discarding live operator intent is the defect task 4888 exists to fix).
"""
from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from escalation.merge_lane_resolution import (
    InvalidMergeLane,
    LaneChoice,
    resolve_merge_lane,
    validate_requested_lane,
)

# ---------------------------------------------------------------------------
# Cross-package orchestrator imports — mirrors test_merge_state_wiring.py's guard.
# ---------------------------------------------------------------------------
try:
    from orchestrator.merge_queue import MERGE_LANES  # type: ignore[reportMissingImports]
    _ORCHESTRATOR_AVAILABLE = True
except ImportError:
    _ORCHESTRATOR_AVAILABLE = False
    MERGE_LANES: Any = None  # type: ignore[assignment,misc]


class TestValidateRequestedLane:
    """The loud half: what a CALLER may supply."""

    def test_none_passes_through(self):
        """``None`` means "no argument supplied" and is not an error."""
        assert validate_requested_lane(None) is None

    @pytest.mark.parametrize('lane', ['high', 'normal'])
    def test_in_vocabulary_lane_passes_through(self, lane):
        assert validate_requested_lane(lane) == lane

    @pytest.mark.parametrize(
        'bad',
        [
            'higgh',  # the motivating typo
            '',  # empty string is NOT "unspecified"; None is
            'HIGH',  # case matters — the vocabulary is exact
            0,  # arrives off the MCP wire, so non-str is reachable
            True,
            ['high'],
        ],
        ids=['typo', 'empty', 'wrong-case', 'int', 'bool', 'list'],
    )
    def test_out_of_vocabulary_lane_raises(self, bad):
        with pytest.raises(InvalidMergeLane):
            validate_requested_lane(bad)

    def test_raised_error_exposes_value_and_vocabulary_structurally(self):
        """The reject carries STRUCTURED data, not a prose message to re-parse.

        ``merge_request`` renders its ``{error, code, hint}`` envelope from
        these attributes, so the valid-lane list in the hint cannot drift
        from the vocabulary the check keyed on.
        """
        with pytest.raises(InvalidMergeLane) as excinfo:
            validate_requested_lane('higgh')
        err = excinfo.value
        assert err.value == 'higgh'
        assert tuple(err.valid_lanes) == tuple(MERGE_LANES or err.valid_lanes)

    def test_invalid_merge_lane_is_a_value_error(self):
        """So a caller that does not know this type still catches it."""
        assert issubclass(InvalidMergeLane, ValueError)


class TestResolveMergeLanePrecedence:
    """``argument > metadata.merge_lane > 'normal'``, with provenance."""

    def test_explicit_argument_wins_over_metadata(self):
        choice = resolve_merge_lane(
            requested='normal', task_metadata={'merge_lane': 'high'}
        )
        assert choice == LaneChoice(lane='normal', source='argument')

    def test_explicit_high_argument_reports_argument_source(self):
        choice = resolve_merge_lane(requested='high', task_metadata={})
        assert choice == LaneChoice(lane='high', source='argument')

    def test_metadata_is_honoured_when_no_argument_given(self):
        choice = resolve_merge_lane(requested=None, task_metadata={'merge_lane': 'high'})
        assert choice == LaneChoice(lane='high', source='task_metadata')

    @pytest.mark.parametrize('metadata', [{}, None], ids=['empty', 'none'])
    def test_no_argument_and_no_metadata_is_the_default(self, metadata):
        choice = resolve_merge_lane(requested=None, task_metadata=metadata)
        assert choice == LaneChoice(lane='normal', source='default')

    def test_inherited_typo_normalises_silently_and_reports_its_source(self):
        """The deliberate asymmetry against ``validate_requested_lane``.

        An inherited value was written by a different actor at a different
        time and must never be able to fail a submission, so it fails OPEN to
        ``'normal'`` — but the source is still reported honestly as
        ``'task_metadata'``, because the key WAS present.
        """
        choice = resolve_merge_lane(
            requested=None, task_metadata={'merge_lane': 'higgh'}
        )
        assert choice == LaneChoice(lane='normal', source='task_metadata')

    def test_metadata_source_is_presence_not_truthiness(self):
        """A present-but-empty ``merge_lane`` still came FROM the metadata.

        ``'normal'`` alone cannot distinguish "the task asked for normal"
        from "nothing asked at all" — which is the entire reason
        ``LaneChoice`` carries a source.
        """
        choice = resolve_merge_lane(requested=None, task_metadata={'merge_lane': ''})
        assert choice == LaneChoice(lane='normal', source='task_metadata')

    def test_metadata_asking_for_normal_is_not_reported_as_default(self):
        choice = resolve_merge_lane(
            requested=None, task_metadata={'merge_lane': 'normal'}
        )
        assert choice == LaneChoice(lane='normal', source='task_metadata')


class TestLaneChoiceShape:
    def test_lane_choice_is_frozen(self):
        choice = resolve_merge_lane(requested='high', task_metadata=None)
        with pytest.raises(dataclasses.FrozenInstanceError):
            choice.lane = 'normal'  # type: ignore[misc]


@pytest.mark.skipif(not _ORCHESTRATOR_AVAILABLE, reason='orchestrator package not installed')
class TestVocabularyIsNotForked:
    """The module must never mint a lane vocabulary of its own."""

    @pytest.mark.parametrize(
        ('requested', 'metadata'),
        [
            (None, None),
            (None, {}),
            (None, {'merge_lane': 'high'}),
            (None, {'merge_lane': 'normal'}),
            (None, {'merge_lane': 'higgh'}),
            (None, {'merge_lane': 42}),
            ('high', {'merge_lane': 'normal'}),
            ('normal', {'merge_lane': 'high'}),
        ],
    )
    def test_every_resolvable_lane_is_a_merge_lanes_member(self, requested, metadata):
        choice = resolve_merge_lane(requested=requested, task_metadata=metadata)
        assert choice.lane in MERGE_LANES

    def test_validate_accepts_exactly_the_merge_lanes_vocabulary(self):
        for lane in MERGE_LANES:
            assert validate_requested_lane(lane) == lane

    def test_lane_source_vocabulary_is_closed(self):
        sources = {
            resolve_merge_lane(requested='high', task_metadata=None).source,
            resolve_merge_lane(requested=None, task_metadata={'merge_lane': 'high'}).source,
            resolve_merge_lane(requested=None, task_metadata={}).source,
        }
        assert sources == {'argument', 'task_metadata', 'default'}
