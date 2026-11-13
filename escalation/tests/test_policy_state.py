"""Per-queue policy-class switches: ``<queue_dir>/policy-table-state.json``.

``escalation.policy_state`` is the single owner of that file (task 8422; PRD
``plans/decision-stream-prd.md`` §3 decision 2). These tests drive it against a
real temporary queue directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType

import pytest
from _escalation_seed import seed_escalation

from escalation.authority import POLICY_TABLE, ClassSwitch, IllegalClassTransition
from escalation.policy_state import (
    POLICY_STATE_FILENAME,
    PolicyStateUnreadable,
    load_class_states,
    set_class_state,
)
from escalation.queue import EscalationQueue


def _initial_states() -> dict[str, ClassSwitch]:
    return {row.policy_class: row.initial_state for row in POLICY_TABLE}


def _write_state(queue_dir: Path, text: str) -> Path:
    queue_dir.mkdir(parents=True, exist_ok=True)
    path = queue_dir / POLICY_STATE_FILENAME
    path.write_text(text)
    return path


class TestLoadClassStates:
    def test_missing_file_gives_every_row_its_initial_state(self, tmp_path: Path) -> None:
        states = load_class_states(tmp_path)
        assert isinstance(states, MappingProxyType)
        assert dict(states) == _initial_states()

    def test_file_overlays_the_initial_states(self, tmp_path: Path) -> None:
        _write_state(tmp_path, json.dumps({'classes': {'found_on_main': 'live'}}))
        states = load_class_states(tmp_path)
        assert states['found_on_main'] is ClassSwitch.LIVE
        expected = _initial_states() | {'found_on_main': ClassSwitch.LIVE}
        assert dict(states) == expected

    @pytest.mark.parametrize(
        'text',
        (
            pytest.param('{not json', id='unparsable'),
            pytest.param('["classes"]', id='top-level-not-object'),
            pytest.param('{"classes": ["found_on_main"]}', id='classes-not-object'),
            pytest.param('{"classes": {"found_on_main": "maybe"}}', id='unknown-switch'),
            pytest.param('{"classes": {"no_such_class": "live"}}', id='unknown-class'),
        ),
    )
    def test_unreadable_state_names_the_file(self, tmp_path: Path, text: str) -> None:
        path = _write_state(tmp_path, text)
        with pytest.raises(PolicyStateUnreadable, match=str(path)):
            load_class_states(tmp_path)


class TestSetClassState:
    def test_shadow_to_live_persists(self, tmp_path: Path) -> None:
        previous = set_class_state(tmp_path, 'landing_provenance', ClassSwitch.LIVE)

        assert previous is ClassSwitch.SHADOW
        assert (tmp_path / POLICY_STATE_FILENAME).is_file()
        assert POLICY_STATE_FILENAME == 'policy-table-state.json'
        states = load_class_states(tmp_path)
        assert dict(states) == _initial_states() | {'landing_provenance': ClassSwitch.LIVE}

    def test_live_to_shadow_persists(self, tmp_path: Path) -> None:
        set_class_state(tmp_path, 'landing_provenance', ClassSwitch.LIVE)
        previous = set_class_state(tmp_path, 'landing_provenance', ClassSwitch.SHADOW)

        assert previous is ClassSwitch.LIVE
        assert load_class_states(tmp_path)['landing_provenance'] is ClassSwitch.SHADOW

    def test_shadow_to_off_is_illegal_and_writes_nothing(self, tmp_path: Path) -> None:
        with pytest.raises(IllegalClassTransition):
            set_class_state(tmp_path, 'landing_provenance', ClassSwitch.OFF)
        assert not (tmp_path / POLICY_STATE_FILENAME).exists()

    def test_off_to_live_is_illegal_and_leaves_the_file_unchanged(self, tmp_path: Path) -> None:
        path = _write_state(tmp_path, json.dumps({'classes': {'landing_provenance': 'off'}}))
        before = path.read_bytes()

        with pytest.raises(IllegalClassTransition):
            set_class_state(tmp_path, 'landing_provenance', ClassSwitch.LIVE)

        assert path.read_bytes() == before

    def test_setting_the_current_state_is_a_no_op(self, tmp_path: Path) -> None:
        previous = set_class_state(tmp_path, 'landing_provenance', ClassSwitch.SHADOW)

        assert previous is ClassSwitch.SHADOW
        assert not (tmp_path / POLICY_STATE_FILENAME).exists()

    def test_unknown_class_is_refused_by_name(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match='no_such_class'):
            set_class_state(tmp_path, 'no_such_class', ClassSwitch.LIVE)

    def test_a_move_drops_a_slug_no_row_has_and_keeps_the_rest(self, tmp_path: Path) -> None:
        """A renamed or removed row's slug must not leave the repair verb unusable."""
        _write_state(
            tmp_path, json.dumps({'classes': {'retired_class': 'live', 'found_on_main': 'live'}}),
        )

        previous = set_class_state(tmp_path, 'landing_provenance', ClassSwitch.LIVE)

        assert previous is ClassSwitch.SHADOW
        expected = _initial_states() | {
            'found_on_main': ClassSwitch.LIVE, 'landing_provenance': ClassSwitch.LIVE,
        }
        assert dict(load_class_states(tmp_path)) == expected

    def test_setting_the_current_state_on_a_stale_file_repairs_it(self, tmp_path: Path) -> None:
        _write_state(tmp_path, json.dumps({'classes': {'retired_class': 'not-a-switch'}}))

        previous = set_class_state(tmp_path, 'landing_provenance', ClassSwitch.SHADOW)

        assert previous is ClassSwitch.SHADOW
        assert dict(load_class_states(tmp_path)) == _initial_states()

    def test_an_illegal_move_on_a_stale_file_leaves_it_unchanged(self, tmp_path: Path) -> None:
        path = _write_state(tmp_path, json.dumps({'classes': {'retired_class': 'live'}}))
        before = path.read_bytes()

        with pytest.raises(IllegalClassTransition):
            set_class_state(tmp_path, 'landing_provenance', ClassSwitch.OFF)

        assert path.read_bytes() == before


def test_the_queue_never_reads_the_state_file_as_an_escalation(tmp_path: Path) -> None:
    queue = EscalationQueue(tmp_path / 'esc')
    seeded = [seed_escalation(queue, level=1, task_id=str(n)) for n in (1, 2)]
    set_class_state(queue.queue_dir, 'landing_provenance', ClassSwitch.LIVE)

    assert sorted(esc.id for esc in queue.get_pending()) == sorted(esc.id for esc in seeded)
