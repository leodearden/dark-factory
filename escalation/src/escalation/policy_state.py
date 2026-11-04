"""The single owner of each escalation queue's per-class policy switch file.

PRD ``plans/decision-stream-prd.md`` §3 decision 2: every queue carries
``<queue_dir>/policy-table-state.json``, holding the
:class:`~escalation.authority.ClassSwitch` of each
:data:`~escalation.authority.POLICY_TABLE` class on that queue. The escalation
server reads it on every executor call; the sitting's audit verbs and the
storm escape write it through :func:`set_class_state`. Nothing else reads or
writes the file.

A missing file, or a class the file does not name, means that row's
``initial_state``. A file that cannot be read as that shape raises
:class:`PolicyStateUnreadable` instead of defaulting, so a corrupt table is
never mistaken for a healthy one.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

from shared.safe_io import atomic_write_text

from escalation.authority import POLICY_TABLE, ClassSwitch, check_class_transition, policy_row
from escalation.queue import escalation_id_lock

POLICY_STATE_FILENAME = 'policy-table-state.json'

# Locks policy-table-state.json.lock. The startup sweep reaps only
# esc-*.json.lock sidecars, so this one is never unlinked under a writer.
_LOCK_ID = 'policy-table-state'


class PolicyStateUnreadable(ValueError):
    """The state file exists but is not ``{"classes": {slug: switch}}`` over table classes."""

    def __init__(self, path: Path, why: str) -> None:
        super().__init__(f'{path}: {why}')
        self.path = path


def policy_state_path(queue_dir: Path) -> Path:
    return Path(queue_dir) / POLICY_STATE_FILENAME


def _initial_states() -> dict[str, ClassSwitch]:
    return {row.policy_class: row.initial_state for row in POLICY_TABLE}


def _read_overrides(path: Path) -> dict[str, ClassSwitch]:
    try:
        text = path.read_text()
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeDecodeError) as exc:
        raise PolicyStateUnreadable(path, f'unreadable ({exc})') from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PolicyStateUnreadable(path, f'not JSON ({exc})') from exc
    if not isinstance(data, dict):
        raise PolicyStateUnreadable(path, f'top level is {type(data).__name__}, not an object')
    classes = data.get('classes')
    if not isinstance(classes, dict):
        raise PolicyStateUnreadable(path, f"'classes' is {type(classes).__name__}, not an object")
    overrides: dict[str, ClassSwitch] = {}
    for policy_class, switch in classes.items():
        if policy_row(policy_class) is None:
            raise PolicyStateUnreadable(path, f'{policy_class!r} names no POLICY_TABLE class')
        try:
            overrides[policy_class] = ClassSwitch(switch)
        except (TypeError, ValueError) as exc:
            raise PolicyStateUnreadable(
                path,
                f'{policy_class!r} has switch {switch!r}; '
                f'expected one of {[member.value for member in ClassSwitch]}',
            ) from exc
    return overrides


def load_class_states(queue_dir: Path) -> Mapping[str, ClassSwitch]:
    """Every table class's switch on *queue_dir*'s queue, read-only."""
    return MappingProxyType(_initial_states() | _read_overrides(policy_state_path(queue_dir)))


def set_class_state(queue_dir: Path, policy_class: str, target: ClassSwitch) -> ClassSwitch:
    """Move *policy_class* to *target* on *queue_dir*'s queue and return its previous switch.

    Moving a class to the switch it already has is a no-op. Any other move
    must be in :data:`~escalation.authority.CLASS_SWITCH_TRANSITIONS`, or
    :class:`~escalation.authority.IllegalClassTransition` is raised and
    nothing is written.
    """
    row = policy_row(policy_class)
    if row is None:
        raise ValueError(f'no POLICY_TABLE class named {policy_class!r}')
    with escalation_id_lock(queue_dir, _LOCK_ID):
        states = load_class_states(queue_dir)
        current = states[policy_class]
        if target == current:
            return current
        check_class_transition(policy_class, current, target)
        initial = _initial_states()
        overrides = {
            slug: str(switch)
            for slug, switch in sorted((dict(states) | {policy_class: target}).items())
            if switch != initial[slug]
        }
        atomic_write_text(
            policy_state_path(queue_dir),
            json.dumps({'classes': overrides}, indent=2) + '\n',
            mkdir=True,
        )
        return current
