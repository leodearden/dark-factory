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
never mistaken for a healthy one. A slug that names no current row (one since
renamed or removed) makes :func:`load_class_states` raise too, but
:func:`set_class_state` drops it, so the write verb can repair the file the
read refuses.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

from shared.safe_io import atomic_write_text

from escalation.authority import POLICY_TABLE, ClassSwitch, check_class_transition, policy_row
from escalation.queue import escalation_id_lock

logger = logging.getLogger(__name__)

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


def _read_overrides(path: Path) -> tuple[dict[str, ClassSwitch], frozenset[str]]:
    """The file's switch for each table class it names, and the slugs it names that no row has."""
    try:
        text = path.read_text()
    except FileNotFoundError:
        return {}, frozenset()
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
    stale: set[str] = set()
    for policy_class, switch in classes.items():
        if policy_row(policy_class) is None:
            stale.add(policy_class)
            continue
        try:
            overrides[policy_class] = ClassSwitch(switch)
        except (TypeError, ValueError) as exc:
            raise PolicyStateUnreadable(
                path,
                f'{policy_class!r} has switch {switch!r}; '
                f'expected one of {[member.value for member in ClassSwitch]}',
            ) from exc
    return overrides, frozenset(stale)


def load_class_states(queue_dir: Path) -> Mapping[str, ClassSwitch]:
    """Every table class's switch on *queue_dir*'s queue, read-only."""
    path = policy_state_path(queue_dir)
    overrides, stale = _read_overrides(path)
    if stale:
        raise PolicyStateUnreadable(path, f'{sorted(stale)} name no POLICY_TABLE class')
    return MappingProxyType(_initial_states() | overrides)


def set_class_state(queue_dir: Path, policy_class: str, target: ClassSwitch) -> ClassSwitch:
    """Move *policy_class* to *target* on *queue_dir*'s queue and return its previous switch.

    Moving a class to the switch it already has writes nothing, unless the
    file names a slug no current row has: this write drops every such slug,
    with a warning. Any other move must be in
    :data:`~escalation.authority.CLASS_SWITCH_TRANSITIONS`, or
    :class:`~escalation.authority.IllegalClassTransition` is raised and
    nothing is written.
    """
    if policy_row(policy_class) is None:
        raise ValueError(f'no POLICY_TABLE class named {policy_class!r}')
    path = policy_state_path(queue_dir)
    with escalation_id_lock(queue_dir, _LOCK_ID):
        overrides, stale = _read_overrides(path)
        states = _initial_states() | overrides
        current = states[policy_class]
        if target == current and not stale:
            return current
        if target != current:
            check_class_transition(policy_class, current, target)
        if stale:
            logger.warning('%s: dropping %s, which name no POLICY_TABLE class', path, sorted(stale))
        _write_overrides(path, states | {policy_class: target})
        return current


def _write_overrides(path: Path, states: Mapping[str, ClassSwitch]) -> None:
    """Persist the switches in *states* that differ from their row's initial state."""
    initial = _initial_states()
    overrides = {
        slug: str(switch) for slug, switch in sorted(states.items()) if switch != initial[slug]
    }
    atomic_write_text(path, json.dumps({'classes': overrides}, indent=2) + '\n', mkdir=True)
