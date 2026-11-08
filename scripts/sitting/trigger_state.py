"""The sitting trigger evaluator's last-run file: the single home of its shape.

The trigger evaluator (ε, ``scripts/sitting/sitting_trigger.py``) is the only
writer; ``return_brief.py`` is the reader that renders the docket's trigger
state line. The evaluator's own cooldown and daily-cap bookkeeping is a
separate file it owns, not this one.

Strict both ways: a body whose key set is not exactly the record's, whose
outcome is unknown or whose stamp does not parse is
:class:`TriggerStateUnreadable`, never a silently defaulted value.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from shared.safe_io import atomic_write_text

DEFAULT_PATH = Path(__file__).resolve().parents[2] / 'data' / 'sitting' / 'trigger-last-run.json'


class TriggerOutcome(StrEnum):
    """What one evaluator run decided."""

    NOT_FIRED = 'not_fired'
    LAUNCHED = 'launched'
    ABSENT = 'absent'
    SUPPRESSED = 'suppressed'


class TriggerStateUnreadable(Exception):
    """The file at :attr:`path` exists but does not hold a last-run record."""

    def __init__(self, path: Path, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f'{path}: trigger last-run state is unreadable ({reason})')


@dataclass(frozen=True)
class TriggerLastRun:
    ran_at: str
    outcome: TriggerOutcome
    reason: str
    last_launch_at: str | None
    lane: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, 'outcome', TriggerOutcome(self.outcome))
        _require_instant('ran_at', self.ran_at)
        if self.last_launch_at is not None:
            _require_instant('last_launch_at', self.last_launch_at)
        if not isinstance(self.reason, str):
            raise ValueError(f'reason is text, not {self.reason!r}')
        if self.lane is not None and not isinstance(self.lane, str):
            raise ValueError(f'lane is text or None, not {self.lane!r}')

    def to_json(self) -> str:
        return json.dumps({**asdict(self), 'outcome': str(self.outcome)}, indent=2, sort_keys=True) + '\n'

    @classmethod
    def from_json(cls, text: str) -> TriggerLastRun:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError(f'the body is {type(data).__name__}, not an object')
        expected = {field.name for field in fields(cls)}
        if set(data) != expected:
            raise ValueError(f'keys must be exactly {sorted(expected)}, got {sorted(data)}')
        return cls(**data)


def load(path: Path | str = DEFAULT_PATH) -> TriggerLastRun | None:
    """The last run recorded at *path*; None when the evaluator has never written it."""
    path = Path(path)
    try:
        text = path.read_text(encoding='utf-8')
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as exc:
        raise TriggerStateUnreadable(path, str(exc)) from exc
    try:
        return TriggerLastRun.from_json(text)
    except (TypeError, ValueError) as exc:
        raise TriggerStateUnreadable(path, f'{type(exc).__name__}: {exc}') from exc


def save(path: Path | str, state: TriggerLastRun) -> None:
    atomic_write_text(path, state.to_json(), mkdir=True)


def _require_instant(name: str, stamp: object) -> None:
    try:
        if not isinstance(stamp, str):
            raise TypeError(type(stamp).__name__)
        datetime.fromisoformat(stamp)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} {stamp!r} is not an ISO-8601 timestamp') from exc
