"""cockpit.panes.session_table — pure session-table formatting + the DataTable widget.

Resolves PRD open-Q4 (failed-spawn/gone-window glyph set): the six Status
members fold into four glyphs -- awaiting-input is "blocked on you",
running/launching are "working", idle is "idle", and exited/
failed-to-start are "dead". An unrecognized/foreign status degrades to
'?' rather than raising (additive-safe, mirroring session_registry's own
no-coerce policy for spawn_mode/display).
"""

from __future__ import annotations

from orchestrator.session_registry import Status

_GLYPHS: dict[Status, str] = {
    Status.AWAITING_INPUT: '⏸',
    Status.RUNNING: '⚙',
    Status.LAUNCHING: '⚙',
    Status.IDLE: '✓',
    Status.EXITED: '☠',
    Status.FAILED_TO_START: '☠',
}

_FALLBACK_GLYPH = '?'


def state_glyph(status: Status | str) -> str:
    """Map a Status (or its wire string) to its display glyph.

    A foreign/unrecognized status value returns the fallback glyph rather
    than raising -- this must stay total over any status the registry
    hands it (fail-soft, PRD §2).
    """
    try:
        resolved = Status(status)
    except ValueError:
        return _FALLBACK_GLYPH
    return _GLYPHS.get(resolved, _FALLBACK_GLYPH)
