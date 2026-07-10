"""cockpit.backends.fakes — FakeBackend, the Protocol-level spy (PRD §6.2 / C4).

FakeBackend implements FocusArrangeBackend directly, in memory, recording
every call into a per-method bucket list — no subprocess, no I/O. It's the
mechanism C5b's B3 relies on: a refresh path that calls only
set_urgency/reorder must leave focus_calls/tile_calls empty. Distinct from
the injected fake CommandRunner used to exercise the real wm/tmux backends
(see test_backends_wm.py / test_backends_tmux.py).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from cockpit.backends.base import DisplayTarget, FocusResult, Zone

AliveSpec = bool | Mapping[DisplayTarget, bool] | Callable[[DisplayTarget], bool]


class FakeBackend:
    """In-memory FocusArrangeBackend spy.

    `focus_result` scripts what every focus() call returns (default
    FocusResult(ok=True)). `alive` scripts is_alive(): a plain bool applies
    to every target, a {target: bool} mapping overrides specific targets and
    falls back to True for targets not listed, and a callable(target) -> bool
    is invoked directly.
    """

    def __init__(
        self,
        focus_result: FocusResult | None = None,
        alive: AliveSpec = True,
    ) -> None:
        self._focus_result = focus_result if focus_result is not None else FocusResult(ok=True)
        self._alive = alive
        self.focus_calls: list[DisplayTarget] = []
        self.set_urgency_calls: list[tuple[DisplayTarget, bool]] = []
        self.reorder_calls: list[Sequence[DisplayTarget]] = []
        self.tile_calls: list[tuple[Sequence[DisplayTarget], Zone]] = []
        self.is_alive_calls: list[DisplayTarget] = []

    def focus(self, target: DisplayTarget) -> FocusResult:
        self.focus_calls.append(target)
        return self._focus_result

    def set_urgency(self, target: DisplayTarget, on: bool) -> None:
        self.set_urgency_calls.append((target, on))

    def reorder(self, targets: Sequence[DisplayTarget]) -> None:
        self.reorder_calls.append(targets)

    def tile(self, targets: Sequence[DisplayTarget], zone: Zone) -> None:
        self.tile_calls.append((targets, zone))

    def is_alive(self, target: DisplayTarget) -> bool:
        self.is_alive_calls.append(target)
        if callable(self._alive):
            return self._alive(target)
        if isinstance(self._alive, Mapping):
            return self._alive.get(target, True)
        return bool(self._alive)
