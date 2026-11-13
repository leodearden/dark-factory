"""Make every EscalationQueue construction, make_id and submit call report to a LoopFreedomProbe before its real work."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from _fm_helpers import LoopFreedomProbe
from escalation.queue import EscalationQueue


def _probed(real: Callable[..., Any], probe: LoopFreedomProbe) -> Callable[..., Any]:
    def wrapper(self: EscalationQueue, *args: Any, **kwargs: Any) -> Any:
        probe.block()
        return real(self, *args, **kwargs)

    return wrapper


def probe_escalation_queue_io(monkeypatch: pytest.MonkeyPatch, probe: LoopFreedomProbe) -> None:
    for name in ('__init__', 'make_id', 'submit'):
        monkeypatch.setattr(EscalationQueue, name, _probed(getattr(EscalationQueue, name), probe))
