"""BackgroundService + LifecycleRegistry — one reusable seam collapsing the
eleven background-loop/service lifecycles in harness.py.

PRD ``plans/harness-supervision-prd.md`` §5.3 (LR-1/2/3): a bounded, uniform
start/stop contract for every long-lived harness background task so the
recurring shutdown-hang class (survey 2.3; tasks 108/161/162/169/875/1080)
becomes structurally impossible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Parity default (Open-Q Q5): the only current failure backoff across the
# eleven harness background loops is this fixed constant — mirrors
# harness._BG_LOOP_FAILURE_BACKOFF_SECS verbatim (see
# test_background_service.py::test_default_backoff_secs_matches_harness_parity_constant).
DEFAULT_BACKOFF_SECS: float = 60.0


@dataclass(frozen=True)
class BackoffPolicy:
    """Backoff delay applied after a failed BackgroundService pass.

    Minimal constant-delay value object: Open-Q Q5 parity requires carrying
    the CURRENT backoff verbatim, and the only current failure backoff across
    the eleven loops is a fixed constant (no exponential/attempt-dependent
    shape exists today). ``delay_for`` still accepts an ``attempt`` argument
    so the call-site shape stays stable if a non-constant policy is
    introduced later.
    """

    delay_secs: float

    def delay_for(self, attempt: int) -> float:
        del attempt  # constant policy: same delay regardless of attempt
        return self.delay_secs
