"""Digest helpers for AFK hardening — per-N-escalation markdown digests + EWA trip.

Task 1327: Every N escalation events the harness writes an append-only markdown
digest summarising recent activity, and tracks an EWA of escalations/done that
pauses the scheduler when it trips.

All I/O in this module is fail-open: helpers return sentinels / zeros / None
and log warnings rather than raising.  The digest is observability, not a
correctness gate.

Design decisions (see plan.json):
- Pure, Harness-free helpers here; harness.py owns the trigger and state.
- EWA state is process-local (reset on restart — consistent with park-stop counters).
- write_digest_entry never raises; digest_dir is auto-created if missing.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# EWA math
# ---------------------------------------------------------------------------


def update_ewa(
    prev_ewa: float,
    escalations_in_step: int,
    done_in_step: int,
    alpha: float,
) -> float:
    """Compute one EWA step.

    EWA(t+1) = alpha * (escalations_in_step / max(done_in_step, 1))
               + (1 - alpha) * prev_ewa

    done_in_step == 0 uses denominator 1 so a step with escalations and zero
    completions (the worst-case signal) pushes EWA up rather than crashing.

    No exception handling — pure arithmetic.
    """
    ratio = escalations_in_step / max(done_in_step, 1)
    return alpha * ratio + (1 - alpha) * prev_ewa
