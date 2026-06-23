"""B+H integration-gate suite for the deterministic task kind, B1–B12.

Exercises the full integration of the deterministic task kind over the landed
β(1899)/γ(1900)/ε(1902)/α(1898)/δ(1901) PRD implementations:
 - Pure gate path: born-at-L2 escalation, quiescence, proceed/no-go resolution,
   restart stamp-clear re-fire, orchestrator-restart replay, strand-reaper
   invisibility, no-lock (B1–B5, B11, B12).
 - Auto-deploy path: cross-unit success, failure+reaper no-rerun, self-restart
   scheduled, δ submit CLI L2, α validation rejection corners (B6–B10).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from escalation.models import Escalation
from escalation.queue import EscalationQueue

from orchestrator.deterministic_runner import DeterministicRunner
from orchestrator.harness import Harness
from orchestrator.scheduler import Scheduler, TaskAssignment
from orchestrator.workflow import WorkflowOutcome
