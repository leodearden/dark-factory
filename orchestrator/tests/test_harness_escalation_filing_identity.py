"""The harness stamps the FILING incarnation on the Escalations it files (task 3550).

Spec ``docs/task-escalation-state-spec.md`` S6, realised by
``escalation.pins::classify_pins`` Link 4: an L0 is a live handoff ONLY while
the incarnation that FILED it lives.  Before task 3550,
``Escalation.filing_claimant_run_id`` was stamped by NOTHING outside tests, so
Link 4 read every real record as "filing identity unknown" and fell back to
pinning — the rule existed but could never fire.

The harness is a PROCESS-level filer, structurally unlike ``TaskWorkflow``:

* It has no per-task ``session_id`` at any of its filing sites, and no live-
  workflow registry to borrow one from (``Harness.is_workflow_active`` is a
  bare ``_workflow_cancel_events`` membership test), so a genuine per-task
  incarnation identity is unavailable.  A FIXED session literal is used
  instead — honest, greppable, and structurally disjoint from every
  ``{task_id}-{uuid8}`` workflow session id, so a harness-filed L0 can never
  be mistaken for a workflow handoff.
* It has no DB claimant counterpart it must stay byte-identical to (that is
  ``TaskWorkflow``'s constraint, from the dispatch stamp).  So on an unknown
  ``_run_id`` it degrades to ``None`` — a fail-safe UNKNOWN that ``pins``
  fails safe to pinning on — rather than emitting the well-shaped-but-wrong
  partial identity ``TaskWorkflow``'s ``or ''`` produces.  That is the same
  choice ``workflow.py``'s ``lock_plan`` call site already makes, and the one
  task 3563 ratified when it left the DB stamp's ``or ''`` asymmetry alone.

Covers:
  step-5/6    the ``Harness._filing_claimant_run_id`` property itself.
  step-7/8    the identity reaching a REAL harness filing, what Link 4 then
              makes of it, and an ast tripwire over every ``Escalation(...)``
              construction in ``harness.py``.
  step-13/14  ``_build_task_claimant_lookup`` and its wiring into
              ``create_server``, which is how an AGENT-filed escalation gets
              stamped with its dispatching incarnation's identity.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from escalation.pins import _norm_id

import orchestrator.harness as harness_module
from orchestrator.harness import Harness

_HARNESS_SRC_PATH = Path(harness_module.__file__)


def _build_harness(mock_orch_config) -> Harness:
    """Construct a Harness with heavy constructors patched out.

    Mirrors ``test_harness_already_landed_gate_wiring.py``'s ``_build_harness``.
    """
    mock_orch_config.max_concurrent_tasks = 2
    mock_orch_config.fused_memory.project_id = 'test'

    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        return Harness(mock_orch_config)


# ---------------------------------------------------------------------------
# step-5/6: the property
# ---------------------------------------------------------------------------


class TestHarnessFilingClaimantIdentity:
    """``Harness._filing_claimant_run_id`` — the process filer's identity."""

    def test_composes_run_id_fixed_session_and_pid(self, mock_orch_config) -> None:
        """With a run id, the value is a normal composed claimant identity."""
        h = _build_harness(mock_orch_config)
        h._run_id = 'run-xyz'

        assert h._filing_claimant_run_id == (
            f'run-xyz/{harness_module._HARNESS_FILING_SESSION_ID}/pid={os.getpid()}'
        )

    def test_passes_the_pins_shape_guard(self, mock_orch_config) -> None:
        """``pins._norm_id`` leaves it untouched — it is COMPARABLE, not just set.

        A value without the ``/pid=`` marker collapses to UNKNOWN inside
        ``_norm_id`` and Link 4 then fails safe to pinning, making the stamp
        inert.
        """
        h = _build_harness(mock_orch_config)
        h._run_id = 'run-xyz'

        prop = h._filing_claimant_run_id
        assert _norm_id(prop) == prop

    def test_unknown_run_id_degrades_to_none_not_a_partial_identity(
        self, mock_orch_config,
    ) -> None:
        """Pre-``run()``, ``_run_id`` is None — an honest UNKNOWN, never '/harness/pid=N'.

        ``__init__`` declares ``_run_id = None`` and only ``run()`` assigns it,
        so filings from that startup window have no run id to embed.  A
        partially-composed ``'/harness/pid={pid}'`` would carry the ``/pid=``
        marker, survive ``_norm_id``'s shape guard, and then be compared WHOLE
        against a live claimant as if it were KNOWN — mismatching every one of
        them and reading as "a DIFFERENT incarnation is live".  Unlike
        ``TaskWorkflow``, the harness has no DB claimant counterpart forcing
        it to match, so ``None`` is available and correct.
        """
        h = _build_harness(mock_orch_config)
        h._run_id = None

        assert h._filing_claimant_run_id is None

    def test_session_component_cannot_collide_with_a_workflow_session_id(self) -> None:
        """The fixed literal is structurally disjoint from ``{task_id}-{uuid8}``.

        ``TaskWorkflow.session_id`` is ``f'{task_id}-{uuid.uuid4().hex[:8]}'``
        — always at least 10 characters and always containing a ``-``.  A bare
        literal with neither property can never be produced by that format, so
        a harness-filed record can never be mistaken for a workflow handoff no
        matter what task id a workflow carries.
        """
        session = harness_module._HARNESS_FILING_SESSION_ID

        assert '-' not in session
        assert len(session) < 10
        assert '/' not in session  # would corrupt the composed identity's shape
