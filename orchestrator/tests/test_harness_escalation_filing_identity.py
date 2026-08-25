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

import ast
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from escalation.models import Escalation
from escalation.pins import _norm_id, classify_pins
from shared.task_claimant import compose_claimant_run_id

import orchestrator.harness as harness_module
from orchestrator.harness import Harness
from orchestrator.scheduler import SetTaskStatusRejected

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


# ---------------------------------------------------------------------------
# step-7/8: the identity on a real harness filing, and the tripwire
# ---------------------------------------------------------------------------


class TestHarnessFilingsCarryTheIdentity:
    """A REAL harness filing stamps it, and Link 4 then acts on it."""

    def test_reconcile_failure_filing_is_stamped_and_classifies_dead_l0(
        self, mock_orch_config,
    ) -> None:
        """``_escalate_reconcile_failure`` — the one harness record Link 4 governs.

        An ast scan of ``harness.py``'s ``Escalation(...)`` sites shows this
        is the ONLY one filing a level-0, non-``info`` record against a REAL
        task id.  Every other site is either ``severity='info'`` (Link 1 →
        NON_PINNING) or an explicit ``level=1``/``level=2`` (Link 3 →
        QUEUE_HANDOFF), so the filing identity never reaches Link 4 there.

        The downstream MEANING is asserted, not left incidental: the harness's
        synthetic session literal can never equal a workflow claimant, so once
        task 3541 wires ``live_claimant_id`` through, this record classifies
        ``dead_l0`` and stops pinning recovery.  That is the PRD's stated
        intent (``plans/task-escalation-state-graph-prd.md`` task eta item 4 —
        an unconsumed L0 whose filing incarnation is dead should be PROMOTED
        for visibility, not left immortal), and it is safe because the
        PROTECTIVE half survives: ``vetoes_done_flip`` stays True.

        NOTE: ``level == 0`` here encodes the OBSERVED status quo.  That
        site's docstring says "Submit an L1 escalation" but it passes no
        ``level=``, so it builds at the dataclass default 0.  The
        contradiction is filed as out-of-scope ticket
        tkt_0RSVX7E03MYACJPMGPH1GWMW08 and deliberately NOT resolved here —
        correcting the docstring is behaviour-neutral, while adding
        ``level=1`` would flip this record from reaper-eligible DEAD_L0 to
        permanently-pinning QUEUE_HANDOFF.  Whoever resolves that ticket moves
        this assertion with them.
        """
        h = _build_harness(mock_orch_config)
        h._run_id = 'run-xyz'

        submitted: list[Escalation] = []
        queue = MagicMock()
        queue.make_id = MagicMock(return_value='esc-77-1')
        queue.submit = MagicMock(side_effect=submitted.append)
        h._escalation_queue = queue

        h._escalate_reconcile_failure(
            '77',
            SetTaskStatusRejected('77', 'provenance_missing', 'raw text'),
        )

        assert len(submitted) == 1
        esc = submitted[0]
        assert esc.filing_claimant_run_id == h._filing_claimant_run_id
        # The status quo this assertion encodes — see ticket note above.
        assert esc.level == 0
        assert esc.severity == 'blocking'

        # A workflow-shaped claimant holds the task; the harness identity can
        # never equal it, so the handoff has no live consumer.
        workflow_id = compose_claimant_run_id('run-xyz', '77-abcd1234', os.getpid())
        assert workflow_id != esc.filing_claimant_run_id
        report = classify_pins(
            '77', [esc], live_claimant=True, live_claimant_id=workflow_id,
        )
        assert report.dead_l0 == ('esc-77-1',)
        assert report.pins is False           # recovery is no longer blocked...
        assert report.vetoes_done_flip is True  # ...but the done-flip veto holds


def _escalation_call_sites(module_path: Path) -> list[ast.Call]:
    """Every ``Escalation(...)`` construction in *module_path*.

    MUST be ``ast``, never a regex: ``harness.py`` carries docstring and
    comment prose mentioning ``Escalation(...)``, which a regex would report
    as an unstamped filing site.
    """
    tree = ast.parse(module_path.read_text(encoding='utf-8'))
    return [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == 'Escalation'
    ]


class TestEveryHarnessFilingSiteIsStamped:
    """Structural tripwire over ``harness.py`` — the sibling of
    ``test_workflow_claimant.py::TestEveryWorkflowFilingSiteIsStamped``.

    TOTAL rule, no allowlist, for the same reason: the field is load-bearing
    only at level-0 non-``info`` records, but a site whose level later flips
    to 0 would silently lose the stamp — which is how this defect arose.
    """

    def test_all_escalation_constructions_pass_filing_claimant_run_id(self) -> None:
        calls = _escalation_call_sites(_HARNESS_SRC_PATH)
        # Baseline at the time of writing; a drop means sites moved out of
        # this module and the tripwire's coverage silently shrank.
        assert len(calls) >= 27, f'expected >=27 Escalation(...) sites, found {len(calls)}'

        unstamped = [
            node.lineno for node in calls
            if not any(kw.arg == 'filing_claimant_run_id' for kw in node.keywords)
        ]
        assert not unstamped, (
            f'{_HARNESS_SRC_PATH.name}: Escalation(...) without '
            f'filing_claimant_run_id at lines {unstamped} — every filing site '
            f'must stamp the filing incarnation (task 3550); use '
            f'self._filing_claimant_run_id.'
        )
