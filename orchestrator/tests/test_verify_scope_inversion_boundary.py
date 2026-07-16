"""B+H boundary suite for the verify-scope-inversion PRD (task ξ).

plans/verify-scope-inversion-prd.md's task ξ is the PRD's **integration
gate**: ONE module driving the REAL merge-queue/workflow verify seams BOTH
WAYS (producer plan/gate state AND consumer observable) for every row of the
PRD's 10-row Boundary-test sketch. Every scenario below asserts BOTH sides of
its seam. This module passing green in CI is the leaf signal that the whole
inversion composes end-to-end — the merge-gate correctness guarantee.

Every capability asserted here is delivered by upstream siblings, all DONE on
main before this task started:

- κ (2588): plan-authoritative execution — ``run_scoped_verification``
  derives→executes→aggregates the ``VerifyPlan``; ``VerifyResult.plan`` is
  the EXECUTED plan (verify.py:4503, :2516).
- λ (2589): role-differentiated policy + the ``merge_verify_breadth`` knob —
  ``derive_verify_plan(role=...)`` forks; merge+full expands every
  registered ``ModuleConfig`` to its FULL_SUITE per-module; task-role adds
  an owning-module pytest floor; breadth='scoped' stays byte-identical
  legacy (config.py/defaults.yaml/verify_plan.py).
- μ (2590): merge-gate baseline attribution — ``seed_main_baseline`` /
  ``cached_main_baseline_failing_ids`` / ``diff_new_failures`` (verify.py);
  junit ``failing_test_ids`` on ``VerifyResult``; the merge worker cites
  NEW-only ids vs routing MAIN_HEALTH_RED (merge_queue.py
  ``_classify_main_health_red`` / ``_run_post_merge_verify``).
- ν (2591): infra-transient outcomes consume zero attempts at both the
  workflow ``_verify_debugfix_loop`` and merge accounting
  (``INFRA_TRANSIENT_CATEGORIES``, verify_categories.py).

Row → seam map (PRD's 10-row Boundary-test sketch):

=====  =========================================  ===============================================
Row    Name                                        Seam(s) driven
=====  =========================================  ===============================================
1      source-only sibling break rejected          producer: ``run_scoped_verification`` (merge+full
                                                     widens to the untouched sibling module); consumer:
                                                     ``_run_post_merge_verify`` (real merge gate, BLOCKED
                                                     citing the sibling's failing id)
2      task-role signal                            producer+observable: ``run_scoped_verification``
                                                     (role='task' — the floor never widens beyond the
                                                     owning module)
3      docs-only trivial, both roles                producer+consumer: ``run_scoped_verification`` at
                                                     both role='task'/'merge' (TRIVIAL short-circuit,
                                                     zero commands)
4      new-vs-preexisting baseline attribution      consumer: ``_run_post_merge_verify`` (mixed
                                                     baseline — blocked citing only the NEW id)
5      wholly pre-existing + cache hit              consumer: ``_run_post_merge_verify`` x2 (MAIN_HEALTH_RED
                                                     routing, no branch charge, one probe across two merges)
6      infra non-consumption, both consumers        task-verify: ``TaskWorkflow._verify_debugfix_loop``;
                                                     merge-verify: ``_run_post_merge_verify`` accounting
7      train amortization                           ``MergeWorker._do_merge`` (exactly one broad,
                                                     per-module merge-role verify of the train tip)
8      rollback path (breadth='scoped')              producer: ``run_scoped_verification`` (byte-identical
                                                     legacy merge-role plan)
9      fallback narrowing (never the global chain)   producer: ``run_scoped_verification`` (scoped fallback
                                                     only, never the whole-repo opaque chain)
10     plan authority, both roles                    producer: ``run_scoped_verification`` (executed ==
                                                     ``plan.runs``; ``result.plan`` is the EXECUTED plan)
=====  =========================================  ===============================================

Harness approach — drive REAL seams both ways, fake runners, no ssh (the
1737/2260/2309 shape): the PRODUCER/plan+execution side calls the REAL
``run_scoped_verification`` under an instrumented fake
``orchestrator.verify.run_verification`` (never a real subprocess) and
inspects the executed ``ModuleConfig``(s) / ``VerifyResult.plan`` /
``VerifyResult.summary``. The CONSUMER/merge-gate side drives the REAL
``_run_post_merge_verify`` chokepoint (merge_queue.py), opting OUT of the
autouse ``_mock_merge_queue_verification`` stub via
``@pytest.mark.exercise_merge_verify`` so ``orchestrator.merge_queue.
run_scoped_verification`` stays bound to the REAL function — which, in turn,
calls the SAME patched ``orchestrator.verify.run_verification`` fake. One
patched seam, two chokepoints, both real end-to-end.

conftest.py's autouse fixtures (``_clear_probe_cache`` — resets BOTH
``verify._PROBE_CACHE`` and ``verify._BASELINE_FAILING_IDS_CACHE`` before
and after every test — ``_mock_merge_queue_verification``,
``_neutralize_verify_admission``) already apply to this module; nothing to
import or opt into beyond the ``exercise_merge_verify`` marker on the rows
that need it. Neither this module nor its dependencies touch conftest.py or
any production file — scope is this single new test module.

Golden diff (row 1) — CONSTRUCTED, not mined: the confusion-codebook's
``verify-scope-asymmetry`` entry (docs/legibility/confusion-codebook.yaml:75,
16 ``sightings_2026_06``) records the cluster's *count* but its ``sightings``
list is empty — no per-incident diff is extractable from it. A quick
git-log mining pass (``git log --all -i --grep=MAIN_HEALTH_RED`` /
``--grep='sibling.*test'`` over verify.py/merge_queue.py, and repo-wide
``--grep='fix-forward'``) surfaced only the (also-DONE) task-2564/task-1690
main-health machinery commits themselves — no standalone minimal
source-breaks-sibling-test incident diff to mine, confirming the codebook's
empty ``sightings`` list. Per the task's recipe and capability manifest, row
1 therefore uses the CONSTRUCTED minimal two-module shape (a source edit
under module A that breaks a test in sibling module B) — explicitly labeled
constructed here, NOT dressed up with a fabricated "historical" attribution.
The cluster's 16 incidents validate the premise either way: a source-only
diff breaking a sibling module's suite escapes a scoped merge gate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal
from unittest.mock import AsyncMock, patch

import pytest
from test_merge_queue_main_health import _make_config, _make_git_ops, _make_req
from test_verify_scope_kappa import _executed_module_configs, _run_verification_spy
from test_verify_scope_lambda import _two_module_registry

from orchestrator import verify
from orchestrator.config import ModuleConfig, OrchestratorConfig
from orchestrator.merge_queue import (
    MAIN_HEALTH_RED_REASON_PREFIX,
    MergeOutcome,
    _run_post_merge_verify,
)
from orchestrator.verify import VerifyResult, run_scoped_verification, seed_main_baseline
from orchestrator.verify_categories import INFRA_TRANSIENT_CATEGORIES  # noqa: F401 — row 6 (later iteration)

# ---------------------------------------------------------------------------
# Golden diff (row 1) — CONSTRUCTED. See module docstring "Golden diff
# (row 1)" above for why this is constructed rather than mined.
# ---------------------------------------------------------------------------

# The touched file (the "diff" proper) — a plain source edit under module A.
MODA_SOURCE_PATH: str = 'moda/helpers.py'
MODA_SOURCE_CONTENT: str = 'def helper():\n    return 1\n'

# The untouched sibling test coupled to modA's contract — present on disk for
# the narrative fidelity of the two-module shape, but never itself part of
# any row's task_files: modB is discovered purely via the registry + a
# merge+full breadth expansion (see _two_module_registry, reused unmodified
# from test_verify_scope_lambda.py), never because it was directly touched.
MODB_SIBLING_TEST_PATH: str = 'modb/tests/test_sibling.py'
MODB_SIBLING_TEST_CONTENT: str = (
    'from moda.helpers import helper\n\n\n'
    'def test_sibling_contract():\n'
    '    assert helper() == 1\n'
)

# The pytest node id the row-1/4/5 instrumented fakes report as failing for
# modB — shaped like a real junit-derived id (VerifyResult.failing_test_ids).
MODB_FAILING_TEST_ID: str = 'modb/tests/test_sibling.py::test_sibling_contract'


def _row1_golden_diff(
    tmp_path: Path, *, breadth: Literal['scoped', 'full'] = 'full',
) -> tuple[ModuleConfig, ModuleConfig, OrchestratorConfig]:
    """Write the CONSTRUCTED row-1 golden diff to *tmp_path* and return the
    2-module registry (modA/modB) it belongs to.

    Modeled on (and delegates to) ``test_verify_scope_lambda._two_module_registry``
    for the ModuleConfig/OrchestratorConfig registry — this function's own
    addition is writing the golden-diff FILES: modA's touched source file
    (``MODA_SOURCE_PATH``) and modB's untouched sibling test
    (``MODB_SIBLING_TEST_PATH``), matching row 1's "a source edit under
    module A that breaks a test in sibling module B" shape. Reused by every
    row built on this same golden diff (1, 2, 8).
    """
    mod_a, mod_b, config = _two_module_registry(tmp_path, breadth=breadth)

    moda_full = tmp_path / MODA_SOURCE_PATH
    moda_full.parent.mkdir(parents=True, exist_ok=True)
    moda_full.write_text(MODA_SOURCE_CONTENT)

    modb_full = tmp_path / MODB_SIBLING_TEST_PATH
    modb_full.parent.mkdir(parents=True, exist_ok=True)
    modb_full.write_text(MODB_SIBLING_TEST_CONTENT)

    return mod_a, mod_b, config
