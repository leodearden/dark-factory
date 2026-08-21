"""Meta-guard: every whole-tree AST-scanning test in this directory must carry a
module-level ``pytest.mark.timeout`` (task 4215).

THE FAILURE MODE, END TO END.  A dozen guard tests under ``orchestrator/tests/``
sweep the ENTIRE repo -- ``rglob('*.py')`` over ~500 files -- and ``ast.parse``
each one.  Unloaded and serial that costs 6-9s per call here; the orchestrator
suite however runs under ``-n auto`` (``orchestrator/pyproject.toml:178``), so on
a 32-core box those sweeps run concurrently with everything else and the same
call has been MEASURED at 17.85 / 21.32 / 30.75s at loadavg 120-176 -- a ~4.8x
load inflation.  Worker deaths were then observed at loadavg 250-423, one
further inflation step past the 60s default at ``pyproject.toml:152``.

What makes the breach so expensive is the two settings around it:

* ``timeout_method = "thread"`` (``pyproject.toml:153``) means a breach does NOT
  fail the test -- pytest-timeout's thread method ``os._exit()``s the whole xdist
  worker ("node down: Not properly terminated");
* ``--max-worker-restart=0`` (``pyproject.toml:178``, task 1907) then declines to
  replace that worker, degrading the run to a TRUNCATED whole-suite session whose
  surviving failure names some innocent guard that merely happened to share the
  dead worker.

So a single slow tree-scan does not cost one red test -- it costs the whole
verify run, and it misattributes the blame.  Escalations esc-3980-1 and
esc-3787-1 are both instances.

THE FIX the pyproject itself sanctions ("Slow tests opt out with
``@pytest.mark.timeout(N)``", ``pyproject.toml:154``) is a module-level
``pytestmark`` on each scanner.  This module pins the two constants that mark
uses, and (from step-5 on) recomputes the census of scanners from source on
every run so the Nth new scanner is caught at commit time rather than by another
truncated verify.

NOT widening the global ``timeout``: that would blunt the 60s ceiling for the
other ~16000 tests, which is exactly the ceiling that catches real hangs.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

from _orch_helpers import PYPROJECT_DEFAULT_TIMEOUT, WHOLE_TREE_SCAN_TEST_TIMEOUT

# Resolved from THIS FILE, never the process CWD: merge-verify runs pytest from
# the orchestrator/ cwd while a plain `pytest orchestrator/tests` runs from the
# repo root, and both must resolve identically. Same idiom as
# test_marker_registration_drift.py:597-601.
_ORCH_PYPROJECT = Path(__file__).resolve().parents[1] / 'pyproject.toml'


def _pytest_ini_options() -> dict[str, object]:
    """``[tool.pytest.ini_options]`` parsed from the REAL orchestrator pyproject."""
    data = tomllib.loads(_ORCH_PYPROJECT.read_text(encoding='utf-8'))
    return data['tool']['pytest']['ini_options']


class TestTimeoutConstants:
    """The two constants this guard's remediation advice depends on."""

    def test_pyproject_default_timeout_mirrors_pyproject(self) -> None:
        """``PYPROJECT_DEFAULT_TIMEOUT`` must equal the REAL configured default.

        The constant is a mirror of ``[tool.pytest.ini_options].timeout``, and
        its previous incarnation (test_merge_queue_concurrent_verify.py, task
        3492) admitted the defect in its own comment: "it is still a literal and
        CAN drift if that setting changes without a matching edit here; there is
        no automated link between the two."  This test IS that link -- it reads
        orchestrator/pyproject.toml with ``tomllib`` at runtime rather than
        citing a line number in a comment, which is precisely how the earlier
        mirror was free to go stale.

        ``timeout_method`` is asserted alongside it because the entire cost model
        above rests on it: under thread mode a breach ``os._exit()``s the xdist
        worker instead of failing the test.  If someone switches back to signal
        mode, the rationale for every mark this guard demands changes, and that
        should surface loudly here rather than be discovered later.
        """
        ini_options = _pytest_ini_options()

        assert ini_options['timeout'] == PYPROJECT_DEFAULT_TIMEOUT, (
            f'PYPROJECT_DEFAULT_TIMEOUT ({PYPROJECT_DEFAULT_TIMEOUT}) no longer '
            f"mirrors [tool.pytest.ini_options].timeout ({ini_options['timeout']}) "
            f'in {_ORCH_PYPROJECT}. Update the constant in '
            'orchestrator/tests/_orch_helpers.py -- and re-check that '
            'WHOLE_TREE_SCAN_TEST_TIMEOUT, which is derived from it, still '
            'clears the measured worst case.'
        )
        assert ini_options['timeout_method'] == 'thread', (
            'timeout_method is no longer "thread" in '
            f'{_ORCH_PYPROJECT} (got {ini_options["timeout_method"]!r}). The '
            'whole rationale for this guard -- a breach os._exit()s the xdist '
            'worker rather than failing the test, and --max-worker-restart=0 '
            'turns that into a truncated whole-suite run -- is specific to '
            'thread mode. Revisit this module before changing it.'
        )

    def test_whole_tree_scan_timeout_clears_default_with_margin(self) -> None:
        """``WHOLE_TREE_SCAN_TEST_TIMEOUT`` must clear the default several times over.

        Measured basis, all first-hand except where the task record is cited:

        * 8.25s / 6.70s / 6.46s per call unloaded and serial (``-n0``) on a
          32-core box for test_merge_queue_reachback_patch_guard,
          test_event_loop_antipattern_guard and
          test_serial_merge_worker_import_guard respectively;
        * 17.85 / 21.32 / 30.75s per call for test_serial_merge_worker_import_guard
          at loadavg 120-176 (task 4215's record) -- ~4.8x its unloaded figure;
        * xdist worker deaths observed at loadavg 250-423 (esc-3980-1,
          esc-3787-1), i.e. past the 60s default.

        A 5x multiple leaves ~36x headroom over the unloaded worst case and ~10x
        over the measured-under-load worst case.  Asserted as ``>=`` rather than
        ``==`` so raising the constant later is never blocked by this test --
        the never-narrow polarity the neighbouring shared timeouts use.
        """
        assert WHOLE_TREE_SCAN_TEST_TIMEOUT >= 5 * PYPROJECT_DEFAULT_TIMEOUT, (
            f'WHOLE_TREE_SCAN_TEST_TIMEOUT ({WHOLE_TREE_SCAN_TEST_TIMEOUT}) must '
            f'clear 5x the pyproject default ({5 * PYPROJECT_DEFAULT_TIMEOUT}s). '
            'A whole-tree AST sweep measured 30.75s at loadavg 120-176 and the '
            'xdist worker deaths this guard exists to prevent were seen at '
            'loadavg 250-423; anything tighter re-arms that cliff.'
        )
