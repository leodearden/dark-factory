"""Tests for the Harness<->Scheduler constructor-injected callback seam
(task 2235, W10-alpha).

Replaces the nine post-construction ``scheduler._on_*`` monkey-patches
(installed in ``Harness.__init__`` after the Scheduler is built) with a
single ``SchedulerCallbacks`` bundle passed to ``Scheduler(config,
callbacks=...)`` at construction time.

Covers:
  (a) callback-install grep-guard -- harness.py must not post-construction
      ASSIGN any of the nine retired scheduler callback attributes.
  (b) constructed-with-callbacks -- a freshly-built Harness's scheduler has
      all 9 hooks wired (non-None) with no half-wired window, and driving a
      representative trigger (park-stop trip) actually routes to the
      harness's own ``pause_scheduler`` method.

Step-9 (this file) is RED against the current harness.py: the nine hooks
are still installed as post-construction attribute assignments (dead
writes -- the Scheduler no longer reads them, per task 2235 steps 1-8) and
the Harness never passes ``callbacks=`` to the Scheduler constructor.
Step-10 migrates harness.py to close this gap.
"""

from __future__ import annotations

import asyncio
import dataclasses
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventStore
from orchestrator.harness import Harness
from orchestrator.run_store import RunStore

_HARNESS_SRC_PATH = Path(__file__).parent.parent / 'src' / 'orchestrator' / 'harness.py'

# Assignment-form (LHS ``self.scheduler.<attr> =``) reach-in patterns for the
# nine retired scheduler callback attributes. Deliberately scoped to
# assignment form (trailing ``\s*=``) so this guard does NOT false-positive
# on the retired-pattern docstrings (~4177-4458) that mention e.g.
# ``self.scheduler._on_external_dep_block`` in backticked prose without a
# trailing ``=`` -- those are reworded by step-10, not deleted, and a naive
# whole-file substring scan would wrongly flag them.
_CALLBACK_ASSIGNMENT_PATTERNS = (
    re.compile(r'self\.scheduler\._on_\w+\s*='),
    re.compile(r'self\.scheduler\._warm_base_health_probe\s*='),
    re.compile(r'self\.scheduler\._suppress_blocked_write\s*='),
)


def _find_callback_assignment_reachins(content: str) -> list[str]:
    """Return ``"LINENO: text"`` for each line in *content* that assigns
    directly to one of the nine retired scheduler callback attributes.

    Line-based (not a single ``re.MULTILINE`` scan) so failures report an
    actionable line number instead of just "somewhere in this 9000-line file".
    """
    hits = []
    for lineno, line in enumerate(content.splitlines(), start=1):
        if any(pattern.search(line) for pattern in _CALLBACK_ASSIGNMENT_PATTERNS):
            hits.append(f'{lineno}: {line.strip()}')
    return hits


class TestCallbackInstallGrepGuard:
    """(a) harness.py must not post-construction-assign the nine retired
    scheduler callback attributes -- they must flow through the
    SchedulerCallbacks constructor seam instead (task 2235)."""

    def test_no_assignment_form_reachins(self) -> None:
        content = _HARNESS_SRC_PATH.read_text()
        hits = _find_callback_assignment_reachins(content)
        assert not hits, (
            'harness.py still post-construction-assigns retired Scheduler '
            'callback attributes; task 2235 migrates these to '
            f'Scheduler(config, callbacks=SchedulerCallbacks(...)): {hits}'
        )

    def test_excluded_post_construction_gates_survive_untouched(self) -> None:
        """Sanity check: the two intentionally-excluded post-construction
        gates (``_landed_outbox_gate`` task 2156, ``_already_landed_gate``
        task 2313) are NOT nine-callback reach-ins, so this guard must not
        (and does not) flag them -- they legitimately stay post-construction.
        """
        content = _HARNESS_SRC_PATH.read_text()
        assert 'self.scheduler._landed_outbox_gate = ' in content, (
            'expected the excluded _landed_outbox_gate install (task 2156) '
            'to still be present in harness.py'
        )
        assert 'self.scheduler._already_landed_gate = ' in content, (
            'expected the excluded _already_landed_gate install (task 2313) '
            'to still be present in harness.py'
        )
        hits = _find_callback_assignment_reachins(content)
        assert not any('_landed_outbox_gate' in hit for hit in hits)
        assert not any('_already_landed_gate' in hit for hit in hits)


class TestConstructedWithCallbacks:
    """(b) A freshly-built Harness wires all 9 hooks into the Scheduler's
    SchedulerCallbacks bundle AT CONSTRUCTION TIME -- no half-wired window
    where the Scheduler exists but its callbacks are unset."""

    def test_all_nine_hooks_non_none_after_construction(self, tmp_path: Path) -> None:
        config = OrchestratorConfig(project_root=tmp_path)
        harness = Harness(config)

        callbacks = harness.scheduler._callbacks
        for field in dataclasses.fields(callbacks):
            assert getattr(callbacks, field.name) is not None, (
                f'SchedulerCallbacks.{field.name} is None immediately after '
                'Harness construction -- task 2235 requires the Harness to '
                'build the Scheduler with callbacks=SchedulerCallbacks(...) '
                'at construction time (no post-construction install window).'
            )

    @pytest.mark.asyncio
    async def test_park_stop_trip_routes_to_harness_pause_scheduler(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Driving the park-stop trip through the Scheduler must route to
        the Harness's OWN ``pause_scheduler`` -- observed via its RunStore
        persistence side effect, not merely ``scheduler.is_paused`` (which
        the scheduler's synchronous latch sets on its own, regardless of
        whether the trip callback is wired at all -- see
        ``_maybe_fire_park_stop_trip``'s docstring)."""
        config = OrchestratorConfig(
            project_root=tmp_path,
            max_per_module=1,
            park_stop_parked_threshold=1,
            park_stop_parked_window_hours=1.0,
        )
        harness = Harness(config)

        mock_run_store = MagicMock(spec=RunStore)
        harness._run_store = mock_run_store
        harness._run_id = 'run-test-0001'
        harness.event_store = EventStore(tmp_path / 'events.db', 'run-test-0001')

        monkeypatch.setattr(
            'orchestrator.scheduler.mcp_call', AsyncMock(return_value={}),
        )

        await harness.scheduler.set_task_status('T1', 'blocked')
        # Yield to the event loop so the ensure_future'd trip callback runs.
        await asyncio.sleep(0)

        mock_run_store.save_scheduler_pause.assert_called_once()
