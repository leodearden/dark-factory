"""One race shape for every plan.json writer: a writer against a held plan mutex.

The holder enters ``artifacts.plan_mutex()``, reads plan.json raw off disk, and
starts the writer on a second thread. It waits ``probe_secs`` for the writer,
then applies its own edit and writes while STILL holding the mutex. A writer
that does not wait for the mutex therefore lands inside the holder's
read-modify-write window and is clobbered by the holder's stale write; a writer
that does wait runs after the holder releases, against the holder's result, so
both writes survive.

The holder reads raw rather than through ``read_plan`` so that it never
triggers read_plan's normalization write-back itself.
"""
from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from orchestrator.artifacts import TaskArtifacts

PROBE_SECS = 0.2
HOLDER_MARKER_KEY = '_race_holder_landed'
_WRITER_FINISH_SECS = 10.0


@dataclass(frozen=True)
class RaceOutcome:
    blocked: bool
    plan: dict
    writer_error: BaseException | None


def mark_holder(plan: dict) -> None:
    plan[HOLDER_MARKER_KEY] = True


def _read_raw(artifacts: TaskArtifacts) -> dict:
    return json.loads((artifacts.root / 'plan.json').read_text())


def race_writer_against_held_mutex(
    artifacts: TaskArtifacts,
    writer: Callable[[], Any],
    *,
    holder_edit: Callable[[dict], None] = mark_holder,
    probe_secs: float = PROBE_SECS,
) -> RaceOutcome:
    errors: list[BaseException] = []

    def run_writer() -> None:
        try:
            writer()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run_writer, daemon=True)
    with artifacts.plan_mutex():
        plan = _read_raw(artifacts)
        thread.start()
        thread.join(probe_secs)
        blocked = thread.is_alive()
        holder_edit(plan)
        artifacts.write_plan(plan, mutex_held=True)
    thread.join(_WRITER_FINISH_SECS)
    assert not thread.is_alive(), (
        f'writer still running {_WRITER_FINISH_SECS}s after the holder released'
    )
    return RaceOutcome(
        blocked=blocked,
        plan=_read_raw(artifacts),
        writer_error=errors[0] if errors else None,
    )
