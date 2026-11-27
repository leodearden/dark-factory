"""Waits that let a DecisionQueue finish reflowing before a pilot test asserts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.geometry import Size

if TYPE_CHECKING:
    from textual.pilot import Pilot

    from cockpit.panes.decision_queue import DecisionQueue


def _queue_layout(queue: DecisionQueue) -> tuple:
    return (
        queue.size,
        queue.virtual_size,
        queue.scrollable_content_region,
        queue.scroll_offset,
        queue.row_count,
        tuple(column.width for column in queue.columns.values()),
    )


async def await_queue_settled(queue: DecisionQueue) -> None:
    """Return once a screen refresh leaves *queue*'s layout unchanged.

    Pilot.pause() is not enough: DecisionQueue finishes a reflow in
    after-refresh callbacks, its own re-measure
    (cockpit/src/cockpit/panes/decision_queue.py::DecisionQueue._reflow) and
    DataTable.move_cursor's deferred scroll to the cursor, and Pilot.pause()'s
    CPU-idle heuristic can return before those run on a contended host
    (task 11997). wait_for_refresh() resolves only after every callback the
    queue scheduled before it, so an unchanged pass means the layout reached
    a fixpoint. That fixpoint means nothing is left pending only because
    every reflow that matters changes a column width or the scroll, which the
    equality guard in DecisionQueue._reflow_if_width_changed ensures; a
    reflow rescheduled with an unchanged layout is not waited for.
    """
    while True:
        before = _queue_layout(queue)
        await queue.wait_for_refresh()
        if _queue_layout(queue) == before:
            return


async def resize_terminal_and_settle(
    pilot: Pilot, queue: DecisionQueue, width: int, height: int
) -> None:
    """Resize the terminal to a new size, then wait for *queue* to settle.

    A refresh barrier can pass before the app-level Resize reaches the
    screen, so this first waits for the screen's layout to take the new
    size, which Textual alone drives. The closing pause drains what the
    queue bubbled to the app, such as a RowHighlighted. Textual ignores a
    resize to the current size, and the resize must change *queue*'s size;
    both are asserted with values. There is no time bound; the calling
    test's pytest timeout guards liveness.
    """
    target = Size(width, height)
    assert pilot.app.size != target, f'the terminal is already {target}'
    before = queue.size
    await pilot.resize_terminal(width, height)
    while pilot.app.screen.outer_size != target:
        await pilot.pause()
    await await_queue_settled(queue)
    await pilot.pause()
    assert queue.size != before, f'resizing to {target} left the queue at {before}'
