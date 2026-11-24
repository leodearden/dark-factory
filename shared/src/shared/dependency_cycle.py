"""The task store's one dependency-cycle rule (task 9490).

An edge task -> depends_on ('task depends on depends_on') closes a cycle iff
depends_on already reaches task over the LITERAL dependencies table: edges of
EVERY status count, so reopening a done or cancelled task cannot revive a latent
cycle. The graph is local: qualified ``project:id`` external deps are never
walked. Callers: ``fused_memory.backends.sqlite_task_backend.SqliteTaskBackend``
(``add_dependency`` and ``update_task``) and
``shared.cancel_dispositions.check_targets``.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence


def cycle_closed_by_edge(
    task: int, depends_on: int, edges: Callable[[int], Sequence[int]]
) -> tuple[int, ...] | None:
    """The cycle (task, depends_on, ..., task) the edge would close, or None.

    Breadth-first from depends_on, visiting each node's dependencies in the order
    edges yields them, so the path is the shortest one and ties break
    deterministically. task's own out-edges are never expanded.
    """
    if depends_on == task:
        return (task, task)
    parent: dict[int, int | None] = {depends_on: None}
    frontier = deque([depends_on])
    while frontier:
        node = frontier.popleft()
        for dep in edges(node):
            if dep == task:
                return (task, *_path_to(node, parent), task)
            if dep not in parent:
                parent[dep] = node
                frontier.append(dep)
    return None


def _path_to(node: int, parent: Mapping[int, int | None]) -> tuple[int, ...]:
    path = [node]
    while (previous := parent[path[-1]]) is not None:
        path.append(previous)
    return tuple(reversed(path))
