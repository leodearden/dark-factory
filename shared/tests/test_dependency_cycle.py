"""shared.dependency_cycle: the task store's one dependency-cycle rule (task 9490).

An edge task -> depends_on means 'task depends on depends_on'; it closes a cycle
iff depends_on already reaches task. Each graph is a plain dict behind an edges
callable, as the backend and the cancel-repoint check both supply one.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from shared.dependency_cycle import cycle_closed_by_edge


def _edges(graph: Mapping[int, Sequence[int]]) -> Callable[[int], Sequence[int]]:
    return lambda task: graph.get(task, ())


def test_a_three_cycle_names_the_whole_path() -> None:
    assert cycle_closed_by_edge(1, 2, _edges({2: [3], 3: [1]})) == (1, 2, 3, 1)


def test_the_shortest_path_wins_in_the_order_edges_are_yielded() -> None:
    graph = {2: [3, 4], 3: [5], 4: [1], 5: [1]}

    assert cycle_closed_by_edge(1, 2, _edges(graph)) == (1, 2, 4, 1)


def test_a_diamond_built_one_edge_at_a_time_closes_no_cycle() -> None:
    graph: dict[int, list[int]] = {}
    for task, depends_on in ((1, 2), (1, 3), (2, 4), (3, 4)):
        assert cycle_closed_by_edge(task, depends_on, _edges(graph)) is None
        graph.setdefault(task, []).append(depends_on)


def test_a_self_edge_is_a_cycle_of_one_edge() -> None:
    assert cycle_closed_by_edge(5, 5, lambda task: ()) == (5, 5)


def test_a_cycle_that_never_reaches_the_task_terminates_unclosed() -> None:
    assert cycle_closed_by_edge(1, 2, _edges({2: [2, 3], 3: [2]})) is None


def test_a_direct_back_edge_is_a_two_edge_cycle() -> None:
    assert cycle_closed_by_edge(1, 2, _edges({2: [1]})) == (1, 2, 1)


def test_the_tasks_own_out_edges_are_never_expanded() -> None:
    assert cycle_closed_by_edge(1, 3, _edges({1: [2], 2: [3]})) is None
    assert cycle_closed_by_edge(1, 3, _edges({1: [9], 3: [1]})) == (1, 3, 1)
