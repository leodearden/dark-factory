"""Fake ``session_ownership_probes.ProcessTree`` values describing a process tree for a hook event."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from orchestrator.session_ownership_probes import ProcessTree

#: The hook's own parent in a ``claude_process_tree``; no test pid uses it.
_HOOK_SHELL_PID = 7


def fake_process_tree(chain: Sequence[tuple[int, str]]) -> ProcessTree:
    """A tree whose ancestry is *chain*: child-first ``(pid, comm)`` pairs.

    The chain starts at THIS hook's parent, so the ancestor walk starts
    inside it rather than at the real parent.
    """
    if not chain:
        raise ValueError('fake_process_tree needs at least the hook parent')
    parents = {pid: chain[i + 1][0] for i, (pid, _) in enumerate(chain[:-1])}
    comms = dict(chain)
    return ProcessTree(
        hook_parent_pid=lambda: chain[0][0],
        parent_pid_of=lambda pid: parents.get(pid),
        comm_of=lambda pid: comms.get(pid),
    )


def claude_process_tree(pid: int | None, parent_pid: int | None = None) -> ProcessTree:
    """A tree whose owning claude is *pid* and that claude's direct parent *parent_pid*.

    ``pid=None`` means no claude ancestor at all; ``parent_pid=None`` means the
    claude is found but its parent cannot be read.
    """
    chain = [(_HOOK_SHELL_PID, 'sh')]
    if pid is not None:
        chain.append((pid, 'claude'))
        if parent_pid is not None:
            chain.append((parent_pid, 'bash'))
    return fake_process_tree(chain)


def counting_walks(tree: ProcessTree) -> tuple[ProcessTree, list[int]]:
    """*tree*, plus a list that gains one entry per ancestor walk started on it."""
    walks: list[int] = []

    def _hook_parent_pid() -> int:
        walks.append(1)
        return tree.hook_parent_pid()

    return dataclasses.replace(tree, hook_parent_pid=_hook_parent_pid), walks
