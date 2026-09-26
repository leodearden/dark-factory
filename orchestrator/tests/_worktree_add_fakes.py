"""One parameterised fake ``orchestrator.git_ops._run`` for the two test
modules that drive ``GitOps._worktree_add_with_retry`` (task 5140).

``test_create_merge_worktree_retry.py`` and ``test_ephemeral_worktree.py``
exercise the SAME shared retry driver, so they share one fake: a change to
the driver's argv shape lands in exactly one place instead of being chased
through a per-module copy each.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path

FakeRun = Callable[..., Awaitable[tuple[int, str, str]]]


def make_fake_run(
    add_results: Sequence[tuple[int, str, str]],
    calls: list[list[str]],
    *,
    mkdir_on_failure: bool = False,
    exists_at_entry: list[bool] | None = None,
    rev_parse_sha: str | None = None,
) -> FakeRun:
    """Fake ``orchestrator.git_ops._run`` recording every argv into *calls*.

    ``git worktree add`` results are consumed in order from *add_results* as
    ``(rc, stdout, stderr)`` triples; the last entry repeats once exhausted.
    A successful add mkdirs its target (``cmd[-2]`` in either ``add [--detach]
    <path> <ref>`` shape), mirroring real ``git worktree add``.  With *mkdir_on_failure* a FAILED add mkdirs it too —
    also what real git does, since it creates the target directory before
    the add can fail.  When *exists_at_entry* is supplied, each add records
    whether the target already existed on entry, which is how the
    between-attempts residue clearing is observed.  When *rev_parse_sha* is
    supplied, ``git rev-parse`` answers it with the trailing newline real
    git emits, so a caller's ``.strip()`` is genuinely exercised.  Every
    other command (e.g. ``git worktree remove``) succeeds silently.
    """
    state = {'add_calls': 0}

    async def _fake_run(cmd, **kwargs) -> tuple[int, str, str]:
        calls.append(list(cmd))
        if 'worktree' in cmd and 'add' in cmd:
            target = Path(cmd[-2])
            if exists_at_entry is not None:
                exists_at_entry.append(target.exists())
            rc, out, err = add_results[
                min(state['add_calls'], len(add_results) - 1)
            ]
            state['add_calls'] += 1
            if rc == 0 or mkdir_on_failure:
                target.mkdir(parents=True, exist_ok=True)
            return (rc, out, err)
        if rev_parse_sha is not None and 'rev-parse' in cmd:
            return (0, f'{rev_parse_sha}\n', '')
        return (0, '', '')

    return _fake_run
