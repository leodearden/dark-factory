"""Tests for orchestrator.session_ownership_probes, through its public seam.

Covers the default ``ProcessTree`` readers, the owning-claude identity walk
and its bound, the ``CLAUDE_SPAWN_OWNER_PPID`` verdict, the liveness-pid
choices, and the ``EventProbes`` per-event memo.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable

import pytest
from _process_tree_fakes import claude_process_tree, counting_walks, fake_process_tree

from orchestrator.session_ownership_probes import (
    MAX_CLAUDE_ANCESTOR_HOPS,
    EventProbes,
    ProcessTree,
)

# ---------------------------------------------------------------------------
# The default (live) ProcessTree readers
# ---------------------------------------------------------------------------


def test_default_process_tree_reads_this_process_parent() -> None:
    tree = ProcessTree()
    assert tree.hook_parent_pid() == os.getppid()
    assert tree.parent_pid_of(os.getpid()) == os.getppid()


def test_default_process_tree_answers_none_for_an_unresolvable_parent() -> None:
    # Never raises: an absent /proc entry (or no /proc at all) is a plain
    # "cannot tell", which every caller treats as a best-effort miss.
    assert ProcessTree().parent_pid_of(-1) is None


def test_default_process_tree_answers_none_for_an_unresolvable_name() -> None:
    # Never raises -- the same best-effort contract as its parent read.
    assert ProcessTree().comm_of(-1) is None


def test_default_process_tree_reads_this_process_name() -> None:
    assert ProcessTree().comm_of(os.getpid()) is not None


def test_default_process_tree_session_leader_is_this_process_session() -> None:
    assert ProcessTree().session_leader_pid() == os.getsid(0)


# ---------------------------------------------------------------------------
# The owning-claude identity walk
# ---------------------------------------------------------------------------

# The tree measured end-to-end with a probe hook on claude 2.1.241:
# python -> hook.sh -> sh -c -> claude -> the payload bash. The `sh -c`
# wrapper Claude Code interposes is the level a fixed-depth grandparent
# lookup mistook for the firing claude.
_REAL_HOOK_CHAIN = [
    (2028035, 'hook.sh'),
    (2028032, 'sh'),
    (2025347, 'claude'),
    (2025344, 'bash'),
]


def test_owning_claude_pid_skips_the_shell_wrapper_claude_code_interposes() -> None:
    # The GRANDPARENT (2028032) is the ephemeral `sh`, not the claude.
    # Resolving by identity must reach past it to 2025347 -- otherwise the
    # OWNER_PPID verdict compares one level short and judges every genuine
    # owner an inheritor.
    probes = EventProbes({}, fake_process_tree(_REAL_HOOK_CHAIN))
    assert probes.owning_claude_pid() == 2025347


def test_owning_claude_pid_still_resolves_without_a_shell_wrapper() -> None:
    # Identity resolution is indifferent to whether the wrapper is present,
    # so a harness that execs the hook directly still resolves.
    tree = fake_process_tree([(500, 'hook.sh'), (400, 'claude')])
    assert EventProbes({}, tree).owning_claude_pid() == 400


def test_owning_claude_pid_returns_none_when_no_claude_ancestor_is_found() -> None:
    # No /proc (macOS), a mid-read race, or a hook reparented away: the
    # answer is "cannot prove ownership", never a guessed pid.
    tree = fake_process_tree([(500, 'hook.sh'), (400, 'bash'), (300, 'init')])
    assert EventProbes({}, tree).owning_claude_pid() is None


def test_owning_claude_pid_walk_is_bounded() -> None:
    # A claude sitting deeper than the bound is not reached, so a runaway
    # walk can never reach init and mint a bogus owner.
    chain = [(1000 - i, 'bash') for i in range(MAX_CLAUDE_ANCESTOR_HOPS)]
    chain.append((1000 - MAX_CLAUDE_ANCESTOR_HOPS, 'claude'))
    assert EventProbes({}, fake_process_tree(chain)).owning_claude_pid() is None


# ---------------------------------------------------------------------------
# The CLAUDE_SPAWN_OWNER_PPID verdict
# ---------------------------------------------------------------------------


def test_owner_ppid_verdict_accepts_the_genuine_owner_through_the_wrapper() -> None:
    # End-to-end over the measured tree: CLAUDE_SPAWN_OWNER_PPID carries the
    # payload bash (2025344), i.e. the firing claude's DIRECT parent. The
    # verdict must be True -- a grandparent lookup would return False.
    env = {'CLAUDE_SPAWN_OWNER_PPID': '2025344'}
    probes = EventProbes(env, fake_process_tree(_REAL_HOOK_CHAIN))
    assert probes.owner_ppid_verdict() is True


def test_owner_ppid_verdict_still_rejects_a_nested_claude() -> None:
    # A nested `claude -p` from an agent's Bash tool: its own first claude
    # ancestor is ITSELF, whose parent is the tool's shell -- not the
    # inherited payload-bash pid, so it mismatches and is judged an
    # inheritor even though it carries the env var verbatim.
    tree = fake_process_tree([
        (900035, 'hook.sh'),
        (900032, 'sh'),
        (900010, 'claude'),  # the NESTED claude
        (900005, 'bash'),  # the Bash tool's shell
        (2025347, 'claude'),  # the owner, further up
        (2025344, 'bash'),
    ])
    env = {'CLAUDE_SPAWN_OWNER_PPID': '2025344'}
    assert EventProbes(env, tree).owner_ppid_verdict() is False


def test_owner_ppid_verdict_is_none_for_every_unresolvable_input() -> None:
    # Fail-soft: never False (fork) on an input the probe simply cannot read.
    def verdict(env: dict[str, str], tree: ProcessTree) -> bool | None:
        return EventProbes(env, tree).owner_ppid_verdict()

    owner = claude_process_tree(4193501, 4193500)
    assert verdict({}, owner) is None
    assert verdict({'CLAUDE_SPAWN_OWNER_PPID': '   '}, owner) is None
    assert verdict({'CLAUDE_SPAWN_OWNER_PPID': 'not-a-pid'}, owner) is None
    assert verdict({'CLAUDE_SPAWN_OWNER_PPID': '1'}, owner) is None
    assert verdict({'CLAUDE_SPAWN_OWNER_PPID': '4193500'}, owner) is True
    assert verdict({'CLAUDE_SPAWN_OWNER_PPID': '4193999'}, owner) is False

    no_claude_ancestor = claude_process_tree(None)
    assert verdict({'CLAUDE_SPAWN_OWNER_PPID': '4193500'}, no_claude_ancestor) is None
    unreadable_parent = claude_process_tree(4193501)
    assert verdict({'CLAUDE_SPAWN_OWNER_PPID': '4193500'}, unreadable_parent) is None


# ---------------------------------------------------------------------------
# Liveness pids for a new record
# ---------------------------------------------------------------------------


def test_hand_launched_liveness_pid_is_the_session_leader_not_the_claude() -> None:
    tree = dataclasses.replace(claude_process_tree(424242), session_leader_pid=lambda: 515151)
    assert EventProbes({}, tree).hand_launched_liveness_pid() == 515151


def test_nested_claude_liveness_pid_is_the_nested_claude_not_the_wrapper() -> None:
    # Stamping the `sh` wrapper's pid (2028032) would give a forked-inheritor
    # record an ALREADY-DEAD launcher_pid, so `stale_pid` could reap the
    # record of a still-running nested session. The stamped pid must be the
    # claude itself.
    probes = EventProbes({}, fake_process_tree(_REAL_HOOK_CHAIN))
    assert probes.nested_claude_liveness_pid() == 2025347


def test_nested_claude_liveness_pid_falls_back_to_the_durable_pid() -> None:
    # No /proc, a mid-read race, or a hook reparented to init: degrade to
    # the coarse-but-durable session leader rather than guessing.
    tree = dataclasses.replace(claude_process_tree(None), session_leader_pid=lambda: 515151)
    assert EventProbes({}, tree).nested_claude_liveness_pid() == 515151


# ---------------------------------------------------------------------------
# EventProbes -- one process-tree observation per hook event
# ---------------------------------------------------------------------------


def test_event_probes_walks_proc_once_for_repeated_pid_reads() -> None:
    tree, walks = counting_walks(claude_process_tree(4193501))
    probes = EventProbes({}, tree)

    assert probes.owning_claude_pid() == 4193501
    assert probes.owning_claude_pid() == 4193501
    assert probes.owning_claude_pid() == 4193501
    assert len(walks) == 1


def test_event_probes_walks_proc_once_for_repeated_verdicts() -> None:
    tree, walks = counting_walks(claude_process_tree(4193501, 4193500))
    probes = EventProbes({'CLAUDE_SPAWN_OWNER_PPID': '4193500'}, tree)

    assert probes.owner_ppid_verdict() is True
    assert probes.owner_ppid_verdict() is True
    assert len(walks) == 1


_ProbeRead = tuple[Callable[[EventProbes], object], object]
_VERDICT: _ProbeRead = (EventProbes.owner_ppid_verdict, True)
_OWNER_PID: _ProbeRead = (EventProbes.owning_claude_pid, 4193501)
_NESTED_LIVENESS_PID: _ProbeRead = (EventProbes.nested_claude_liveness_pid, 4193501)


@pytest.mark.parametrize(
    'reads',
    [(_VERDICT, _OWNER_PID), (_OWNER_PID, _VERDICT), (_VERDICT, _NESTED_LIVENESS_PID)],
    ids=['verdict-then-pid', 'pid-then-verdict', 'verdict-then-nested-liveness-pid'],
)
def test_event_probes_share_one_walk_across_the_verdict_and_pid_reads(reads: tuple[_ProbeRead, ...]) -> None:
    """The verdict and every pid read consume ONE walk, in any order."""
    tree, walks = counting_walks(claude_process_tree(4193501, 4193500))
    probes = EventProbes({'CLAUDE_SPAWN_OWNER_PPID': '4193500'}, tree)

    for read, expected in reads:
        assert read(probes) == expected
    assert len(walks) == 1


def test_event_probes_caches_none_as_a_resolved_answer() -> None:
    """None is a legitimate resolved answer, not "not yet computed"."""
    tree, walks = counting_walks(claude_process_tree(None))
    probes = EventProbes({'CLAUDE_SPAWN_OWNER_PPID': '4193500'}, tree)

    assert probes.owning_claude_pid() is None
    assert probes.owning_claude_pid() is None
    assert len(walks) == 1

    tree_v, walks_v = counting_walks(claude_process_tree(None))
    probes_v = EventProbes({'CLAUDE_SPAWN_OWNER_PPID': '4193500'}, tree_v)
    assert probes_v.owner_ppid_verdict() is None
    assert probes_v.owner_ppid_verdict() is None
    assert len(walks_v) == 1
