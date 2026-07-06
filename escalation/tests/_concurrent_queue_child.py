"""Subprocess child-runner for two-OS-process concurrency tests.

Called by TestAddMembersToL2Concurrency and TestAttachDedupeChildConcurrency
in test_queue.py.  This script is intentionally self-contained: it injects
the worktree ``src`` onto sys.path exactly as conftest.py does, so it
imports the in-tree escalation package rather than any editable/installed copy.

Usage::

    python _concurrent_queue_child.py <queue_dir> <op> <parent_id> <prefix> <count>

Arguments
---------
queue_dir
    Path to the EscalationQueue directory (same tmp_path as the parent test).
op
    Operation to run.  One of:
    - ``add_members`` — calls ``add_members_to_l2(parent_id, [f'{prefix}-{i}'])``
      on each iteration.
    - ``attach`` — calls ``attach_dedupe_child(parent_id, f'{prefix}-{i}')`` on
      each iteration.
    - ``make_id_submit`` — calls ``eid = queue.make_id(parent_id)`` (here
      *parent_id* is reused as the ``task_id`` — no pre-seeded parent
      escalation is required for this op) then
      ``queue.submit(Escalation(id=eid, task_id=parent_id, ...))`` on each
      iteration, minting and submitting one fresh escalation per iteration.
parent_id
    The escalation id of the pre-seeded parent (submitted by the parent
    process).  For ``make_id_submit`` this slot instead carries the shared
    ``task_id`` that both child processes mint ids against.
prefix
    String prefix for the generated child/member ids (e.g. ``'a'`` or ``'b'``).
    Unused by ``make_id_submit`` (accepted for CLI-argument-count
    uniformity — pass any placeholder).
count
    Number of iterations to run.  Each iteration appends exactly ONE element,
    maximising RMW interleaving between the two concurrent child processes.

Exit codes: 0 on success, non-zero on any exception.  No assertions; this is
a harness only.  The parent test reads the final state and asserts the union.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Inject worktree src — same idiom as escalation/tests/conftest.py so this
# child imports the in-tree escalation package, not any installed copy.
_SRC = Path(__file__).parent.parent / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from escalation.models import Escalation  # noqa: E402  (import after path fix)
from escalation.queue import EscalationQueue  # noqa: E402  (import after path fix)


def main() -> None:
    if len(sys.argv) != 6:
        print(
            f'Usage: {sys.argv[0]} <queue_dir> <op> <parent_id> <prefix> <count>',
            file=sys.stderr,
        )
        sys.exit(1)

    queue_dir_str, op, parent_id, prefix, count_str = sys.argv[1:]
    count = int(count_str)
    queue = EscalationQueue(Path(queue_dir_str))

    if op == 'add_members':
        for i in range(count):
            queue.add_members_to_l2(parent_id, [f'{prefix}-{i}'])
    elif op == 'attach':
        for i in range(count):
            queue.attach_dedupe_child(parent_id, f'{prefix}-{i}')
    elif op == 'resolve':
        # count is ignored; call resolve once per process (idempotent after first call).
        queue.resolve(parent_id, f'resolved-by-{prefix}')
    elif op == 'make_id_submit':
        # parent_id slot is reused as task_id — no pre-seeded parent required.
        task_id = parent_id
        for _ in range(count):
            eid = queue.make_id(task_id)
            queue.submit(Escalation(
                id=eid,
                task_id=task_id,
                agent_role='orchestrator',
                severity='blocking',
                category='task_failure',
                summary='concurrent make_id test',
                level=0,
            ))
    else:
        print(f'Unknown op: {op!r}', file=sys.stderr)
        sys.exit(2)


if __name__ == '__main__':
    main()
