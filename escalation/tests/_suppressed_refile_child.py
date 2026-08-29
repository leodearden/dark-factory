"""Subprocess child-runner for the note_suppressed_refile concurrency test.

Called by ``TestNoteSuppressedRefile.test_concurrent_bumps_lose_no_count`` in
test_queue.py.  Self-contained in the same way ``_concurrent_queue_child.py``
is: it injects the worktree ``src`` onto sys.path exactly as conftest.py does,
so it imports the in-tree escalation package rather than any
editable/installed copy.

WHY A SECOND CHILD SCRIPT (task 4499, amendment pass).  The natural home for
this is a new ``note_suppressed_refile`` op on ``_concurrent_queue_child.py``,
which already serves four such ops.  That file was outside the editable scope
of the task that needed this test, and a NEW file cannot collide with another
task's concurrent edits, so the op lives here instead.  Folding it back into
``_concurrent_queue_child.py`` as a fifth op — and deleting this file — is a
clean, behaviour-preserving follow-up for anyone already editing that file.

Usage::

    python _suppressed_refile_child.py \
        <queue_dir> <escalation_id> <count> <ready_file> <go_file>

Arguments
---------
queue_dir
    Path to the EscalationQueue directory (same tmp_path as the parent test).
escalation_id
    Id of a pre-seeded RESOLVED record (submitted and resolved by the parent
    process).  ``note_suppressed_refile`` is terminal-only, so a pending
    record would make every call a silent no-op and the test vacuous.
count
    Number of bumps to run.  Each iteration performs exactly ONE increment,
    maximising read-modify-write interleaving between the two concurrent child
    processes — which is the whole point: the counter is an INCREMENT, not a
    field SET, so a lost update is invisible in the final value unless the
    parent knows the exact total to expect.
ready_file / go_file
    A RENDEZVOUS BARRIER, and it is load-bearing rather than tidiness.  Without
    it the two children race their own interpreter startup, not the counter:
    process A can finish every bump before B has imported, so the runs never
    overlap and the test passes even with the lock deleted (measured — that is
    exactly how the first draft of this harness went vacuous).  The child
    therefore does ALL its setup, touches *ready_file*, and spins until
    *go_file* appears; the parent releases both children only once both are
    parked at the barrier.

Exit codes: 0 on success, 3 if the go-file never appears (a barrier timeout, so
a wedged rendezvous fails the parent test loudly instead of silently degrading
to the sequential case it exists to avoid), non-zero on any exception.  No
assertions; this is a harness only.  The parent test re-reads the record from
disk and asserts the total.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

#: Generous — this only has to outlast the sibling process's interpreter
#: startup, and is a wedge-detector rather than a tuning knob.
_BARRIER_TIMEOUT_SECS = 60.0

_SRC = Path(__file__).parent.parent / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from escalation.queue import EscalationQueue


def main():
    if len(sys.argv) != 6:
        print(
            f'Usage: {sys.argv[0]} <queue_dir> <escalation_id> <count> '
            '<ready_file> <go_file>',
            file=sys.stderr,
        )
        sys.exit(1)

    queue_dir_str, escalation_id, count_str, ready_str, go_str = sys.argv[1:]
    count = int(count_str)
    queue = EscalationQueue(Path(queue_dir_str))

    # Park at the barrier with every cost already paid — interpreter startup,
    # imports, queue construction — so what overlaps is the bumping and nothing
    # else.
    go = Path(go_str)
    Path(ready_str).write_text('ready')
    deadline = time.monotonic() + _BARRIER_TIMEOUT_SECS
    while not go.exists():
        if time.monotonic() > deadline:
            print('barrier timeout: go-file never appeared', file=sys.stderr)
            sys.exit(3)
        time.sleep(0.002)

    for _ in range(count):
        queue.note_suppressed_refile(escalation_id)


if __name__ == '__main__':
    main()
