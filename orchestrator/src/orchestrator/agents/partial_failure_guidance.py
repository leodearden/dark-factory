"""Role-prompt guidance: a command given several paths can fail on one and print the rest.

`orchestrator/src/orchestrator/agents/roles.py::_BASH_CAPABLE_ROLE_PREAMBLE`
splices `MULTI_PATH_PARTIAL_FAILURE_GUIDANCE` into every role holding
unqualified `Bash`. It is a leaf module, imported downward by roles.py, only so
that already oversized file does not grow further.

`wc`, `cat`, `head`, `grep` and `ls` print genuine output for every readable
path beside one error line per missing one, so the failure reads as noise.

`orchestrator/tests/test_roles_multi_path_partial_failure.py` checks the
block's shape and its splice (carrier roles, count, order after
`GREP_LOOKAROUND_GUIDANCE`), never its wording.
"""

MULTI_PATH_PARTIAL_FAILURE_GUIDANCE = """
## A command given several paths can fail on one and print the rest

`wc`, `cat`, `head`, `grep` and `ls` handed several paths do not stop at a
missing one. They print correct output for every path they could read, one
"No such file or directory" line for each they could not, and exit non-zero:
1 from `wc`, `cat` and `head`, 2 from `grep` and `ls` (measured 2026-10-04).
The rows that did print are genuine, which is exactly why the one error line
is easy to read past.

Two things about that output mislead even when you look. The `total` line
from `wc` sums only the files it read: asked for three files with one
missing, it printed the total of the other two, and nothing on that line says
a file is absent. And the error line need not sit where the missing path was
in your argument list: `wc` and `grep` printed it between the rows, `ls`
printed it above all of them.

The evidence is also easy to destroy before you see it.
`wc -l a missing b 2>/dev/null | tail -1` and
`wc -l a missing b 2>&1 | tail -1` each printed only that partial total, and
the call exited 0. Do not discard stderr or trim the output while the paths
are guesses rather than known.

So, whenever one command takes several paths:

- Under "Exit code 1" or "Exit code 2" (for `grep`, only 2: its 1 means no
  line matched), output that looks complete means some argument failed: find
  the line naming it before you use any number.
- Count the per-path rows against the paths you passed. A missing row is a
  path that did not resolve, not a file with nothing in it.
- Treat the missing path as a wrong premise to re-derive, not noise to skip:
  the file you meant usually exists under another path, and any total or
  conclusion computed without it is wrong.
- When the paths are guesses, confirm them first, with `Glob` for a pattern
  or `ls -d` on the list (it names every missing one on stderr), and run the
  command over the confirmed set.
"""
