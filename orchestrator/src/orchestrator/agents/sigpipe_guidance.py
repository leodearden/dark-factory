"""Role-prompt guidance: under `pipefail`, a pipe reader that quits early turns success into exit 141.

`orchestrator/src/orchestrator/agents/roles.py::_BASH_CAPABLE_ROLE_PREAMBLE`
splices `SIGPIPE_UNDER_PIPEFAIL_GUIDANCE` into every role holding unqualified
`Bash`. It is a leaf module, imported downward by roles.py, only so that
already oversized file does not grow further.

With `pipefail` on, `head`, `grep -q` or `grep -m` closing the pipe SIGPIPEs a
producer that is still writing, and the pipeline's 141 then reads like a
failure or a "no match". It is guidance rather than a guard because whether a
141 is benign depends on intent, which command text does not show.

`orchestrator/tests/test_roles_sigpipe_under_pipefail.py` checks the block's
shape and its splice (carrier roles, count, order after
`GREP_LOOKAROUND_GUIDANCE`), never its wording.
"""

SIGPIPE_UNDER_PIPEFAIL_GUIDANCE = """
## Under `pipefail`, a pipe reader that quits early turns success into exit 141

The `Bash` tool's own shell runs with `pipefail` off. It is ON inside any
script that sets it (a `set -euo pipefail` header is common in repository
scripts and test helpers) and in any command where you turn it on yourself.
With it on, a pipeline's status is the last NON-ZERO status of any stage, not
the last stage's.

That makes a reader that stops early dangerous. `head -n1`, `grep -q` and
`grep -m1` exit as soon as they have their answer and close the pipe. If the
producer upstream is still writing, SIGPIPE kills it with status 141
(128 + 13), and `pipefail` reports that 141 as the whole pipeline's status.
Measured with `pipefail` on:

- `seq 1 200000 | head -n1` printed `1`, the right answer, and exited 141.
  The `Bash` result opened with "Exit code 141" above that correct output.
- `if seq 1 200000 | grep -q '^1$'; then ...; else ...; fi` took the ELSE
  branch, status 141, although the line is present. A genuinely absent line
  takes the same branch with status 1. Only the status tells them apart.
- In a `set -euo pipefail` script, `first=$(seq 1 200000 | head -n1)` ended
  the script with status 141 and printed nothing at all.
- `git log | head -n1` exited 141 the same way. A Python producer does not
  die of SIGPIPE: it printed a "BrokenPipeError" traceback and exited 1, or
  120 when the broken write was its final flush at exit.

It is intermittent, so "it worked when I ran it" proves nothing. It fires
only when the producer still has output to write after the reader has quit.
`grep -q` on a present line never returned 141 for 4 KB of input, returned
it in 15 of 20 runs for 24 KB, and in every run for 1.2 MB. Run alone, the
same producer exits 0.

RECOGNISE IT. A 141, or a "BrokenPipeError" traceback, from a pipeline that
ends in `head`, `grep -q` or `grep -m` means the producer was cut off after
the reader already had what it wanted. It is neither a failure nor a "no
match", and it is not a bug in your lookup logic. Confirm it in one step:
run the producer alone and read its own status, or run
`declare -p PIPESTATUS` immediately after the pipeline, before any other
command overwrites it. That prints the whole array, and its first element is
the producer's status.

GUARD IT whenever you write or edit a script that sets `pipefail`:

- For a yes/no membership test, capture first, then test the captured text:
  `keys=$(producer)`, then `grep -qxF -- "$key" <<<"$keys"`. The capture
  fails only if the producer really failed, and the test's 0 or 1 is the
  answer.
- Otherwise, use a reader that consumes all of its input: `sed -n 1p` in
  place of `head -n1`, and `grep -c PATTERN >/dev/null` in place of
  `grep -q PATTERN`, which still exits 1 when nothing matches. Both
  returned status 0 on the 1.2 MB input.
- `|| true` after the pipeline silences the 141, but it also silences a real
  failure: `false | head -n1 || true` returned 0 with empty output. Use it
  only where the producer's own failure does not matter.
"""
