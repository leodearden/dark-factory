"""Role-prompt guidance: a pipeline's exit status is not its first command's, with `pipefail` off or on.

`orchestrator/src/orchestrator/agents/roles.py::_BASH_CAPABLE_ROLE_PREAMBLE`
splices `PIPELINE_EXIT_STATUS_GUIDANCE` into every role holding unqualified
`Bash`. It is a leaf module, imported downward by roles.py, only so that
already oversized file does not grow further.

The block is one section built from two named halves.
`PIPELINE_LAST_STAGE_STATUS` covers `pipefail` off, where `$?` after
`cmd | head` is the reader's status. `SIGPIPE_UNDER_PIPEFAIL` covers `pipefail`
on, where a reader that quits early turns the status into 141. It is guidance
rather than a guard because whether a pipeline's status matters, or a 141 is
benign, depends on intent, which command text does not show.

`orchestrator/tests/test_roles_pipeline_exit_status.py` checks the block's
shape, its composition and its splice, never its wording.
"""

PIPELINE_LAST_STAGE_STATUS = """
## A pipeline's exit status is not your command's

The `Bash` tool's own shell runs with `pipefail` off. So after
`cmd | head -N` or `cmd | tail -N`, both `$?` and the call's own "Exit code N"
are the LAST stage's status: `head`'s or `tail`'s, which is almost always 0.
They are not `cmd`'s. A command that failed outright reads as a success
(measured 2026-10-13, bash 5.2.21):

    bash -c 'echo "ERROR: unknown argument" >&2; exit 64' 2>&1 | head -4; echo "EXIT=$?"

printed the ERROR line, then `EXIT=0`, for a command that exited 64. That is
how real sessions have read a verify script's exit 64 as EXIT=0, and taken
compiler errors printed above "exit=0" for a clean build.

CAPTURE, THEN READ THE STATUS, THEN TRUNCATE. Keep the command out of the pipe:

    cmd > /tmp/<task-id>-out.log 2>&1; echo "rc=$?"

Then `Read` the log, or run `head` or `tail` on the FILE. Measured, the same
command gave `rc=64`. Make the path unique to your task: /tmp is shared by
every agent on the host.

`set -o pipefail` is NOT the fix. With it on, a reader that closes early can
kill the command with SIGPIPE before it finishes, and the status becomes 141,
which hides the command's own code. Measured: a producer that wrote 200000
lines and then exited 64, piped to `head -1` under `pipefail`, returned 141.
The same producer redirected to a file returned 64.
"""

SIGPIPE_UNDER_PIPEFAIL = """
UNDER `pipefail`, A READER THAT QUITS EARLY TURNS SUCCESS INTO EXIT 141.
`pipefail` is ON inside any script that sets it (a `set -euo pipefail` header
is common in repository scripts and test helpers) and in any command where you
turn it on yourself. With it on, a pipeline's status is the last NON-ZERO
status of any stage, not the last stage's.

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
only when the producer still has output to write after the reader has quit,
and that depends on scheduling as well as size. `grep -q` on a present line
returned 141 in 15 of 20 runs for 24 KB of input and in every run for
1.2 MB, and under host load a payload of a few hundred bytes has been
measured to miss too. Small output is not safe: output size never clears a
pipeline. Run alone, the same producer exits 0.

RECOGNISE IT. A 141, or a "BrokenPipeError" traceback, from a pipeline that
ends in `head`, `grep -q` or `grep -m` means the producer was cut off after
the reader already had what it wanted. It is neither a failure nor a "no
match", and it is not a bug in your lookup logic. To confirm it, run
`declare -p PIPESTATUS` immediately after the pipeline, before any other
command overwrites it. That prints the whole array, and a 141 in its first
element confirms the reader closed early. It is not the producer's real
status and does not tell you what that would have been: to learn it, run the
producer alone, captured to a file, and read its own status.

GUARD IT whenever you write or edit a script that sets `pipefail`:

- For a yes/no membership test, capture first, then test the captured text
  as a condition: `keys=$(producer)`, then
  `if grep -qxF -- "$key" <<<"$keys"; then ...; fi`. The capture fails only
  if the producer really failed, and the test's 0 or 1 is the answer. Keep
  the test in an `if`, `&&` or `||` condition: run bare under `set -e`, its
  1 for an absent key ends the script.
- Otherwise, use a reader that consumes all of its input: `sed -n 1p` in
  place of `head -n1`, and `grep -c PATTERN >/dev/null` in place of
  `grep -q PATTERN`, which still exits 1 when nothing matches. Both
  returned status 0 on the 1.2 MB input.
- `|| true` after the pipeline silences the 141, but it also silences a real
  failure: `false | head -n1 || true` returned 0 with empty output. Use it
  only where the producer's own failure does not matter.
"""

PIPELINE_EXIT_STATUS_GUIDANCE = PIPELINE_LAST_STAGE_STATUS + SIGPIPE_UNDER_PIPEFAIL
