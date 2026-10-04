"""Role-prompt guidance: reading the one exit status of a chained `Bash` call.

`orchestrator/src/orchestrator/agents/roles.py::_BASH_CAPABLE_ROLE_PREAMBLE`
splices `CHAINED_COMMAND_STATUS_GUIDANCE` into every role holding unqualified
`Bash`. It is a leaf module, imported downward by roles.py, only so that
already oversized file does not grow further.

A call whose steps are joined with `&&` or `;` reports one status, the last-run
command's, and `&&` stops at the first step that exits non-zero.

The prose is machine-checked by the `test_roles_*` modules, which scan every
role prompt.
"""

CHAINED_COMMAND_STATUS_GUIDANCE = """
## A chained `Bash` call reports one exit status: its last command's

A `Bash` call returns ONE exit status. It belongs to the LAST command that
actually ran, not to the call as a whole and not to any earlier step. When it
is non-zero, the result opens with "Exit code N". Read it with the operator
that joined your steps in mind (all measured 2026-10-02):

- `a && b && c` stops at the first step that exits non-zero, and nothing after
  it runs. The status is the stopping step's. A later step's missing output
  means NEVER RAN, not "found nothing": a `grep` with no match stopped a chain
  right after its first header line, and nothing behind it ran.
- `a; b; c` runs every step but reports only `c`'s status, so an earlier
  step's failure leaves no trace in it. A failing last step, such as an
  `ls -d` of an optional path, puts "Exit code 2" over output in which an
  earlier step already printed the answer you wanted.
- Some commands exit non-zero to ANSWER, not to fail: `grep` exits 1 for no
  match, `test` exits 1 for false, `git merge-base --is-ancestor` exits 1 for
  "not an ancestor", and `git diff --quiet` exits 1 for "differs". As the last
  step, that answer becomes the call's "Exit code 1". In front of `&&`, it
  silently cancels every step after it.
- A "syntax error" or "unexpected end of file" from "/bin/bash: eval" means the
  line it broke on never ran: neither the broken part nor the steps in front
  of it on that line. Lines ABOVE it in a multi-line call did run, and the line
  number it reports need not match your own. Do not assume an earlier
  `git add` or `cd` on the broken line took effect: check the state, fix the
  syntax, and re-run.

So:

- Put independent lookups in SEPARATE `Bash` calls, issued together in one
  turn, so each brings back its own output and its own status.
- Use `&&` only where a later step must not run unless the earlier one
  succeeded, as in `cd dir && make`. Never put a predicate or an optional
  existence check in front of a step you still need.
- To keep several independent steps in one call, separate them with `;` or a
  line break, and label each step's status in the output:

      grep -rn 'needle' src; echo "grep rc=$?"
      git merge-base --is-ancestor main HEAD; echo "ancestor rc=$?"

  Each label prints its step's code while the call itself reports success, so
  a predicate's answer reads as an answer rather than as a failed call.
- `rc=$?` labels ONE command's status. After a pipeline it holds the last
  stage's status, so keep the command whose status you are labelling out of a
  pipe.
"""
