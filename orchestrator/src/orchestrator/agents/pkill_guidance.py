"""Role-prompt guidance: `pkill -f` and `pgrep -f` match the `Bash` wrapper shell.

`orchestrator/src/orchestrator/agents/roles.py::_BASH_CAPABLE_ROLE_PREAMBLE`
splices `PKILL_SELF_MATCH_GUIDANCE` into every role holding unqualified `Bash`.
It is a leaf module, imported downward by roles.py, only so that already
oversized file does not grow further.

Every `Bash` call runs in a `/bin/bash -c` wrapper whose argv carries the
command text, so a `-f` pattern typed into the command matches that wrapper.

The prose is machine-checked by the `test_roles_*` modules, which scan every
role prompt.
"""

PKILL_SELF_MATCH_GUIDANCE = """
## `pkill -f` and `pgrep -f` match the shell that runs them

Every `Bash` call runs inside a `/bin/bash -c` wrapper whose own command line
carries your whole command text verbatim. `pkill -f PATTERN` matches PATTERN
against the FULL command line of every process on the host and spares only
itself — not that wrapper. Your pattern is always in the wrapper's command
line, because you typed it there, so the wrapper matches and is killed: nothing
after the `pkill` runs, and the call reports "Exit code 144" (measured
2026-10-02). That exit is the self-kill — none of the codes the exit-code
section above reads against the clock, not a sandbox denial, not a flaky
harness — and retrying the identical command reproduces it.
`kill $(pgrep -f PATTERN)` is the same trap: the list it returns includes the
wrapper and the substitution's own subshell.

A safe kill pattern needs two things. First, anchor it on something only your
own target carries — your worktree's path, never a bare program name. Every
worktree on this host runs the same programs, and a `pkill -f` naming only
`pytest` has killed another worktree's merge-lane verify run here. Second, put
one character of it in a bracket class so it cannot match its own text:
`pkill -f "[.]worktrees/<your-id>/.venv/bin/pytest"`. The regex still matches
`.worktrees/<your-id>/.venv/bin/pytest` in the target's command line, while the
wrapper's command line holds the text `[.]worktrees`, which it does not match.
Measured: the plain form ended its own call with exit 144; the bracketed form
killed its target and the call ran on. The bracket protects you only if the
plain target string appears NOWHERE else in the same call — launching the
target, echoing the string or grepping for it in that call puts the plain text
back into the wrapper's command line — so kill in a call of its own.

Preview before you kill. `pgrep -af` with the same bracketed pattern prints the
pid and full command line of every process the kill would hit. A `.venv` or log
path under another worktree is the tell that a process is not yours. Once the
list is exactly your target, `kill` those pids by number. A plain, unbracketed
`pgrep -af` that lists a `/bin/bash -c` line containing your own command is the
self-match showing itself.

A command you started with `Bash`'s `run_in_background` needs no pattern at
all: stop it by its task ID with the termination tool the wait section above
describes.
"""
