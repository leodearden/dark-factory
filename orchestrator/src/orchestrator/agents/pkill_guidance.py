"""Role-prompt guidance for the `pkill -f` / `pgrep -f` self-match hazard.

`orchestrator/src/orchestrator/agents/roles.py::_BASH_CAPABLE_ROLE_PREAMBLE`
splices `PKILL_SELF_MATCH_GUIDANCE` into every role holding unqualified `Bash`.

Provenance. Task 5961 is a cross-repo refile of reify legibility-census
candidate #7922, routed here under `plans/confusion-reduction-prd.md` §6
decision 4. dark_factory's own codebook carries the same confusion as
cand-20260903-16, cand-20260904-5, cand-20260904-12, cand-20260906-9,
cand-20260906-12 and cand-20260909-17, whose recorded causes misattribute exit
144 to a timeout or to SIGUSR1. The codebook is census-owned and is not edited
here (§6 decision 1).

Mechanism, measured 2026-09-27 and re-measured 2026-10-02. Every `Bash` call
runs in a `/bin/bash -c ... eval '<command>'` wrapper whose argv carries the
command text. A `pkill -f` whose pattern appears in the command kills that
wrapper: the tool reports "Exit code 144" and nothing after the `pkill` runs.
The bracket form (`[z]z_marker`) survives and still kills the target.
`$(pgrep -f X)` lists the wrapper and the substitution's own subshell.

Why prose and not a guard: a PreToolUse hook would live in the user-global
`~/.claude/settings.json`, which the per-task config dir symlinks, so it is not
reachable from this repo. Follow-up task 5981 evaluates such a guard.

Machine-checked prose constraints: no literal braces
(`test_roles_pkill_self_match.py`); no fully-qualified MCP tool name
(`test_roles_ancestry_check.py`); no backticked CamelCase word that is not a
real harness tool (`test_roles_harness_tool_inventory.py`), which is why
"Exit code 144" is double-quoted and never backticked; no `Test`-prefixed
CamelCase identifier (`test_cited_test_class_drift.py`). By convention it also
carries no sighting count and no byte-size figure.
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

Make the pattern unable to match its own text by putting one character in a
bracket class: `pkill -f "[p]ytest orchestrator/tests"`. The regex still
matches `pytest orchestrator/tests` in the target's command line, while the
wrapper's command line holds the text `[p]ytest`, which it does not match.
Measured: the plain form ended its own call with exit 144; the bracketed form
killed its target and the call ran on. The bracket protects you only if the
plain target string appears NOWHERE else in the same call — launching the
target, echoing the string or grepping for it in that call puts the plain text
back into the wrapper's command line — so kill in a call of its own.

Preview before you kill. `pgrep -af "[p]attern"` prints the pid and full
command line of every process the kill would hit. Other sessions' processes
run on this host too, and a pattern loose enough to match your target can match
theirs. Once the list is exactly your target, `kill` those pids by number. A
plain, unbracketed `pgrep -af` that lists a `/bin/bash -c` line containing your
own command is the self-match showing itself.

A command you started with `Bash`'s `run_in_background` needs no pattern at
all: stop it by its task ID with the termination tool the wait section above
describes.
"""
