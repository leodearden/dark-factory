"""Role-prompt guidance about the agent harness's tools, and the preamble that composes it.

Every role with a literal system prompt and unqualified `Bash` splices
`BASH_CAPABLE_ROLE_PREAMBLE` straight after its role statement, in
`orchestrator/src/orchestrator/agents/roles.py`. JUDGE builds its own chain
there from `TOOL_CALL_REJECTION_GUIDANCE`, `ERROR_REMEDY_HINT_GUIDANCE` and
`GREP_LOOKAROUND_GUIDANCE_READ_ONLY`, and
`orchestrator/src/orchestrator/agents/briefing.py` interpolates
`WAIT_PATTERN_REMINDER` into the amender turn prompt. Those two modules import
this one; it imports only the `*_guidance.py` leaves appended at the preamble
tail.

Every constant here holds these rules, each with its enforcer:
- no literal brace: checked per block in its `test_roles_*` module, except
  `BACKGROUND_TASK_WARNING`, which holds it unchecked;
- no `mcp__<family>__<name>` tool name, because a block reaches roles with
  differing grants:
  `orchestrator/tests/test_roles_ancestry_check.py::test_role_holds_every_mcp_tool_its_prompt_names`;
- every backticked CamelCase word is a real harness tool or a listed non-tool:
  `orchestrator/tests/test_roles_harness_tool_inventory.py`;
- a test cited here is named `path::lowercase_function`, so
  `orchestrator/tests/test_cited_test_class_drift.py` has no class to resolve.
One rule no test enforces: no sighting count and no byte-size figure, because
a prose copy does not move when the thing it counts does.

Each block's finding and rationale live in its `test_roles_*` module; the
measurements live in the commits that introduced the block.
"""

from orchestrator.agents.bash_cwd_guidance import BASH_CWD_ANCHOR_GUIDANCE
from orchestrator.agents.chained_command_guidance import CHAINED_COMMAND_STATUS_GUIDANCE
from orchestrator.agents.file_lookup_guidance import FILE_LOOKUP_GUIDANCE
from orchestrator.agents.grep_pattern_guidance import GREP_PATTERN_ESCAPING_GUIDANCE
from orchestrator.agents.partial_failure_guidance import MULTI_PATH_PARTIAL_FAILURE_GUIDANCE
from orchestrator.agents.path_not_found_guidance import PATH_NOT_FOUND_GUIDANCE
from orchestrator.agents.pipeline_status_guidance import PIPELINE_EXIT_STATUS_GUIDANCE
from orchestrator.agents.pkill_guidance import PKILL_SELF_MATCH_GUIDANCE
from orchestrator.agents.premise_refutation_guidance import PREMISE_REFUTATION_GUIDANCE
from orchestrator.agents.python_literal_guidance import PASTED_TEXT_PYTHON_LITERAL_GUIDANCE

# Task 2761. Spliced only through BACKGROUND_WAIT_GUIDANCE: its "~25-minute
# threshold described below" resolves only inside that unit.
BACKGROUND_TASK_WARNING = """
## CRITICAL: Never end your turn while a backgrounded command is still running

Do NOT launch a long-running command in the background (`Bash` with
`run_in_background=true`) and then end your turn to "wait for it to finish" or
"wait for the completion notification". This is a headless `claude --print`
process: the moment you end your turn it exits reporting SUCCESS, and the
still-pending background work is silently abandoned mid-task — leaving a
half-done tree that is falsely recorded as a completed, successful run.

- Run long commands (builds, full test suites, verification) in the FOREGROUND
  and wait for them to complete — up to the ~25-minute threshold described
  below, past which you must background it and poll instead. A foreground
  command that legitimately runs that long is fine; an abandoned background one
  never is.
- If you genuinely must use `run_in_background=true`, you MUST see it through
  BEFORE ending your turn. The launch hands you an output FILE PATH; `Read` on
  that path is how you collect the result, so read it until the command has
  actually finished — or terminate the task outright. Never end the turn with a
  background command still pending.
"""


# Task 3607; the `Monitor` bullet is task 5969. The ~25-minute threshold and
# the per-role ceilings are derived at
# orchestrator/tests/test_roles_wait_pattern.py::test_session_idle_bound_config_drift_tripwire.
# The built-ins it names are reachable whatever a role's allowed_tools say:
# orchestrator/tests/test_roles_harness_tool_inventory.py.
WAIT_PATTERN_GUIDANCE = """
## CRITICAL: How to wait for something

The rule above is one half of the contract — it tells you not to end your turn
on pending background work. This is the other half: how to actually wait.

SANCTIONED
- Default, for anything that finishes inside ~25 minutes: run it in the
  FOREGROUND with an explicit Bash `timeout` sized to the expected wall clock
  plus margin. It is in MILLISECONDS and DEFAULTS to 120000 (2 minutes) when
  omitted. A 5-minute command silently killed at that 2-minute default is the
  exit-143 failure mode you will otherwise misdiagnose as a hung command.
  Foreground-and-finished beats backgrounded-and-abandoned.
- BACKGROUND anything that will not produce a NEW assistant turn for more than
  ~25 minutes. Sizing a long run to a big foreground `timeout` does NOT buy you
  that time. The binding constraint is not the harness Bash cap (3900000 ms /
  65 minutes) — it is the orchestrator watchdog that kills your entire SESSION.
  A blocking foreground Bash call emits NO turn while it runs, so a 45-minute
  verify sized to a 2700000 timeout sits well under the harness cap and still
  gets the session killed around 30 minutes, losing everything it observed —
  the same lose-all-observations failure the `until`-loop bullet warns about.
  KNOW WHICH CEILING YOU ARE UNDER, because it decides what polling buys you:
  - architect, implementer, debugger, merger, simple_task are watched on IDLE
    time — the kill fires only after no new transcript turn for
    max(working_idle_secs, your role timeout), stock 1800s = 30 minutes. Here
    polling genuinely helps: any tool call that engages the background task —
    a `Read` of its output file — IS a new turn and resets that clock, which is
    what makes a long backgrounded run survivable.
  - As STEWARD your ceiling is FLAT wall clock — 1800s = 30 minutes from
    session start — and NOTHING resets it. Polling does not extend it. Work
    that cannot finish inside that window must be sized down or handed off; no
    wait strategy rescues it, so do not start a 40-minute job and plan to poll.
  - As deep_reviewer no watchdog ceiling fires at all; only the harness Bash
    cap binds you. The don't-end-your-turn rule above still applies in full.
  So: `Bash` with `run_in_background=true`. The launch returns a task ID AND
  an output FILE PATH, and says it will notify you. `Read` that path — it is
  the entire collection mechanism, and there is NO poll tool in this build to
  go hunting for. While the command runs the file holds the output so far;
  once it finishes the file holds the full output plus an "[exited with code
  N]" trailer carrying the real exit status, so a single Read gives you both
  the result and whether it passed (measured 2026-09-20 on a probe that exited
  7). Keep reading until you have that trailer, BEFORE ending your turn. To
  terminate instead, find the build's termination tool with `ToolSearch`
  first — it is DEFERRED, so calling it cold is the `InputValidationError`
  described below; today that tool is `TaskStop` (measured 2026-09-20), and if
  `ToolSearch` stops returning that name it has been renamed, so discover the
  current one rather than improvising. If a tool you reach for is absent, do
  NOT fall back to a `ps`/`pgrep` liveness loop — that improvisation is the
  failure this guidance exists to prevent; read the output file instead.
  Launching it detached with `setsid` and reading its log file is the same
  sanctioned shape; so is backgrounding a wait command that EXITS on its own
  when the condition holds. A pid-based foreground wait
  (`timeout N tail --pid=<pid> -f /dev/null`) is a LAST resort, and only if you
  captured the pid AT LAUNCH: `run_in_background` hands you a task ID and a
  path, never a pid, and a pid guessed with `pgrep -f` matched a wrapper
  process in measurement, so the wait blocked its full timeout and exited 124
  though the job had finished long before. Reading something you launched is
  NOT the ad-hoc wait prohibited below: you have a real completion signal, and
  you stay until you have it.
- `Monitor` is NOT a way to wait for something you launched: it takes no task
  or shell id to join on. It runs a shell command of its own and turns each
  line that command prints into a notification, so it fits a recurring event
  feed. Cold calls to it almost all guessed a wait-for-this-id shape
  (`target`, `shellId`, `taskId`) and were rejected with
  `InputValidationError`. To await your own background command, `Read` its
  output file as above. When a harness message (the `Bash` tool description,
  or the sleep guard's block) says to "use Monitor with an until-loop", run
  that until-loop as a `run_in_background` `Bash` command and read its output
  file instead; the loaded Monitor description itself steers a
  one-notification wait there too. If you do need a feed: Monitor is DEFERRED,
  so load its schema with `ToolSearch("select:Monitor")` BEFORE the first
  call, and take every parameter name from that schema, never from memory —
  its parameters have changed within a month. Even loaded, it cannot hold a
  wait for you: every Monitor expires at a deadline the harness caps, and the
  loaded schema states that cap. Its notifications reach you only while you
  are still working, so ending your turn to await one exits this session,
  exactly as a pending background command does.

NEVER — each of these cost a real session a turn or an entire wait
- `sleep N; tail ...` / `sleep N && cat ...` chained to poll background output.
  The harness anti-polling guard BLOCKS foreground sleep and spends one of your
  turns teaching you this. Do not retry with a different sleep spelling.
- Back-to-back `Read`/`Bash` calls fired as an ad-hoc DELAY — re-reading an
  unchanged file to "let some time pass". That is not waiting: it burns turns
  and context and observes nothing new. (Polling a genuine completion signal,
  as above, is a different thing and is fine.)
- A hand-rolled FOREGROUND `until`/`while` busy-loop. It runs until the Bash
  timeout SIGTERMs it (exit 143) and you lose EVERYTHING it had observed —
  strictly worse than not waiting at all. The same loop is fine BACKGROUNDED,
  where no foreground timeout applies.
- `ScheduleWakeup` as a generic delay. It is scoped to `/loop` dynamic mode,
  requires its `prompt` argument, and does nothing for a one-shot dispatched
  session.
"""


# Task 5519. It reads shell 128 + N codes against the clock;
# orchestrator/src/orchestrator/verify_classify.py::is_external_kill_rc is a
# different rule, over negative asyncio returncodes, that ignores a shell 137.
EXTERNAL_KILL_GUIDANCE = """
## Reading an exit code: your own timeout, or an external kill?

Read the code against the CLOCK: a command that died far short of the
`timeout` you set was not timed out, whatever the code says. The three codes:
- 143 = 128 + 15, SIGTERM: your own Bash `timeout` (120000 ms when omitted).
- 124: a GNU `timeout N` prefix YOU wrote into the command, as above.
- 137 = 128 + 9, SIGKILL. Well short of your budget, that is an EXTERNAL
  killer: the Linux OOM killer, or an operator or sweep `kill -9`. It is NOT
  the session watchdog: that kill takes your whole session, so if you are
  reading the code, it was not the watchdog. Raising `timeout` cannot fix an
  external kill, and re-running the identical command usually reproduces it.

A killed run says NOTHING about the code under test: it is neither failing nor
hanging. Never report it as a test failure or a hang. Say it was killed
externally, and quote the raw code.

Do NOT pipe a long run into `| tail`, `| head` or `| grep`. A kill that takes
the process group also takes whatever was buffered in the pipe, leaving a code
and no output. Redirect to a file and `Read` the file: a killed run then still
leaves its partial output on disk. For what a pipe does to the exit code
itself, see the pipeline exit-status section below.

To diagnose a 137, re-run it once, on ONE line:

    /usr/bin/time -v <cmd> > <log> 2>&1; echo "rc=$?"

Make the log path unique (carry your task id): /tmp is shared by every agent
on the host. `Read` the END of the log: "Command terminated by signal 9", the
elapsed wall clock, and the max RSS (the largest single process; xdist workers
are not summed). Before and after that run, read the `oom_kill` line of
`/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)/memory.events`. A rise is
strong evidence of OOM, not proof: other agents share that cgroup.

Then, in order:
- Elapsed close to your budget: it was your timeout. Re-size it, or
  background it per the rules above.
- Short of budget with `oom_kill` risen or a large max RSS: shrink the run.
  Shard by directory, cut xdist workers (`-n 4`, not `-n auto`, which starts
  one per core), or run the heaviest subset alone.
- Neither: escalate rather than guess a cause (`escalate_blocker` with
  `category='infra_issue'` if you hold it, otherwise `escalate_info`), quoting
  the raw code, elapsed time, max RSS and `oom_kill` before and after.
"""


# The single splice unit: never splice a member alone. Pinned by
# orchestrator/tests/test_roles_wait_pattern.py::test_combined_guidance_composes_every_rule.
BACKGROUND_WAIT_GUIDANCE = (
    BACKGROUND_TASK_WARNING + WAIT_PATTERN_GUIDANCE + EXTERNAL_KILL_GUIDANCE
)


# The pointer form, for a TURN prompt whose role already carries
# BACKGROUND_WAIT_GUIDANCE: today build_amender_prompt, under IMPLEMENTER. It
# must still state the operative rules itself. Its precondition, and the half
# no test pins: orchestrator/tests/test_roles_wait_pattern.py::test_amender_reminder_cannot_dangle.
WAIT_PATTERN_REMINDER = """
Reminder — verification is exactly where the wait rules above bite. If it will
finish inside ~25 minutes, run it in the FOREGROUND with an explicit Bash
`timeout` (milliseconds; 120000 default). If it will run longer than that
WITHOUT emitting a new assistant turn, background it and collect the result by
reading the output file path the launch returns, until that file shows the
command has exited, before you end your turn — the deciding limit is the
working-regime watchdog, which kills this SESSION after ~30 minutes with no new
turn, well short of the 3900000 harness ceiling, and each read is a new turn
that resets that clock. Never end this turn with verification still pending:
this session is one-shot, and abandoned work is recorded as a successful run."""


# Census 2026-08-16 §1.1 (plans/confusion-census-2026-08-16.md), task 4273.
# Finding and carrier set: orchestrator/tests/test_roles_tool_call_rejection.py.
_TOOL_CALL_REJECTION_KNOWN_SHAPES = """
## Reading a tool-call rejection before you retry

`InputValidationError` reports a defect in the CALL you just made, and it
echoes back the exact bytes you sent — read that echo before retrying. It
reports several distinct shapes, and they do not share a fix:

- "could not be parsed as JSON" means the call's SYNTAX is broken and no
  parameter VALUE is implicated. Re-emit the entire call from scratch; never
  copy the rejected call and edit one field. Catalogued sightings of this
  shape are `Read` calls malformed in the `offset`/`limit` pagination
  region — an empty slot, a stray extra value, a bare value with no key, or
  the closing brace placed before `limit` — and the defect you see may be
  none of these. Whatever the echo shows, the fix is the same re-emit from
  scratch: editing a value such as `limit` and resubmitting the same
  malformed structure reproduces the identical rejection, because editing a
  value cannot fix a syntax error.
- A rejection naming a parameter the schema does not have is the opposite
  problem: the call parsed fine, but a field is wrong — typically a deferred
  tool called before `ToolSearch` loaded its schema. Consult the schema; do
  not re-emit blindly.
"""


# Census 2026-08-21 §1.1 (plans/confusion-census-2026-08-21.md), task 4578: a
# third InputValidationError shape, composed in as a bullet.
MISSING_REQUIRED_PARAMETER_REJECTION = """\
- A rejection naming a REQUIRED parameter that is MISSING is a third shape,
  distinct from both above: the call parsed fine and the parameter it
  names IS a real, accepted field — a required sibling was left out. No
  schema-not-sent trailer, and the named parameter is real rather than
  invented, means a required sibling is missing rather than the
  deferred-tool problem above — re-emit the whole call with every required
  field supplied; do not edit or drop the field that was already accepted,
  and do not reach for `ToolSearch`. Catalogued sighting: an `Agent` call
  that supplied `prompt` and omitted the required `description`, rejected
  before any sub-agent launched.
"""


# The single splice unit.  SYSTEM prompts embed THIS, never either half on its
# own -- test_roles_tool_call_rejection.py asserts the composition so the
# three shapes cannot drift apart.
TOOL_CALL_REJECTION_GUIDANCE = _TOOL_CALL_REJECTION_KNOWN_SHAPES + MISSING_REQUIRED_PARAMETER_REJECTION


# Census 2026-08-31 §1.1 (plans/confusion-census-2026-08-31.md), task 4964.
# Finding, scope and editing rule: orchestrator/tests/test_roles_error_remedy_hint.py.
ERROR_REMEDY_HINT_GUIDANCE = """
## A tool error's remedy hint may name a parameter — check your tool list first

When a tool result rejects your call and its remedy clause names a bare
identifier ("set X to true", "pass Y"), the discriminator is whether that
identifier is in your own tool list. If it is NOT, it is a PARAMETER of
the tool that just errored — not a tool you can call. Re-issue THAT SAME
tool call with the parameter set, carrying over every argument from the
failed call; the remedy names the one field to change, not a standalone
action. If it IS in your tool list — e.g. a deferred tool's rejection
naming `ToolSearch` — call it for real: that is a genuine "call this tool
first," already covered by the section above, not this one.

Catalogued sighting of the parameter case: `Edit` failed with "Found 3
matches of the string to replace, but replace_all is false", and the next
turn called a bare `replace_all` with an empty argument object — rejected
with "No such tool available: replace_all", one turn lost. `replace_all`
never appears in any tool list; the empty argument object is the other
tell: read as a tool, the name carries nothing from the failed edit, so
there is nothing for it to act on.

For that `Edit` case both remedies are legitimate and you choose on
INTENT: re-issue `Edit` with `replace_all: true` only if you really do
mean every occurrence; otherwise lengthen `old_string` until it matches
uniquely. The hint reports why the edit was refused; it is not a claim
that replacing all of them is what you wanted.
"""


# Legibility-census finding, task 5331. Its "the section just above" means
# ERROR_REMEDY_HINT_GUIDANCE, true only while every splice abuts the two
# (orchestrator/tests/test_roles_grep_lookaround.py::test_variant_placement_is_structural);
# reword it if that pin changes. Variant rationale: that test module.
_GREP_ENGINE_LIMITS = """
## The `Grep` tool's regex engine rejects look-around outright

`Grep` runs ripgrep's default engine: no look-ahead, no look-behind, no
backreferences. A pattern using any of them is REJECTED before the search
runs — `error: look-around, including look-ahead and look-behind, is not
supported`, and you get no results at all rather than a partial or degraded
match. Know this before you write the pattern. `multiline: true` does not
change it: same engine either way.

The rejection prints "Consider enabling PCRE2 with the --pcre2 flag". That
remedy is addressed to ripgrep's command line, NOT to the `Grep` tool, which
exposes no such parameter under any spelling — not `--pcre2`, not `pcre2`,
and not a flag smuggled into `pattern`. Do not retry the call. This is the
discriminator against the section just above: THERE the remedy named a real
parameter of the erroring tool and re-issuing with it set is the fix; HERE it
names a flag of the WRAPPED BINARY that the tool does not expose, so no
re-issue of `Grep` can ever succeed and you must change approach instead.

FIRST-LINE RECOVERY, needing no other tool: express the intent without
look-around — match broadly, then exclude from the results. A rejected
`config\\.(?!git|project_root|verify_env)[a-z_]+` means "every `config.<key>`
except those three names"; the working form is `Grep` for `config\\.[a-z_]+`
with `-o`, then drop the three from what comes back. "Not followed by", "not
preceded by" and "the same capture twice" are all this one shape: the engine
cannot express them, and a second pass over the matches can.
"""


# Recourse for the roles holding unqualified `Bash`. Concatenated onto
# _GREP_ENGINE_LIMITS below -- never used alone.
_GREP_PCRE_BASH_RECOURSE = """
When look-around or a backreference is genuinely required, the escape hatch
is `Bash`:

    command grep -rP --include='*.py' --exclude-dir=.venv --exclude-dir=node_modules --exclude-dir=.git '<pattern>' <path>

`-P` is real PCRE, so the rejected pattern runs unchanged, and `--include=`
is the glob filter `Grep` would have applied.

The `--exclude-dir=` flags are not boilerplate: they stand in for a SECOND
filter `Grep` applied for you and real grep does not. `Grep` reads ignore
files — `.gitignore` at each level, including the catch-all one a virtualenv
writes inside itself — while GNU grep reads none of them and walks straight
into a vendored tree. Measured in this worktree, `--include='*.py'` ALONE
still matched hundreds of third-party site-packages files under `.venv` for a
pattern as ordinary as `config\\.[a-z_]+`, not one of which `Grep` would have
returned. So point `<path>` at a source directory rather than the repo root,
and if the rejected `Grep` call omitted `path`, do not simply substitute `.`.

The `command` prefix is load-bearing, not decoration. The harness shell
snapshot shadows some search commands with shell functions, and WHICH ones is
not stable across sessions or hosts: one measured shadow wrapped `grep` in a
.gitignore-respecting tool that silently skipped an ignored tree and returned
a fast, confident, empty result. `command` reaches the real `/usr/bin/grep`
either way.

`rg --pcre2` also works, and needs none of those exclusions — `rg` IS the
engine `Grep` wraps, so it honours the same ignore files. Its one limit is
where it survives. `rg` here is currently a shell FUNCTION that re-execs the
Claude Code bundle, not a binary on `PATH`, so it is lost in a nested shell
PROCESS: a `bash -c` or `sh -c`, or a standalone script file you write and
then run. It DOES survive a pipeline, a `$(...)` substitution and an explicit
`( ... )` subshell in the top-level `Bash` call — those stay in the shell that
defines it. `which rg` finds nothing either way, so do not use that to decide.
Reach for `command grep -rP` when a nested shell process is involved.
"""


# JUDGE's recourse. Its `Bash` grant is `Bash(git:*)`, so the escape hatch
# above would land it a permission denial rather than a result -- naming a
# command the role cannot run is the failure this variant exists to avoid.
# Concatenated onto _GREP_ENGINE_LIMITS below -- never used alone.
_GREP_PCRE_READ_ONLY_RECOURSE = """
Shelling out is NOT available to you here: your `Bash` grant is restricted to
git commands, so a `grep -P` call would land a permission denial rather than
a result. The restructure above IS your recovery — match broadly with `Grep`
and exclude afterwards. Where the candidate region is small enough, `Read` it
and judge the text directly instead of searching it.
"""


# The two splice units: Bash-capable roles take the first, JUDGE the second.
GREP_LOOKAROUND_GUIDANCE = _GREP_ENGINE_LIMITS + _GREP_PCRE_BASH_RECOURSE
GREP_LOOKAROUND_GUIDANCE_READ_ONLY = _GREP_ENGINE_LIMITS + _GREP_PCRE_READ_ONLY_RECOURSE


# The harness guidance every role with a literal system_prompt and unqualified
# `Bash` carries, spliced straight after its one-line role statement. One
# composite, so a new block is one edit here rather than one per carrier. Each
# block's own test module still pins its carrier set against a capability, so
# a role for which some block stops applying builds its own chain rather than
# dropping that block from this one -- JUDGE already does: no wait block and
# the read-only grep variant.
#
# APPEND-ONLY AT THE TAIL. Each block through GREP_LOOKAROUND_GUIDANCE has its
# own test module pin it to start exactly where its predecessor ends; blocks
# appended after it pin only ORDER (after GREP_LOOKAROUND_GUIDANCE), so
# concurrent appends do not break each other. A tail block's anchor test
# subclasses orchestrator/tests/_role_splice_contract.py::PreambleTailContractTests,
# which applies that order pin; do not clone its tests. BACKGROUND_WAIT_GUIDANCE's
# heading must stay the prompt's first `##`.
# _GREP_ENGINE_LIMITS's prose also points at ERROR_REMEDY_HINT_GUIDANCE as "the
# section just above". Inserting anywhere but the end breaks a pin, or
# silently redirects that pointer.
BASH_CAPABLE_ROLE_PREAMBLE = (
    BACKGROUND_WAIT_GUIDANCE
    + TOOL_CALL_REJECTION_GUIDANCE
    + ERROR_REMEDY_HINT_GUIDANCE
    + GREP_LOOKAROUND_GUIDANCE
    + PKILL_SELF_MATCH_GUIDANCE
    + CHAINED_COMMAND_STATUS_GUIDANCE
    + GREP_PATTERN_ESCAPING_GUIDANCE
    + PASTED_TEXT_PYTHON_LITERAL_GUIDANCE
    + MULTI_PATH_PARTIAL_FAILURE_GUIDANCE
    + PATH_NOT_FOUND_GUIDANCE
    + BASH_CWD_ANCHOR_GUIDANCE
    + FILE_LOOKUP_GUIDANCE
    + PIPELINE_EXIT_STATUS_GUIDANCE
    + PREMISE_REFUTATION_GUIDANCE
)
