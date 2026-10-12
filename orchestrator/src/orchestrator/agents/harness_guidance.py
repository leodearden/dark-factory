"""Role-prompt guidance about the agent harness, and the Bash-capable preamble that composes it."""

from orchestrator.agents.bash_cwd_guidance import BASH_CWD_ANCHOR_GUIDANCE
from orchestrator.agents.chained_command_guidance import CHAINED_COMMAND_STATUS_GUIDANCE
from orchestrator.agents.file_lookup_guidance import FILE_LOOKUP_GUIDANCE
from orchestrator.agents.grep_pattern_guidance import GREP_PATTERN_ESCAPING_GUIDANCE
from orchestrator.agents.partial_failure_guidance import MULTI_PATH_PARTIAL_FAILURE_GUIDANCE
from orchestrator.agents.path_not_found_guidance import PATH_NOT_FOUND_GUIDANCE
from orchestrator.agents.pkill_guidance import PKILL_SELF_MATCH_GUIDANCE
from orchestrator.agents.premise_refutation_guidance import PREMISE_REFUTATION_GUIDANCE
from orchestrator.agents.python_literal_guidance import PASTED_TEXT_PYTHON_LITERAL_GUIDANCE
from orchestrator.agents.sigpipe_guidance import SIGPIPE_UNDER_PIPEFAIL_GUIDANCE

# Layer-1 guard against the headless --print background-task footgun (task
# 2761).  A single source of truth, no longer spliced on its own: task 3607
# folded it into BACKGROUND_WAIT_GUIDANCE (below), which is what the seven
# Bash-capable role prompts now embed — see constraint (b) there.  Like
# _ESCALATION_INSTRUCTIONS / MANDATED_STAGING_COMMAND it is anchored by a
# regression test (test_roles_background_warning.py) so a prompt refactor can't
# silently drop it.
#
# The first bullet's "~25-minute threshold described below" forward-references
# WAIT_PATTERN_GUIDANCE, and resolves ONLY because the two always ship together
# as BACKGROUND_WAIT_GUIDANCE (constraint (b) already forbids splicing either
# half alone).  Do NOT drop that qualifier back to the pre-3607 wording ("a
# foreground command that legitimately runs long is fine", unqualified): in the
# composed block it sat a few lines above the rule that a long FOREGROUND call
# is exactly what gets the session killed, so the agent read "long foreground is
# fine" immediately before "long foreground kills your session".
#
# Plain text, NO literal `{`/`}` braces.  Defensive rather than load-bearing:
# nothing interpolates this constant today (briefing.py takes
# WAIT_PATTERN_REMINDER), and role prompts are plain `+` concatenation — see the
# MANDATED_STAGING_COMMAND note above, they are deliberately NOT f-strings
# because they contain literal braces.  Staying brace-free keeps it safe if a
# future splice site does interpolate it.
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


# Census-R3 companion to BACKGROUND_TASK_WARNING (task 3607, census
# 2026-08-02 §1.1).  2761 told agents what NOT to do; it never told them what
# to do instead, so sessions kept improvising a wait — blocked `sleep` chains,
# back-to-back tool calls as an ad-hoc delay, hand-rolled `until` busy-loops
# that get SIGTERMed at the Bash timeout.  This states the sanctioned wait
# paths.  Anchored by test_roles_wait_pattern.py, which pins two HARD
# constraints:
#   (a) NO literal `{`/`}` braces.  Defensive, NOT load-bearing, and the
#       distinction matters to whoever edits these next: this constant reaches
#       role prompts only by plain `+` concatenation, which is brace-safe by
#       construction (role prompts are deliberately not f-strings — see the
#       MANDATED_STAGING_COMMAND note above — precisely because they DO contain
#       literal braces).  The one constant here that genuinely must stay
#       brace-free is WAIT_PATTERN_REMINDER, which briefing.py interpolates into
#       build_amender_prompt's f-string.  Keeping this one brace-free too costs
#       nothing and keeps it interpolation-safe if a future splice site needs it.
#   (b) BACKGROUND_WAIT_GUIDANCE (defined below) is the ONLY thing that may be
#       spliced into a role prompt.  Splicing either half on its own reopens
#       the exact footgun-trade the composition mandate exists to prevent: the
#       prohibition without the pattern breeds busy-loops, the pattern without
#       the prohibition breeds abandoned background work.
#
# TOOL AVAILABILITY -- read what this block does and does not establish.  It
# records that built-ins are REACHABLE.  It is NOT, and cannot be, an inventory
# of WHICH built-ins exist -- see the retraction below, which this block has
# now earned twice.  For that, re-derive with `ToolSearch`.
#
# REACHABILITY -- measured 2026-08-05, do NOT re-derive.  The tools named below
# (`Monitor`, `ToolSearch`, `ScheduleWakeup`) are absent from every role's
# `allowed_tools`, and that is CORRECT, not a bug: the `--allowed-tools` flag
# cli_invoke passes (cli_invoke.py:1830) is a PERMISSION allowlist -- what may
# run without a prompt -- not a tool-registry filter.  Built-in harness tools
# remain present and callable regardless.  Verified from inside a dispatched
# implementer session whose cmdline was exactly `--allowed-tools Read Edit
# Write Bash Glob Grep mcp__...`: `ToolSearch` and `ScheduleWakeup` were live
# in that session's tool list, and a `ToolSearch` select query for deferred
# tools returned their schemas.  So do NOT "fix" this block by trimming those
# tools out of it on the theory that the roles cannot reach them -- they can.
#
# RETRACTED 2026-09-20 (task 5332), in place rather than deleted, so the next
# editor can tell which of this block's claims rest on what.  The block used to
# continue: "The same holds for the `BashOutput`/`KillShell` that
# BACKGROUND_TASK_WARNING has named since task 2761."  That sentence was never
# measured -- it inherited the authority of the genuine measurement above it
# through the words "The same holds for", and later edits then cited the whole
# block as settled.  It is false: neither name has ever resolved from the
# registry, and no transcript in the fleet census ever shows one being called.
# THE MEASUREMENT AND ITS DATES ARE NOT RESTATED HERE -- they live once, at
# tests/test_roles_harness_tool_inventory.py::MEASURED_ABSENT_TOOLS, for the
# same reason the size figures below live at exactly one site.
#
# The retraction GENERALISES, and that is the part worth carrying forward: this
# block has now aged wrong TWICE in the same way.  Its own worked example above
# used to cite a `ToolSearch` select query returning a `TaskOutput` schema --
# and ten days later that name no longer resolved at all (both measurements at
# the citation above).  Reachability is durable; the roster is not.  So the
# guidance below is deliberately built on the things that do not churn --
# `Bash`'s own `run_in_background` parameter, `Read` on the output file path it
# returns, and `ToolSearch` as the way to DISCOVER the current termination tool
# -- and names a concrete tool only where unavoidable, stamped with its
# measurement date.  tests/test_roles_harness_tool_inventory.py now holds the
# "a named built-in must be a real built-in" invariant mechanically.
#
# THE ~25-MINUTE BACKGROUND THRESHOLD is deliberate and is NOT the harness Bash
# cap; do not "simplify" it back to 3900000.  But the watchdog mechanism differs
# by DISPATCH SITE, and the prompt text says so explicitly, because assuming it
# is uniform is the error the task-3607 review caught (round 5).  Measured, not
# assumed -- cli_invoke.py gates the whole idle regime on
# `extension_engaged = seen_turn and working_idle_secs is not None and
# absolute_cap_secs is not None`:
#
#   * workflow._invoke (workflow.py:11791-11792) passes BOTH extension params,
#     so extension_engaged latches True and the watchdog measures IDLE time: it
#     kills only after no NEW transcript turn for
#     max(working_idle_secs, timeout_seconds).  Polling a backgrounded command
#     genuinely resets that clock, since each read of the background task's
#     output file is a new turn.
#     Covers architect (2400s), implementer/debugger (1800s), merger (1800s),
#     simple_task (7200s) -- each being max(working_idle_secs=1800, role).
#   * steward.py:806 passes `timeout_seconds=timeouts.steward` and NEITHER
#     extension param, so extension_engaged is False and cli_invoke takes the
#     `else` arm: a FLAT `elapsed >= timeout_seconds` measured from
#     watchdog_start, i.e. 1800s from session start.  Polling resets NOTHING
#     for the steward.  Do not tell the steward otherwise.
#   * review_checkpoint.py:211 passes no timeout_seconds at all, so the else
#     arm's own `timeout_seconds is not None` guard is False and NO watchdog
#     kill fires for deep_reviewer; only the harness Bash cap binds it.
#
# ~25 minutes is the safe default because 1800s = 30 min is the tightest real
# ceiling among the roles that have one, and it is the FLAT one, so it has no
# slack to reclaim.  The 120000 default / 3900000 max remain true harness facts
# and are still stated as such -- they are just not the background-vs-foreground
# decision threshold.
#
# Threading the extension params into the steward/deep_reviewer dispatch sites
# so the idle mechanism engages uniformly is a deliberate NON-goal here: it
# would change steward's effective budget from a flat 1800s to
# invocation_timeout (7200s), a real operational change that belongs in its own
# task rather than riding along in a prompt-wording one.
# (Task 3607 review rounds 1 and 5, reviewer_comprehensive.)
# Task 5969 (refile of reify #7907/#7915) rewrote the `Monitor` bullet to lead
# with what Monitor is NOT: cold calls kept guessing a join-on-an-id shape
# despite the ToolSearch-first clause.  It states no cap figure because the
# harness cap churns; the measurements are in that commit's message.
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


# Third member of the wait block (task 5519): reading the exit code a wait
# returned -- your own timeout, or an external kill.  Constraint (b) above binds
# it, and it stays brace-free like the two members above.  It reads shell
# 128 + N codes against the clock; verify_classify.py::is_external_kill_rc is a
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

Do NOT pipe a long run into `| tail`, `| head` or `| grep`. Bash here runs
without `pipefail`, so a pipeline reports its LAST stage's status and the real
exit code is thrown away. A kill that takes the process group also takes
whatever was buffered in the pipe, leaving a code and no output. Redirect to a
file and `Read` the file: a killed run then still leaves its partial output
on disk.

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


# The single splice unit.  SYSTEM prompts embed THIS, never any member on its
# own -- see constraint (b) above; test_roles_wait_pattern.py asserts the
# composition so the three rules cannot drift apart.
BACKGROUND_WAIT_GUIDANCE = (
    BACKGROUND_TASK_WARNING + WAIT_PATTERN_GUIDANCE + EXTERNAL_KILL_GUIDANCE
)


# Pointer form for a TURN prompt whose role system_prompt already carries the
# full block above.  Restating the whole block at the failure site would hand
# the agent the identical text twice in one session -- the very double-splice
# that test_combined_guidance_appears_exactly_once_per_role polices WITHIN a
# system prompt, just spread across the system/turn pair where that test cannot
# see it.  The point of the at-the-failure-site injection was always ADJACENCY
# (put the rule next to the action item that trips it), and the pointer buys
# that for ~8% of the block's bytes.
#
# THE ONLY SIZE FIGURES IN THIS FEATURE LIVE HERE.  Measured on this revision:
# BACKGROUND_WAIT_GUIDANCE 10022 B (= BACKGROUND_TASK_WARNING 1193 +
# WAIT_PATTERN_GUIDANCE 6411 + EXTERNAL_KILL_GUIDANCE 2418),
# WAIT_PATTERN_REMINDER 766 B -> 766/10022 = 7.6%.
# Re-derive rather than trust these after any edit to the strings:
#   python -c "from orchestrator.agents.roles import *; \
#              print(len(BACKGROUND_WAIT_GUIDANCE), len(WAIT_PATTERN_REMINDER))"
# Every other mention of the block's size in roles.py and
# test_roles_wait_pattern.py is deliberately QUALITATIVE ("the full block").
# An earlier revision hand-copied the figure to 8 sites (2 here, 6 in the test
# file) and every one was wrong: 5 said "~3.2 kB" and 2 said "~2.6 kB" against
# a real 4.4 kB, and 1 said the pointer costs "~2% of the tokens" against the
# then-real ~15% -- a 7x error (task 3607 review).  Those are the figures as
# that review found them; both constants have grown since, which is exactly
# why the live numbers above are re-derived rather than carried forward.
# Prose copies do not move when the string they describe does.  Do not reintroduce a number anywhere else --
# cite this comment instead.
#
# Use this ONLY where the receiving role is statically known to carry
# BACKGROUND_WAIT_GUIDANCE, or the pointer dangles and the agent is left with
# neither half.  Today that is build_amender_prompt alone, which is invoked
# under IMPLEMENTER at exactly one call site (workflow.py `_invoke(IMPLEMENTER,
# ...)`).  Note precisely what is and is not machine-checked: the IMPLEMENTER
# half of that precondition IS pinned (test_roles_wait_pattern.py asserts
# IMPLEMENTER carries the block), but "the amender runs under IMPLEMENTER" is
# NOT -- that call site is reachable only by reading workflow.py, and a
# source-text scan for `_invoke(IMPLEMENTER` is exactly the brittle
# literal-over-prose pin this feature's tests were twice told to drop.  So if
# you re-point build_amender_prompt at another role, no test will stop you;
# check that role carries the block yourself.  Same no-literal-braces rule as
# the constants above.
#
# Keep it SUBSTANTIVE when editing: a pointer still has to STATE the operative
# rules at the failure site -- foreground with an explicit `timeout`, or else
# background it and read its output file to completion before ending the turn --
# because that adjacency is the whole reason for the injection.  Do not let it
# decay into a bare "see the rule above" cross-reference; a reader who follows
# no cross-reference gets nothing.  This is an editorial expectation, so it
# lives here rather than as an assertion: a length floor would pass on filler
# and fail on a tighter, better-worded reminder.
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


# Census-2026-08-16 §1.1 companion to WAIT_PATTERN_GUIDANCE's `Monitor` bullet
# above (task 4273; plans/confusion-census-2026-08-16.md).  Third sighting of
# the same shape catalogued as `cand-20260806-12` and `cand-20260812-19` in
# docs/legibility/confusion-codebook.yaml: a `Read` tool call whose echoed
# input carried an empty parameter slot between `offset` and `limit`
# (`"offset": 1240, , "limit": 1`) was rejected by `InputValidationError` as
# unparseable JSON, and three turns later the agent reissued the IDENTICAL
# malformed structure with only `limit` edited (1 -> 260) -- rejected
# identically.  The retry encoded a misdiagnosis: the agent read the
# rejection as a bad `limit` VALUE rather than a JSON SYNTAX defect.
#
# WHAT IS AND IS NOT ADDRESSED.  The stray comma's CAUSE is upstream of this
# repository -- tool-call generation/serialization for a Claude Code builtin
# (`Read`) -- and no in-repo code path produces or can intercept it, so this
# constant does not attempt to detect or repair the malformed JSON.  Only the
# RETRY facet is in scope: the observed retry edited `limit` while leaving
# the syntax defect untouched, so the rejection reproduced identically and
# the turn was wasted.  Retry behaviour of a dispatched agent IS this repo's
# to fix, through the role system prompts.
#
# DISCRIMINATION, NOT DUPLICATION.  WAIT_PATTERN_GUIDANCE's `Monitor` bullet
# above (roles.py:490-495) already names the OTHER `InputValidationError`
# shape -- a parameter name the schema does not have, typically a deferred
# tool called before `ToolSearch` loaded its schema -- which needs the
# OPPOSITE fix (consult the schema, don't re-emit blindly).  This constant
# points at that discrimination rather than repeating it; do not delete
# either bullet as redundant with the other.  Do NOT trim that mention down
# to a generic "consult the schema" on the theory that a dispatched role
# cannot reach `ToolSearch` -- it can, regardless of whether `ToolSearch`
# appears in that role's `allowed_tools`; see the TOOL AVAILABILITY comment
# above WAIT_PATTERN_GUIDANCE (roles.py:401-413), which verified this from
# inside a live dispatched session and explicitly forbids the same trim
# there for the same mistaken reason.
#
# JUDGE IS DELIBERATELY INCLUDED, not merely uncovered by an exclusion rule.
# Task 3607 excluded judge (and reviewer_comprehensive) from
# BACKGROUND_WAIT_GUIDANCE partly on a cost framing -- neither holds
# unqualified `Bash`, so "the whole block would be dead weight in every one
# of their sessions" (the comment above
# orchestrator/tests/test_roles_wait_pattern.py::_CONTRACT).  JUDGE runs after
# EVERY implementer iteration, making it the
# highest per-invocation multiplier of the eight roles this constant is
# spliced into, and unlike the wait block, JUDGE genuinely can hit this
# rejection: it holds `Read` and runs on a tight 30-turn budget, so a
# rejected paginated `Read` it cannot diagnose costs it a turn with no other
# way to recover.  The token cost this adds to JUDGE's prompt is real and is
# accepted for JUDGE specifically -- it is not waved through by reusing the
# wait block's Bash-capability line, which would have excluded it.  The only
# exclusion in THIS constant's splice set (`_UNPINNED_PROMPT_ROLES` in
# test_roles_tool_call_rejection.py) rests on the artifact-pinning mechanism;
# there is no separate, non-arbitrary cost line to draw beside it.
#
# HARD CONSTRAINT -- do NOT add an `mcp__<family>__<name>` example to this
# constant.  test_roles_ancestry_check.py::test_role_holds_every_mcp_tool_its_prompt_names
# (task 4107) is parametrized over every role and asserts any fully-qualified
# MCP tool named in a role's system_prompt is in THAT role's allowed_tools.
# This constant is spliced into 8 roles with differing allowlists, so naming
# one MCP tool would fail for whichever roles lack the grant.  `Read`,
# `ToolSearch` and `InputValidationError` are all builtins and do not match
# that regex.
#
# Plain text, NO literal `{`/`}` braces.  Defensive rather than load-bearing,
# same as WAIT_PATTERN_GUIDANCE: role prompts reach this only by plain `+`
# concatenation, which is brace-safe by construction -- role prompts are
# deliberately NOT f-strings (see the MANDATED_STAGING_COMMAND note at
# roles.py:332) precisely because they contain literal braces.  Staying
# brace-free costs nothing and keeps this safe if a future splice site does
# interpolate it.
#
# No sighting COUNT is stated in the text below -- the codebook accrues a new
# sighting every census cycle, and a number baked into a role prompt goes
# stale silently while "catalogued sightings all show the same shape" stays
# true.
_TOOL_CALL_REJECTION_KNOWN_SHAPES = """
## Reading a tool-call rejection before you retry

`InputValidationError` reports a defect in the CALL you just made, and it
echoes back the exact bytes you sent — read that echo before retrying. It
reports several distinct shapes, and they do not share a fix:

- "could not be parsed as JSON" means the call's SYNTAX is broken and no
  parameter VALUE is implicated. Re-emit the entire call from scratch; never
  copy the rejected call and edit one field. Catalogued sightings of this
  shape all show a `Read` call whose echoed input carries an empty slot
  between `offset` and `limit` — editing `limit` and resubmitting the same
  malformed structure reproduces the identical rejection, because editing a
  value cannot fix a syntax error.
- A rejection naming a parameter the schema does not have is the opposite
  problem: the call parsed fine, but a field is wrong — typically a deferred
  tool called before `ToolSearch` loaded its schema. Consult the schema; do
  not re-emit blindly.
"""


# Census-2026-08-21 §1.1 companion to _TOOL_CALL_REJECTION_KNOWN_SHAPES above
# (task 4578; plans/confusion-census-2026-08-21.md). A dispatched session
# called the `Agent` tool supplying only a `prompt` field; the schema also
# requires `description`, and the omission was rejected by
# `InputValidationError: The required parameter 'description' is missing`
# before any sub-agent launched.
#
# WHAT IS AND IS NOT ADDRESSED. The CAUSE is upstream of this repository --
# harness-side tool-call construction for a Claude Code builtin -- and no
# in-repo code path produces or can intercept it, so this constant does not
# attempt to detect or repair the malformed call. Only the RECOVERY facet is
# in scope, the same carve-out the constant above already takes for the
# stray-comma sighting: a dispatched agent's response to the rejection is
# this repo's to fix, through the role system prompts.
#
# DISCRIMINATION, NOT DUPLICATION. This is a THIRD `InputValidationError`
# shape, not a rewording of either bullet above -- do not fold it into, or
# delete, either existing bullet as redundant. The raw error string ("The
# required parameter `X` is missing") is IDENTICAL to the deferred-tool
# shape's founding evidence above, so the string alone does not discriminate
# between the two mechanisms; read the echo instead. The deferred-tool case
# carries a trailer stating the tool's schema was not sent to the API, and
# names a parameter the schema does not have -- an INVENTED name. This case
# carries neither: the named parameter is real and accepted, and only a
# required sibling is missing.
#
# HARD CONSTRAINTS, same as the block above. No `mcp__<family>__<name>`
# example -- the ancestry-check test that guards role/tool-grant consistency
# is parametrized over every role and would fail for whichever of the 8
# roles lacks the grant; `Agent`, `ToolSearch` and `InputValidationError` are
# all builtins and do not match that test's regex, so naming them is safe,
# and `Agent` is genuinely reachable by a dispatched role regardless of
# whether it appears in that role's allowed_tools -- the same fact the
# comment above already records for `ToolSearch`. No literal `{`/`}` braces
# -- role prompts reach this only by plain `+` concatenation. No sighting
# COUNT -- the codebook accrues a new sighting every census cycle, and a
# number baked into a role prompt goes stale silently.
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


# Census-2026-08-31 §1.1 finding (task 4964; plans/confusion-census-2026-08-31.md),
# codebook entry entry-cand-20260827-23, founding session
# 7dae04c6-f0a5-400d-b199-8932d65a8790. An `Edit` call landed
# `<tool_use_error>Found 3 matches of the string to replace, but replace_all
# is false...`, and the very next turn called a bare `replace_all({})` --
# rejected with `Error: No such tool available: replace_all`, one turn lost.
# The agent read the rejection's remedy clause ("set replace_all to true")
# and bound the bare identifier `replace_all` to the wrong syntactic
# category: a callable tool rather than a parameter on a re-issued `Edit`
# call.
#
# WHAT IS AND IS NOT ADDRESSED. `Edit`'s multi-match error text is emitted
# by the harness and is upstream of this repository; no in-repo code path
# produces or can intercept it, so this constant does not attempt to
# change, wrap or detect that message. Only the RESPONSE facet is in
# scope -- an agent that reads the remedy clause correctly does not spend a
# turn on a phantom tool call. This is the same carve-out tasks 4273 and
# 4578 documented for the two `InputValidationError` findings above, and it
# is why a harness-rooted census finding still lands in this file.
#
# DISCRIMINATION, NOT DUPLICATION -- now against THREE neighbouring shapes,
# not two. `TOOL_CALL_REJECTION_GUIDANCE` above covers `InputValidationError`:
# a defect in the CALL you just made, with the echoed bytes to read, in
# three shapes (unparseable JSON / invented parameter name / missing
# required sibling). This constant covers a DIFFERENT error class: a
# well-formed call that succeeded in parsing, failed on CONTENT, and
# printed a remedy -- where the failure is in parsing the remedy, not the
# call. The census recorded "No such tool available" as spanning at least
# five catalogued mechanisms, so neither block may be collapsed into the
# other and neither may be deleted as redundant with the other.
#
# ACCURACY CAVEAT the text below must honour: the census explicitly does
# NOT establish that `replace_all: true` was the semantically correct fix
# for that particular edit. The guidance presents both remedies as chosen
# on intent -- it must never assert that setting the parameter is always
# right.
#
# HARD CONSTRAINTS, same as both neighbours above. No `mcp__<family>__<name>`
# example -- this constant is spliced into 8 roles with differing
# allowlists, and
# test_roles_ancestry_check.py::test_role_holds_every_mcp_tool_its_prompt_names
# asserts any fully-qualified MCP tool a prompt names is in THAT role's
# allowed_tools; `Edit` and `ToolSearch` are both builtins and do not
# match that regex. No literal `{`/`}` braces -- role prompts reach this
# only by plain `+` concatenation. No sighting COUNT in the prose -- the
# codebook accrues a new sighting every census cycle, and a number baked
# into a role prompt goes stale silently.
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


# Census finding, task 5331 (`metadata.source: legibility_census`),
# reproduced first-hand in the founding session and again on revalidation
# 2026-09-17:
#
#     Grep(pattern='config\.(?!git|project_root|verify_env)[a-z_]+', ...)
#     -> error: look-around, including look-ahead and look-behind, is not
#        supported
#        Consider enabling PCRE2 with the --pcre2 flag
#
# The search never runs. Two facts make that a legibility defect rather than
# a plain capability limit. The `Grep` tool's own description advertises
# "Full regex syntax" and warns only about literal braces, so the agent
# learns the limit by failure. And the printed remedy is UNREACHABLE through
# the tool that printed it: `Grep`'s schema exposes pattern/path/glob/type/
# output_mode/-i/-n/-A/-B/-C/-o/multiline/head_limit/offset and nothing that
# can carry `--pcre2`.
#
# WHAT IS AND IS NOT ADDRESSED. `Grep` is a Claude Code builtin: no in-repo
# code path produces, wraps or can intercept its pattern rejection, so
# nothing here can add a `--pcre2` option or change that message. Only the
# told-upfront/recovery facet is in scope -- an agent told the limit BEFORE
# it writes the pattern does not spend a turn discovering it, and one that
# reads the remedy correctly does not spend a second turn trying to pass
# `--pcre2` to a tool with no such parameter. Same carve-out tasks 4273,
# 4578 and 4964 documented for the three findings above, and the reason a
# harness-rooted census finding still lands in this file.
#
# DISCRIMINATION, NOT DUPLICATION -- a FOURTH shape, against three
# neighbours. `TOOL_CALL_REJECTION_GUIDANCE` covers `InputValidationError`:
# a defect in the CALL you just made, in three shapes (unparseable JSON /
# invented parameter name / missing required sibling), every one of them
# fixed by re-emitting the call correctly. `ERROR_REMEDY_HINT_GUIDANCE`
# covers a well-formed call that failed on CONTENT and printed a remedy
# naming a real PARAMETER of the erroring tool -- fixed by re-issuing with
# it set. This is neither: the call was well-formed, it failed on content,
# and the remedy names a flag of the WRAPPED BINARY that the tool exposes no
# parameter for, so NO re-issue can ever succeed and the correct response is
# to restructure the pattern or switch tools. Folding this into either
# neighbour would collapse the discrimination it exists to draw; neither
# block may be deleted as redundant with the other.
#
# THE PROSE DEPENDS ON THE PLACEMENT PIN. _GREP_ENGINE_LIMITS draws its
# discrimination by pointing at "the section just above", which is true only
# because every splice site abuts ERROR_REMEDY_HINT_GUIDANCE. That abutment is
# machine-checked by
# orchestrator/tests/test_roles_grep_lookaround.py::test_variant_placement_is_structural;
# moving this block elsewhere in a prompt silently turns that pointer into a
# reference to whatever now precedes it, so reword the sentence if the pin ever
# has to change.
#
# TWO VARIANTS, because two orthogonal capability dimensions decide who
# needs what. All 9 roles hold `Grep`, so every role can hit the rejection
# and needs the LIMITATION; but the `grep -P` escape hatch is gated on the
# `Bash` grant, and `judge` holds only `Bash(git:*)`, so prescribing it
# there would walk that role into a permission denial -- the same "agent
# follows its own instructions into a denial" defect that
# orchestrator/tests/test_roles_ancestry_check.py::test_role_holds_every_mcp_tool_its_prompt_names
# guards against for MCP tools, and which does NOT cover builtins. Composed
# the way _SCOPE_BOUNDARY_FACTS + _SCOPE_BOUNDARY_RECOURSE compose
# SCOPE_BOUNDARY_GUIDANCE below: the shared facts stated once so the two
# variants cannot drift, each half self-contained with its own
# leading/trailing blank line, plain `+` concatenation, and only the
# composed units ever spliced into a prompt.
#
# THE SHELL MEASUREMENTS BEHIND THE ESCAPE HATCH, with the commands that
# produced them, taken in a dispatched agent session in a task worktree
# (2026-09-17):
#     type -a rg               -> "rg is a function": a shell function the
#                                 harness snapshot installs, re-execing the
#                                 Claude Code bundle via CLAUDE_CODE_EXECPATH
#     which rg                 -> empty
#     bash -c 'command -v rg'  -> not found
#     type -a grep             -> /usr/bin/grep, /bin/grep; no function shadow
#     grep --version           -> GNU grep 3.11; no ugrep installed here
#     command grep -rPoh --include='*.py' <the look-ahead pattern above> \
#         orchestrator/src/orchestrator/verify.py
#                              -> exit 0, the intended matches returned
#
# TWO MEASUREMENTS ADDED ON THE AMENDMENT PASS (2026-09-19), each correcting a
# sentence that had generalized past what the 09-17 run above actually showed.
#
# (1) WHERE the `rg` shell function survives. `bash -c 'command -v rg'` tests
# one thing only -- a nested shell PROCESS -- and an earlier revision of
# _GREP_PCRE_BASH_RECOURSE read that as "never put `rg` in a script, a subshell
# or any nested shell", which is false for a subshell. Re-measured directly on
# `rg` rather than on a stand-in function:
#     rg --pcre2 -o <the look-ahead pattern above> \
#         orchestrator/src/orchestrator/verify.py | head -3
#                                       -> matches returned; pipeline is fine
#     echo "subst: $(rg --version)"     -> ripgrep 14.1.1
#     ( rg --version )                  -> ripgrep 14.1.1
#     bash -c 'rg --version'            -> rg: command not found
#     bash <standalone script file>     -> rg: command not found
# A pipeline, a command substitution and an explicit ( ... ) subshell all stay
# inside the shell that defines the function; only a new shell PROCESS loses
# it. The prose now draws the boundary there.
#
# (2) The IGNORE-FILE asymmetry the escape hatch silently dropped. `Grep`
# (ripgrep) reads ignore files; GNU grep reads none, so `--include=` alone does
# not reproduce `Grep`'s filtering:
#     rg -l -g '*.py' 'config\.[a-z_]+' .                    -> 711 files
#     command grep -rlP --include='*.py' 'config\.[a-z_]+' .  -> 1101 files
# The 385-file difference is entirely under `.venv`, and `rg --no-ignore` is
# what unlocks it -- the mechanism is `.venv/.gitignore`, a catch-all the
# virtualenv writes inside ITSELF, which is why `git check-ignore .venv` exits
# 1 (the directory is not ignored; its contents are) and why naming .gitignore
# at the repo root alone would have mis-stated the cause. Hence the
# `--exclude-dir=` flags and the scope-the-path sentence in the prose: without
# them the prescribed recovery returns a wall of third-party matches, which is
# the same wasted turn this constant exists to prevent.
#
# Those are the claims that were measured; nothing broader is asserted about
# the environment. In particular the shadowed SET is NOT stable across time
# or host: a 2026-08-11 project memory records `grep` ITSELF shadowed by a
# ugrep-wrapping shell function that honoured .gitignore and silently
# skipped a whole ignored tree, returning a fast, confident, wrong zero --
# which does not hold here. That instability is precisely why the guidance
# prescribes the `command` prefix (it resolves the real /usr/bin/grep
# whichever way a given snapshot falls) rather than depending on which
# commands happen to be shadowed, and why `rg --pcre2` is mentioned only
# with its caveat instead of being led with despite the error message
# suggesting it. Leading with `rg` would repeat the defect census
# 2026-09-10 sec 1.1 caught: role wait-guidance naming `BashOutput`, a tool
# absent from every CLI build in the corpus, sending the agent after
# something that is not there.
#
# HARD CONSTRAINTS, four, every one machine-checked:
#   - No literal `{`/`}` in the constants. Role prompts reach them by plain
#     `+` concatenation, but they are held brace-free defensively so a
#     future interpolating splice site stays safe -- CODE_QUALITY_GUIDANCE
#     below is the live example, brace-free BY CONTRACT because it reaches
#     _REVIEWER_HEURISTICS_TEMPLATE's str.format().
#   - No `mcp__<family>__<name>` name in the prompt text: the ancestry check
#     cited above is parametrized over every role, and these constants land
#     in 8 roles with differing allowlists. `Grep`, `Bash` and `Read` are
#     builtins and do not match its regex.
#   - No `Test` + CamelCase class citation that does not resolve to a real
#     class. orchestrator/tests/test_cited_test_class_drift.py statically
#     requires every such identifier in a comment or string token of any
#     member's src tree to resolve to a real test class, so every test cited
#     from here is named `path/to/test_module.py::lowercase_function`, which
#     is out of scope by shape.
#   - No sighting COUNT and no byte-size figure in the prose. The codebook
#     accrues a sighting every census cycle, and the BACKGROUND_WAIT_GUIDANCE
#     comment above records what byte-size figures cost: an earlier revision
#     hand-copied one to 8 sites and every one was wrong, one by 7x, because
#     a prose copy does not move when the string it describes does.
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
    + SIGPIPE_UNDER_PIPEFAIL_GUIDANCE
    + PREMISE_REFUTATION_GUIDANCE
)
