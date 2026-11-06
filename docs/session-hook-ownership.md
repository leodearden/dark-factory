# Session hook ownership model

This explains how the Claude Code hook trio (SessionStart, Notification, Stop)
decides which session registry record an event belongs to. The code is
`orchestrator/src/orchestrator/session_hooks.py` and
`orchestrator/src/orchestrator/session_ownership_probes.py`. The inputs it
relies on are `CLAUDE_SPAWN_SESSION_ID` and `CLAUDE_SPAWN_OWNER_PPID`, both
exported by `skills/spawn/spawn-claude.sh` into the session it launches, and
the registry's session-id and pid pointers.

## Record keying

Each spawned session has exactly one registry record. `spawn-claude.sh`
writes it as `launching` and exports that record's slug as
`CLAUDE_SPAWN_SESSION_ID`. The hooks adopt the value as the slug as-is,
after `session_registry.sanitize_slug`. They do not pass it back through
`build_session_slug`, which would prefix it twice and miss the record. All
three hook events, plus `spawn-claude.sh`'s own `launching` and `exited`
writes, therefore target the same record. It moves from launching to
running, then to idle or awaiting-input, then to exited. This was introduced
by task 2511.

A hand-launched session has no `CLAUDE_SPAWN_SESSION_ID`. It keys on the
Claude Code `session_id` from the hook's stdin JSON, through
`session_registry.build_session_slug`, with the `session_id` in the
`launcher_pid` slot as the uniqueness token. That `session_id` is the same
for SessionStart, Notification and Stop within one session, so all three
resolve to the same `record.json`. Both kinds keep their record across a
re-mint (see §Session-id re-mints).

A session with no `CLAUDE_SPAWN_SESSION_ID` first tries two registry
pointers, both siblings of `sessions/`: `sessions-by-claude-id/<session_id>`,
then `sessions-by-pid/<owning claude pid>`. A pointer counts only while the
record it names vouches for it, by an equal `claude_session_id` or
`claude_owner_pid`. The slug a pointer names then goes through the same
ownership decision as an env slug. So an event whose `session_id` mismatches
the record falls through to the hand-launched keying, unless it is a re-mint
reached through the pid pointer. Every bind, and every dead-owner reclaim
(see §Session-id re-mints), writes the session-id pointer, and every write
of a record whose `claude_owner_pid` is the event's claude writes the pid
pointer, so spawned, T3-hosted and hand-launched records carry
the same pointers. This is the only way a T3-hosted session reaches its spawn record,
because it has no `CLAUDE_SPAWN_*` environment at all: the T3 lane binds the
record to the session id of the transcript that names the record dir, and
writes that session-id pointer itself (task 7489).

Code: `orchestrator/src/orchestrator/session_hooks.py::hook_session_slug`,
`::_resolve_hook_slug`, `::_pointer_resolution`, `::_stamp_session_pointers`;
`orchestrator/src/orchestrator/session_registry.py::resolve_session_slug_for_claude_session_id`,
`::resolve_session_slug_for_pid`.

## Owner or inheritor

`CLAUDE_SPAWN_SESSION_ID` is inherited by everything the spawned session
starts. A nested `claude` (an agent's Bash-tool `claude -p`, a script that
shells out) carries the variable too, and adopting it would let the nested
session flip the spawner's status mid-turn. An ancestry test cannot tell the
two apart, because a nested `claude` is also a descendant of the launcher.
The hooks therefore use these discriminators, in order. This was introduced
by task 4193.

1. **A bound `record.claude_session_id`.** The first event that adopts a slug
   and proves its ownership stamps its stdin `session_id` onto the record.
   The binding is set once and never overwritten, so it stays a stable
   discriminator. A later event with a different `session_id` is an
   inheritor. It falls through to the hand-launched keying and gets its own
   record.
2. **Before binding, `CLAUDE_SPAWN_OWNER_PPID` parent equality.**
   `spawn-claude.sh` exports, from inside the payload shell, the pid of the
   bash that is about to run `claude`. That `claude` runs without `exec`, so
   the owning claude's direct parent is always exactly that pid, on every
   backend. A nested `claude` inherits the value but has a different parent.
   This test needs no persisted state, so it works from the very first event.
   It is keyed on whether the record is unbound, not on its status, so it
   also covers sibling-mode records, which `resolve_sibling` has already
   flipped to `running` by the time a hook runs.
3. **For a SessionStart re-mint, `claude_owner_pid` equality.** The owning
   pid stamped at bind time must equal the pid of this event's owning claude
   (see §Session-id re-mints).

Lineage against `CLAUDE_SPAWN_LAUNCHER_PID` cannot replace any of these. The
variable is never exported into the spawned session, since `sanitize_env`
strips the namespace and only a few values are re-exported. Under a detached
terminal emulator the launcher is not even an ancestor of the owning claude,
because gnome-terminal reparents the payload to gnome-terminal-server.

Code: `orchestrator/src/orchestrator/session_hooks.py::_env_slug_ownership`,
`::_bind_claude_session_id`,
`orchestrator/src/orchestrator/session_ownership_probes.py::EventProbes.owner_ppid_verdict`.

## Session-id re-mints

Claude Code re-mints the stdin `session_id` in place on `/clear` and on
compaction. Without special handling, the owner's next SessionStart would
look like an inheritor and fork the session onto a second record. A
SessionStart whose `source` is `clear` or `compact` is accepted as the owner
when both of these hold:

- The `source` is in the owner-only set. These sources have no command-line
  spelling, so the emitter already holds a session. This alone is not
  enough: it proves the emitter holds a session, not this one. Automatic
  compaction fires `compact` in any long-running nested `claude -p`.
- The event's owning claude pid equals the `claude_owner_pid` bound on the
  record. `/clear` and compaction run inside the owning process and keep its
  pid. A nested claude is a different process.

When both hold, SessionStart re-binds the record to the new id.

A session with no `CLAUDE_SPAWN_SESSION_ID` re-binds the same way through
its pid pointer: the new id misses `sessions-by-claude-id`, the owning
claude's pid finds the record, and the pid condition above holds. The
re-bind also writes a session-id pointer for the new id. A T3 thread that T3
resumes in a new process keeps its session id.

When no live process holds the recorded `claude_owner_pid`, the resumed
process's event re-stamps it with the new claude. This dead-owner reclaim
applies on two lanes:

- an event reached through its session-id pointer;
- an event reached through the hand-launched keying, whose slug embeds that
  same session id, so a record bound to it is proven the session's own. A
  forked inheritor's record is keyed the same way (see §Forked-inheritor
  identity), so it shares this lane.

An event that adopts its `CLAUDE_SPAWN_SESSION_ID` does not reclaim: there a
reclaim would also satisfy the self-exit owner-pid precondition in the same
event. The reclaim writes both pointers. A later re-mint in the new process
therefore re-binds, and a record bound before the session-id pointer existed
(task 7489) gains one at its next event. This is how a thread opened directly
in T3's UI keeps its pid pointer across T3's resume in a new process: it has
no spawn record and is keyed by the hand-launched keying (task 7614). A live
recorded owner is left in place, because then the event is a nested
`claude --resume` of the same session. A thread whose resume fails starts a
fresh session id in a new process, which no pointer links to the record; it
gets a hand-launched record.

The re-bind moves `claude_session_id` only. `claude_launch_session_id` keeps
the id the first binding stamped and is never re-bound, so the launch
transcript `<id>.jsonl` stays findable after a `/clear` or compaction
(task 6069). Records bound before that field existed carry none and are not
backfilled.

Code: `orchestrator/src/orchestrator/session_hooks.py::_bind_claude_session_id`,
`::_reclaim_dead_owner_pid`, `::_HookSlugResolution.session_id_keyed`,
`scripts/legibility/check_transcript_persistence.py::find_matching_transcript`.

`resume` is excluded. `--resume` and `--continue` make a brand-new process
report it, so a nested `claude -c -p` would present it. Honouring it would
let an inheritor re-bind its spawner's record to itself. A spawned session
that is `/resume`d in place forks onto its own record and recovers at its
next `/clear` or compaction, which is the safe direction. `compact` is kept
even though nested processes can emit it. Dropping it would fork the owner's
own record on every routine auto-compaction. The pid condition handles both
cases.

Only SessionStart may forgive a re-mint. Notification and Stop never carry
`source`, and honouring one there would give any inheritor a one-word bypass.
The owner does not need it on those events, because its SessionStart has
already re-bound the record.

Code: `orchestrator/src/orchestrator/session_hooks.py::_OWNER_ONLY_SESSION_START_SOURCES`,
`::_is_owner_only_session_start`, `::_env_slug_ownership`.

## Fail-soft direction

Every probe that cannot answer resolves to **adopt**, never to fork. This
covers no stdin `session_id`, an unreadable record, no `CLAUDE_SPAWN_OWNER_PPID`
(a hand-launched session, or one spawned by an older `spawn-claude.sh`), an
owning pid that cannot be resolved, and a platform with no `/proc` such as
macOS. Adopting falls back to "collapse onto one record", which does no
lasting harm. Forking where ownership is merely unproven would split the
owner's own record on routine events.

Adopting and binding are separate decisions, returned as two bits: `owned`
and `may_bind`. A binding is permanent, so it needs positive proof. An event
that adopts only because ownership could not be disproved refreshes the
record but does not bind it. The record stays open for its real owner.
Without this split, a nested `claude` that reached an unbound record first
would capture it and exile the owner on its next event. Hand-launched and
forked slugs embed the stdin `session_id`, so binding there is tautological
and `may_bind` is always true.

Code: `orchestrator/src/orchestrator/session_hooks.py::_env_slug_ownership`.

## The launch window

A record that is `LAUNCHING` and unbound has no discriminator yet. The owner's
SessionStart has not landed, and a nested inheritor could equally be the
sender. A Notification or Stop that arrives in this window, and that
`CLAUDE_SPAWN_OWNER_PPID` does not identify as the owner's, has unknown
provenance. It is withheld: only the mtime heartbeat is written. Its status,
its pending question and any binding are dropped. An event positively
identified as the owner's is never withheld.

Nothing else bounds this window. There is no `LAUNCHING` reaper, and the
stale-pid reaper needs a dead `launcher_pid`, which is `spawn-claude.sh`'s own
shell and lives as long as the session. A SessionStart that never lands, for
example one killed by the hook timeout, would leave the session invisible to
the attention rail for its whole life. `_LAUNCH_WINDOW_WITHHOLD_MAX_SECS`
(600s) bounds the withhold. It is well past the observed launch-window p90
and maximum, so a slow but healthy spawn is still withheld and only a stuck
one falls through. A record with no readable `start_ts` cannot be shown to be
inside the window, so it counts as expired.

Trade-off: if the owner's SessionStart never lands, its own later events are
also withheld until the bound expires. The alternative, forking every
pre-SessionStart event onto a new slug, would misroute the owner whenever its
SessionStart was merely slow.

Code: `orchestrator/src/orchestrator/session_hooks.py::_LAUNCH_WINDOW_WITHHOLD_MAX_SECS`,
`::_launch_window_withholding_expired`, `::_in_unbound_launch_window`,
`::_withhold_from_launching`, `::_run_status_refresh_and_retitle`.

## One read, one decision, one write

A record read is tri-state (`_RecordSnapshot`): present, absent, or
unreadable. Absent and unreadable must not collapse into one `None`, because
the two consumers of a read want opposite things from a fault. The ownership
probe adopts on both. A refresh may create a new record over an absent one,
but must not over an unreadable one: `session_registry.py::refresh_record`
raises `CorruptSessionRecord` rather than overwriting a corrupt body. That
body holds role, prompt and result_file, and it is the record
`spawn-claude.sh` writes `exited` to.

Each event reads the record it writes exactly once. On the adopt path, the
ownership probe's own read is passed forward on
`_HookSlugResolution.snapshot`, under the invariant
`snapshot is None or snapshot.slug == slug`. The snapshot travels only on the
adopt branch. On the fork branch the returned slug is a different record, and
passing the spawner's snapshot forward would apply a nested claude's decision
to the spawner's record. The withhold decision and the body written derive
from the same snapshot, so they cannot disagree. Status, question and binding
land in one `write_record`.

On a Stop owner event (`may_bind` true and not withheld) the self-exit
evaluation's `exit_request` stamp lands in that same write. A withheld or
unproven event stamps nothing. The only second write is the error path: a
failed signal stamps `kill_failed` after it. The preconditions are
`plans/review-all-program-tooling-prd.md` §4 H5's.

Every lane goes through one shared withhold, refresh, bind, question and
write block: adopt, fork, hand-launched, and the unreadable fault. The
unreadable lane differs only in where its snapshot comes from: it re-reads
once and then falls through. Do not give any lane its own copy of that block.
A lane that copies the write policy but not the decision policy writes
without evaluating the launch window, and inverts ownership on a nested
claude's event.

Registry writes stay last-writer-wins, as a deliberate non-goal.
`session_registry.py::write_record` is an atomic whole-body replace with no
compare-and-swap. Adding one would change the write contract for every
registry caller and belongs in its own change.

Code: `orchestrator/src/orchestrator/session_hooks.py::_RecordSnapshot`,
`::_read_record_snapshot`, `::_HookSlugResolution`, `::_resolve_hook_slug`,
`::run_session_start`, `::_run_status_refresh_and_retitle`;
`orchestrator/src/orchestrator/session_self_exit.py::SelfExit`. Introduced by
task 4662; the self-exit stamp by task 7193.

## Forked-inheritor identity

When an inherited `CLAUDE_SPAWN_SESSION_ID` is rejected, the event belongs to
a nested `claude`. Every other `CLAUDE_SPAWN_*` value in scope then describes
the spawner, not this session. On that path:

- `parent_session_id` is the rejected slug, which names exactly the spawner.
  `CLAUDE_SPAWN_PARENT_ID` would name the spawner's own parent and show the
  nested session as its sibling.
- `CLAUDE_SPAWN_WM_TITLE` is not resolved. It names the spawner's window, and
  claiming it would send cockpit focus to the parent's terminal. `TMUX` and
  `WINDOWID` still apply, because the terminal the nested process actually
  runs in sets them.
- The record's liveness pid is the nested claude (see §Process-tree probes).
- `_SPAWNER_IDENTITY_ENV_KEYS` are stripped from the env used for the record
  body and the OSC retitle. Role, project, task id, escalation id and title
  then fall through to this session's own defaults. This was introduced by
  task 4663.

The slug itself is still derived from the raw env. It is an opaque key: every
consumer either keys on it or prints it, and none parses an identity back out
of it. SessionStart, Notification and Stop each derive it independently, so
it must be built from inputs that are identical across all three events.
Otherwise one forked session would split across two records.

Code: `orchestrator/src/orchestrator/session_hooks.py::resolve_hook_identity`,
`::_SPAWNER_IDENTITY_ENV_KEYS`, `::_own_identity_env`, `::_resolve_hook_slug`,
`::_resolve_display`, `::run_session_start`.

## Process-tree probes

The owning claude is found by identity: the first ancestor whose comm is
`claude`, within `MAX_CLAUDE_ANCESTOR_HOPS` of the hook's parent. A fixed depth
does not work, because Claude Code runs hook commands through `sh -c`. The
grandparent is then the short-lived wrapper shell, and a depth-based owner
check would come out one level short and reject every real owner. The measured
tree is `orchestrator/tests/test_session_ownership_probes.py::_REAL_HOOK_CHAIN`.
The hop bound leaves room for an extra wrapper level without letting a
reparented hook walk to init. This is an identity resolution compared by
equality, not an ancestry test: a nested `claude`'s first claude ancestor is
itself.

Each hook event builds one `EventProbes` over one `ProcessTree` and passes it
through. The adopt/fork decision, the launch-window withhold and the
`claude_owner_pid` stamp all read the same walk. This bounds the `/proc` cost
against the hook timeout. It is also a correctness property: two decisions
made from one observation cannot disagree because the tree changed between
two walks. The memo lives for one event and is never shared across events,
because the process tree is live state.

A new record's `launcher_pid` is its liveness signal for the stale-pid
reaper:

- **Hand-launched record:** the POSIX session leader (`os.getsid(0)`), which
  is normally the interactive shell or `claude` that owns the terminal. The
  hook's own parent shell exits seconds after the hook, and an idle session
  would then be reaped while its terminal is still open.
- **Forked-inheritor record:** the nested claude itself, which dies with the
  nested session. The session leader outlives every nested claude in the
  terminal, so a forked record stamped with it would never be reaped.
- When no claude ancestor resolves, the forked record falls back to the
  session leader. This only makes the signal coarser. The reaper also needs
  an hour of heartbeat silence, so a fallback pid cannot reap a record that is
  still being written.

Code: `orchestrator/src/orchestrator/session_ownership_probes.py::ProcessTree`,
`::EventProbes`, `::_owning_claude_pid`.
