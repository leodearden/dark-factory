# One canonical build path and one shared compiler cache for reify

**Status:** active. Authored 2026-11-18 (`/team /prd`, Leo attending). Amended the same day: the cache
policy is one setting per host that covers the warm base and verifies (decision 9), and gate κ measures
three policies. Amended 2026-11-19 after gate α passed (Leo's rulings on its report,
`plans/canonical-build-path-alpha-2026-11-19/REPORT.md`): the primitive restarts a server whose start-up
check collided (C5), and the drift probe covers the system libraries reify's build scripts probe (C13).
**Code anchors** are cited by symbol and were checked against dark-factory main `0dfdd820f1` and reify
main `11a5990e46` (2026-11-18). Main moves fast; re-locate symbols at implementation time.
**Evidence** (all under `plans/sccache-path-key-study-2026-11-18/`): `REPORT.md` (the study, a laptop
trial, two addenda); `prd-scout/S1`–`S10` (ten scouting reports); `prd-scout/CRITIQUE-prd.md` and
`prd-scout/FOLD-VERIFY-prd.md` (two adversarial reviews of earlier drafts of this PRD, both incorporated).
Reify's short statement of the mechanism is `docs/notes/sccache-key-path-sensitivity.md` in reify.

## 1. Goal

Every reify build, wherever it runs, sees its worktree at one absolute path and reads and writes one
compiler cache. A unit compiled once, by any verify, task agent or interactive session on either host, is
a cache hit for every later build of the same source.

What an operator observes when this lands:

- A reify task verify on `leo-laptop`, in a worktree that was never built before, compiles nothing the
  workstation has already compiled: its sccache counters show hits where today they show misses, and the
  workstation's build-cache observation log has a row for it with backend `shared`.
- Under the `shared` policy, `cargo build` typed in a fresh interactive worktree of reify's current main
  finishes without compiling the workspace's libraries, because a merge verify already built them.
- When the shared store cannot be reached, builds still pass using a local cache, and one escalation
  appears and clears itself when the store returns.

## 2. Background and evidence

sccache 0.14.0 hashes the compile's working directory, `CARGO_MANIFEST_DIR`/`CARGO_MANIFEST_PATH`, the value
of every variable a crate reads with `env!` (so `OUT_DIR`), every inherited `CARGO_*` variable, and the
bytes of every dependency artefact into each Rust key. A workspace library therefore misses at any path
it was not built at, and so does about a third of registry libraries. No sccache release keys Rust
compiles independently of the path, and remapping flags make it worse (study, Q1–Q4).

| Measurement | Result | Source |
|---|---|---|
| Task-verify build steps at a new path, today | 728 hits / 422 misses, 90.8 CPU-minutes | study, stage 1 |
| The same from a never-built worktree bound at one fixed path in a mount namespace, the whole verify and its sccache server inside | 1,150 hits / 0 misses, 49.4 CPU-minutes | addendum 1, run N2 |
| Tests that read files through compile-time-baked paths, run inside that namespace | 2,121 of 2,121 passed | addendum 1 |
| One of those test binaries run where the fixed path is not bound | fails with a missing-file error | addendum 1 |
| Two sccache servers over one local disk cache directory | a running server never sees the other's new entries | addendum 2 |
| Two servers over one WebDAV endpoint (`rclone serve webdav` over plain files) | entries visible at once | addendum 2 |
| The laptop reading entries the workstation wrote (toy workspace, same path string) | 11 of 11 hits, Rust and C | addendum 2 |
| Toolchain, C compiler binaries, system packages, prebuilt static libraries on the two hosts | identical today | addendum 2; S8 |
| Store unreachable when a server starts | the server refuses to start; every compile would fail | S4, S5 |
| Store refusing connections while a server runs | compiles pass uncached | S5 |
| Store frozen while a server runs | each lookup waits up to 60 s; writes have no timeout | S5 |
| A second TCP-addressed server in another namespace | its compile was served by the first namespace's server and wrote into the first worktree | S4, S10 |
| Any sccache client that finds no listener | starts a detached server from its own environment; no switch disables this | critique B4 |
| A build inside a landlock sandbox trying to enter a namespace | fails; landlock forbids mounts | S9 |
| A landlock grant on a worktree, written through a second mount of the same directory | allowed; rules attach to inodes | S9 |
| Plain `cargo test` at the real path after a build at the fixed path | no recompile; tests fail on the baked path | S10 |
| Agents' builds today | incremental, no sccache; cargo puts the incremental setting in each unit's identity, so they reuse nothing from a base that verifies built | S9; fold-verify M7 |
| From the host, a namespaced process's working directory | reads as the fixed path; every path-based liveness check misses it | S3 |
| From the host, `stat /proc/<pid>/root/<fixed path>` | gives the bound worktree's device and inode, on both hosts | S3 |
| `rclone serve webdav` under 1 to 64 concurrent clients, realistic object sizes | no errors on a prepared store; GET p50 8–166 ms | S5 |

Three facts shape the design:

1. **Cargo does not notice a change of path.** Its fingerprints are path-independent, while build-script
   outputs, dep-info files and test binaries record absolute paths (S2 F2, S7, S10). A build directory
   should therefore be built at one path for its whole life.
2. **The sccache server reads sources and writes outputs itself**, so it must run where the fixed path is
   bound to the right worktree, and no other worktree's build may reach it (S4, S10).
3. **What has run on real reify is the trial's shape**: the whole verify inside one namespace. The shape
   below differs in how the server is addressed, started and stored. Task α measures those differences on
   real reify before anything is built on them.

## 3. Rulings (Leo, 2026-11-18)

1. Pursue the canonical path. Task agents and interactive sessions must get warm builds too, not only
   verifies.
2. One compiler cache shared by both hosts is the goal. One per host is acceptable if sharing fails.
3. The store is the simplest thing that works, unless it has a relevant performance or reliability cost.
4. When the store is unreachable, a build falls back to a host-local cache and escalates above a low
   threshold. A fallback must not quietly become the normal state.
5. Measuring real-reify hits across hosts is the first gated leaf.

Standing rule that binds every decision below: a cache must never change a verify verdict.

## 4. Resolved design decisions

1. **The canonical path is `/opt/reify-canon/ws` on every host** ("CANON"). `/opt/reify-canon` is created
   once per host by root, with an empty `ws`. Root ownership means nothing can populate it and `bwrap`
   cannot create it by accident (S9). It is outside `/tmp` because scripts delete `/tmp/reify-*` globs, and
   inside a namespace such a delete would empty the bound worktree. The literal's home is reify's
   primitive (decision 3). Dark-factory holds it once, in configuration, only for the session binding of
   decision 7; the in-service check compares the two (C13).
2. **A build directory built at CANON says so.** It carries the stamp file `.reify-build-root` containing
   CANON, written before its first build and never changed. A build directory with no stamp is a real-path
   directory, exactly as today. A directory is never converted in place: changing its root means deleting
   it. This lets the scheme be switched on per host and per tree with no flag day.
3. **Reify ships one primitive, `scripts/canon_run.py`** (Python, standard library only). It decides the
   mode of the build directory a command will use (C1), and either runs the command untouched (real-path
   mode: exactly today's behaviour) or runs it with the worktree at CANON (C3) under the host's cache
   policy, with a private cache server where the policy uses one (C4, C5). It is the only place that knows the store, the server's lifecycle and
   the cache environment.
4. **Every launch site that reify or dark-factory owns calls the primitive explicitly; `PATH` is only for
   people and agents typing `cargo`.** A `PATH` shim alone is defeated by `~/.cargo/env` re-ordering `PATH`
   and by systemd units with their own `PATH` (critique B2). So dark-factory prefixes each verify leg and
   offline-lane run with the primitive (C8); reify's git hooks and warm-base scripts call it themselves
   (C12); a `cargo` trampoline on `PATH` covers ad-hoc cargo in agent and interactive sessions (C7).
5. **What a bypass can and cannot do.** A cargo that reaches a CANON-stamped directory without the
   primitive builds at the real path. Both roots name the same tree, so it cannot read another tree's
   files or produce a wrong result; it fails on a missing CANON path (a baked `OUT_DIR`, a test fixture)
   or rebuilds. That is loud or wasteful, never silently wrong, which is why no enforcement wrapper is
   put in front of every compile: the one cargo offers, a configured rustc wrapper, is overridden by the
   `RUSTC_WRAPPER` variable that `verify.sh` always sets, and it never sees build scripts (fold-verify B2).
6. **The whole verify leg runs at CANON**, as in the trial, not each cargo call separately. Dark-factory
   prefixes the leg's command; `verify.sh` and everything it starts then run with the worktree at CANON,
   so no line of `verify.sh` changes (it is mid-port to Python, reify task 9112) and no cargo call inside
   it can bypass. The primitive exports the real root for the two things that need it: the verify
   ledger's lane (`REIFY_VERIFY_LEDGER_LANE_DIR`) and the grammar cache (`REIFY_TS_CACHE_HOME`, which is
   otherwise keyed on the path and would collide at CANON; S2 F3, F6).
7. **Two ways to have CANON bound.**
   - *Per command*: the primitive enters a mount namespace with `/usr/bin/bwrap` for the life of the
     command. Used by verify legs, hooks, unsandboxed agent roles and interactive sessions.
   - *Per session*: a sandboxed agent cannot create mounts, so dark-factory binds the lane at CANON before
     landlock is applied. Dark-factory composes that prefix itself from its own configuration and executes
     nothing from the lane before the sandbox exists (critique M4). The session's working directory stays
     the real lane path, so file tools, git and transcripts are unchanged, and the lane is visible at both
     paths. The primitive detects the binding and only changes directory.
8. **Fresh build directories go to CANON only where CANON can be provided.** A build directory that cargo
   has not built in yet becomes a CANON directory if the host is enabled and either CANON is already bound
   to this tree or a namespace can be entered; otherwise it becomes a real-path directory. A sandboxed
   agent that builds some second tree (a test's scratch workspace) therefore gets a real-path build there,
   not a failure. Only a directory already stamped CANON, where CANON cannot be provided, is refused.
9. **Cache policy is one setting per host, and it covers every build there: verifies, the warm base,
   agents and interactive sessions.** Cargo puts the incremental setting into each unit's identity, so a
   build reuses another's work only if both used the same setting; a policy chosen per caller would leave
   agents reusing nothing from the base, which is today's state (seeded-lane study). The values:
   - `shared` (default): sccache with the shared store for every crate, incremental compilation off. A
     build reuses anything already built anywhere. An edited crate and all its dependents compile in full.
   - `incremental`: no sccache; workspace crates compile incrementally. Fastest rebuild after a small
     edit in a build directory that already built nearby source (reify measured a touched crate with 23
     dependents at 16 s against 104 s); nothing to reuse in a directory that has not.
   - `mixed`: workspace crates compile incrementally and are not cached; registry crates, which cargo
     never compiles incrementally, use sccache and the shared store.
   The primitive applies the policy at every cargo invocation, inside the command, so it holds even where
   a script in between exports its own settings: `verify.sh` always exports `RUSTC_WRAPPER=sccache` and
   `CARGO_INCREMENTAL=0`. Under a policy other than `shared` the verify plan's printed environment
   therefore differs from what cargo gets; the event and the journal record the policy in force (C6).
   Which policy is cheapest for a host is measured, not argued: gate κ runs all three (§9).
10. **Cargo subcommands that neither build nor run artefacts are passed through; everything else goes to
    the build directory's mode.** The trampoline passes through `metadata`, `locate-project`, `pkgid`,
    `tree`, `search`, `login`, `logout`, `new`, `init`, `add`, `remove`, `update`, `fetch`, `vendor`,
    `generate-lockfile`, `help`, `version`, and `install` when it has neither `--path` nor a target
    directory inside the tree. Every other subcommand, including aliases and plugins it does not know,
    goes through the primitive. The list is of what is safe to leave alone, so an unknown subcommand fails
    safe (critique B3).
11. **A private sccache server per outermost primitive invocation**, started in the foreground as a child
    before the command, addressed by a unix socket in the build directory, with no idle timeout, stopped
    and its socket removed when the command ends. Never a TCP port (fact 2). An invocation is *nested*
    when the environment says a primitive is active for a tree with this tree's device and inode and CANON
    resolves to it; a nested invocation reuses its parent's server. If the foreground server dies,
    sccache's client starts a detached one from the command's environment; that environment carries only
    the local fallback cache and an 1,800 s idle timeout, so the stray server is harmless and short-lived,
    and the event counts it.
12. **No PID namespace.** It would contain a stray server cleanly, but tests that build temporary names
    from their process id would collide across builds that share `/tmp`, and scripts that check a holder's
    liveness through `/proc/<pid>` would misjudge it (S2 F1).
13. **The shared store is `rclone serve webdav` over a plain directory on the workstation**, bound to
    loopback and the workstation's tailscale address, with basic authentication (ruling 3: rclone is
    installed, the files are inspectable, and the load test found no performance cost). A pruning job
    replaces the eviction sccache does for a local cache.
14. **The store is probed before the server starts, and cut off, not swapped, if it is lost.** The probe
    chooses the backend, because sccache has no fallback of its own. The server reaches the store through
    a loopback relay owned by the primitive. While the command runs the primitive probes the store
    directly; after two consecutive failed or slow probes it closes the relay and resets every connection
    through it. Every later cache request then fails at once, and sccache treats a refused store as misses
    and compiles normally (measured, S5). The command is never aborted and the server never replaced, so
    the client's auto-start cannot fire. A store that answers probes quickly but serves large objects
    slowly is not detected; it costs time, and gate ι's watch covers it.
15. **Compiles become part of the verify they belong to.** Today rustc runs in the host sccache unit's
    cgroup, outside the verify's CPU weight and memory cap, under one host-wide job pool (S4). A private
    server puts compiles inside the verify's scope and gives each server its own pool, with cargo's
    jobserver still the effective limit. That is intended; gate ι watches the laptop's 48 GiB verify cap
    for a compile killed by it.
16. **Process-liveness checks translate paths through mount namespaces.** For a same-user process in
    another mount namespace, the check finds whether the lane is mounted in that namespace (by device and
    inode of its mount points, read through `/proc/<pid>/root`) and, if so, translates the process's
    working directory, open files and mappings from that mount point back to the lane before the existing
    prefix match. One namespace is resolved once per scan. A namespace whose root cannot be read is
    skipped, as unreadable processes are today: a build's own namespace is always readable by its user,
    and a fail-closed rule would count browser sandboxes as lane references and signal them (critique B5).
17. **Facts travel on the existing per-leg event channel**, `REIFY_VERIFY_EVENT_LOG`. The primitive appends
    one `build_cache` event per outermost invocation. Dark-factory folds a leg's events into one
    `build_cache` field on `VerifyResult`. The verify ledger's own sccache figures stay correct, because
    `verify.sh` runs with the private server's socket in its environment.
18. **One streak alarm for "build cache degraded on host H"**, with the ladder the task-verify-lanes PRD
    adopted (L1 at 2 consecutive, again every 2, L2 at 10, a healthy result clears). The ladder is
    extracted into one module that both alarms use. A host whose configuration says it is converted
    reports a real-path verify as degraded, so an enabled host that silently stops using the scheme is
    caught (critique M15). Builds by agents and interactive sessions emit no event; a store that is down
    for them is down for the host's verifies too.
19. **Observations have one persisted home**: `data/verify-logs/build-cache-observations.jsonl` in the
    project, one row per verify result that carried a `build_cache` fact, written where the alarm observes
    it. The in-service check reads it. The existing `verify_runner.py::SccacheStats` and
    `config.py::SccacheConfig` describe the host-wide server and are left as they are.
20. **Machine-dependent compiler flags bypass the cache.** The hosts' CPUs differ; `-march=native`,
    `-mtune=native`, `-mcpu=native` or `-C target-cpu=native` in a compile's arguments (where flags from
    `RUSTFLAGS` and `CFLAGS` also end up) gives one key and different code. None is in use (S8). Where the
    policy uses sccache the primitive puts a directory holding one file, a guard named `sccache`, first on `PATH`, so
    that `RUSTC_WRAPPER=sccache` and the `cc` crate (which routes C compiles only through a wrapper with
    that file stem) both reach it. The guard runs a flagged compile directly, uncached, counts it, and
    otherwise calls the real sccache by absolute path. Other compiler-steering options (`-fplugin`,
    `-specs=`, profile-guided flags) are not in use and not guarded; trigger to add them: the drift probe
    or a dependency review finds one.
21. **Key hygiene is the primitive's job**: it sets the wrapper and the incremental setting the policy
    calls for, one pinned `CARGO_HOME` and `PWD`, and removes presentation-only `CARGO_*` variables. `PWD` matters
    because gcc records it in debug information when it names the working directory, which would split C
    keys per tree (critique M8).
22. **Rollout is per host and per tree, by operator gates** (ι, κ), and reversible in this order: set the
    host to disabled, so new build directories are real-path; rebuild the warm base; only then unset
    dark-factory's `build_canon.root`. Existing CANON directories keep working through the primitive
    throughout. As a backstop dark-factory gives a worktree a fresh build directory whenever it is about
    to run a leg unprefixed in a CANON-stamped one (C8).
23. **Cold-control verifies are not special-cased.** Runs that start from an empty build directory (shadow
    compare, the merge-lane safety valve, the cold-control sweep) will get cache hits they miss today.
    They already use sccache; what they verify does not change, only what they cost.

Superseded positions: the 2026-09 verify-speed ruling rejected a shared Redis backend because Redis holds
the cache in RAM, and named an object store as the right backend if sharing was ever pursued; reify's
verify-ledger ruling against a second sccache server dates from when nothing needed one.

## 5. Contract

### C1. Mode of a build directory

"Built in" means the directory contains a `.fingerprint` directory at any depth of two or less below a
profile directory, which only a cargo build creates. Other content does not count: reify's tree-sitter
epoch file is written before any cargo call, and `cargo metadata` writes `.rustc_info.json` without
building (fold-verify B1).

| Stamp `.reify-build-root` | Built in | Mode |
|---|---|---|
| contains CANON | either | CANON |
| absent | yes | real path |
| absent | no | CANON if the host is enabled and CANON is bound to this tree or a namespace can be entered (decision 8); the primitive then writes the stamp. Otherwise real path. |
| any other content | either | refuse, exit 126 |

A build directory outside the worktree (`CARGO_TARGET_DIR` elsewhere) is always real-path mode.

### C2. Primitive interface (`scripts/canon_run.py`)

| Invocation | Behaviour |
|---|---|
| `canon_run.py exec -- <cmd…>` | Decide the mode. Real path: exec `<cmd>` with the environment untouched. CANON: C3–C6, then exit with the command's status. |
| `canon_run.py cargo <args…>` | The same for cargo, after the pass-through list of decision 10. |
| `canon_run.py status` | One JSON object: CANON, whether it is provisioned, host enabled, store reachable, this build directory's mode, the host's policy. |

Exit 126 means "this command cannot run in the mode its build directory requires"; stderr names the fix.
`REIFY_CANON_ROOT` overrides CANON, and `REIFY_CANON_HOST_CONFIG` the configuration file, for tests only.

### C3. Entering CANON

- Already bound: CANON is a real directory (not a symlink), a mount point in `/proc/self/mountinfo`, and
  its device and inode equal the worktree root's. Then change directory to
  `CANON/<path relative to the worktree root>`.
- Otherwise: `/usr/bin/bwrap --dev-bind / / --bind <worktree root> CANON --chdir CANON/<relative>
  --die-with-parent --info-fd <fd>`, with the outer process's working directory left at the real path.
  The outer process does not exec bwrap. It stays resident as bwrap's parent, in the caller's process
  group, and exits with bwrap's status (the command's status, or 128+N for signal N).
- Signals at the namespace boundary (ruled by Leo 2026-11-23, esc-10418-3, reify task 10418). bwrap is
  started with TERM, INT and HUP ignored, except a signal the caller already ignored, which stays ignored.
  Inside, the command runs under `/usr/bin/env --default-signal=<those signals> --`, so it gets the
  caller's dispositions back. The outer process forwards TERM, INT and HUP to the `child-pid` bwrap writes
  to the info fd, through a pidfd. A signal that arrives before the pid is known is delivered once it is.
  The rule: entering the namespace is invisible to whoever kills the command. On every kill path (TERM or
  INT to the started pid, TERM or INT to its process group, TERM to each descendant, SIGKILL to the pid
  or the group), the command's own trap runs, or nothing survives, exactly as in real-path mode. A plain
  exec of bwrap fails this: bwrap dies of the signal and its parent-death signal SIGKILLs the command
  mid-trap (measured, bwrap 0.9.0). There is one signal policy, in canon_cache.py's supervisor. The
  namespace parent reuses its forwarder, and the supervisor never leaves the command ignoring a signal
  the caller did not ignore.
- `PWD` is set to the new working directory. Arguments and `CARGO_*` values that are paths under the
  worktree's real root are rewritten to CANON (`--manifest-path`, `--target-dir`, `CARGO_TARGET_DIR`).
- Exported for the command: `REIFY_CANON_ACTIVE` (the tree's device and inode), and
  `REIFY_VERIFY_LEDGER_LANE_DIR` and `REIFY_TS_CACHE_HOME` derived from the real root.

### C4. Cache environment, by policy

The command's `PATH` starts with two directories of the tree's `scripts/canon-bin/`: `guard/` (holding
only the guard, named `sccache`) and the trampoline's, then `$HOME/.cargo/bin`. Every cargo started inside
the command therefore passes through the trampoline, which calls the primitive as a nested invocation;
that invocation sets the variables below immediately before the real cargo runs, replacing whatever was
inherited.

| Policy | Server | Set for cargo |
|---|---|---|
| `shared` | yes | `RUSTC_WRAPPER=sccache`, `CARGO_INCREMENTAL=0` |
| `mixed` | yes | `RUSTC_WRAPPER=sccache`, `CARGO_INCREMENTAL=1` (sccache declines an incremental compile, so the client runs it uncached) |
| `incremental` | no | `RUSTC_WRAPPER` and the sccache variables removed, `CARGO_INCREMENTAL=1` |

In every policy: one pinned `CARGO_HOME`, and presentation-only `CARGO_*` variables removed.

Where there is a server:

- Socket: `SCCACHE_SERVER_UDS=<CANON>/target/.reify-sccache.<pid>.sock`; `SCCACHE_SERVER_PORT` removed.
- The outermost invocation starts it in the foreground as its child, with file descriptors above 2 closed
  and `SCCACHE_IDLE_TIMEOUT=0`, under a host start lock (bounded wait; on timeout it starts anyway). Its
  environment carries the backend of C5. The primitive waits until the socket answers before running the
  command, reads the server's counters before stopping it, stops it and removes the socket at exit.
- The command's environment carries the socket, the local fallback directory and
  `SCCACHE_IDLE_TIMEOUT=1800`, and no store variables.

### C5. Store, relay and fallback

- Probe: one GET with a 2 s timeout before the server starts. Success: backend `shared`, with
  `SCCACHE_WEBDAV_ENDPOINT` pointing at a loopback relay the primitive listens on, which forwards to the
  store. Failure, or no store configured: backend `local_fallback`, `SCCACHE_DIR` = the fallback directory.
- Watchdog: while the command runs, probe the store directly on an interval. Two consecutive probes that
  fail or exceed the timeout close the relay and reset its open connections for the rest of the command.
- Start-up check collisions. sccache's start-up check reads and then writes one file,
  `<prefix>/.sccache_check`, and servers starting together collide on it (gate α: 0 to 14 of 20 servers
  read-only per burst, 10 of 20 in a cross-host burst; S5). A concurrent write gets HTTP 423 and the
  server runs read-only for its life, printing `storage write check failed` on stderr; a read racing a
  write gets a truncated body and the server exits (`cache storage failed to read`), after the probe has
  already passed. So, after starting the server, the primitive checks it: if the server exits before its
  socket answers, or its stderr carries `storage write check failed`, it stops it, waits a random 0.5 to
  2 s and starts it again, up to three attempts in all; after the third the backend is `local_fallback`
  with reason `store_unreachable` (the server never started) or `read_only_server`. The command starts
  only once a server is read-write or the fallback is chosen. The start lock removes most collisions
  within a host; the retry covers the rest and those across hosts.
- A server still read-only at exit (the retry is a start-time check) shows every miss as a write error
  and no writes; it is reported with reason `read_only_server`. The counters alone do not identify it,
  because same-key races also produce write errors (S5); the stderr line does.

### C6. `build_cache` event and the invocation journal

The event uses the channel's existing record envelope (reify
`docs/prds/v0_6/run-all-runner-infra-harness.md` §4 C2). Unknown event names are already ignored by
`verify_event_channel.py::_channel_from`. One event per outermost invocation on an enabled host, appended
to `REIFY_VERIFY_EVENT_LOG` when that variable is set:

| Field | Values |
|---|---|
| `mode` | `canon`, `real_path` |
| `policy` | `shared`, `mixed`, `incremental` |
| `backend` | `shared`, `local_fallback`, `none` (real-path mode, or `incremental` policy) |
| `reason` | `''`, `store_unreachable`, `store_lost`, `store_not_configured`, `read_only_server`, `legacy_build_dir`, `canon_unavailable` |
| `hits`, `misses`, `write_errors`, `uncached_machine_flags`, `detached_server` | integers |
| `wall_secs`, `cpu_secs` | the command's wall time, and the processor time of the command and the server with all their children |

Write errors are reported but are not a degradation signal: two builds writing one key at once is normal
under a canonical path and counts as a write error (S5).

The same record, with the cargo subcommand, is appended to `<build dir>/.reify-canon-journal.jsonl` for
every outermost invocation, whoever the caller. It is the record of agent and interactive builds, which
have no event channel, and the data gate κ compares policies with. The seed removes it from a copied
build directory.

### C7. Trampoline and guard (reify `scripts/canon-bin/`)

- `cargo` is the trampoline. Installed copies live in `~/.local/share/reify-canon/bin` on each host. It
  finds the worktree root by walking up to a directory holding both `Cargo.lock` and
  `scripts/canon_run.py`, applies decision 10, and otherwise execs the next `cargo` on `PATH`.
- `guard/sccache` is the guard of decision 20. Nothing else lives in its directory.

### C8. Dark-factory wiring

A new configuration section, hot-reloadable (in `config.py::RELOADABLE_FIELDS`), read from each host's own
yaml and never carried in `verify_env`, which the workstation copies to the laptop (S1):

| Key | Default | Meaning |
|---|---|---|
| `build_canon.root` | `''` | CANON. Empty: no leg prefix and no session binding. |
| `build_canon.shim_dir` | `''` | The trampoline's directory, prepended to `PATH` for agent launches, with `$HOME/.cargo/bin` kept after it so `~/.cargo/env` does not reorder them. |
| `build_canon.expect_canon` | `false` | The host is converted: a real-path verify is a degradation (C10). |

- **Verify legs.** The leg is a shell string run by `/bin/bash -c` inside `verify.py::run_verification`
  (its nested `_run_leg_command`). When `build_canon.root` is set and the worktree holds
  `scripts/canon_run.py`, the process is started as `<python> <worktree>/scripts/canon_run.py exec --
  /bin/bash -c <string>`, composed where the nice prefix is applied, after the command rewriters. The same
  for `offline_lane_runs.py::launch`. Otherwise unchanged.
- **Unprefixed leg in a stamped directory.** If a leg is about to run unprefixed (root unset, or the
  worktree has no primitive) and one of the project's `git.reap_build_artifact_dirs` carries the stamp
  file, that directory is removed first and the fact is logged.
- **Sandboxed agent launches** (`agents/sandbox_dispatch.py::wrap_command`, reached from
  `agents/invoke.py::_sandbox_wrap`): when `build_canon.root` is set, the command is prefixed, before the
  landlock helper, with `/usr/bin/bwrap --dev-bind / / --bind <lane> <root>`. The session's working
  directory and write-set are unchanged. A bwrap that fails is a failed launch. There is no
  `--die-with-parent` (ruled by Leo 2026-11-23, esc-10418-3). Agent sessions are always stopped by their
  process group (`shared/cli_invoke.py`: TERM to the group, a grace period, then SIGKILL), and bwrap stays
  in that group. So the session gets TERM directly, as an unwrapped landlock launch does today. Adding the
  flag would SIGKILL it as soon as bwrap died of that same TERM.
- **All agent launches** (`workflow.py::TaskWorkflow._build_agent_env`): `PATH` as above.
- `verify.py::run_main_tip_sweep`'s vanished-worktree detection also recognises `build_canon.root` in error
  text, since compiler errors name CANON for a leg that ran there.

### C9. `build_cache` on the verify result

`verify.py::VerifyResult` gains one optional field `build_cache`: `None`, or a dict with `mode` and
`backend` (the worst of the run's events: `real_path` before `canon`, `local_fallback` before `shared`),
`policy`, `reason` (the first non-empty) and the summed counters and times. `verify.py::_aggregate_results` folds the legs.
`verify_runner.py::result_from_dict` accepts the field; `remote_gate.py::wire_fingerprint` is computed from
field shapes and follows by itself.

### C10. Observation log and degraded-cache alarm

One observer, bound late in `harness.py::Harness._start_escalation_server` the way that method binds its
other sinks, is called where every task verify returns (`task_verify_dispatch.py::TaskVerifyDispatcher.__call__`,
local and remote) and where every merge verify returns (`verify_runner.py::VerifyRunnerPool.dispatch`). For
each result whose `build_cache` is not `None` it appends one row to
`data/verify-logs/build-cache-observations.jsonl` (time, executing host, role, task id or merge commit,
the dict) and feeds the alarm.

- **Degraded**: `backend == 'local_fallback'`; or `reason == 'read_only_server'`; or `mode == 'real_path'`
  on a host with `build_canon.expect_canon`.
- **Healthy**: `mode == 'canon'`, `reason == ''`, and `backend == 'shared'`, or `backend == 'none'` under
  `incremental` policy, where no cache is expected.
- **Neutral**: `build_cache` is `None`, or `real_path` on a host not yet expected to be converted.
  Neutral results neither extend nor reset a streak.

Per executing host, the ladder of decision 18 runs under the sentinel `__build_cache_degraded__<host>`,
category `infra_issue`, filed the way the lane alarm of task 10737 files its own.

### C11. Liveness

`shared/proc_group.py::scan_process_groups_under_path`, and `lib_live_refs.sh::live_ref_present` in reify's
copy and in dark-factory's (`orchestrator/scripts/warm-lane/lib_live_refs.sh`), apply decision 16 to
working directories, open files and mappings. `shared/proc_group.py::cpu_ticks_under_path` applies it to
working directories only, which is all that function reads. An open file moved out of the bound tree
reads as `/` from the host and stays unattributable, as it is today for any process. Each implementation
passes the scenario of boundary row 19.

### C12. Reify launch sites, warm base and seed

- `hooks/pre-commit` and `hooks/pre-merge-commit` run `verify.sh` through `canon_run.py exec`; so do
  `scripts/seed-warm-base-initial.sh` and `scripts/ensure-warm-base.sh` for their cargo. The warm base is
  therefore built under the host's policy, like everything seeded from it.
- `seed-warm-lane.sh`, when the copied build directory carries the CANON stamp:
  - if the lane's tree holds the primitive: no path relocation and no `env!` relink; leftover socket
    files and the invocation journal are removed;
  - if it does not (a branch from before this work): today's relocation and relink run with CANON as the
    root to rewrite from, and the stamp is removed, so the lane has a real-path directory as it does now.
  With no stamp it behaves as today. `refresh-warm-base.sh` is unchanged apart from ignoring socket files.
- Changing a host's policy changes every workspace unit's identity. The old units stay in the build
  directory unused, so a policy change is followed by dropping the merge lane's build directory, which
  rebuilds the base under the new policy.

### C13. Host configuration, store service and operator tools

Host configuration, read by the primitive: `~/.config/reify-canon/host.toml`, mode 0600, written by the
operator gates with the installer's help.

| Key | Meaning |
|---|---|
| `enabled` | bool. False or file absent: fresh build directories are real-path. |
| `store_url`, `store_user`, `store_password` | the WebDAV endpoint as this host reaches it. Absent: `store_not_configured`. |
| `fallback_dir`, `fallback_cap` | the local fallback cache: under the user's home, or under `/tmp` when the caller is a sandboxed agent, which can write nowhere else outside its lane. |
| `policy` | `shared` (default), `mixed` or `incremental` (decision 9). One value for every build on the host. |

Dark-factory `scripts/deploy/`:

- `install_build_cache_store.sh` (workstation; idempotent; `--check`): the store directory with its leaf
  directories pre-created (concurrent first writes into a new directory fail; S5), the rclone user unit
  (loopback and tailscale addresses only, basic auth, `Restart=always`), the pruning timer, and a host
  configuration file when asked for one.
- `build_cache_store_prune.py`: removes least-recently-used objects (by the later of access and
  modification time) down to a size cap, skipping files younger than ten minutes and never removing
  directories, then signals rclone to drop its directory cache.
- `build_cache_drift_probe.sh`: prints one digest line per cache-key input of a host (compiler binaries
  including `cc1`, `cc1plus` and `as`, toolchain, system package versions, prebuilt archives, tool
  versions). `cc1`, `cc1plus` and `as` are printed although sccache hashes only the compiler driver: a
  host whose driver matches but whose `cc1` differs would serve the other host's objects. It is the
  committed form of the study's `prd-scout/S8-drift-probe.sh`, plus one line per pkg-config module that
  reify's dependencies probe, with its version and `libdir`, or `ABSENT`. The module list is derived
  from reify's `Cargo.lock` each run: every registry crate's `[package.metadata.system-deps]` entries
  and `x11-dl`'s probe list (30 modules at reify `27c69f616f`). A target-scoped `cfg(...)` table of
  system-deps entries is descended one level, and every target's modules are listed without evaluating
  the cfg: a module of a non-Linux target prints `ABSENT` on both Linux hosts, so it never makes a
  difference. A build script that bakes a pkg-config answer into generated source makes its crate and
  every dependent miss on a host that answers differently; gate α's 18 cross-host misses were
  `libxss-dev` present on one host only (`xscrnsaver`), which the fixed package list did not see.
  `#info.cargo.registry-sources-missing` names the locked registry crates whose source is not on the
  host. It is informational and never compared: which sources a host holds depends on what it last
  fetched (a Linux build never downloads `windows-*` or `objc2-*` crates), not on a key input. A missing
  crate that declares system-deps still shows, as `pkg-config.<module>` lines absent on that host only.
  Whole-system package or module lists are not compared: the hosts legitimately differ in hundreds of
  unrelated ones.
- `check_build_cache_in_service.py --observations <file> --host <runner> --since <ISO> --min-runs N
  [--canon-root <path> --status-json <file>]`:

  | Exit | Meaning |
  |---|---|
  | 1 | the most recent observation for the host is degraded, or `--canon-root` differs from the CANON in the given `canon_run.py status` output |
  | 0 | at least N observations since `--since` are healthy and the most recent is healthy |
  | 2 | no observation for the host since `--since`: nothing is producing them |
  | 75 | fewer than N healthy observations yet |

  It prints the summed hits and misses.

## 6. Boundary tests

Reify rows 1–8 are unit-level and need no namespace, so a sandboxed implementer can run them. Rows 9–19
need `bwrap`, and some a private `rclone`; they live in members classified `intra-run-serial` in
`tests/infra/run-all-classification.manifest`, use `REIFY_CANON_ROOT` pointing at a scratch directory, and
skip with a counted, named reason when namespaces cannot be entered (an implementer's sandbox) or
`rclone` is absent (the laptop until gate ι), so the workstation's verify is what exercises them. Their
scratch repository has two Rust crates, one reading a fixture through `env!("CARGO_MANIFEST_DIR")`, and one
compiling a C file through `cc`. No row asserts a wall-clock bound.

| # | Side | Scenario | Asserts |
|---|---|---|---|
| 1 | reify | C1's table, each row; a directory holding only a tree-sitter epoch file and a `.rustc_info.json` | the mode named; that directory is treated as not built in |
| 2 | reify | pass-through: `metadata`, `install` with no path, `install --path .`, an alias `t`, an unknown plugin | first two untouched; the other three go to the primitive |
| 3 | reify | argument and environment rewriting | `--manifest-path <real>/Cargo.toml` and `CARGO_TARGET_DIR=<real>/target` become CANON paths; `PWD` is the new directory |
| 4 | reify | policy application: host policy `incremental`, the caller's environment exporting `RUSTC_WRAPPER=sccache` and `CARGO_INCREMENTAL=0` as `verify.sh` does | cargo runs with no wrapper and incremental compilation on; no server is started; the journal row says `incremental` |
| 5 | reify | guard: `-C target-cpu=native`; a C compile with `-march=native`; an ordinary compile | first two run the compiler directly and are counted; the third calls sccache |
| 6 | reify | probe fails before start (fake store refusing) | backend `local_fallback`, reason `store_unreachable` |
| 7 | reify | watchdog with an injected clock and a fake store that stops answering, one transfer in flight | relay closed and the transfer's connection reset after the second failed probe; reason `store_lost`; the command's exit status unchanged |
| 8 | reify | the event written by the primitive | read back through dark-factory's envelope rules, held in reify as a conformance sample of the channel contract |
| 9 | reify | host disabled, fresh build directory | real cargo, environment byte-identical to the caller's, no stamp, no event |
| 10 | reify | enabled; the workspace at two different paths, built in turn, `shared` policy | second build: every cacheable Rust and C unit is a hit; both stamps CANON; the fixture test passes in both |
| 11 | reify | a CANON-stamped directory, namespaces unavailable, CANON not bound | exit 126 with the reason; nothing compiled |
| 12 | reify | fresh directory, namespaces unavailable, CANON bound to a different tree | real-path build, no stamp, passes |
| 13 | reify | two concurrent builds in two trees | two servers, two sockets; each binary carries its own tree's marker |
| 14 | reify | nested: a test in tree A runs the primitive on scratch tree B | B gets its own namespace and server; A's server is not used for B |
| 15 | reify | the command killed by its process group; and a normal exit | no server or compiler process remains; after the normal exit no socket file |
| 16 | reify | twenty primitives started at once against one store | every command ran against a read-write server or the fallback; no server that started read-only or exited at start was used while attempts remained; any event saying `read_only_server` names the third attempt |
| 17 | reify | session binding: the bind prefix, then dark-factory's landlock helper, then `cargo test` through the trampoline, under `incremental` policy (β1) and under `shared` and `mixed` (β2) | passes at CANON without entering a namespace; the working directory outside cargo is the real path; under `mixed` the workspace crate is compiled incrementally and not cached while the registry crate is a cache hit on the second tree |
| 18 | reify | seed a lane from a CANON-stamped base with the primitive in the lane's tree; without it; and from an unstamped base | first: no file rewritten, no source touched, the lane's build is all fresh. Second: relocated from CANON, stamp gone, builds at the real path. Third: output identical to today's |
| 19 | each | a process in a mount namespace with (a) its working directory, (b) only an open file, inside a bound lane, its visible parent killed; plus an unrelated same-user process in a foreign namespace whose root cannot be read | scans report the lane referenced for (a) and (b); CPU ticks are counted for (a); the unrelated process is not reported and is never signalled |
| 20 | DF | `build_canon.root` unset | the started leg process, its environment and the agent command line identical to today's |
| 21 | DF | `build_canon.root` set; a worktree with and without the primitive; a stamped build directory in the one without; a sandboxed and an unsandboxed role | leg started through the primitive only where it exists; the stamped directory removed in the other; bind prefix before the landlock helper for the sandboxed role only; `PATH` and class set for both |
| 22 | both | the real primitive under a real verify leg, with a private store or none | the result's `build_cache` matches the primitive's events; a two-leg run sums the counters; one observation row is written |
| 23 | DF | results in order: 2 degraded; a third; a healthy one; with the L1 closed by hand, 2 more degraded; 10 consecutive; neutral results interleaved; a `real_path` result with and without `expect_canon`; a `read_only_server` result; a `backend none` result under `incremental` policy | L1 at the second; nothing at the third; cleared by the healthy one; refiled; L2 at the tenth with the L1 superseded; neutral changes nothing; `real_path` is degraded only with `expect_canon`; read-only is degraded; `none` under `incremental` is healthy |
| 24 | DF | a result with `build_cache` `None` | neutral; no observation row |
| 25 | DF | the lane alarm of the task-verify-lanes PRD after the ladder is extracted | that PRD's own boundary row 20 passes unchanged |
| 26 | reify | a command at CANON entered through the namespace (C3), whose TERM/INT trap takes 0.3 s; TERM and INT to the started pid and to its group, TERM to each descendant, SIGKILL to the pid and to the group | after a TERM or INT, the trap completes and its status is returned; after a SIGKILL, nothing the command started remains (reify task 10418) |

## 7. Pre-conditions

- `bwrap` on both hosts (present; unprivileged use measured on both) and `rclone` on the workstation
  (present, v1.75.0).
- `/opt/reify-canon/ws` provisioned by root on each host: a step of gate ι, not a task.
- Task α's measurements pass, or its resolution amends this PRD (ruling 5). Passed 2026-11-19: second-host
  misses 18 of 1,150 (1.57%), 2,121 of 2,121 tests, the agent session's tests passing
  (`plans/canonical-build-path-alpha-2026-11-19/REPORT.md`); C5 and C13 amended on its findings.
- Dark-factory task 10737 (the lane alarm whose ladder η extracts) and 10736 (which edits the same
  configuration and verify modules as ζ) land first.

## 8. Cross-PRD relationship

| Other work | Direction | Seam | Owner | Status |
|---|---|---|---|---|
| `plans/task-verify-lanes-prd.md` (dark-factory tasks 10736–10739) | composes | A lane's kept build directory holds CANON artefacts once the lane is reset after enablement. Its decision 8 (a host server's compiles outliving a cancel) stops applying to CANON runs, whose server is a child of the leg. Its lane-prepare process scan gains C11. Its alarm's ladder is extracted by η. | this PRD for the extraction and C11; that PRD for lanes | pending; ζ depends on 10736, η on 10737 |
| dark-factory tasks 10757 and reify 10002 (run_all INFRA verdict on the same event channel) | same parser and emitter files | `verify_event_channel.py`; reify `scripts/lib_event_log.sh` | each its own event | pending; rebase whichever lands second |
| dark-factory tasks 9686, 9688, 9716, 9722, 9856 (one escalation-filing entry point) | adjacent | η files as task 10737's alarm does; that program migrates both | those tasks | pending |
| dark-factory tasks 10459, 9128, 9130, 9150, 9134 | they add or move `VerifyResult` fields, calls and failure vocabulary | `VerifyResult` | each its own | pending; cite by symbol, rebase whichever lands second |
| dark-factory tasks 8338, 8339 | same functions | `proc_group.py::cpu_ticks_under_path`, `merge_lane/liveness.py::verify_tree_cpu_ticks` | those tasks for cost and start-time filtering; ε for namespace translation | pending; rebase whichever lands second |
| `plans/warm-lane-infra-repatriation-prd.md` (dark-factory tasks 3077, 7865, 3078) | reify's copies of the warm-lane scripts are the ones that run until the flip | `lib_live_refs.sh` in both repositories | δ and ε change both copies | pending |
| reify PRD `verify-structured-plan-and-role-table` (reify task 9112) | avoided | this PRD changes no line of `verify.sh` | — | pending |
| reify task 10169 (measure agent builds under the verify's settings at real paths) | superseded in part | it asks whether agents should build with the verify's settings; decision 9 makes the setting one per host, and gate κ measures the three policies at the canonical path | task 10169 for its own measurement; κ for the policy | pending; no dependency |
| reify tasks 7429, 9055 (a 14-day A/B of incremental without sccache against sccache on all lanes, through an arm variable `verify.sh` would read) | overlapping | this PRD's host policy is a second way to put a host's builds, verifies included, on either arm, and gate κ runs the comparison first; 7429's other parts (debug information, linker) are untouched | those tasks; their owner decides whether κ's result answers 9055 | pending, parked |
| reify tasks 10167, 10170, 9054 (seed relocation of dep-info paths; fewer relinked tests; baked paths in `src/` tests) | unnecessary for CANON directories, still needed for real-path lanes | `seed-warm-lane.sh` | those tasks | pending; γ edits the same file, rebase whichever lands second |
| reify tasks 9317, 9769, 10168, 9603, 10152, 9712 (seed, base stamp and refresh work) | same files | `seed-warm-lane.sh`, `refresh-warm-base.sh`; the base's existing per-generation stamp records where the base tree lives and is unchanged, while this PRD's stamp records where a build directory was built | those tasks | pending; rebase whichever lands second |
| reify task 9260 (one helper for environment variables infra tests must not inherit) | consumed | the primitive's `REIFY_CANON_*` variables join that list | β1 adds them where the list lives when it is implemented | pending, parked |
| reify `CLAUDE.md`, "Vendored sandbox helpers" (bwrap not used for the GUI sidecar because of a crash on kernel 6.17) | amended | the statement is scoped to the sidecar; bwrap runs the Claude CLI on the current kernel (S9) | λ | — |

## 9. Decomposition plan

Reify tasks are filed in reify's task store and linked across projects. Sizes are expected changed lines
including tests. The reify tasks are not parked behind reify's alpha-spine holder: they have a consumer
clock, this PRD's gates. Leaves that edit one file are chained by dependencies. No leaf is pinned; hub
files (`config.py`, `verify.py`, `harness.py`) are contended, and pinning the chain is the operator's call.

- **α. Measure the drafted shape on real reify, on both hosts** (dark-factory; operator gate, no code;
  ruling 5). Adapt the study's trial harness (`plans/sccache-path-key-study-2026-11-18/trial/exp2.sh`, on
  disk beside the study) to this PRD's shape: whole leg at CANON, a foreground server on a unix socket,
  the store a private `rclone serve webdav` on the workstation reached from the laptop, `PWD` set, no PID
  namespace. Then: compare the hosts' key inputs with `prd-scout/S8-drift-probe.sh`; build a task verify's
  four steps on one host into the empty store; build the same commit on the other host in a never-built
  worktree and run the three baked-path test packages there; start twenty servers at once and count
  read-only ones; launch one real sandboxed agent session behind the bind prefix and have it run
  `cargo test` on a small crate.
  Signal: the second host's counters, the list of missed units, the read-only count and the agent
  session's outcome are recorded in the resolution.
  Criterion: misses on the second host are at most 2% of cacheable requests (the same-host trial had 0 of
  1,150; the toy cross-host run 0 of 11), all three test packages pass, and the agent session's tests pass.
  Otherwise the resolution names the cause and amends this PRD before β1 starts; a cross-host shortfall
  alone means one store per host (ruling 2). Unlocks β1, θ.
- **δ. Liveness through mount namespaces, reify side** (reify, about 350 lines). C11 in
  `scripts/lib_live_refs.sh`; boundary row 19.
  Files: `scripts/lib_live_refs.sh`, `tests/infra/test_live_refs_namespace.py` and its `.sh` wrapper (new),
  `tests/infra/run-all-classification.manifest`.
  Signal: `live_ref_present <lane>` returns live for a namespaced process holding only an open file in the
  lane after its visible parent is killed, `warm-lane-gc.sh` on that scratch lane refuses to delete, and an
  unrelated foreign-namespace process changes nothing. Unlocks β1, ι, κ.
- **β1. The primitive: modes, namespace entry, trampoline** (reify, about 900 lines). C1–C3, C7's
  trampoline, and `incremental` policy of C4 end to end; decisions 2, 3, 8–10, 12; boundary rows 1–4, 9,
  11, 12, and 17 under `incremental` policy.
  Files: `scripts/canon_run.py` (new), `scripts/canon-bin/cargo` (new), `tests/infra/test_canon_run_units.py`
  and `tests/infra/test_canon_run_host.py` with their `.sh` wrappers (new),
  `tests/infra/run-all-classification.manifest`, `scripts/verify-pipeline-paths.txt`.
  Signal: `canon_run.py status` prints the mode of a scratch tree's build directory and the host's policy;
  `cargo test` through the trampoline passes at CANON under `incremental` policy with the stamp written,
  even when the caller exported `CARGO_INCREMENTAL=0`; with the host disabled the same command is
  byte-identical to plain cargo. Depends on α (cross-project) and δ. Unlocks β2, γ, ζ.
- **β2. Cache server, guard, store relay, fallback, the event and the journal** (reify, about 1,200
  lines). `shared` and `mixed` policy of C4; C5, C6, C7's guard; decisions 11, 14, 20, 21; boundary rows
  5–8, 10, 13–16, and 17 under `shared` and `mixed`; the note
  `docs/notes/sccache-key-path-sensitivity.md` and the event name added to the channel's contract.
  Files: `scripts/canon_run.py`, `scripts/canon_cache.py` (new), `scripts/canon-bin/guard/sccache` (new),
  `tests/infra/test_canon_cache_units.py` and `tests/infra/test_canon_cache_host.py` with wrappers (new),
  `tests/infra/run-all-classification.manifest`, `docs/notes/sccache-key-path-sensitivity.md`,
  `docs/prds/v0_6/run-all-runner-infra-harness.md`.
  Signal: the scratch workspace built at a second path under `shared` policy reports every cacheable unit
  as a hit and writes a `build_cache` event and a journal row with `mode` `canon`, `backend` `shared` and
  the command's processor time; with the store stopped it reports `local_fallback`. Depends on β1.
  Unlocks η, λ, ι.
  G7 waiver: no-silent-fail-soft — fallbacks of agent and interactive builds raise no escalation of their
  own; they are journaled per build directory and escalated per host through verify results, which share
  the same store (decision 18).
- **γ. Reify launch sites and the seed** (reify, about 450 lines). C12; boundary row 18.
  Files: `hooks/pre-commit`, `hooks/pre-merge-commit`, `scripts/seed-warm-base-initial.sh`,
  `scripts/ensure-warm-base.sh`, `scripts/seed-warm-lane.sh`, `scripts/refresh-warm-base.sh`,
  `tests/infra/test_seed_warm_lane.sh`.
  Signal: seeding a scratch lane from a CANON-stamped base leaves every file byte-identical to the base and
  prints no relocation or relink line; seeding a lane whose tree lacks the primitive leaves no stamp; a
  commit through the hook in a tree with a CANON-stamped build directory records a ledger row whose lane
  is the real path. Depends on β1. Unlocks ι, κ.
- **ε. Liveness through mount namespaces, dark-factory side** (dark-factory, about 450 lines). C11;
  boundary row 19.
  Files: `shared/src/shared/proc_group.py`, `shared/tests/test_proc_group.py`,
  `orchestrator/scripts/warm-lane/lib_live_refs.sh`, `orchestrator/tests/test_live_refs_lib.py`.
  Signal: `scan_process_groups_under_path` returns the group, and `cpu_ticks_under_path` a positive count,
  for a namespaced process with no visible parent; an unrelated foreign-namespace process is absent from
  the result. Unlocks ι, κ.
- **ζ. Launch wiring** (dark-factory, about 700 lines). C8; boundary rows 20, 21.
  Files: `orchestrator/src/orchestrator/config.py`, `verify.py`, `offline_lane_runs.py`,
  `agents/sandbox_dispatch.py`, `agents/invoke.py`, `workflow.py`, and a test beside each.
  Signal: with `build_canon.root` set, a verify leg on a fixture repository whose primitive is a recording
  stub shows the stub was started with `exec -- /bin/bash -c …`, and a sandboxed agent
  launch shows the bind prefix before the landlock helper; unset, both are identical to today's. Depends
  on β1 (cross-project) and dark-factory task 10736. Unlocks η, ι, κ.
- **η. `build_cache` facts, the observation log and the degraded-cache alarm** (dark-factory, about 900
  lines). C9, C10; the ladder extracted to one module; boundary rows 22–25.
  Files: `orchestrator/src/orchestrator/verify_event_channel.py`, `verify.py`, `verify_runner.py`,
  `streak_alarm.py` (new), `build_cache_observer.py` (new), `task_verify_lane_alarm.py`,
  `task_verify_dispatch.py`, `harness.py`, `orchestrator/tests/test_build_cache_observer.py` (new),
  `orchestrator/tests/test_task_verify_lane_alarm.py`.
  Signal: a verify leg run through the real primitive appends an observation row whose `mode` and `backend`
  match the primitive's event; two consecutive `local_fallback` results file one L1 under
  `__build_cache_degraded__<host>` and a healthy result resolves it. Depends on β2 (cross-project), ζ and
  dark-factory task 10737. Unlocks ι.
- **θ. Store service, pruning, drift probe and the in-service check** (dark-factory, about 700 lines).
  C13, with tests using a private rclone and fixture observation files, and an `OPERATIONS.md` section:
  provisioning `/opt/reify-canon`, the host configuration file, enabling, disabling and rolling back a
  host in the order of decision 22, reading observations, the alarm, pruning, and shrinking the old host
  cache.
  Files: `scripts/deploy/install_build_cache_store.sh`, `scripts/deploy/build_cache_store_prune.py`,
  `scripts/deploy/build_cache_drift_probe.sh`, `scripts/deploy/check_build_cache_in_service.py` (all new),
  `scripts/tests/test_build_cache_store.py` (new), `OPERATIONS.md`.
  Signal: after the installer, `--check` exits 0 and a PUT then GET through the endpoint round-trips; the
  pruner brings a fixture store under its cap, oldest first, leaving a file younger than ten minutes; the
  in-service check returns 1, 0, 2 and 75 on fixture observation files; the drift probe prints a
  pkg-config line for `xscrnsaver` and for each module of a fixture `Cargo.lock`'s system-deps crates.
  Depends on α. Unlocks ι.
- **λ. Interactive sessions** (reify, about 300 lines). `scripts/setup-dev.sh` installs the trampoline into
  `~/.local/share/reify-canon/bin`, puts it first on login shells' `PATH`, and reports when something
  shadows it; `scripts/run-gui.sh` and `scripts/run-gui-dev.sh` build and launch through
  `canon_run.py exec`; a "Canonical build path" section in reify's `CLAUDE.md` (cargo goes through the
  trampoline; a test binary built at CANON runs through cargo or `canon_run.py exec`; CANON paths appear
  in compiler output and backtraces and must not be written into plans, escalations or task records;
  never run `git worktree repair` from CANON, which rewrites the lane's admin record, S9), and the
  sandbox-helper note of §8 scoped.
  Files: `scripts/setup-dev.sh`, `scripts/run-gui.sh`, `scripts/run-gui-dev.sh`, `CLAUDE.md`,
  `tests/infra/test_canon_interactive.py` and its wrapper (new),
  `tests/infra/run-all-classification.manifest`.
  Signal: in a scratch home, after `setup-dev.sh`, `cargo` resolves to the trampoline and
  `canon_run.py status` reports CANON mode for a fresh build directory on an enabled host. Depends on β2.
  Unlocks κ.
- **ι. Enable on `leo-laptop`** (dark-factory; operator gate, no code). Provision `/opt/reify-canon` on
  both hosts; install the store on the workstation and `rclone` on the laptop; write the laptop's host
  configuration and `build_canon` in its yaml; reset the laptop's task-verify lanes and merge lane so
  their build directories are recreated; wait for laptop verifies; run the drift probe on both hosts and
  compare (on 2026-11-19, after `libxss-dev` and `libocct-draw-dev` were installed on the workstation, the
  study's probe and the reify-probed pkg-config modules showed no difference); run the in-service check with `--min-runs 5` and `--canon-root`; then set `expect_canon` for
  the laptop. Watch the laptop's verify memory cap for a killed compile, and verify wall time for a slow
  store (decisions 14, 15).
  Signal: the in-service check exits 0 for the laptop, and an observation row for a laptop verify shows
  `mode` `canon`, `backend` `shared`. Depends on α, β2, γ, δ, ε, ζ, η, θ.
- **κ. Enable on the workstation and choose its policy** (dark-factory; operator gate, no code). Write
  the workstation's host configuration and `build_canon`; reload reify's orchestrator configuration on a
  commit containing ζ and η. Then, for each of `shared`, `incremental` and `mixed` in turn: set the host's
  policy; with the merge queue halted and idle, drop the persistent merge lane's build directory so the
  next merge verify rebuilds the warm base at CANON under that policy (one cold build each), and release
  the queue; let a comparable number of tasks run; collect, from the invocation journals of their lanes
  and from the observation log, processor time per task for agent builds and for verifies, and the cold
  base build's own. For the two incremental policies also record whether a newly seeded lane's first
  build reuses the base's incremental state or recompiles the workspace. Choose the policy with the least
  processor time per task, set it, and rebuild the base once more if it is not the last one measured. Then:
  confirm a newly seeded lane's stamp is CANON; run `setup-dev.sh` for interactive shells and launch the
  GUI once from a CANON build directory; set `expect_canon`; lower the old host cache's cap once the store
  holds a week of builds. Set the laptop to the chosen policy and reset its lanes, since hosts on
  different policies share at most registry crates. Rollback at any step follows decision 22.
  Signal, and the goal signal of this PRD: the three policies' measured processor time per task and the
  choice are recorded in the resolution; the in-service check exits 0 for the workstation; and a build of
  current main in a never-built interactive worktree shows no registry-library misses, and under `shared`
  policy no workspace-library misses either.
  Depends on ι, γ, λ.

## 10. Out of scope

- Other projects' builds.
- A store on a host other than the workstation, or replication between stores.
- Upgrading sccache. Version 0.18 adds a two-level cache (local first, shared second) that would make
  hits faster and an outage softer. Trigger to revisit: `store_lost` observations more than once a week.
- Keeping one sccache server alive across commands (decision 11). Trigger: the per-command start shows up
  in the agent build costs measured in gate κ.
- A standing, scheduled comparison of the two hosts' key inputs. The drift probe is run at the gates;
  drift afterwards costs hits, not correctness. Trigger: the laptop's hit share falls in the observation
  log with no store outage.
- Reporting fallbacks of agent and interactive builds individually (decision 18).
- An enforcement wrapper in front of every compile (decision 5).
- rust-analyzer and other editor tooling, which run cargo themselves; none is installed on these hosts.
- Removing the seed's relocation and relink code, still needed for real-path lanes; a later cleanup.

## 11. Open questions (tactical)

1. **Which `CARGO_*` variables the primitive removes.** Start with `CARGO_TERM_*` and `CARGO_LOG`; decide
   at β2 from what live agent and verify environments carry.
2. **The pinned `CARGO_HOME`.** Use the value reify's verify configuration pins, with a test that the two
   agree. Decide at β2.
3. **Where the trampoline goes for T3-hosted and other non-login sessions.** Decide at λ from how those
   sessions get their `PATH`.
4. **Composition with reify's existing `scripts/agent-bin/cargo` shim.** It must run before the change of
   directory or its lane start-log is skipped. Decide at β1.
5. **The store's size cap and pruning cadence.** Start at 300 GiB and daily; revisit at κ.
6. **Watchdog probe interval and what counts as slow.** Start at 15 s and 2 s. Decide at β2.
7. **Whether incremental state survives seeding.** A lane's build directory is a copy of the base's. If
   rustc does not accept the copied incremental state, the two incremental policies recompile the
   workspace once per lane. Measured at κ, not assumed.
8. **How the incremental setting is passed.** sccache hashes every `CARGO_*` variable, so setting
   `CARGO_INCREMENTAL` differently under `mixed` and `shared` also splits registry crates' keys between
   the two. Passing it as a cargo `--config` argument would not. Decide at β2.
9. **How the observer reaches `VerifyRunnerPool.dispatch`.** A late-bound process-wide sink avoids editing
   the pool's five construction sites; decide at η against what task 10737 does for its own alarm.
