# Gate 10782 (task α): the canonical-path design measured on real reify, on both hosts

**Date:** 2026-11-19. **PRD:** `plans/canonical-build-path-shared-cache-prd.md`, task α (ruling 5).
**Outcome: the criterion is met.** Second-host misses were 18 of 1,150 cacheable requests (1.57%, limit
2%), all 2,121 tests of the three baked-path packages passed, and the sandboxed agent session's
`cargo test` passed. Two findings do not affect the criterion but bear on later leaves: the store's
start-up check collides between servers started together (§4), and one development package differs
between the hosts outside what the drift probe checks (§3). Proposed amendments are in §7.

Harness: this directory (`alpha.sh`, `launch.sh`, `gov.sh`, `snap.py`, `summarize.py`,
`agent-bin/cargo`). Small results: `results/`. Raw logs stay on the hosts under
`/home/leo/canon-alpha-20261119/results/` (workstation) and
`/media/leo/Internal-2nd/canon-alpha-20261119/results/` (leo-laptop).

## 1. What was run, and how it differs from the 2026-11-18 trial

The trial harness (`plans/sccache-path-key-study-2026-11-18/trial/exp2.sh`) reshaped to the PRD:

| | 2026-11-18 trial | This run |
|---|---|---|
| Canonical path | a directory on the laptop's second disk | `/home/leo/.reify-canon-alpha/ws`, the same string on both hosts (stand-in for `/opt/reify-canon/ws`, which needs root; kept outside `/tmp` so the landlock check in step 5 is real, since the helper grants `/tmp` wholesale) |
| Leg | whole leg in one `bwrap` namespace | the same |
| sccache server | daemon on TCP port 4228 | **foreground child of the leg** (`SCCACHE_START_SERVER=1 SCCACHE_NO_DAEMON=1`, idle timeout 0), addressed by **`SCCACHE_SERVER_UDS=<CANON>/target/.reify-sccache.<pid>.sock`**, no `SCCACHE_SERVER_PORT` anywhere; stats read and server stopped at the end of the leg |
| Cache | local disk directory | **`rclone serve webdav`** on the workstation (port 4231, loopback and the tailscale address, basic auth, leaf directories pre-created); the laptop reached it over tailscale |
| Command's own sccache environment | — | the socket, a local fallback directory, idle timeout 1,800 s, no store variables (C4) |
| `PWD` | inherited | set to CANON (`bwrap --setenv PWD`) and asserted |
| `CARGO_HOME` | `~/.cargo` | `/tmp/reify-agent-cargo-home`, the value reify's `verify_env` and agents pin |
| PID namespace | none | none |

Unchanged: the commit (reify `27c69f616f`, the trial's X, with `main` at `84d5608509`, so the scope is the same
24 packages and the trial's N1/N2 are direct comparators); the verify's four build steps extracted from
`verify.sh --print-plan`; a systemd user unit at `CPUWeight=30` with a clean environment; the governor
freezing the unit while a production verify waited at its gates. Neither run touched the production
sccache server (port 4226) or `~/.cache/sccache`.

## 2. The numbers

| Run | Host | Worktree | Hits / misses (cacheable) | Writes | Read errors, timeouts, write errors | Build CPU-min | Comparator |
|---|---|---|---|---|---|---:|---|
| W1 | workstation | never built, empty store | 12 / 1,138 | 1,138 | 0, 0, 0 | 113.6 | trial N1 (cold): 2 / 1,148, 109.7 |
| **L1** | **leo-laptop** | **never built, same commit, store holds W1** | **1,132 / 18 (1.57%)** | 18 | 0, 0, 0 | **51.8** | trial N2 (same host): 1,150 / 0, 49.4 |

L1 by step: debug nextest build 494 / 0; GUI nextest build 222 / 9; clippy 237 / 0; GUI check 179 / 9.
All 56 C/C++ objects and both assembler objects hit across hosts. The governor froze L1 three times
(about 70 s in total). W1's unit used 114 CPU-minutes in all, L1's 56 and the agent session's 1; with the
probes and bursts, the gate cost about 175 CPU-minutes.

**Tests at CANON on the laptop** (the debug step's command with `--no-run` removed and the filter narrowed
to `reify-syntax`, `reify-config` and `reify-audit`): **2,121 run, 2,121 passed, 7 skipped**. The same
count as the trial.

W1's end-of-leg stats were lost to a harness bug (the EXIT trap named a variable local to the function);
its counters come from the snapshot after its last step, which is cumulative. L1 used the fixed harness.

## 3. The second host's missed units, and why

The 18 misses are 9 units, each missed by both GUI steps:
`x11_dl`, `tao`, `wry`, `tauri_runtime_wry`, `tauri`, `tauri_plugin_shell`, `tauri_plugin_fs`,
`tauri_plugin_dialog`, `reify_gui`.

Cause, measured: **`libxss-dev` is installed on leo-laptop (since 2026-03-22) and not on the workstation.**
`x11-dl`'s build script asks pkg-config for `xscrnsaver` and writes the answer into its `OUT_DIR/config.rs`
(`results/x11-dl-config.*.rs`; the files differ in one line, `xscrnsaver: None` against
`Some("/usr/lib/x86_64-linux-gnu")`). The compile arguments of every missed unit were identical on both
hosts; the key differs through that generated source, and every unit downstream of `x11_dl` follows
because its key covers its dependencies' artefacts. All 142 build scripts' printed outputs were otherwise
identical except tree-sitter's, whose units still hit.

The drift probe (step 1) did not see this: it digests a fixed list of packages and the OCCT headers, not
the system libraries that build scripts probe for the GUI. It found one other difference, also a
development package present only on the laptop (`libocct-draw-dev`, 182 extra headers in
`/usr/include/opencascade`), which lie outside the C++ wrappers' include closure (identical digests on both
hosts) and cost no miss. Everything else the probe covers was identical: toolchain, sysroot, compiler and
assembler binaries, the toolchain packages, the prebuilt archives, the cargo configuration, the
environment variables.

Further things seen and ruled out:

- 26 "Cache miss" lines in L1's server log against 18 counted misses: the other 8 are build-script
  feature probes of `anyhow`, `thiserror` and `proc-macro2` (`--emit=dep-info,metadata`), which are
  expected to fail, are counted as `compile_fails` and are never stored. They produced the same keys and
  the same lines on W1.
- The plans' environment differed in `LD_LIBRARY_PATH` (`/snap/freecad/current/usr/lib` on the laptop
  only). It enters no compile argument and no key here.
- The default stable toolchain differs between the hosts (1.98.1 against 1.99.1). Reify pins 1.96.0, so
  this does not touch reify keys; it would split any cache shared by a crate without a pin.

So with `libxss-dev` on both hosts (or on neither) the expectation is 0 misses, as in the toy cross-host
run. Not re-measured.

## 4. Twenty servers started at once (PRD boundary row 16, C5)

Each burst starts N foreground servers within about a second, each on its own socket and all against the
store, then compiles one unique C file through each and reads its counters.

| Burst | Servers | Refused to start | Started read-only | Read-write |
|---|---:|---:|---:|---:|
| workstation, existing prefix | 20 | 1 | 6 | 13 |
| workstation, fresh prefix (no `.sccache_check` yet) | 20 | 2 | 14 | 4 |
| workstation, existing prefix, repeat b | 20 | 0 | 0 | 20 |
| repeat c | 20 | 0 | 10 | 10 |
| repeat d | 20 | 0 | 3 | 17 |
| repeat e | 20 | 0 | 0 | 20 |
| both hosts, 10 + 10 at the same second | 20 | 0 | 10 (3 workstation, 7 laptop) | 10 |

The mechanism is sccache's start-up check, which reads and then writes the one file
`<prefix>/.sccache_check`:

- **Read-only:** a concurrent PUT of that file gets HTTP 423 Locked from rclone; sccache prints
  `storage write check failed: … Locked` and runs read-only for its whole life (36 such lines).
- **Refused to start (not in the PRD):** a GET of that file while another server is writing it returns a
  truncated body; sccache aborts with `cache storage failed to read: … end of file before message length
  reached` (3 servers). The PRD's probe (C5) would already have passed, so a primitive following C5 as
  written would start the command against a server that is not there; the client would then auto-start a
  detached server from the command's environment (the local fallback, decision 11), which is harmless
  but is the case the design counts as `detached_server`.
- **What a read-only server looks like at the end:** every miss counts one write error and there are no
  writes (in every burst the read-only servers were exactly the ones that wrote nothing). C5's wording
  ("misses without writes or write errors") should read "misses with write errors and no writes". Write
  errors also come from ordinary same-key races (S5), so the counters alone cannot tell read-only from a
  race; the server's stderr line can, exactly.

The C5 start lock serialises starts within a host only. The cross-host burst shows collisions across
hosts are as common as within one when starts coincide. They will be rarer in practice (starts are spread
out), but nothing bounds them.

## 5. The sandboxed agent session (PRD decision 7, C8)

On the workstation, in W1's worktree (built at CANON by the verify steps), launched like an orchestrator
sandboxed role:
`/usr/bin/bwrap --dev-bind / / --bind <lane> <CANON> --die-with-parent -- python3 <DF>/orchestrator/src/orchestrator/agents/landlock_exec.py --writable <lane>/.task --writable <lane> --writable <results> -- claude -p …`,
working directory the real lane path, `CLAUDE_CONFIG_DIR` under the lane's `.task`, a pool account's
token as `invoke.py` passes it, and on `PATH` first a stand-in for the PRD's trampoline (`agent-bin/cargo`:
`shared` policy, a foreground server on a socket in the lane's build directory through CANON, the store,
and a journal row).

The agent was asked to run `cd <CANON>/crates/reify-syntax && cargo test --offline -p reify-syntax`.
**Result: PASS, cargo exit 0** (the agent's summary covers the last test binary, 193 passed; the exit
status covers them all). `reify-syntax`'s tests read the language spec through a compile-time-baked CANON
path, so they could only pass with CANON bound. Journal row: 3 Rust misses, all written, no write errors,
the server read-write, 44 s wall. The server was stopped and its socket removed. So, under landlock: the
lane was writable through CANON, a unix socket could be created there, and the store was reachable (landlock
here does not restrict the network).

Two launch details the real wiring (ζ) already handles: the token must be in `CLAUDE_CODE_OAUTH_TOKEN`
as well as the config directory's credentials file (without it the CLI says "Not logged in"), and the
first two attempts failed on exactly that before reaching cargo.

## 6. Smaller observations

- A server killed with its leg (here by `--die-with-parent` when the harness bug ended W1's shell) leaves
  its socket file in the build directory. C12 already has the seed remove socket files; the primitive's
  start should also tolerate a stale one with another pid in its name (it does by construction: the name
  carries the pid).
- The foreground server with `SCCACHE_NO_DAEMON=1` and a unix socket worked unchanged on both hosts; no
  client ever auto-started a server (the harness waits for the socket file before its first query).
- Store latency did not show: L1's build CPU (51.8) is within 5% of the same-host trial's all-hit run
  (49.4), with 18 compiles the trial did not do.

## 7. Proposed PRD amendments (for Leo to rule; none is needed before β1)

1. **C5, start-up collisions (β2).** After starting the server, the primitive checks it: if the server
   exits before its socket answers, or its stderr carries `storage write check failed`, it stops it,
   waits a random 0.5 to 2 s and starts again, up to three attempts; then `local_fallback` with reason
   `store_unreachable` or `read_only_server`. The detection at exit stays as a backstop, reworded as in
   §4. Boundary row 16 then asserts that no server stays read-only while retries remain.
2. **θ's drift probe** also digests `pkg-config --list-all` with each module's version, and the set of
   installed `*-dev` packages, so a probed system library is caught.
3. **ι's preconditions** add: install `libxss-dev` on the workstation (or remove it from the laptop) and
   rerun the drift probe until it reports no difference.

## 8. State left behind

The private store (`canonalpha-store` user unit) has been stopped and the store, worktrees and clones
removed on both hosts; results and raw logs are kept at the paths at the top. `/home/leo/.reify-canon-alpha/ws`
remains on both hosts as an empty directory.
