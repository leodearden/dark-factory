# Cross-PRD seams discovered by PRD-1 (append-only)

Protocol: see the seam ownership register in `plans/escalation-flow-2026-06-04-prd-briefs.md`.
Entries below are newly discovered seams NOT in the static register. Siblings: glob
`plans/escalation-flow-gaps-prd*.md` before finalizing your PRD.

## 2026-06-04 — PRD-1 entry 1: invoke env lives in shared/, not orchestrator/

The register row "orchestrator/.../harness.py — watcher-supervisor region + invoke env"
mislocates the invoke env: the verbatim `os.environ` copy is at
`shared/src/shared/cli_invoke.py:822`, a package consumed by **every** agent invocation
(implementers, stewards, curator, watcher). PRD-1 deliberately makes **no edit to shared/**:
`invoke_with_cap_retry(**invoke_kwargs)` forwards `env_overrides` to `invoke_agent` (verified),
so the BASH_MAX_TIMEOUT_MS injection happens entirely at the `_run_watcher_rotation` call site
(PRD-1-owned harness region). `shared/cli_invoke.py` remains untouched and unowned — if PRD-2/3
need env changes for their own invocations, the same call-site `env_overrides` route applies;
do not edit the shared copy loop.

## 2026-06-04 — PRD-1 entry 2: sweep wiring adds a server-startup region to escalation/server.py

PRD-1 wires the queue-root sweep + archive prune into the escalation server's startup path
(inside the escalation package, at/near `create_server()` — the pre-serving single-writer
window). The register assigns PRD-3 only the CATEGORIES list and `resolve_issue` docstring in
that file; this is the expected same-file/different-section case. **Heads-up to PRD-3:** if your
doc-truth edits touch server.py module-level prose, expect a new startup-sweep block near
`create_server` after PRD-1's ε task lands.

## 2026-06-04 — PRD-1 entry 3: watcher.py initial-scan changes arm-time semantics for ALL consumers

PRD-1 adds an initial scan to `escalation.watcher` (arm inotify FIRST, then scandir for
already-pending matches, emit-and-exit if found). Net effect: the watcher now returns
*immediately* when a matching pending escalation already exists at launch — previously it
blocked until the next inotify event. This is safe under the documented invariant (watcher is
wake-signal-only; drains are authoritative; consumers re-drain after every return and tolerate
spurious wakes), and it is precisely the fix for the drain-before-up race on both tiers.
**Heads-up to PRD-3:** the L2 watcher-launch prose you own in `escalation-watcher/SKILL.md`
stays correct as written (drain-after-up already landed, ee58fe3464), but if you rewrite those
sections, the "watcher may fire instantly at launch if something is already pending" behaviour
is now guaranteed rather than incidental — a one-line note there would not go amiss.

## 2026-06-04 — PRD-1 entry 4: ack of PRD-2 entry 1 (b3-state.json / afk-digest.md protection)

Constraint adopted as a PRD-1 design decision (D6): the sweep/reaper extension keeps the
existing `esc-*.json` glob (sweep.py:116, queue.py:70) and never widens it; non-escalation
queue-root residents (`b3-state.json`, `afk-digest.md`, future state files) are structurally
invisible to it. PRD-1's ε task carries this as an explicit non-goal plus a regression
assertion (sweep leaves non-`esc-*` files untouched).
