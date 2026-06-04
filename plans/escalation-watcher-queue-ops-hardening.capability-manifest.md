# Capability manifest — PRD-1 escalation watcher & queue ops hardening

Companion to `plans/escalation-watcher-queue-ops-hardening-prd.md`. Binds each leaf signal's
asserted capabilities to evidence (mechanized G3+G6). All bindings verified against the working
tree 2026-06-04 in the authoring session; **no FAIL bindings** — batch clear to queue.

Evidence vocabulary: `grep:<file>:<line>` = wired on main; `producer:task-<label>` = delivered
by an upstream task in this batch; `stdlib`/`venv` = environment-verified; `live` = observed on
the running system this session.

## α — Harden escalation.watcher CLI (intermediate; carries its own CLI signals)

| Capability | Evidence |
|---|---|
| `inotify_simple.INotify.read(timeout=…)` | venv: `inspect.signature` → `(self, timeout=None, read_delay=None)` |
| Escalation parse + status/level/severity fields read by watcher | grep: watcher.py:67,71,77 (production read path) |
| `BORN_AT_L2_SEVERITIES` importable for the ntfy mapping | grep: escalation/models.py (imported at server.py:15) |
| Queue-root scandir for initial scan (pending filter) | grep: queue.py:70 `glob('esc-*.json')` pattern reused |

## β — Sidecar flock for all queue-root mutators (leaf)

| Capability | Evidence |
|---|---|
| Mutator set enumerable (resolve/dismiss/add_members_to_l2/attach_dedupe_child/submit) | grep: queue.py:337,355,473+ (production write paths) |
| `fcntl.flock(LOCK_EX)` | stdlib |
| Writers are atomic tmp+rename (why data-file flock fails → sidecar) | audit-verified (briefs §Brief 1 non-issues); queue.py write helpers |
| Two-process concurrency fixture runnable in CI | project convention: subprocess tests opt-up `@pytest.mark.timeout(N)` ≥3× nominal (fused-memory 60s default) |

## γ — Auto-watcher Main Loop rewrite (leaf)

| Capability | Evidence |
|---|---|
| `--timeout`, exit-124, initial-scan semantics | producer: task-α (upstream — DAG direction ✓) |
| Turn budget fits bounded waits | floor: `watcher_max_turns=400` (config.py:983) ≫ ~27 waits × 2–4 turns ≈ 54–108 per quiet 4h rotation |
| Rotation deadline inputs injected into the agent | grep: harness.py:3640-3644 (ROTATION_ESCALATIONS / ROTATION_HOURS in user prompt) |
| Journal signal observable (force-kill signature exists today as the before-state) | live: journal shows 10/10 unclean at 14702s = 4·3600+300 grace |

## δ — BASH_MAX_TIMEOUT_MS injection at rotation call site (leaf)

| Capability | Evidence |
|---|---|
| `env_overrides` reaches subprocess env | grep: shared/src/shared/cli_invoke.py:773 (param), :825-826 (merge) |
| `invoke_with_cap_retry` forwards kwargs to `invoke_fn` | grep: cli_invoke.py:409+ `**invoke_kwargs` |
| `timeout_secs` available at call site for sizing | grep: harness.py:3653 |
| `BASH_MAX_TIMEOUT_MS` honoured by Claude Code Bash tool | documented settings env var; **belt-only** — if a future CLI ignores it, behaviour degrades to status quo (D1), not a correctness failure |

## ε — Server-start sweep + reaper wiring (leaf)

| Capability | Evidence |
|---|---|
| Sweep reconciliation logic exists (`--apply`) | grep: sweep.py:116 (root glob), :157-162 (reconcile paths) |
| Archive prune exists | grep: archive.py:46 `prune_archive` |
| Pre-serving single-writer window | grep: harness.py:610-611 (`_start_escalation_server` before task execution at :662); external writers are MCP-only; fused-memory recon writes to `data/reconciliation/escalations` (separate root) |
| Glob invariant `esc-*.json` (protects `b3-state.json`, `afk-digest.md`) | grep: sweep.py:116, queue.py:70 — already narrow; ε adds regression assertion (D6) |
| Orphan relief achievable | live: 683 root orphans (445 resolved + 238 dismissed), 11 loose archive-top files counted this session; signal phrased drift-safe ("root contains only pending") |
| Per-id sidecar lock for relocations | producer: task-β (upstream — DAG direction ✓) |
