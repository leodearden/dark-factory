# Cross-PRD seams discovered by PRD-2 (append-only)

Protocol: see the seam ownership register in `plans/escalation-flow-2026-06-04-prd-briefs.md`.
Entries below are newly discovered seams NOT in the static register. Siblings: glob
`plans/escalation-flow-gaps-prd*.md` before finalizing your PRD.

## 2026-06-04 — PRD-2 entry 1: b3-state.json vs PRD-1's queue-root reaper extension

PRD-2 adds a durable gate-state file at `<project_root>/data/escalations/b3-state.json`
(rolling 24h merge-cap charges + per-proposal launch records; flock + tmp+rename). Brief 1
issue 3 has PRD-1 extending the reaper to "root-orphans and loose archive files" in the same
directory. **Request to PRD-1:** the reaper must match escalation files only (`esc-*.json`) or
allowlist known non-escalation residents — `afk-digest.md` (existing) and `b3-state.json` (new).
A reaper that sweeps unknown root files would silently destroy the B3 runaway guard's state.

## 2026-06-04 — PRD-2 entry 2: attended-mode B3 stales PRD-3-owned category-handler prose

PRD-2 makes B3 posture-configurable (config `attended_b3_enabled` + session override), replacing
the "AFK-only" framing inside the B3 subsection (PRD-2-owned). But the `task_failure` /
`review_issues` category handlers — PRD-3-owned sections — hardcode the old rule at
`skills/escalation-watcher/SKILL.md:378` and `:386`: "**In AFK mode:** try the low-risk
auto-unblock gate first … Spawn the interactive session only when a human is present."
**Request to PRD-3:** when rewriting those sections, defer B3 applicability to the subsection
instead of restating it — e.g. "if the low-risk auto-unblock gate applies (see 'Low-risk
auto-unblock gate (B3)' for when it does), try it first." PRD-2's T4 rewrites the subsection to
self-define applicability so the pointer form stays true under either posture.

## 2026-06-04 — PRD-2 entry 3: ack of PRD-1 entry 4 / D6 (seam 1 closed)

PRD-2 entry 1 is wired: PRD-1's D6 pins the sweep/reaper glob to `esc-*.json` with a regression
assertion protecting non-escalation queue-root residents. `b3-state.json` placement in
`data/escalations/` is confirmed safe. PRD-1 entries 1–3 reviewed: no PRD-2 surface intersection
(b3_gate is a plain CLI, not an agent invocation needing env_overrides; PRD-2 touches neither
server.py nor watcher-launch prose).
