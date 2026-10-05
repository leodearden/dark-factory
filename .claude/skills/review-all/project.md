# review-all overlay — dark-factory

Project specialization for the generic `/review-all` skill (source `skills/review-all/` in this repo, exposed via `~/.claude/skills/review-all`). Schema: `skills/review-all/references/project-overlay.md`. Created 2026-10-04 alongside the skill; its path→area rule, fable routing and first-run hotspot refresh are Leo's rulings of 2026-10-05.

- **Identity**: `project_id="dark_factory"`, `project_root="/home/leo/src/dark-factory"`, `agent_id="claude-interactive"` (CLAUDE.md §Write-Tagging Convention).
- **Briefing**: `review/briefing.yaml` (default); area keys are its `subprojects` — `fused-memory`, `orchestrator`, `shared`, `dashboard`, `escalation`, `sampler` — plus `repo`.
- **Quality definition source**: `docs/code-quality.md` is present — seats `Read` it; `args.quality_guidance` is `null`.
- **Path → area, paths under no member** — CONFIRMED (Leo, 2026-10-05):
  - `scripts/legibility/**` → `shared` (the legibility pipeline's home; its config and codebook live under `docs/legibility/`, its library seams in `shared/`).
  - other `scripts/**` → `repo` (operator and CI helpers with no owning member).
  - `cockpit/` → **not mapped**: a top-level package the briefing omits is a briefing defect (contract §3); the run reports it for `/review-briefing` and excludes it, rather than inventing an area.
- **Source and test roots**: `*/src/**/*.py`, `scripts/**/*.py`; tests `*/tests/**/*.py`, `dashboard/tests/**`.
- **Metrics instruments**: Python defaults — `radon`, `complexipy` (both in the workspace venv: 6.2.0 / 6.0.1), the AST helpers of `scripts/merge_lane_metrics.py`; `scripts/quality_metrics_snapshot.py` once `plans/quality-metrics-snapshot-prd.md` lands. Per-function cognitive ceiling 15.
- **Size ceilings**: the quality doc's defaults, 1,500 soft / 2,000 alarm.
- **Slice seeds** (re-derive sizes from the snapshot each run; measured 2026-10-04 in source lines): `orchestrator` (209k) → `merge-lane` (`merge_lane/`), `harness-workflow-agents` (`harness.py`, `workflow*.py`, `agents/`), `scheduler-verify-gitops` (`scheduler*`, `verify*`, `git_ops.py`, `warm_lane_pool.py`, `worktree_identity.py`), `routing-escalation-rest`; `fused-memory` (156k) → `server-tools`, `reconciliation`, `memory-services-clients`, `task-layer-curator`; `dashboard` (113k) → two slices by package; `shared`, `escalation`, `sampler` → one slice each. Read budget 15k lines fully per slice.
- **Instrument report homes**: hotspot `plans/bug-hotspot-survey-*` (+ findings artefact), review `review/reports/`, codebook `docs/legibility/confusion-codebook.yaml`, previous review-all `plans/review-all-dark_factory-*`, metrics `plans/quality-metrics/*.json`.
- **Staleness thresholds**: defaults (hotspot ≥ 50 source commits or its gate `done`; review on any source change).
- **First run**: the first `/review-all` refreshes the hotspot survey inside its own Phase 2 (`/hotspot-survey --refresh`, stale by the default threshold since the 2026-07-06 survey); nothing waits for a separate hotspot run first (Leo, 2026-10-05).
- **Output**: `plans/`, committed with `git commit --only` (CLAUDE.md §Working in the main checkout; direct commits race the live merge queue).
- **Cross lenses**: the four defaults.
- **Fable**: synthesis and critic pre-authorised on fable — a STANDING ruling (Leo, 2026-10-05), so no per-run `--fable` is needed. Opus / xhigh is the fallback only when fable is unavailable, and the report's `cost` line says so.
- **Known-context sources**: fused-memory `search` plus the auto-memory index `~/.claude/projects/-home-leo-src-dark-factory/memory/MEMORY.md` (standing rulings, accepted gaps, incident families) → `leads`.
- **Hand-off**: PRDs `plans/<slug>-prd.md`; program doc `plans/review-all-program-dark_factory-<date>.md`; spawn briefs `~/.claude/spawn-briefs/`; trigger chain per `skills/_shared/filing-the-trigger-chain.md` (form probe in this checkout, which is also the factory root).
- **Prior art**: `plans/bug-hotspot-remediation-program-2026-07-06.md` is the proven program-doc shape; the agent-capacity study (`plans/agent-capacity-study-2026-09-09.md`, untracked) priced the self-review this skill supersedes.
- **Anti-triggers**: none beyond the generic skill's.
