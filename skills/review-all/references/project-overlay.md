# Project overlay schema — `<ROOT>/.claude/skills/review-all/project.md`

Same mechanism as `skills/prd/references/project-overlay.md` and `skills/hotspot-survey/references/project-overlay.md`: the generic skill (this directory, symlinked into `~/.claude/skills/review-all`) reads the overlay at Step 0 and treats it as authoritative extensions and overrides. Do not create a competing `SKILL.md` under the project's `.claude/skills/review-all/`; a directory holding only `project.md` is ignored by skill discovery and the overlay loads cleanly.

Free-form markdown covering these slots; omit any that match the generic default.

| Slot | What it supplies | Generic default |
|---|---|---|
| **Identity** | fused-memory `project_id`, `project_root`, `agent_id` convention | elicit from `dark-factory-orchestrator.yaml` / CLAUDE.md |
| **Briefing path** | where `review/briefing.yaml` lives if not at the default; the area keys are always its `subprojects` (contract §3) | `review/briefing.yaml` |
| **Path → area map** | the one rule of contract §3 is fixed (the briefing subproject whose member directory contains the path); the overlay names the rule for paths under no member (`scripts/` in dark-factory), and which uncovered directories are briefing defects to report for `/review-briefing` rather than map | member directory → area; uncovered paths reported, never silently mapped |
| **Source and test roots** | globs for source and for tests, per language | Python: `*/src/**/*.py`, `*/tests/**/*.py`, `scripts/**/*.py` |
| **Quality definition source** | how the definition reaches every judging prompt (contract §10): `Read docs/code-quality.md` when the project carries the file, otherwise the lead embeds the render of `orchestrator/src/orchestrator/agents/code_quality.py::guidance` (run from the dark-factory checkout) as `args.quality_guidance`; a project never restates the doc in its overlay | `Read` when `<ROOT>/docs/code-quality.md` exists, else embed |
| **Metrics instruments** | the per-language equivalents of the quality doc's instruments (cognitive complexity, raw/prose counts, import graph, test patch targets) and how to invoke them; the per-function cognitive ceiling the project uses | Python: `radon raw`, `complexipy`, `scripts/quality_metrics_snapshot.py` (measures in `scripts/source_measures.py`); ceiling 15 |
| **Size ceilings** | project-specific `soft_lines` / `alarm_lines` for heuristic 14 | 1,500 / 2,000 (quality doc) |
| **Slice seeds** | known package → slice groupings and a per-slice read budget, to seed the hand-authored slice list | derive from the import graph; ~40k source lines per slice, ~15k read fully |
| **Instrument report homes** | glob for the latest hotspot findings artefact, `/review` reports dir, codebook path, previous `/review-all` record, metrics snapshots | `plans/bug-hotspot-survey-*`, `review/reports/`, `docs/legibility/confusion-codebook.yaml`, `plans/review-all-*`, `plans/quality-metrics/*.json` |
| **Staleness thresholds** | commits-since figures that make the hotspot and review inputs stale | hotspot: 50 source commits or its human gate `done`; review: any source change |
| **Output directory** | where the report, findings JSON and program doc land, and whether committed | `plans/`, committed (`docs/notes/` if the repo has no `plans/`) |
| **Cross lenses** | additional cross-area lenses beyond the four defaults (never fewer) | the four in `orchestration.md` |
| **Fable authorisation** | whether synthesis/critic may run on fable without `--fable` (a standing project ruling) | provenance rule of `skills/team/SKILL.md` |
| **Known-context sources** | memory queries, incident docs, operator notes that seed `leads` | fused-memory `search` only |
| **Hand-off conventions** | PRD path convention, program-doc location, `/spawn` brief directory; the trigger chain is never overlay-specific — it is filed per `skills/_shared/filing-the-trigger-chain.md`, whose form probe runs in the dark-factory checkout, not in the project | `plans/<slug>-prd.md`, `plans/review-all-program-*`, `~/.claude/spawn-briefs/` |
| **Anti-triggers** | project skills that must not be shadowed | none |

Example skeleton:

```markdown
# review-all overlay — <project>

- Identity: project_id `<id>`, project_root `<path>`, agent_id `claude-interactive`.
- Path → area (paths under no member): `scripts/legibility/**` → `shared`; other `scripts/**` → `repo`; `<dir>/` → briefing defect, report for /review-briefing.
- Quality definition: no `docs/code-quality.md` here — embed the `code_quality.py::guidance` render.
- Metrics: Rust — `cargo clippy -W clippy::cognitive_complexity` for the per-function pair, `tokei` for raw/prose, `cargo modules` for the import graph; ceiling 25.
- Ceilings: soft 1,200 / alarm 1,800.
- Slice seeds: `<area>` → `<slice-slug>` (`<package dirs>`), ...; read budget 15k lines fully per slice.
- Reports: hotspot `plans/bug-hotspot-survey-*`, review `review/reports/`, codebook `docs/legibility/confusion-codebook.yaml`.
- Fable: synthesis and critic pre-authorised (ruling <date>).
- Hand-off: PRDs `plans/<slug>-prd.md`; program `plans/review-all-program-<id>-<date>.md`; briefs `~/.claude/spawn-briefs/`.
```
