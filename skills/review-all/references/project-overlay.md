# Project overlay schema — `<ROOT>/.claude/skills/review-all/project.md`

Same mechanism as `skills/prd/references/project-overlay.md` and `skills/hotspot-survey/references/project-overlay.md`: the generic skill (this directory, symlinked into `~/.claude/skills/review-all`) reads the overlay at Step 0 and treats it as authoritative extensions and overrides. Do not create a competing `SKILL.md` under the project's `.claude/skills/review-all/`; a directory holding only `project.md` is ignored by skill discovery and the overlay loads cleanly.

Free-form markdown covering these slots; omit any that match the generic default.

| Slot | What it supplies | Generic default |
|---|---|---|
| **Identity** | fused-memory `project_id`, `project_root`, `agent_id` convention | elicit from `dark-factory-orchestrator.yaml` / CLAUDE.md |
| **Briefing path** | where `review/briefing.yaml` lives if not at the default; the area keys are always its `subprojects` (contract §3) | `review/briefing.yaml` |
| **Path → area map** | the one rule of contract §3 is fixed (the briefing subproject whose member directory contains the path); the overlay names the rule for paths under no member (`scripts/` in dark-factory), and which uncovered directories are briefing defects to report for `/review-briefing` rather than map. When the briefing names members in prose, not as member lists, this slot is the one home of the mapping: an ordered rule list (first match wins, a catch-all per area, sub-package rules where a package spans areas by file), with each inferred mapping listed as a briefing defect (SKILL.md Phase 0 step 2) | member directory → area; uncovered paths reported, never silently mapped |
| **Source and test roots** | globs for source and for tests, per language | Python: `*/src/**/*.py`, `*/tests/**/*.py`, `scripts/**/*.py` |
| **Quality definition source** | how the definition reaches every judging prompt (contract §10): `Read docs/code-quality.md` when the project carries the file, otherwise the lead embeds the render of `orchestrator/src/orchestrator/agents/code_quality.py::guidance` (run from the dark-factory checkout) as `args.quality_guidance`; a project never restates the doc in its overlay | `Read` when `<ROOT>/docs/code-quality.md` exists, else embed |
| **Metrics instruments** | the per-language equivalents of the quality doc's instruments (cognitive complexity, raw/prose counts, import graph, test patch targets) and how to invoke them; the per-function complexity ceiling the project uses. A project that is neither a Python uv workspace nor a single src-layout Python package must fill this slot: the Python snapshot cannot run there (SKILL.md Phase 1). For a Cargo workspace the instrument is this skill's `scripts/rust_metrics_snapshot.py`, and the overlay supplies the rules in its PROJECT RULES block (domain excludes, area rules, inferred mappings, test-file pattern, lizard excludes, fix-density start date) — reify's are the shipped example. A Python repo that also carries another language (TypeScript under a Python uv workspace, say) names a second instrument for that language here: the Python snapshot does not see its files. Where none is named, the report lists each such area as **unranked** with its file and line counts, and the lead does not rank it against measured areas (scp's first run left 153 TS files, 16k lines, unmeasured, and one area with no ranking instrument) | Python: `radon raw`, `complexipy`, `scripts/quality_metrics_snapshot.py` (measures in `scripts/source_measures.py`); ceiling 15 |
| **Language note** | for a project not written in Python: how to read the quality doc's Python-flavoured heuristic-13 and Tests-stance symptoms in the project's language, and what the complexity figure measures; the generator copies it into `args.language_note` and the area prompt inserts it after YOUR SLICE (`references/orchestration.md` §`args` shape) | none (Python reads the symptoms as written) |
| **Size ceilings** | project-specific `soft_lines` / `alarm_lines` for heuristic 14 | 1,500 / 2,000 (quality doc) |
| **Slice seeds** | known package → slice groupings and a per-slice read budget, to seed the hand-authored slice list; which test suites count as src (the area's product) and which one-small-file areas are read inside a neighbouring slice | derive from the import graph; ~40k *src* lines per slice, a package's tests attached to the slice holding most of its src, ~8–12k read fully (SKILL.md Phase 1 §Read shortfall) |
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
- Metrics: Rust — `skills/review-all/scripts/rust_metrics_snapshot.py` (tokei prose share, lizard CCN pair, `use`-graph, impl-block spread) with its PROJECT RULES block replaced by this overlay's rules; CCN ceiling 15. In a Python repo with another language, add that language's instrument as a second entry (`<lang>` — `<tool and invocation>`), or write `<lang>: none — area unranked`.
- Language note: `<how the quality doc's symptoms read in this language, and what the complexity figure measures>` (reify's, below, is the example).
- Ceilings: soft 1,200 / alarm 1,800.
- Slice seeds: `<area>` → `<slice-slug>` (`<package dirs>`), ...; read budget 15k lines fully per slice.
- Reports: hotspot `plans/bug-hotspot-survey-*`, review `review/reports/`, codebook `docs/legibility/confusion-codebook.yaml`.
- Fable: synthesis and critic pre-authorised (ruling <date>).
- Hand-off: PRDs `plans/<slug>-prd.md`; program `plans/review-all-program-<id>-<date>.md`; briefs `~/.claude/spawn-briefs/`.
```

The **Language note** reify's first run (`review-all-reify-20261010`) used, the example for a Rust project:

> this is a Rust workspace with a TypeScript GUI and shell/Python factory infra. Read the quality doc's Python-flavoured symptoms as their Rust/TS equivalents: a reach-back import is 'use super::'/a parent-module import from a child file; a deferred import is a 'use' inside a fn body; a re-export shim is a 'pub use' kept so an old path resolves; the heuristic-13 "satellite over a parent's private state" is an 'impl Type' block for one type spread across many files (the metrics list impl-block spread per file); the Tests stance's "patching privates" is 'pub(crate)'/'#[doc(hidden)] pub' items that exist for tests, '#[cfg(test)]' modules reading private state, and vi.mock/patch targets in TS/Python tests. The instruments' complexity figure is lizard cyclomatic complexity (CCN), read as the pair (max per function, file total); it is not Sonar cognitive complexity, so re-measure before resting a claim on it.
