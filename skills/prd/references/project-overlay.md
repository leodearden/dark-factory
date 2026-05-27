# Project overlay — how to specialize this skill for a project

The generic `/prd` skill is project-agnostic. A project specializes it by shipping an **overlay** at:

```
<project-root>/.claude/skills/prd/project.md
```

## Why this location (and not a competing SKILL.md)

- The generic skill is a **personal** skill (`~/.claude/skills/prd/`, usually a symlink to `dark-factory/skills/prd/`). Per Claude Code precedence, a personal skill **shadows** a same-named project skill — so a project `SKILL.md` named `prd` would be ignored anyway.
- A directory under `.claude/skills/<name>/` with **no `SKILL.md`** is silently ignored by skill discovery. So `.claude/skills/prd/` containing only `project.md` + `references/` is *not* registered as a skill — it's pure data the generic skill reads at Step 0.
- `.claude/skills/` is also the conventional place projects already track `.claude` content (e.g. reify's `.gitignore` is `.claude/*` + `!.claude/skills/`), so the overlay lands in a tracked, worktree-visible path.

**Do not** add a `SKILL.md` here. Put project specifics in `project.md` and any reference files under `references/`.

## What the overlay defines

The overlay is free-form Markdown the generic skill reads as authoritative extensions/overrides. Cover whatever the project needs; the generic skill looks for these:

### 1. Identity & paths (required for decompose mode)
- `project_id` — the fused-memory tag (e.g. `reify`, `dark_factory`).
- `project_root` — absolute path used in every fused-memory call.
- **PRD path convention** — where authored PRDs are saved (e.g. `docs/prds/<vM_N>/<slug>.md`), and how the milestone segment is chosen.
- **Commit convention** — message prefix/format if the project has one.

### 2. G2 signal vocabulary
Project-specific user-observable signal types beyond the generic menu (CLI / API / persisted-state / UI / log / diagnostic / CI example). E.g. "viewport state via debug MCP", "LSP hover content", "a stdlib example in the project's language".

### 3. G3 substrate verifier
The concrete check that proves an assumed capability exists, with command + pass/fail semantics, and a reference file if the procedure is involved. Examples:
- Language/DSL project: a **grammar gate** — extract syntax fragments to fixtures, parse them; exit 0 = pass. (See reify's `references/grammar-gate.md`.)
- Web/service project: "the route exists in the router table"; "the migration adds the column".
- If the project has no such substrate, say so → G3 reduces to the generic manual check.

### 4. G1 integration-seam catalogue (optional)
If the project has a fixed set of legitimate in-system integration seams (dispatch points, route table, plugin registry), list them so an in-system mechanism's named consumer must plug into a catalogued seam, and a NEW seam is flagged as a G4 design question.

### 5. G4 known contested pairs (optional)
Known reciprocal-ownership seam fights, so a new PRD doesn't add another instance.

### 6. G5 load-bearing seams + threshold overrides
The project's high-stakes seams (the ones where integration tasks starve under narrow locks), and any tuning of the blast-radius / mechanism-count / cross-PRD-consumer thresholds.

### 7. G6 domain flag + hazards
Whether the project is numerically/scientifically heavy (branches 1–2 fire often) or not (mostly branch 3). List domain-specific premise hazards (e.g. FEA element locking, spline end-conditions, numerical conditioning) and any worked cautionary examples.

### 8. Exemplars
Paths to gold-standard PRDs in the repo, labelled by shape (B+H full / bare-B large / G4-strong), so author mode can point at a worked example.

### 9. Memory namespace
The project's relevant memory slugs (decisions, conventions, prior PRDs) the skill should `search` / cite.

### 10. Provenance (optional)
Why these gates matter for this project (e.g. an architecture-audit that motivated them), and pointers to the audit/design docs the skill may cite at G4 / META time.

### 11. Substrate-confirmed metadata field name
The decompose-mode metadata flag name for "substrate exists" (reify uses `grammar_confirmed`; a web project might use `route_confirmed`). Defaults to `grammar_confirmed` if unspecified.

## Reference files

The overlay may ship additional files under `<root>/.claude/skills/prd/references/`. The generic skill Reads them when `project.md` points to them. Reify ships:
- `references/grammar-gate.md` — the G3 verifier (tree-sitter mechanics).
- `references/gates.md` / `references/decompose-mode.md` — **pointer stubs** preserved so older in-repo docs that linked to those paths still resolve; they redirect to the generic skill + the overlay.

## Minimal overlay skeleton

```markdown
# <Project> PRD overlay

project_id: <id>
project_root: <abs path>
PRD path: <convention>

## G2 signals      — <project-specific signal types>
## G3 verifier     — <command + semantics, or "none">
## G5 seams        — <load-bearing seams; threshold overrides>
## G6 domain       — numerical | not; <hazards>
## Exemplars       — <paths by shape>
## Memory          — project_id <id>; slugs: <...>
```
