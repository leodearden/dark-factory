# Project overlay schema — `<ROOT>/.claude/skills/hotspot-survey/project.md`

Same pattern as the /prd overlay: the generic skill (this directory, symlinked into `~/.claude/skills/`) reads the overlay at Step 0 and treats it as authoritative extensions/overrides. **Do not** create a competing `SKILL.md` under the project's `.claude/skills/hotspot-survey/` — a directory without `SKILL.md` is correctly ignored by skill discovery, so the overlay loads cleanly.

The overlay is free-form markdown covering these slots (omit any that match the generic defaults):

| Slot | What it supplies | Generic default |
|---|---|---|
| **Memory identity** | fused-memory `project_id`, `agent_id` convention | elicit from user / CLAUDE.md |
| **Task tracker source** | where fix-task history lives + its shape (e.g. `.taskmaster/tasks/tasks.json` with `data.master.tasks`; or a SQLite db + query recipe) and whether to mine the file directly or via MCP | probe in Phase 0; skip lane if absent |
| **Output directory** | where survey artifacts land (`plans/` vs `docs/notes/`) and whether they are committed | `plans/`, committed |
| **Subsystem vocabulary seed** | known subsystem → files mapping to seed Phase 0's cluster list | derive from repo layout + churn |
| **Fix-commit vocabulary** | project-specific commit markers beyond fix/bug/regression (e.g. dark-factory's `amend:` post-merge patch-ups and `red-main` commits) | the generic grep set |
| **History window** | default `--since` (project epoch, or when autonomous commits began) | ~6 months |
| **Deterministic audit fold-in** | a project detector CLI (e.g. reify's `/audit` — phantom-done/orphan detectors) to run inline in Phase 0 and hand to reviewers as known context | none |
| **Doc corpora** | postmortem/PRD locations for the mine:plans lane | `plans/`, `docs/`, CHANGELOG.md, DESIGN.md |
| **Hand-off conventions** | the project's /prd path conventions, program-doc location, release-gate mechanism (e.g. dark-factory deterministic pure-gate tasks vs reify escalate-on-dispatch milestones) | generic /prd |
| **Known-context sources** | extra memory queries or standing incident docs to seed cluster `context` paragraphs | fused-memory search only |
| **Anti-triggers** | project skills that must not be shadowed (e.g. "invariant detector sweeps → /audit, not this") | none |

Example overlay skeleton:

```markdown
# hotspot-survey overlay — <project>

- project_id: `<id>`; agent_id: `claude-interactive`.
- Task tracker: `<path>` — <shape>. Mine directly with python3, not via MCP round-trips.
- Output: `docs/notes/bug-hotspot-survey-<date>.md` (+ `-full-findings.json`), committed.
- Fix vocabulary: add `--grep='<project marker>'`.
- Subsystem seeds: <key> (<files>), ...
- Audit fold-in: run `<cli> --pattern P1,P2,P5 --since <window>` in Phase 0; give findings to the matching clusters as known context.
- Hand-off: PRDs under `docs/prds/`; gate deferred batches with escalate-on-dispatch milestones (see task 5117 precedent).
```
