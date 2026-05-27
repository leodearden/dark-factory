# Author mode — conversational flow

A design session that ends with a committed PRD on disk. The PRD is the **output** of the conversation, not a template to fill in. The skill drives discussion through the gates; the user brings the goal and the engineering judgment.

> **Overlay first.** Before Stage 0, complete Step 0 of `SKILL.md` — load `<root>/.claude/skills/prd/project.md` if present. It supplies the PRD output path, signal vocabulary, the G3 verifier, project mechanism-checks, exemplars, and the memory namespace this flow refers to as **[overlay]**.

## Termination condition

The session is done when this can be answered "yes":

> If I decompose and queue this PRD without further oversight, will the architecture of what gets implemented be complete, coherent, cohesive, and **good**?

If not, keep designing. No open **design** questions remain at PRD-save time. Tactical questions go in `## Open questions`. See `gates.md` § META for the design/tactical boundary.

## Conversational style

- Terse, technical. No preamble.
- Surface design choices as 2–4 way `AskUserQuestion` menus when the choice is genuinely independent of other context; otherwise ask inline.
- Do **not** recommend a single answer unless analysis genuinely converges; when you do recommend, label it `(Recommended)` and put it first.
- Push back if the framing has an unstated assumption. Default toward asking the question rather than answering it.
- Keep the conversation moving — don't recite the gates as a checklist, weave them into substantive discussion.

## Flow

### Stage 0 — Frame the work

Establish:
- **Goal.** What does the user want? What problem does this PRD solve? 1–3 sentences.
- **Milestone / placement.** Drives the output path. **[overlay]** defines the path convention (e.g. `docs/prds/<vM_N>/<slug>.md`); in generic mode, ask the user where PRDs live.
- **Slug.** kebab-case filename. If unclear, propose 2–3 candidates.
- **Type.** Greenfield PRD / contract resolving an existing accreted PRD / extension of a shipped PRD.

If a relevant memory exists (similar past PRD, related decisions), surface it via `search(query="<topic> design decisions", project_id="<project_id from overlay>")`.

### Stage 1 — Goal + motivating signal (drives G1)

Have the user describe what a user observes if this PRD lands. Push for **specifics** — what command, what response, what state, what artifact. "A user can …" sentences with concrete artifacts. This seeds G1 (the consumer) and the decomposition plan's user-observable signals.

### Stage 2 — Enumerate mechanisms (drives G1, G3)

For every mechanism the PRD will introduce, capture in conversation:
1. **What it is** — value type, struct, fn, syntax surface, endpoint, runtime entry, UI affordance.
2. **Consumer** — which PRD or user surface consumes it. Push back on "future consumer in an unfiled PRD".
3. **Substrate reality check** — if the mechanism assumes any capability that might not exist yet (novel syntax, an endpoint, a schema column, a flag), schedule a G3 verification now. Fail-fast: rewrite or queue substrate work *before* sinking design effort into a fiction. **[overlay]** supplies the verifier and any project-specific mechanism patterns to watch for (e.g. known runtime/dispatch gaps gated on a tracked task).

### Stage 3 — Cross-PRD seams (drives G4)

Identify every other PRD this one touches. For each: **direction** (produces/consumes), **mechanism** (the specific fn/event/file/trait crossing the boundary), **owner** (which PRD's decomposition holds the integration task). Resolve reciprocal "the other owns it" ambiguity *now*. Build the `## Cross-PRD relationship` table inline. **[overlay]** may list known contested pairs — confirm this PRD doesn't add a new instance.

### Stage 4 — Approach choice (drives G5)

Apply the G5 heuristic from `gates.md`. If cross-module blast radius ≥ 3, mechanism count ≥ ~8, a load-bearing seam is touched (**[overlay]** names them), or cross-PRD consumers ≥ 2, prompt for the B-vs-B+H choice. If B+H, the next two stages produce the contract section and boundary-test sketch — they shape the decomposition.

### Stage 5 — Contract section (if B+H)

Draft signatures + invariants for the seam. Goal: an architect reading this section can implement the producer side correctly without further discussion — function signatures, lifecycle rules, error semantics, ordering invariants. **[overlay]** points to a worked exemplar.

### Stage 6 — Boundary-test sketch (if B+H)

Draft a table of test scenarios facing **both** sides of the seam. Each row: scenario (one sentence), preconditions (what state must hold), postconditions (what the test asserts). The boundary-test sketch is the integration-gate task's observable signal at decompose time (closing G2's loop).

### Stage 7 — Decomposition plan

Draft the decomposition. For every task:
1. **Title** — verb + noun, ≤ ~70 chars.
2. **Modules touched** — which crates/packages/services this task modifies.
3. **Observable signal** — see G2. Leaf tasks name a user-observable signal; intermediate tasks name the downstream prerequisite they unlock.
4. **Prereqs** — intra-batch (label by letter: α, β, γ, …) and out-of-batch (PRD names, existing task IDs).

Use Greek-letter or numeric labels in the PRD; actual task IDs are assigned at decompose time.

If B+H was chosen, the decomposition includes: Phase 1 — foundation supplements; Phase 2 — vertical slice (minimum-viable end-to-end producing the named consumer signal); Phase 3+ — incremental phases each adding one slice; and a **companion correction-tasks phase** for cross-PRD prose updates this PRD's resolution requires. If bare B, a simpler linear or shallow DAG of vertical slices.

### Stage 8 — Open questions

A `## Open questions` section catches **tactical** questions explicitly deferred:

```markdown
## Open questions (surfaced but not decided in this session)

1. **<question>**. <context>. **Suggested resolution:** <default if any>. Decide during <task α / impl phase>.
```

Allowed to be empty.

### Stage 9 — META check

Before writing the file, **run the termination question aloud**:

> If I decompose and queue this PRD without further oversight, will the architecture of what gets implemented be complete, coherent, cohesive, and good?

If yes → save + commit. If no → identify the design-level open questions, resolve them in conversation, re-ask. The skill is allowed to **fail** here: "not yet good enough" is a valid outcome; close with an unsaved draft + a hand-off note naming the open design questions.

### Stage 10 — Save + commit

Path: the overlay's convention (generic: the path agreed in Stage 0). Write the file, then commit in the same skill turn:

```bash
git add <prd-path>
git commit -m "$(cat <<'EOF'
docs(prd): <one-line goal summary>

<2–3 sentence summary of what this PRD covers + the load-bearing design decisions resolved>

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

The commit happens **before** any decompose-mode work — task agents run in worktrees branched from main and need the PRD on disk.

### Stage 11 — Transition to decompose?

After committing, ask: "session has room — continue into decompose mode now, or stop and trigger decompose later (possibly a fresh session)?" If continuing → switch to `decompose-mode.md`. If stopping → write a hand-off note summarizing what was authored and what's pending.

## PRD section template (content-matched, not literal)

A "good" PRD has these sections (names may vary; content is what matters):

1. **Title + status line** — milestone, "deferred" / "active" / "contract resolving …", date.
2. **Goal** — what user-observable behaviour ships.
3. **Background** — why this exists; architecture-doc references; prior work.
4. **Why deferred** *or* **Activation status** — what gates this on.
5. **Sketch of approach** — surface + mechanism overview. (Often where novel substrate appears; G3 watches here.)
6. **Resolved design decisions** — the choices made in this session.
7. **Pre-conditions for activating** — upstream PRDs / tasks / substrate prerequisites.
8. **Cross-PRD relationship** — the G4 seam-owner table.
9. **Decomposition plan** — the task DAG with observable signals.
10. **Out of scope for this PRD** — explicit exclusions; future-PRD pointers.
11. **Open questions** — tactical-only.
12. **(B+H only)** **Contract section** + **Boundary-test sketch**.

**[overlay]** points to worked exemplars for each shape (B+H full, bare-B large, G4-strong).
