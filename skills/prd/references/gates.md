# Gates — G1, G2, G3, G4, G5, G6, META

Each gate's section names:
- **What it catches** — the class of failure it prevents.
- **Application** — the exact algorithm to apply during author or decompose mode.
- **Level** — `block` / `prompt` / `prompt with heuristic`.

These gates are project-agnostic. Where a gate has a project-specific dimension (a signal vocabulary, a substrate verifier, a seam catalogue, a numerical domain), the **project overlay** (`<root>/.claude/skills/prd/project.md`) supplies it — those hooks are flagged inline as **[overlay]**.

---

## G1 — Consumer named

**Level:** **block** (both modes; checked at PRD save and re-checked at decompose).

**What it catches.** Producer-orphans: a mechanism is fully built but no named consumer ever wires it in, so the integration task stays pending indefinitely. This is the single most common implementation-chain failure under a narrow-file-lock orchestrator.

**Application.** For every mechanism the PRD introduces — value type, struct, fn, syntax surface, runtime entry, API endpoint, UI affordance, kernel hook — the PRD must name **at least one consumer**:
1. A specific other PRD by slug/path, AND/OR
2. A specific user-observable surface (CLI command, API response, UI behaviour, IDE diagnostic, example artifact that runs in CI).

A "mechanism" is anything for which you can write a one-sentence end-to-end test ("does X work end to end?").

If no consumer can be named today, the PRD is incomplete by construction. Two valid resolutions:
- **(a)** Defer the producer work until the consumer-side PRD exists. Mark this PRD blocked-on-consumer in `Pre-conditions for activating`.
- **(b)** Author the consumer-side PRD first (or as a paired commit), then return.

Do **not** accept "future consumer in an unfiled PRD" as a named consumer. That's the failure mode the gate exists to prevent.

**[overlay] Integration-seam sub-check.** A project may define a catalogue of legitimate in-system integration seams (e.g. an engine's dispatch points, a service's route table). When the overlay defines one, an in-system-seam mechanism's named consumer must plug into a catalogued seam; a NEW seam is itself a cross-PRD design question (fold into G4 or author a norm-extension first).

**In author mode:** conversational — walk the introduced mechanisms one by one, ask for the consumer of each, push back on fictional / future consumers.

**In decompose mode:** re-check by reading the saved PRD. A mechanism without a named consumer → escalate before queueing.

---

## G2 — User-observable leaf

**Level:** **block** (decompose only; author-mode informs the decomposition plan but the hard check is at decompose time).

**What it catches.** Tasks marked done with load-bearing wiring absent — frontend ready but no backend event source, a trait wired but the actual walk stubbed, a task closed via a docs-only commit. The policy: every leaf task names a user-observable signal proving completion.

**Application.** For each task in the decomposition:
1. Classify: **leaf** (no other task in this batch depends on it) or **intermediate** (other batch tasks consume its output).
2. **Leaf tasks must declare a user-observable signal.** Generic menu:
   - CLI output difference (a command emits specific text / a diagnostic / an exit code).
   - API/service response difference (an endpoint returns a specific shape; a status changes observably).
   - Persisted-state change observable through the product's own read path (not by peeking at storage).
   - UI state change observable through the product (or a UI-driving harness).
   - A log line / metric / emitted event a user or operator can see.
   - A user-facing diagnostic code.
   - An example/fixture that exercises the new path and runs in CI.

   **[overlay]** may extend this menu with project-specific signal types (e.g. viewport state via a debug MCP, LSP hover content, a stdlib example in the project's own language).
3. **Intermediate tasks must declare which downstream prerequisites they unlock** — the consumer task ID or title. Producer-only intermediate tasks with no named downstream consumer are not acceptable.
4. The signal becomes the task's `user_observable_signal` metadata at filing.

If a leaf task's only "signal" is "a unit test passes against synthetic input", **reject**. That's the failure shape the gate exists to prevent — synthetic-input unit tests close cleanly while no user observes anything different.

**Escape hatch.** Foundation-style tasks that genuinely cannot demonstrate a user-observable signal in isolation are acceptable IFF they are roped into a paired **integration-gate task** within the same batch — the integration-gate task is the leaf, the foundation tasks are intermediates that unlock it. (This is the **C-as-integration-gate** pattern.)

---

## G3 — Assumed-substrate verified

**Level:** **block** (both modes).

**What it catches.** PRDs that assume a substrate capability which does not actually exist yet: a parser/grammar production, an API endpoint, a DB schema/migration, a config key, a CLI flag, a library function, a feature flag. The work is designed against a fiction and stalls when an implementer discovers the substrate isn't there.

**Application.** Enumerate every substrate capability the PRD's mechanisms / signals assume. For each, **verify it exists OR queue it as an explicit prerequisite task**:

1. **[overlay] If the overlay defines a substrate verifier**, run it. The canonical example is a *grammar gate* for a language/DSL project: extract each novel syntax fragment to a fixture and parse it (`tree-sitter parse --quiet`), exit 0 = pass. Other verifiers: "does this route exist in the router?", "does this column exist in the schema?", "does this flag parse?". The overlay specifies the command and pass/fail semantics and may ship a reference file (e.g. `references/grammar-gate.md`).
2. **If no verifier is defined**, do it manually: for each assumed capability, find concrete evidence it exists today (a definition, a test, a doc). If you can't, it's unverified.

Every unverified capability must be resolved before save/queue. Two valid resolutions:
- **(a)** Rewrite the PRD to use a capability that does exist; re-verify the rewrite.
- **(b)** Queue the substrate work as an explicit prerequisite task in the decomposition, make every dependent task `depends_on` it, and name it in `Pre-conditions for activating`.

Do **not** accept "the substrate will exist by the time this PRD activates" unless that work is filed and tracked as a hard prerequisite task in the DAG.

For PRDs that introduce **no novel substrate assumptions** (pure-infrastructure wiring of existing capabilities), G3 is a no-op — note "no novel substrate — G3 N/A" and move on.

---

## G4 — Cross-PRD seam ownership

**Level:** **prompt** (both modes).

**What it catches.** Contested-ownership seams: PRD A says "the integration is owned by B" while B says "owned by A", so neither decomposition holds the integration task and the seam never lands.

**Application.** For every cross-PRD reference in the PRD:
1. Identify the mechanism the seam owns (the function, event, file, or trait whose implementation crosses the boundary).
2. Ask the user: **which PRD owns the seam?**
3. The named owner gets the integration task in its decomposition. The other PRD references the seam-owner task as a dependency.
4. **Detect reciprocal ambiguity.** If this PRD reads "X is owned by the other PRD" while the other reads the same back, surface it. The user picks an owner; the other PRD gets updated in a paired commit OR a follow-up edit task.

Bookkeeping artifact: every saved PRD has a `## Cross-PRD relationship` (or equivalent) section with a table:

| Other PRD | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| `path/to/other.md` | consumes / produces | `Foo::bar()` / `Event::baz` | this-prd OR other-prd | wired / queued / blocked |

In author mode the skill drafts this table from conversation; in decompose mode it re-checks before queueing.

**[overlay]** may list known contested-ownership pairs to check against, so a PRD doesn't introduce a fourth instance of a known seam fight.

**Conditional adoption.** G4 only fires for PRDs that touch existing PRD territory or have load-bearing cross-PRD dependencies. Standalone foundational PRDs may not need the section — note "no cross-PRD seams" inline.

---

## G5 — Design-first when stakes are high (approach H)

**Level:** **prompt with heuristic** (author only; informational at decompose).

**What it catches.** Under the orchestrator's narrow-file-lock model, integration-step tasks that span crates/modules get starved or never get queued. Approach **B** (vertical slice) is fine for architecturally-simple features; approach **B + H** (contract document + interface tests + two-way boundary tests) is required for high-stakes / architecturally-complex features, so the integration is specified up front and lands as a first-class task rather than getting starved at medium priority.

**Heuristic.** A PRD needs **B + H** rather than bare B when **any of**:
- **Cross-module blast radius ≥ 3** (crates/packages/services touched).
- **Mechanism count ≥ ~8** — a coarse signal the PRD is too large to vertical-slice without a contract.
- **High stakes** — touches a load-bearing seam. **[overlay]** names the project's load-bearing seams; absent an overlay, use judgment (auth, persistence, the core domain engine, the parser, the public API).
- **Cross-PRD consumers ≥ 2** — multiple downstream PRDs assume this one's output.

Thresholds are **[overlay]**-tunable. When any condition holds, prompt: "This PRD looks B+H-shaped. Add a §contract section (seam signatures + invariants) and a §boundary-test sketch (cross-module scenarios facing both producer and consumer sides) before saving? Otherwise we accept the risk that integration tasks starve under the narrow-lock orchestrator." Default **yes for high-stakes seams**, **no for self-contained features**.

**What B + H adds to the PRD.**
1. A **contract section** — seam signatures, invariants, ordering rules, error semantics.
2. A **boundary-test sketch** — a table of scenarios with preconditions + postconditions, facing both the producer side and the consumer side.
3. The decomposition plan's integration-gate task names the boundary-test sketch as its observable signal — closing the loop into G2.

---

## G6 — Premise validity

**Level:** **block** (both modes; checked at PRD save and re-checked at decompose).

**What it catches.** A failure class orthogonal to G1–G5: a leaf signal whose **substantive premise** is false, unreachable, or misattributed. G1–G5 validate the *structure* of the implementation chain; G6 validates the *truth* of the claim embedded in the signal. A signal can pass every other gate and still assert something impossible. The danger zone is signals baked into a RED test — the false premise surfaces only when an implementer provably can't turn it green, costing an escalation and a planner-tier amendment.

**Application.** For every observable / leaf signal, classify its assertion and apply the matching check. Most signals — "emits diagnostic `E_*`", "compile test", "endpoint returns 200" — assert no quantitative premise and pass trivially.

1. **Numeric bound / threshold** ("within X%", "≤ ε", "≥ N", "to M digits"). Cite an *achievability basis*: an existing validated test/reference that already hits that accuracy on a comparable problem, OR a back-of-envelope error estimate for the method at the planned resolution, OR a reference computation. If none exists, the bound is a **guess** — set it to a defensible value, or mark it provisional and file a calibration task. **Reject bare guessed thresholds.** A fixture comment claiming "Tuned" is not a basis.

2. **Closed-form exactness / reproduction** ("exact within 1e-12", "reproduces P(t) exactly", "round-trips losslessly"). State the **mathematical identity** that makes it true, then confirm the asserted **configuration** satisfies it. Exactness is almost always configuration-dependent (boundary condition, element order, end conditions, basis degree) — name the configuration that earns it.

3. **End-to-end capability** ("produces a Mesh", "evaluates to `Value::X`", "the union renders"). Trace every capability the signal requires to the task's **dependency set**: each must be delivered by this task or one of its **prerequisites** — never by a task that **depends on** this one. If a required capability is owned by a downstream task, the signal belongs on that downstream leaf (the C-as-integration-gate pattern from G2), not here.

Branches 1 and 2 are **domain-weighted**: they fire heavily for numerical/scientific projects and rarely for CRUD/web/tooling projects. **[overlay]** sets the project's domain flag and supplies domain-specific premise hazards (e.g. FEA element locking, spline end-conditions). Branch 3 is **universal** — it's a dependency-correctness check that applies to every project.

**Resolution when a premise fails:**
- **(a)** Move the signal to the task that can actually produce it (fixes misattribution).
- **(b)** Weaken the assertion to what's achievable now, file a follow-up for the stronger property.
- **(c)** Change the asserted configuration so the claim becomes true.

---

## META — "is this PRD good?"

**Level:** **block** (author mode only; the final check before saving).

**What it catches.** A structural-headers-all-present PRD can still be incomplete if it leaves load-bearing **design** questions undecided.

**Application.** Before writing the PRD to disk, ask:

> If I decompose and queue this PRD without further oversight, will the architecture of what gets implemented be complete, coherent, cohesive, and **good**?

If not, identify the open **design** questions and resolve them inline. Tactical/implementation-time questions go in `## Open questions`. The boundary:
- **Design** — if a downstream architect could choose differently and arrive at an architecturally inferior result. Resolve now.
- **Tactical** — local, recoverable; an architect could pick either and the system stays coherent. Defer to `## Open questions`.

When unsure, ask the user: "design-level or tactical?" Default toward design-level.

---

## Gate-application order (author mode)

Walk in this rough order; iterate freely as discussion surfaces new mechanisms:

1. **G1 first.** Establish who consumes this; otherwise the rest is exercise.
2. **G3 second.** If a novel substrate assumption fails verification, drop / queue it before designing further.
3. **G4 third.** Identify cross-PRD seams; resolve ownership before writing the relationship table.
4. **G5 fourth.** Decide B vs B+H; if H, draft contract + boundary-test sketch now (they shape the decomposition).
5. **G2 in the decomposition plan** — name an observable signal per task even though the hard check is at decompose time.
6. **G6 alongside the G2 draft** — validate each drafted leaf signal's substantive premise.
7. **META last.** Final sanity check before save.

## Gate-application order (decompose mode)

1. **G1, G3, G4 re-check** against the saved PRD (fast; mostly drift detection).
2. **G2 walk** — enumerate every task, classify leaf/intermediate, attach `user_observable_signal` / `consumer_ref`.
3. **G6 re-check** — validate each leaf signal's premise. Escalate before filing if one can't be substantiated (cheaper than an implementer discovering it against a RED test).
4. **G5 informational** — note B vs B+H; if B+H, verify the integration-gate task exists and points at the boundary-test sketch.
5. File the batch (see `decompose-mode.md`).
