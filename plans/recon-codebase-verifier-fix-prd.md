# PRD — Fix the reconciliation codebase verifier (correct root + turn cap)

**Status**: active · 2026-08-24 · source: esc-3241-4 L2 brief
(`/home/leo/.claude/spawn-briefs/prd-recon-verifier-fix-2026-08-24.md`)

> **Code anchors** verified against main `ea876cb624` (2026-08-24). Main
> moves fast — cite-by-symbol; re-locate lines at implementation time.

## Goal

`fused-memory`'s reconciliation codebase verifier — the agent spawned when a
task transitions to `done` and memory about it is sparse
(`TargetedReconciliation._on_task_done`, the `len(related) < 2` gate) —
currently fails on **100% of invocations** (`verify|codebase|error` = 6/6
rows since the task-4343 audit landed 2026-08-19) and, were it working,
would read the **wrong repository on 58% of invocations** (46 of 79
historical gate openings were for non-dark_factory projects, while
`CodebaseVerifier` resolves one global `explore_codebase_root` =
`$PROJECT_ROOT` = `/home/leo/src/dark-factory`).

Leo's ask, verbatim in intent: *"fix the code verifier by both increasing
the max turns to a reasonable cap that should catch a runaway agent but not
prevent any reasonable execution and by making sure that it's pointed at
the correct code."*

After this PRD lands:

- The verify-outcome census (`SELECT operation, COUNT(*) FROM run_actions
  WHERE action_type='verify' AND operation != 'post_verify_error' GROUP BY 1`
  on `data/reconciliation/reconciliation.db`) stops being 100% `error`:
  real `confirmed`/`contradicted`/`inconclusive` rows appear.
- A verify invocation for a reify (or any non-dark_factory) task resolves
  evidence against **that project's** tree — or produces an **audited,
  structured refusal**, never a wrong-tree verdict.
- A `contradicted` verdict reaches a consumer that can act on it (an L1
  escalation in the target project's own queue), and the memory it writes
  no longer opens by asserting completion.

## Background

The verifier chain: `TargetedReconciliation._on_task_done` (gate + audit
row + memory write, `reconciliation/targeted.py`) → `CodebaseVerifier.verify`
(`reconciliation/verify.py`) → `AgentLoop` (`reconciliation/agent_loop.py`,
sole production caller: verify.py). Blast radius of `agent_loop.py` changes
is this feature only (grepped: one construction site).

**Limb A premise (validated, regression):** `AgentLoop._call_claude_cli`
passes `max_turns=1` alongside `output_schema=CLAUDE_CLI_RESPONSE_SCHEMA`,
with a comment claiming schema tool-use completes within the same turn.
Measured false: the model stochastically emits a prose turn before calling
the synthetic `StructuredOutput` tool; at a cap of 1 the CLI returns
`subtype='error_max_turns'` with no payload, nothing is salvaged, and
`_call_claude_cli` raises. Regression introduced by `c2eaa95586`
(2026-04-21), which flipped `max_turns=3 → 1` in both `agent_loop.py` and
`judge.py`; task 3067 restored `judge.py` (`_JUDGE_CLI_MAX_TURNS = 3`);
`agent_loop.py` is the last unfixed site, and task **3241** (branch ready,
audited) exists for exactly that. Evidence, four independent lines: (a) the
production census (5 of 6 error rows carry `error_max_turns`, salvage never
fired); (b) live probe on CLI 2.1.241 driving the real production shape:
`mt=1 → 0/3`, `mt=10 → 3/3` (known-positive control); (c) earlier
measurements on 2.1.236/2.1.233: `mt=1 → 0/6` stable across three CLI
versions, intermediate rates stochastic; (d) positive control from
production history — 8 verdict memories across 2 tasks, both **before** the
2026-04-21 regression, proving the whole downstream chain
(`_CLIResponseAdapter` → `extract_agent_verdict` → `VerificationResult` →
memory write) has worked end-to-end.

Census caveat carried forward: `num_turns` as reported by the CLI is a
post-hoc counter, **not** the counter `--max-turns` bounds (successful runs
have reported `num_turns` above the cap). The load-bearing failure signals
are `subtype='error_max_turns'` plus salvage-never-fired. No comment
landed by this PRD may claim otherwise.

**Limb B premise (validated):** `CodebaseVerifier.__init__` resolves
`config.explore_codebase_root` once, globally; `verify()` receives no
project scope. One fused-memory process serves every project slug from one
`reconciliation.db`; the live process env pins `PROJECT_ROOT` to the
dark-factory checkout. A verifier hunting a reify task's evidence in the
dark-factory tree finds nothing and can return `contradicted` — writing a
Mem0 memory falsely asserting a genuinely-completed task's work is absent.
That is memory-corpus poisoning, and agents search that corpus.

**Prior decision this PRD reverses:** task 2548 (done) item 2 found
`CodebaseVerifier.verify()`'s then-unused `project_id` parameter and offered
"wire it up (resolve codebase_root per task project_id …) or delete the
dead parameter". Deletion was chosen. 2548 framed it as dead code; what was
actually missing was the *wiring*, and the deletion made the scope bug
unfixable-by-inspection. This PRD reinstates the wiring — with the root
passed by the caller rather than re-resolved from a registry, see D3.

## Resolved design decisions

**D1 — Ordering: limb B lands at or before limb A (hard dependency).**
Limb A *activates* the verifier; today it is 100% dead, so the wrong-root
bug is inert. Fixing the cap first would ship a verifier that reads the
wrong repository on 58% of invocations and can write `contradicted`
memories against completed work — strictly worse than today's loud
failure. Encoded as real `add_dependency` edges (task 3241 depends on α
and β), never as prose.

**D2 — Limb B shape: thread project scope through (brief option a).**
Rejected: (b) gate-the-branch — leaves 58% of the population permanently
unverified, and "skip silently unless audited" is the exact INV-2 shape
task 4343 existed to kill; (c) retire the path — a legitimate outcome on
the consumer numbers, but Leo explicitly asked for the fix, and D7 names
real consumers, so retirement is not taken (see Out of scope).

**D3 — The root comes from the caller's `ProjectScope`, not a registry
lookup inside the verifier.** `TargetedReconciliation.reconcile_task`
already receives a validated per-task `project_root`
(`require_project_root` → `ProjectScope(ProjectId(...), ProjectRoot(...))`)
and every handler holds `scope`. `_on_task_done` passes
`codebase_root=Path(scope.project_root)` to `verify()` as a required
keyword argument. `CodebaseVerifier.__init__` stops resolving
`explore_codebase_root` entirely (the config key remains for the other
task-1989-adjudicated call sites: judge, stages, cli_stage_runner —
out of scope here). One authority for the root, no second resolver to
drift (INV-9).

**D4 — Fail closed, loudly and structured.** Before spawning the agent,
`verify()` validates the root with cheap stat checks (directory exists and
`<root>/.git` exists — no subprocess on the event loop, INV-8). On failure
it returns
`VerificationResult(verdict=inconclusive, agent_failed=True,
failure_token='codebase_root_unresolved', summary=<offending path>)` —
which rides the existing task-4343 machinery unchanged: `_on_task_done`
writes an audited `verify|codebase|agent_failed` outcome row carrying the
token, and **no memory is written**. Invariant: a wrong or unresolvable
root can never produce `contradicted` (or any verdict-bearing memory).
The refusal is a census-visible row, not a silent skip.

**D5 — The agent's cwd follows the target root.** `AgentLoop` gains a
`cwd` constructor parameter; `verify()` passes `codebase_root`. Task 1989
deliberately kept the CLI's cwd at the codebase root so the auto-loaded
CLAUDE.md is the agent's passive codebase signal — that rationale now
demands the **correct project's** CLAUDE.md, not dark-factory's. The
`.mcp.json`-exposure mitigations at that call site
(`disallowed_tools=['*']` expanded to the built-ins denylist, plus
`no_mcp_servers_config()` + `strict_mcp_config=True`) are cwd-independent
and cover any target project's checkout the same way. `cwd` defaults to
`None` → falls back to `config.explore_codebase_root`, preserving existing
test construction; the sole production caller always passes it.

**D6 — Per-invocation cap = 10, by adopting branch `task/3241`.** The cap
is a per-CLI-invocation ceiling, not a target: `AgentLoop.run()` drives
multi-turn *externally* (`agent_max_steps: 50` is the separate
conversation budget), so one invocation needs room for at most one prose
turn plus the `StructuredOutput` call — headroom is free when unused.
Runaway protection is layered: spend is bounded by `invoke_with_cap_retry`'s
default `$5.00/invocation` budget, wall-clock by
`agent_cli_timeout_seconds` (180s), and conversation length by
`agent_max_steps`. 10 is established precedent (curator batch path caps at
10; curator default 8; judge and path-scope adjudicator floor at `ge=3`)
and measured green (`mt=10 → 3/3` on 2.1.241, `6/6` on 2.1.236). The
existing branch `task/3241` (tip `0c8459f273`, audited 2026-08-24: one
behaviour change `max_turns=1 → _AGENT_CLI_MAX_TURNS = 10`, corrects the
false claim at all 7 comment copies across 5 files, adds
`fused-memory/scripts/probe_schema_max_turns.py`, regression test asserts
`max_turns >= 3` and fails on revert) is **adopted, not superseded or
re-implemented**. Its landing folds in one residual wording fix from the
branch audit: the task_curator.py comment's "nothing to salvage … simply
hard-fails" sentence gets scoped to the measured verify shape (its own
hedge three lines later already admits the curator's shape was not
re-measured).

**D7 — Consumers (G1).** Two, both real today:

1. **The memory corpus, searched by agents.** `observations_and_summaries`
   verdict memories are read via semantic search by every session and
   dispatched agent working these projects. This consumer imposes a
   *quality* obligation this PRD takes on (β): the current single content
   template opens **both** verdicts with `Completed task '<title>': …`, so
   a `contradicted` verdict writes a memory asserting completion and hopes
   Mem0's fact extraction strips the framing. β replaces it with
   verdict-specific templates — confirmed: `Verified completion of task
   '<title>' against the codebase: <summary>`; contradicted:
   `Codebase evidence CONTRADICTS the completion claim of task '<title>':
   <summary>` — and hardens `EXPLORE_AGENT_SYSTEM_PROMPT` to state that the
   summary becomes a permanent memory record: cite concrete repo evidence
   (paths, symbols), never narrate the agent's own tooling or process (the
   historical `confirmed` set is tool-narration noise; the `contradicted`
   set is line-cited and genuinely useful — keep the latter shape).
2. **The per-project escalation ladder, for `contradicted`.** β files an
   L1 escalation into the **target project's** own queue
   (`<project_root>/data/escalations`), mirroring the in-file precedent
   `TargetedReconciliation._sweep_escalate_l1`: `agent_role='reconciler'`,
   `severity='info'`, `level=1`, existing `category='risk_identified'`
   (deliberately **not** a new category — the Escalation model's category
   comment makes the next addition promote the vocabulary to an enum;
   this PRD does not take that on), `suggested_action='reopen_task|create_followup_task|dismiss'`.
   Detail carries **pointers, not copies** (INV-9): task_id, run_id,
   verdict + confidence, and the top evidence paths; the memory record and
   the census row remain the finding's homes. The consumer surface is the
   existing L1 → escalation-watcher-auto → L2 human ladder. Filing is
   guarded by the existing `_HAS_ESCALATION` import guard and fires only on
   `verdict == 'contradicted' and not agent_failed`.

   `confirmed` keeps its memory write (with the honest template): the gate
   selects tasks with *sparse memory*, and an evidence-citing confirmation
   is the backfill the path was built for. No verdict changes task status —
   nothing auto-closes or auto-reopens on an LLM verdict (esc-3105-3 pin;
   INV-3: the escalation is an alert for a human, not an action).

**D8 — Post-landing soak gate (ε).** This repo's measured failure mode is
"landed, tested, deployed — and inert" (gate 3841's fix, the 2447 guard,
three enabled-and-blind guards). ε is a human-gated operational check that
the census actually moved: at least one non-`error` outcome row, and at
least one non-dark_factory invocation that either resolved evidence
against its own tree or refused with `codebase_root_unresolved`. The
census query is quoted in the task; `trigger_reconciliation` on a chosen
sparse done task is the documented fast path instead of waiting for
organic traffic (~17 sparse-done events/month fleet-wide).

**D9 — Follow-up sequencing.** Task 4463 (retract the three memory-corpus
records asserting the refuted max_turns mechanism) gains a dependency on
3241: its retraction text cites `_AGENT_CLI_MAX_TURNS = 10` as landed
fact, which is only true after the branch merges.

## Pre-conditions (G3 — all verified on main `ea876cb624`)

- Task 4343's audit machinery (one `verify|codebase|<outcome>` row per
  invocation, `failure_token` plumbing) — done, on main since 2026-08-19.
  It is the measurement instrument for every signal below.
- Task 3067 (judge restored to 3 turns on the shared structured-output
  path) — done.
- `ProjectScope` with validated per-task `project_root` at every
  `_on_task_done` call site — on main (`reconcile_task`).
- Escalation substrate importable from targeted.py behind
  `_HAS_ESCALATION`; file-based `EscalationQueue` per project root — on
  main (`_sweep_escalate_l1` is the production precedent).
- Branch `task/3241` exists, merges clean onto main
  (`git merge-tree --write-tree` rc=0 re-checked 2026-08-24), merge-result
  suite green (461 passed across the three affected test files).
- `done_provenance`/`completion_fast` fast path (the deterministic sibling
  answering "was this really done?" from the merge SHA) — live, 9,122 rows.

No novel substrate is assumed beyond mechanisms this PRD itself introduces
(the `cwd` parameter, the refusal token, the escalation call) — each has
its named consumer above.

## Cross-task relationships (G4)

No cross-PRD seams; the seams are existing tasks:

| id | status | relationship |
|---|---|---|
| 3241 | blocked | Limb A's implementation, branch ready. **Adopted** as this PRD's δ — deps re-wired, details amended; not duplicated. Stays blocked until esc-3241-4 is resolved by its L2 session. |
| 4344 | pending | Adjacent (residual `error_max_turns` at any cap). Its "~20% residual" premise was **not reproduced** on CLI 2.1.236/2.1.241 (`mt=10 → 6/6`, `3/3`) — re-measure with the probe script and the post-landing census before treating it as live. Untouched by this batch. |
| 4463 | pending | Corpus retraction of the same refuted claim. Sequenced after δ (D9). |
| 2548 | done | Item 2 deleted the `project_id` parameter this PRD's α re-wires (as a root argument). Reversed with rationale — see Background. |
| esc-3241-4 | pending | Owned by the L2 session that produced this PRD's brief. **This batch must not resolve it.** |

## Contract (compact)

- `CodebaseVerifier.verify(claim, context='', scope_hints=None, *,
  codebase_root: Path) -> VerificationResult`. The caller
  (`_on_task_done`) passes `Path(scope.project_root)`. No default.
- Root validation precedes any agent spawn: not-a-directory or missing
  `<root>/.git` ⇒ structured refusal
  (`agent_failed=True, failure_token='codebase_root_unresolved'`); stat
  checks only, no subprocess.
- **A wrong or unresolvable root can never produce `contradicted`**, nor
  any verdict-bearing memory write. Pinned by test.
- `AgentLoop(config, system_prompt, tools, terminal_tool, cwd=None)`;
  `cwd=None` falls back to `config.explore_codebase_root`; verify.py
  always passes `codebase_root`.
- Escalation fires only on `contradicted` ∧ `not agent_failed`, into
  `Path(scope.project_root)/'data/escalations'`, category
  `risk_identified`, severity `info`, level 1; escalation-write failure is
  caught and logged, never breaks the reconciliation run (mirrors
  `_sweep_escalate_l1`).
- Memory content templates are verdict-specific (D7.1 wording anchors);
  the shared `Completed task '<title>'` template is retired.

## Decomposition plan

| # | task | kind | deps | user-observable signal |
|---|---|---|---|---|
| α | Thread per-task codebase root into `CodebaseVerifier.verify` + `AgentLoop` cwd; fail-closed refusal | normal, high | — | Intermediate (unlocks δ, ε). Tests pin: tool closures + agent cwd resolve against the per-call root; unresolvable root ⇒ `codebase_root_unresolved` refusal, no memory write, audited row. Production signal owned by ε (verifier still dead until δ). |
| β | Verdict-specific memory templates + prompt hardening + `contradicted` → L1 escalation in target project's queue | normal, high | α | Intermediate (unlocks δ, ε). Tests pin: contradicted memory opens with the contradiction template; escalation lands in a real `EscalationQueue` under a tmp project root with category/severity/level per contract; confirmed/inconclusive file nothing. |
| γ | Correct the `schema_salvaged` docstring in `shared/src/shared/cli_invoke.py` (upstream ancestor of the false claim) | normal, simple, low | — | Leaf. The docstring no longer claims `error_max_turns` "commonly" arrives paired with a completed payload (measured: it almost never does); grep-checkable, see manifest. |
| δ | = existing task 3241 (adopted): land branch `task/3241` (cap 10 + 7 comment corrections + probe script), rebasing over α/β, folding in the task_curator.py hedge reword | existing | α, β | Census stops being 100% `error`: first `confirmed`/`contradicted`/`inconclusive`/`agent_failed` row. Regression test `max_turns >= 3` green (fails on revert). |
| ε | Soak gate: verify-census shows the verifier live and correctly scoped | operational gate, medium | δ | Leaf (integration gate). The quoted census query returns ≥1 non-`error` outcome row, and ≥1 non-dark_factory invocation shows evidence from its own tree or an audited `codebase_root_unresolved` refusal. Fast path: `trigger_reconciliation` on a chosen sparse done task. |

Out-of-batch wiring: 4463 → depends on δ (D9).

G7 walk: recorded in the capability manifest. One waiver —
`G7 waiver: storm-escape-required` on α's refusal path: the refusal is
rate-bounded by construction (it can fire at most once per sparse-done
gate opening, ~17/month fleet-wide — there is no volume for a streak
counter to count) and each firing is a census-visible audit row; ε checks
the census once post-landing. All other invariants: INV-2 satisfied by
D4's structured token; INV-3 by the alert-not-action shape of D7.2;
INV-5 — the turn-cap comment corrections are prose at 7 pre-existing
copies (branch-landed shape, not extended); INV-8 by D4's stat-only
checks; INV-9 by D3 (one root authority) and D7.2 (pointers, not copies).

## Out of scope

- **Retiring the verify path** (brief option c). Rejected per D2. If ε's
  soak shows the activated verifier producing only noise, retirement is
  the follow-up to file — with the branch's comment corrections kept
  regardless.
- **Per-project cwd for judge / stages / cli_stage_runner.** Those sites'
  cwd was adjudicated by task 1989 (sweep verdict) and they are not
  verdict-writing verifiers; re-opening that is a separate design
  question.
- **Task 4344's residual-failure work** — premise currently unreproduced;
  the post-landing census is the instrument to re-check it with.
- **Resolving esc-3241-4** — owned by its L2 session.
- **Escalation category vocabulary enum-promotion** (the model's stated
  refactor trigger) — deliberately avoided by reusing `risk_identified`.
- **Retro-fixing the 4 pre-regression verdict memories** — 4463 covers
  the refuted-mechanism records; the 2026-04 verdict memories are honest
  history.

## Open questions (tactical)

1. **Wall-clock at cap 10 vs `agent_cli_timeout_seconds=180`.** Raising
   the cap converts a fast `error_max_turns` failure (4.5–24.5s observed)
   into a possible 180s timeout on a pathological run. Unmeasured;
   accepted open — it fails loud either way (timeout ⇒ audited error
   row). Watch the first production rows' latency during ε; tune the
   timeout only if the census shows timeouts.
2. **Escalation confidence threshold.** β files on every `contradicted`
   regardless of confidence (volume is gate-bounded and tiny; the L1
   watcher triages). Revisit only if the ladder sees noise.
3. **`AgentLoop` cwd parameter required vs defaulted.** Defaulted per D5
   for test compatibility; an implementer may tighten to required if the
   test churn is acceptable at rebase time.
