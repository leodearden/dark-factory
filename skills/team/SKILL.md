---
name: team
description: "Run a piece of work as an agent team — decompose it into seats, route every seat to the model tier (haiku / sonnet / opus / fable) and reasoning effort (low → max) its brief actually rewards, dispatch through the Workflow tool (or the Agent tool for small, adaptive teams), then integrate and verify the results as the lead. ALWAYS use this for: /team commands (/team --fable pre-authorises fable seats for the run), 'use an agent team', 'fan this out to agents', 'parallelise this with subagents', 'run this as a team and use cheap models where you can', 'put haiku/sonnet on the grunt work', or any request to do work with several agents at task-appropriate model and effort levels. NOT for: single-chain work you can do inline, the purpose-built multi-agent surveys (/hotspot-survey), handing a decision to a fresh context (/do), or the dark-factory orchestrator (/orchestrate)."
argument-hint: "[--fable] [task — omit to use the direction just agreed in this conversation]"
---

# /team — route the work to the right seats, then lead

**Task:** $ARGUMENTS

> If the line above is empty, the task is whatever direction was just agreed in this conversation. Infer it; don't ask the user to restate it.
>
> A leading `--fable` token is not part of the task: it pre-authorises fable seats for this run (see *Fable — who may route to it*). Strip it before reading the rest.

A team exists to spend the *right* model and the *right* amount of thinking on each part of a job, in parallel where the job allows it, instead of running everything through this session's own model at this session's own effort. You are the lead: you scout, decompose, route, integrate and verify. Seats do the bounded work. The `/team` invocation is the user's explicit opt-in to multi-agent orchestration — the Workflow tool's opt-in rule is satisfied by it.

## 1. Scout inline — just enough to decompose

Read what you need to see the *shape* of the work: the files or subsystems involved, the natural work-list (files, modules, findings, questions), what is mechanical and what needs judgment, what can run concurrently without touching the same files. Stop when more reading stops changing the roster. Don't delegate this step — a roster built on guesses is the most expensive mistake a lead makes, and a seat cannot see the conversation that produced the task.

Two outcomes are legitimate here and you should say so plainly when they occur:

- The task is one sequential chain with no parallel or context-heavy part. Then the win, if any, is routing alone: run it as a team of one well-routed seat, or do it inline. A team is not a goal.
- The task is ambiguous in a way that changes the roster and a human is present: ask once, then proceed.

## 2. Build the roster

One row per seat. Post it before launching so the user sees what is about to run and what it costs (a seat at sonnet / medium–high runs ~50–150k tokens and 5–20 minutes; the roster's cost is roughly seats × 100k). Launch without waiting for a reply unless the roster is large (≳10 seats or ≳1M tokens) or step 1 left a real question open.

| Seat | Brief (one line) | Deliverable | Model | Effort | Isolation | After |
|---|---|---|---|---|---|---|

- **Deliverable** is a shape, not a hope: a JSON schema for Workflow seats, a fenced-block layout for Agent-tool seats. Seats return raw results for you to integrate, never a narrative for the user.
- **Isolation** is `worktree` only for seats that mutate files while other seats are also mutating files. It costs real setup time and disk; a read-only seat never needs it.
- **After** names the seat(s) whose output this one consumes. Prefer pipelining over barriers — a verifier for finding A should start while B is still being reviewed.

### Routing — the model axis

Pick the tier by the **judgment density** of the brief and the **cost of a plausible-but-wrong answer**, not by how important the overall task feels.

| Tier | Route here | Don't route here |
|---|---|---|
| `haiku` | Mechanical and fully specified, cheap to check: inventories, greps, format conversions, applying an already-settled pattern across many sites, summarising one bounded document, running a command and reporting its output. | Anything with a judgment call, and anything whose output flows downstream unchecked. **Haiku ignores effort** — a brief that needs thinking cannot be rescued by an effort setting here. |
| `sonnet` | Routine engineering with a clear spec: a scoped change with tests, tests for known behaviour, a first-pass diff review, checking one concrete claim against the code, a research sweep over a bounded corpus. The default worker tier. | Open-ended design, root-causing, synthesis across many inputs. |
| `opus` | Judgment-dense work: design choices, root causes, architectural review, cross-input synthesis, adversarial verification of subtle claims — wherever plausible-but-wrong is expensive. | Bulk mechanical work — split the mechanical part out to a cheaper seat rather than paying opus rates for it. |
| `fable` | Where it is *likely* to be the optimum, not only where that is proven: synthesis of conflicting complex reports, and adversarial review of such a synthesis; difficult design judgment and architectural thinking; open-ended, high-ambiguity work — problem framing, open-world exploration and orientation, strategy, project planning — where a slightly better intermediate result compounds into a much better outcome; and anywhere opus has already struggled. On this work the cost of a wrong answer is orders of magnitude above the cost of the seat, so under-routing is the expensive mistake. | Mechanical or well-specified roles — review-by-checklist, verification of a concrete claim, transformation, inventory — where opus or sonnet returns the same answer at a fraction of the price. And never by inheritance: see below. |

Every seat gets an explicit model. The Workflow reference advises omitting `model` so a seat inherits the session model — for `/team` that advice is inverted, because routing is the point. An unset model under a fable lead bills every mechanical seat at fable rates: one heavyweight check workflow left with all roles on the session default burned a week of fable credit in a day, for answers sonnet would have given verbatim. Inheriting is a deliberate choice for a seat where the session model is the right tier, never a default.

### Fable — who may route to it

Whether you may put a seat on `fable` on your own judgment depends on what *you* are running on (your system prompt names your model):

- **A fable lead routes fable seats freely**, under the criteria above. Being in a fable session is itself the user's signal that the problem as a whole deserves fable-level thinking, and the lead has the best available judgment for where within it that pays.
- **An opus (or lower) lead proposes; it doesn't presume.** Route a seat to fable only with clear direction that it is appropriate here — the `--fable` token on the invocation (the explicit form: fable seats are pre-authorised for this run, still under the criteria above), a statement in the conversation, or the project's own guidance. Without that, mark the seat in the roster as `proposed: fable (else opus / xhigh)` and ask before launching when a human is present; unattended, run the fallback and say in the report that a fable seat was wanted and why. The user is generally glad to consider the request — what they don't want is fable appearing as a matter of course.

### Routing — the effort axis

Pick effort by **how much reasoning the brief rewards**, independently of the model.

| Effort | Route here |
|---|---|
| `low` | Lookups, mechanical edits, running commands, reformatting, applying a decided pattern. |
| `medium` | Routine implementation, standard review, checking a concrete claim against the code. |
| `high` | Design, debugging, synthesis, adversarial verification of anything subtle. |
| `xhigh` / `max` | The hardest reasoning: tricky concurrency, a proof-shaped argument, a root cause that survived a `high` attempt. `max` only when a wrong answer costs far more than the tokens. |

The axes are orthogonal — don't couple them reflexively ("opus so high, sonnet so low" throws the second axis away). `sonnet` at `high` is the right seat for a bounded problem that rewards thinking; `opus` at `low` is the right seat for a quick call that needs strong priors. A level the model doesn't support falls back to the highest supported level at or below it.

### Shapes worth reaching for

- **Find → verify**: finders at `sonnet`/`medium`, one skeptic per finding at `sonnet`/`medium` prompted to *refute*, pipelined; a synthesis seat at `opus`/`high`, or `fable` when the findings conflict and the stakes are design-level (provenance rule applies). Zero refutations is a yellow flag, not a clean bill.
- **Judge panel**: N independent attempts at `opus`/`high` from different angles, scored by parallel judges, synthesised from the winner — the synthesis is the seat most likely to earn `fable`. Beats one attempt iterated when the solution space is wide.
- **Frame → explore → decide**: for open-ended or strategic work, a `fable`/`high` seat frames the problem and names the questions, cheap seats gather evidence against each question, and a `fable` seat decides — spend the expensive thinking at the two ends where a better intermediate compounds, not in the middle.
- **Mechanical sweep**: discovery at `haiku`, transformation at `sonnet`/`low`–`medium` with worktree isolation, one `sonnet`/`medium` seat verifying the merged result.
- **Understand**: parallel readers at `sonnet`/`medium` over disjoint subsystems returning a structured map; you synthesise.

## 3. Dispatch

**Default: the Workflow tool.** It is the only mechanism that takes model *and* effort per seat in one call, and it gives you pipelining, structured-output schemas and a resumable journal. Load `workflow-authoring` for the script API, then write the script inline. Set `{model, effort}` on every `agent()` call from the roster; put the phase title on each call so the progress tree reads like the roster.

**Use the Agent tool instead** when a seat needs *this conversation's* context (`subagent_type: "fork"`), when you expect to continue a seat by `SendMessage`, when the team is one to three seats with no pipeline shape, or when the Workflow tool is unavailable. Effort cannot be set per Agent-tool call, so route effort through the seat types that carry it in their definition — `team-low`, `team-medium`, `team-high`, `team-xhigh`, `team-max` (`agents/` beside this file, symlinked into `~/.claude/agents/`) — and pass `model` on the call; the call's model wins over the definition's. Built-in types (`Explore`, `Plan`) are fine seats when their tool restrictions fit; they take `model` but carry their own effort. Launch independent seats in one message so they run concurrently, and record `total_tokens` / `duration_ms` from each completion notification as it arrives — it is the only time that data exists.

**The brief.** A seat starts from zero context, so the brief is its whole world. Every brief carries:

1. the goal in one sentence, and why it matters to the whole (so the seat can tell a blocker from a detail);
2. exact inputs — paths, symbols, commands, prior seats' outputs — never "the module we discussed";
3. the deliverable's exact shape;
4. boundaries: what not to touch, and an instruction to report an under-specified brief or a blocker precisely and stop rather than guess or widen scope;
5. the verification the seat must run before returning;
6. cross-file references as `path/to/module.py::symbol`.

Seats receive the same CLAUDE.md files you did (built-in `Explore`/`Plan` excepted) — name the one rule a brief depends on rather than pasting rules.

## 4. Integrate and verify — you, not a seat

A seat's output is evidence about what that seat saw, not truth. Read every result; spot-check anything that changes the answer; route correctness-critical claims through a skeptic seat or verify them inline yourself. A structured output that passed its schema can still be degenerate (empty evidence, placeholder text) — check it semantically before it feeds another seat, and re-run a degenerate seat once before dropping its lane and saying so.

Large results (a Workflow return or a completion notification arrives truncated past a few hundred KB) go to the scratchpad and get digested with a small script — never ingested raw. Merge worktree seats' changes yourself and run the project's real verification (tests, lint, type-check) on the merged result, not on each seat's branch.

## 5. Report

End with the roster-and-outcome table — seat, model, effort, tokens and minutes where known, one-line outcome — followed by what was dropped, unverified or degraded (a lane that failed twice, a seat skipped, a claim you could not check). This table is what lets the user recalibrate the routing next time: a `sonnet`/`medium` seat that had to be redone at `opus`/`high` is a routing lesson worth more than the task's own answer. Then give the task's answer as you would for any other piece of work.
