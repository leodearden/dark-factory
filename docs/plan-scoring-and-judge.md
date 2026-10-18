# Plan: ε composite-score fix + ζ completion-judge exit condition

**Read this first, then start.** This file is a self-contained prompt for
a new Claude Code session in `/home/leo/src/dark-factory`. It does not
assume you've read `docs/vllm-eval-status.md`, though that doc has fuller
historical context if you want it.

## Context: what this fixes

vLLM-hosted local models (REAP-139B-NVFP4, MiniMax-M2.5-FP8, etc.) in the
orchestrator eval pipeline consistently write correct code that passes
tests/lint/typecheck gates but never produce `outcome=done` with a
non-zero composite score. Two compounding bugs cause this:

**Bug ε — scoring.** `compute_composite()` in
`orchestrator/src/orchestrator/evals/metrics.py:60` multiplies quality by
`plan_completion_pct`, which is `(steps with status=done) / total_steps`.
Local models don't maintain `plan.json` step statuses — they write real
code but never flip the status fields — so `plan_completion_pct=0.0` for
every vLLM run, driving the composite score to 0 even on successful
`done` outcomes.

**Bug ζ — workflow exit condition.** `_execute_iterations()` in
`orchestrator/src/orchestrator/workflow.py:748` loops
`while self.artifacts.get_pending_steps():` — it only breaks when all
plan.json step statuses become `done`. Local models pass the actual
verify gates but skip the plan.json bookkeeping, so the loop runs to
`max_execute_iterations` (20) and returns `WorkflowOutcome.BLOCKED`. The
task is genuinely complete; the workflow just can't tell.

ε is a 5-line scoring fix that unblocks the existing result corpus
without changing behavior. ζ is a structural fix: add a "completion
judge" LLM that runs after each implementer iteration and decides
whether to exit the loop based on the *actual code*, not plan.json.

**Order of operations is strict: land ε first, then ζ.** ε is tiny and
independent. Re-score the existing corpus with ε, sanity-check the
numbers, update `docs/vllm-eval-status.md`, then start on ζ.

---

## ε — scoring fix

### Change

**File:** `orchestrator/src/orchestrator/evals/metrics.py`
**Function:** `compute_composite(m: EvalMetrics) -> float` at line 60

Current body (lines 68-74):

```python
if not m.tests_pass:
    return 0.0
steps = max(m.plan_steps, 1)
blocking_rate = m.review_blocking_issues / steps
quality = 1.0 - (blocking_rate * 2.0) - (m.debug_cycles * 0.05)
quality = max(quality, 0.0)
return round(quality * m.plan_completion_pct, 4)
```

Replace the final line with:

```python
quality = min(quality, 1.0)
return round(quality, 4)
```

Update the docstring above (lines 61-67) to reflect that score is now
pure quality, bounded 0..1. Note in the docstring that
`plan_completion_pct` is still collected as a diagnostic signal (visible
in result JSON) but no longer gates the composite.

Leave `plan_completion_pct` and `plan_steps` on the `EvalMetrics`
dataclass (lines 26-27) unchanged. Don't delete fields.

### Backfill existing results

Write a small script at
`orchestrator/scripts/backfill_composite_scores.py` (new file, new
directory if needed) that:

1. Globs `orchestrator/src/orchestrator/evals/results/*.json`.
2. For each file: loads the JSON, extracts the `metrics` dict.
3. **Filters the metrics dict to known `EvalMetrics` fields** before
   reconstruction — old files may have extra/missing fields. Use the
   same pattern as the existing `EvalResult` loader at
   `orchestrator/src/orchestrator/evals/runner.py:361-362`:

   ```python
   import dataclasses
   from orchestrator.evals.metrics import EvalMetrics, compute_composite

   known = {f.name for f in dataclasses.fields(EvalMetrics)}
   clean = {k: v for k, v in metrics_dict.items() if k in known}
   em = EvalMetrics(**clean)
   new_score = compute_composite(em)
   ```
4. Writes back: `data['metrics']['composite_score'] = new_score`. Use
   `json.dump(data, f, indent=2)` to preserve pretty-printing.
5. Prints a summary: N files scanned, M files updated, K files unchanged.

Don't mutate fields other than `composite_score`. Don't add new fields
to existing result files.

### ε acceptance criteria

1. Unit sanity check (can be a quick REPL, no test file required):
   ```python
   m = EvalMetrics(tests_pass=True, plan_completion_pct=0.0, plan_steps=8,
                   review_blocking_issues=0, debug_cycles=0)
   assert compute_composite(m) == 1.0
   ```
2. Backfill the corpus. Spot-check:
   `df_task_13__minimax-m25-fp8-new__e78000d4.json` should go from
   `composite_score=0.0` to ≥ 0.9 (it's T/T/T with 0 blocking issues and
   1 debug cycle, so quality ≈ 1.0 - 0.05 = 0.95).
3. Run `uv run pytest orchestrator/tests/` and confirm no regressions.
   (Some existing tests may assert on composite scores; update them if
   they hardcoded the plan_completion_pct behavior — search for
   `plan_completion_pct` in `orchestrator/tests/`.)
4. Update the per-config table in `docs/vllm-eval-status.md` preliminary
   report to reflect re-scored vLLM results. Don't rerun any evals.

### ε non-goals

- Don't touch `plan_completion_pct` collection in `collect_metrics()` —
  it's useful diagnostic signal, just no longer in the score.
- Don't touch outcome classification (`done`/`blocked`/...). That's ζ.
- Don't add heuristic scoring based on `lines_changed` or `files_changed`
  — out of scope.

---

## ζ — completion judge LLM

### The design in one paragraph

After each implementer iteration in `_execute_iterations()`, invoke a
new **completion judge** agent (read-only, sonnet, small budget). The
judge receives: (a) the plan (plan.json contents), (b) the recent
iteration log, (c) the full diff of the worktree against the pre-task
baseline. The judge returns structured JSON: `complete: bool`,
`substantive_work: bool`, `uncovered_plan_steps: list[str]`, `reasoning:
str`. If the judge says `complete=True AND substantive_work=True`, exit
the implementer loop early with `WorkflowOutcome.DONE`. Otherwise
continue to the next iteration. Any judge failure (exception, malformed
output, `success=False`) falls through silently to the next iteration —
current behavior is preserved as the worst case.

### Critical naming warning — DO NOT TOUCH `evals/judge.py`

**`orchestrator/src/orchestrator/evals/judge.py` already exists.** It is
a 214-line Elo-pairwise blinded comparison judge used for cross-config
ranking, with `JudgeVerdict` and `JUDGE_SCHEMA` symbols. It is unrelated
to the completion judge you're adding.

- **Do not import from `evals/judge.py`.**
- **Do not modify `evals/judge.py`.**
- Use these names for the new judge to avoid confusion:
  - Prompt builder: `build_completion_judge_prompt`
  - Schema constant: `COMPLETION_JUDGE_SCHEMA`
  - Dataclass: `CompletionJudgeVerdict`
  - `AgentRole` Python constant: `JUDGE` (lives in
    `agents/roles.py`, no collision there)
  - `AgentRole.name = 'judge'` (the string is what the `role_key` split
    in `_invoke()` uses to look up per-role config; see workflow.py:1351)

### Step 1: `CompletionJudgeVerdict` dataclass + schema

**File:** `orchestrator/src/orchestrator/agents/briefing.py`

At the top of the file (after imports, before `BriefingAssembler`), add:

```python
from dataclasses import dataclass

@dataclass
class CompletionJudgeVerdict:
    complete: bool
    reasoning: str
    uncovered_plan_steps: list[str]
    substantive_work: bool

COMPLETION_JUDGE_SCHEMA = {
    'type': 'object',
    'properties': {
        'complete': {'type': 'boolean'},
        'reasoning': {'type': 'string'},
        'uncovered_plan_steps': {
            'type': 'array', 'items': {'type': 'string'},
        },
        'substantive_work': {'type': 'boolean'},
    },
    'required': ['complete', 'reasoning', 'uncovered_plan_steps', 'substantive_work'],
    'additionalProperties': False,
}
```

### Step 2: `JUDGE` AgentRole

**File:** `orchestrator/src/orchestrator/agents/roles.py`

Insert after `REVIEWER_COMPREHENSIVE` (currently at lines 325-340), before
the next role constant (likely `MERGER` around line 343). Mirror
`REVIEWER_COMPREHENSIVE` structurally but leaner:

```python
JUDGE = AgentRole(
    name='judge',
    system_prompt="""\
You are a completion judge. You decide whether an implementer agent has
*substantively* completed a task's work, regardless of whether the plan.json
bookkeeping reflects that.

## Context

You run AFTER each implementer iteration inside the orchestrator's execute
loop. You are read-only — you cannot edit code. Your job is to compare the
plan's intended behavior against the code that currently exists in the
worktree, and return a structured JSON verdict.

## Inputs you receive

1. The original plan (plan.json contents)
2. The iteration log so far (recent implementer activity)
3. The full diff of the worktree against the task's pre-task base commit

## What you must decide

- **complete**: Has the substantive work described by the plan's steps
  actually been implemented in the code diff? Ignore plan.json step
  statuses — they may be stale because some implementers don't update them.
  Judge by the code.
- **substantive_work**: Is the diff non-trivial and actually implements
  the plan? Return `false` if the diff is empty, only touches .task/,
  only contains whitespace/comment edits, deletes tests that were
  previously failing, or consists of trivially-passing tests with no
  corresponding production code.
- **uncovered_plan_steps**: Which plan step IDs appear unimplemented in
  the diff? Empty list means all steps are covered. This is advisory;
  your overall `complete` verdict is authoritative.
- **reasoning**: 2-5 sentences. Cite specific files and behaviors you
  checked to reach the verdict. Be concrete.

## Safety rules

- If `substantive_work` is `false`, you MUST return `complete=false`.
  An empty or trivial diff cannot be a completed task.
- When uncertain, prefer `complete=false`. The cost of a false negative
  is one extra implementer iteration; the cost of a false positive is a
  shipped incomplete task.
- Do not be swayed by plan.json status fields. The whole reason you exist
  is that those fields can be wrong.
- Do not run tests or modify anything. Use Read/Glob/Grep only.

## Output

You MUST output ONLY valid JSON matching the schema provided by the
--json-schema flag. No markdown fences, no prose outside the JSON.
""",
    allowed_tools=[*_READ_ONLY_TOOLS, *_JCODEMUNCH_TOOLS],
    disallowed_tools=['Edit', 'Write'],
    default_model='sonnet',
    default_budget=0.50,
    default_max_turns=15,
)
```

Register in the `ROLES` dict (currently at line 583):

```python
ROLES = {
    ...,
    'judge': JUDGE,
}
```

### Step 3: Per-role config fields in ALL SIX sub-models

**File:** `orchestrator/src/orchestrator/config.py`

You must add a `judge` field to all six per-role sub-config models. The
`_invoke()` method at workflow.py:1351 computes `role_key = role.name`
and does `getattr(config.models, role_key, ...)`, so missing a sub-model
means the judge falls back to defaults silently, which is usually bad.

Add:

- `ModelsConfig` (line 104): `judge: str = Field(default='sonnet')`
- `BudgetsConfig` (line 118): `judge: float = Field(default=0.50)`
- `TurnsConfig` (line 132): `judge: int = Field(default=15)`
- `EffortConfig` (line 146): `judge: str = Field(default='medium')`
- `TimeoutsConfig` (line 160): `judge: float = Field(default=300.0)`
- `BackendsConfig` (line 174): `judge: str = Field(default='claude')`

Top-level `OrchestratorConfig` (around line 281). Add near
`max_execute_iterations` (line 291):

```python
judge_after_each_iteration: bool = Field(default=False)
```

**Default False.** Production orchestrator runs are unaffected unless
explicitly opted in.

**Do not edit `orchestrator/src/orchestrator/defaults.yaml`** unless
grep shows that file already enumerates all per-role fields explicitly
(in which case add `judge` to each list to keep it consistent). Pydantic
defaults apply without touching defaults.yaml.

### Step 4: `build_completion_judge_prompt()` in briefing

**File:** `orchestrator/src/orchestrator/agents/briefing.py`

Insert after `build_reviewer_prompt()` (around line 199), following the
same style as the reviewer/debugger builders. Truncate large diffs to
50,000 chars (same cap as `build_reviewer_prompt` at line 186):

```python
async def build_completion_judge_prompt(
    self,
    plan: dict,
    iteration_log: list[dict],
    diff: str,
    task_id: str | None = None,
    context: str | None = None,
) -> str:
    """Build prompt for the completion judge agent."""
    effective_tid = task_id or plan.get('task_id')
    if context is None:
        context = await self._get_memory_context(effective_tid)

    identity = self._agent_identity(effective_tid, 'judge')

    # Truncate diff (same cap as reviewer)
    if len(diff) > 50000:
        diff = diff[:50000] + '\n\n... [diff truncated] ...'

    # Last 5 iteration log entries (reviewer uses 3; judge benefits from
    # seeing more of the arc of work)
    log_section = ''
    if iteration_log:
        recent = iteration_log[-5:]
        lines = [
            f"- iter {e.get('iteration', '?')} [{e.get('agent', '?')}]: "
            f"{e.get('summary', 'N/A')}"
            for e in recent
        ]
        log_section = "## Recent Iterations\n\n" + '\n'.join(lines)

    # Serialize only the plan fields the judge needs
    plan_json = json.dumps({
        'task_id': plan.get('task_id'),
        'title': plan.get('title'),
        'analysis': plan.get('analysis'),
        'prerequisites': plan.get('prerequisites', []),
        'steps': plan.get('steps', []),
    }, indent=2)

    return f"""\
{context}

{identity}

# Plan

```json
{plan_json}
```

{log_section}

# Code Diff (worktree vs pre-task base)

```diff
{diff}
```

# Action

Read the code in the worktree as needed to verify behavior. Then return
your verdict as JSON matching the schema. Follow the safety rules: if the
diff is empty or trivial, `substantive_work=false` and `complete=false`.
"""
```

### Step 5: Reuse `git_ops.get_diff_from_base()` — DO NOT write a new diff helper

`orchestrator/src/orchestrator/git_ops.py:364` already has:

```python
async def get_diff_from_base(self, worktree: Path, base_commit: str) -> str
```

It returns the full diff **content** as a string (not `--stat` output).
Use it directly. Do not write a new diff helper. If you need a fallback,
`get_diff_from_main(worktree)` exists at around line 356 in the same file.

### Step 6: Reuse the existing `output_schema` / `structured_output` pattern

The `TaskWorkflow._invoke()` method at `workflow.py:1335` already
accepts `output_schema: dict | None = None` as a fourth argument and
threads it through to the CLI backend. The return value is an
`AgentResult` whose `structured_output: Any = None` field
(`shared/src/shared/cli_invoke.py:69`) holds the parsed JSON when
`output_schema` was passed.

**Existing precedent:** `workflow.py:1049-1053` shows how the reviewer
uses this:

```python
result = await self._invoke(
    role, prompt, self.worktree, output_schema=review_schema
)
if result.structured_output:
    return result.structured_output
```

Follow the exact same pattern for the judge. No new invoke plumbing
needed.

### Step 7: Wire the judge into `_execute_iterations()`

**File:** `orchestrator/src/orchestrator/workflow.py`

Add imports at the top, alongside existing role imports:

```python
from orchestrator.agents.roles import (
    ..., IMPLEMENTER, DEBUGGER, JUDGE, ...,
)
from orchestrator.agents.briefing import COMPLETION_JUDGE_SCHEMA
```

**Insertion point inside `_execute_iterations()`** (currently lines
748-849): **after** the iteration log append at line 832, **after** the
post-implementer ownership validation at lines 834-841, **before** the
`while` loop continues (natural flow to the next iteration). Insert:

```python
            # --- Judge: decide whether to exit early ---
            if self.config.judge_after_each_iteration:
                judge_verdict = await self._run_completion_judge(iteration_log)
                if judge_verdict is not None and judge_verdict.get('complete') is True:
                    # Safety: reject complete=True if substantive_work=False
                    if not judge_verdict.get('substantive_work', False):
                        logger.warning(
                            f'Task {self.task_id}: judge returned complete=True '
                            f'with substantive_work=False — ignoring verdict'
                        )
                    else:
                        self.metrics.judge_early_exits += 1
                        logger.info(
                            f'Task {self.task_id}: judge signaled completion at '
                            f'iteration {self.metrics.execute_iterations} — '
                            f'reasoning: {judge_verdict.get("reasoning", "")[:200]}'
                        )
                        self.artifacts.append_iteration_log({
                            'iteration': self.metrics.execute_iterations,
                            'agent': 'judge',
                            'event': 'early_exit',
                            'complete': True,
                            'substantive_work': True,
                            'uncovered_plan_steps': judge_verdict.get('uncovered_plan_steps', []),
                            'summary': judge_verdict.get('reasoning', '')[:500],
                            'source': 'orchestrator',
                        })
                        return WorkflowOutcome.DONE
```

Add `_run_completion_judge` as a new helper method immediately below
`_execute_iterations` (around line 849):

```python
async def _run_completion_judge(
    self, iteration_log: list[dict]
) -> dict | None:
    """Invoke the completion judge. Returns parsed verdict dict or None on failure."""
    assert self.worktree is not None and self.artifacts is not None

    base_commit = self.artifacts.read_base_commit()
    if base_commit:
        diff = await self.git_ops.get_diff_from_base(self.worktree, base_commit)
    else:
        diff = await self.git_ops.get_diff_from_main(self.worktree)

    prompt = await self.briefing.build_completion_judge_prompt(
        plan=self.plan,
        iteration_log=iteration_log,
        diff=diff,
        task_id=self.task_id,
    )

    pre_cost = self.metrics.total_cost_usd
    try:
        result = await self._invoke(
            JUDGE, prompt, self.worktree,
            output_schema=COMPLETION_JUDGE_SCHEMA,
        )
    except Exception as exc:
        logger.warning(
            f'Task {self.task_id}: judge invocation raised {type(exc).__name__}: {exc} — '
            f'continuing iteration loop'
        )
        return None

    self.metrics.judge_invocations += 1
    self.metrics.judge_cost_usd += (self.metrics.total_cost_usd - pre_cost)

    if not result.success:
        logger.warning(
            f'Task {self.task_id}: judge invocation returned success=False — '
            f'continuing iteration loop'
        )
        return None

    verdict = result.structured_output
    if not isinstance(verdict, dict):
        logger.warning(
            f'Task {self.task_id}: judge returned non-dict structured_output — '
            f'continuing iteration loop'
        )
        return None

    required = {'complete', 'reasoning', 'uncovered_plan_steps', 'substantive_work'}
    if not required <= verdict.keys():
        logger.warning(
            f'Task {self.task_id}: judge verdict missing keys '
            f'{required - verdict.keys()} — continuing iteration loop'
        )
        return None

    return verdict
```

### Step 8: Cost accumulation is additive (not disjoint)

The judge's cost already flows into `self.metrics.total_cost_usd` via
`_invoke()` at workflow.py:1407-1413. Track it **separately** in a new
`judge_cost_usd` field by delta-ing `total_cost_usd` around the
`_invoke` call (as shown in `_run_completion_judge` above).

**This means `judge_cost_usd` is a SUBSET of `total_cost_usd`, not
disjoint.** Existing budget guards and cost reports that use
`total_cost_usd` continue to work unchanged. Document this explicitly
in the `EvalMetrics` docstring so future report generators don't
double-count.

### Step 9: Metrics plumbing

**File:** `orchestrator/src/orchestrator/workflow.py`

Add to `WorkflowMetrics` (currently at lines 112-128):

```python
judge_invocations: int = 0
judge_cost_usd: float = 0.0
judge_early_exits: int = 0
```

**File:** `orchestrator/src/orchestrator/evals/metrics.py`

Add to `EvalMetrics` (currently at lines 18-54), after the "Efficiency"
block (after line 34):

```python
judge_invocations: int = 0
judge_cost_usd: float = 0.0  # subset of cost_usd, not disjoint
judge_early_exits: int = 0
```

Wire into `collect_metrics()` at lines 77-135. In the `EvalMetrics(...)`
constructor call around line 111, add:

```python
judge_invocations=wf_metrics.judge_invocations,
judge_cost_usd=wf_metrics.judge_cost_usd,
judge_early_exits=wf_metrics.judge_early_exits,
```

**Backwards compatibility is free.** The `EvalResult` loader at
`orchestrator/src/orchestrator/evals/runner.py:361-362` already filters
unknown keys when deserializing, so old result files without these
fields still load cleanly.

### Step 10: Enable in eval mode only

**File:** `orchestrator/src/orchestrator/evals/runner.py`

In `build_eval_orch_config()` at lines 85-166, add `judge` entries to
each per-role config sub-model alongside the existing `reviewer`
entries:

- Around line 117: `ModelsConfig(..., reviewer='opus', judge='sonnet', ...)`
- Around line 127: `BudgetsConfig(..., reviewer=5.0, judge=0.50, ...)`
- Around line 135: `EffortConfig(..., reviewer='high', judge='medium', ...)`
- Around line 144: `BackendsConfig(..., reviewer='claude', judge='claude', ...)`
- (Plus `TurnsConfig` and `TimeoutsConfig` if they appear in this
  builder; use the Pydantic defaults otherwise.)

In the returned `OrchestratorConfig(...)` around line 147, add:

```python
judge_after_each_iteration=task.get('judge_after_each_iteration', True),
```

**Default True in eval mode.** Per-task override via spec field — same
pattern as `max_review_cycles=task.get('max_review_cycles', 1)` at
line 155. This lets individual task specs turn it off if needed.

**Do NOT modify production orchestrator configs.** The top-level Pydantic
default remains `False`, so production runs are unaffected.

### Step 11: Design decision locked in — no verify-before-judge in v1

Do not add a `run_verification()` call before invoking the judge.
Running full verification before each judge call would add 30s-2min per
iteration × up to 20 iterations × up to 25 task-config pairs ≈ 12.5 hours
worst-case extra wall clock per matrix run. Unacceptable.

**v1 design:** the judge receives plan + diff + iteration log only. No
verify output. The judge reads files itself with Read/Glob/Grep and
decides from code. The final `_verify_debugfix_loop` at workflow.py:917
still runs after `_execute_iterations` returns, and the debugger loop
handles any correctness failures the judge missed. The judge is a
*loop-exit hint*, not a replacement for verify.

If you find the judge is too lenient in practice, the follow-up
iteration can add a snapshot of the last-known verify state
(pass/fail summary, not full test output) to the prompt. Don't do that
in v1.

### Step 12: Tests

**New file:** `orchestrator/tests/test_briefing_judge.py`

Scoped to the new prompt builder. Four cases:

1. `test_build_completion_judge_prompt_includes_plan`: synthetic plan + empty
   iteration log + small diff → assert prompt contains plan step IDs,
   contains `# Plan`, contains `# Code Diff`, contains the "Follow the
   safety rules" action line.
2. `test_build_completion_judge_prompt_truncates_large_diff`: 60k-char diff
   → assert result contains `[diff truncated]`.
3. `test_build_completion_judge_prompt_recent_iterations_capped_at_5`:
   iteration log with 10 entries → assert only the last 5 appear.
4. `test_build_completion_judge_prompt_empty_iteration_log`: empty log →
   no `## Recent Iterations` section.

Patch `_get_memory_context` with `unittest.mock.patch.object` to return a
fixed string (avoids MCP calls). Look at how existing briefing tests (if
any) mock this, otherwise use the pattern in other `orchestrator/tests/`
test files that touch the briefing module.

**Augment:** `orchestrator/tests/test_workflow_e2e.py`

Extend `AgentStub._detect_role` (currently around line 187) with a new
branch at the top of the role detection:

```python
if 'completion judge' in system_prompt.lower():
    return 'judge'
```

Add a `_judge` method to `AgentStub` that returns a canned
`AgentResult(success=True, structured_output={...})` verdict, driven by
a list/flag attribute so each test can customize (follow the same
pattern as `_verify_results` or similar mechanisms already in
AgentStub).

Add these test functions to `test_workflow_e2e.py`:

1. `test_execute_iterations_exits_early_when_judge_says_complete`:
   judge returns `complete=True, substantive_work=True` → assert
   `outcome == WorkflowOutcome.DONE`, `metrics.execute_iterations == 1`,
   `metrics.judge_invocations == 1`, `metrics.judge_early_exits == 1`.
2. `test_execute_iterations_rejects_judge_complete_when_substantive_work_false`:
   judge returns `complete=True, substantive_work=False` → loop does
   NOT exit early, `judge_early_exits == 0`.
3. `test_execute_iterations_continues_on_judge_exception`: judge stub
   raises `ConnectionError` → loop continues normally, test-visible
   logging, eventual plan completion via implementer.
4. `test_execute_iterations_continues_on_judge_malformed_output`: judge
   returns `structured_output=None` or dict missing required keys →
   loop continues.
5. `test_execute_iterations_disabled_by_default`: without
   `judge_after_each_iteration=True`, judge stub is never called,
   `judge_invocations == 0`. This is the critical regression test.

**Run:** `uv run pytest orchestrator/tests/ -x` after each step. The
disabled-by-default test must pass first before any other changes land.

### Step 13: Validation — targeted subset run, not a full matrix

Before committing, run one real eval task end-to-end with the judge
enabled to verify the happy path works in production-ish conditions:

```bash
cd /home/leo/src/dark-factory
uv run orchestrator eval \
  --task orchestrator/src/orchestrator/evals/tasks/reify_task_12.json \
  --config-name claude-sonnet-max \
  --config orchestrator/config.yaml \
  --force
```

Pick `reify_task_12` because: (a) it's known to be tractable for
sonnet-max, (b) it's the task where vLLM runs historically reach T/T/T
at the iteration cap — it exercises the early-exit path.

Expected result file (under
`orchestrator/src/orchestrator/evals/results/reify_task_12__claude-sonnet-max__*.json`):
- `outcome: "done"`
- `metrics.judge_invocations > 0`
- `metrics.judge_early_exits >= 1`
- `metrics.iterations < 20` (ideally ≤ 5)
- `metrics.composite_score > 0.9` (ε + ζ both working)

**Do not run a full matrix until this targeted test passes.** If
`judge_early_exits == 0` on this task, either the judge prompt is too
conservative or the insertion point is wrong. Debug before expanding.

---

## The plan-production predicate (task 3302)

**The rule: `plan_steps > 0`. Never `plan_quality > 0`, never `outcome ==
'done'`.** Wherever the pipeline or a downstream consumer —
`select_survivors`, `compute_composite`, `blend_composite`,
`build_composite_report`, `build_plan_quality_report` — needs to know whether
an architect actually produced a plan, that is the question it asks, via
`plan_predicates.produced_a_plan(metrics_dict)`.

**Why not `plan_quality > 0`.** The two plan scorers *disagreed* exactly on a
stepless artifact. `judge.score_plan_structure` returns `0.0` for one as a
deliberate anti-fabrication guard (through `plan_predicates.is_scorable_plan`, which also
catches the truthy header-only stub `create_plan` writes); the LLM plan judge
had no such guard and scored the same artifact confidently. Observed in the
2026-07-29 corpus cell
`reify_task_12__architect-opus-high__52c66767.json`: `plan_steps=0` alongside
`plan_quality=0.31`. That number cannot have come from the floor at all —
`PLAN_QUALITY_RUBRIC`'s weights sum to `8.0` and the floor returns
`round(satisfied_weight / 8, 4)`, so its outputs are multiples of `0.125` and
`0.31` is not one. It came from the ungated judge.

**That disagreement is now closed at the instrument (task 3303)** — see below —
but the rule stands regardless, and for a reason independent of the fix:
`plan_quality` is the number being *decided*, so it can never be evidence for
the decision. Using it as the predicate makes a no-plan cell certify itself.
Recorded in Graphiti episode `e2066ec6` (edges `23d9c24f` / `8092af24`).

**Why not `outcome == 'done'`.** Done-without-a-plan and blocked-with-a-good-plan
cells both occur — task 2863 AMENDMENT §1.

**One artifact predicate, two persisted twins (task 3118).** The artifact-level
question is `plan_predicates.is_scorable_plan(plan)`: a dict carrying at least
one step. The runner's cap-taint decision and the floor's `0.0` short-circuit
were once two independent tests — raw dict truthiness in the runner, `not
plan.get('steps')` in the floor — and they disagreed exactly on the truthy
header-only stub `create_plan` writes, so a session cap landing right after
that first plan-tools call left the cell untainted and then floored it to a
fabricated `0.0`. Both call sites (and the plan judge's own refusal) are now
literally that one function. `produced_a_plan` is its twin over the persisted
`plan_steps`, which `run_architect_eval` derives with the identical
`len(plan.get('steps') or [])`;
`test_eval_architect.py::TestSteplessPlanIsNeverJudged.test_persisted_metrics_agree_with_the_artifact_level_twin`
pins the equivalence end-to-end through the real runner.

**Two failure causes, two treatments, two counts, neither silent:**

| Cause | Treatment | Count |
|---|---|---|
| Transport refusal (`cap_tainted`: 429 / auth / model-not-found / wedge) — we never got to ask | **EXCLUDED** from every pool | `plan_quality_cap_excluded` (composite row) / `cap_excluded` (θ table) |
| No plan produced (`not produced_a_plan`) — we asked, the answer was nothing | **FLOORED** to `0.0` on the quality axis *and* **HARD-GATED** to `composite = 0.0` (`blend_composite(no_plan=True)`), kept in every pool and counted | `plan_quality_no_plan` (composite row) / `no_plan` (θ table) |

A no-plan cell is additionally barred from *setting* its `(fixture,
'plan_only')` cost/latency baseline, though its real spend still enters the
row's pools: it is cheap and fast precisely because it failed
(`plans/eval-architect-effort-verdict-2026-07-27.md` measured $0.5–$3 against a
real plan's ~$3.7), so seeding the floor with it deflates every candidate that
succeeded. Barring it from the baseline was *not sufficient on its own*: it
closes only the intra-group route. Flooring the quality axis bounds just the
0.6 quality weight, and a no-plan cell that is the sole cell of its `(fixture,
'plan_only')` group takes the all-trials fallback baseline — earning ratios of
`1.0` on both efficiency axes and banking the remaining 0.4 as credit for
having failed. Measured on the pre-fix pipeline: a no-plan cell scored `0.40`,
outranked a config that produced a real 6-step plan at `0.26`, and survived
`select_survivors(top_k=2)` while the plan-producing one was cut. Hence the
composite hard gate — the plan-only analogue of the workflow `tests_pass` gate,
closing the cross-group route.

**Fixed at all three points.**

| Point | Fix | Task |
|---|---|---|
| The **instrument** | `judge_plan_quality` refuses a non-`is_scorable_plan` artifact up front — before a prompt is built, before `invoke_agent` — and returns the floor's own value | 3303 |
| The **writer** | `run_architect_eval` gates the judge call site on `is_scorable_plan`, so no new incoherent cell is written | 3302 |
| The **reader** | `report._plan_quality_score` floors a no-plan cell at read time, because the 2026-07-27 / 07-29 corpus is already on disk carrying the old shape | 3302 |

### The validity bound: judged without a reference (task 3628)

The two counts above are *failure* counts: something went wrong, and the
treatment was to exclude or to floor. This third one is neither. It is not a
failure cause, it changes no treatment, and it is **not** a third row of that
table. What it records is that a cell's `plan_quality` is *real but unbounded* —
the LLM judge produced it from an **empty `reference_diff`**, so it grades the
plan on plausibility rather than on fidelity to the diff that actually landed.

**Why a bound and not an exclusion or a floor.** The score is a genuine
measurement: the architect ran, produced a scorable plan, and the judge scored
it. Averaging those cells out would discard real data; flooring them would
assert a failure nobody observed. So they stay — in `n`, in
`mean_plan_quality`, in `plan_quality`, in `composite`, in `cost_usd` — at
their real values. What is reduced is the *confidence* in the number, not the
number. That is also what makes this count disjoint from both counts above: a
`cap_tainted` cell has no `plan_quality` to bound at all, and a no-plan cell's
score comes from the structural floor, which never consults a reference and is
therefore valid ground-truth-independently.

| Surface | Key | Rendered |
|---|---|---|
| Cell metrics | `judged_without_reference` | — |
| Composite row | `plan_quality_judged_without_reference` | `pq_no_ref` on `format_composite_table`, beside `pq_excluded` / `pq_no_plan` |
| θ table | `judged_without_reference`, per-config **and** report-level | `judged_without_reference` in the `plan_quality by config:` block, plus a trailer line stating the total against the scored pool (suppressed only when the bound is *measured* and zero) |

**Absence is not `False`.** The marker is three-valued everywhere it is read.
`EvalMetrics.to_dict` writes the key on *every* architect cell, so a healthy
cell says `False` explicitly; a cell whose metrics predate task 3628 carries no
key at all, and its validity was therefore never measured. The report layer
keeps those apart (`report._judged_without_reference`): one keyless cell in a
config's admitted pool makes that config's count — and the report-level total —
`None`, rendered `-`, with `judged_without_reference_unmeasured` carrying how
many cells drove it and the trailer printing `UNMEASURED` rather than being
suppressed. Reporting `0` there would let a rebuild over the 2026-07-27 / 07-29
corpus — the campaign this doc's provenance note says was judged blind on half
the hard subset — render byte-identically to a campaign graded entirely against
landed diffs, which is the very degradation σ exists to remove, reintroduced at
the reader. It is the same read-time discipline `report._plan_quality_score`
applies to the no-plan floor, and the same rule
`scripts/run_fable_trial_v2_campaign.py` (task 3632) states as its rule 3 and
enforces per cell with `MARKER_KEY not in metrics`; the two consumers of this
one field must not answer the same question differently.

**Provenance — this doc was already half-telling the story.** The
`reify_task_12` cell cited above carried no `reference` block, and neither did
`reify_task_27` or `df_task_18`: all three shipped with a top-level
`post_task_commit` but nothing for `run_architect_eval` to read, so it built an
empty diff and the judge scored plausibility. The 2026-07-29 v1 campaign graded
half the hard subset that way — including `reify_task_12`, the fixture carrying
the entire v1 result — and it was discoverable only by archaeology. Task 3628
back-filled the three `reference` blocks *and* added the marker, so the next
referenceless fixture is loud at run time — a `logger.warning` naming the task,
and a count on both tables — instead of being found in the corpus months later.

### The derived reliability surface (task 3379)

Those two *failure* counts are collected, but the figure an operator actually
compares candidates on is the *ratio* over them. Task 3379 derives it — no new
collection, no new predicate:

```
plan_rate     = (n_admitted - no_plan) / n_admitted
cost_per_plan = (spend over ADMITTED plan-only cells) / (n_admitted - no_plan)
```

| Figure | Composite row key | θ aggregate key | Rendered column |
|---|---|---|---|
| `plan_rate` | `plan_rate` (beside `plan_quality_n`, the denominator) | `plan_rate` | `plan_rate` on `format_composite_table` **and** in the `plan_quality by config:` block |
| `cost_per_plan` | `cost_per_plan` (immediately after `cost_usd`) | — (see below) | `cost_per_plan` on `format_composite_table` |

Both reduce through the shared `report._plan_rate` / `report._cost_per_plan`,
and `_plan_rate` is called by **both** builders — the `_mean_plan_quality`
discipline, so the two tables the CLI prints adjacently cannot drift.
`cost_per_plan` is composite-row-only because `build_plan_quality_report`
carries no cost data at all; threading some there would create a *second* cost
surface free to drift from `cost_usd`.

**The denominator rule: the ADMITTED θ pool, never `trials`.** A cap-tainted
cell leaves *both* the numerator and the denominator. It is a transport refusal
— we never got to ask — so it is neither a cell that planned nor a cell that
failed to plan, and counting it would report a candidate as *less reliable* for
a question a session-cap window stopped it from answering: the
schedule-attributable penalty tasks 3118 and 3099 spent two rounds removing,
reintroduced on the surface `select_survivors` ranks on. It also matches the
campaign this automates —
`plans/eval-architect-effort-verdict-2026-07-27.md` computes its own `planRate`
over **19** fixtures, dropping the 3 cap-contaminated ones, not over all 22.
`plan_quality_n` rides on the row so the ratio is verifiable from the row alone;
it is the same number the θ table reports as `n`.

**The no-plan cells' real spend stays in the `cost_per_plan` numerator.** That
asymmetry is the whole figure: you paid for the failed attempt and got nothing,
which is exactly what "$ per *usable* plan" prices. The verdict doc's own
arithmetic makes the point — fable read `$3.456`/fixture against the opus-max
incumbent but `$4.731` per usable plan, i.e. *no cheaper at all*, while failing
to plan 5× as often. Netting the failed attempts out of the numerator would
report that illusion rather than remove it. An *unmeasurable* cell's `$0.00` is
still excluded, for the same reason `_is_unmeasurable` keeps it out of the
`cost` pool: it is the price of a run that never happened, and letting it in
would report the cap window as a discount.

**`select_survivors` ranks on `plan_rate` as a TIE-BREAK, below
`plan_quality`** (axes 5–6 of 7; the name tiebreak falls to 7). Not above:
a no-plan cell is already paid for twice — floored into `plan_quality` and
hard-gated in `composite` — so ranking on reliability primarily would charge
the same failure a third time. Its residual value is real, though: one cell at
`0.0` beside one at `1.0` yields the *same* mean as two cells at `0.5` while
the configs differ sharply in reliability, and without this axis that tie falls
through to the alphabet — the defect the name tiebreak is meant to be a last
resort against.

**Both figures are `None` (rendered `-`) on an empty pool, never `0.0`.** For
`plan_rate` the fabrication would be two-sided: a `0.0` slanders a workflow
config that never ran an architect cell ("it never planned"), a `1.0` flatters
it ("it always planned"). For `cost_per_plan`, `n_planned == 0` means "we got no
plan at any price", which must not render as `$0.0000` — nor raise.

**The gap this closes.** The 2026-07-27 operator had to hand-compute `planRate`
and `$/plan` from the per-cell result JSONs because no report surface exposed
either, which is precisely why the campaign's headline finding — *what separates
these candidates is how often they emit a plan at all* — could not be read off
the pipeline's own ranking table. It can now.

### The anti-fabrication floor applies uniformly (task 3303)

Gating only the writer left the instrument's coherence resting entirely on its
ONE caller remembering to gate — so a second caller (a backfill/re-scoring
script, a new eval path, `prompt_opt`, a resume wave) would silently re-open the
defect above. **The floor therefore applies to whichever scoring path runs, and
is enforced AT the instrument**, using the same `is_scorable_plan` predicate the
floor short-circuits on and the runner gates on. Three call sites, one
predicate: they cannot drift into disagreeing about what a plan is, which is the
argument `is_scorable_plan`'s own docstring records for the level below.

Three details are load-bearing:

- **The refusal's score is DERIVED**, `score_plan_structure(plan)`, never a
  hardcoded `0.0`. A literal would be a third independent statement of what an
  unjudgeable artifact is worth, free to diverge if the floor's semantics ever
  change. Deriving it is what lets the test assert the identity
  `outcome.value.plan_quality == score_plan_structure(plan)` rather than a
  magic number.
- **`Scored` at `0.0`, not `Unmeasured`.** `Unmeasured` means "no judgement
  available, degrade to the floor": a transport refusal or an unparseable
  answer. A stepless plan from a healthy architect is a definite content
  verdict worth `0.0`, so no `invocation_errors` record is written either;
  conflating the two would collapse the content-failure / infra-failure
  distinction tasks 3118 and 3302 both turn on.
- **The refusal is logged at WARNING.** Reaching it means a caller did not gate.
  It cannot double-log through the normal path (the runner's gate
  short-circuits first), so the record stays rare and means exactly one thing.

The runner-side gate is *not* redundant now: it still owns the taint decision,
the `task_id × config.name` log line, and the skipped opus call — it simply
stops being the sole correctness guarantee.

**Known remaining asymmetry, filed as follow-up:** `judge_plan_quality` does
`float(raw_quality)` with no clamp, while `score_plan_structure` returns
`round(min(max(score, 0.0), 1.0), 4)`. `PLAN_QUALITY_SCHEMA` declares
`minimum`/`maximum`, but the `json.loads(result.output)` fallback path bypasses
schema enforcement, so an out-of-range judge score would persist verbatim. That
is a *range* invariant rather than the anti-fabrication floor, deliberately left
out of 3303's scope rather than overlooked.

**Prior art in-repo — this makes the pipeline agree with surfaces that already
got it right:** `scripts/eval_bootstrap_smoke.sh` gates its smoke on
`metrics.plan_steps > 0 & outcome == 'done'`, and
`plans/eval-architect-effort-verdict-2026-07-27.md` hand-computed the campaign's
`planRate` / `meanPQ_all` ("scores a no-plan cell as 0") from `plan_steps > 0`
because the pipeline's own ranking could not be used.

### Unmeasured cells: typed causes (task 6734)

Code: `orchestrator/src/orchestrator/evals/measurement.py` (`UnmeasuredCause`,
`classify_unmeasured`, `measured_call`), persisted on
`metrics.py::EvalMetrics.invocation_errors`, mapped from legacy cells by
`results.py::_invocation_errors_from_marker`, rendered by
`report.py::format_invocation_errors`.

An LLM invocation behind an eval cell either produced a measurement (`Scored`)
or did not (`Unmeasured`, naming its stage, cause, reason and spend). The cell
persists one record per stage that went unmeasured, architect first:

```json
"invocation_errors": [
  {"stage": "architect", "cause": "cap_hit", "reason": "You've hit your session limit · resets 8pm"}
]
```

`None` means no infra failure was observed. That includes an ordinary content
failure, an architect that ran fine and produced a bad or absent plan, which is
a genuine reliability signal and keeps scoring on content.

**Why each transport tier is unmeasurable.** `classify_unmeasured` delegates to
`shared.invocation_outcome.classify_invocation(strict_confirm=True)` rather than
re-matching cap strings in the eval layer: that seam owns the cap, auth and
model-not-found tables under a drift guard, and its `OK` short-circuit is what
stops a healthy run whose output merely quotes a cap message from being
tainted. The ladder, in order:

- `cap_hit`, `auth_failed`, `model_not_found`: refused before the model answered.
- An `OK` outcome is never a refusal.
- `api_error`: a body-less HTTP 429. 429 is excluded from `AuthFailed` because it
  normally carries a cap body; without this arm a body-less one would go
  unmarked. It precedes the wedge tier, so a 429 that also timed out with zero
  turns is reported as the 429 it is.
- `server_error`: HTTP 5xx, 529 Overloaded included. The provider refused. It
  ranks above the wedge tier because a SIGTERM-flushed 5xx on a watchdog-killed
  CLI is a provider outage, not a local wedge (the 2026-07-29 incident).
- `wedge`: a full timeout with zero transcript turns produced no answer at all;
  scoring it `0.0` would be the fabricated zero this exists to remove.
- Anything else (`NearCap`, `CliLocalError`, an unclassified failure) is not a
  refusal.

The runner adds the causes no `AgentResult` can show: `timeout`,
`cap_exhausted` (every account stayed capped past the eval's cap-wait patience),
and `harness_error` (our own crash: worktree, config, briefing or artifact read
raised). The plan judge adds `raised` and `unparseable` (a non-JSON, null or
non-finite answer).

**Recorded vs tainted.** `invocation_errors` records what happened;
`cap_tainted` is the scoring decision. A tainted cell has no model content at
all, so the plan_quality aggregates and a plan-only trial's composite and
quality pools EXCLUDE it and count the exclusion (`plan_quality_cap_excluded`,
`cap_excluded`), rather than averaging in a fabricated zero that would penalise
whichever candidate happened to run inside a cap window. The `trials`
denominator still counts every trial. The two deliberately diverge:

- A timeout is recorded but not tainted. The candidate was asked and did not
  finish inside the operator's budget, which is candidate-attributable;
  excluding it would let a slow candidate dodge the penalty its rivals paid.
- A transport refusal that still left a plan with steps is recorded but not
  tainted: the structural floor is a genuine content score.
- A judge-only failure is recorded (stage `judge`) and never taints.

`cap_tainted` stays an explicit boolean rather than being inferred from
`plan_quality is None`, because a null score alone is ambiguous between "not an
architect run" and "an architect run we could not measure" — the null-gating
discipline `_is_false_green` and `_is_null_work` already apply to `tests_pass`.

**Spend.** Every outcome carries the spend of the invocation behind it, summed
over every attempt: `measured_call` banks each attempt itself, because
`invoke_with_cap_retry` returns only the last result. A refused or unparseable
call still burned tokens, so it reports them; only a path that never invoked
(the unjudgeable-artifact floor above) reports `0.0`. `Scored` and `Unmeasured`
reject a non-finite or negative cost at construction, and producers coerce
through `measurement.py::coerce_cost_usd`, which is `isinstance`-gated because a
bare `MagicMock`'s `float()` is `1.0` and would fabricate spend. A judge that
raises leaves `judge_invocations` and `judge_cost_usd` at zero, since whether it
was ever reached is unknowable, but its `raised` record keeps the cell from
reading as judge-free. The judge's spend is a subset of the cell's `cost_usd`,
never an addend.

**Causes are not interchangeable.** The flag is named for the session cap that
motivated it, but its causes range from transient and schedule-attributable
(`cap_hit`, `cap_exhausted`: rerun later) to permanent configuration errors
(`model_not_found`, `auth_failed`: this candidate can never run). So the report
breaks `cap_excluded_by_cause` out by `UnmeasuredCause` value, keyed by the
cell's FIRST record (architect precedes judge, and a judge-only failure never
taints), and the per-cell `excluded` marker stays cause-neutral beside the
rendered records. A tainted row carrying no record raises, naming the cell.

**Legacy cells (PRD §7 D3).** Cells written before task 6734 carry one string,
`invocation_error`, such as `architect:cap_exhausted: …`.
`ResultStore.load_all` maps it once, at load, to a single record: the first
`;`-separated marker gives the stage and cause, and the reason is everything
after the string's first two `:` with exactly one leading space removed, so
`format_invocation_errors` reproduces the original string byte for byte. An
unrecognised stage or cause loads as `(architect, legacy_unclassified, <raw
string>)`, and a tainted legacy cell with no string gets `(architect,
legacy_unclassified, '')`. A live run never produces `legacy_unclassified`.

### Scoring instruments: unmeasured, never a tie or zero (task 6736)

Code: `reviewer_trial/scorer.py::match_issues`, `judge.py::run_judge` and
`::run_elo_tournament`, `compare.py::compare_models`,
`elo.py::MAX_UNMEASURED_PER_PAIR`, and the reducers
`reviewer_trial/campaign.py::summarize_arm` and
`reviewer_trial/report.py::build_trial_report` (all under
`orchestrator/src/orchestrator/evals/`).

**Instruments measure or say they did not (PRD D1).** The reviewer-trial
matcher, the pairwise judge, the comparison assessor and the synthesizer are
instruments, like the plan judge above. Each runs through `measured_call`, so a
refusal is `Unmeasured`. So is an answer that does not carry the measurement: a
missing winner, a winner outside A/B/tie, a missing or non-finite confidence,
or a `matches` that is not a list of objects. Before this task those cases came
back as a tie (the pairwise judge and the assessor), an overall tie (the
synthesizer) or an empty match list and F1 0.0 (the matcher). Each of those
is a measurement nobody made. Prose fields such as reasoning, summary and
strengths stay lenient, because rejecting on them would throw away a valid
score. A verdict record (`JudgeVerdict`, `TaskAssessment`) has `winner` None
exactly when `unmeasured` is set, enforced at construction by
`measurement.py::require_answer_iff_measured`. The winner vocabulary has one
home, `elo.PAIRWISE_RESULT`, and every schema enum derives from it.

**Reducers exclude and count (PRD D2).** Every reducer above an instrument
leaves an unmeasured record out of its figures and reports how many it left
out: `ArmMetrics.unmeasured`, `VariantSummary.unmeasured`,
`HeadToHead.unmeasured`, `ComparisonReport.unmeasured` and
`TournamentTally.unmeasured`. The `corpus-sanity` command does the same, and
it exits 1 while any cell is unmeasured, because the sanity run is incomplete.
A group with nothing measured reports None, never 0.0. Per-group cost ratios
are taken over measured records only, so an unmeasured record moves no mean,
cost-per-review included. Its spend is still reported, beside the ratios
(`unmeasured_cost_usd`) and in the trial and campaign totals, because a
refused call still spent. The synthesizer reads the measured assessments
only, and it is not called when none measured (`synthesis` is None).

**Elo.** An unmeasured verdict never reaches `record_match`, so it moves no
rating and adds nothing to the pair's `MAX_PER_PAIR` count. Unmeasured calls
spend `max_rounds` like measured ones. A pair that returns
`MAX_UNMEASURED_PER_PAIR` (2) unmeasured verdicts is retired for the rest of
that run, so a capped account cannot spin the loop. A pair with a missing
contender joins the same per-run skip set. That replaces the old write of a
fabricated `pair_counts[key] = MAX_PER_PAIR`, which `is_pairwise_data_dense`
then read as density. Nothing about the skip set is persisted in `TaskPool`,
so a later run measures those pairs again. As a second line of defence,
`record_match` raises on a winner outside `PAIRWISE_RESULT` rather than
scoring it 0.5.

**Cache and gates.** The reviewer-trial `campaign` command never caches an
unmeasured score, so a rerun measures that cell again. The matcher draws on the
campaign's existing usage gate. The pairwise judge, the assessor and the
synthesizer get no gate, because their drivers build none (PRD §12 q1). Wiring
one in later is a single `usage_gate=` argument to `measured_call`.

**Backfill (INV-13).** Score files a campaign cached before task 6736 carry no
`unmeasured` key, so they cannot be told apart from a refused matcher's
fabricated zero. They are abandoned, not migrated, and they load as measured.
A campaign that needs a trustworthy mean must rerun into a fresh
`--results-dir`.

### Terminal outcome (`terminal_kind`, task 4760)

The three counts above all describe the *quality* axis. This one describes what
the architect actually *said*, and it exists because `no_plan` never could: it
answers HOW MANY architect cells emitted no plan and never WHY.

**The closed vocabulary.** `metrics.TERMINAL_KINDS` — resolved by
`orchestrator/src/orchestrator/evals/metrics.py::resolve_terminal_kind`,
persisted on `metrics.py::EvalMetrics.terminal_kind`, read back through
`metrics.py::terminal_kind_of`:

| Value | Meaning |
|---|---|
| `planned` | A scorable plan (`plan_predicates.is_scorable_plan`) was the terminal statement |
| `already_done` | `plan_tools::report_task_already_done` — the work already landed |
| `blocking_dependency` | `plan_tools::report_blocking_dependency` — something else must land first |
| `false_premise` | `plan_tools::report_false_premise` — the task's premise does not hold |
| `unactionable` | `plan_tools::report_unactionable_task` — the task cannot be acted on as written |
| `ready_to_merge` | `plan_tools::report_ready_to_merge` — the branch is already mergeable |
| `none` | Neither a scorable plan nor any decline artifact — the genuine "could not plan", or an infra failure |
| `None` (key present, value null) | NOT MEASURED: a non-architect cell |
| key ABSENT | NOT MEASURED: a cell persisted before this field existed |

`to_dict` is a bare `asdict`, so the live instrument puts the key on **every**
cell — which is what makes key ABSENCE the one unambiguous legacy signal, the
same property `judged_without_reference` depends on. The write must never be
made conditional.

The last two rows are nevertheless read back **alike**: `terminal_kind_of`
answers `None` for both, and `terminal_kind_unmeasured` counts both. Over the
architect cells that count ranges on, the two state one fact — nobody resolved
this cell's terminal statement — and giving the written null its own arm would
leave it in neither `declined` nor `terminal_kind_unmeasured`, i.e. read back as
"measured, did not decline". Key absence stays the unambiguous legacy signal for
a reader who tests presence directly; no consumer has to.

The five decline kinds (`metrics.py::DECLINE_KINDS`) are single-sourced against
the `artifacts.TaskArtifacts.read_*` readers that produce them via
`metrics.py::_DECLINE_READERS`, so the vocabulary cannot drift away from the
artifacts it names.

**A DECLINE IS NOT A FAILURE.** It is the architect reading the fixture and
correctly refusing to plan it — the same exit an architect takes on a real task,
and the workflow treats it as a legitimate terminal state, not an error. So a
`plan_steps = 0` cell that carries a decline kind is evidence about the
*fixture*, not about the candidate. **Future scoring work should validate
declines against ground truth — is the premise in fact false? did the work in
fact already land? — and never penalise a decline as though it were a failure to
plan.** Scoring a correct refusal as a zero measures the corpus's staleness and
reports it as the candidate's incompetence.

**The ordering policy: LAST TERMINAL CALL WINS.** A cell can do both — confirm a
plan and then decline — so the two claims have to be ordered. Both producers
already stamp comparable ISO-8601 UTC timestamps
(`artifacts.py::TaskArtifacts.write_false_premise` and its four siblings stamp
`reported_at`; `plan_tools.py::_confirm_plan` stamps `_finalized_at` on
`plan.json`), so the resolver compares them rather than inferring an order.
Ambiguity — an unparseable or absent stamp, a tie, an unconfirmed plan carrying
no `_finalized_at` — resolves toward **the decline**, because a decline is an
explicit statement the architect had to call a tool to make, whereas a plan that
was never confirmed is one the workflow itself refuses to advance.

*Worked example* — run `e522b1b0`, cell `reify_task_4026`: a confirmed 6-step
plan, then `report_task_already_done` at a later stamp. It persists
`plan_steps=6, terminal_kind='already_done'`. **Both facts survive**: the cell is
in `declined` and NOT in `no_plan`, so a downstream reader can bucket it either
way and neither reading is destroyed. `declined - no_plan_declined` recovers that
population exactly.

**planRate is DELIBERATELY unchanged.** `produced_a_plan` is still
`plan_steps > 0`, `_plan_rate` still reduces the same two numbers, and
`is_scorable_plan` / `score_plan_structure` / `cap_tainted` are untouched — so
every campaign artifact already committed stays comparable. The split is
reported ALONGSIDE it, by
`report.py::build_plan_quality_report` (`declined`, `declined_by_kind`,
`no_plan_declined`, `terminal_kind_unmeasured`, over the SAME admitted pool `n`)
and surfaced verbatim by `scripts/run_fable_trial_v2_campaign.py`. **The number
to read beside `plan_rate` is `no_plan_declined`**: when it equals `no_plan`,
that candidate emitted no plan *only* where it explicitly refused, which is a
positive reliability result.

**Banding reads the split too, since task 4766.**
`scripts/run_fable_trial_v2_campaign.py::band_for_cell` was on that untouched
list when the counts above first landed, and is not any more: its no-plan rung
now splits into `no_plan` (this cell emitted no plan and said nothing about why)
and `declined` (it took one of the five explicit plan-tools decline exits), read
through the same `metrics.py::terminal_kind_of` accessor the report layer uses.
What did NOT change is the thing the sentence above was protecting.
**Both bands are RETAINED** — PRD D6 discards only the unambiguous ceiling band
— and `declined` sits immediately after `no_plan` in the fixture-level
precedence, which makes the split a pure label REFINEMENT: over every
one-to-three-cell fixture shape, each fixture keeps its old label or gains
`declined` exactly where the old label was `no_plan`, and none moves between
`retained` and `discarded`. That is enumerated exhaustively by
`test_the_split_cannot_re_select_the_pool`, so already-committed campaign
artifacts stay comparable in the partition as well as in `planRate`.

The split lives strictly inside the no-plan rung, so the worked example above is
untouched: `reify_task_4026` planned before it declined, never reaches that rung,
and still bands on its plan's merits — both facts still survive.

**The provenance.** Tranche 1 recorded `plan_steps = 0` on 47 of 53 architect
cells and the readout reported it as an ~89% planning failure. All 47 were
explicit, server-accepted decline calls — none silent, none capped — made after
a median of 41–98 assistant turns of real investigation. Establishing that took
manual forensics over `~/.claude/projects/*run-<id>/` transcripts, because the
decline artifacts themselves were already gone: `snapshots.py::cleanup_eval_worktree`
`rmtree`s the relocated meta root from `runner.py::run_architect_eval`'s
`finally`. The runner therefore reads the decline artifacts **inside that
`finally`, immediately before the cleanup call** — the one site reached by all
four exits (success, timeout, cap exhaustion, harness error) and the last moment
the artifacts exist.

### The architect cell: invoke, classify, assemble (task 6737)

Code: `orchestrator/src/orchestrator/evals/architect_eval.py`, composed by
`runner.py::run_architect_eval`.

`run_architect_eval` runs one architect cell as a short composition. The
architect is invoked and its attempt recorded as a frozen `ArchitectAttempt`:
outcome, plan, declines, usage, resolved architect cost and, when something
went wrong, one architect `Unmeasured`. The reference diff is read
(`read_reference_diff`). `classify_attempt` decides how the plan is scored.
The plan judge is asked (`judge_plan`) only when that decision is `JUDGE`.
`assemble_metrics` builds the persisted `EvalMetrics`, and
`log_architect_cell` writes the route's warning and the completion line.
`classify_attempt` and `assemble_metrics` are pure. Every effect boundary of a
cell (worktree, artifacts, invoke, judge, reference-diff read, gate factory,
cap-wait patience, store) is a field of `ArchitectEvalDeps`, whose defaults are
the production collaborators, so a test injects fakes rather than patching
module paths.

**The invoke step (`invoke_architect`).** It is the only step with effects on
the architect, and it drives only the architect: the worktree at the fixture's
`pre_task_commit`, `TaskArtifacts.init` at the relocated `.task-meta/<name>/`
root, the replay-framed architect prompt (docs/eval-replay-frame.md), one gated
`invoke_with_cap_retry` with this candidate's model, backend, effort and
`env_overrides`, then `read_plan`. Every downstream role is frozen. The
architect reaches plan-tools and no other MCP server
(`architect_eval_mcp_scope`): plan-tools is wired through the same seam real
dispatch uses, so the server writes `plan.json` where the artifacts read it,
and the strict flag drops the worktree's ambient `.mcp.json` with its live
escalation and fused-memory servers. The cell's address (`CellName`) is built
right after the worktree exists, before anything spends, so an unaddressable
cell fails without a run.

Each way the invoke can end is one row:

| Exit | `outcome` | Architect `Unmeasured` | Plan | Usage |
|---|---|---|---|---|
| Answered | `done`, or `blocked` when not `success` | the transport refusal, if `classify_unmeasured` finds one | `read_plan()` | the last attempt's |
| Answered, then `read_plan` raised | `blocked` | the refusal if any, else `harness_error` | `{}` | the last attempt's |
| `TimeoutError` | `timeout` | `timeout` | not read | zeros |
| `AllAccountsCappedException` | `blocked` | `cap_exhausted` | re-read, `{}` if that raises | zeros |
| Any other exception (config, artifacts, briefing, invoke) | `blocked` | `harness_error` | `{}` | zeros |

Every marker carries the spend banked so far. A classifier that raises leaves
the cell unmarked rather than losing it.

- The operator's `--timeout` bounds each attempt, never the cap-retry loop.
  Around the loop, a cap wait longer than the timeout would surface as
  `outcome='timeout'`, which is candidate-attributable and so not tainted: a
  cap-starved cell would be scored on a fabricated 0.0 and included in the
  mean, the differential-exclusion bias φ exists to remove. Per attempt, the
  two clocks keep their meanings: slowness is a timeout and scores on content,
  cap starvation is cap exhaustion and taints. A capped attempt is near-instant
  and free, so the attempt that burns the whole timeout is never the one being
  retried.
- Spend is banked per attempt, as each attempt returns.
  `invoke_with_cap_retry` returns only the last `AgentResult`, so a cap landing
  mid-run, after the architect spent real money, would otherwise vanish. That
  matters because the cell is now included: under-reported cost biases the
  cost comparison the same way exclusion biased quality. Duration is the
  winning attempt's alone, because cap waits and abandoned attempts are infra
  time, not candidate latency.
- Cap exhaustion re-reads the plan. The exception is raised at the invoke, so
  the ordinary `read_plan` never ran, and a plan written through plan-tools
  before the pool ran dry would be discarded, turning a scorable cell into an
  exclusion. The re-read makes exhaustion follow the same taint row an inline
  mid-run 429 follows. It is caught ahead of the generic handler so it is not
  labelled our own crash.
- A transport refusal outranks a later harness error: when `read_plan` raises
  after a classified refusal, the refusal is the more specific statement of
  why the cell is not measured.
- The decline artifacts are read in the step's `finally`, after every exit and
  before `cleanup_worktree`, which removes the meta root they live under: it
  is the last moment they exist. A read that raises costs the declines, never
  the cell.

The architect component's cost is resolved per Invariant P5 through
`metrics.resolve_cost_usd`, the seam `collect_metrics` uses. Its model is
`config.model`, the model the invoke was handed; `orch_config.models.architect`
is pinned to the frozen `opus` default for every architect eval and would
price a fable or vLLM candidate as opus. Its prices come from the built orch
config, which seeds `claude_endpoint_price_table()` for a proxied candidate
(task 2820), and its proxy flag is `is_proxied_endpoint(config.env_overrides)`,
the predicate that seeding keys on. A cell that spent nothing (no tokens, $0.00
banked) skips resolution and reads `(0.0, 'cli')`: there is no provenance to
resolve, and resolving would fire the unpriced-proxy warning for spend that
never happened. The judge's spend keeps its CLI figure (the judge is always a
native opus call) and the two are composed by `compose_cost_source`.

**The taint table (`classify_attempt`).** "Never asked" means the attempt
carries an architect `Unmeasured` whose cause is not candidate-attributable;
`CANDIDATE_ATTRIBUTABLE_CAUSES` holds `timeout` alone. "Scorable" is
`plan_predicates.is_scorable_plan`, the same predicate the floor and the
judge's own refusal consult.

| Never asked | Scorable plan | Scoring | `plan_quality` |
|---|---|---|---|
| yes | no | `TAINTED` | `None`, `cap_tainted=True` |
| yes | yes | `FLOORED` | `score_plan_structure(plan)` |
| no | no | `FLOORED` | `score_plan_structure(plan)`, i.e. `0.0` |
| no | yes | `JUDGE` | the judge's score, or the floor when the judge is unmeasured |

Per row, and why the line falls there (the general recorded-vs-tainted
argument is in "Unmeasured cells: typed causes" above):

- A transport refusal (cap hit, auth failure, model not found, body-less 429,
  5xx, zero-output wedge), cap exhaustion or a harness error, with no scorable
  plan, is tainted. The candidate was never asked, and charging a schedule,
  a configuration or our own crash to it would be a fabricated score. "No
  scorable plan" covers an absent artifact and the header-only stub
  `create_plan` writes.
- The same failures after a plan with steps landed (a cap landing mid-run,
  after the architect wrote real steps through plan-tools) are floored, not
  tainted: model content exists, so the structural floor is a genuine content
  measurement. The judge is skipped, because inside the same cap window it
  would 429 too.
- A healthy architect with no scorable plan is floored to `0.0` and not marked
  at all: that is a real reliability signal. The judge is skipped, so the
  score is never the judge's opinion of an artifact with nothing to judge, and
  the judge refuses such an artifact itself as well (task 3303).
- A timeout is marked but is candidate-attributable, so it keeps scoring on
  content: floored without a plan, judged with one.
- A failure of the judge alone is recorded (stage `judge`) and never taints:
  the cell falls back to the floor of a real plan.

The table is today's ladder exactly. Deriving "never asked" from the cause,
instead of carrying a second flag on the attempt, keeps one fact in one place,
and a test sweeps every `UnmeasuredCause` so a new cause lands on a row
deliberately.

**The reference diff and the two warnings.** The reference diff is read on
every route, so the fixture-level warning ("fixture carries no
`reference.post_task_commit`") fires once per cell even when the cell is
tainted and never reaches the judge. That warning states only the fixture-level
fact. The claim that a score was judged blind is made separately, by
`judge_plan`, exactly where the judge is invoked with an empty diff.
`judged_without_reference` is set only where the judge's own number is kept
(`Scored`), for the reasons in "The validity bound" above: every other path
persists the structural floor, which never consults a reference.

**Judge spend and calls.** `judge_invocations` is 1 whenever the judge was
called, even for a $0.00 refusal or an unparseable answer. A judge that raised
is recorded as `raised` with `cost_usd=0.0` and credits no call, because
whether the judge was ever reached is unknowable from the caller; its record
keeps the cell from reading as judge-free. `judge_cost_usd` is a subset of the
cell's `cost_usd`, as "Spend" above states.

**`tests_pass` is `None`.** No test signal exists for a plan-only cell:
implementer, debugger, reviewer and verify are frozen, so verification never
runs. `None` is the documented "unknown" value. The dataclass default `False`
would read as "the tests failed" and hard-gate `blend_composite` to `0.0`,
collapsing every architect row's composite and leaving `select_survivors`'
alphabetical tie-break as the whole selection mechanism. `True` is wrong too:
`build_composite_report` draws each fixture's cost and latency floor from
passing trials, and `ofat_candidates()` mixes architect, implementer and judge
candidates over the same fixtures, so a cheap plan-only cell marked passing
would become the floor for full-workflow cells, and it would fabricate a 100%
`tests_pass_rate` for a cell that never ran a test.

### Eval usage-gate lifecycle (task 6737)

Code: `orchestrator/src/orchestrator/evals/eval_gate.py`
(`build_eval_usage_gate`, `owned_usage_gate`, `campaign_usage_gate`,
`GATE_UNSET`, `InjectedGate`, `EVAL_CAP_WAIT_SANITY_SECS`). Consumers:
`runner.py`'s three executors and stage fan-outs, `cli.py::_run_single_eval`
and `reviewer_trial/__main__.py::campaign`.

**Why an eval invoke is gated at all (eval-revival φ).** Ungated, a capped
account refuses the architect outright and the cell is recorded `cap_tainted`,
so it is excluded from the reported mean. The 2026-07-28 architect wave lost
about 40% of its cells that way (37 blocked). Exclusion is not neutral: the
costlier candidate runs longer, is more cap-exposed, and so loses more cells
than its cheaper rival. The surviving cells are then a biased sample, and the
comparison the campaign exists to make is silently wrong. Failing over to a
healthy account measures the candidate instead of the schedule.

**One builder.** `build_eval_usage_gate` is the single construction every eval
entry point uses. It replaced three byte-identical copies, the shape that
drifts: the zero-account degrade below would otherwise have had to be
remembered in three places. It comes back ungated (`None`) three ways, all
warn-and-degrade rather than raise. Losing failover costs cells, but raising
costs the whole campaign, for a reason that has nothing to do with the
candidate under test.

1. `usage_cap.enabled=False`. This is a real production configuration: a
   deployment that opted out of the account pool must not pay for probe dirs
   and account state it never asked for.
2. The constructor raised.
3. The gate constructed but resolved zero accounts. `_init_accounts` logs "No
   accounts configured and no default credential found" and returns `[]`, so
   construction succeeds and the failure is deferred to the first
   `before_invoke`, which raises `RuntimeError('No OAuth accounts available
   ...')`. That lands in the caller's harness-error handler, so the cell is
   tainted and excluded. With `usage_cap.enabled=True` but no `CLAUDE_OAUTH_*`
   exported and no `~/.claude/.credentials.json` (an ordinary CI shell), every
   architect cell would be lost, where the same campaign ran fine ungated on
   ambient credentials before φ. An empty pool has nothing to fail over to, so
   degrading forfeits nothing. The discarded gate is still shut down
   (best-effort): even with no accounts the constructor allocated a fallback
   probe `TaskConfigDir` and installed a SIGHUP handler, and this runs once
   per cell.

**The three readings of a `usage_gate=` argument (`InjectedGate`, task 4427).**
A `UsageGate` is a live gate a campaign owner handed down. `None` is a campaign
owner that is deliberately ungated: run without failover and do not build one.
`GATE_UNSET`, the default, means nobody supplied anything: build a gate and own
it. A plain `None` default cannot express this, because `build_eval_usage_gate`
legitimately returns `None` three ways. A campaign whose gate degraded would
hand `None` to every cell, and each cell would read that as "nothing was
supplied" and build its own, silently restoring the per-cell construction the
campaign gate exists to remove (and re-paying a probe-dir allocation plus a
SIGHUP-handler steal per cell on the zero-account path). `GATE_UNSET` is an
instance of a private class so the annotation and the `isinstance` check are
pyright-clean.

**Whoever built the gate tears it down, and nobody else.** `owned_usage_gate`
is that contract, stated once for all three executors. Given `GATE_UNSET` it
builds through its factory, yields, and shuts the gate down on exit, whether
the body returned or raised. Given a gate or `None` it yields the argument and
touches nothing.

- An owned gate must come down. A campaign loops fixtures × candidates ×
  trials in one process, so a gate left running leaks a live account-resume
  probe loop firing real CLI probes for the rest of the campaign, and each new
  gate steals the SIGHUP handler from predecessors kept alive by exactly those
  background tasks.
- A borrowed gate must not. It carries the one cap-state view every sibling
  cell shares, so cell N+1 no longer re-leases the account cell N proved
  capped. Shutting it down would take failover from every sibling still to
  run, including ones already in flight.
- Teardown is best-effort everywhere: the cell (or the campaign) is already
  complete by then, so a failing `shutdown()` logs and never turns a finished
  result into a raise.

In the architect cell the owned gate wraps only the `invoke_with_cap_retry`
call, the one call that uses it. A harness error before the invoke therefore
builds no gate at all.

**One gate per campaign (`campaign_usage_gate`).** A campaign is any loop that
runs several cells in one process. It builds through `owned_usage_gate`, so
build, yield and teardown have one implementation. It builds from
`base_config`, which is right because the builder reads exactly one leaf,
`usage_cap`, and every cell's orch config is
`apply_eval_profile(base_config).model_copy(update=...)` with neither
`EVAL_PROFILE` nor the update dict mentioning `usage_cap`. The leaf is
inherited verbatim and the campaign gate is identical to the one each cell
would have built. That equivalence is emergent across two files, so
`test_eval_architect.py::TestCampaignGatePremise` pins it as a tripwire. A
`None` base config yields `None` without building: there is nothing to build
from, and every stage function defaults it. A `None` yield, from that or from
a degraded build, must be threaded to cells as an explicit `usage_gate=None`,
never dropped, for the reason in the readings above.

Sharing one gate across concurrent cells is the production shape, not a new
one: `orchestrator.harness.Harness` builds one `UsageGate` per orchestrator
process, shares it across every concurrently running workflow, and tears it
down once. Account leasing exists to arbitrate concurrent consumers, and
sharing is what makes cap state mean anything: a cell that proves an account
capped must inform its in-flight siblings, not merely its successors.

Sharing also makes the gate's runtime state campaign-scoped, including
`UsageGate._cumulative_cost`. `before_invoke` raises `SessionBudgetExhausted`
once that tally reaches `usage_cap.session_budget_usd`, and nothing resets it
between cells, so one gate means one session ceiling for the whole campaign
where a per-cell gate gave each cell a fresh budget. With the ceiling set, a
long campaign would trip partway through and every remaining cell would block
or be stamped `harness_error`, losses the report would attribute to the
candidate. This is latent, not live: `session_budget_usd` defaults to `None`
and no config in this repo sets it. A campaign that wants per-cell budgeting
should build from a `usage_cap` copy with `session_budget_usd=None` and let
each invocation's `max_budget_usd` bound it instead.

**Cap-wait patience: 48 h (`EVAL_CAP_WAIT_SANITY_SECS`).** It was raised from
1800 s on 2026-08-25 (esc-3634-1). The 1800 s house value was justified as
"fail loud rather than park a campaign for two weeks", and that rationale does
not survive contact with the gate:

- A short bound never prevented the park. When every account is capped,
  `before_invoke` waits on the gate's own unbounded `_open.wait()`, released
  only by a real cap reset or a successful resume probe. Freeing a campaign
  slot buys no progress, because the next cell parks in the same wait. The
  short bound only decided how many in-flight cells were destroyed on the way
  to parking.
- Tainting costs more than waiting. A cap lands mid-cell, after the architect
  has spent real money (γ1 mean: $8.88 banked per cap-excluded cell).
  `invoke_with_cap_retry` resumes the capped session with `--resume` on a
  freed account, so waiting costs about nothing extra, while tainting discards
  the banked spend and pays full freight again on the re-run.
- Restart is not a cheap escape. `run_ofat_stage`, the path campaign drivers
  use, has no skip-existing guard (that is `run_eval_matrix`'s
  `ResultStore.exists` check). Cells persist individually, so a kill loses no
  data, but a relaunch re-runs and re-spends them unless the operator
  restricts `--tasks-dir` by hand.

48 h is sized to the observed account regime: one to three accounts typically
sit uncapped against the weekly cap and are hit repeatedly against their
5-hour caps, giving short all-capped windows regularly plus one long window
each week. 1800 s cannot skate a 5-hour window; 48 h skates it with margin. A
weekly exhaustion outlasts even this bound, but no value of this constant
makes that case loud, so the choice costs nothing there. There is deliberately
no `max_cap_retries` companion: cooldown doubles per full cycle through the
pool, so a fixed count would make a bigger, healthier pool give up sooner.

What the bound does not cover: `cap_wait_sanity_secs` is consulted only in
`_check_cap_wait`, in the cap-hit branch after an invocation returned and was
classified as a cap. It bounds cap-retry patience and nothing else; the
unbounded `_open.wait()` is out of its reach. That park is no longer silent
(task 4945): `shared/usage_gate.py::UsageGate.before_invoke` emits a throttled
`all_capped_park` WARNING carrying elapsed seconds, the account count and the
soonest expected reopen, so a parked campaign is distinguishable from a hung
one by reading the log, without aborting a park whose banked spend `--resume`
would otherwise recover.

---

## Sharp edges — read these before starting

1. **Cost ceiling at full matrix scale.** Worst case: 5 tasks × 5
   configs × 20 iterations × ~$0.15/call ≈ $75 per full matrix. Likely
   actual is much less because early exits cut the iteration count.
   Not a blocker for v1, but consider adding a `max_judge_invocations`
   safety knob to `OrchestratorConfig` in a follow-up if real-world cost
   exceeds expectations.

2. **`uncovered_plan_steps` is advisory in v1.** The judge's overall
   `complete` verdict is authoritative. If the judge says `complete=True`
   but `uncovered_plan_steps` is non-empty, we trust the overall
   verdict. The future session may want to harden this — e.g., require
   `complete=True AND uncovered_plan_steps == []` — but not in v1.

3. **`substantive_work=False` is the main guard against false-green
   empty-diff completion.** Our separate verify false-green bug —
   `_verify_debugfix_loop` reports PASS on an unchanged worktree because
   the baseline is clean — is **out of scope** for ζ. Flag as a
   follow-up in `docs/vllm-eval-status.md`. The judge's `substantive_work`
   check partially mitigates it at the loop-exit point when
   `judge_after_each_iteration=True`.

4. **`plan_completion_pct` will look wrong on judge-early-exit runs.**
   Example: judge exits after step 1 of 3, `plan_completion_pct=0.33`.
   After ε lands this no longer affects `composite_score`, but any
   dashboard that displays `plan_completion_pct` will look confusing for
   early-exit runs. Document in `docs/vllm-eval-status.md` and update any
   dashboard widgets that use the field.

5. **`role_key` resolution for the judge.** `TaskWorkflow._invoke()` at
   workflow.py:1351 does `role_key = role.name.split('_')[0]`. For
   `role.name='judge'`, `role_key='judge'`. The per-role config lookup
   finds `config.models.judge`, `config.budgets.judge`, etc. No special
   handling like the `role.name.startswith('reviewer')` branch at
   workflow.py:1362-1368 — `'judge'` is a clean name.

6. **Eval-only initially.** Do not enable judge in production orchestrator
   runs in this PR. Get eval data first, validate the prompt quality,
   then flip a follow-up PR that turns it on in `defaults.yaml` or the
   project-level config.yaml for live use.

---

## Critical files (one-line index)

- `orchestrator/src/orchestrator/evals/metrics.py:60` — `compute_composite` (ε)
- `orchestrator/src/orchestrator/evals/metrics.py:18-54` — `EvalMetrics` dataclass
- `orchestrator/src/orchestrator/evals/metrics.py:77-135` — `collect_metrics`
- `orchestrator/src/orchestrator/evals/runner.py:85-166` — `build_eval_orch_config`
- `orchestrator/src/orchestrator/evals/runner.py:361-362` — EvalResult loader field filter (pattern to reuse)
- `orchestrator/src/orchestrator/evals/judge.py` — **DO NOT TOUCH** (Elo pairwise judge; naming collision warning)
- `orchestrator/src/orchestrator/workflow.py:112-128` — `WorkflowMetrics`
- `orchestrator/src/orchestrator/workflow.py:748-849` — `_execute_iterations`
- `orchestrator/src/orchestrator/workflow.py:1049-1053` — `output_schema`/`structured_output` reviewer precedent
- `orchestrator/src/orchestrator/workflow.py:1335-1415` — `_invoke` (accepts `output_schema`)
- `orchestrator/src/orchestrator/agents/roles.py:325-340` — `REVIEWER_COMPREHENSIVE` (template for JUDGE)
- `orchestrator/src/orchestrator/agents/roles.py:583` — `ROLES` dict
- `orchestrator/src/orchestrator/agents/briefing.py:62-199` — existing prompt builders
- `orchestrator/src/orchestrator/config.py:104-186` — six per-role config sub-models
- `orchestrator/src/orchestrator/config.py:281-351` — `OrchestratorConfig`
- `orchestrator/src/orchestrator/git_ops.py:364` — `get_diff_from_base()` (reuse, don't rewrite)
- `shared/src/shared/cli_invoke.py:47-77` — `AgentResult` with `structured_output`
- `orchestrator/tests/test_workflow_e2e.py:~187` — `AgentStub._detect_role` extension point
