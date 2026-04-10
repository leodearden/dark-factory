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
