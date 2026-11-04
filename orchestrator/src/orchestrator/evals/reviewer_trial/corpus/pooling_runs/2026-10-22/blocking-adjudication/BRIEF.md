# Blind adjudication of reviewer blocking flags

## Why this matters

An offline experiment compared code reviewers on a corpus of past changes. Each change comes with
an answer key of known issues. Many issues the reviewers marked BLOCKING ("do not merge until
fixed") matched nothing in that key, so the scoring counted them as false alarms. The answer key
may be incomplete, though. Your job is to establish, from the code itself, whether each flagged
problem is real and how serious it is. The result decides whether the scoring was unfair to one
of the reviewers. You are not told which reviewer raised which item, and you must not try to
find out.

## Inputs

- Your packet, at the path in your task prompt. It lists one or more fixtures. For each fixture
  it gives:
  - a scratch git clone holding the reviewed tree. HEAD is the change under review and HEAD^ is
    its base, so `git -C <clone> show HEAD` shows the diff;
  - the task the change was written for;
  - the items, each with an `id`, `category`, `location`, `description` and `suggested_fix`.
- You have full read access to each clone. Read any file, grep, use git log/blame/show, and
  follow callers and callees across the codebase as far as you need. Small, targeted executions
  are allowed, such as a `python -c` that imports a module or evaluates an expression. Do NOT run
  test suites, builds (cargo, npm, tsc, pytest on directories) or anything long-running: the
  host is heavily loaded.

## For each item, decide

1. `real`: is the specific problem the item claims actually present in the reviewed tree (HEAD)?
   - `yes`: the claim holds as stated.
   - `partly`: a real problem exists, but the item misstates its trigger, scope or effect.
   - `no`: the claim is wrong. The code already handles it, the premise misreads the code, or the
     scenario cannot occur.
2. `in_change`: is the problem introduced, or materially touched, by this change (HEAD^..HEAD)?
   Use `no` if it lives only in pre-existing code the change did not touch, and `n/a` if
   `real` is `no`.
3. `corpus_severity`, if real or partly real, under the answer key's own standard:
   - `blocking`: merging as-is ships broken correctness or security. Examples: wrong results, a
     crash, data loss, a race, a security hole, a broken contract with callers, or a test that
     cannot detect the defect it claims to test.
   - `suggestion`: style, minor issues, or maintainability.
   - `n/a` if not real.
4. `quality_block`: only when real or partly real AND `corpus_severity` is `suggestion`. Answer
   `yes` if it is a maintainability or design problem serious enough that a careful senior
   reviewer would still refuse to merge, and you can name a concrete next change it would make
   riskier or costlier. Otherwise `no`. Use `n/a` in every other case.
5. `confidence`: `high`, `medium` or `low`.
6. `evidence`: at most 3 sentences, citing `path:line` in the clone. Say what you checked and
   what settled it.

Several items within a fixture may describe the same underlying problem; they come from
different reviews. Judge each one on its own wording. If an item is identical in substance to
an earlier one, you may give the same verdict and write "same as <id>" in the evidence.

## Output

Write a JSON file to `<verdicts dir>/<your seat>.json` (the exact path is in your task prompt).
It must be a JSON array with exactly one object per item in your packet:

```json
{"id": "F01-I01", "fixture": "<fixture id>", "real": "yes|partly|no", "in_change": "yes|no|n/a",
 "corpus_severity": "blocking|suggestion|n/a", "quality_block": "yes|no|n/a",
 "confidence": "high|medium|low", "evidence": "..."}
```

Then reply with only:
- the output path;
- the item count;
- a count of each `real` value;
- any item you could not adjudicate and why.

## Boundaries

- Read only your packet and the clones it names. Do not open anything else under the scratchpad,
  anything under `/home/leo/src/dark-factory/plans/`, or any experiment results. That would
  unblind you.
- Do not modify the clones. Use no MCP tools (no fused-memory, no escalation).
- Before writing, check that your JSON parses, has one object per packet item, and that every
  enum value is one of those listed above.
- If an item is too vague to adjudicate, mark it `real: "no"` only when the code refutes it.
  Otherwise give your best verdict with `confidence: "low"` and explain in the evidence. Never
  skip an item.
