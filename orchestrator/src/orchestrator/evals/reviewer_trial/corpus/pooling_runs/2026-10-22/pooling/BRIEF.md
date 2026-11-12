# Pooling the answer key: adjudicate unmatched review findings

## Why this matters

An offline experiment scores code reviewers against an answer key of known issues for each
past change. A finding that matches no answer-key issue is counted as a false alarm. A first
blind check showed the key is badly incomplete: most "false alarm" blocking findings were real
problems. We are now repairing the key. Every finding any reviewer raised that matched nothing
is checked against the code. Each distinct real problem becomes a new answer-key entry, and the
whole experiment is then re-scored with its rule unchanged.

What you produce becomes ground truth, so precision matters more than speed. Two errors damage
the key:
- adding a problem that is not real, or not part of the change;
- adding a duplicate of an issue the key already has.

You are not told which reviewer raised which finding, and you must not try to find out.

## Inputs

Your packet is a JSON file (path in your task prompt) holding a list of fixtures. Each fixture
has these fields:

- `clone`: a scratch git clone. What it holds depends on `tree_state`:
  - `applied`: HEAD is the reviewed change and HEAD^ its base, so `git -C <clone> show HEAD` is
    the diff.
  - `base-only: ...`: the stored diff would not apply, so HEAD is the change's BASE and the
    change is NOT in the tree. Read the change from `diff_file` and the surrounding code from the
    clone.
  - `none: ...`: there is no usable tree (`clone` is null). Judge from `diff_file` alone, and lower
    your confidence wherever a claim depends on code outside the diff.
- `diff_file`: the stored diff text.
- `task`: the task the change was written for (title, description, test_strategy, details) and
  `plan`, the implementation plan, possibly truncated. Either may be null.
- `answer_key`: the issues the key already has.
- `findings`: the unmatched reviewer findings, each with `id`, `category`, `location`,
  `description` and `suggested_fix`. Many repeat the same underlying problem in different words,
  because they come from several independent reviews.

You have full read access to the clones. Read any file, grep, use git log/blame/show, and follow
callers and callees as far as you need. Small targeted executions are allowed, such as a
`python -c` probe. Do NOT run test suites, builds, or anything long-running: the host is heavily
loaded. Do not fetch from the network.

## What to do, per fixture

1. Group the findings into distinct underlying problems. Two findings belong together when fixing
   one fix would resolve both.
2. Decide each problem:
   - `same_as_key_issue`: the id of an existing answer-key issue if this problem IS that issue
     (the matcher missed the pairing), else null. Check this first and carefully.
   - `real`:
     - `yes`: the problem is present as the findings describe it.
     - `partly`: a real problem exists, but the findings misstate it. Write the corrected version
       in `description`.
     - `no`: the code refutes it.
   - `in_change`:
     - `yes`: introduced, or materially touched, by the change. A task requirement the change
       fails to deliver, such as a missing test the task asked for, counts as `yes`.
     - `no`: it lives only in pre-existing code the change left alone.
     - `n/a`: if not real.
   - `corpus_severity`, if real:
     - `blocking`: merging as-is ships broken correctness or security. Examples: wrong results, a
       crash, data loss, a race, a security hole, a broken contract with callers, a task
       requirement left undelivered, or a test that cannot detect the defect it claims to test.
     - `suggestion`: style, minor issues, or maintainability.
     - `n/a`: if not real.
   - `quality_block`: only when real and `suggestion`. Answer `yes` if it is still a maintainability
     or design problem serious enough that a careful senior reviewer would refuse to merge, and
     you can name a concrete next change it makes riskier. Otherwise `no`. Use `n/a` otherwise.
   - `location` (`path:line`, in the change), `category` (short snake_case) and `description`: a
     canonical, self-contained statement of the problem as it exists in the code. Write it as an
     answer-key entry would be written, not as a review comment.
   - `confidence` (`high`, `medium` or `low`) and `evidence`: at most 3 sentences citing
     `path:line`, saying what you checked.
3. Every finding id must appear in exactly one problem's `finding_ids`. Vague or wrong findings
   go into a problem marked `real: "no"`; a single "not real" problem may collect several.

## Output

Write `<verdicts dir>/<seat>.json` (path in your task prompt). It is a JSON array with one object
per fixture in your packet:

```json
{"fixture": "<id>", "problems": [
  {"pid": "P1", "finding_ids": ["<id>", "..."], "same_as_key_issue": null,
   "real": "yes|partly|no", "in_change": "yes|no|n/a",
   "corpus_severity": "blocking|suggestion|n/a", "quality_block": "yes|no|n/a",
   "location": "path:line", "category": "snake_case", "description": "...",
   "confidence": "high|medium|low", "evidence": "..."}
]}
```

Before writing, check:
- the JSON parses;
- every finding id in your packet appears exactly once;
- every enum value is one listed above.

Then reply with only:
- the output path;
- per fixture: the number of findings, the number of distinct problems, and how many are new,
  real and in the change;
- anything you could not adjudicate.

## Boundaries

- Read only your packet, the files it names, and the clones. Do not open anything else under the
  scratchpad, anything under `/home/leo/src/dark-factory/plans/`, or any experiment results. That
  would unblind you.
- Do not modify the clones. Use no MCP tools.
- Never skip a finding. If you cannot decide, give your best verdict with `confidence: "low"` and
  explain.
