# Skeptic pass: try to refute "real correctness blocker" verdicts

## Why this matters

An offline experiment scored code reviewers against an answer key. A blind adjudication then
found that many reviewer findings the key counted as false alarms are real problems that should
block a merge. If that is right, the experiment's verdict was unfair. If it is wrong, we would
"repair" the key with false entries. Your job is to find out which, by trying hard to break the
adjudicator's conclusions. A pass that refutes nothing is acceptable only if you genuinely tried.

## Inputs

Your packet is a JSON file (path in your task prompt) holding a list of fixtures. Each fixture
has:

- `clone`: a scratch git clone. HEAD is the reviewed change and HEAD^ its base, so
  `git -C <clone> show HEAD` is the diff.
- `task` and `plan`: what the change was written for.
- `items`: each has a reviewer's `claim` and the earlier adjudicator's `prior_verdict`. The prior
  verdict always says the problem is real, in the change, and a correctness or security blocker,
  with that adjudicator's evidence.

Several items describe the same underlying problem, raised in independent reviews.

You have full read access to the clones: read anything, grep, use git, follow callers. Small
targeted executions are allowed, such as a `python -c` probe. Do NOT run test suites or builds.
Do not fetch from the network.

## For each item, try to refute each part of the prior verdict

1. **Real.** Is the scenario actually reachable? Look for guards, callers that make it
   impossible, types that rule it out, or a misreading of the code.
2. **In the change.** Was the problem introduced or materially touched by HEAD^..HEAD, or does it
   live in old code the change left alone?
3. **Correctness or security blocker.** Does merging actually ship broken behaviour: wrong
   results, a crash, data loss, a race, a security hole, a broken caller contract, a task
   requirement left undelivered, or a test that cannot catch what it claims to? Or is it really
   a style, minor or maintainability point?

## Output

Write `<verdicts dir>/<seat>.json` (path in your task prompt). It is a JSON array with one object
per item:

```json
{"id": "...", "fixture": "...", "problem": "<short name of the underlying problem; reuse the same name across items that share it>",
 "upheld": "yes|no", "failed_part": "none|real|in_change|blocking",
 "corrected_severity": "blocking|suggestion|not_real",
 "confidence": "high|medium|low", "evidence": "<at most 3 sentences citing path:line>"}
```

Check before writing that the JSON parses and has exactly one object per item. Then reply with
only:
- the output path;
- the number of items upheld and refuted;
- the distinct problem names, each with its verdict.

## Boundaries

- Read only your packet and the clones it names. Do not open anything else under the
  scratchpad, anything under `/home/leo/src/dark-factory/plans/`, or any experiment results.
- Do not modify the clones. Use no MCP tools.
