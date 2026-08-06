# Cross-Account Session-Resume Measurement (task 3484)

**Question:** when a Claude CLI session is started on OAuth account A and then
`--resume`d on a *different*, healthy OAuth account B, does B's turn recall the
context established by A?

**Answer, measured 2026-08-05: YES — context IS preserved.** Three valid runs,
three `preserved` verdicts, zero void runs, same-account control passing in
every run.

| Field | Value |
|---|---|
| Measurement date | **2026-08-05T20:04:11Z – 20:05:15Z** (gate satisfied → last run) |
| claude CLI version | **2.1.222** — read from the `version` field of the r1 transcript records themselves (see [Transcript corroboration](#transcript-corroboration)), not from a same-day `claude --version` |
| Account pair | `CROSS_ACCOUNT_RESUME_TOKENS='F,C'` — **A = `CLAUDE_OAUTH_TOKEN_F`** started the session (r1), **B = `CLAUDE_OAUTH_TOKEN_C`** issued the `--resume` (r2) |
| Valid runs | **3** (`preserved`, `preserved`, `preserved`) |
| Void runs in this round | **0** (no `void_capped`, no `void_error`) |
| Harness | `shared/tests/test_cli_invoke_integration.py::TestCrossAccountResume`, `-vs -m integration` |
| Invocation | haiku, `max_turns=1`, `max_budget_usd=0.05`, `cwd=/tmp`, `allowed_tools=[]`, `effort=low` (`_INVOKE_DEFAULTS`) |
| Durable evidence | `data/3484-cross-account-resume-evidence/` — `evidence.jsonl`, `run{1,2,3}.log`, `runner.log`, `status.json`, `runner.py` |
| Related tasks | 3454 (the 2026-08-01 INCONCLUSIVE round) · 3483 (single-homed cap corpus) · 3484 (this round) |
| Verdict lives in code at | the `MEASURED` comment above the cap-hit resume branch in `shared/src/shared/cli_invoke.py` (`invoke_with_cap_retry`) |

> **Evidence location.** Cite `data/3484-cross-account-resume-evidence/`. The
> runner's original scratch dir `/tmp/3484-window/` is on temp storage and may
> be reaped at any time; its absence is not evidence loss. (`data/` is
> gitignored, so it is durable on the host but not in git — which is why the
> raw records are reproduced verbatim below.)

---

## How the measurement was run

A detached runner (`runner.py`, copied into the evidence dir so the gating
logic is auditable) was armed by the task steward against account C's stated
session-cap reset at 20:00Z. It honoured prerequisite **pre-1 as a hard gate**:
probe first, and spend on the measurement **only if two accounts are
simultaneously healthy**. Probing goes through `invoke_claude_agent` — the same
code path the measurement uses — at a generous `max_budget_usd=0.50` so a probe
can never abort on budget. An account is HEALTHY iff it answers `PONG`.

The runner deliberately stopped at evidence capture: it wrote no verdict into
`cli_invoke.py`, because deciding what the evidence means is not a script's job.

### Pre-1 probe output (verbatim, from `status.json`)

| At (UTC) | `CLAUDE_OAUTH_TOKEN_C` | `CLAUDE_OAUTH_TOKEN_F` | Gate |
|---|---|---|---|
| 2026-08-05T19:58:04Z | CAPPED — `"You've hit your session limit · resets 9pm (Europe/London)"` | `"not probed (C still capped)"` | **NOT satisfied** — no measurement run, nothing spent |
| 2026-08-05T20:04:11Z | **HEALTHY** — `"PONG"` | **HEALTHY** — `"PONG"` | **SATISFIED** → pair `F,C`, 3 runs |

The window was genuinely narrow: at 19:58:04Z — six minutes before the gate
opened — C was still capped. That is why the pair is `F,C` and not the default
first-two-available pair (`B,C`), and why the `CROSS_ACCOUNT_RESUME_TOKENS`
override exists at all.

### Runner log (verbatim)

```
2026-08-05T18:34:16+00:00  runner armed; pre-1 is a HARD GATE — measurement runs only if 2 healthy
2026-08-05T18:34:16+00:00  sleeping 5023s until 19:58Z
2026-08-05T19:58:04+00:00  probe: C=CAPPED F=CAPPED/skipped
2026-08-05T20:04:11+00:00  probe: C=HEALTHY F=HEALTHY
2026-08-05T20:04:11+00:00  GATE SATISFIED — pair=F,C; running measurement 3x
2026-08-05T20:04:11+00:00  --- measurement run 1/3 ---
2026-08-05T20:04:32+00:00  run 1: rc=0 null_run_suspected=False
2026-08-05T20:04:32+00:00  --- measurement run 2/3 ---
2026-08-05T20:04:52+00:00  run 2: rc=0 null_run_suspected=False
2026-08-05T20:04:52+00:00  --- measurement run 3/3 ---
2026-08-05T20:05:15+00:00  run 3: rc=0 null_run_suspected=False
2026-08-05T20:05:15+00:00  DONE. evidence verdicts=['preserved', 'preserved', 'preserved']
```

`null_run_suspected=False` is the check that the `-m integration` marker
actually took: a run that reports `3 deselected` executed nothing and would
look green having spent nothing. All three runs report `3 passed`, so none is a
null run.

---

## The runs

Each section gives the evidence line **verbatim** from
`data/3484-cross-account-resume-evidence/evidence.jsonl`, plus the pytest
summary from the corresponding `runN.log`.

### Run 1 — `preserved`

```json
{"account_a": "CLAUDE_OAUTH_TOKEN_F", "account_b": "CLAUDE_OAUTH_TOKEN_C", "codeword_recalled": true, "control_passed": true, "r1_session_id": "6a259899-315b-4cd3-94cd-8448c982daaf", "r1_transcript_present": true, "r1_transcript_records": 11, "r2_output": "ZEPPELIN", "r2_stderr": "", "r2_subtype": "success", "r2_success": true, "verdict": "preserved"}
```

```
tests/test_cli_invoke_integration.py::TestCrossAccountResume::test_invoke_returns_session_id PASSED
tests/test_cli_invoke_integration.py::TestCrossAccountResume::test_session_resume_same_account_baseline PASSED
tests/test_cli_invoke_integration.py::TestCrossAccountResume::test_session_resume_preserves_context_across_accounts PASSED
============================== 3 passed in 18.96s ==============================
```

### Run 2 — `preserved`

```json
{"account_a": "CLAUDE_OAUTH_TOKEN_F", "account_b": "CLAUDE_OAUTH_TOKEN_C", "codeword_recalled": true, "control_passed": true, "r1_session_id": "75f9c167-7e91-4743-995e-5d943fac2326", "r1_transcript_present": true, "r1_transcript_records": 11, "r2_output": "ZEPPELIN", "r2_stderr": "", "r2_subtype": "success", "r2_success": true, "verdict": "preserved"}
```

```
============================== 3 passed in 19.05s ==============================
```

### Run 3 — `preserved`

```json
{"account_a": "CLAUDE_OAUTH_TOKEN_F", "account_b": "CLAUDE_OAUTH_TOKEN_C", "codeword_recalled": true, "control_passed": true, "r1_session_id": "362bb71e-0489-4f61-b930-0cf772982d04", "r1_transcript_present": true, "r1_transcript_records": 11, "r2_output": "ZEPPELIN", "r2_stderr": "", "r2_subtype": "success", "r2_success": true, "verdict": "preserved"}
```

```
============================== 3 passed in 22.21s ==============================
```

### Per-run fields, tabulated

| Field | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| r1 `session_id` | `6a259899-315b-4cd3-94cd-8448c982daaf` | `75f9c167-7e91-4743-995e-5d943fac2326` | `362bb71e-0489-4f61-b930-0cf772982d04` |
| r1 transcript present | `true` | `true` | `true` |
| r1 transcript records | `11` | `11` | `11` |
| r2 `output` | `"ZEPPELIN"` | `"ZEPPELIN"` | `"ZEPPELIN"` |
| r2 `stderr` | `""` | `""` | `""` |
| r2 `success` | `true` | `true` | `true` |
| r2 `subtype` | `success` | `success` | `success` |
| `codeword_recalled` | `true` | `true` | `true` |
| `control_passed` (same-account) | `true` | `true` | `true` |
| `verdict` | `preserved` | `preserved` | `preserved` |

Three **distinct** r1 session ids: each run started a fresh session rather than
re-reading one warm transcript, so the result is not an artifact of a single
session.

<a id="transcript-corroboration"></a>

## Transcript corroboration (measured 2026-08-06 from the on-disk transcripts)

Read directly out of the three r1 transcripts, which were still on disk under
the worktree's ambient config dir at
`.task/claude-config-3484/projects/-tmp/<session_id>.jsonl` — the `-tmp` slug
comes from `cwd=/tmp`, exactly as the MECHANISM paragraph predicts:

| Session | Records now | `version` | First ts | Last ts |
|---|---|---|---|---|
| `6a259899-…` | 19 | `2.1.222` | 2026-08-05T20:04:24.675Z | 2026-08-05T20:04:30.599Z |
| `75f9c167-…` | 19 | `2.1.222` | 2026-08-05T20:04:44.642Z | 2026-08-05T20:04:50.781Z |
| `362bb71e-…` | 19 | `2.1.222` | 2026-08-05T20:05:07.417Z | 2026-08-05T20:05:14.134Z |

Two things follow, both independent of the test's own assertions:

1. **The CLI version is pinned at 2.1.222** by the records themselves. (The
   runner did not capture `claude --version`; the host reads `2.1.223` as of
   2026-08-06, i.e. the CLI has since been upgraded.)
2. **The cross-account resume appended to the same local file: 11 → 19
   records.** The test measured 11 records after r1 and before r2; the file now
   holds 19. The appended turns are r2's, on account C:

   ```
   record 14  type=user       "What was the codeword I told you? Reply with just the word."
   record 17  type=assistant  "ZEPPELIN"
   ```

   This is the same behaviour 3454 observed (12 → 20) — but where 3454's r2
   turn was the verbatim text `"You've hit your weekly limit · resets Aug 5,
   11am"`, here it is a real model turn that recalls the codeword.

These transcripts live under a gitignored `.task/` config dir inside the task
worktree and will not survive a worktree reset — which is why the facts are
transcribed here rather than cited by path alone.

---

## Attempt history, including the void and null rounds

The full history matters: this question took three rounds to answer, and each
failed round failed in a *different* way that the harness now guards against.

| Round | Date | Outcome | Why |
|---|---|---|---|
| 1 (task 3454) | 2026-08-01, CLI 2.1.220 | **1 VOID cross-account run, 0 valid** | r2's transcript turn was verbatim `"You've hit your weekly limit · resets Aug 5, 11am"` — account B was capped and no model turn ever ran. Same-account control PASSED (`8e4d1819-…`, 12 records, codeword recalled), so the harness was sound. 4 of 5 accounts in env were capped. Recorded honestly as `VERDICT: INCONCLUSIVE`. |
| 2 (task 3484, first attempt) | 2026-08-05, daytime | **0 runs** — no window | At 15:15Z only one account was uncapped; a cross-account resume needs two. No measurement was attempted, and nothing was spent on one. |
| 3 (task 3484, this round) | 2026-08-05T20:04Z, CLI 2.1.222 | **3 valid runs, verdict `preserved`** | Gate satisfied at 20:04:11Z with C and F both healthy. |

Three traps were found and closed along the way; each had already produced, or
was about to produce, a non-context failure dressed up as context loss:

- **Cap text drift (found in round 1, closed by task 3483).** The module's skip
  guard matched `"you've hit your usage"` while the real text is `"you've hit
  your weekly limit"`, so a capped account failed the ZEPPELIN assertion loudly
  instead of skipping. The corpus now lives single-homed in
  `shared/tests/_capacity_skip.py`, pinned against the verbatim string and
  cross-checked against production's `classify_invocation`.
- **The probe one-liner does not work (measured 2026-08-05T16:04–16:29Z, CLI
  2.1.222; commit `7e10b0b172`).** A bare
  `claude -p 'Say exactly: PONG' --model haiku --max-turns 1` hung until killed
  on all six tokens (`timeout` rc=124 at both 90s and 180s), emitting only
  `Execution error`, under the ambient config dir and a fresh isolated one
  alike. It reads as "all six capped" and burns the window it was meant to
  find. The same six tokens probed through `invoke_claude_agent` in the same
  minutes answered cleanly (1 healthy, 5 capped with verbatim limit messages) —
  which is why the runner probes through `invoke_claude_agent`.
- **The budget-abort trap (measured 2026-08-05; commit `07f7b5b720`).** At the
  then-current `max_budget_usd` of $0.01, one run aborted with `success=False`,
  `subtype='error_max_budget_usd'` and an EMPTY output while another passed
  having spent $0.0083674 — 84% of the ceiling. A budget abort is not a
  capacity failure, so the cap guard did not skip it; its output is empty, so
  no codeword was found — and the record scored it `not_preserved`, i.e. a
  budget abort recorded as a production defect. Closed on both layers: the
  ceiling is now $0.05 (~6× the observed worst case), and any non-cap
  `r2.success is False` scores `verdict='void_error'`, so `not_preserved` now
  requires r2 to have SUCCEEDED and still not recalled the codeword.

None of those three traps fired in this round: all three runs recorded
`r2_success=true`, `r2_subtype='success'`, empty `r2_stderr`, and a non-empty
`r2_output` containing a real model answer.

---

## Verdict

**Cross-account session resume PRESERVES conversation context.** Measured
2026-08-05, claude CLI 2.1.222, accounts `CLAUDE_OAUTH_TOKEN_F` → `_C`, 3 valid
runs, 3 × `preserved`, 0 void.

The reasoning that ties the verdict to the runs:

1. **A model turn really ran on account B.** `r2_success=true`,
   `r2_subtype='success'`, `r2_stderr=''`, and `r2_output='ZEPPELIN'` — a real
   answer, not a limit message and not the empty output of a budget abort. The
   two ways this measurement has previously produced a false reading (a capped
   account, a budget abort) are both excluded by the recorded fields, and both
   now have their own verdict class (`void_capped`, `void_error`) that no run
   in this round took.
2. **The account really did change between r1 and r2.** `account_a` and
   `account_b` are distinct env vars, `select_token_pair` rejects a pair naming
   the same account twice, and both accounts were independently probed healthy
   6 minutes before the first run.
3. **The harness was sound in the same process.** `control_passed=true` in all
   three runs: the same-account baseline (FLAMINGO on account A) recalled its
   codeword in the very same pytest session. A cross-account result taken while
   the control is red would be uninterpretable; this one was not.
4. **The result reproduced across three fresh sessions**, with three distinct
   r1 session ids, in three separate pytest processes.
5. **The mechanism explains it, and the transcripts show the mechanism.** The
   resume appended to the *same local JSONL file* r1 wrote (11 → 19 records,
   with r2's user turn and its `ZEPPELIN` answer among the appended records).
   Resume is governed by transcript REACHABILITY — same config dir, same cwd —
   not by OAuth identity. Round 1 had already ruled out the transcript-absent
   explanation; this round supplies the positive half.

**Consequence for production.** The cap-hit resume branch in
`shared.cli_invoke.invoke_with_cap_retry` — which resumes a capped session on
the next account in the rotation — is doing something that works. There is no
production defect on this axis, so 3484's "IF THE ANSWER TURNS OUT TO BE
'context is NOT preserved'" contingency does not apply and nothing is escalated.
The reachability guard that task 3454 added on that branch remains load-bearing
for the *other* failure mode: it is what stops a resume against a transcript
that is gone (cleaned-up `TaskConfigDir`, different config dir, swept temp dir),
which really does start an effectively empty session.

**Scope of the claim.** This was measured on claude CLI 2.1.222 with a local
transcript reachable by both invocations. It is a statement about *this*
mechanism, not a guarantee from Anthropic's API: if a future CLI moves sessions
server-side and scopes them per account, the mechanism changes and the answer
could change with it. The regression guard is
`TestCrossAccountResume::test_session_resume_preserves_context_across_accounts`
— re-run it (with `-m integration`, and `CROSS_ACCOUNT_RESUME_TOKENS` aimed at
two healthy accounts) after a CLI upgrade that touches session handling.

---

## Reproducing / re-checking

```bash
# 1. Probe health through invoke_claude_agent (NOT a bare `claude -p` one-liner).
#    An account is healthy iff it answers PONG; you need TWO at once, and a
#    probe older than ~15 minutes is stale.
CROSS_ACCOUNT_RESUME_TOKENS='F,C' uv run --project shared --directory shared \
    pytest tests/test_cli_invoke_integration.py -m integration -q \
    -k test_invoke_returns_session_id

# 2. Measure. The `-m integration` marker is MANDATORY — both shared/pyproject.toml
#    and the root pyproject.toml deselect it, so without it every test is silently
#    deselected and the run looks green having executed nothing. Confirm the summary
#    says "3 passed", not "3 deselected".
export CROSS_ACCOUNT_RESUME_TOKENS='F,C'          # aim at the healthy pair
export CROSS_ACCOUNT_EVIDENCE_PATH=/tmp/evidence.jsonl
uv run --project shared --directory shared \
    pytest tests/test_cli_invoke_integration.py::TestCrossAccountResume -vs -m integration
```

A run whose record says `verdict='void_capped'` or `verdict='void_error'` is
VOID — it does not count, and it is **not** evidence of context loss. If a run
is red with an unfamiliar limit message, add that phrasing to
`REAL_CLI_CAP_MESSAGES` in `shared/tests/_capacity_skip.py` (single-homed and
drift-guarded) rather than reasoning around it locally.

## See also

- `shared/src/shared/cli_invoke.py` — the `MEASURED` comment above the cap-hit
  resume branch in `invoke_with_cap_retry`: the single source of truth in code
  for what is and is not established about cross-account resume.
- `shared/tests/test_cli_invoke_integration.py` — the harness, the
  `CROSS_ACCOUNT_RESUME_TOKENS` / `CROSS_ACCOUNT_EVIDENCE_PATH` knobs, and the
  regression guard.
- `shared/tests/_cross_account_evidence.py` — `select_token_pair`,
  `format_run_evidence`, `emit_run_evidence` (unit-tested by
  `shared/tests/test_cross_account_evidence.py`).
- `shared/tests/_capacity_skip.py` — the single-homed cap-message corpus
  (task 3483).
- Tasks: **3454** (round 1, INCONCLUSIVE) · **3483** (cap corpus) · **3484**
  (this measurement).
