# ALLOWANCE seat: how the shared agent usage allowance is consumed, and what a weight on it could mean

Seat: allowance. Written 2026-11-12 ~15:45Z. Read-only.
Code read: Dark Factory `main` at `7848173a7d` (2026-11-12T13:55Z). The running services predate part of it.
Every orchestrator unit started between 2026-11-10T20:56Z and 23:48Z, and fused-memory started 2026-11-12T14:47Z.
Since 11-10T20:00Z, `shared/src/shared/usage_gate.py` gained one change, the probe-slot hold bound
(`813cb4d8ed`/`65284b3e96`, task 9327). The running orchestrators do not have it. Nothing else in the selection
or cap path changed.
Data: every `data/orchestrator/runs.db` of the nine orchestrator units, plus fused-memory's curator store
`data/reconciliation/curator_events.db`. Extracted from 2026-09-15 on; the newest row is 2026-11-12T15:37Z.
Scripts and their outputs are in `/tmp/team-weights-1112/allowance/`: `extract.py` writes `raw.pkl`, then
`consume.py`, `tiers.py`, `caps.py`, `predict.py`, `stalls.py`, `herding.py` and `verify.py` each write a matching
`*_out.txt`.
Windows: W0 = 2026-10-25T00:00Z..11-08T16:00Z (14.67 d). W1 = 2026-11-09T14:13Z..11-12T15:00Z (3.03 d) unless stated.

**Consumption unit: `invocations.cost_usd`.** It is the notional API-price cost the Claude CLI reports for one call
(the subscription is flat-rate, so nobody is billed it). Each `invoke_with_cap_retry` call writes one row, for its
final attempt, attributed to the final account (`shared/src/shared/cli_invoke.py::invoke_with_cap_retry`).
Usage probes write no row.
**Which unit the cap meters could not be determined.** At the weekly cap, cost_usd, total tokens and output tokens
predict equally well: their spreads differ by 0.03 or less (F7). The fleet is 90% one model (claude-opus-5-5), so
cost and tokens move together and the data cannot tell them apart.

## Findings

```
F1. The allowance pool is one shared list of seven accounts. Every orchestrator unit and fused-memory use it, and
    no unit overrides it. The nine units, reify included, all run from the Dark Factory checkout and read the
    same token env.
    figure: 7 accounts (in list order: max-d, max-e, max-f, max-g, max-b, max-c, max-h); 9 orchestrator units +
            fused-memory curator + reconciliation harness + legibility census/return brief draw on it
    rule:   accounts_file resolved per config: project yaml if it sets usage_cap.accounts_file, else
            orchestrator/src/orchestrator/defaults.yaml usage_cap.accounts_file (= DF config/usage-accounts.yaml)
    window: config as of 2026-11-12          population: 9 units under systemctl --user, fused-memory config
    kind:   measured total
    evidence: DF/pump-web-ui yaml usage_cap.accounts_file; reify config/usage-accounts.yaml symlink; solar-challenge,
            solar-challenge-platform yaml have no usage_cap section, so defaults.yaml applies (orchestrator/src/
            orchestrator/config.py::YamlSettingsSource.__call__ deep-merges defaults); fused-memory/config/
            config.yaml usage_cap.accounts_file; every unit has WorkingDirectory=/home/leo/src/dark-factory.
            Account names were read; no token values. Data agrees: all 7 accounts appear in every large project's
            invocations (consume_out.txt "accounts seen per source").
```

```
F2. W0 pool shares by project. Dark Factory took a quarter. Two projects the pause never touched (scoping-intake,
    stream-touch) took 29%. The three paused projects took 44%.
    figure: dark-factory 24.7% | scoping-intake 20.8% | solar-challenge-platform 16.9% | solar-challenge 14.7% |
            reify 12.6% | stream-touch 8.1% | pump-web-ui 2.1% | know-live 0.1%; pool total $25,864 = $12.3k/week
    rule:   sum cost_usd by project, invocations completed in window; fused-memory curator rows assigned to the
            project_id they served (curator is 1.9% of the pool in total); all rows kept, capped=1 included (71 rows)
    window: W0                                population: 16,661 invocation rows, 9 runs.db + curator_events.db
    kind:   ratio (project cost / pool cost)
    evidence: allowance/verify.py (straight SQL, re-run) = allowance/tiers.py
    W1 for comparison: dark-factory 60.6% | scoping-intake 22.1% | stream-touch 17.2% | the three paused ≈0.1%;
            pool $4,148 = $9.6k/week. Dark Factory went from $3.0k/week (W0) to $5.8k/week (W1).
```

```
F3. In W0 61% of the pool went to tasks whose current tier is low, and 15% to high or critical. Shares by role:
    implementer 40%, architect 38%, comprehensive reviewer 14%. All other roles together took under 8%. Work with
    no task attached took 3.9%.
    figure: W0 tier: low 61.1 | medium 19.9 | high 14.8 | critical 0.2 | no-task 3.9 (%)
            W1 tier: low 38.9 | medium 29.5 | high 26.9 | no-task 4.7 (%)
            W0 low share within project: scoping-intake 84%, pump-web-ui 85%, solar-challenge 71%,
            solar-challenge-platform 69%, stream-touch 59%, reify 47%, dark-factory 37%
    rule:   tier = tasks.priority in that project's .taskmaster/tasks/tasks.db TODAY. This is the current own tier,
            not the tier at dispatch, so a task re-tiered since then is counted at its new tier. Rows with no
            integer task id are 'no-task' (curator, escalation-watcher-auto, deep_reviewer)
    window: W0, W1                            population: as F2
    kind:   ratio (tier cost / pool cost)
    evidence: allowance/tiers.py (tiers_out.txt); role split allowance/consume.py
```

```
F4. The binding limit is the WEEKLY one. Every weekly reset falls at a fixed hour each week, a different one per
    account. Every all-capped minute in W1, and 4 of the 8 W0 stalls of 30 min or more, had all seven accounts
    on a weekly cap. The other 4 (10-27..28) had six on a weekly cap and one on a session cap. Each all-weekly
    stall ends at the moment the next account's weekly reset arrives.
    figure: weekly reset times (UTC, from resets_at, unchanged week to week since mid-Sep): max-c Wed 10:00,
            max-f Wed 13:00, max-d Thu 13:00, max-b Fri 12:00, max-e Sat 17:00, max-g Sun 13:00, max-h Mon 21:00.
            The longest gap between resets is Mon 21:00 -> Wed 10:00 (37 h).
            Stall end = a weekly reset, for all 6 all-weekly stalls: 10-28T10:00 (max-c), 11-04T10:00 (max-c),
            11-07T17:00 (max-e), 11-08T13:00 (max-g), 11-09T21:00 (max-h), 11-11T10:00 (max-c).
            Cap messages since 09-15: 1,837 rows "weekly limit", 446 "session limit", 19 "5-hour limit", 210 old-format
            label-only rows, 1 heuristic (of 2,513 cap_hit rows)
    rule:   cap kind read from the CLI's own message text in account_events.details.reason; reset time from the
            gate-parsed details.resets_at
    window: 2026-09-15..11-12 (reset times); W0/W1 (stalls)   population: 2,513 cap_hit rows, all 10 stores
    kind:   measured total
    evidence: allowance/predict.py (reset-time block), allowance/caps.py (stall table), caps_out.txt
```

```
F5. The pool was fully capped for 23.8% of W1 and 10.2% of W0. Each account alone was capped about three
    quarters of W0.
    figure: all-capped: W1 16.4 h of 68.8 h = 23.8% (to 11-12T11:00, the capacity.md F3 window; 22.5% to 15:00);
            W0 36.0 h of 352 h = 10.2%, in 8 stalls of 30 min or more (W0 stalls 1-4, 10-27..28, each had one account on a
            session cap and the rest on weekly caps; all W1 stalls were 7 weekly).
            Per-account capped share W0: max-d 82%, max-g 81%, max-e 79%, max-f 76%, max-h 72%, max-b 61%, max-c 48%.
            W1: max-b and max-e 100% (their weekly allowance was spent on 11-07 and 11-08, before W1); max-d and
            max-g 97%; max-c, max-f and max-h 60-65%.
    rule:   capacity.md F3 rule applied across ALL sources: each non-scoped cap_hit row opens [t, min(resets_at,
            next 'resumed' for that account from any source, start of next >60 s invocation on that account from
            any source)]; intervals merged per account; a minute is all-capped when all 7 accounts are inside an
            interval at its midpoint. Fable per-model scope caps excluded (the account keeps serving)
    window: W0, W1                            population: 7 accounts; cap_hit/resumed from 10 stores
    kind:   ratio (minutes / window minutes)
    evidence: allowance/caps.py (run with 2026-11-12T11:00:00+00:00 and default 15:00); reproduces the brief's 24%
```

```
F6. Each process discovers each cap separately. The average weekly or session cap was detected by 7.9 different
    processes. Selection is first-open-account in list order, so all processes pile onto the same account and
    drain the pool one account at a time.
    figure: 826 cap_hit rows / 39 distinct episodes in W0 (7.9 sources per episode); W1 42 / 2 (8.0).
            88% of agent-minutes ran on the first open account in list order, and 95% on the minute's busiest
            account, even in minutes with 2 or more accounts open (W0: 13,325 such minutes; W1: 88% and 92%)
    rule:   episode = distinct (account, kind, resets_at to 30 min); agent-minutes from invocation spans
            [completed - duration, completed], all sources; open = not inside an F5 cap interval
    window: W0, W1                            population: pool accounts, all sources
    kind:   ratio
    evidence: allowance/caps.py, allowance/herding.py (herding_out.txt)
```

```
F7. A weekly cap can be forecast from recorded consumption on 5 of the 7 accounts. On those, cost spent since
    the weekly reset reaches about $1.97k at the cap, with a spread of 7-14%. A session cap is much less
    predictable. max-b and max-c are bimodal: some weeks they cap at about a third of the usual amount.
    figure: weekly cost_usd at cap. Per account: n, min, median, max, then CV for cost / tokens / output tokens:
            max-d 8: 1543/2009/2507  0.14/0.14/0.16 | max-e 7: 1677/1978/2136  0.08/0.10/0.07 |
            max-f 6: 1777/1857/2216  0.08/0.05/0.09 | max-g 6: 1830/2001/2257  0.07/0.06/0.07 |
            max-h 3: 1889/1918/2184  0.07/0.07/0.06 | max-b 6: 666/1703/2561  0.39 | max-c 6: 728/1389/2629  0.52
            Pooled excluding max-b/c: n=30, median $1,971, range 1,543-2,507, CV 0.10.
            Median hours from weekly reset to cap: 45 h (n=42; range 17-149).
            Session (5 h) caps: $71-$462 at cap (e.g. max-d n=17: 71 to 305).
            Sum of per-account weekly medians = $12.9k/week. W0 consumption was $12.3k/week (96% of that).
    rule:   cost on that account across all 10 recorded stores, each invocation spread evenly over its span, inside
            [resets_at - 7 d, first detection] (weekly) or [resets_at - 5 h, first detection] (session). Excludes
            episodes whose window starts before 2026-09-15 or whose resets_at lies outside the window (2 parser
            outliers). Consumers the stores do not record are NOT included (see U2)
    window: weekly episodes with a window from 2026-09-15 to 11-10   population: 42 weekly, 78 session episodes
    kind:   measured total per episode; spread = CV (pstdev / mean); $12.9k/week is a projection
    evidence: allowance/predict.py (predict_out.txt)
```

```
F8. Low-tier work dominated what drained the pool before each all-capped stall. Across all 10 stalls, 50-62% of
    the allowance whose absence caused the stall went to tasks now tiered low, and 13-20% to high or critical.
    In W1, 37-45% of that allowance had been spent by the three projects paused later, because most of it was
    spent before the pause.
    figure: lens A (the allowance the stall lacked). Low+polish share per stall: W0 50, 50, 50, 50, 52, 62, 59,
            59%; W1 57% (11-09T16:01, 5.0 h), 52% (11-10T22:37, 11.4 h). Each stall's lens-A pool is
            $11.2k-13.8k, about one week of the whole pool. W1 projects, lens A: dark-factory 26%/31%,
            scoping-intake 22%/21%, paused three 45%/37%, stream-touch 7%/12%.
            lens B (last 6 h before the stall): low 45-74%, but only $320-640, because by then most accounts were
            already capped.
    rule:   A = for each account capped at stall start, cost on that account inside the window of the binding
            episode (F7 windows); B = all pool cost in the 6 h before stall start. Tier as F3 (current tier).
            BOUND: low share is an UPPER bound on what a reserve admitting only medium+ work would have withheld
            before that stall. It is NOT stall time removed: withheld work still needs allowance later, and a low
            task may gate a high one through dependencies
    window: the 10 stalls of 30 min or more in W0 and W1   population: as F2
    kind:   bound
    evidence: allowance/stalls.py (stalls_out.txt)
```

```
F9. No near-cap warning arrived in the window, and nothing can read remaining quota. A forecast would have to
    come entirely from the factory's own accounting.
    figure: near_cap events 0 of 6,492 account_events since 2026-09-15 (failover 3,121, cap_hit 2,513, resumed 625,
            lease_stale 129, auth_failed 88, auth_resumed 16)
    rule:   count by event_type, all 10 stores
    window: 2026-09-15..11-12                 population: 10 stores
    kind:   measured total
    evidence: allowance raw.pkl event tally; shared/src/shared/usage_gate.py::UsageGate.check_at_startup ("The usage
            API (claude.ai/api/oauth/usage) is no longer available"); shared/src/shared/config_models.py::
            UsageCapConfig._reject_legacy_pause_threshold (proactive pause removed)
```

## Mechanisms (from code at 7848173a7d)

M1. **Selection.** `shared/src/shared/usage_gate.py::UsageGate.try_lease` walks `self._accounts` in accounts-file
order. It skips accounts that are capped, have a probe in flight, or failed auth; with a Fable scope it also skips
accounts scope-capped for that model. It returns the first remaining account. A PROBING account is claimed for
exactly one caller (PROBE_IN_FLIGHT). `before_invoke` calls `try_lease` under an `asyncio.Lock`. If nothing is
open it first runs `_refresh_capped_accounts`, which moves an account whose resets_at has passed to PROBING, and
otherwise parks on `_open` until an account reopens. `orchestrator/src/orchestrator/routing.py` picks model,
effort and budget only; it never chooses an account. Nothing here knows the project, task, tier or role.
The scheduler dispatches without asking the gate. The only gate reads in orchestrator code are
`harness.py::...check_at_startup` (a no-op) and `workflow.py`'s scope snapshot. A dispatched task therefore holds
its slot while parked in `before_invoke` (capacity.md F5: 20.6% of W1 slot-hours were cap stalls).

M2. **On a cap.** `UsageGate.detect_cap_hit` routes to `_handle_cap_detected`. That sets acct.resets_at from the
CLI banner (parsed by `shared/src/shared/invocation_outcome.py::_parse_resets_at`), moves the account to CAPPED
through `_transition` (which writes `cap_hit`), and starts `_account_resume_probe_loop`. The loop sleeps
min(`probe_interval_secs`·2^n, `max_probe_interval_secs`, time to resets_at) and fires a one-turn probe
(`_run_probe`). On success it moves the account to PROBING and writes `resumed`; the first real caller then gets
the probe slot. `invoke_with_cap_retry` resumes the capped session (`--resume`) on the next account after a
5 s base cooldown, doubling per full cycle up to 300 s. A near-cap banner sets `near_cap` but does not block. A
Fable cap marks only that (account, model) scope (`scoped_cap_models`).

M3. **Per process, all in memory.** Each process builds its own `UsageGate`: 9 orchestrators
(`orchestrator/src/orchestrator/harness.py`), the fused-memory curator (`fused-memory/src/fused_memory/server/
main.py`), the reconciliation harness (`fused-memory/src/fused_memory/reconciliation/harness.py`, no cost store),
the one-shot legibility pool (`scripts/legibility/account_pool.py`, `wait_for_reset=False`, never probes), and
evals (`orchestrator/src/orchestrator/evals/eval_gate.py`). Account phase, resets_at, near_cap, probe count and
scope caps all live in `AccountState` in that process's memory.
Shared between processes:
- the accounts YAML, read once at config load (`UsageCapConfig._load_accounts_file`);
- the token env vars, from DF's `.env`;
- per-(account, pid) probe config dirs under /tmp, which keep processes apart rather than coordinating them;
- the SIGHUP force-reset, which is per process.

Account events and invocations go to each process's OWN store (`shared/src/shared/cost_store.py`, i.e. that
project's runs.db or curator_events.db). No process reads another's. No lock and no file coordinates selection
across projects. This is why one cap is rediscovered about 8 times (F6). Since 6214, each rediscovery costs one
failed start at $0 plus the 5 s pause.

M4. **Ordering effect.** Because every process walks the same list, the fleet drains one account at a time
(88% of agent-minutes on the first open account, F6). Fleet load piling onto one account explains why W0 hit
session caps (median ~$250 per 5 h) as well as weekly ones. Who gets an account is decided purely by who asks
first. A project's share of the pool is just its demand: concurrent slots times turn cost, with nothing
weighting it.

M5. **Leo's rulings that constrain any design.**
- (a) 2026-10-06, task 5944: max-b and max-h stay listed at the end of the order and are never disabled or
  commented out. The accounts file comment says switching them on at Anthropic's side is "the whole restore".
  Both served in W0, max-h for 17.6% of the pool, so both are currently switched on.
- (b) 2026-11-12, task 8253 (memory file plus `~/.claude/review-all/review-all-dark_factory-20261008/digests/
  leo-rulings-8253.md`), exactly:
  - "The cap file is not built … No process, long-running or one-shot, reads or writes a cap file."
  - No retry-at-reset for the return-brief preparer.
  - The census need not be timely.
  - The 5 s pause is irrelevant.

  Why: after 6214, a one-shot run loses 18-49 s at $0 rediscovering capped seats. The only lost night had all
  seven seats capped, which no memory of caps fixes. "the benefit is seconds".
  How to apply: "Do not propose cap persistence, a cap authority, or shortening the 5 s failover pause again
  without new evidence that a run was lost while a seat was open."

  The ruling targeted cap DISCOVERY. Its words, though ("no process … reads or writes a cap file", "a cap
  authority"), cover any shared record of which accounts are capped. A weight or reserve needs a different
  shared fact: consumption per account since that account's weekly reset. Whether that counts as a
  "cap authority" is Leo's call, not an inference this seat can make.

M6. **What the allowance looks like as a resource** (combining F4, F5 and F7).
- Seven weekly lumps of about $1.97k each (lower on max-b/max-c some weeks) arrive at fixed hours spread
  across the week.
- The fleet spends each lump in a median of 45 h. W0 used 96% of the weekly total.
- Stalls therefore fall wherever the gap between resets is longer than the time to spend the last lump. The
  biggest gap is Mon 21:00 -> Wed 10:00, which matches the stalls of 11-03/04 and 11-10/11.

Two consequences for any weight:
- Total weekly throughput is fixed by the allowance, as long as each lump is spent before its reset (F7: about
  45 h against 168 h). A weight changes only WHO spends it and WHEN.
- Allowance left unspent at a reset is lost, so a reserve held past its account's reset is a real loss.

## Options (analysis, not measurement; no recommendation)

The quantities all four options would draw on:
- the weekly reset time per account (known from resets_at, F4);
- consumption per account since its reset, across ALL consumers (F7; today spread over 10 stores, plus
  consumers no store records, U2);
- a cap estimate per account (F7: about $1.97k, ±10% on 5 accounts).

The arithmetic in each option is a projection from W0 rates.

**(a) Static per-project shares of each account's window.**
- Needs: a per-project budget per account-week, and a running tally of each project's spend against it, i.e.
  cross-process consumption per (account, project), and an action when a project reaches its share: skip that
  account, or park. A project can count its own spend from its own runs.db alone. No cross-project read is
  needed to stop at its own share, only the reset times and cap estimates.
- Guarantees: a floor and a ceiling per project for each week.
- Does not guarantee:
  - urgency within or across projects;
  - full use: an idle project's unspent share is lost at reset unless shares can be borrowed, and borrowing
    puts back the shared state;
  - accuracy while the consumers no store records (U2) and max-b/max-c's bimodal caps (F7) move the real cap.
- Example: W0 shares (F2) as weights gives DF about $3.0k/week. W1 DF consumed at $5.8k/week.
- Rulings: compatible with (a) if each process only reads its own store. A borrowing variant needs a shared
  record.

**(b) Reserve released only to high-urgency work when a cap is forecast.**
- Needs:
  - a pool-level forecast: remaining = Σ over open accounts of (cap estimate − spend since reset), against time
    to the next reset (F4 gives the schedule; the Mon->Wed gap is the predictable danger);
  - consumption from every consumer;
  - one urgency order across projects (tiers are per-project labels today; DF high and reify high are equal
    only by convention);
  - a gate at DISPATCH, not just at `before_invoke`. A low-tier task already dispatched holds a slot and its
    turns finish on the reserve.
- Sizing: high+critical took 15% of W0 ($1.85k/week); high+critical+medium took 35% ($4.3k/week).
- Guarantees: high-tier work keeps running through a forecast stall, if the forecast is right and the reserve
  is sized to the high-tier rate.
- Does not guarantee:
  - fewer total stall-hours for low work;
  - safety from forecast error: ±10%, worse on max-b/max-c and with unrecorded consumers;
  - no waste: a reserve still unspent at its account's reset is lost (M6).
- F8 bounds the pool lens-A allowance (what the stall lacked) that went to low-tier work before each stall at
  50-62%.
- Rulings: a shared forecast of remaining allowance comes close to the "cap authority" Leo declined. It
  differs in purpose (allocation, not discovery) and in what it records (spend, not cap flags). It could be
  computed statelessly by each process reading the existing runs.db files read-only, with no new file. Leo must
  rule whether that is inside the 8253 ruling. max-b and max-h are untouched.

**(c) Dedicating accounts to projects.**
- Needs: a per-project accounts list. The mechanism exists today: `usage_cap.accounts_file` or inline
  `accounts` per project yaml (M3, F1). Nothing shared at runtime.
- Guarantees: complete isolation. One project cannot spend another's allowance. Each project rediscovers only
  its own accounts' caps.
- Does not guarantee:
  - urgency across projects;
  - full use: an account idle at its reset loses its unspent allowance;
  - a smooth division: with 7 accounts and 7 consumers, weights come in sevenths, and max-b/max-c are worth
    less in some weeks.
- Example: DF's W0 share (24.7%) is about 2 accounts of 7. Its W1 rate (about $5.8k/week) is about 3.
- Rulings: compatible on persistence. Fixing which project lists max-b/max-h is fine as long as they stay
  listed somewhere and are never commented out; the ruling's wording is about the shared file, so moving them
  into a per-project list needs Leo's reading.

**(d) Urgency-ordered consumption with no shares.**
- Needs: one cross-project order of pending agent work and a single point that admits the next invocation in
  that order. Every gate and scheduler would consult one ranking, so this needs shared state at dispatch even
  without a forecast. Without a forecast it reorders only within a stall's queue: who gets the account the
  moment it reopens.
- Guarantees: the most urgent work is served first whenever demand exceeds supply.
- Does not guarantee:
  - any share for a low-weight project: starvation risk, given low tiers dominate (F3);
  - help before the pool runs dry. Without a forecast, high work still stalls when the pool hits zero, because
    low work spent the lump first (F8).
- Rulings: a cross-project admission point is the largest departure from "no shared cap state". It also
  resembles host-wide verify admission, which was ruled out for CPU (lead's brief; outside this seat).

**Cheap adjunct, independent of all four:** spreading selection across open accounts instead of first-in-list
(M4) would mostly remove session caps (W0's session stalls). It would neither add weekly allowance nor change
any weight.

## Unknowns

- U1. **What the cap meters.** cost_usd, total tokens and output tokens are equally good predictors (F7).
  Anthropic's metering is not observable here.
- U2. **Unrecorded consumers.** The reconciliation harness's gate has no cost store
  (`fused-memory/src/fused_memory/reconciliation/harness.py`, `UsageGate(self.config.usage_cap)` without
  cost_store). The legibility census, return brief, evals and usage probes write no invocation rows here.
  Their spend is inside every cap but outside F2 and F7. F7's 7-10% spread on 5 accounts bounds their
  week-to-week VARIATION, not their level.
- U3. **max-b and max-c bimodal caps.** Weeks capped at $666-$1,126 against about $2k otherwise:
  max-b in the 10-09, 10-30 and (partly) 11-06 windows; max-c in 10-07, 10-28 and 11-04. Cause not determined.
  Candidates: a plan tier change, the 2026-09-29/10-06 org-disable episodes, or use outside these stores.
  This makes them unforecastable without more evidence.
- U4. **DST.** resets_at is parsed from "(Europe/London)" banners. Reset hours show the same UTC clock before and
  after 2026-10-25 (e.g. max-d 13:00 on 10-22 and 10-29). Either the reset is fixed in UTC or the parser drops
  the zone. October episode windows may be off by 1 h; November ones are unaffected.
- U5. **Cost attribution across a cap.** One row per call is attributed to the final account. If the CLI's
  cost_usd after `--resume` includes the capped attempt's spend, that spend is credited to the wrong account.
  Rows flagged capped: 71 of 16,661 in W0, 18 of 2,799 in W1. Effect presumed small; not measured.
- U6. **Tier is today's tier** (F3, F8). Re-tiering since dispatch is not corrected for. Per-task priority
  overrides (`scheduler_overrides.db`) were not applied.
- U7. **Pre-2026-10-12 cap rows carry only a free-text label** (no reason or resets_at). Episodes before that
  date come only from rows in the newer format.
- U8. **Whether a shared consumption forecast falls under Leo's 2026-11-12 "no cap file / no cap authority"
  ruling** is a ruling question. It is reported here, not decided (M5).
