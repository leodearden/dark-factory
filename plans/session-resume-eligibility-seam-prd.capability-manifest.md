# Capability manifest — session-resume-eligibility-seam-prd

Binds each leaf's user-observable signal to substrate evidence, mechanizing
G3 + G6 once here instead of once per task at dispatch.
PRD: `plans/session-resume-eligibility-seam-prd.md` (committed `fa070f8913`).
Built 2026-08-05 during `/prd decompose`. Machine-readable twin:
`plans/session-resume-eligibility-seam-prd.capability-manifest.yaml`.

Greek labels follow PRD §10: α, β, γ, δ, ε.

Every `grep:` citation below was re-run against `main` on 2026-08-05; the
PRD's own §4 pre-condition table was re-verified line by line and is correct
except where a finding says otherwise.

---

## Gate findings resolved during this decompose

Ten bindings did not clear on the PRD's own text. All resolutions are carried
into the filed task descriptions, not just recorded here.

**F1 — α: PRD §4's "α touches only `shared/` and is clear [of 3256]" is FALSE
(`DAG-direction` hazard).**
§4's live-work-hazard paragraph asserts α is clear of task 3256's `harness.py`
edit. α's own §10 entry contradicts it (*Modules:* `shared/src/shared`,
`orchestrator/src/orchestrator` (event field only)), and §10 is the correct
one: the **sole** `session_resume_fallback` emission site is
`harness.py:7663-7668`, which sits inside 3256's declared edit region
`harness.py:7649-7674`. There is nowhere else to attach `archive_available`.
**Resolution:** α additionally `depends_on` **3256** — the same resolution the
sibling `transcript-preservation-seam` manifest reached for its own leaf 2
(its finding F3). D9's latency argument survives: α is one already-high-priority
task deep, not behind the 3618 → 3619 → 3578 chain.
*Non-finding, checked:* α and γ do **not** collide inside `shared/`.
`shared/src/shared/__init__.py` re-exports neither `transcript_archive` nor
`config_dir` symbols (verified — `archive_task_transcripts` is imported
module-qualified at `git_ops.py:73` and `workflow.py:46`), so α adds
`durable_archive_path` to `transcript_archive.py` without touching
`__init__.py` or `shared/tests/test_public_api.py`.

**F2 — δ: the signal asserts an end-to-end capability whose critical leg is
explicitly unverified (`producer-extent-short`, G6 branch 3).**
δ's signal requires the claude CLI to accept `--resume` against a transcript
whose recorded cwd has moved. PRD §6 puts that HARD GATE out of scope and
Open question 2 records why it may never be exercised: 3578's own amendment A4
makes dark-factory the validation case *precisely because the cwd-move question
is vacuous there*, so 3578 can close green having proven only same-path restore.
A signal whose truth depends on an upstream deliverable that may not cover the
needed extent is the `producer-extent-short` shape.
**Resolution (G6-b, weaken to what is achievable + bind the rest as an
acceptance criterion):** δ's signal becomes two-tier —
*(i)* **achievable tier, dark-factory:** an `EventType.session_resume` appears
in dark-factory's own `runs.db`, where the lifetime count is **0** across 217
fallbacks (task 3221, re-verified 2026-08-05). Non-zero is a real, unambiguous,
same-path signal that the archive-mediated predicate works at all.
*(ii)* **cross-lane tier, reify:** the §1 signal as written, **conditioned on
3578's HARD GATE answer**. δ carries an explicit acceptance criterion: obtain
that answer from 3578's outcome and, if the CLI rejects a moved cwd, **report
it and stop** rather than working around it — the instruction 3578's own text
already gives, hoisted to the task that consumes the answer.

**F3 — β: INV-4 (`storm-escape-required`) hit — recorded as a WAIVER, not as
"satisfied" (G7).**
PRD §12 marks INV-4 "Satisfied, with the interim window named", resting on α's
`archive_available` rate as the interim loud signal. That over-claims: INV-4's
checkable question is "if this fallback fires 100× in an hour, **who hears about
it, and via what counter?**". α ships a queryable *field on an event*; nothing
*fires*. Between β and ε the honest answer is **nobody**.
**Resolution:** recorded as `metadata.g7_waivers` on β with the PRD's own
rationale — the window is bounded by ε, the fallback path stays fully
observable throughout, and the alternative (retaining a storm that fires on
by-design outcomes and misdirects the operator to NTP) is strictly worse.
Additionally pinned in β's text: β **must not delete**
`_file_session_resume_storm_escalation` (`harness.py:6059`). The mechanism is
retained and only its feeder changes; ε re-points it. Without this pin β's
"the storm stops firing" signal is satisfiable by deleting the escalation.

**F4 — γ: D7's second conjunct goes VACUOUS if the reaper is sited one boot
step later (`corroborate-before-acting`).**
MEASURED in `harness.run()`: step **2c** `_recover_crashed_tasks()`
(`harness.py:2362`) runs **before** step **2d**
`_reconcile_stranded_in_progress()` (`harness.py:2372`), which sweeps stranded
`in-progress` tasks back to `pending`. Sited at 2c — where PRD §10 puts it,
adjacent to `_adopt_recovered_session` — the "task is not in-progress"
conjunct reads pre-reconcile ground truth and is meaningful. Sited **after** 2d
every task reads not-in-progress, the conjunct silently evaluates true for
everything, and the reaper eats the sidecars of tasks the orchestrator is about
to legitimately re-dispatch — exactly the hazard D7's second conjunct exists to
prevent.
**Resolution:** the ordering constraint is pinned in γ's text as a hard
acceptance criterion plus a test that fails if the reaper is moved after 2d.

**F5 — I-E / B7 (`sole locator`) has no enforcing task in the batch.**
§9 B7 is a consumer-side grep that lands "when 3578's restore path landed", but
§10 assigns it to no leaf, so nothing in the batch fails if a second finder
appears.
*Verified on main today:* I-E **holds** — the two existing archive readers do
whole-tree enumeration, not session-id-keyed lookup
(`fused-memory/scripts/memory_eval_transcript_corpus.py:697`
`root.glob('*/**/*.jsonl.gz')`; `scripts/legibility/inventory.py:420`
`root.rglob('*.jsonl*')`). The risk is entirely that 3578 adds one.
**Resolution:** B7 assigned to **δ** — the only batch task downstream of both α
and 3578 — as an explicit acceptance criterion, with a mechanical
`expect: absent` delivered-check for a session-id-interpolated archive glob
outside `shared/transcript_archive.py`.

**F6 — 3256 and 3221 carry premises this PRD refutes; amended in place.**
PRD §7 flags both ("premise drift"; "overlapping and now partly stale …
flagged for reconciliation at decompose, not silently duplicated") but §10
files no amendment leaf for either. Left alone, 3256 dispatches against a
discriminator (`reason == 'no_transcript'`) that selects **zero** reify events,
and 3221 dispatches against a refuted bound.
**Resolution:** both amended in place at decompose (recording the refutation
and pointing at β), alongside the §7-directed 3161 B6 amendment. No new leaves —
these are bookkeeping amendments, exactly as §7 asks.

**F7 — γ: D7's "import `shared.config_dir`'s `_pid_alive`; do not add a fifth
copy" is off by one AND points at the wrong copy (`no-lockstep-duplication`).**
MEASURED: there are already **five** production copies, not four —
`shared/src/shared/config_dir.py:51`,
`fused-memory/src/fused_memory/services/orchestrator_detector.py:119`,
`orchestrator/src/orchestrator/session_registry.py:1076`,
`orchestrator/src/orchestrator/task_ground_truth.py:186`, and
**`orchestrator/src/orchestrator/harness.py:466`** (plus one in
`orchestrator/tests/test_harness_verify_scope_reaper.py:259`). `config_dir.py`'s
own docstring still says "the fourth instance"; a fifth landed after it.
Decisively: `harness.py` — the file γ edits — **already defines `_pid_alive`
itself at :466**, byte-identical in semantics to `config_dir`'s including the
load-bearing `PermissionError → alive` branch. Following D7 literally would add
a cross-package import into a module that already has the helper, leaving two
liveness paths in one file.
**Resolution:** γ uses **`harness._pid_alive` (`harness.py:466`)**. This
satisfies D7's *intent* (no new copy) more cleanly than D7's *instruction* — no
new copy **and** no new import edge — and it removes γ's only `shared/` edit, so
γ becomes pure-`orchestrator/`. Promoting `shared.config_dir._pid_alive` to
public is dropped from γ's scope as unnecessary. The four-into-five count
correction is recorded here; deduplicating the copies stays out of scope per §6.

**F8 — β must depend on α: same-region collision, and it is D8's own ordering.**
α adds `archive_available` to the event `data` dict at `harness.py:7667`; β
rewrites the surrounding branch structure at `harness.py:7649-7674`. Same ~25
lines. The PRD's DAG has both depending on 3256 but not on each other, while
D8 ("instrument before acting") states the sequencing in prose: "α … reports
its answer on the existing fallback event, and changes no behaviour. β then
fixes the reason semantics."
**Resolution:** β additionally `depends_on` **α**. Encodes D8 in the DAG rather
than in prose, and guarantees β's implementer sees the field already present.

**F9 — γ must depend on 3619: both add a sweeper to the same boot scan, and γ
can strip the identity 3619 needs.**
PRD §10's γ note asks decompose to "confirm the edge if 3619 is still open when
γ dispatches". 3619 is **open** (`pending`, deps 3618 + 3256). Both hook
`_recover_crashed_tasks`: 3619 adds a startup sweeper that archives transcripts
for the SIGKILL tail; γ clears sidecars for dead owners. The interaction is
concrete, not merely adjacent — the `agent_session.json` sidecar is what
carries `session_id`, so a γ reap that runs before 3619's sweeper can remove
the session identity the sweeper needs to name its archive.
**Resolution:** γ `depends_on` **3619**. Cost stated plainly: γ now waits behind
3618 → 3619, so the zombie-sidecar drain lands later than the PRD's
value-front-loading paragraph implies. Accepted because β alone already stops
the false storm; γ's value is draining the fuel, which is real but not urgent
once the L1 is quiet.

**F10 — 3578 must depend on α, or §1's whole G1 argument is unenforced.**
PRD §1 says naming the seam here "is what stops 3578's architect inventing a
second, divergent glob (INV-5)", and §7 records 3578 as consuming
`durable_archive_path()`. But 3578 is already filed with deps `[3256, 3619]`
and no edge to α. A PRD paragraph does not gate a dispatch; only an edge does.
Absent the edge, 3578 can dispatch before α exists and will write its own
finder — the exact lock-step duplication D6 and I-E exist to prevent.
**Resolution:** `3578 depends_on α` wired as a real edge, and 3578 amended to
name `durable_archive_path()` as the locator it must call rather than
re-implement. No cycle: α → 3256; 3578 → {3256, 3619, α}.

---

## α — `durable_archive_path` lookup + `archive_available` on the fallback event

*Deps:* 3256 (F1). *Consumers:* task 3578 (restore), δ (eligibility), operators.

| Capability the signal asserts | Binding | Verdict |
|---|---|---|
| An archive lookup helper keyed on `(task_id, session_id)` | capability→producer — **built by α**. Confirmed absent today: `transcript_archive.py`'s only public entry is `archive_task_transcripts` (`:128`); the module is write-only | PASS |
| The archive path is derivable and cwd-globbable | grep:`shared/src/shared/transcript_archive.py:110` — `dest = archive_root / task_id / rel.parent / (rel.name + '.gz')`, **wired on the production write path** (called at `git_ops.py:11788` and `workflow.py:11860`, not only from tests) | PASS |
| Format-agnostic across the gzip drop (I-C) | grep:`transcript_archive.py:110` writes `.gz` today; task **3618** flips the writer to plain `.jsonl` and bulk-gunzips (3578 amendment A3). I-C matching `.jsonl*` spans both, which is what removes any α→3618 edge | PASS |
| A `session_resume_fallback` emission site to carry the new field | grep:`orchestrator/src/orchestrator/harness.py:7663-7668` — `EventType.session_resume_fallback` emitted in `_run_slot`, the production dispatch path. **Sole site.** DAG-direction hazard vs 3256 resolved by F1 | PASS (dep wired) |
| `archive_root` resolvable at that emission site | grep:`harness.py:2108` `self.config.project_root`; `config.py:3975` `transcript_archive: TranscriptArchiveConfig`; precedent composition at `git_ops.py:11785` `self.project_root / self.transcript_archive.root` | PASS |
| `task_id` + `session_id` in scope at the emission site | grep:`harness.py:7646` `task_id=assignment.task_id`; `:7638` `recovered_session.get('session_id')` — both already in the event payload | PASS |
| Numeric premise: the 63.5% figure | **Not a bound α must hit** — §2's 125/197 is a *reference measurement*, and 3619 will move it deliberately. G6 branch 1 N/A; α's signal is rewritten to assert the field is **populated and queryable on every fallback event**, with 63.5% named as the 2026-08-05 reference, not a target | PASS |

## β — composite reason reporting + by-design carve-out from the storm streak

*Deps:* 3256, α (F8). *Consumers:* operators triaging the storm L1; δ; ε.

| Capability the signal asserts | Binding | Verdict |
|---|---|---|
| A by-design carve-out precedent to extend | grep:`harness.py:7653-7661` — `capped` already gets its own event and explicitly does **not** feed the streak; `config.py:868` documents it as by-design throttling. D4 extends this rather than inventing a mechanism | PASS |
| The predicate can report all true reasons | grep:`harness.py:3060-3110` `_session_resume_eligible` — four independent predicates already evaluated in one body, each currently `return`-ing on first match. Composite reporting is a restructure of existing checks, no new substrate | PASS |
| Single call site to update for the signature change | grep:`harness.py:7633` — the only production caller; `orchestrator/tests/test_session_resume_integration_gate.py:814` references it in prose only | PASS |
| The storm mechanism survives the carve-out | grep:`harness.py:6059` `_file_session_resume_storm_escalation`, wired from `harness.py:7674`. Pinned `expect: present` so β cannot satisfy "the storm stops firing" by deleting it (F3) | PASS |
| P1/P2's co-occurrence population is real | MEASURED (§2) — 25/25 `stale`, 0 `no_transcript` on reify's current boot, over a population whose transcripts are also absent; the flip traced on five tasks (5848, 5893, 5766, 5344, 5238) | PASS |
| INV-4 during the β→ε window | **G7 WAIVER** — see F3. No counter fires between β and ε | WAIVED |

## γ — sidecar reaper: dead owner AND task not in-progress

*Deps:* 3256, 3619 (F9). *Consumers:* the eligibility path (fuel supply for P1/P2); operators.

| Capability the signal asserts | Binding | Verdict |
|---|---|---|
| A pid-liveness helper reachable without a new copy or import | grep:`orchestrator/src/orchestrator/harness.py:466` `_pid_alive` — **already in the file γ edits**, semantics byte-identical to `config_dir.py:51` including `PermissionError → alive`. Supersedes D7's instruction; see F7 | PASS |
| The sidecar carries `owner_pid`, populated non-sentinel on the production path | **field-population check** — grep:`artifacts.py:50` declares `owner_pid: int`; `artifacts.py:1086` writes `owner_pid=os.getpid()` on the production sidecar write. A real pid, not a placeholder | PASS |
| A sidecar-clearing primitive to reuse | grep:`harness.py:2967` `_clear_recovery_artifact` — clears from **both** the `.task-meta` and legacy roots. γ reuses it rather than unlinking | PASS |
| Live task status readable at the reap site | grep:`harness.py:3250` `await self.scheduler.get_status(rec.task_id)` and `:3331`/`:3587` `await self.scheduler.get_task(...)` — already the established idiom **inside `_recover_crashed_tasks`**. Bulk form available at `harness.py:2436` `self.scheduler.get_statuses()` | PASS |
| The "not in-progress" conjunct is non-vacuous where sited | **ordering constraint, F4** — step 2c `_recover_crashed_tasks` (`harness.py:2362`) precedes step 2d `_reconcile_stranded_in_progress` (`harness.py:2372`). Pinned as an acceptance criterion + test | PASS (pinned) |
| The zombie population exists | MEASURED (§2) — 16 surviving sidecars on reify, 13 already stale, oldest 176h (task 5730); `owner_pid` recorded but never liveness-checked anywhere in the tree | PASS |
| INV-2/INV-4 on the reaper itself | **resolved, not waived** — γ emits a structured per-reap fact (sidecar path, `task_id`, `owner_pid`, age) plus a per-boot reap count, so a mass reap is countable rather than silent | PASS |

## δ — reachability outranks freshness, with a derived outer bound

*Deps:* α, β, 3578. *Consumers:* reify + dark-factory dispatch paths; operators.

| Capability the signal asserts | Binding | Verdict |
|---|---|---|
| The archive lookup | producer:**task-α**, upstream in the transitive closure. Extent matches exactly: δ needs `Path \| None` for `(task_id, session_id)`, which is α's whole deliverable | PASS |
| Restore of the located transcript into the config dir | producer:**task-3578**, upstream. In 3578's declared scope ("Make the dispatch path RESTORE the relevant `.jsonl` into the freshly created config dir before arming `--resume`") | PASS |
| The CLI accepts `--resume` against a moved cwd | **`producer-extent-short` risk — resolved by F2.** Signal re-cast two-tier: achievable dark-factory tier (a `session_resume` where lifetime = 0 / 217 fallbacks) + reify cross-lane tier conditioned on 3578's HARD GATE, with report-and-stop as an explicit acceptance criterion | PASS (re-scoped) |
| D3's absolute outer bound | **G6 branch 1 — no guessed threshold is shipped.** PRD Open question 1 requires deriving it from observed in-flight duration in `runs.db`, pinning the derivation not the number, mirroring **task 3621**, whose own text carries the "MUST NOT BE VACUOUS" rule (a rate that is itself a hardcoded constant makes the assertion arithmetic on literals). Verified 3621 is filed and `pending` | PASS |
| `session_resume.*` tunable without a restart | grep:`config.py:4929` `_submodel_leaf_paths('session_resume', SessionResumeConfig)` — green-tier whole-submodel group in `RELOADABLE_FIELDS`, so the bound is hot-appliable | PASS |
| I-E sole-locator holds after 3578 lands | **B7, assigned to δ by F5.** Holds on main today (both existing readers do whole-tree enumeration, not session-id lookup); mechanically checked `expect: absent` for a session-id-interpolated archive glob outside `shared/transcript_archive.py` | PASS |

## ε — re-point the storm streak at eligible-but-FAILED

*Deps:* β, 3578. *Consumers:* operators triaging the storm L1.

| Capability the signal asserts | Binding | Verdict |
|---|---|---|
| A genuine-breakage feeder population exists | producer:**task-3578**, upstream. Its own signal declares it verbatim: "every CLI-level resume failure emits an event, so the fallback rate is measurable rather than journal-only". Extent confirmed against 3578's text, not name-matched | PASS |
| The storm escalation mechanism to re-point | grep:`harness.py:6059` `_file_session_resume_storm_escalation` + `harness.py:7674` call site; dedup via `has_open_l1` at `:6076` | PASS |
| The escalation's prose currently misdirects | grep:`harness.py:6103-6110` — the `detail` says "Check host clock skew (NTP)…" and `suggested_action` says "Investigate clock skew (NTP)…". This is the operator-visible half of the signal; ε replaces it with structured facts naming the failed restore (INV-2). Checked `expect: absent` on the NTP literal | PASS |
| The injected-fault path is reachable | capability→producer — ε builds the fault-injection gate over the restore path 3578 delivers; the by-design half (a boot of pure by-design fallbacks files nothing) is β's carve-out, upstream | PASS |

---

## Amendments filed at decompose (not leaves)

| Task | Amendment | Basis |
|---|---|---|
| **3161** | Narrow boundary test B6 per D1 — keep "lane B sees no sidecar" verbatim as the credential/state-isolation assertion; replace "and no resume is attempted against lane A's session" with "and any resume that does occur is mediated by the task-keyed durable archive, never by the lane's sidecar". Record that the original clause encoded a then-true status quo, not a designed invariant | PRD §7 (directed) |
| **3256** | Record P1/P2: its `reason == 'no_transcript'` discriminator now selects **zero** reify events (25/25 `stale` on the current boot); `stale` short-circuits ahead of the transcript check and masks it. β supersedes the single-string approach with D5's composite | PRD §7 + F6 |
| **3221** | Record that its bound "only ~14 were `stale`, which bounds how much corroboration work could ever recover" is **refuted** (39 lifetime, 25/25 this boot), and that its root-cause scope is now covered by 3619 + 3578 | PRD §7 + F6 |
| **3578** | Name `durable_archive_path()` (α, §8) as the locator it must call rather than re-implement; new blocking edge on α | F10 |
