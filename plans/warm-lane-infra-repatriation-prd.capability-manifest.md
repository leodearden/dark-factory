# Capability manifest — warm-lane infrastructure repatriation

Binds every leaf signal's asserted capabilities to evidence, mechanizing G3 + G6.
Machine-readable twin: `warm-lane-infra-repatriation-prd.capability-manifest.yaml`.
PRD: `plans/warm-lane-infra-repatriation-prd.md` (committed `264968127a`).

**Verdict summary: 18 bindings, 18 PASS, 0 blocking.** The one binding that would
have failed as `declared-only` — `LaneState.IN_USE` — is homed on leaf **δ** as its
*producer*, and every downstream consumer (ε, and γ's B4/B5 rows) is wired to δ by a
real dependency edge rather than assuming the capability already exists.

## Substrate findings that shaped the bindings

| Capability | Status at authoring | Consequence |
|---|---|---|
| `<worktree_base>/.lane-state/<lane>.json` durable records, readable from bash | **CONFIRMED** — 56 live records read 2026-07-26; `{state, task_id, branch, seeded_from_sha, updated_at}` | β binds directly; no oracle-callback needed |
| `LaneState` distinguishes free from assigned | **CONFIRMED** — `lane_lifecycle.py:55-63`; census 55 × `assigned`, 1 × `released` | γ's preserve predicate is real today |
| `LaneState.IN_USE` (assigned-**building** vs assigned-**idle**) | **DECLARED-ONLY** — in the enum (`:61`) and transition table (`:77-78`), read in `harness.py`, **zero writers**; 55/55 live records are `assigned` | δ becomes a named producer leaf and a hard prereq of ε. Without it ε makes every assigned lane permanently unreclaimable and re-creates the 2026-07-10 ENOSPC outage |
| Six `project_root/scripts/<name>.sh` resolution sites | **CONFIRMED** — `git_ops.py:3701, 3765, 3816, 3922, 3970, 4095` | α's preference-order change has an exact footprint |
| Absent-script path is silent | **CONFIRMED DEFECT** — `git_ops.py:3923-3926` `logger.debug` + rc 127 | α must make it loud (B8); this is the migration landmine |
| systemd ExecStart is a hardcoded reify path | **CONFIRMED** — `reify/deploy/systemd/reify-warm-lane-gc.service:28` | η is a one-line change plus installer sed |
| `refresh-warm-base.sh` RUSTFLAGS coupling | **CONFIRMED** — `:26, :107, :121, :136, :444` | θ's generalization has an exact footprint |

## Bindings

### α — relocate the seven generic scripts, with resolution preference order

| Capability | Binding | Verdict |
|---|---|---|
| `df-ships-generic-warm-lane-scripts` | capability→producer — α creates `orchestrator/scripts/warm-lane/*.sh` from reify's copies (0 toolchain tokens each; §2.1 audit) | PASS |
| `script-resolution-preference-order` | capability→producer (wired) — six existing resolution sites at `git_ops.py:3701,3765,3816,3922,3970,4095` are the exact edit set | PASS |
| `absent-script-no-longer-silent` | rejection-mechanism — current silent path confirmed at `git_ops.py:3923-3926` (`logger.debug`); α replaces it with a logged fact (B8) | PASS |

### α2 — port the generic bash tests into dark-factory

| Capability | Binding | Verdict |
|---|---|---|
| `generic-warm-lane-bash-tests-run-in-df` | capability→producer — α2 ports the subset of reify's 23 `tests/infra/test_warm_lane_*.sh` covering relocated scripts; reify's originals stay green until κ | PASS |

### β — `lib_lane_state.sh` (dark-factory-authoritative data, readable from bash)

| Capability | Binding | Verdict |
|---|---|---|
| `durable-lane-record-readable-from-bash` | substrate CONFIRMED — `.lane-state/<lane>.json` verified live 2026-07-26 (`_lane-28` → `assigned 5551`, `_lane-50` → `assigned 5416`) | PASS |
| `recordless-lane-returns-unknown` | capability→producer — β's fail-open branch; recordless dirs (`_iact-*`) exist in the live pool today | PASS |
| `protected-prefixes-rendered-not-mirrored` | INV-5 extraction — current lockstep duplication confirmed at `warm-lane-gc.sh:318` (comment: "mirrors dark-factory's PROTECTED_PREFIXES … only ever grows") | PASS |
| `audit-reader-unified-not-duplicated` | INV-5 extraction, **extract-after** — reify 5363 lands first (already implemented, merge-pending on `_merge-verify.lock` contention since 2026-07-26T16:51) shipping its own `.lane-state` reader in `warm-lane-audit.sh`; β folds it into the shared helper rather than 5363 consuming β. Direction corrected at decompose | PASS |

### γ — reclaim consults the durable record per lane, before reset, under the flock

| Capability | Binding | Verdict |
|---|---|---|
| `assigned-lane-preserved-any-entry-point` | capability→producer, β upstream (wired) — closes esc-5334-6; B1 | PASS |
| `per-lane-recheck-not-snapshot` | INV-3 — B2 is the executable guard; D5 forbids `--assigned-lanes CSV` marshalling precisely because it re-creates the TOCTOU reify 5572 fixes | PASS |
| `free-lane-still-reclaims` | non-regression of task 5326 — B3; the ENOSPC accretion path must stay open | PASS |
| `recordless-lane-falls-back-to-proc-scan` | capability→producer **upstream: reify 5572** (external dep wired) — γ inherits and supersedes 5572's scan, retaining it for recordless lanes; B6 | PASS |

### δ — populate `LaneState.IN_USE`

| Capability | Binding | Verdict |
|---|---|---|
| `lane-state-in-use-written` | **declared-only TODAY → producer:δ.** Enum + transition table exist; zero writers; 55/55 live records `assigned`. δ IS the producer, so this binds PASS here and would have been a blocking `declared-only` FAIL on any downstream leaf that assumed it | PASS |

### ε — whole-assignment lease

| Capability | Binding | Verdict |
|---|---|---|
| `lease-held-across-whole-assignment` | capability→producer — generalizes DF 3027's two `task_verify_lease` sites (`workflow.py:3330, 7117`) to acquire→release; B5, B10 | PASS |
| `assigned-idle-still-reclaimable` | producer:δ upstream (wired) — B4 **and** B3 must be green in the same run as B5; this is the 2026-07-10 ENOSPC regression guard | PASS |
| `pressure-degrades-to-backpressure-not-corruption` | rejection-mechanism (INV-4) — B11 asserts reclaim resets **nothing** at the floor and `warm-lane-disk-guard.sh` exit-75 blocks admission; there is deliberately no reclaim-an-`in_use`-lane override | PASS |

### ζ — cutover readiness (operational)

| Capability | Binding | Verdict |
|---|---|---|
| `df-copies-confirmed-live-before-deletion` | producer:α upstream (wired) — the go/no-go reads α's resolved-path INFO line; this is the ordering that defuses the migration landmine | PASS |

### η / θ / ι / κ — reify-side cutover

| Capability | Binding | Verdict |
|---|---|---|
| `systemd-unit-points-at-df` | substrate CONFIRMED — hardcoded ExecStart at `reify/deploy/systemd/reify-warm-lane-gc.service:28` | PASS |
| `build-fingerprint-generalized` | substrate CONFIRMED — RUSTFLAGS coupling at `refresh-warm-base.sh:26,107,121,136,444` is the entire footprint (7 tokens) | PASS |
| `docs-no-longer-claim-reclaim-time-enforcement` | rejection-mechanism (`expect: absent`) — CLAUDE.md currently claims one-consumer-per-lane "enforced at RECLAIM time", which RC-1 falsified | PASS |
| `reify-scripts-reduced-to-contract-primitives` | rejection-mechanism (`expect: absent`) — the seven relocated scripts absent from reify; `verify-pipeline-paths.txt` no longer lists them | PASS |

## Cross-repo note

Leaves η, θ, ι, κ are filed in **reify**'s tracker; α, α2, β, γ, δ, ε, ζ in **dark-factory**'s.
Cross-repo edges are qualified `project_id:task_id` `depends_on` refs routed to
`metadata.external_deps` and gated at dispatch via `get_external_statuses`.
`commit_planning` stamps sidecar labels only for the project whose checkout holds the
PRD, so reify-side labels are stamped by hand in the same turn.
