# Capability manifest — plans/write-triage-flip-readiness-prd.md

Binds each leaf signal's asserted capabilities to evidence (G3 + G6), as of main `ca06d49340` (2026-09-23). Mechanical `delivered_check`s live in the YAML twin and are copied onto producer tasks by `commit_planning`. The YAML verdict vocabulary is `PASS | FAIL | OPEN`; the nuance ("on landing", "provisional", "deferred") lives in this column.

| leaf | capability | binding | verdict |
|---|---|---|---|
| κ1 | retrieved-slate eval + production-parity recall + aliases | producer: branches `iact/wt-option-c-harness` @ 72aba0725b, `iact/wt-option-c-recall` @ 695f808694; 354 + 225 tests green on trees 12 commits behind main — re-verify after rebase | PASS (on landing) |
| κ1 | committed artifacts carry `production_shape` and `retrieval_mode` | grep after landing (YAML) | PASS (on landing) |
| Γ1 | gate predicate script exists and is executable | `scripts/check_write_triage_readiness_gate.py`, 8 tests, committed with this PRD | PASS |
| τ | `MemoryService.search(anchor_topics=True)` promotes a topic-tagged canonical | task 3111 landed `f62767929b`; measured 48/63 rank-1 in legacy mode 2026-09-23 | PASS |
| τ | a pinned record can be scored explicitly | embedder reachable from the write path (convergence PRD §6) | PASS |
| ρ1 | at least one D1 arm reachable without new credentials | gpt-4o-mini pairwise uses the existing openai arm | PASS (skipped arms recorded, never silent) |
| Γ2 | thresholds achievable | the gate task's `before_done.args` is the home; filing-time basis: cosine rank-1 0.19, recall@20 0.79 (§9) | OPEN (provisional) |
| ρ2 | fail-open shares the triage storm counter | `write_triage.py::TriageFailOpenCounter` | PASS |
| σ | judge credentials resolve through one function | `write_triage_judge.py::_provider_credentials` | PASS |
| μ | anthropic arm exists for a claude-haiku row | `write_triage_judge.py::_KNOWN_PROVIDERS` | PASS |
| μ | judge_candidate_count overridable per run | κ1's `--judge-candidate-count` (in-memory) | PASS (on landing) |
| Γ3 | thresholds achievable and jointly feasible | args are the home; filing-time basis §9 (0.79 × 0.653 = 0.516; 38 strict of 75 with ≤ 76 attaches) | OPEN (provisional) |
| ν | Jev API shape | vendor docs: `POST /v1/systemone`, `choice` ≤255 options, 64k/32k budgets; early access, key required | OPEN (deferred) |
| Γ4 | flip predicate passes on main today | run 2026-09-23: exit 0 in 6.3 s | PASS |
| Γ4 | 4949's consumption probe | task 4949 in-progress (mid-merge 2026-09-23); Γ4 depends on it | PASS (producer upstream) |

No binding resolves to declared-only, producer-downstream or rejection-absent. The OPEN rows are numeric thresholds whose gate escalation carries the observed values for the operator to re-base, and one deferred leaf.

## Amendment 2026-10-01 (PRD §11), as of main `1e6a7074c0`

Γ2 and ρ2 are retired (D12) and ν is retired (D13); their rows above stand as the record. μ's bindings are replaced; Γ3's basis is re-based (D15). New leaves:

| leaf | capability | binding | verdict |
|---|---|---|---|
| ι | one definition of the flip metrics; refuses on unrated pairs | `score_write_triage_pairs.py::score_pairs` (YAML grep); the judged-candidate column is produced upstream by task 6007 | PASS (on landing) |
| ω | openai arm posts the Responses API; `judge_reasoning_effort` hot-reloadable | measured 2026-09-30: luna/terra/sol reject `max_tokens` and answer on `responses.create`; leaf registered in `reload.py` (YAML greps) | PASS (on landing) |
| ψ | the `contests` clause is the ruling's wording, one home | `JUDGE_SYSTEM_PROMPT` (YAML grep); wording × model measured on 359 hand links | PASS (on landing) |
| χ | briefing renders a contesting child; recon marker writes exempt | `briefing.py::render_memory_results` has no contested branch today; `write_triage.py::declares_attach_keys` exists | PASS (on landing) |
| π | population ≥ 300 judge-band writes; slates at production parameters | 591 qualifying writes on 2026-10-01; `prefetch_retrievals` takes any record list | PASS (on landing) |
| λ | every judge-named pair rated | operational; Γ3's `unrated_pairs ≤ 0` makes incompleteness a fail, not a pass | OPEN (operational) |
| μ | best_config carries ι's quality block | YAML grep on `false_contested_rate` | PASS (on landing) |
| Γ3 | D15 thresholds achievable | every bound sits on the far side of a measured arm (Sol/Opus) and on the near side of today's judge | OPEN (provisional) |
| Γ4 | population bound replaces the fixture bound | args edit only; never cancelled (D4) | PASS |

No new binding resolves to declared-only, producer-downstream or rejection-absent.
