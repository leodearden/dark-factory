# Capability manifest — Coupling-tolerant train former (Lever A′)

Mechanizes G3 + G6 per task. Evidence verified 2026-06-09 against `main` (cite-by-symbol).
**No binding resolves to a FAIL value → batch is not blocked.**

**CORRECTED 2026-06-09 (PRD §4 D7).** Of the two economic premises, **union ≈ single is resolved a priori** —
reify's merge gate is unconditionally full-`--scope all` (verify.sh:348 C2; `-p` narrowing structurally off at
scope=all, :521-527), so a 1-task and an N-task merge verify are the byte-identical full plan ⇒ union ≡ single by
construction. The only remaining premise, `s(N) > 1/N`, is **measured** (exp2) and consumed by a **human** go/no-go
(not a task node — a no-code decision task churns the orchestrator). The design-doc M2 "~170 s, $0.90" was a
misattribution — it is the verify-PHASE agent-invocation median, not the merge verify (which is ~90 min cold).

Legend as in the C manifest. `reify:` = filed against the reify project (cross-project dep).

---

## exp1 — RETIRED / CANCELLED (reify task 4454)
union ≡ single is an identity under the full-scope merge gate — there is no scoped union vs scoped single to time.
The original "~170 s single-task baseline" was the verify-phase agent-invocation median, not the merge verify. No binding.

## exp2 — s(N) thrash proxy from history *(leaf; reify task 4455; the ONLY economic gate)*
| Capability | Binding | Verdict |
|---|---|---|
| `main` first-parent history + runs.db | present (design §4 measured from them) | PASS |
| committed s(N) estimate (N=2,3) + `s(N) > 1/N` check | producer:task-exp2 | PASS |

## go/no-go — automated inside the s(N) task (4455), escalate-on-margin; NOT a separate node (D7.3)
| Capability | Binding | Verdict |
|---|---|---|
| s(N) estimate + sample size | producer:reify:4455 (exp2, PART 1) | PASS |
| deterministic rule `s(N) > 1/N` + action | producer:reify:4455 (exp2, PART 2): clear go → flip β…ε `deferred`→`pending`; clear no-go → cancel + info-escalate; marginal → escalate to human. Marginal = `|s(N)−1/N| < 0.2·(1/N)` ∨ <~10 clusters ∨ ambiguous attribution | PASS |

## α — Union-scope the train verify *(intermediate; NO gate — lands regardless, D6)*
| Capability | Binding | Verdict |
|---|---|---|
| `GroupMergeRequest` tip-scoped `task_files`/`module_configs` (the bug) | grep:workflow.py:~653-654 wired on main (this IS the latent under-verify defect) | PASS |
| `merge_verify_workspace` force-workspace fallback knob | grep:config.py:~811 wired (default false) | PASS |
| union-over-members closure computation + `train_scope=union` event | net-new · producer:task-α | PASS |

## β — Former core (select N≤3 stackable + assign train metadata) *(intermediate; depends_on gate, α)*
| Capability | Binding | Verdict |
|---|---|---|
| δ₂ `_maybe_enqueue_group_merge` trigger | grep:workflow.py:~547 wired | PASS |
| `metadata.train` membership + `_train` reader | grep:workflow.py:~499-508 wired | PASS |
| line-level-stackability test + the former | net-new · producer:task-β | PASS |
| go decision (DAG-direction) | producer:task-gate upstream | PASS |
| union-scope on formed trains | producer:task-α upstream | PASS |

## γ — Branch stacking + conflict-eject *(intermediate; depends_on β)*
| Capability | Binding | Verdict |
|---|---|---|
| `rebase_onto_main` | grep:git_ops.py wired | PASS |
| `_train_predecessor` sibling-stacking | grep:git_ops.py:~500 wired | PASS |
| conflict-eject (drop member, keep ≥2) | net-new · producer:task-γ | PASS |

## δ — Failure attribution (re-verify singles) *(intermediate; depends_on γ)*
| Capability | Binding | Verdict |
|---|---|---|
| existing single-task merge path | grep:merge_queue.py / workflow.py wired | PASS |
| on-train-fail → re-verify members singly | net-new · producer:task-δ | PASS |

## ε — Integration gate (real reify train lands) *(LEAF; depends_on δ)*
| Capability (G6 end-to-end — from ε's dependency closure only) | Binding | Verdict |
|---|---|---|
| a train lands N≥2 tasks on **one union** verify | producer:task-α (union-scope) + β/γ/δ (former) upstream | PASS |
| CAS-retry-rate / verifies-per-landed-task metrics | grep existing merge events wired on main | PASS |
| amortization win (union/single ratio ≈ 1) | producer:reify:exp1 upstream (measured) | PASS |
