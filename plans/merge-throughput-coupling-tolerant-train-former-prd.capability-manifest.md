# Capability manifest — Coupling-tolerant train former (Lever A′)

Mechanizes G3 + G6 per task. Evidence verified 2026-06-09 against `main` (cite-by-symbol).
**No binding resolves to a FAIL value → batch is not blocked.**

The two economic premises (union ≈ single; `s(N) > 1/N`) are **measured** (exp1/exp2) and consumed
by the go/no-go gate — they are NOT frozen into any RED test (the G6 "guessed bound in a RED test"
failure is structurally avoided: the build is gated *behind* the measurement, not asserted against it).

Legend as in the C manifest. `reify:` = filed against the reify project (cross-project dep).

---

## exp1 — Union-verify wall-time measurement *(leaf; reify; off-peak / out-of-band)*
| Capability | Binding | Verdict |
|---|---|---|
| cargo verify scoped to a 2–3-task union closure | reify cargo/verify toolchain present | PASS |
| ~170 s single-task baseline (M2) | design §4 M2 measured | PASS |
| committed union/single ratio bench doc | producer:task-exp1 | PASS |
| *constraint:* must run off-peak (contention — warm-builds D2) | recorded in description; filed **deferred** (out-of-band), not auto-dispatched | PASS |

## exp2 — s(N) thrash proxy from history *(leaf; reify)*
| Capability | Binding | Verdict |
|---|---|---|
| `main` first-parent history + runs.db | present (design §4 measured from them) | PASS |
| committed s(N) estimate (N=2,3) + `s(N) > 1/N` check | producer:task-exp2 | PASS |

## gate — A′ go/no-go *(intermediate; depends_on exp1, exp2)*
| Capability | Binding | Verdict |
|---|---|---|
| exp1 + exp2 results | producer:reify:exp1 + reify:exp2 upstream (cross-project) | PASS |
| explicit go criteria (union≈single AND s(N)>1/N) | stated in PRD §5 / task — agent-evaluable, escalates on ambiguity | PASS |

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
