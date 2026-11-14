# signal_leaves backfill, tranche ε1 (task 6586)

Task 6586 is tranche ε1 of `plans/dashboard-prd-lanes-prd.md` §9.
Rule applied: a leaf is named only where the PRD's own text, including its `.capability-manifest.md` twin, names the integration gate or goal-signal leaf. A leaf is never inferred from the dependency graph.

| Key | Result | Leaf (label → task) | Named by (section heading) / Why unnamed |
|---|---|---|---|
| `docs/prds/claimant-invariant-detection.md` | unnamed | — | No text names a leaf. The Goal has five co-equal outcomes; d2–d5 are tagged "leaf" only structurally. |
| `docs/prds/claimant-invariant-enforcement.md` | unnamed | — | No text names a leaf. The G2 note names γ/ζ/η only as DAG sinks, and detection was removed by D8. |
| `docs/prds/falkordb-index-provisioning.md` | named (in the `.capability-manifest.md` twin) | ζ → 3711 | Twin `docs/prds/falkordb-index-provisioning.capability-manifest.md`, "## ζ — activation verification + measured retrieval improvement *(intermediate → η; the integration gate)*". The PRD's Decomposition plan calls ζ the "verification leaf". The downstream η is not named: its twin heading "## η — documentation + correct the false certifications *(leaf)*" makes it a doc-and-runbook follow-up, and its signal is a grep over OPERATIONS.md and a plans/ certification row, not the Goal's measured end state (briefing recall above today's 0/24), which ζ asserts. |
| `docs/prds/landed-not-done-recovery.md` | named | θ → 4652 | "## Decomposition plan": "θ — Integration gate: the landed-recovery boundary suite". |
| `docs/prds/memory-briefing-and-fusion.md` | unnamed | — | No text names a leaf. α and β are independent per-leaf signals, and γ 3660 is cancelled. |
| `docs/prds/memory-eval-dashboard.md` | named | gamma → 3217 | "## 7. Decomposition plan": "γ — escalation-parity integration gate … Leaf.". The §4 G5 note agrees. |
| `docs/prds/memory-eval-program.md` | named (in the `.capability-manifest.md` twin) | epsilon → 3211 | Twin `docs/prds/memory-eval-program.capability-manifest.md`, "## Gate outcomes", G5: "ε is the integration-gate task". |
| `docs/prds/memory-metadata-vocabulary.md` | unnamed | — | No text names a goal-signal leaf. η is named only as the shape-ratification decision gate (§3, §9, D9), not as the integration gate. |
| `docs/prds/memory-write-path-convergence.md` | unnamed | — | No text names a leaf; 16 leaves each carry their own G2 signal. Separately, every bound task lacks a raw `prd_path` (it is attributed via `prd`), so the drift audit would flag any naming. |
| `docs/prds/merge-worktree-lifecycle-integrity.md` | named | ζ → 2929 | "Decomposition plan": "ζ — 5326-replay integration gate (H) … this is the PRD's done gate". |
| `docs/prds/recurring-deterministic-tasks.md` | named | r6 → 4681 | Decomposition plan table, r6 row: "leaf (integration gate)". The G1 paragraph agrees. |
| `docs/prds/truth-propagation-record-mechanics.md` | unnamed | — | No text names a leaf. It has four independent leaves, and γ 4699 is a backstop decision gate. |
| `plans/agent-transcript-archival-prd.md` | named | ε → 2732 | "§8 Decomposition": "ε — End-to-end boundary gate (B+H)". |
| `plans/architect-already-complete-exits.md` | named | β → 3031 | "## Decomposition plan", Phase 1, β: "Signal (leaf / integration-gate)". |
| `plans/census-incremental-prd.md` | named | G → 6409 | "§9 Decomposition plan": "G: live census conformance gate" and "the C-as-integration-gate pattern". §1 Goal agrees. |
| `plans/completion-driven-triggers-prd.md` | named | λ → 6393 | "§10 Decomposition plan": "λ — integration gate: trigger chain end to end". |
| `plans/confusion-reduction-prd.md` | named (new minimal sidecar) | epsilon → 2581 | "§11 Decomposition plan": "ε — Nightly trickle assembly … Integration-gate leaf.". Only task 2581 carries label epsilon for this PRD (not 2584 epsilon-prime). |
| `plans/dashboard-availability-prd.md` | unnamed | — | The text names out-of-batch task 3289 'restore-gate' as the batch's terminal gate, but it has no sidecar block, per the twin's "Unbound task labels (task 4907 adjudication)" ruling that none be added. No in-plan leaf α–θ is named. |
| `plans/dashboard-one-datum-one-path-prd.md` | named | ι → 5598 | Decomposition plan row ι: "Integration gate … Leaf (gate)"; heading "## Boundary-test sketch (B + H) — the integration gate ι's signal". |

12 named, 7 unnamed. Adding the confusion-reduction sidecar surfaces 9 non-live (all done) UNBOUND LABEL rows in the drift audit; they are expected and are not signal-binding findings.
