---
outcome: done
changed: commit a629f24 on main (docs-only: S16 PRD §4 D11/§11/§12/§13 RULING 2026-10-16 notes + capability-manifest δ2 block); NEW task 1299 (S16·δ2, pending, deps 1130/1138/1236/1241); dated details notes on 1236, 1241, 422, 1272; fused-memory decisions_and_rationale (ruled_by human)
action_needed: review-all lead — add 1299 (δ2) to the program doc's S16 FILED row and S16 author-pass amendments (Q1 is now ruled B, not "default as authored"); decide whether 1299 joins gate 1065 (not touched here; S16 had no leaf on it); optionally add a dated G4 cross-reference in docs/prds/review-all-governed-ledgers-on-one-mixin-and-reader.md (S21's contract gains EffectiveDatedLedger.prepend + PREPENDABLE via 1299). The 1236 description still says "Q1 D11 stands as authored"; its details note supersedes it.
---
**Ruling (Leo, 2026-10-16): S16 Q1, option B.** While an account holds no statement, its first tariff assignment follows its opening.
- **Interim, in task 1236 (δ):** D11's 409 stays as the interim guard, triggered on `window_after(...).opens_on` earlier than `first_assignment_day`. Its 409 text names the interim remedy: close with `closed_on` = `opened_on`, then re-create.
- **New rule in δ (Leo, second question):** δ also refuses the gap D11 missed. A first assignment through `POST /accounts/{id}/tariff` dated after the start of the supply span gets a 409. Without it, the account is skipped `no_tariff` every period, for good.
- **The replacement, new task 1299 (δ2):** after ε and S21 β/ε it replaces the interim refusal with:
  - a generic `EffectiveDatedLedger.prepend`, fail-closed through `config/authority.py::PREPENDABLE = {AccountTariffAssignment}`;
  - an account-row lock, `RestatesBilledDays.check_assignment` and ε's version-in-force 422;
  - a partial unique index on closed `effective_to`, which closes the race between two concurrent prepends.
- **Invariant weakened, exactly:** for assignments only, S21's "nothing lands before a partition's first row" becomes "insert ending exactly at the earliest row while unbilled".

**What decided it:**
- Main already lets an unbilled opening move either way (`scaling-defects-1k-10k.md` D9, task 313). D11 as authored narrowed that rule quietly.
- The close-and-re-create runbook leaves a junk account in `/me/accounts`, a new Stripe customer, and probably a new DD mandate. That last point is unverified: `ensure_mandate` looks like a placeholder.

**Rejected:** refusing permanently; restatement as a later PRD; a bitemporal ledger (D); flooring the first assignment at the tariff's first version; billing extending the first span.

**Sequencing default set by this session, not by Leo:** δ keeps D11 as the interim guard rather than dropping it before δ2 lands. δ2 runs after ε, so ε needs no edge change.

**Team:** sonnet/medium seats A and B (about 95k tokens each) and opus/high seat C (169k), plus my own integration. The PRD author's hand-back file named in the brief does not exist; its Q1 content was taken from the program doc and PRD §11. Program-doc amendments are listed under action_needed above.
