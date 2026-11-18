---
outcome: done
changed: commit 8ad584b (docs-only — S13 PRD §7 + P-Member PRD §0 A1 RULING notes); tasks 582, 617, 618 (title/description/metadata rewritten + dated notes); dated notes on 972, 461, 462, 476, 457, 493, 1195, 1200, 624; edge 1200→582 added
action_needed: (1) Leo: amend ~/mission-control/consulting/solar-challenge/decisions.md 2026-09-29 entry — the cbs_billing clause is superseded (contract, provisional) — and put the billing Art 6 basis to the board with the DPIA; (2) review-all lead: program-doc amendments listed below; (3) optional: drop 617's edges to 442/461 (they served test flips that moved to 582)
---
**Ruling (Leo, 2026-10-16):**
- **D1 = A-narrow.** Task 582 owns `LawfulBasis` {CONSENT, CONTRACT, LEGITIMATE_INTEREST} and `LAWFUL_BASIS` beside `REQUIRED_SCOPE` in consent/purposes.py. It also owns a basis-first branch in `ConsentGate.check`:
  - it reads no consent row and no account;
  - in scope → `OTHER_LAWFUL_BASIS`, otherwise `SCOPE_INSUFFICIENT`;
  - there is no `NO_ACTIVE_ACCOUNT` reason, which would collide with S7 η 1269's delivered check.
- **617 shrinks** to TermsVersion, offered, the digests and the migration.
- **618 keeps** GET /consent/purposes, ConsentScreen and the 409 refusals.
- **D2: cbs_billing = CONTRACT.** A1 is adopted, provisional pending the board; this supersedes the 09-29 clause.
- **platform_operation = CONTRACT.**

**Consequences the lead carried into task text:**
- 582 inherits the δ1a test flips (billing now permits withdrawn customers) and removes the portal's cbs_billing toggle.
- Because billing and ingest never read consent_current after 582, S13 §10 rows B6, B7 and B8 are restated:
  - 1200 now depends on 582;
  - 493 and 1195 carry notes;
  - 461, 462 and 476 drop the consent-denied limb of B4.
- Stale delivered_checks were fixed:
  - 617 required `def categories_for(` and "catalogue" in gate.py;
  - 618 had three `categories_for` call-site checks, and S13 forbids that name.

esc-582-5 was already dismissed by the orphan reaper and its L1 copy (esc-582-6) is resolved, so nothing was left to resolve.

**Program-doc amendments for the review-all lead (I did not edit the program doc):**
- S13 FILED row: 582 is "lawful-basis table + basis branch (A-narrow, 2026-10-16)", no longer "+ GET /consent/purposes".
- Lead Q2 ruling: the "GET surface" half moves to 618.
- Note the 1200→582 edge, and that the 617 and 618 descriptions and checks were rewritten.

The memory (decisions_and_rationale, ruled_by human) is written.
