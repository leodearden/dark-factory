---
outcome: done
changed: commit d6ea845 (docs-only, PRD docs/prds/review-all-provider-mode-as-a-typed-axis.md §4 D6 ruling note + §12 item 5); dated RULING notes on tasks 1276, 1279, 466, 653, 461, 815, 1077 (+addendum), 1084, 444, 465, 1278, 489, 490, 491, 1282; one decisions_and_rationale memory (ruled_by human)
action_needed: review-all lead — (1) apply the program-doc amendments below (R15 wording, S15 Q3 line, S24 stream scope); (2) file the staging-banner follow-up task (ops console + member portal "pre-production, no real money"); (3) decide whether S24·γ (1077) carries the staging rung or a new S24 leaf is split from it
---
**Ruling (Leo, 2026-10-16, S15 Q3):** production fails closed, and a new `staging` ENVIRONMENT rung carries demos and pre-production testing. Leo ruled from three requirements:
1. The real production deployment cannot accidentally run any non-production mode or component.
2. Demos and progressive pre-production testing are possible in a safe, controlled way.
3. CI and integration tests keep working.

How the rungs work:
- **The rungs:** development/test, then staging, then production/prod.
- **staging** is hardened like production. It admits non-production modes, each set explicitly. It refuses provider mode `live`.
- **Every guard row declares which environments refuse it.** This is one mechanism in S24·γ.
  - *Hardening rows* are refused on staging and production: 444's secret rules, db-sqlite, mqtt-tls-off, telemetry-store-volatile, and **consent-off** (Leo's call).
  - *Reality rows* are refused in production only: provider-mode-not-live, clock-override, registry-yaml-only (the lead's classification), and a new **simulated-fleet** row.

Options rejected:
- **A (as ruled):** ν's drain could then only run through an unrecorded ENVIRONMENT override.
- **B1 (a free flag):** fails requirement 1.
- **B2 (sandbox on a simulated fleet):** the fleet is itself a non-production component.
- **C (pull 461 forward):** rider R-a carries its useful part.
- **D:** adopted in a cheaper form, as a third rung rather than a second axis.

**Corrections the investigation found in the original Q3 note:**
- PROVIDER_ROLES stays {BILL} until 465, so the exit-2 window would not close at 461. Rider R-a fixes this: 461 drops BILL.
- 1278 and 461 clash on BillGraph's provider binding. Whichever lands second adapts.
- 444's O4 case breaks after 1279 (rider R-b).
- ν (491) and the M1 "production path" demo conflicted with R15 plus 815, and the original note missed it.
- The finding fk-7d95cbeb4eac never asked for sandbox to be refused in production.
- The author's hand-back (~/.claude/fleet/sessions/prd-solar_challenge_platform-4191476/result.md) no longer exists. Q3 was taken from the program doc and task 1279.

**Program-doc amendments for the review-all lead** (not applied here):
- **R15 ruling text:** change "refused under is_production unless live" to "refused in the production deployment (ENVIRONMENT=production/prod) unless live; refused on staging when live; staging is a hardened pre-production rung (Leo 2026-10-16, S15 Q3)".
- **S15 Q3 amendment line:** replace "stands as ruled …" with "RULED by Leo 2026-10-16: staging rung; see PRD §4 D6 ruling note (d6ea845); riders R-a (461 drops BILL from PROVIDER_ROLES; 465 adds the issue role; 1278/461 second-to-land adapts), R-b (1279 updates 444's O4), R-c (653 = pytest lane)".
- **S24 stream:**
  - γ (1077) gains the staging rung, the hardened vs real-deployment predicates and per-row environment scope; consent-off becomes hardening.
  - ε2 (1084) admits staging at render and refuses a simulated fleet in production.
  - Both are recorded on the tasks. The S24 PRD needs a dated note; S24's owner should add it, as this session didn't edit the S24 PRD.
- **New follow-up to file:** a staging banner in the ops console and the member portal.

**Team roster** (lead: Opus 5.5):

| Seat | Model / effort | Tokens / time | Outcome |
|---|---|---|---|
| A code facts | sonnet / medium | 87k / 8 min | 815 is the render's production gate (main passes ENVIRONMENT through); the close uses fakes unconditionally |
| B task graph | sonnet / medium | 147k / 17 min | found the PROVIDER_ROLES window error, the ν/M1 conflict and the O4 break |
| C design space | sonnet / high | 163k / 20 min | S24 has no waiver mechanism; the finding never asked to refuse sandbox; costed B and D |

Routing held: no seat needed a redo.
