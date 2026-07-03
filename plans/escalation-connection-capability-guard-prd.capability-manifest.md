# Capability manifest — escalation connection-level capability guard

Mechanizes **G3 (assumed-substrate)** + **G6 (premise validity)** for
`plans/escalation-connection-capability-guard-prd.md`. One block per task; every
capability the task's observable signal asserts is bound to evidence. Any binding
resolving to a FAIL value (`declared-only` / `test-only` / `producer-absent` /
`producer-extent-short` / `producer-downstream` / `fixture-ERROR` / `bound≤floor`
/ `rejection-absent`) **blocks the batch** until resolved.

**Verified at decompose:** 2026-07-03, against `main` @ `18d2a490e9` (venv
`fastmcp 3.2.2`). Line numbers are decompose-time anchors, not contracts.

**Domain flag:** infrastructure / wiring — **no numeric premises** anywhere
(G6 branches 1–2 are N/A). The live G6 branches are **3 (end-to-end capability →
dependency-set)** and **4 (negative assertion → rejection-mechanism-backed)**.

**Substrate-confirmed flag emitted on every task:** `substrate_confirmed=True`
(generic analogue of the overlay's `grammar_confirmed`).

---

## α — Server-side per-connection capability enforcement (leaf-capable, CI)

*Signal:* an HTTP integration test drives the escalation server — a client
sending `X-Escalation-Levels: 0,1` gets `level_forbidden` (record untouched) on a
seeded **L2** resolve; the same client resolving an **L1** succeeds and the
archived `resolved_by` equals the `X-Escalation-Identity` header; a **header-less**
client resolves the L2 successfully; `promote_to_l2` from the `0,1` client still
succeeds.

| Capability asserted | Evidence | Verdict |
|---|---|---|
| FastMCP tool can read a per-request header via `get_http_headers()` | `grep: from fastmcp.server.dependencies import get_http_headers` — imports & resolves in-venv (`fastmcp 3.2.2`) | **PASS** (verified exists) |
| `resolve_issue` has a single pre-write choke where a level-gate can guard **all** actions | `grep:escalation/src/escalation/server.py:533-556` — action-validation at :533-534; `park` branch at :535-546; pre-stamp `pending.resolution_action=action` at :554-556. Insert the gate **after :534, before :535** → covers park + resolve/close/abandon | **PASS** (wired choke exists) |
| `queue.park` / `queue.resolve` accept `resolved_by` (so Identity can override the tool arg) | `grep:server.py:540-542` (park) + `:558-561` (resolve) both pass `resolved_by=…` | **PASS** (wired) |
| `promote_to_l2` is a **separate** tool, so leaving it ungated is a no-op edit | `grep:server.py:618` `async def promote_to_l2` — distinct tool; cascade doc at :635 | **PASS** (verified separate) |
| `esc.level` is a readable int to compare against the parsed level set | `grep:escalation/src/escalation/models.py:66` `level: int = 0`; consumed at `server.py:597` `e.level` | **PASS** (field populated on production path) |
| **Rejection mechanism** `level_forbidden` / `bad_capability_header` fires on a forbidden call (G6 branch-4) | **authored HERE**, upstream of β/γ/δ; the rejection is **observed to fire** in α's own HTTP test (rejection built + observed = the PASS pattern) | **PASS** (rejection built & observed in-task) |

**Prereqs:** none. **Leaf/intermediate:** intermediate (β, γ, δ depend on it) — but
carries its own CI signal, so it is not a bare foundation task.

---

## β — Orchestrator attaches the capability header for the supervised watcher (leaf-capable, live smoke)

*Signal:* with the built escalation server running, a connection configured exactly
as the watcher's `mcp_config` (Levels=`0,1` + Identity) is **denied** an L2
`resolve_issue` (proving the header reaches the server end-to-end), while its L1
resolve + `promote_to_l2` succeed with server-stamped `resolved_by`. If the header
does **not** arrive → escalate, do **not** mark done.

| Capability asserted | Evidence | Verdict |
|---|---|---|
| `mcp_config_json` escalation block is a plain dict a `headers` field can be added to | `grep:orchestrator/src/orchestrator/mcp_lifecycle.py:697-701` — `config['mcpServers']['escalation'] = {'type':'http','url':…}` plain dict | **PASS** (trivial extension, β's own edit) |
| `_run_watcher_rotation` builds the watcher's `mcp_config` (the wiring site) | `grep:orchestrator/src/orchestrator/harness.py:5925` `_run_watcher_rotation`; `:5956` `mcp_config = self.mcp.mcp_config_json(escalation_url=…)`; allowlist at `:158` | **PASS** (wiring site exists) |
| Server **enforces** the header once β sets it (the whole point of β) | `producer:α` — α authors the enforcement; **α is upstream of β** (β `depends_on` α) | **PASS** (producer upstream, ✓ DAG-direction) |
| **CLI forwards a `--mcp-config` `headers` field on each request (substrate (ii))** | `live-smoke` — **task-time proof, not pre-verifiable**. Novel: the escalation block carries no headers today. β's signal **is** the proof; on absence β **escalates + pivots** (URL-token / dedicated restricted surface — PRD Open Q3), never a silent done | **PASS-conditional** (designed prove-or-escalate spike; **not** a fictional premise — the failure path is a loud escalation, the failure mode the manifest guards against is a *silent* stall) |

**Prereqs:** α. **Leaf/intermediate:** intermediate (γ depends on it) — carries its
own live-smoke signal.

> **Note on the one non-static binding.** Substrate (ii) is the single capability
> in this PRD not provable by grep at decompose. It is **not** a G3 FAIL: there is
> no producer task to be absent, and the task is explicitly structured to *prove or
> loudly escalate* the CLI round-trip, with a documented pivot. This is the correct
> handling of a genuine substrate unknown — a scoped spike, not a false premise
> baked into a RED test. Surfaced here so a dispatch-time architect does not treat
> it as pre-verified.

---

## γ — Integration gate: two-way boundary tests (LEAF)

*Signal:* the seven boundary scenarios (PRD §Boundary-test sketch) all pass —
watcher-side conn denied L2 resolve **and** park but allowed L1 resolve + promote;
human-side (header-less) conn retains full L2 resolve/park; `l2-cascade` member
resolution still fires when the human resolves an L2 with members; malformed-header
call fails closed (`bad_capability_header`).

| Capability asserted | Evidence | Verdict |
|---|---|---|
| `level_forbidden` denial on watcher-side L2 resolve **and** park (scen. 1,2) | `producer:α` (rejection authored in α; gate guards both the park branch and the resolve path) — **α upstream** | **PASS** (producer upstream, ✓ DAG) |
| `bad_capability_header` on malformed-Levels (scen. 7, G6 branch-4) | `producer:α` (fail-closed parse authored in α) — **α upstream** | **PASS** (producer upstream, ✓ DAG) |
| Watcher-configured connection actually reaches the server with its header (scen. 1–4) | `producer:β` (orchestrator header wiring) — **β upstream** | **PASS** (producer upstream, ✓ DAG) |
| Header-less connection retains full L2 authority (scen. 5, default-open) | `producer:α` (default-open path = absence-of-header ⇒ all levels) — **α upstream** | **PASS** (producer upstream, ✓ DAG) |
| `l2-cascade` member resolution fires on a permitted parent-L2 resolve (scen. 6) | `grep:escalation/src/escalation/queue.py:486` `cascade_resolved_by = f'l2-cascade:{escalation_id}'`; cascade runs server-internally after lock release (:425-497) — **existing substrate, not re-checked against Levels** | **PASS** (verified exists, unaffected) |

**Prereqs:** α, β. **Leaf.** Realizes the B+H contract — the integration-gate task
whose signal **is** the boundary-test sketch (closes G5→G2).

---

## δ — Docs + memory correction (LEAF, companion-correction)

*Signal:* both watcher skills grep-match the new identity string
`orchestrator-escalation-watcher-auto` **and** the L2-enforcement invariant; the
interactive skill's rogue-detection section names that string as the **trusted
supervised owner** (not a rogue).

| Capability asserted | Evidence | Verdict |
|---|---|---|
| Canonical identity string `orchestrator-escalation-watcher-auto` is decided/authoritative | `producer:α` (server stamps `resolved_by` from `X-Escalation-Identity`; value set by β) — **α upstream** (δ `depends_on` α "needs the final identity string + error semantics") | **PASS** (producer upstream, ✓ DAG) |
| Error semantics `level_forbidden` / `bad_capability_header` to document | `producer:α` — **α upstream** | **PASS** (producer upstream, ✓ DAG) |
| Auto-watcher skill file exists to edit | `grep:skills/escalation-watcher-auto/SKILL.md` (32 KB) | **PASS** (target exists) |
| Interactive watcher skill file exists to edit | `grep:skills/escalation-watcher/SKILL.md` (45 KB) | **PASS** (target exists) |
| Memory note with inverted ownership framing exists to correct | `grep:memory/feedback_escalation_watcher_check_for_canonical_owner.md` (4 KB) | **PASS** (target exists) |

**Prereqs:** α. **Leaf** (companion-correction; doc-grep signal — not a
synthetic-input unit test).

---

## Batch verdict

**No FAIL bindings.** Every capability is either verified-present on `main`
(`grep`), delivered by an upstream producer in the dependency closure with correct
DAG-direction (`producer:α|β` upstream), or — for the single genuine substrate
unknown (CLI header round-trip, substrate (ii)) — a **deliberately-scoped
prove-or-escalate spike** on β whose failure path is a loud escalation, not a
silent stall. **G3 + G6 clear → batch may queue.**
