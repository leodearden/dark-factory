# Capability manifest: review-all S21 legibility pipeline

PRD: `plans/review-all-s21-legibility-pipeline-prd.md`. Machine-readable twin: `plans/review-all-s21-legibility-pipeline-prd.capability-manifest.yaml` (task ids stamped by `commit_planning`).

**Code anchors** verified against main `ae3047ec3a` (2026-10-10). Cite-by-symbol; re-locate at implementation time.

Each leaf's signal is bound capability by capability. `PASS` = the evidence holds today or the producer is this leaf or an upstream one. `OPEN` = the producer is another stream's leaf not yet filed; it is bound at decompose and blocks filing until then.

## α — One detector layer for sampler and digest

| Capability | Check | Evidence | Verdict |
|---|---|---|---|
| `detectors.py`, `reingestion.py` exist | producer | α creates both (D1) | PASS |
| `detectors.record_hits` | producer | α (contract 5.1) | PASS |
| sampler pattern tables deleted | anti-orphan (deletion) | `sampling.py::_NOT_FOUND_PATTERNS` and three siblings present on main | PASS |
| one agent-class classifier | anti-orphan (deletion) | `classify_agent_class` defined in both `sampling.py` and `digest.py` on main | PASS |
| 0 class-presence disagreement on the live population | G6 exactness | both scorers reduce one `record_hits` over one parse contract (`inventory.iter_json_lines` ≡ `digest.load_transcript`); presence, not count, is asserted, because digest counts per matched pattern | PASS |

## β — One codebook merge result and one committer

| Capability | Check | Evidence | Verdict |
|---|---|---|---|
| `codebook.apply_coding_records`, `MergeResult` | producer | β (D3) | PASS |
| three merge loops collapse | wired | loops in `nightly.py::run_nightly`, `census.py::run_census`, `codebook.py::_cmd_apply` on main | PASS |
| deletion directive skipped, not raised | G6 rejection→report | reproduction: `validate_coding_record` returns `[]` and `apply_coding_record` raises `NeverDeleteError` on the coder-shaped record | PASS |
| `unlanded.commit_only` with retry | producer | moved from `nightly.py::_git_commit_docs_only` / `_LOCK_ERROR_PATTERNS` (present on main) | PASS |
| bare sibling imports gone | anti-orphan (deletion) | `census.py` bare `import codebook/coder/filing_policy/config` on main | PASS |

## γ — Typed run outcome and trickle state; barren escalation recorded

| Capability | Check | Evidence | Verdict |
|---|---|---|---|
| `RunOutcome` with `deferred` | producer | γ; `trickle_state.py::OUTCOME_*` has four values on main | PASS |
| `state_docs.py` over `shared.safe_io` | substrate | `shared/src/shared/safe_io.py::atomic_write_text(mkdir=)`, `load_json_or_warn`, stdlib-only; units run under `uv run --project shared` | PASS |
| `barren_escalated` written and read | field population | written by nightly before `record_run` (D6); a pre-γ document reads as unrecorded (INV-13) | PASS |
| status strings removed | anti-orphan (deletion) | 5 sites on main (coder 2, nightly 1, census 2) | PASS |
| production row | INV-13 | read through `check_trickle_progress.py` after a real nightly | PASS (manual) |

## δ — run_nightly as stage functions with one result builder

| Capability | Check | Evidence | Verdict |
|---|---|---|---|
| `nightly._finish` | producer | δ (D16) | PASS |
| no dotted-path stage patches in tests | anti-orphan (deletion) | 6 names patched in `test_legibility_nightly.py` on main | PASS |
| final journal line | producer | δ; observed in the real nightly journal | PASS (manual) |

## ε — One alarm poster over the shared MCP client

| Capability | Check | Evidence | Verdict |
|---|---|---|---|
| `alarm.raise_alarm`, `fail_loud_boundary` | producer | ε (D11) | PASS |
| poster constant copies deleted | anti-orphan (deletion) | `_ESCALATION_TOOL_NAME` in two probes on main; nightly's and census's envelope builders | PASS |
| `census_trigger.py` transport deleted, no importer left | anti-orphan (deletion) | 18 code references on main incl. `scripts/_config_reload_gate.py` | PASS |
| shared synchronous MCP client | capability→producer, extent, DAG direction | **task 6711 (S14 ζ)**, upstream of ε: `shared/src/shared/mcp_session.py::SyncMcpSession`. Its filed extent matches what ε needs: handshake for stateful escalation servers (server-minted `mcp-session-id` captured and echoed) and stateless fused-memory, `/mcp` via `mcp_endpoint_url` + `MCP_POST_HEADERS`, SSE decode, explicit session release, `isError` typed, one neutral shared error type. `shared/src/shared/mcp_post.py` is async-only on main | PASS (bound at decompose 2026-10-10) |
| real-server e2e | substrate | `escalation/tests/test_legibility_census_escalation_e2e.py::serve_escalation_mcp` drives census, nightly, transcript-check on main | PASS |

## ζ — Census trigger reads validated config and a typed census state

| Capability | Check | Evidence | Verdict |
|---|---|---|---|
| second reader deleted | anti-orphan (deletion) | `census_trigger.py::CensusConfig.from_mapping`, `load_census_config` on main | PASS |
| strict validation rejects bad values | G6 rejection | pydantic `StrictInt`/`ge=0` raise `ValidationError`; live dark_factory and reify `census:` blocks are unquoted ints, so neither nightly breaks | PASS |
| `layout.py`, `census_state.py` | producer | ζ (D7, D9) | PASS |
| inline census-state paths removed | anti-orphan (deletion) | `census.py::main`, `census_trigger.py::decide_for_project` on main | PASS |
| census-incremental chain end | DAG direction | task 6408 upstream of ζ | PASS (edge at decompose) |

## η — Candidate identity by merger-minted id; lifecycle as a transition table

| Capability | Check | Evidence | Verdict |
|---|---|---|---|
| title joins deleted | anti-orphan (deletion) | `_find_pending_candidate_id`, `_find_adjudicated_candidate` on main | PASS |
| `Resolution` variants | producer | η (contract 5.2) | PASS |
| lifecycle transforms moved | anti-orphan (deletion) | four census dict-edit transforms on main | PASS |
| live codebook passes new `validate` checks | G6 | 2026-10-10: 90 promoted, 0 dangling `promoted_to`, no out-of-set `filing` | PASS |

## θ — run_census as stage functions with an explicit persist order

| Capability | Check | Evidence | Verdict |
|---|---|---|---|
| `CensusEffects`, `CensusStage`, `persist` | producer | θ (D15) | PASS |
| effect order observed | boundary row 16 | fake effects record call order | PASS |

## ι — census.py layers and a public verifier module

| Capability | Check | Evidence | Verdict |
|---|---|---|---|
| public verifier builder in `census_verify.py` | producer | ι | PASS |
| `_post_mcp_tool_call` gone | anti-orphan (deletion) | present in `census.py` and patched by tests on main | PASS |
| layers import downward | heuristic 13 | no `census_*.py` imports `census` on main | PASS |
| tests on public names | boundary row 17 | 7 private names / 52 references at the pin | PASS (manual) |

No binding is FAIL or OPEN. ε's shared-client binding was OPEN at authoring and was bound to task 6711 at decompose (2026-10-10).
