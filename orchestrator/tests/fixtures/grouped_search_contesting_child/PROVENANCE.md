# Provenance — `grouped_search_contesting_child` fixtures

These fixtures exist so the orchestrator's contesting-child render is tested
against the bytes fused-memory's `search` tool actually emits, not against a
shape written from its docstring. A contested entry's `content` must come from
the producer, never from a hand edit: a hand-built body would let the full-body
render pass whatever production actually carried.

## The rule

> **If a test that reads these bytes fails, the fused-memory/orchestrator seam
> has drifted.** Re-record the fixture from the producer (below) and fix the
> **consumer**. Do **NOT** edit a fixture to make a test pass.

---

## `child-also-matched.json` and `only-parent-matched.json`

| | |
|---|---|
| **Grounding** | **Real producer over a stub scenario.** The real `search` MCP tool (`fused-memory/src/fused_memory/server/tools.py::create_mcp_server` → `fused-memory/src/fused_memory/server/grouped_read.py::group_search_results`) ran over a stub memory service; only the service's reads are stubbed. |
| **Producing code sites** | `fused-memory/src/fused_memory/server/grouped_read.py::_amendment_entry` gives every amendment entry `content`: the child's text cut by `grouped_read.py::amendment_text` (over `fused-memory/src/fused_memory/utils/clip_disclosure.py::elide_text`) at `write_triage.judge_field_chars`, which `fused-memory/src/fused_memory/server/tools.py` resolves per call via `fused-memory/src/fused_memory/server/write_triage_judge.py::resolve_judge_field_chars`. It stamps `contested: true` from `grouped_read.py::is_contested_child`, the single definition of contested. |
| **Pinning test** | `fused-memory/tests/server/test_contesting_child_search_reply.py` re-produces both replies and asserts equality, printing the produced JSON on a mismatch. |
| **Readers** | `orchestrator/tests/test_memory_recall.py` and `orchestrator/tests/test_briefing_project_scope.py`, through `orchestrator/tests/_briefing_helpers.py::recorded_search_text`. |
| **Serialisation** | `pydantic_core.to_jsonable_python` of the tool result, then `json.dumps(..., indent=2, ensure_ascii=False)` plus one trailing newline. |

The scenario: parent `11111111-…-111111111111` carries two amendments, an
ordinary one (`…221`, created first) and a contesting one (`…222`, body longer
than the 240 characters the pre-7518 digest showed).

Load-bearing facts a reader must not re-derive from prose:

* **Every amendment entry carries `content` and none carries `digest`.** The
  contested entry differs from the ordinary one only by `contested: true`; its
  `content` is the child's text, whole here because the body is under the
  judge's field cap.
* **In `child-also-matched` the full body is on the wire twice**: as the
  same-id top-level result (uncapped) and as the entry's `content` (capped).
  In `only-parent-matched` the entry's `content` is the only full copy.
* **The ordinary amendment sorts first** in `grouped.amendments`, so a
  renderer that puts the contesting child first has done so by reading the
  marker, not by inheriting the list order.
* **The top-level child also carries `metadata.x_contested`.** The orchestrator
  must not read it: grouped_read's `contested` marker is the one home of the
  flag on the wire.
