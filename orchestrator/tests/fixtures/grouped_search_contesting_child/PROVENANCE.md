# Provenance — `grouped_search_contesting_child` fixtures

These fixtures exist so the orchestrator's contesting-child render is tested
against the bytes fused-memory's `search` tool actually emits, not against a
shape written from its docstring. A hand-built fixture that gave the contested
digest a `content` body would let the full-body render pass while production
never carried one.

## The rule

> **If a test that reads these bytes fails, the fused-memory/orchestrator seam
> has drifted.** Re-record the fixture from the producer (below) and fix the
> **consumer**. Do **NOT** edit a fixture to make a test pass.

---

## `child-also-matched.json` and `only-parent-matched.json`

| | |
|---|---|
| **Grounding** | **Real producer over a stub scenario.** The real `search` MCP tool (`fused-memory/src/fused_memory/server/tools.py::create_mcp_server` → `fused-memory/src/fused_memory/server/grouped_read.py::group_search_results`) ran over a stub memory service; only the service's reads are stubbed. |
| **Producing code sites** | `fused-memory/src/fused_memory/server/grouped_read.py::_digest_entry` stamps `contested: true` on a digest entry from `grouped_read.py::is_contested_child`, the single definition of contested. |
| **Pinning test** | `fused-memory/tests/server/test_contesting_child_search_reply.py` re-produces both replies and asserts equality, printing the produced JSON on a mismatch. |
| **Readers** | `orchestrator/tests/test_memory_recall.py` and `orchestrator/tests/test_briefing_project_scope.py`, through `orchestrator/tests/_briefing_helpers.py::recorded_search_text`. |
| **Serialisation** | `pydantic_core.to_jsonable_python` of the tool result, then `json.dumps(..., indent=2, ensure_ascii=False)` plus one trailing newline. |

The scenario: parent `11111111-…-111111111111` carries two amendments, an
ordinary one (`…221`, created first) and a contesting one (`…222`, body over the
240-character digest cut).

Load-bearing facts a reader must not re-derive from prose:

* **The contested digest entry carries no `content`.** Its `digest` is cut at
  240 characters and ends with `…`. A contested child is never pinned into its
  parent's block, so the digest is all the grouped block has.
* **The full body is on the wire only as the same-id top-level result**, and
  only when the child itself matched the query (`child-also-matched`). In
  `only-parent-matched` the digest is the only copy.
* **The ordinary amendment sorts first** in `grouped.amendments`, so a
  renderer that puts the contesting child first has done so by reading the
  marker, not by inheriting the list order.
* **The top-level child also carries `metadata.x_contested`.** The orchestrator
  must not read it: grouped_read's `contested` marker is the one home of the
  flag on the wire.
