# Provenance — `grouped_search_superseded_child` fixtures

These fixtures exist so the orchestrator's superseded-child render is tested
against the bytes fused-memory's `search` tool actually emits, not against a
shape written from its docstring. A hand-built digest that carried
`metadata.x_superseded_by` let the render pass while production stripped the
mark from every nested entry (task 7156's review).

## The rule

> **If a test that reads these bytes fails, the fused-memory/orchestrator seam
> has drifted.** Re-record the fixture from the producer (below) and fix the
> **consumer**. Do **NOT** edit a fixture to make a test pass.

---

## `superseded-children.json`

| | |
|---|---|
| **Grounding** | **Real producer over a stub scenario.** The real `search` MCP tool (`fused-memory/src/fused_memory/server/tools.py::create_mcp_server` → `fused-memory/src/fused_memory/server/grouped_read.py::group_search_results`) ran over a stub memory service; only the service's reads are stubbed. |
| **Producing code sites** | `fused-memory/src/fused_memory/server/grouped_read.py::_nested_entry_metadata`, through `grouped_read.py::_digest_entry` (the amendment digest) and `grouped_read.py::_pin_matched_child` (the pinned sighting). |
| **Pinning test** | `fused-memory/tests/server/test_grouped_read_supersession.py::TestTheRecordedSupersededReplyIsWhatSearchProduces` re-produces the reply and asserts equality, printing the produced JSON on a mismatch. |
| **Readers** | `orchestrator/tests/test_memory_recall.py::TestSupersededRender`, through `orchestrator/tests/_briefing_helpers.py::recorded_search_text` with `fixtures=SUPERSEDED_CHILD_FIXTURES`. |
| **Serialisation** | `pydantic_core.to_jsonable_python` of the tool result, then `json.dumps(..., indent=2, ensure_ascii=False)` plus one trailing newline. |

The scenario: parent `11111111-…-111111111111` (a matched hit) carries amendment
`22222222-…-222222222221` (not a hit) and sighting `33333333-…-333333333331`
(a matched hit). Both children carry `x_superseded_by` =
`0f6c2a4e-8b1d-4c3e-9a7f-5d2e1b0c9a8f`, the id of the record that superseded
them.

Load-bearing facts a reader must not re-derive from prose:

* **The amendment digest carries `metadata.x_superseded_by`**, projected from
  the child's raw payload.
* **The sighting appears only under `grouped.matched_children`**, with its full
  `content` and `metadata.x_superseded_by`. It is not a top-level result: the
  reply has exactly one.
* **The parent's own `metadata` is `{}`.** Only the children are superseded, so
  a renderer that tags the parent has read the mark from the wrong entry.
