# Provenance — `grouped_search_amendment_bodies` fixtures

These fixtures exist so the orchestrator's ceilings on amendment text in a
briefing are pinned against the bytes fused-memory's `search` tool actually emits for the worst
case the amendment text cap accepts (task 7518), not against a shape written
from its docstring. A hand-built body could pass a ceiling the producer
exceeds.

## The rule

> **If a test that reads these bytes fails, the fused-memory/orchestrator seam
> has drifted.** Re-record the fixture from the producer (below) and fix the
> **consumer**. Do **NOT** edit a fixture to make a test pass.

---

## `ten-amendments-over-the-cap.json`

| | |
|---|---|
| **Grounding** | **Real producer over a stub scenario.** The real `search` MCP tool (`fused-memory/src/fused_memory/server/tools.py::create_mcp_server` → `fused-memory/src/fused_memory/server/grouped_read.py::group_search_results`) ran over a stub memory service; only the service's reads are stubbed. The service's `config.write_triage.judge_field_chars` is a mock attribute, not a positive int, so `fused-memory/src/fused_memory/server/write_triage_judge.py::resolve_judge_field_chars` falls back to the shipped `write_triage.judge_field_chars` default (4,000). |
| **Producing code sites** | `fused-memory/src/fused_memory/server/grouped_read.py::_amendment_entry` gives each amendment entry `content` = `grouped_read.py::amendment_text` at that default: the text cut by `fused-memory/src/fused_memory/utils/clip_disclosure.py::elide_text` with the counted marker `…[+N chars elided]`. The scroll is capped at `grouped_read.py::AMENDMENT_LIST_CAP` (10). |
| **Pinning test** | `fused-memory/tests/server/test_amendment_bodies_search_reply.py` re-produces the reply and asserts equality, printing the produced JSON on a mismatch. |
| **Readers** | `orchestrator/tests/test_memory_recall.py::TestAmendmentBodyRender` and `orchestrator/tests/test_briefing_project_scope.py::TestTheBriefingsAmendmentTextIsBounded`, through `orchestrator/tests/_briefing_helpers.py::recorded_search_text` with `fixtures=AMENDMENT_BODY_FIXTURES`. |
| **Serialisation** | `pydantic_core.to_jsonable_python` of the tool result, then `json.dumps(..., indent=2, ensure_ascii=False)` plus one trailing newline. |

The scenario: parent `11111111-…-111111111111`, the only hit, carries ten
uncontested amendments `22222222-…-000000000000` to `…-000000000009`. Each body
is 4,500 characters: an opening sentence naming a marker `DETAIL-00` to
`DETAIL-09`, then deterministic filler.

Load-bearing facts a reader must not re-derive from prose:

* **There are 10 amendments**, the most a grouped document lists, and
  `amendment_count` is 10, so the block carries no `truncated` marker.
* **Every body is over the cap.** Each entry's `content` is the first 4,000
  characters followed by `…[+500 chars elided]`.
* **Each `DETAIL-nn` marker lies past character 240**, where the pre-7518
  digest cut every amendment, and inside the 4,000-character cap.
* **No entry carries a `digest` key**, and none is marked `contested`.
