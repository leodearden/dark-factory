"""Predicate-output specimens shared by the predicate test files.

Imported by bare module name (``from _predicate_specimens import ...``),
like ``_proc_supervision_doubles`` -- ``orchestrator/tests/`` has no
``__init__.py``.

One copy of ``POLLUTED_PREDICATE_OUTPUT``, used by
``test_predicate_output.py`` (the summarizer's unit tests) and by
``test_deterministic_runner.py`` (the predicate pass path that stamps the
summarized note).
"""

# ---------------------------------------------------------------------------
# Task 3286 — the real task-2902 specimen shape.
#
# `orchestrator/src/orchestrator/before_done_script.py::run_script` merges
# stderr into stdout and returns `[-2000:]`, so a
# chatty predicate script's server-log noise reached `done_provenance.note`
# verbatim — and from there fused-memory's `_format_outcome_echo` wrote it into
# a Mem0 completion summary.  The live note on task 2902 is exactly 1999 chars:
# it starts MID-TOKEN (the 2000-char tail cut), carries FalkorDB identity-scan
# WARNINGs naming the unrelated project `my_solar_challenge` plus `httpx` HTTP
# request lines, and ends with the script's own pretty-printed dry-run JSON
# verdict — the ONE part actually worth keeping.
#
# Shape-faithful, not byte-identical: reproduced from the plan's analysis (the
# live task is a preserved forensic specimen and is never re-read or mutated
# by this suite).  An abridged copy lives in
# scripts/tests/test_scan_provenance_note_log_leaks.py — the two suites cannot
# share imports across orchestrator/tests/ and scripts/tests/.
# ---------------------------------------------------------------------------

POLLUTED_PREDICATE_OUTPUT = """\
_tariff_pence_per_kwh' in group 'my_solar_challenge' (exact-name identity gate should prevent this — investigate)
2026-07-30 16:39:00,523 fused_memory.backends.graphiti_client WARNING identity scan found 3 candidate nodes for 'import_tariff_pence_per_kwh' in group 'my_solar_challenge'
2026-07-30 16:39:00,584 fused_memory.backends.graphiti_client WARNING identity scan found 2 candidate nodes for 'export_tariff_pence_per_kwh' in group 'my_solar_challenge'
2026-07-30 16:39:00,625 httpx INFO HTTP Request: GET http://localhost:6333 "HTTP/1.1 200 OK"
2026-07-30 16:39:00,701 httpx INFO HTTP Request: POST http://localhost:6333/collections/mem0/points/scroll "HTTP/1.1 200 OK"
2026-07-30 16:39:00,742 fused_memory.backends.graphiti_client WARNING identity scan found 4 candidate nodes for 'battery_state_of_charge' in group 'my_solar_challenge'
{
  "dry_run": true,
  "before": {
    "total_source": 0,
    "total_with_kind": 0
  },
  "orphan_count": 0,
  "orphan_ids": [],
  "verdict": "clean"
}"""
