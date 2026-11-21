"""Tests for orchestrator.predicate_output (task 3286's sanitizer and C1's verdict)."""

from __future__ import annotations

import json

import pytest
from _predicate_specimens import POLLUTED_PREDICATE_OUTPUT

from orchestrator.predicate_output import (
    PREDICATE_NOTE_MAX_PAYLOAD_CHARS,
    predicate_verdict,
    summarize_predicate_output,
)

# ---------------------------------------------------------------------------
# Task 3286: `summarize_predicate_output` — the ALLOWLIST sanitizer standing
# between a predicate script's raw stdout tail and `done_provenance.note`.
#
# The note is not a private field: fused-memory's `_format_outcome_echo` reads
# it and appends it to a Mem0 completion-summary write, so anything that lands
# here is ingested into the knowledge graph.  Task 2902 is the specimen that
# proved raw forwarding corrupts it.
# ---------------------------------------------------------------------------

class TestSummarizePredicateOutput:
    """The sanitizer keeps the structured verdict and drops the log noise."""

    def test_specimen_starts_with_deterministic_verdict_prefix(self):
        """The verdict prefix is unconditional — a note is never empty."""

        result = summarize_predicate_output(POLLUTED_PREDICATE_OUTPUT, rc=0)

        assert result.startswith('predicate check passed (rc=0)'), result

    @pytest.mark.parametrize(
        'marker',
        [
            'fused_memory.backends.graphiti_client',
            'httpx',
            'my_solar_challenge',
            'HTTP/1.1 200 OK',
            'WARNING',
        ],
    )
    def test_specimen_server_log_markers_are_dropped(self, marker: str):
        """No server-log noise survives into the note (the task-2902 leak)."""

        result = summarize_predicate_output(POLLUTED_PREDICATE_OUTPUT, rc=0)

        assert marker not in result, (
            f'{marker!r} leaked into the provenance note: {result!r}'
        )

    def test_specimen_trailing_json_verdict_survives_compacted(self):
        """The script's OWN structured verdict is the part worth keeping."""

        result = summarize_predicate_output(POLLUTED_PREDICATE_OUTPUT, rc=0)

        # Compact json.dumps separators — no space after ':' or ','.
        assert '"orphan_count":0' in result, result
        assert '"verdict":"clean"' in result, result

    def test_specimen_result_is_single_line(self):
        """A note flowing into a Mem0 summary must not carry newlines."""

        result = summarize_predicate_output(POLLUTED_PREDICATE_OUTPUT, rc=0)

        assert '\n' not in result, result

    def test_specimen_mid_token_first_line_is_dropped(self):
        """The 2000-char tail cut starts mid-word — that fragment is noise."""

        result = summarize_predicate_output(POLLUTED_PREDICATE_OUTPUT, rc=0)

        assert '_tariff_pence_per_kwh' not in result, result

    # -- Tier 2: one clean final line survives -----------------------------
    #
    # Each shape below is a REAL in-repo predicate script's output, and each
    # is load-bearing for a pre-existing green test.  A sanitizer that dropped
    # them would be throwing away the verdict it exists to preserve.

    @pytest.mark.parametrize(
        ('out', 'expected'),
        [
            # test_deterministic_runner.py's own predicate-mode fixtures.
            ('check ok: 0 flakes', 'check ok: 0 flakes'),
            # scripts/check_merge_flakiness.sh — drives the REAL-subprocess
            # test at test_milestone_integration_gate.py's exemplar-pass case.
            (
                'check_merge_flakiness: value=1 threshold=5 window_days=7 '
                '-- invariant holds',
                'invariant holds',
            ),
            # scripts/check_esc_analytics_perf.sh.
            (
                'measured_median_ms=12 attempts=5 threshold_ms=2000 '
                'url=http://127.0.0.1:8080/api/escalations/analytics',
                'measured_median_ms=12 attempts=5',
            ),
        ],
    )
    def test_clean_final_line_is_preserved(self, out: str, expected: str):
        """A single clean verdict line is the payload — kept verbatim."""

        result = summarize_predicate_output(out, rc=0)

        assert result.startswith('predicate check passed (rc=0)'), result
        assert expected in result, result

    def test_trailing_blank_lines_do_not_defeat_extraction(self):
        """A script ending with a newline still yields its verdict line."""

        result = summarize_predicate_output('check ok: 0 flakes\n\n', rc=0)

        assert 'check ok: 0 flakes' in result, result

    # -- Log-shaped lines are rejected -------------------------------------

    def test_pure_log_output_yields_bare_verdict(self):
        """Nothing but logger lines -> no payload at all, not a log line."""

        out = (
            '2026-07-30 16:39:00,523 fused_memory.backends.graphiti_client '
            'WARNING identity scan found 3 candidate nodes\n'
            '2026-07-30 16:39:00,625 httpx INFO HTTP Request: GET '
            'http://localhost:6333 "HTTP/1.1 200 OK"'
        )

        result = summarize_predicate_output(out, rc=0)

        assert result == 'predicate check passed (rc=0)', result

    @pytest.mark.parametrize(
        'out',
        [
            '2026-07-30 16:39:00,523 fused_memory.x INFO done',
            # No timestamp — the standalone level token alone is enough.
            'INFO: all checks passed',
        ],
    )
    def test_level_token_final_line_is_rejected(self, out: str):
        """Deliberately conservative: a level token forfeits the payload.

        Losing a payload is the safe failure direction — the verdict prefix
        always survives and the raw text is logged (step-6).
        """

        result = summarize_predicate_output(out, rc=0)

        assert result == 'predicate check passed (rc=0)', result

    def test_formatter_less_log_line_survives_as_documented_limitation(self):
        """Tier 2 is a DENYLIST, and this pins the gap it leaves open.

        A ``%(name)s %(message)s`` formatter emits no timestamp and no level
        token, so ``_LOG_LINE_RE`` does not reject it and the line is kept as
        the payload.  Tier 2 cannot be tightened into a grammar-based
        allowlist without also dropping ``check ok: 0 flakes`` and
        ``-- invariant holds`` (see the parametrized cases above), which are
        structurally identical prose.

        The exposure is bounded rather than closed — ONE line, capped — and
        this test exists so the boundary is asserted rather than assumed.  If
        a future change narrows tier 2, this expectation should flip
        deliberately, not silently.
        """

        out = 'httpx HTTP Request: GET http://localhost:6333 secret=abc'

        result = summarize_predicate_output(out, rc=0)

        assert result == f'predicate check passed (rc=0): {out}', result
        # Still bounded: a single line, never the multi-KB 2902 blob.
        assert '\n' not in result, result

    # -- The verdict word is derived from rc, never hardcoded ---------------

    def test_nonzero_rc_verdict_does_not_claim_passed(self):
        """The prefix cannot contradict the code it renders.

        Only the ``rc == 0`` branch calls this today, but a hardcoded
        ``passed`` would let a future caller stamp
        ``predicate check passed (rc=2)`` into a provenance note that flows
        into a Mem0 completion summary.
        """

        result = summarize_predicate_output('check failed: 3 flakes', rc=2)

        assert result.startswith('predicate check failed (rc=2)'), result
        assert 'passed' not in result, result
        assert 'check failed: 3 flakes' in result, result

    # -- Size bound: elide wholesale, never slice mid-structure -------------

    def test_oversized_payload_is_elided_not_sliced(self):
        """Task 2054's lesson: a mid-structure cut is worse than no payload.

        A raw ``note[:N]`` slice there garbled ``8679,8680`` into
        ``8679,868``.  A sliced JSON object is worse still — unparseable, yet
        still structured-looking to a reader.
        """

        obj = {'ids': list(range(2000))}
        compact = json.dumps(obj, separators=(',', ':'))
        assert len(compact) > PREDICATE_NOTE_MAX_PAYLOAD_CHARS, 'fixture too small'

        result = summarize_predicate_output(json.dumps(obj, indent=2), rc=0)

        # Bounded: the cap plus the verdict prefix and the marker's own text.
        assert len(result) <= PREDICATE_NOTE_MAX_PAYLOAD_CHARS + 200, len(result)
        # The marker names the dropped size and where the full text lives.
        assert str(len(compact)) in result, result
        assert 'elided' in result, result
        # Critically: NOT a prefix-slice of the compact dump.
        assert compact[:100] not in result, result

    def test_payload_at_the_cap_is_kept(self):
        """The bound elides only what exceeds it — it is not a blanket drop."""

        out = 'ok ' + 'x' * (PREDICATE_NOTE_MAX_PAYLOAD_CHARS - 10)
        result = summarize_predicate_output(out, rc=0)

        assert out in result, result
        assert 'elided' not in result, result

    # -- Degenerate inputs -------------------------------------------------

    @pytest.mark.parametrize('out', ['', None, 42, b'check ok', ['check ok']])
    def test_degenerate_output_yields_bare_verdict(self, out: object):
        """A falsy/non-str seam return never raises and never leaks a repr."""

        result = summarize_predicate_output(out, rc=0)

        assert result == 'predicate check passed (rc=0)', result


class TestPredicateVerdict:
    """``predicate_verdict`` keeps only a trailing JSON OBJECT, bounded by the note cap."""

    def test_trailing_json_object_is_the_verdict(self):
        out = 'some log line\n{"verdict": "clean", "n": 0}'

        assert predicate_verdict(out) == {'verdict': 'clean', 'n': 0}

    def test_trailing_json_array_is_not_a_verdict(self):
        assert predicate_verdict('x\n[1, 2]') is None

    def test_clean_final_line_is_not_a_verdict(self):
        assert predicate_verdict('check ok: 0 flakes') is None

    def test_non_str_output_is_not_a_verdict(self):
        assert predicate_verdict(None) is None

    def test_oversized_object_is_recorded_by_size_alone(self):
        obj = {'ids': list(range(2000))}
        compact = json.dumps(obj, separators=(',', ':'))
        assert len(compact) > PREDICATE_NOTE_MAX_PAYLOAD_CHARS, 'fixture too small'

        assert predicate_verdict(json.dumps(obj, indent=2)) == {
            'elided': True,
            'chars': len(compact),
        }
