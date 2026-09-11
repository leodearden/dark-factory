"""Tests for the stale-gate-citation guard (task 4919).

Closes the incident where task 3708's evidence-relay prose kept citing
``3660`` as a pending external gate for three relay cycles after 3660 was
coalesced into 4856 — see
``fused_memory.reconciliation.stale_gate_citation_guard``'s module docstring
for the full incident record and the corpus measurement.

Every fixture constant below is a VERBATIM excerpt from task 3708's live
``details`` field, re-confirmed present in ``.taskmaster/tasks/tasks.db``
(tag=master) on 2026-09-11. The expected value for each is the scanner output
observed in that run, not a guess.
"""

from __future__ import annotations

from fused_memory.reconciliation import stale_gate_citation_guard
from fused_memory.reconciliation.stale_gate_citation_guard import find_gate_citation_ids

# --------------------------------------------------------------------------- #
# Live relay excerpts (task 3708, tag=master, re-confirmed 2026-09-11)
# --------------------------------------------------------------------------- #

# The two STALE spellings: the defect itself. Live `dependencies` is
# [3658, 3659, 3707, 4856, 4987], so the cited 3660 is stale.
STALE_A = (
    'the only real remediation lever remains external deps 3658/3659/3660 '
    'landing so this task (γ) can be dispatched'
)
STALE_B = (
    'The only real remediation lever remains external deps 3658/3659/3660/4856 '
    'landing so this task (γ) can be dispatched'
)

# The 2026-08-31 correction relay. The capture stops at ' only', so the
# HISTORICAL 3660 named after it is never read as a citation — no separate
# negation heuristic is needed.
CORRECTION = (
    'All future evidence-log relays into this field must cite pending external '
    'gates as 3658/3659/4856 only — 3660 must be dropped permanently, it no '
    'longer exists as a separate gate'
)

# Correct relays that a CLAUSE-SCOPED rule would wrongly block: the ids after
# the capture are TRANSITIVE gates (gates of this task's gates), legitimately
# absent from this task's own `dependencies`.
TRANSITIVE = (
    'The only real remediation lever remains external deps 3659 and 4856 '
    'landing (via their own upstream gates 3212 and 4006 respectively)'
)
PARENTHETICAL = (
    'The only real remediation levers remain: 3659 (blocked on 3212→3207), '
    '4856 (blocked on 3659+4006), and now also 4987 (blocked on 4932+4986) '
    'landing'
)

# OUT-OF-SAMPLE. Both of these are relay prose written AFTER this algorithm was
# designed — they appear in the 2026-09-11 corpus scan but not the 2026-09-08
# one — and both must PASS. They are the evidence that the marker vocabulary
# generalises rather than being fitted to the incident. CAPS_GATING also
# exercises the ALL-CAPS spelling (hence re.IGNORECASE) and ARROW_LEVERS the
# colon-less connector.
CAPS_GATING = (
    'GATING DEPENDENCY 4987 OBSERVED (append-only; not evidence of a status '
    'change)'
)
ARROW_LEVERS = (
    'The only real remediation levers remain 3659 (→3212), 4856 (→3659+4006), '
    'and 4987 (→4932+4986)'
)


class TestFindGateCitationIds:
    """The pure marker-anchored contiguous id-list scanner."""

    def test_stale_a_captures_the_three_cited_gates(self):
        assert find_gate_citation_ids(STALE_A) == {3658, 3659, 3660}

    def test_stale_b_captures_all_four_cited_gates(self):
        assert find_gate_citation_ids(STALE_B) == {3658, 3659, 3660, 4856}

    def test_correction_relay_stops_before_the_historical_id(self):
        # Capture stops at ' only'; the trailing historical 3660 is NOT a
        # citation, so no negation heuristic is needed to exonerate it.
        assert find_gate_citation_ids(CORRECTION) == {3658, 3659, 4856}

    def test_transitive_gates_are_excluded(self):
        # Stops at ' landing' — upstream gates 3212/4006 are not this task's
        # dependencies and must not be read as citations of them.
        assert find_gate_citation_ids(TRANSITIVE) == {3659, 4856}

    def test_capture_stops_at_a_parenthetical(self):
        # Deliberate under-fire: only the first id of the list is captured.
        # Fail-open beats false-positive.
        assert find_gate_citation_ids(PARENTHETICAL) == {3659}

    def test_all_caps_marker_matches_case_insensitively(self):
        # Out-of-sample; stops at ' OBSERVED'.
        assert find_gate_citation_ids(CAPS_GATING) == {4987}

    def test_marker_without_a_connector_still_anchors(self):
        # Out-of-sample; no colon after the marker, stops at the arrow
        # parenthetical.
        assert find_gate_citation_ids(ARROW_LEVERS) == {3659}

    def test_marker_with_no_adjacent_id_list_is_a_no_op(self):
        assert find_gate_citation_ids(
            'FRESH STATUS CHECK ON ALL FIVE GATING DEPENDENCIES:'
        ) == set()

    def test_text_with_no_marker_is_a_no_op(self):
        assert find_gate_citation_ids(
            'actual_total=2 / expected_total=38 (36 missing)'
        ) == set()

    def test_blocked_on_is_not_a_marker(self):
        # Measured to false-positive on real relay prose: '3659: pending —
        # blocked on 3212' cites a correct TRANSITIVE gate.
        assert find_gate_citation_ids('blocked on 3212') == set()

    def test_upstream_gates_is_not_a_marker(self):
        # Same reason: these are gates of this task's gates.
        assert find_gate_citation_ids(
            'via their own upstream gates 3212 and 4006'
        ) == set()

    def test_empty_text_is_a_no_op(self):
        assert find_gate_citation_ids('') == set()

    def test_text_with_no_digits_is_a_no_op(self):
        assert find_gate_citation_ids(
            'the only real remediation lever remains external deps landing'
        ) == set()

    def test_marker_alternation_is_module_level(self):
        # The regex is frozen at import time, not rebuilt per call.
        assert stale_gate_citation_guard.GATE_CITATION_RE.groups == 1


class TestTerminalOutcomeEscape:
    """A RETROSPECTIVE statement about gates that already landed is a
    legitimate relay, even when written against an already-emptied
    `dependencies` array. Suppressing those is the false-positive class the
    trailing-tail escape exists to remove."""

    def test_have_landed_is_not_a_pending_gate_assertion(self):
        assert find_gate_citation_ids(
            'external deps 3658/3659 have landed and this task is unblocked'
        ) == set()

    def test_are_all_done_is_not_a_pending_gate_assertion(self):
        assert find_gate_citation_ids(
            'The external deps 3658/3659/4856 are all done'
        ) == set()

    def test_were_merged_is_not_a_pending_gate_assertion(self):
        assert find_gate_citation_ids(
            'pending external gates 3658/3659 were merged last week'
        ) == set()

    def test_escape_does_not_suppress_the_live_stale_spelling(self):
        # THE LOAD-BEARING ASSERTION. The corpus spells the defect
        # '… 3658/3659/3660 landing so this task (γ) can be dispatched', and
        # TERMINAL_OUTCOME_RE's `\blanded\b` does not match 'landing'. Measured:
        # with the escape applied, the would-fire count over task 3708's 9
        # matches stays exactly 3.
        assert find_gate_citation_ids(STALE_A) == {3658, 3659, 3660}

    def test_escape_does_not_suppress_the_transitive_relay(self):
        # Tail is ' landing (via their own upstream gates …' — no terminal cue.
        assert find_gate_citation_ids(TRANSITIVE) == {3659, 4856}

    def test_escape_does_not_suppress_the_out_of_sample_caps_relay(self):
        # Tail is ' OBSERVED (append-only; not evidence of …' — no terminal cue.
        assert find_gate_citation_ids(CAPS_GATING) == {4987}

    def test_terminal_cue_beyond_the_tail_window_does_not_suppress(self):
        # 'done' here is far past the end of the citation, describing something
        # else entirely; only a cue in the immediate tail is a retrospective.
        assert find_gate_citation_ids(
            'external deps 3658/3659 landing so this task can finally be '
            'dispatched once everything is done'
        ) == {3658, 3659}
