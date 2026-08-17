"""Tests for shared.decision_pairing — the semantic cross-pairing detector.

Task 3967. The damage class is a ``design_decisions`` entry whose ``decision``
and ``rationale`` are each perfectly well-formed prose but whose *association*
is wrong. Nothing is malformed and no sentinel is present, so
``shared.toolcall_markup.detect`` is structurally blind to it — the two damage
classes are disjoint at the detector, which is why this is a separate module
rather than a widening of that one.

The only machine-visible trace is the **correction entry an author appends
after noticing**. That is what this predicate keys on, and it is why every
count derived from it is a strict lower bound.

TDD pair 1 (step 1/2): :func:`detect_mispairing`, the entry-level predicate.
TDD pair 2 (step 3/4): :func:`scan_design_decisions`, the document walker.
TDD pair 3 (step 5/6): the committed-corpus replay.

## Specimens are modelled on real victim text

Every hand-authored specimen below is paraphrased from an entry actually
observed in a live plan (the task id is named in each docstring), not invented.
The committed corpus in ``TestCommittedCorpus`` replays the real text verbatim;
these exist to state the CONTRACT in a form a reader can check by eye, and to
cover the one branch no live specimen exercises (see
``test_pairing_language_in_rationale_alone_counts``).

## Sentinel-literal hazard — DO NOT "helpfully" un-escape these

Every envelope literal in this file is spelled with the ``\\x3c`` escape for
the opening angle bracket, exactly as ``shared/tests/test_toolcall_markup.py``
and ``fused_memory/utils/toolcall_xml_leak.py`` lines 77-86 require. Writing
one verbatim here would force any agent editing this file to emit that literal
inside its own tool-call envelope, reproducing a defect adjacent to the one
these tests pin. It is byte-identical at runtime and never appears verbatim in
the file text. Leave it escaped.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from shared.decision_pairing import (
    HEADER_MARKERS,
    PAIRING_FIELDS,
    PAIRING_MARKERS,
    MispairingHit,
    detect_mispairing,
    scan_design_decisions,
)

# ---------------------------------------------------------------------------
# Specimens, paraphrased from live victim plans.
# ---------------------------------------------------------------------------

# Modelled on 3098[1] / 3216[1] / 3473[1] — by far the commonest live shape.
CORRECTION_DECISION = (
    'CORRECTION of the preceding entry, whose rationale was mis-paired with '
    'its statement. The two were recorded in one call and the arguments were '
    'associated the wrong way round; read this entry as the authoritative one.'
)
CORRECTION_RATIONALE = (
    'The preceding entry states the scoping choice but carries the rationale '
    'for the timeout override, which belongs to a different decision entirely.'
)

# Modelled on 3298[1] — the header form with an explicit slash.
RESTATEMENT_DECISION = (
    'CORRECTION/RESTATEMENT of decision #1, which was mis-paired with the '
    'rationale for decision #2 when both were recorded.'
)

# Modelled on 3567[3] and 3209[3].
READ_THIS_INSTEAD_DECISION = (
    'READ THIS INSTEAD OF DECISIONS 2 AND 3, whose decision-lines and '
    'rationales were cross-paired: each rationale sits under the other '
    'decision-line.'
)

# Modelled on 3042[2] and 4096[3] — the SUPERSEDES header used for a genuine
# mis-pairing, which is what makes 3382[5] below a discriminating control.
SUPERSEDES_DECISION = (
    'SUPERSEDES the entry above, whose rationale was mis-paired: the text '
    'recorded there argues for the atomic-write choice, not for this one.'
)

# Modelled on 3210[1].
CORRECTED_DECISION = (
    'CORRECTED entry. The rationale immediately above was recorded against '
    'the wrong decision-line; this entry restores the intended association.'
)

# Modelled on 3727[1], the wire-evidence specimen.
MIS_TITLED_DECISION = (
    'CORRECTION of the mis-titled entry above: the preceding rationale '
    'belongs to THIS decision, not to the one it was filed under.'
)

# Modelled on 3209[3] — the `swapped` phrasing.
SWAPPED_DECISION = (
    'READ THIS INSTEAD OF THE TWO ENTRIES ABOVE, whose rationales were '
    'swapped at the point they were recorded.'
)

ORDINARY_RATIONALE = (
    'Re-verified on this worktree against the live code path before the '
    'entry was written; the measurement is reproducible from the scanner.'
)


# ---------------------------------------------------------------------------
# Negative controls, quoted from the four observed near-miss classes.
# ---------------------------------------------------------------------------

# 3382[5] verbatim in shape: a GENUINE design reversal. The header conjunct
# holds and the entry really does supersede another — but nothing was
# mis-paired, so the pairing conjunct must reject it.
GENUINE_SUPERSESSION_DECISION = (
    'SUPERSEDES decision #3 ("mirror the uv_cache precedent exactly: '
    'grant-only, no pre-creation helper"). Its ACCEPTED RESIDUAL RISK is '
    'REJECTED as a confirmed blocking defect.'
)
GENUINE_SUPERSESSION_RATIONALE = (
    'Decision #3 argued the cache directory is TOOL-owned like the uv cache, '
    'so the grant-only precedent applies. That reasoning does not survive '
    'scrutiny: the grammar builds run INSIDE the sandbox.'
)

# 3209[1] in shape: `supersedes` used mid-sentence as the name of a metadata
# edge kind. No start-anchored header at all.
INCIDENTAL_KEYWORD_DECISION = (
    'The dangling census is split across TWO metrics — one count over all '
    'three pointer keys, and a tripwire over `supersedes` edges only — '
    'rather than one, and the correction is applied at read time.'
)

# 3298[2] in shape: a restatement named PARENTHETICALLY, mid-prose. Only the
# start-anchor separates this from a positive.
PARENTHETICAL_RESTATEMENT_DECISION = (
    'Annotate the PRD\'s historical "~42s" in place rather than overwriting '
    'the number (restatement of decision #1 with its correct rationale).'
)

# 3692[8] in shape: prose ABOUT another plan's mis-pairing. This is the false
# positive the originating task description acknowledged, and the one the
# start-anchor is what sheds.
META_PROSE_DECISION = (
    "No RED test asserts that task 3567's plan is repaired, despite the task "
    'description naming it as the concrete damage specimen to use as a test '
    'case.'
)
META_PROSE_RATIONALE = (
    'Its damage is SEMANTIC cross-pairing (a rationale recorded against the '
    'wrong decision), which carries no sentinel and is therefore invisible to '
    'the envelope detector.'
)


class TestDetectMispairing:
    """The entry-level predicate: a CONJUNCTION, not a keyword scan."""

    @pytest.mark.parametrize(
        ('decision', 'expected_header'),
        [
            (CORRECTION_DECISION, 'CORRECTION'),
            (RESTATEMENT_DECISION, 'CORRECTION'),
            (READ_THIS_INSTEAD_DECISION, 'READ THIS INSTEAD'),
            (SUPERSEDES_DECISION, 'SUPERSEDES'),
            (CORRECTED_DECISION, 'CORRECTED'),
            (MIS_TITLED_DECISION, 'CORRECTION'),
            (SWAPPED_DECISION, 'READ THIS INSTEAD'),
        ],
    )
    def test_each_live_header_form_fires(self, decision: str, expected_header: str) -> None:
        """Every header form observed in a live victim plan is detected."""
        hit = detect_mispairing(decision, ORDINARY_RATIONALE)
        assert hit is not None, f'header form {expected_header!r} was not detected'
        assert hit.header == expected_header

    @pytest.mark.parametrize(
        ('phrase', 'expected_marker'),
        [
            ('the rationale above was mis-paired with this statement', 'mis-paired'),
            ('the two entries were cross-paired when recorded', 'cross-paired'),
            ('this corrects the mis-titled entry above', 'mis-titled'),
            ('the rationale was mis-attributed to the wrong entry', 'mis-attributed'),
            ('that text was recorded against the wrong decision', 'recorded against the wrong'),
            ('the two rationales were swapped at the point of writing', 'swapped'),
            ('the preceding rationale belongs to THIS decision', 'belongs to THIS decision'),
        ],
    )
    def test_each_pairing_marker_fires(self, phrase: str, expected_marker: str) -> None:
        """Every pairing phrase observed in a live victim plan is detected.

        The hit NAMES the matched marker, so a triager is never sent to
        log-scrape which literal fired (INV-2).
        """
        hit = detect_mispairing('CORRECTION. ' + phrase + '.', ORDINARY_RATIONALE)
        assert hit is not None, f'pairing marker {expected_marker!r} was not detected'
        assert hit.marker == expected_marker

    def test_hit_names_both_matched_markers_and_the_field(self) -> None:
        """A hit is self-describing: header, pairing marker, and where it matched."""
        hit = detect_mispairing(CORRECTION_DECISION, CORRECTION_RATIONALE)
        assert isinstance(hit, MispairingHit)
        assert hit.header == 'CORRECTION'
        assert hit.marker == 'mis-paired'
        assert hit.field == 'decision'
        # The markers a hit reports are the module's OWN declared literals, not
        # a re-spelling. If this drifts, some consumer has a second copy.
        assert hit.header in HEADER_MARKERS
        assert hit.marker in PAIRING_MARKERS

    def test_pairing_language_in_rationale_alone_counts(self) -> None:
        """Header on ``decision``, pairing language only in ``rationale``.

        NO LIVE SPECIMEN EXERCISES THIS BRANCH. Measured over all 23 corpus
        positives: every one carries its pairing marker in ``decision``; none
        carries it in ``rationale`` only, and none in both. (The task's plan
        asserted specimen 3727[1] splits the evidence this way; it does not —
        it reads "CORRECTION of the mis-titled entry above: the preceding
        rationale belongs to THIS decision", both halves in one field.)

        The disjunction is deliberate recall headroom for a correction phrased
        with a bare header and its reasoning deferred to the rationale. Because
        no corpus record covers it, this hand-authored specimen is the ONLY
        thing pinning it — do not delete it as redundant.
        """
        hit = detect_mispairing(
            'CORRECTION of the entry immediately above.',
            'Its rationale was mis-paired with a statement from a different '
            'decision when both were recorded in one call.',
        )
        assert hit is not None
        assert hit.header == 'CORRECTION'
        assert hit.marker == 'mis-paired'
        assert hit.field == 'rationale'

    def test_decision_field_wins_when_both_carry_the_marker(self) -> None:
        """``decision`` is searched first, so the reported field is stable."""
        hit = detect_mispairing(CORRECTION_DECISION, 'Also mis-paired, for the record.')
        assert hit is not None
        assert hit.field == 'decision'

    # -- Negative controls, one per observed near-miss class -----------------

    def test_genuine_supersession_without_pairing_language_is_clean(self) -> None:
        """3382[5]: a real design REVERSAL, not a mis-pairing.

        The header conjunct holds. Dropping the pairing conjunct would sweep
        this in, which is the precision cost the conjunction buys off.
        """
        assert detect_mispairing(
            GENUINE_SUPERSESSION_DECISION, GENUINE_SUPERSESSION_RATIONALE
        ) is None

    def test_incidental_mid_prose_keyword_is_clean(self) -> None:
        """3209[1]: `supersedes` and `correction` used mid-sentence.

        Dropping the start-anchor would sweep in three separate 3209 entries.
        """
        assert detect_mispairing(INCIDENTAL_KEYWORD_DECISION, ORDINARY_RATIONALE) is None

    def test_parenthetical_restatement_is_clean(self) -> None:
        """3298[2]: a restatement named parenthetically, mid-prose.

        The hardest control: the word ``restatement`` is a declared header
        marker AND the entry really does restate another. Only the START-ANCHOR
        separates it from a positive.
        """
        assert detect_mispairing(
            PARENTHETICAL_RESTATEMENT_DECISION, ORDINARY_RATIONALE
        ) is None

    def test_meta_prose_about_another_plans_mispairing_is_clean(self) -> None:
        """3692[8]: prose ABOUT task 3567's mis-pairing.

        The acknowledged false positive of the originating task description's
        own phrase-list predicate. Its rationale contains the pairing phrase
        ``recorded against the wrong`` verbatim; the start-anchor sheds it.
        """
        assert detect_mispairing(META_PROSE_DECISION, META_PROSE_RATIONALE) is None

    @pytest.mark.parametrize('decision', ['', '   ', '\n\t  \n'])
    def test_empty_or_whitespace_decision_is_clean(self, decision: str) -> None:
        assert detect_mispairing(decision, 'mis-paired with something') is None

    def test_header_alone_without_pairing_language_is_clean(self) -> None:
        """Both conjuncts are required; neither alone suffices."""
        assert detect_mispairing('CORRECTION. The timeout is 150s, not 30s.', '') is None

    def test_pairing_language_alone_without_a_header_is_clean(self) -> None:
        assert detect_mispairing(
            'The rationale here was mis-paired in an earlier draft.', ''
        ) is None

    # -- Totality ------------------------------------------------------------

    @pytest.mark.parametrize(
        ('decision', 'rationale'),
        [
            (None, None),
            (None, 'CORRECTION. mis-paired.'),
            (CORRECTION_DECISION, None),
            (CORRECTION_DECISION, 12345),
            (12345, CORRECTION_RATIONALE),
            ({'decision': CORRECTION_DECISION}, []),
            ([CORRECTION_DECISION], object()),
            (b'CORRECTION. mis-paired.', b''),
            (b'CORRECTION. mis-paired.', 'mis-paired'),
        ],
    )
    def test_never_raises_for_non_str_inputs(self, decision, rationale) -> None:
        """Total for any input. A detector that raises is a detector that is off.

        The walker in ``scan_design_decisions`` guards these shapes too, but the
        entry predicate must be independently total: it is called directly by
        the scanner script over plan documents nobody validated.

        This asserts TOTALITY and a well-typed verdict, deliberately NOT that
        every row is clean. Two rows carry a valid ``str`` decision that already
        satisfies both conjuncts on its own; a non-``str`` rationale must not
        suppress that hit, which is pinned separately below. Asserting ``is
        None`` here would have conflated "does not raise" with "finds nothing"
        and would have demanded the wrong behaviour.
        """
        verdict = detect_mispairing(decision, rationale)
        assert verdict is None or isinstance(verdict, MispairingHit)

    def test_non_str_rationale_does_not_suppress_a_decision_only_hit(self) -> None:
        """A garbage rationale never masks evidence that is complete in ``decision``.

        The commonest live shape by far: 23 of 23 corpus positives carry both
        conjuncts in ``decision``. If a non-``str`` sibling could suppress the
        hit, the walker would go quiet on exactly the adversarial plan shapes it
        exists to survive.
        """
        for rationale in (None, 12345, [], object()):
            hit = detect_mispairing(CORRECTION_DECISION, rationale)
            assert hit is not None, f'suppressed by a {type(rationale).__name__} rationale'
            assert hit.field == 'decision'

    @pytest.mark.parametrize(
        'decision',
        [None, 12345, {'decision': CORRECTION_DECISION}, [CORRECTION_DECISION], b'CORRECTION.'],
    )
    def test_non_str_decision_is_always_clean(self, decision) -> None:
        """The header conjunct is anchored on ``decision`` ALONE.

        A non-str ``decision`` can never satisfy it, however the rationale
        reads — so the disjunction never promotes a rationale-only match into a
        hit on its own.
        """
        assert detect_mispairing(decision, 'CORRECTION. It was mis-paired.') is None

    def test_leading_whitespace_before_the_header_is_tolerated(self) -> None:
        hit = detect_mispairing('\n  ' + CORRECTION_DECISION, ORDINARY_RATIONALE)
        assert hit is not None
        assert hit.header == 'CORRECTION'

    def test_header_and_marker_matching_are_case_insensitive(self) -> None:
        """Live specimens shout their headers; the predicate must not depend on it."""
        hit = detect_mispairing(
            'Correction of the entry above, whose rationale was Mis-Paired.', ''
        )
        assert hit is not None
        assert hit.header == 'CORRECTION'
        assert hit.marker == 'mis-paired'

    def test_marker_sets_are_immutable_and_non_empty(self) -> None:
        """One owner, and no consumer can mutate the sets out from under it."""
        assert isinstance(HEADER_MARKERS, tuple)
        assert isinstance(PAIRING_MARKERS, tuple)
        assert HEADER_MARKERS and PAIRING_MARKERS
        assert all(isinstance(m, str) and m for m in HEADER_MARKERS)
        assert all(isinstance(m, str) and m for m in PAIRING_MARKERS)

    def test_the_entry_field_names_are_a_public_constant_consumers_can_import(
        self,
    ) -> None:
        """``PAIRING_FIELDS`` is part of the single-owner surface, not private.

        The scanner script needs the same two names to look up the entry whose
        envelope verdict it reports beside this predicate's. Re-spelling them
        there would let the two damage classes be asked about DIFFERENT fields
        — a wrong column rather than a missing one — so the tuple is public and
        imported. Pinned so a rename back to a private name fails here, at the
        owner, rather than silently in the importer.

        The ORDER is asserted too: ``decision`` first is what makes a hit's
        reported ``field`` deterministic when both texts carry a marker.
        """
        assert PAIRING_FIELDS == ('decision', 'rationale')
        assert isinstance(PAIRING_FIELDS, tuple)

    def test_an_envelope_literal_does_not_by_itself_make_a_hit(self) -> None:
        """The two damage classes are disjoint AT THE DETECTOR.

        A string carrying an envelope literal is the OTHER class's business
        (``shared.toolcall_markup``); this predicate must not fire on it absent
        its own two conjuncts. Specimen 3209[8] is exactly this shape in the
        committed corpus, where it is a negative control.
        """
        envelope = '\x3c/decision>\n\x3cparameter name="rationale">'
        assert detect_mispairing('The census is computed inline. ' + envelope, '') is None


class TestScanDesignDecisions:
    """The document-level walker: total for adversarial plan shapes, read-only.

    Its guard set mirrors ``plan_tools._walk_repairable`` (skip a missing key, a
    non-list collection, a non-dict item, a non-str field) so the two walkers
    agree on what an adversarial plan shape means.
    """

    @staticmethod
    def _plan(*entries: object) -> dict:
        return {'task_id': '9999', 'design_decisions': list(entries)}

    @staticmethod
    def _entry(decision: object, rationale: object = ORDINARY_RATIONALE) -> dict:
        return {'decision': decision, 'rationale': rationale}

    def test_finds_a_single_mispairing_and_names_its_index(self) -> None:
        plan = self._plan(
            self._entry('Return a Path, never a bool; callers spell existence as `is not None`.'),
            self._entry(CORRECTION_DECISION),
        )
        hits = scan_design_decisions(plan)
        assert len(hits) == 1
        assert hits[0].index == 1
        assert hits[0].header == 'CORRECTION'
        assert hits[0].marker == 'mis-paired'

    def test_two_mispairings_in_one_plan_both_reported_in_index_order(self) -> None:
        """The real 3209 / 3567 / 4096 shape: two corrections in one plan.

        Three of the twenty victim plans carry two matched entries each, so a
        walker that stopped at the first would silently under-report 3 of 20.
        """
        plan = self._plan(
            self._entry('An ordinary decision with nothing wrong with it.'),
            self._entry(READ_THIS_INSTEAD_DECISION),
            self._entry('Another ordinary decision.'),
            self._entry(SUPERSEDES_DECISION),
        )
        hits = scan_design_decisions(plan)
        assert [h.index for h in hits] == [1, 3]
        assert [h.header for h in hits] == ['READ THIS INSTEAD', 'SUPERSEDES']

    def test_indices_are_ascending_even_with_unwalkable_entries_interleaved(self) -> None:
        """A skipped item still consumes its index, so indices stay addressable.

        If the walker enumerated only the entries it could read, a reported
        index would not address the entry a human opens the plan to find.
        """
        plan = self._plan(
            'not a dict at all',
            self._entry(CORRECTION_DECISION),
            None,
            self._entry(CORRECTED_DECISION),
        )
        hits = scan_design_decisions(plan)
        assert [h.index for h in hits] == [1, 3]
        assert plan['design_decisions'][1]['decision'] is CORRECTION_DECISION
        assert plan['design_decisions'][3]['decision'] is CORRECTED_DECISION

    # -- Totality for adversarial plan shapes --------------------------------

    @pytest.mark.parametrize(
        'plan',
        [
            {},
            {'task_id': '9999'},
            {'design_decisions': None},
            {'design_decisions': 'CORRECTION. mis-paired.'},
            {'design_decisions': {}},
            {'design_decisions': 42},
            {'design_decisions': []},
            {'design_decisions': [None, 42, 'text', [], object()]},
            {'design_decisions': [{}, {'decision': None}, {'rationale': None}]},
            {'design_decisions': [{'decision': 42, 'rationale': []}]},
            {'design_decisions': [{'decision': b'CORRECTION. mis-paired.'}]},
        ],
        ids=[
            'empty-plan', 'key-absent', 'value-none', 'value-str', 'value-dict',
            'value-int', 'value-empty-list', 'non-dict-items', 'fields-absent',
            'fields-non-str', 'field-bytes',
        ],
    )
    def test_adversarial_shapes_yield_no_hit_and_no_exception(self, plan: dict) -> None:
        assert scan_design_decisions(plan) == []

    def test_non_dict_plan_is_tolerated(self) -> None:
        """The scanner hands it whatever ``json.loads`` returned from a plan file.

        A plan file holding a bare list or string parses fine and reaches the
        walker, so the top-level shape is as untrusted as the entries.
        """
        for plan in (None, [], 'CORRECTION. mis-paired.', 42, object()):
            assert scan_design_decisions(plan) == []

    # -- Read-only by construction -------------------------------------------

    def test_never_mutates_the_plan(self) -> None:
        """Read-only is ASSERTED, not merely documented.

        The walker is handed live plan documents by the scanner. A walker that
        normalized a field in passing would be mutating runtime state that a
        running task may be reading.
        """
        plan = self._plan(
            self._entry('An ordinary decision.'),
            self._entry(CORRECTION_DECISION, CORRECTION_RATIONALE),
            'not a dict',
            self._entry(42, None),
        )
        before = copy.deepcopy(plan)
        hits = scan_design_decisions(plan)
        assert hits, 'guard: the specimen must actually produce a hit'
        assert plan == before

    def test_does_not_mutate_an_adversarial_plan_either(self) -> None:
        """The skip paths are where a normalizing walker would be tempted."""
        plan = {'design_decisions': [None, {}, {'decision': 42}, {'rationale': 'x'}]}
        before = copy.deepcopy(plan)
        assert scan_design_decisions(plan) == []
        assert plan == before

    def test_returns_a_fresh_list_each_call(self) -> None:
        """No shared accumulator: a caller mutating the result cannot poison the next."""
        plan = self._plan(self._entry(CORRECTION_DECISION))
        first = scan_design_decisions(plan)
        first.clear()
        assert len(scan_design_decisions(plan)) == 1

    def test_delegates_its_verdict_to_detect_mispairing(self) -> None:
        """One mechanism, two readers (INV-5).

        The walker must not re-implement the predicate: if it did, the entry
        tests above and the corpus replay below could pass while the scanner —
        the only consumer anyone actually runs — disagreed with both.
        """
        for decision in (
            CORRECTION_DECISION,
            GENUINE_SUPERSESSION_DECISION,
            PARENTHETICAL_RESTATEMENT_DECISION,
            META_PROSE_DECISION,
        ):
            plan = self._plan(self._entry(decision))
            entry_verdict = detect_mispairing(decision, ORDINARY_RATIONALE)
            walker_verdict = scan_design_decisions(plan)
            assert bool(walker_verdict) == (entry_verdict is not None), decision[:60]
            if entry_verdict is not None:
                assert walker_verdict[0].header == entry_verdict.header
                assert walker_verdict[0].marker == entry_verdict.marker
                assert walker_verdict[0].field == entry_verdict.field


# ---------------------------------------------------------------------------
# Committed-corpus replay (TDD pair 3).
# ---------------------------------------------------------------------------

CORPUS_PATH = Path(__file__).parent / 'fixtures' / 'decision_pairing_corpus.jsonl'

#: The victim task ids the corpus was extracted from, named here so a future
#: edit CANNOT SILENTLY DROP A SPECIMEN. Dropping one means editing this list in
#: the same commit, which is a reviewable event; a widened predicate that
#: quietly loses a victim plan is not. This is an assertion about the COMMITTED
#: corpus's contents, never about the live .worktrees tree — nothing in this
#: file reads that.
VICTIM_TASK_IDS = frozenset({
    '3042', '3098', '3201', '3209', '3210', '3216', '3298', '3337', '3363',
    '3382', '3415', '3473', '3567', '3664', '3668', '3727', '3757', '3918',
    '4030', '4096',
})

#: The four near-miss classes observed in live plans. Each must be represented
#: by at least one negative control, because each is a distinct way the
#: predicate could lose precision if a conjunct were dropped.
NEAR_MISS_CLASSES = frozenset({
    'genuine-supersession-no-pairing-language',
    'incidental-mid-prose-keyword',
    'parenthetical-restatement-not-headed',
    'meta-prose-about-another-plans-mispairing',
})


def load_corpus() -> list[dict]:
    """Every committed specimen, parsed.

    The fixture stores each opening angle bracket as its ``\\u003c`` JSON
    escape, so the FILE TEXT carries none while the parsed value is
    byte-identical to the harvested text. ``json.loads`` undoes it here; do not
    "helpfully" rewrite the fixture to hold them verbatim.

    Read at ``encoding='utf-8'`` rather than in the ambient locale, matching
    ``scan_plan_file``. The corpus is pure ASCII today so nothing misbehaves
    now; the risk is a later regeneration admitting one non-ASCII character,
    after which this replay — the durable artifact the whole prevalence finding
    rests on — would give a different answer on two machines.
    """
    lines = CORPUS_PATH.read_text(encoding='utf-8').splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def replay(record: dict) -> MispairingHit | None:
    """The verdict under test for one record — spelled once, used everywhere."""
    return detect_mispairing(record['text'], record['rationale_text'])


class TestCommittedCorpus:
    """Replay real victim text, and assert AGREEMENT WITH THE COMMITTED COLUMN.

    Never a bare count threshold. A future predicate that legitimately detects
    more must update the ``expect`` column in the same commit, which is a
    reviewable improvement rather than a RED test. And no live count is
    asserted anywhere: ``.worktrees/.task-meta`` grew from 1,196 to 1,297 plans
    in about eight days, so a pinned prevalence would be flaky by construction
    AND would invert the signal, making a better predicate read as a regression.
    """

    def test_corpus_is_present_and_non_empty(self) -> None:
        records = load_corpus()
        assert records, 'the committed corpus is empty — the durable artifact is gone'
        assert all(r['expect'] in {'mispaired', 'clean'} for r in records)

    def test_the_fixture_text_carries_no_raw_opening_angle_bracket(self) -> None:
        """The authoring hazard, asserted rather than trusted to a convention.

        Three specimens genuinely carry envelope literals (3201[1], 3382[1],
        3209[8]). If a future regeneration wrote them verbatim, an agent
        editing this fixture would emit that literal inside its own tool-call
        arguments and truncate its own write.
        """
        assert chr(60) not in CORPUS_PATH.read_text(encoding='utf-8')

    def test_every_record_matches_its_committed_expectation(self) -> None:
        disagreements = []
        for record in load_corpus():
            verdict = replay(record)
            actual = 'mispaired' if verdict is not None else 'clean'
            if actual != record['expect']:
                disagreements.append(
                    f"{record['task_id']}[{record['index']}]: "
                    f"expected {record['expect']}, got {actual}"
                )
        assert not disagreements, (
            'The detector disagrees with the committed corpus:\n  '
            + '\n  '.join(disagreements)
            + '\n\nIf the predicate legitimately improved, update the `expect` '
              'column (and the README sidecar) IN THE SAME COMMIT. Never add a '
              'per-specimen exception to the detector — widen or tighten the '
              'marker sets, which live in exactly one place.'
        )

    def test_positives_report_a_declared_header_and_pairing_marker(self) -> None:
        """A hit names its own evidence, from the module's own tuples (INV-2)."""
        for record in load_corpus():
            if record['expect'] != 'mispaired':
                continue
            hit = replay(record)
            assert hit is not None
            assert hit.header in HEADER_MARKERS, f"{record['task_id']}[{record['index']}]"
            assert hit.marker in PAIRING_MARKERS, f"{record['task_id']}[{record['index']}]"
            assert hit.field in {'decision', 'rationale'}

    def test_the_committed_field_column_agrees_with_the_detector(self) -> None:
        """The stored ``field`` is a measurement, so the detector must reproduce it."""
        for record in load_corpus():
            hit = replay(record)
            if hit is None:
                assert record['field'] is None, f"{record['task_id']}[{record['index']}]"
            else:
                assert hit.field == record['field'], f"{record['task_id']}[{record['index']}]"

    def test_every_victim_task_id_is_still_represented(self) -> None:
        """A widened predicate must not silently drop a victim plan."""
        detected = {r['task_id'] for r in load_corpus()
                    if r['expect'] == 'mispaired' and replay(r) is not None}
        missing = VICTIM_TASK_IDS - detected
        assert not missing, (
            f'victim task ids no longer detected: {sorted(missing)}. '
            'A specimen was dropped or the predicate narrowed.'
        )

    def test_every_near_miss_class_has_a_negative_control(self) -> None:
        """Precision is pinned per FAILURE MODE, not by an aggregate count."""
        present = {r['near_miss_class'] for r in load_corpus() if r['expect'] == 'clean'}
        assert present >= NEAR_MISS_CLASSES, (
            f'near-miss classes with no negative control: '
            f'{sorted(NEAR_MISS_CLASSES - present)}'
        )

    def test_every_negative_control_is_actually_clean(self) -> None:
        """Stated separately from the bulk agreement test so a precision
        regression names itself as one rather than as an anonymous diff row."""
        fired = [
            f"{r['task_id']}[{r['index']}] ({r['near_miss_class']})"
            for r in load_corpus() if r['expect'] == 'clean' and replay(r) is not None
        ]
        assert not fired, f'negative controls the predicate now falsely fires on: {fired}'

    def test_replay_is_deterministic(self) -> None:
        """Two passes give identical results — no ordering or state dependence."""
        first = [replay(r) for r in load_corpus()]
        second = [replay(r) for r in load_corpus()]
        assert first == second

    def test_positives_and_negatives_are_both_represented(self) -> None:
        """A corpus of only positives is a recall pin; only negatives, a precision
        pin. This one is both, and must stay both."""
        expectations = {r['expect'] for r in load_corpus()}
        assert expectations == {'mispaired', 'clean'}
