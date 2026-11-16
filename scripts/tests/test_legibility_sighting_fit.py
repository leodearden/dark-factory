"""Tests for scripts/legibility/sighting_fit.py (task 6787): which of an
entry's sightings count toward the singleton filing gate, and the census fit
check that decides the ones a judgment attached."""
from __future__ import annotations

import pytest
from legibility import codebook, sighting_fit
from legibility.sighting_fit import Standing
from legibility_sighting_fit_fixtures import (
    ENTRY_ID,
    FOUNDING_SESSION,
    MISATTACHED_SESSIONS,
    reify_codebook_at_rest,
    reify_codebook_with_quotes,
)

_RUN_ID = "census-reify-20261116"


def _entry(cb: dict) -> dict:
    return codebook.record_by_id(cb["entries"], ENTRY_ID, kind="entry")


def _sighting(session: str, **fields) -> dict:
    return {
        "date": "2026-11-10",
        "project": "reify",
        "session": session,
        "origin_phase": "unknown",
        "manifested_phase": "unknown",
        **fields,
    }


def _with_sighting(cb: dict, sighting: dict) -> dict:
    _entry(cb)["sightings"].append(sighting)
    return cb


def _with_fit_row(cb: dict, session: str, *, exhibits: bool) -> dict:
    _entry(cb).setdefault(codebook.SIGHTING_FITS_KEY, []).append(
        {"session": session, "exhibits": exhibits, "run_id": _RUN_ID, "reason": "judged"}
    )
    return cb


# ---------------------------------------------------------------------------
# step-3: the counting rule
# ---------------------------------------------------------------------------

def test_the_reify_entry_counts_only_its_founding_session():
    cb = reify_codebook_at_rest()

    standings = sighting_fit.standings(cb, _entry(cb))

    assert sighting_fit.counted_sessions(cb, _entry(cb)) == frozenset({FOUNDING_SESSION})
    assert standings[FOUNDING_SESSION] is Standing.FOUNDING
    assert {standings[session] for session in MISATTACHED_SESSIONS} == {Standing.UNEVIDENCED}


def test_quoted_misattached_sightings_stand_unchecked_and_do_not_count():
    cb = reify_codebook_with_quotes()

    standings = sighting_fit.standings(cb, _entry(cb))

    assert {standings[session] for session in MISATTACHED_SESSIONS} == {Standing.UNCHECKED}
    assert sighting_fit.counted_sessions(cb, _entry(cb)) == frozenset({FOUNDING_SESSION})


@pytest.mark.parametrize("method", ["title", "key"])
def test_a_rule_attached_sighting_counts(method):
    cb = _with_sighting(
        reify_codebook_at_rest(),
        _sighting("s-rule", attached_by={"method": method, "run_id": _RUN_ID}),
    )

    assert sighting_fit.standings(cb, _entry(cb))["s-rule"] is Standing.ATTACHED_BY_RULE
    assert "s-rule" in sighting_fit.counted_sessions(cb, _entry(cb))


def test_a_similarity_attached_sighting_waits_for_a_fit_verdict():
    cb = _with_sighting(
        reify_codebook_at_rest(),
        _sighting(
            "s-similar",
            evidence_quote="ls docs/guess 2>/dev/null -> [exit 2]",
            attached_by={"method": "similarity", "score": 0.91, "run_id": _RUN_ID},
        ),
    )

    assert sighting_fit.standings(cb, _entry(cb))["s-similar"] is Standing.UNCHECKED
    assert "s-similar" not in sighting_fit.counted_sessions(cb, _entry(cb))


def test_a_correction_audit_sighting_does_not_count():
    cb = _with_sighting(
        reify_codebook_at_rest(),
        _sighting("s-correction", note="Reframed: the cause is the guessed path, not the redirect."),
    )

    assert sighting_fit.standings(cb, _entry(cb))["s-correction"] is Standing.UNEVIDENCED
    assert "s-correction" not in sighting_fit.counted_sessions(cb, _entry(cb))


@pytest.mark.parametrize(
    ("exhibits", "standing", "counted"),
    [(True, Standing.CONFIRMED, True), (False, Standing.REFUTED, False)],
    ids=["confirmed", "refuted"],
)
def test_a_fit_row_decides_its_session(exhibits, standing, counted):
    session = MISATTACHED_SESSIONS[0]
    cb = _with_fit_row(reify_codebook_with_quotes(), session, exhibits=exhibits)

    assert sighting_fit.standings(cb, _entry(cb))[session] is standing
    assert (session in sighting_fit.counted_sessions(cb, _entry(cb))) is counted


def test_an_entry_without_a_founding_candidate_counts_its_first_sighting():
    cb = reify_codebook_at_rest()
    cb["candidates"] = []

    assert sighting_fit.founding_sessions(cb, _entry(cb)) == frozenset({FOUNDING_SESSION})
    assert sighting_fit.counted_sessions(cb, _entry(cb)) == frozenset({FOUNDING_SESSION})


def test_a_founding_session_stays_founding_whatever_a_row_says():
    cb = _with_fit_row(reify_codebook_at_rest(), FOUNDING_SESSION, exhibits=False)

    assert sighting_fit.standings(cb, _entry(cb))[FOUNDING_SESSION] is Standing.FOUNDING
    assert FOUNDING_SESSION in sighting_fit.counted_sessions(cb, _entry(cb))
