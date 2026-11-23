"""Tests for scripts/legibility/sighting_fit.py (task 6787): which of an
entry's sightings count toward the singleton filing gate, and the census fit
check that decides the ones a judgment attached."""
from __future__ import annotations

import copy
import functools
import json

import pytest
from legibility import census_filing, codebook, filing_policy, session_runner, sighting_fit
from legibility.sighting_fit import FitReplyError, FitVerdict, OfferedSighting, Standing
from legibility_sighting_fit_fixtures import (
    CAUSE,
    ENTRY_ID,
    FOUNDING_EVIDENCE,
    FOUNDING_SESSION,
    MISATTACHED_EVIDENCE,
    MISATTACHED_NOTES,
    MISATTACHED_SESSIONS,
    TITLE,
    judging,
    recording_fit_fn,
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


# ---------------------------------------------------------------------------
# step-5: selecting the entries whose release a fit check decides, and
# running the checks
# ---------------------------------------------------------------------------

def _at_stake(cb: dict, releases=None) -> tuple[sighting_fit.FitCheck, ...]:
    """at_stake over the reify entry, gated as the census gates it unless
    *releases* says otherwise."""
    gate = functools.partial(census_filing.entry_releases, cb) if releases is None else releases
    return sighting_fit.at_stake(cb, [_entry(cb)], releases=gate)


def test_at_stake_offers_the_quoted_sightings_of_a_withheld_entry_one_short_of_the_gate():
    cb = reify_codebook_with_quotes()

    [check] = _at_stake(cb)

    assert check.entry_id == ENTRY_ID
    assert check.title == TITLE
    assert check.cause == CAUSE
    assert check.founding_evidence == (FOUNDING_EVIDENCE,)
    assert check.offered == tuple(
        OfferedSighting(session=session, evidence_quote=MISATTACHED_EVIDENCE[session])
        for session in MISATTACHED_SESSIONS
    )
    assert [note for note in MISATTACHED_NOTES.values() if note in repr(check)] == []


def test_at_stake_offers_nothing_when_no_sighting_quotes_evidence():
    cb = reify_codebook_at_rest()

    assert _at_stake(cb) == ()


def test_at_stake_offers_nothing_for_an_entry_already_at_the_gate():
    cb = _with_sighting(
        reify_codebook_with_quotes(),
        _sighting("s-rule", attached_by={"method": "title", "run_id": _RUN_ID}),
    )

    assert _at_stake(cb) == ()


def test_at_stake_offers_nothing_when_even_every_unchecked_sighting_stays_short(monkeypatch):
    monkeypatch.setattr(filing_policy, "MIN_UNREMEDIATED_SIGHTINGS", 5)
    cb = reify_codebook_with_quotes()

    assert _at_stake(cb) == ()


@pytest.mark.parametrize(
    "releases",
    [lambda entry, count: True, lambda entry, count: False],
    ids=["gate-already-releases", "gate-never-releases"],
)
def test_at_stake_offers_nothing_whose_release_the_gate_does_not_turn_on_the_unchecked(releases):
    assert _at_stake(reify_codebook_with_quotes(), releases) == ()


def test_at_stake_asks_the_gate_at_the_counted_and_at_the_reachable_count():
    asked = []

    def releases(entry, count):
        asked.append((entry["id"], count))
        return count >= 3

    [check] = _at_stake(reify_codebook_with_quotes(), releases)

    assert check.entry_id == ENTRY_ID
    assert asked == [(ENTRY_ID, 1), (ENTRY_ID, 1 + len(MISATTACHED_SESSIONS))]


def _check(cb, fit_fn):
    return sighting_fit.check(cb, _at_stake(cb), fit_fn=fit_fn, model="m", run_id=_RUN_ID)


def test_check_stamps_a_refuting_row_per_offered_session_on_a_new_codebook():
    cb = reify_codebook_with_quotes()
    before = copy.deepcopy(cb)
    fit_fn = recording_fit_fn(judging(False))

    stamped, [outcome] = _check(cb, fit_fn)

    assert cb == before
    assert [call["model"] for call in fit_fn.calls] == ["m"]
    rows = _entry(stamped)[codebook.SIGHTING_FITS_KEY]
    assert [(row["session"], row["exhibits"], row["run_id"]) for row in rows] == [
        (session, False, _RUN_ID) for session in MISATTACHED_SESSIONS
    ]
    assert all(row["reason"] == "judged from the quote" for row in rows)
    assert codebook.validate(stamped) == []
    assert sighting_fit.counted_sessions(stamped, _entry(stamped)) == frozenset({FOUNDING_SESSION})
    assert outcome.refuted == MISATTACHED_SESSIONS
    assert outcome.confirmed == ()
    assert outcome.unchecked == ()
    assert outcome.error is None


def test_check_counts_a_confirmed_session():
    stamped, [outcome] = _check(reify_codebook_with_quotes(), recording_fit_fn(judging(True, False)))

    assert sighting_fit.counted_sessions(stamped, _entry(stamped)) == frozenset(
        {FOUNDING_SESSION, MISATTACHED_SESSIONS[0]}
    )
    assert outcome.confirmed == (MISATTACHED_SESSIONS[0],)


def test_check_ignores_a_verdict_on_an_unoffered_session_and_leaves_an_omitted_one_unchecked():
    def decide(check):
        return [
            FitVerdict(session="s-never-offered", exhibits=True, reason="?"),
            FitVerdict(session=MISATTACHED_SESSIONS[0], exhibits=False, reason="other probe"),
            FitVerdict(session=MISATTACHED_SESSIONS[1], exhibits=False, reason="other probe"),
        ]

    stamped, [outcome] = _check(reify_codebook_with_quotes(), recording_fit_fn(decide))

    rows = _entry(stamped)[codebook.SIGHTING_FITS_KEY]
    assert [row["session"] for row in rows] == list(MISATTACHED_SESSIONS[:2])
    assert outcome.unchecked == (MISATTACHED_SESSIONS[2],)
    assert codebook.validate(stamped) == []


def test_check_keeps_the_first_verdict_a_reply_gives_a_session():
    session = MISATTACHED_SESSIONS[0]

    def decide(check):
        return [
            FitVerdict(session=session, exhibits=False, reason="first"),
            FitVerdict(session=session, exhibits=True, reason="second"),
        ]

    stamped, _ = _check(reify_codebook_with_quotes(), recording_fit_fn(decide))

    [row] = _entry(stamped)[codebook.SIGHTING_FITS_KEY]
    assert (row["exhibits"], row["reason"]) == (False, "first")


def test_check_leaves_every_offered_session_unchecked_when_the_judge_is_unreachable():
    def decide(check):
        raise session_runner.InvocationFailed("judge unavailable")

    cb = reify_codebook_with_quotes()
    stamped, [outcome] = _check(cb, recording_fit_fn(decide))

    assert codebook.SIGHTING_FITS_KEY not in _entry(stamped)
    assert outcome.unchecked == MISATTACHED_SESSIONS
    assert outcome.confirmed == outcome.refuted == ()
    assert outcome.error is not None
    assert "judge unavailable" in outcome.error


def test_check_lets_a_defect_in_the_fit_fn_propagate():
    def decide(check):
        raise TypeError("a fit_fn defect")

    with pytest.raises(TypeError, match="a fit_fn defect"):
        _check(reify_codebook_with_quotes(), recording_fit_fn(decide))


def test_a_fit_outcome_record_is_json_safe():
    _, [outcome] = _check(reify_codebook_with_quotes(), recording_fit_fn(judging(True, False)))

    record = json.loads(json.dumps(outcome.to_record()))

    assert record == {
        "entry_id": ENTRY_ID,
        "title": TITLE,
        "confirmed": [MISATTACHED_SESSIONS[0]],
        "refuted": list(MISATTACHED_SESSIONS[1:]),
        "unchecked": [],
        "error": None,
    }


# ---------------------------------------------------------------------------
# step-7: the production fit_fn seam
# ---------------------------------------------------------------------------

def _fake_invoke(reply):
    calls = []

    def invoke(prompt, model):
        calls.append((prompt, model))
        return reply

    invoke.calls = calls
    return invoke


def _reify_check():
    [check] = _at_stake(reify_codebook_with_quotes())
    return check


def _encoded(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


def test_the_fit_prompt_carries_the_entry_and_the_verbatim_evidence_but_no_note():
    invoke = _fake_invoke(json.dumps({"sightings": []}))

    sighting_fit.build_fit_fn(invoke)(_reify_check(), model="m")

    [(prompt, model)] = invoke.calls
    assert model == "m"
    for text in (TITLE, CAUSE, FOUNDING_EVIDENCE, *MISATTACHED_SESSIONS, *MISATTACHED_EVIDENCE.values()):
        assert _encoded(text) in prompt, text
    for note in MISATTACHED_NOTES.values():
        assert note not in prompt
        assert _encoded(note) not in prompt


_REPLY = {
    "sightings": [
        {"session": MISATTACHED_SESSIONS[0], "exhibits": False, "reason": "stderr kept by 2>&1"},
        {"session": MISATTACHED_SESSIONS[1], "exhibits": True, "reason": "same guessed probe"},
    ]
}


@pytest.mark.parametrize(
    "raw",
    [json.dumps(_REPLY), "```json\n" + json.dumps(_REPLY) + "\n```"],
    ids=["bare", "fenced"],
)
def test_a_fit_reply_parses_into_verdicts(raw):
    verdicts = sighting_fit.build_fit_fn(_fake_invoke(raw))(_reify_check(), model="m")

    assert tuple(verdicts) == (
        FitVerdict(session=MISATTACHED_SESSIONS[0], exhibits=False, reason="stderr kept by 2>&1"),
        FitVerdict(session=MISATTACHED_SESSIONS[1], exhibits=True, reason="same guessed probe"),
    )


@pytest.mark.parametrize(
    "item",
    [
        {"session": MISATTACHED_SESSIONS[0], "exhibits": "yes", "reason": "?"},
        {"session": MISATTACHED_SESSIONS[0], "exhibits": 1, "reason": "?"},
        {"session": 7, "exhibits": True, "reason": "?"},
    ],
    ids=["string-exhibits", "int-exhibits", "int-session"],
)
def test_a_malformed_fit_reply_item_is_dropped(item):
    reply = {"sightings": [item, _REPLY["sightings"][1]]}

    verdicts = sighting_fit.build_fit_fn(_fake_invoke(json.dumps(reply)))(_reify_check(), model="m")

    assert [verdict.session for verdict in verdicts] == [MISATTACHED_SESSIONS[1]]


@pytest.mark.parametrize(
    "raw",
    ["no judgment here", json.dumps({"verdicts": []}), json.dumps({"sightings": {"a": 1}})],
    ids=["unparseable", "no-sightings", "sightings-not-a-list"],
)
def test_a_fit_reply_that_is_no_judgment_raises(raw):
    with pytest.raises(FitReplyError):
        sighting_fit.build_fit_fn(_fake_invoke(raw))(_reify_check(), model="m")


def test_check_leaves_an_entry_unchecked_when_the_fit_reply_is_no_judgment():
    cb = reify_codebook_with_quotes()

    _, [outcome] = _check(cb, sighting_fit.build_fit_fn(_fake_invoke("no judgment here")))

    assert outcome.unchecked == MISATTACHED_SESSIONS
    assert outcome.error is not None
    assert "FitReplyError" in outcome.error


# ---------------------------------------------------------------------------
# step-9: the census release loop counts only what the rule counts
# ---------------------------------------------------------------------------

def test_the_release_loop_holds_the_reify_entry_back():
    assert census_filing.filing_queue([], reify_codebook_at_rest()) == []


def test_the_release_loop_releases_an_entry_a_fit_verdict_carries_over_the_gate():
    cb = _with_fit_row(reify_codebook_with_quotes(), MISATTACHED_SESSIONS[0], exhibits=True)

    [pending] = census_filing.filing_queue([], cb)

    assert pending.reoffered_entry_id == ENTRY_ID


def _bare_entry(entry_id: str, **fields) -> dict:
    return {
        "id": entry_id,
        "title": f"Confusion {entry_id}",
        "severity": "low",
        "status": "open",
        "origin_phase": "unknown",
        "manifested_phase": "unknown",
        "sightings": [_sighting(f"s-{entry_id}")],
        **fields,
    }


def test_withheld_awaiting_count_is_the_withheld_open_entries_no_fileable_cluster_releases():
    cb = reify_codebook_at_rest()
    withheld = {codebook.ENTRY_FILING_KEY: codebook.ENTRY_FILING_WITHHELD}
    cb["entries"] += [
        _bare_entry("entry-partially", status="partially", **withheld),
        _bare_entry("entry-named", **withheld),
        _bare_entry("entry-retired", status="retired", **withheld),
        _bare_entry("entry-unmarked"),
    ]
    fileable = [({"title": "Confusion entry-named"}, "entry-named")]

    awaiting = census_filing.withheld_awaiting_count(fileable, cb)

    assert [entry["id"] for entry in awaiting] == [ENTRY_ID, "entry-partially"]


def test_an_entry_cluster_quotes_evidence_only_from_counted_sessions():
    cb = reify_codebook_with_quotes()

    cluster = census_filing.entry_cluster(cb, _entry(cb))

    assert cluster["evidence"] == [FOUNDING_EVIDENCE]
    assert len(cluster["sightings"]) == 1 + len(MISATTACHED_SESSIONS)


def test_an_entry_cluster_quotes_a_confirmed_sighting_but_never_a_refuted_one():
    confirmed, refuted, _unchecked = MISATTACHED_SESSIONS
    cb = _with_fit_row(reify_codebook_with_quotes(), confirmed, exhibits=True)
    cb = _with_fit_row(cb, refuted, exhibits=False)

    cluster = census_filing.entry_cluster(cb, _entry(cb))

    assert cluster["evidence"] == [FOUNDING_EVIDENCE, MISATTACHED_EVIDENCE[confirmed]]
