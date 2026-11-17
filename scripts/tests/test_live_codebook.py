"""Live-file guards over the committed docs/legibility/confusion-codebook.yaml
and docs/legibility/coding-records/*.jsonl: the two committed data artifacts
must still agree with each other and with the codebook's sole writer.

New codebook-correction guards belong here. The census landing gate
(docs/legibility/legibility.yaml census.landing_gate) runs this file. The
merger's own unit tests stay in test_codebook.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from codebook_merge_report import clean_merge_report, merge_one, merge_report
from legibility import codebook as mod
from legibility import coder, disposition, verdict
from legibility_task_store_fake import FakeTaskStore
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)


# ---------------------------------------------------------------------------
# step-17: RED (live-file guard, PRD §11 γ observable) — committed codebook
# ---------------------------------------------------------------------------

# scripts/tests/test_live_codebook.py -> scripts/tests/ -> scripts/ -> repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_LIVE_CODEBOOK_PATH = _REPO_ROOT / "docs" / "legibility" / "confusion-codebook.yaml"


def test_live_codebook_is_v2_and_validates_green():
    """Pins that the committed registry conforms post-migration and guards
    future hand-edits: version == 2 and validate() returns zero errors.
    RED until the in-place v1->v2 migration lands (file is still v1)."""
    codebook = mod.load(_LIVE_CODEBOOK_PATH)
    assert codebook["version"] == 2
    assert mod.validate(codebook) == []


def test_live_codebook_has_no_pending_twin_of_an_adjudicated_candidate():
    """Live-file guard against the resurrection that already happened.

    (a) is the resurrection signature: the merger fabricated a *pending* twin
    for a title the census had already adjudicated (rejected cand-20260722-28
    came back as pending cand-20260724-2). (b) is the forward guard — already
    green today — asserted so the invariant is stated whole.

    This is the checkable invariant the merge fix guarantees going forward: a
    title with an existing pending record takes the `pending_match` branch; a
    title whose records are all adjudicated takes the conflict branch (sighting
    appended, disposition untouched, NO new pending); only a genuinely unseen
    title reaches the create branch. census.py only flips dispositions and
    never creates candidates, so it can only violate (a) by rejecting one of
    two same-title pendings — which (b) proves cannot currently arise.

    Deliberately asserted on the pending-vs-adjudicated split rather than on
    duplicate titles outright: the never-delete contract forbids removing the
    resurrected record, so the repair can only restore its verdict, not erase
    it, and a "no duplicate titles" assertion could never go green.
    """
    codebook = mod.load(_LIVE_CODEBOOK_PATH)

    by_title = {}
    for candidate in codebook.get("candidates") or []:
        by_title.setdefault(candidate.get("title"), []).append(candidate)

    def _describe(title):
        return "{!r}: {}".format(
            title,
            [
                (c.get("id"), c.get("disposition"), c.get("first_seen"))
                for c in by_title[title]
            ],
        )

    # (a) THE RESURRECTION SIGNATURE — a pending record alongside an
    #     already-adjudicated one for the byte-identical title.
    resurrected = [
        title
        for title, group in by_title.items()
        if any(c.get("disposition") == "pending" for c in group)
        and any(c.get("disposition") in {"rejected", "promoted"} for c in group)
    ]
    assert resurrected == [], (
        "pending candidate(s) coexist with an already-adjudicated candidate of "
        "the same title — the census verdict was resurrected: "
        + "; ".join(_describe(t) for t in resurrected)
    )

    # (b) forward guard — never two pending records for one title.
    duplicated = [
        title
        for title, group in by_title.items()
        if sum(1 for c in group if c.get("disposition") == "pending") > 1
    ]
    assert duplicated == [], (
        "more than one pending candidate shares a title: "
        + "; ".join(_describe(t) for t in duplicated)
    )


# ---------------------------------------------------------------------------
# Shared machinery for the live-file guards below (tasks 4892, 5198, 5687,
# 5729).
# Each asserts that a committed §7.3 coding record and the merger-owned
# registry still agree; the comparison itself is single-sourced here, so each
# guard states only what is distinctive about its own correction and a fix to
# the comparison lands once rather than once per guard.
# ---------------------------------------------------------------------------

# What apply_coding_records copies verbatim from a match/candidate/correction
# payload into the sighting it appends. The optional three are emitted only
# when truthy, so a payload that omits one is compared as omitting it.
_SIGHTING_PAYLOAD_FIELDS = (
    "origin_phase",
    "manifested_phase",
    "invariant_violated",
    "note",
    "evidence_quote",
)

# What apply_coding_records copies from the record's header into every
# sighting it appends (the session is how a sighting is found, so it is the
# lookup key rather than a compared field).
_SIGHTING_HEADER_FIELDS = ("date", "project")


def _sightings_for(holder: dict, session: str) -> list[dict]:
    return [s for s in holder.get("sightings") or [] if s.get("session") == session]


def _session_record(record_path: Path, session: str) -> dict:
    """The one §7.3 coding record in JSONL *record_path* that carries
    *session*, asserted valid."""
    records = [
        json.loads(line)
        for line in record_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    mine = [r for r in records if r.get("session") == session]
    assert len(mine) == 1, (
        f"expected exactly one {session!r} record in {record_path}, found "
        f"{len(mine)}; sessions present: "
        f"{sorted({r.get('session') for r in records})}"
    )
    assert mod.validate_coding_record(mine[0]) == [], (
        f"invalid §7.3 coding record for session {session!r}"
    )
    return mine[0]


def _live_entry(codebook: dict, entry_id: str) -> dict:
    entries = [e for e in codebook.get("entries") or [] if e.get("id") == entry_id]
    assert len(entries) == 1, (
        f"expected exactly one {entry_id!r} entry, found {len(entries)}"
    )
    return entries[0]


def _assert_sighting_carries(record: dict, payload: dict, holder: dict) -> None:
    """Assert *holder* carries exactly one sighting for *record*'s session,
    and that it is what the sole writer appends for *payload*: the record's
    header fields and the payload's fields, verbatim. This is the
    in-place-rewrite detector for a committed sighting."""
    session = record["session"]
    carried = _sightings_for(holder, session)
    assert len(carried) == 1, (
        f"expected exactly one {session!r} sighting on {holder.get('id')!r}, "
        f"found {len(carried)}"
    )
    expected = {field: record[field] for field in _SIGHTING_HEADER_FIELDS} | {
        field: payload[field] for field in _SIGHTING_PAYLOAD_FIELDS if payload.get(field)
    }
    drift = {
        field: (want, carried[0].get(field))
        for field, want in expected.items()
        if carried[0].get(field) != want
    }
    assert drift == {}, (
        f"the {session!r} sighting on {holder.get('id')!r} no longer matches the "
        "committed record — a committed sighting was rewritten in place "
        f"{{field: (record, codebook)}}: {drift}"
    )


def _assert_correction_absorbed(
    record_path: Path,
    session: str,
    entry_id: str,
    *,
    refuted_tokens: tuple[str, ...],
    expected_status: str,
    original_sightings: dict[str, str],
) -> tuple[dict, dict, dict]:
    """Assert the *session* correction committed in *record_path* is fully
    absorbed by the committed registry, and hand back its
    ``(record, correction, entry)`` triple for whatever else a caller pins.

    A correction REWRITES merger-owned fields in place — the one op in the
    §7.3 vocabulary that is not a pure append — so every guard over one needs
    the same six properties, each invariant under every transition the system
    is designed to make to these two files:

    (0) WELL-FORMED — *record_path* is JSONL, exactly one of its records
        carries *session*, it validates, and it holds exactly one
        `corrections` op naming *entry_id*. Both counts are scoped BY
        SESSION, not file-wide: the sanctioned way to re-correct an entry
        later is a NEW session in the same file (task 4892's record holds
        several ops across two), and a file-wide count would break the moment
        anyone appends. Exactly-one matters because the merger admits one
        sighting per (session, entry), so a second op for this entry in this
        session would be counted in `correction_skipped` with its field
        writes lost.
    (1) AGREEMENT — the entry's title/cause/status equal the record's, and one
        sighting carries the record field for field. This is the
        in-place-rewrite detector for the merger-owned YAML, and it is what
        proves the correction's full text is carried verbatim — which is why
        no caller needs to pin cause or note prose.
    (2) WITHDRAWN — no *refuted_tokens* member survives in the title, the one
        field `scripts/legibility/coder.py::build_codebook_index` renders to
        the nightly coder unconditionally.
    (3) STATUS — the entry is *expected_status*, pinned so a later sweep can
        neither retire a live defect nor reopen a settled one while leaving
        the guard green.
    (4) NEVER-DELETE and ORDER — every session in *original_sightings*
        survives, still carrying its mapped date, and the correction's own
        sighting follows all of them.
    (5) NO-OP — re-applying the record changes nothing, which makes a rebase
        onto a nightly-rewritten main resolvable by re-running one CLI command
        instead of hand-editing 22k lines of generated YAML.

    An entry's TOTAL sighting count is deliberately never pinned: the nightly
    census may legitimately append more, to retired entries too (they stay in
    the coder's index). Presence-of-each-original plus ORDER carries
    never-delete without that coupling. Nor is any substring of a note's prose
    pinned — (1) already proves it is carried verbatim — and schema shape
    stays owned by `test_live_codebook_is_v2_and_validates_green`.
    """
    # (0) WELL-FORMED.
    record = _session_record(record_path, session)
    corrections = record.get("corrections") or []
    assert len(corrections) == 1, (
        f"expected exactly one correction in the {session!r} record, got "
        f"{len(corrections)} — the merger admits one sighting per (session, "
        "entry), so a second op for this entry in this session would be "
        "dropped into `correction_skipped` with its field writes lost"
    )
    correction = corrections[0]
    assert correction.get("entry_id") == entry_id

    codebook = mod.load(_LIVE_CODEBOOK_PATH)
    entry = _live_entry(codebook, entry_id)

    # (1) AGREEMENT — entry fields, then the provenance sighting.
    field_drift = {
        field: (correction[field], entry.get(field))
        for field in ("title", "cause", "status")
        if correction.get(field) and correction[field] != entry.get(field)
    }
    assert field_drift == {}, (
        f"{entry_id!r} no longer carries the committed correction — a "
        f"merger-owned field was rewritten by hand {{field: (record, codebook)}}: "
        f"{field_drift}"
    )
    _assert_sighting_carries(record, correction, entry)

    # (2) WITHDRAWN — the refuted framing is gone from the one field
    #     build_codebook_index renders unconditionally.
    title_lower = entry["title"].lower()
    still_asserted = [t for t in refuted_tokens if t.lower() in title_lower]
    assert still_asserted == [], (
        f"{entry_id!r} still asserts the refuted framing via {still_asserted} "
        f"— build_codebook_index renders this title to the coder every night: "
        f"{entry['title']!r}"
    )

    # (3) STATUS.
    assert entry.get("status") == expected_status, (
        f"{entry_id!r} is {entry.get('status')!r}, not {expected_status!r}"
    )

    # (4) NEVER-DELETE and ORDER.
    sightings = entry.get("sightings") or []
    order = {s.get("session"): i for i, s in enumerate(sightings)}
    for original_session, date in original_sightings.items():
        assert original_session in order, (
            f"the {original_session!r} sighting was deleted from {entry_id!r} "
            "— sightings are immutable dated observations, and a correction "
            "annotates them rather than retracting them"
        )
        original = sightings[order[original_session]]
        assert original.get("date") == date, (
            f"the {original_session!r} sighting was re-dated to "
            f"{original.get('date')!r} (was {date!r}) — a dated observation is "
            "not editable"
        )
    assert order[session] > max(order[s] for s in original_sightings), (
        f"the {session!r} sighting does not follow the sightings it annotates"
    )

    # (5) NO-OP — the committed YAML is exactly what the sole writer produces.
    merge = merge_one(codebook, record)
    assert merge_report(merge) == clean_merge_report(), (
        f"re-applying session {session!r} changed something — the committed "
        f"codebook has not fully absorbed the record: {merge.summary()}"
    )

    return record, correction, entry


# ---------------------------------------------------------------------------
# task 4892 (live-file guard) — both corrections landed through the sole
# writer, and the committed registry still carries the committed record.
# ---------------------------------------------------------------------------

_T4892_ENTRY_ID = "guards-assert-unverified-diagnoses"
_T4892_ORIGINAL_SESSION = "e16af1c5-de98-4252-8504-ce5d13bee3d6"
_T4892_CORRECTION_SESSION = "task-4892-resume-sysprompt-correction"
_T4892_CANDIDATE_SESSION = "task-4892-lettered-option-collision"
_T4892_RECORD_PATH = (
    _REPO_ROOT / "docs" / "legibility" / "coding-records" / "task-4892-corrections.jsonl"
)


def _t4892_carriers(codebook: dict, record: dict):
    """Each payload of *record* with (its key, the live records the sole writer
    appends it to): a match goes to the entry named by its `entry_id`, a
    candidate to the candidate(s) sharing its `title`."""
    for match in record.get("matches") or []:
        key = match.get("entry_id")
        yield key, match, [e for e in codebook.get("entries") or [] if e.get("id") == key]
    for candidate in record.get("candidates") or []:
        key = candidate.get("title")
        yield key, candidate, [
            c for c in codebook.get("candidates") or [] if c.get("title") == key
        ]


def test_live_codebook_carries_the_task_4892_corrections():
    """The committed §7.3 record and the committed registry still agree.

    Both corrections went through the sole writer, which can only APPEND: the
    now-false present-tense claim about `--system-prompt-file` is superseded by
    a sibling sighting rather than rewritten in place, and the
    lettered-option-collision pattern is filed as a candidate for the census to
    adjudicate. Four properties, each invariant under every transition the
    system is designed to make to this file:

    (1) AGREEMENT — every payload in the record is carried, exactly once, by
        exactly one live entry/candidate, field for field. A hand-edit of the
        YAML (the in-place rewrite the append-only contract exists to prevent)
        surfaces here as a mismatch. Two committed data artifacts compared
        against each other — not the prose pin esc-4892-4 removed, which
        asserted a substring of a note this record does not own.
    (2) NEVER-DELETE — the 2026-08-10 observation is still present, still so
        dated. It was true when made; only its present tense expired.
    (3) ORDER — the correction follows what it supersedes: its note points the
        reader at "the wording above".
    (4) NO-OP — re-applying the record appends nothing, which is what makes a
        rebase onto a nightly-rewritten main resolvable by re-running one CLI
        command instead of hand-editing 22k lines of generated YAML.

    Schema shape is not re-checked here: `test_live_codebook_is_v2_and_validates_green`
    owns `validate() == []` over this file. The candidate's `disposition` and
    its `cand-<yyyymmdd>-<n>` id are not pinned either — the census owns the
    first and the merger derives the second from the same-date candidate count,
    so both may legitimately change; each is located by its stable sighting
    session instead.
    """
    records = [
        json.loads(line)
        for line in _T4892_RECORD_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sessions = {r.get("session") for r in records}
    assert {_T4892_CORRECTION_SESSION, _T4892_CANDIDATE_SESSION} <= sessions, (
        f"{_T4892_RECORD_PATH} no longer records both task-4892 corrections: "
        f"{sorted(sessions)}"
    )

    codebook = mod.load(_LIVE_CODEBOOK_PATH)

    # (1) AGREEMENT — the record is carried verbatim, once, by one holder.
    for record in records:
        session = record.get("session")
        assert mod.validate_coding_record(record) == [], (
            f"invalid §7.3 coding record for session {session!r}"
        )
        for key, payload, holders in _t4892_carriers(codebook, record):
            carrying = [h for h in holders if _sightings_for(h, session)]
            assert len(carrying) == 1, (
                f"expected exactly one live record under {key!r} to carry a "
                f"{session!r} sighting, found {[h.get('id') for h in carrying]}"
            )
            carried = _sightings_for(carrying[0], session)
            assert len(carried) == 1, (
                "the sole writer must not duplicate a sighting on a re-run — "
                f"{session!r} appears {len(carried)}x on {carrying[0].get('id')!r}"
            )
            drift = {
                field: (payload[field], carried[0].get(field))
                for field in _SIGHTING_PAYLOAD_FIELDS
                if payload.get(field) and payload[field] != carried[0].get(field)
            }
            assert drift == {}, (
                f"the {session!r} sighting on {carrying[0].get('id')!r} no longer "
                "matches the committed record — a committed sighting was "
                f"rewritten in place {{field: (record, codebook)}}: {drift}"
            )

    # (2) NEVER-DELETE and (3) ORDER, on the annotated entry.
    entries = [e for e in codebook.get("entries") or [] if e.get("id") == _T4892_ENTRY_ID]
    assert len(entries) == 1, f"expected exactly one {_T4892_ENTRY_ID!r} entry"
    sightings = entries[0].get("sightings") or []
    order = {s.get("session"): i for i, s in enumerate(sightings)}

    assert _T4892_ORIGINAL_SESSION in order, (
        f"the 2026-08-10 sighting was deleted from {_T4892_ENTRY_ID!r} — "
        "sightings are immutable dated observations"
    )
    original = sightings[order[_T4892_ORIGINAL_SESSION]]
    assert original.get("date") == "2026-08-10", (
        f"the {_T4892_ORIGINAL_SESSION!r} sighting was re-dated to "
        f"{original.get('date')!r} — a dated observation is not editable"
    )
    assert order.get(_T4892_CORRECTION_SESSION, -1) > order[_T4892_ORIGINAL_SESSION], (
        f"the {_T4892_CORRECTION_SESSION!r} sighting is missing from "
        f"{_T4892_ENTRY_ID!r} or does not follow the 2026-08-10 sighting it "
        "supersedes — its note points the reader at 'the wording above'"
    )

    # (4) NO-OP — the record is already fully absorbed, so re-running the sole
    #     writer over it appends nothing.
    for record in records:
        merge = merge_one(codebook, record)
        codebook = merge.codebook
        assert merge_report(merge) == clean_merge_report(), (
            f"re-applying session {record.get('session')!r} appended something — "
            f"its sightings are not already absorbed by the live codebook: "
            f"{merge.summary()}"
        )


# ---------------------------------------------------------------------------
# task 5198 (live-file guard, esc-5120-2) — the refuted task-5120 framing is
# withdrawn from the committed registry, through the sole writer.
# ---------------------------------------------------------------------------

_T5198_ENTRY_ID = "entry-cand-20260831-16"
_T5198_CORRECTION_SESSION = "task-5198-verify-summary-glob-refutation"
_T5198_RECORD_PATH = (
    _REPO_ROOT / "docs" / "legibility" / "coding-records" / "task-5198-corrections.jsonl"
)

# The three sightings the correction ANNOTATES. They observed a real symptom —
# agents genuinely did hit an opaque JSONDecodeError — and only the cause they
# were filed under was refuted, so each must survive, still carrying its date.
_T5198_ORIGINAL_SIGHTINGS = {
    "273579c5-e299-40c6-ab02-e4c0ab24213b": "2026-08-31",
    "8a483603-92b1-460b-8af1-65c02d70a598": "2026-09-05",
    "d83f509f-a7e5-4eeb-9e2b-74e80289204b": "2026-09-05",
}

# The token pair that made the entry self-ingesting: the census kept matching
# its own quoted error string back to this title.
_T5198_REFUTED_TOKENS = ("glob miss", "JSONDecodeError")


def test_live_codebook_withdraws_the_refuted_task_5120_framing():
    """The committed §7.3 record and the committed registry still agree, and
    the refuted framing is no longer asserted to the coder.

    `scripts/legibility/coder.py::build_codebook_index` renders one line per
    entry — `- {id}: {title} — {cause}` — and deliberately keeps retired
    entries in that index, so retirement alone would leave the refuted title
    in front of the very loop esc-5120-2 diagnoses as self-ingesting. The
    withdrawal therefore had to be a rewritten title AND a 'retired' status,
    which is exactly the pair `_assert_correction_absorbed` pins; nothing
    about this correction is distinctive enough to assert beyond it.
    """
    _assert_correction_absorbed(
        _T5198_RECORD_PATH,
        _T5198_CORRECTION_SESSION,
        _T5198_ENTRY_ID,
        refuted_tokens=_T5198_REFUTED_TOKENS,
        expected_status="retired",
        original_sightings=_T5198_ORIGINAL_SIGHTINGS,
    )


# ---------------------------------------------------------------------------
# task 5687 (live-file guard, confusion census 2026-09-20 §1.2/§5) — the
# refuted em-dash framing is withdrawn from the committed registry, through
# the sole writer.
# ---------------------------------------------------------------------------

_T5687_ENTRY_ID = "entry-cand-20260919-2"
_T5687_CORRECTION_SESSION = "task-5687-embedded-triple-quote-reproduction"
_T5687_RECORD_PATH = (
    _REPO_ROOT / "docs" / "legibility" / "coding-records" / "task-5687-corrections.jsonl"
)

# The founding observation the correction ANNOTATES, and the quote it was
# filed with. The symptom was real — python3 genuinely did report an invalid
# U+2014 — and only the cause it was filed under was refuted, so the sighting
# survives unedited rather than being retracted.
_T5687_ORIGINAL_SIGHTING = {"fbd7b22b-949a-4047-8087-6a719edb1dc2": "2026-09-19"}
_T5687_ORIGINAL_EVIDENCE_QUOTE = "SyntaxError: invalid character '—' (U+2014)"

# The withdrawn framing, as WORD tokens only, all lowercase, compared against
# lowercased text.
#
# `unicode` is deliberately BROADER than the two em-dash spellings and the
# codepoint beside it. python3 reported the symptom as an `invalid character`,
# so the refuted claim respells itself just as naturally as "a unicode
# character in the pasted source breaks the script" — which the three narrow
# tokens would let through unchallenged. The cost is bounded and visible: the
# tuple is scanned against two short surfaces this correction owns (the title,
# and the index line the coder reads), so a future re-framing that genuinely
# needs the word can drop the token in the same commit that needs it.
#
# The bare U+2014 CHARACTER is deliberately NOT in this tuple; it is asserted
# against the TITLE alone, by (b) below. `build_codebook_index` renders each
# entry as `- {id}: {title} — {cause}`, and that separator IS U+2014, emitted
# exactly when the cause is non-empty. Banning the bare character from the
# rendered line would therefore contradict (c)'s demand for a rendered cause —
# no implementation could satisfy both. A title is a leaf value the separator
# cannot contaminate, so the character check belongs there. Do not "helpfully"
# add it back here.
_T5687_REFUTED_TOKENS = ("em-dash", "em dash", "u+2014", "unicode")

# How much of the entry's own collapsed cause must reach the rendered index
# line. Short enough to survive `build_codebook_index`'s truncation under any
# cap it may adopt, so this pins that the cause is carried, never how far.
_T5687_RENDERED_CAUSE_PREFIX_LEN = 80


def test_live_codebook_withdraws_the_refuted_em_dash_framing():
    """`_assert_correction_absorbed` pins the mechanics; this adds the three
    things distinctive to withdrawing a refuted SYMPTOM from a still-open
    entry.

    The entry named a symptom, not a mechanism. Measured against the archived
    transcript, replacing every U+2014 in the failing script with an ASCII
    hyphen leaves it failing — with a different error — while escaping or
    deleting the one embedded `\"\"\"` fixes it with every U+2014 left in
    place. So the reported error string is unstable across recurrences of the
    same defect, and a title keyed to it gives the nightly coder nothing to
    match on.

    (a) STILL OPEN, not retired — passed to the helper as the expected status.
        This correction replaces an explanation; it does not retire a live,
        unfixed defect, and pinning that stops a later sweep quietly retiring
        the entry. (Diverges deliberately from the task-5198 guard above.)
    (b) THE FOUNDING SIGHTING IS ANNOTATED, NOT RETRACTED. The helper pins
        that it survives and keeps its date; its `evidence_quote` is the part
        a later editor is most likely to mistake for the refuted claim and
        "clean up", so the exact quote is pinned here. The symptom it records
        was real — only the cause it was filed under was wrong.
    (c) THE CODER'S VIEW carries the mechanism and none of the symptom. The
        cause is non-empty and reaches the rendered index line, and no refuted
        token survives anywhere in that line. Asserted through the PUBLIC
        renderer over a one-entry codebook, whose output IS the single line
        the coder reads: one scan then covers the title and the truncated
        cause window together, a renderer change surfaces here instead of
        being mirrored in a copy of its format string, and a regression to the
        cause-less `- {id}: {title}` fallback — the very state this task
        exists to end — cannot pass. The cause BODY stays free to name the
        em-dash in order to REFUTE it; only what the coder actually reads is
        policed.
    """
    _, _, entry = _assert_correction_absorbed(
        _T5687_RECORD_PATH,
        _T5687_CORRECTION_SESSION,
        _T5687_ENTRY_ID,
        refuted_tokens=_T5687_REFUTED_TOKENS,
        expected_status="open",  # (a)
        original_sightings=_T5687_ORIGINAL_SIGHTING,
    )
    title = entry["title"]
    assert "—" not in title, (
        f"{_T5687_ENTRY_ID!r} still spells the refuted character in its "
        f"title: {title!r}"
    )

    # (b) THE FOUNDING SIGHTING IS ANNOTATED, NOT RETRACTED.
    for session in _T5687_ORIGINAL_SIGHTING:
        quotes = [s.get("evidence_quote") for s in _sightings_for(entry, session)]
        assert quotes == [_T5687_ORIGINAL_EVIDENCE_QUOTE], (
            f"the {session!r} sighting of {_T5687_ENTRY_ID!r} now carries "
            f"{quotes!r} — the symptom it recorded was real, and is not what "
            "the correction withdraws"
        )

    # (c) THE CODER'S VIEW.
    cause = entry.get("cause")
    assert cause, (
        f"{_T5687_ENTRY_ID!r} has no cause — with none, build_codebook_index "
        "shows the coder the title alone, which is the state this task exists "
        "to end"
    )
    opening = " ".join(str(cause).split())[:_T5687_RENDERED_CAUSE_PREFIX_LEN]
    rendered = coder.build_codebook_index({"entries": [entry]})
    missing = [
        part
        for part, needle in (
            ("id", _T5687_ENTRY_ID),
            ("title", title),
            ("cause", opening),
        )
        if needle not in rendered
    ]
    assert missing == [], (
        f"build_codebook_index left the {missing} of {_T5687_ENTRY_ID!r} out "
        f"of the index line the nightly coder reads: {rendered!r}"
    )
    still_rendered = [t for t in _T5687_REFUTED_TOKENS if t in rendered.lower()]
    assert still_rendered == [], (
        f"the index line the coder actually reads still asserts the refuted "
        f"framing via {still_rendered}: {rendered!r}"
    )


# ---------------------------------------------------------------------------
# task 5729 (live-file guard) — the ef7f6a5c census-verifier sighting misfiled
# on entry-cand-20260919-2 is annotated there and re-routed to
# entry-cand-20260722-18, through the sole writer.
# ---------------------------------------------------------------------------

_T5729_RECORD_PATH = (
    _REPO_ROOT / "docs" / "legibility" / "coding-records" / "task-5729-corrections.jsonl"
)
_T5729_ANNOTATION_SESSION = "task-5729-ef7f6a5c-reroute"
_T5729_MISFILED_SESSION = "ef7f6a5c-3ea0-43b3-b0ee-73fe34d67ccf"
_T5729_MISFILED_DATE = "2026-09-20"
_T5729_HOME_ENTRY_ID = "entry-cand-20260722-18"
_T5729_MISFILED_NOTE = (
    "The periodic-census verifier encountered the em-dash issue itself while "
    "verifying the cluster. Turns 42, 46, 54, 83, 97 show repeated "
    "ansi_c_string encoding rejections when constructing heredocs with "
    "embedded em-dashes. Error neighborhoods confirm SyntaxError with U+2014 "
    "character and multiple failed Bash commands attempting to work with the "
    "problematic em-dash pattern."
)


def test_live_codebook_reroutes_the_misfiled_census_verifier_sighting():
    """The misfiled sighting is annotated where it was filed and re-routed to
    the entry whose mechanism it shows, both through the sole writer.

    (1) ANNOTATION — a note-only correction on the misfiled entry, absorbed as
        `_assert_correction_absorbed` defines it, the entry still open.
    (2) REPLACES NO EXPLANATION — the annotation supplies no title, cause or
        status.
    (3) ANNOTATED, NOT RETRACTED — the misfiled sighting keeps its own note.
    (4) LINK — the annotation names the session it annotates and its new home.
    (5) ROUTE — one match op, under the real session and date, on the home
        entry, naming where it was misfiled; carried once, field for field.
    (6) ROUTE NO-OP — re-applying the route changes nothing.
    """
    _, correction, entry = _assert_correction_absorbed(
        _T5729_RECORD_PATH,
        _T5729_ANNOTATION_SESSION,
        _T5687_ENTRY_ID,
        refuted_tokens=_T5687_REFUTED_TOKENS,
        expected_status="open",
        original_sightings={_T5729_MISFILED_SESSION: _T5729_MISFILED_DATE},
    )

    # (2) REPLACES NO EXPLANATION.
    rewritten = {"title", "cause", "status"} & correction.keys()
    assert rewritten == set(), (
        f"the {_T5729_ANNOTATION_SESSION!r} correction writes {sorted(rewritten)} "
        f"of {_T5687_ENTRY_ID!r} — that entry's framing is owned by task 5687's "
        "record; this one only annotates a sighting"
    )

    # (3) ANNOTATED, NOT RETRACTED.
    notes = [s.get("note") for s in _sightings_for(entry, _T5729_MISFILED_SESSION)]
    assert notes == [_T5729_MISFILED_NOTE], (
        f"the {_T5729_MISFILED_SESSION!r} sighting of {_T5687_ENTRY_ID!r} now "
        f"carries {notes!r} — a dated observation is immutable: it is corrected "
        "by the sibling annotation sighting, never rewritten"
    )

    # (4) LINK.
    unnamed = [
        ident
        for ident in (_T5729_MISFILED_SESSION, _T5729_HOME_ENTRY_ID)
        if ident not in correction["note"]
    ]
    assert unnamed == [], (
        f"the annotation note does not name {unnamed} — naming them is the only "
        "link from an annotation to the sighting it annotates and its new home"
    )

    # (5) ROUTE.
    route = _session_record(_T5729_RECORD_PATH, _T5729_MISFILED_SESSION)
    assert route.get("date") == _T5729_MISFILED_DATE, (
        f"the route must keep the observation's own date, not {route.get('date')!r}"
    )
    assert not route.get("corrections") and not route.get("candidates"), (
        "the route only adds a sighting; it rewrites no entry and files nothing new"
    )
    matches = route.get("matches") or []
    assert len(matches) == 1, f"expected exactly one match op, got {len(matches)}"
    match = matches[0]
    assert match.get("entry_id") == _T5729_HOME_ENTRY_ID
    assert _T5687_ENTRY_ID in match.get("note", ""), (
        f"the route note does not name {_T5687_ENTRY_ID!r}, where the sighting "
        "was misfiled and still stands"
    )

    codebook = mod.load(_LIVE_CODEBOOK_PATH)
    _assert_sighting_carries(route, match, _live_entry(codebook, _T5729_HOME_ENTRY_ID))

    # (6) ROUTE NO-OP.
    merge = merge_one(codebook, route)
    assert merge_report(merge) == clean_merge_report(), (
        f"re-applying the route changed something — the committed codebook has "
        f"not fully absorbed it: {merge.summary()}"
    )


# ---------------------------------------------------------------------------
# task 7448 (live-file guard) — task 5687's whole outcome for
# entry-cand-20260919-2 was its correction, so its done is never read as a fix.
# The 2026-10-24 census read it as one and set the entry fixed (0f443e6c0e).
# ---------------------------------------------------------------------------

_T7448_RECORD_PATH = (
    _REPO_ROOT / "docs" / "legibility" / "coding-records" / "task-7448-corrections.jsonl"
)
_T7448_SESSION = "task-7448-5687-outcome-is-a-correction"
_T7448_CORRECTION_TASK = 5687


def _t7448_refuting_verdict(project_root: str) -> verdict.Verdict:
    return verdict.parse_verdict(
        {
            "verified": False,
            "reason": "re-verified at as_of_sha: the finding is not observed",
            "anchor": "slug:embedded-triple-quote-in-heredoc",
            "tags": ["h13", verdict.KIND_TAG],
            "severity": "medium",
            "severity_reason": "r",
            "route": "mechanical",
        },
        title="a confusion",
        project_root=project_root,
    )


def test_live_codebook_records_task_5687_as_a_correction_and_never_projects_it_fixed():
    """(a) The note-only correction naming task 5687 in `outcome_of_task` is
    absorbed, and the entry stays open. (b) A done 5687 queues no re-verify.
    (c) Even a GONE re-verify against 5687 projects no fix: this reproduces the
    2026-10-24 re-flip over the committed codebook."""
    # (a)
    _, correction, entry = _assert_correction_absorbed(
        _T7448_RECORD_PATH,
        _T7448_SESSION,
        _T5687_ENTRY_ID,
        refuted_tokens=_T5687_REFUTED_TOKENS,
        expected_status="open",
        original_sightings=_T5687_ORIGINAL_SIGHTING,
    )
    rewritten = {"title", "cause", "status"} & correction.keys()
    assert rewritten == set(), (
        f"the {_T7448_SESSION!r} correction writes {sorted(rewritten)}; it only "
        "records what task 5687's outcome was"
    )
    (root,) = {
        row["project_root"]
        for row in entry.get("filed_tickets") or []
        if row.get("task_id") == _T7448_CORRECTION_TASK
    }
    assert correction.get("outcome_of_task") == {
        "project_root": root, "task_id": _T7448_CORRECTION_TASK,
    }

    # (b)
    live = mod.load(_LIVE_CODEBOOK_PATH)
    statuses = {
        str(row["task_id"]): "done"
        for listed in live.get("entries") or []
        for row in listed.get("filed_tickets") or []
        if row.get("task_id") is not None
    }
    outcomes = disposition.read_outcomes(
        live, statuses=statuses, statuses_error=None, observed_root=root, store=FakeTaskStore(),
    )
    assert _T5687_ENTRY_ID not in {due.entry_id for due in outcomes.due}

    # (c)
    gone = disposition.Reverification(
        _T5687_ENTRY_ID,
        disposition.TaskRef(root, _T7448_CORRECTION_TASK),
        disposition.ReverifyResult.GONE,
        found=_t7448_refuting_verdict(root),
    )
    projected, changes = disposition.project(
        live, outcomes, {_T5687_ENTRY_ID: gone}, as_of_sha="c" * 40,
    )
    after = _live_entry(projected, _T5687_ENTRY_ID)
    assert (after["status"], after.get("fixed_at_sha")) == ("open", None)
    assert _T5687_ENTRY_ID not in {change.entry_id for change in changes}
