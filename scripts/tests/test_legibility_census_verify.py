"""Tests for scripts/legibility/census_verify.py — the census's default per-cluster verifier.

Every model call is a fake ``invoke`` seam and every headroom probe a fake
``headroom_probe``, so no test reaches a real ``claude`` process.
"""
from __future__ import annotations

import json
import logging

import pytest
from legibility import census_headroom, census_verify, session_runner, verdict
from legibility_census_scaffold import cap_themed_verdict, make_fake_invoke, quality_block_of
from quality_doc_texts import code_quality
from shared.cap_markers import (
    BLOCKING_BANNER_MARKERS,
    REAL_CLI_CAP_MESSAGES,
    looks_like_blocking_banner,
)


def _verdict(verified=True):
    return json.dumps({"verified": verified, "reason": "because"})


_REMEDIATED_TITLE = "Docs omit the X convention"


# ---------------------------------------------------------------------------
# task 3645 / DEFECT 3: a cap arriving DURING verification must never be
# laundered into a verdict.
#
# census_verify.build_verify_fn fails CLOSED per cluster: any invocation error or
# parse failure appends the cluster to `rejected` and continues. That is the
# right default for a genuinely unverifiable CLAIM, and exactly wrong for an
# infra failure -- a cap mid-verify silently mass-rejects the remaining
# population, and the run exits 0 looking unremarkable.
#
# Three detectors, ordered cheapest-first. These two fire at the FIRST
# corrupted cluster, which is what a periodic counter alone cannot do.
# ---------------------------------------------------------------------------

def _clusters(n):
    return [{"title": f"cluster-{i}"} for i in range(n)]


def _make_recording_probe(*results):
    """Fake `headroom_probe() -> HeadroomResult` seam, recording its calls.

    Returns the i-th result for the i-th call, repeating the last one after
    that, so a test can say "healthy, then capped" without counting calls.
    """
    calls = []

    def probe():
        calls.append(len(calls))
        return results[min(len(calls) - 1, len(results) - 1)]

    probe.calls = calls
    return probe


@pytest.mark.parametrize("message", REAL_CLI_CAP_MESSAGES)
def test_default_verify_fn_raises_when_the_raw_reply_carries_a_banner(message):
    """Detector (a): the cap text arrives as the verify reply itself.

    This is the common case -- the CLI prints its cap banner to stdout, which
    parse_coder_output would otherwise turn into a parse failure and thus a
    rejection. The banner-matching CONTENT is only the TRIGGER: a probe
    CONFIRMS the cap before the run is aborted, so the detector can never
    decide from content alone that the account is capped.

    Parametrized over every recorded real-CLI cap message, so adding a newly
    observed phrasing to the corpus exercises this detector too rather than
    only the marker list.
    """
    cap = message
    replies = [_verdict(), _verdict(), cap, _verdict(), _verdict()]
    calls = []

    def invoke(prompt, model):
        calls.append(prompt)
        return replies[len(calls) - 1]

    probe = _make_recording_probe(
        census_headroom.HeadroomResult(ok=False, reason="probe reply carries a banner marker: 'weekly limit'")
    )
    verify_fn = census_verify.build_verify_fn(
        "/tmp/root", invoke, headroom_probe=probe, probe_every=100,
    )

    with pytest.raises(census_headroom.CensusHeadroomExhausted) as excinfo:
        verify_fn(_clusters(5), model="sonnet")

    exc = excinfo.value
    assert exc.stage is census_headroom.CensusStage.VERIFY
    assert exc.reason and "banner" in exc.reason.lower()
    assert exc.verified == 2, "the two clusters adjudicated before the cap"
    assert exc.unverified == 3, "the hitting cluster plus the two never attempted"
    assert len(probe.calls) == 1, "exactly one confirming probe, on a path that already failed"


# ---------------------------------------------------------------------------
# task 3645 / REVIEW FIX, part 1/2: detector (a) must never re-read a
# WELL-FORMED verdict as a cap banner.
#
# The verify reply is a model-authored {"verified":…, "reason":…} whose reason
# legitimately QUOTES the cluster under adjudication -- and this repo's
# codebook is dominated by clusters ABOUT usage/weekly limits
# (docs/legibility/confusion-codebook.yaml carries ~15 such sightings at lines
# 569, 575, 925, 985, 1048-1163). A loose OR-substring scan over the whole raw
# reply therefore fires on ordinary healthy output.
#
# The consequence is worse than merely over-deferring: _defer returns before
# every write AND before census_state.advance, so last_census_at is untouched,
# the next run re-mines the same window, re-hits the same cluster and aborts
# again -- a permanent census stall plus a stream of false infra_issue
# escalations claiming the account is capped.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("marker", BLOCKING_BANNER_MARKERS)
def test_default_verify_fn_does_not_read_a_cap_themed_verdict_as_a_banner(marker):
    """A reply that PARSES into a verdict IS a verdict, whatever it quotes.

    Parametrized over every marker in the shared contract, so the guarantee is
    "no marker can be smuggled through a reason string", not "the two the test
    author happened to think of".
    """
    def invoke(prompt, model):
        return cap_themed_verdict(marker)

    probe = _make_recording_probe(
        census_headroom.HeadroomResult(ok=False, reason="must never be consulted on a parsed verdict")
    )
    verify_fn = census_verify.build_verify_fn(
        "/tmp/root", invoke, headroom_probe=probe, probe_every=100,
    )

    result = verify_fn(_clusters(3), model="sonnet")

    assert [c["title"] for c in result["verified"]] == ["cluster-0", "cluster-1", "cluster-2"]
    assert result["rejected"] == []
    assert probe.calls == [], "a parsed verdict is never a cap, so nothing to confirm"


@pytest.mark.parametrize("marker", ["usage limit", "weekly limit"])
def test_default_verify_fn_reads_a_cap_themed_refutation_as_an_ordinary_rejection(marker):
    """`verified: false` about a cap-themed cluster is an ordinary verdict too.

    The refutation half of the same class: a well-formed rejection must reject,
    not abort the census.
    """
    def invoke(prompt, model):
        return cap_themed_verdict(marker, verified=False)

    probe = _make_recording_probe(census_headroom.HeadroomResult(ok=False, reason="must never be consulted"))
    verify_fn = census_verify.build_verify_fn(
        "/tmp/root", invoke, headroom_probe=probe, probe_every=100,
    )

    result = verify_fn(_clusters(2), model="sonnet")

    assert result["verified"] == []
    assert [c["title"] for c in result["rejected"]] == ["cluster-0", "cluster-1"]
    assert probe.calls == []


# ---------------------------------------------------------------------------
# task 3645 / REVIEW FIX, part 2/2: the RESIDUAL half of the same false-
# positive class.
#
# Confining the scan to parse FAILURES stops a well-formed verdict being
# re-read as a banner, but an unparseable reply is still arbitrary model
# output about cap-related subject matter: if the model ignores the prompt's
# "STRICT JSON ONLY (no prose, no markdown fences)" instruction and answers in
# prose about a usage-limit cluster, the marker still matches and the run
# still aborts with nothing written -- the identical self-perpetuating stall,
# just reached less often.
#
# The principle these pin: banner-matching CONTENT may only TRIGGER a probe,
# never itself decide that the account is capped.
# ---------------------------------------------------------------------------

_UNPARSEABLE_CAP_THEMED_PROSE = (
    "Verified. The rotation really was interrupted by a usage limit at turns "
    "17 and 27, per the digest."
)


def _prose_at_cluster_two_invoke():
    """`invoke` returning STRICT-JSON-violating prose for cluster #2 of 4.

    The prose is about a cap-themed cluster, so it matches a marker -- and it
    is unparseable, so it reaches the scan. This is the residual case that
    survives the parse-success split.
    """
    calls = []

    def invoke(prompt, model):
        calls.append(prompt)
        if len(calls) == 2:
            return _UNPARSEABLE_CAP_THEMED_PROSE
        return _verdict()

    return invoke


def test_default_verify_fn_rejects_unparseable_cap_themed_prose_when_the_probe_says_healthy():
    """A healthy probe means the marker matched CONTENT, not a real cap.

    The fail-closed default for an unparseable verdict is preserved: the
    cluster rejects and the loop runs to completion, exactly as it would for
    any other malformed reply.
    """
    probe = _make_recording_probe(census_headroom.HeadroomResult(ok=True))
    verify_fn = census_verify.build_verify_fn(
        "/tmp/root", _prose_at_cluster_two_invoke(), headroom_probe=probe, probe_every=100,
    )

    result = verify_fn(_clusters(4), model="sonnet")

    assert len(probe.calls) == 1, "one confirming probe, only on the matched-marker path"
    assert len(result["verified"]) == 3, "the loop ran to completion"
    assert [c["title"] for c in result["rejected"]] == ["cluster-1"]


def test_default_verify_fn_raises_when_an_unparseable_banner_reply_is_probe_confirmed():
    """A probe that reports no capacity turns the same content into an abort."""
    probe = _make_recording_probe(
        census_headroom.HeadroomResult(ok=False, reason="probe reply carries a banner marker: 'weekly limit'")
    )
    verify_fn = census_verify.build_verify_fn(
        "/tmp/root", _prose_at_cluster_two_invoke(), headroom_probe=probe, probe_every=100,
    )

    with pytest.raises(census_headroom.CensusHeadroomExhausted) as excinfo:
        verify_fn(_clusters(4), model="sonnet")

    exc = excinfo.value
    assert exc.stage is census_headroom.CensusStage.VERIFY
    assert exc.verified == 1
    assert exc.unverified == 3
    reason = (exc.reason or "").lower()
    assert "usage limit" in reason, "the reason names the matched marker"
    assert "probe" in reason, "and says a probe confirmed it, not that content decided it"


def test_default_verify_fn_never_infers_a_cap_from_content_without_a_probe():
    """With no probe there is no way to confirm, so a cap is never inferred.

    This is the whole principle in one assertion: content triggers, the probe
    decides. Absent a probe, banner-matching content is just another
    unparseable verdict.
    """
    verify_fn = census_verify.build_verify_fn("/tmp/root", _prose_at_cluster_two_invoke())

    result = verify_fn(_clusters(4), model="sonnet")

    assert len(result["verified"]) == 3
    assert [c["title"] for c in result["rejected"]] == ["cluster-1"]


def test_default_verify_fn_raises_when_an_invocation_error_reprobes_capped():
    """Detector (b): a per-cluster invocation error, then a failing re-probe.

    The cluster that hit the cap must NOT be recorded as rejected. Recording
    it would be the defect in miniature: an infra failure written into the
    codebook as a verdict about the claim.
    """
    calls = []

    def invoke(prompt, model):
        calls.append(prompt)
        if len(calls) == 2:
            raise session_runner.InvocationFailed("claude CLI exited 1: simulated cap")
        return _verdict()

    probe = _make_recording_probe(
        census_headroom.HeadroomResult(ok=False, reason="probe reply carries a banner marker")
    )
    verify_fn = census_verify.build_verify_fn(
        "/tmp/root", invoke, headroom_probe=probe, probe_every=100,
    )

    with pytest.raises(census_headroom.CensusHeadroomExhausted) as excinfo:
        verify_fn(_clusters(4), model="sonnet")

    exc = excinfo.value
    assert exc.stage is census_headroom.CensusStage.VERIFY
    assert exc.verified == 1
    assert exc.unverified == 3
    assert len(probe.calls) == 1, "exactly one re-probe, on a path that already failed"


def test_default_verify_fn_still_rejects_when_the_reprobe_says_healthy():
    """Detector (b), negative: a healthy re-probe preserves the fail-closed contract.

    An invocation error with capacity still available means the claim really
    could not be verified. That must keep rejecting exactly as it does today
    -- the guard narrows the fail-closed default to non-cap causes, it does
    not remove it.
    """
    calls = []

    def invoke(prompt, model):
        calls.append(prompt)
        if len(calls) == 2:
            raise session_runner.InvocationFailed("claude CLI exited 1: unrelated outage")
        return _verdict()

    probe = _make_recording_probe(census_headroom.HeadroomResult(ok=True))
    verify_fn = census_verify.build_verify_fn(
        "/tmp/root", invoke, headroom_probe=probe, probe_every=100,
    )

    result = verify_fn(_clusters(4), model="sonnet")

    assert len(probe.calls) == 1
    assert len(result["verified"]) == 3, "the loop ran to completion"
    rejected_titles = {c["title"] for c in result["rejected"]}
    assert rejected_titles == {"cluster-1"}, "the failing cluster rejects, as today"


def test_default_verify_fn_without_probe_args_behaves_exactly_as_before():
    """The pre-existing signature keeps working, unguarded and fail-closed.

    Every existing call site and test passes no probe; they must be
    completely unaffected by this work.
    """
    calls = []

    def invoke(prompt, model):
        calls.append(prompt)
        if len(calls) == 2:
            raise session_runner.InvocationFailed("claude CLI exited 1: simulated outage")
        return _verdict()

    verify_fn = census_verify.build_verify_fn("/tmp/root", invoke)
    result = verify_fn(_clusters(3), model="sonnet")

    assert len(result["verified"]) == 2
    assert [c["title"] for c in result["rejected"]] == ["cluster-1"]
    assert result["fixed"] == []


def test_default_verify_fn_probes_at_batch_boundaries_only():
    """Detector (c): the periodic backstop probes between batches, not per cluster.

    `invoke` returns a clean verdict every time, so neither the banner
    detector nor the exception detector can fire — this isolates the backstop.
    12 clusters at probe_every=5 means boundaries after clusters 5 and 10:
    twice, not twelve times. The probe is cheap but not free, and verification
    is already one agentic call per cluster.
    """
    probe = _make_recording_probe(census_headroom.HeadroomResult(ok=True))
    verify_fn = census_verify.build_verify_fn(
        "/tmp/root", lambda prompt, model: _verdict(), headroom_probe=probe, probe_every=5,
    )

    result = verify_fn(_clusters(12), model="sonnet")

    assert len(result["verified"]) == 12
    assert len(probe.calls) == 2, "boundaries after clusters 5 and 10 only"


def test_default_verify_fn_backstop_raises_at_the_first_failing_boundary():
    """The backstop aborts at its boundary, leaving the rest untouched."""
    probe = _make_recording_probe(
        census_headroom.HeadroomResult(ok=False, reason="probe reply carries a banner marker")
    )
    verify_fn = census_verify.build_verify_fn(
        "/tmp/root", lambda prompt, model: _verdict(), headroom_probe=probe, probe_every=5,
    )

    with pytest.raises(census_headroom.CensusHeadroomExhausted) as excinfo:
        verify_fn(_clusters(12), model="sonnet")

    exc = excinfo.value
    assert exc.stage is census_headroom.CensusStage.VERIFY
    assert exc.verified == 5
    assert exc.unverified == 7, "clusters 6-12, none of them adjudicated"
    assert len(probe.calls) == 1


def test_default_verify_fn_backstop_does_not_probe_after_the_last_cluster():
    """No probe when nothing remains -- it would guard a stage already over."""
    probe = _make_recording_probe(census_headroom.HeadroomResult(ok=True))
    verify_fn = census_verify.build_verify_fn(
        "/tmp/root", lambda prompt, model: _verdict(), headroom_probe=probe, probe_every=5,
    )

    verify_fn(_clusters(5), model="sonnet")

    assert probe.calls == [], "cluster 5 is the last one; there is nothing left to guard"


# ---------------------------------------------------------------------------
# task 3645 / AMENDMENT: the probe count must include the probes spent INSIDE
# verification.
#
# run_census counted only its own two probes (preflight + the stage gate). The
# three in-verify detectors probe through the `headroom_probe` seam, which
# main() wires with no counter -- so a healthy 20-cluster run at
# probe_every=5 really spends 1 + 1 + 3 = 5 trickle probes and the report
# still printed `headroom-probe=2`. Under-reporting its own spend, in the very
# artifact an operator reads to size a run, is the defect class this pipeline
# exists to find.
# ---------------------------------------------------------------------------

def test_default_verify_fn_reports_the_headroom_probes_it_spent():
    """The verifier reports its probe spend back in the result dict.

    That report is the ONLY way run_census can know what the probes inside
    verification cost: the seam is wired by main(), not by run_census, so a
    counter would have to be threaded through a second wiring site that
    nothing forces anyone to connect.
    """
    probe = _make_recording_probe(census_headroom.HeadroomResult(ok=True))
    verify_fn = census_verify.build_verify_fn(
        "/tmp/root", lambda prompt, model: _verdict(), headroom_probe=probe, probe_every=5,
    )

    result = verify_fn(_clusters(12), model="sonnet")

    assert len(probe.calls) == 2, "backstops after clusters 5 and 10"
    assert result["headroom_probes"] == 2, "and the result SAYS so"


def test_default_verify_fn_reports_zero_probes_when_it_has_no_probe_seam():
    """No probe seam -> nothing spent -> the report says 0, not nothing."""
    verify_fn = census_verify.build_verify_fn("/tmp/root", lambda prompt, model: _verdict())

    result = verify_fn(_clusters(12), model="sonnet")

    assert result["headroom_probes"] == 0


# ---------------------------------------------------------------------------
# task 3645 / AMENDMENT: `headroom_probe` is an INJECTED seam with no
# exception contract.
#
# Production wires `preflight_headroom`, which folds any failure into
# HeadroomResult(ok=False) -- but a seam that RAISES would escape _verify_fn,
# sail past run_census's `except CensusHeadroomExhausted` and land in main()'s
# hard-failure path: exit 1 plus a traceback escalation, after the whole
# mining spend, on a path that has ALREADY failed once. That is strictly worse
# than the deferral this feature exists to produce.
# ---------------------------------------------------------------------------

def _raising_probe(exc=None):
    """A `headroom_probe` seam that raises instead of returning a verdict."""
    def probe():
        raise exc or RuntimeError("probe transport blew up")

    return probe


def test_default_verify_fn_treats_a_raising_probe_as_no_headroom():
    """Detector (b): a raising re-probe defers, it does not crash the census."""
    calls = []

    def invoke(prompt, model):
        calls.append(prompt)
        if len(calls) == 2:
            raise session_runner.InvocationFailed("claude CLI exited 1: simulated cap")
        return _verdict()

    verify_fn = census_verify.build_verify_fn(
        "/tmp/root", invoke, headroom_probe=_raising_probe(), probe_every=100,
    )

    with pytest.raises(census_headroom.CensusHeadroomExhausted) as excinfo:
        verify_fn(_clusters(4), model="sonnet")

    exc = excinfo.value
    assert exc.stage is census_headroom.CensusStage.VERIFY
    assert exc.verified == 1
    assert exc.unverified == 3
    assert "raised" in (exc.reason or "").lower(), (
        "the reason says the probe itself failed, not that a banner was seen"
    )


def test_default_verify_fn_backstop_treats_a_raising_probe_as_no_headroom():
    """Detector (c): same fail-safe stance at the periodic backstop."""
    verify_fn = census_verify.build_verify_fn(
        "/tmp/root",
        lambda prompt, model: _verdict(),
        headroom_probe=_raising_probe(),
        probe_every=5,
    )

    with pytest.raises(census_headroom.CensusHeadroomExhausted) as excinfo:
        verify_fn(_clusters(12), model="sonnet")

    exc = excinfo.value
    assert exc.stage is census_headroom.CensusStage.VERIFY
    assert exc.verified == 5
    assert exc.unverified == 7
    assert "raised" in (exc.reason or "").lower()


def test_default_verify_fn_unparseable_banner_treats_a_raising_probe_as_no_headroom():
    """Detector (a): a raising confirmation probe defers rather than escaping."""
    verify_fn = census_verify.build_verify_fn(
        "/tmp/root",
        _prose_at_cluster_two_invoke(),
        headroom_probe=_raising_probe(),
        probe_every=100,
    )

    with pytest.raises(census_headroom.CensusHeadroomExhausted) as excinfo:
        verify_fn(_clusters(4), model="sonnet")

    exc = excinfo.value
    assert exc.stage is census_headroom.CensusStage.VERIFY
    assert exc.verified == 1
    assert exc.unverified == 3
    assert "raised" in (exc.reason or "").lower()


# ---------------------------------------------------------------------------
# task 4879 step-5: RED — W-B. `CensusHeadroomExhausted` carries `verified`
# and `unverified` but NOT `rejected`, so on any run where a cluster was
# rejected before the hit the two counts silently fail to account for the
# whole run: an operator reading "1 verified, 3 unverified" of a 5-cluster
# run cannot tell whether the missing cluster was work already spent or a
# bookkeeping bug. A rejection is real, paid-for adjudication, not an
# absence. Every case below drives the REAL `census_verify.build_verify_fn` (never
# a hand-raised exception) and arranges at least one PRECEDING rejection, so
# the divergence the current two counts hide is actually present.
# ---------------------------------------------------------------------------

_UNPARSEABLE_PLAIN_PROSE = (
    "The rotation looks fine to me and I have nothing further to add about it."
)


_UNPARSEABLE_BANNERED_PROSE = (
    "I cannot answer right now: this account has hit its weekly limit."
)


def _no_headroom(reason="probe reports no capacity"):
    return census_headroom.HeadroomResult(ok=False, reason=reason)


def test_census_headroom_exhausted_counts_account_for_every_offered_cluster_at_the_backstop():
    """(a) The periodic backstop site, with the hitting cluster ALREADY
    adjudicated (so `unverified == remaining - 1`)."""
    calls = []

    def invoke(prompt, model):
        calls.append(prompt)
        if len(calls) == 1:
            return _UNPARSEABLE_PLAIN_PROSE   # cluster 0 rejects, no banner, no probe
        return _verdict()                      # clusters 1 and 2 verify

    clusters = _clusters(5)
    verify_fn = census_verify.build_verify_fn(
        "/tmp/root", invoke,
        headroom_probe=_make_recording_probe(_no_headroom()),
        probe_every=3,
    )

    with pytest.raises(census_headroom.CensusHeadroomExhausted) as excinfo:
        verify_fn(clusters, model="sonnet")

    exc = excinfo.value
    assert exc.verified == 2, "clusters 1 and 2 came back verified"
    assert exc.rejected == 1, "cluster 0's unparseable verdict is real work already spent"
    assert exc.unverified == 2, "clusters 3 and 4 were never attempted"
    assert exc.verified + exc.rejected + exc.unverified == len(clusters)


def test_census_headroom_exhausted_counts_account_for_every_offered_cluster_at_the_invoke_error():
    """(b) The invocation-error site, where the hitting cluster is NOT yet
    appended (so `unverified == remaining`)."""
    calls = []

    def invoke(prompt, model):
        calls.append(prompt)
        if len(calls) == 1:
            return _verdict()                  # cluster 0 verifies
        if len(calls) == 2:
            return _UNPARSEABLE_PLAIN_PROSE    # cluster 1 rejects
        raise session_runner.InvocationFailed("claude CLI exited 1: simulated cap")

    clusters = _clusters(5)
    verify_fn = census_verify.build_verify_fn(
        "/tmp/root", invoke,
        headroom_probe=_make_recording_probe(_no_headroom()),
        probe_every=100,                       # no backstop — isolate this site
    )

    with pytest.raises(census_headroom.CensusHeadroomExhausted) as excinfo:
        verify_fn(clusters, model="sonnet")

    exc = excinfo.value
    assert exc.verified == 1
    assert exc.rejected == 1
    assert exc.unverified == 3, "the hitting cluster 2, plus 3 and 4"
    assert exc.verified + exc.rejected + exc.unverified == len(clusters)


def test_census_headroom_exhausted_counts_account_for_every_offered_cluster_at_the_banner():
    """(c) The unparseable-banner site — also a not-yet-appended site."""
    assert looks_like_blocking_banner(_UNPARSEABLE_BANNERED_PROSE) is not None, (
        "fixture sanity: the bannered reply must actually match a marker"
    )
    assert looks_like_blocking_banner(_UNPARSEABLE_PLAIN_PROSE) is None, (
        "fixture sanity: the plain reply must NOT match, so it spends no probe"
    )
    calls = []

    def invoke(prompt, model):
        calls.append(prompt)
        if len(calls) == 1:
            return _verdict()                  # cluster 0 verifies
        if len(calls) == 2:
            return _UNPARSEABLE_PLAIN_PROSE    # cluster 1 rejects, no probe spent
        return _UNPARSEABLE_BANNERED_PROSE     # cluster 2: unparseable WITH a marker

    clusters = _clusters(5)
    verify_fn = census_verify.build_verify_fn(
        "/tmp/root", invoke,
        headroom_probe=_make_recording_probe(_no_headroom()),
        probe_every=100,
    )

    with pytest.raises(census_headroom.CensusHeadroomExhausted) as excinfo:
        verify_fn(clusters, model="sonnet")

    exc = excinfo.value
    assert "banner" in exc.reason, "this must be the banner site, not the backstop"
    assert exc.verified == 1
    assert exc.rejected == 1
    assert exc.unverified == 3
    assert exc.verified + exc.rejected + exc.unverified == len(clusters)


# ---------------------------------------------------------------------------
# task 5931 step-9: RED — the default verifier proposes an IN-TREE
# remediation, validated against the observed tree before it is attached.
# ---------------------------------------------------------------------------

def _tree_with_guide(tmp_path):
    """An observed tree holding docs/guide.md, with room OUTSIDE it (tmp_path
    itself) for a path that escapes the tree."""
    root = tmp_path / "tree"
    (root / "docs").mkdir(parents=True)
    (root / "docs" / "guide.md").write_text("guide\n", encoding="utf-8")
    return root


def _remediation_verdict(remediation, *, verified=True):
    return json.dumps({"verified": verified, "reason": "r", "remediation": remediation})


def _verify_one(root, reply, *, cluster=None):
    verify_fn = census_verify.build_verify_fn(str(root), lambda prompt, model: reply)
    return verify_fn([cluster or {"title": _REMEDIATED_TITLE}], model="sonnet")


@pytest.mark.parametrize("absolute", [False, True], ids=["relative", "absolute"])
def test_default_verify_fn_attaches_an_in_tree_remediation(tmp_path, absolute):
    root = _tree_with_guide(tmp_path)
    path = str(root / "docs" / "guide.md") if absolute else "docs/guide.md"
    result = _verify_one(root, _remediation_verdict({"path": path, "change": "Document X"}))
    [cluster] = result["verified"]
    assert cluster["remediation"] == {"path": "docs/guide.md", "change": "Document X"}


def test_default_verify_fn_drops_a_nonexistent_remediation_path_loudly(tmp_path, caplog):
    root = _tree_with_guide(tmp_path)
    with caplog.at_level(logging.WARNING, logger="legibility.census"):
        result = _verify_one(
            root, _remediation_verdict({"path": "docs/missing.md", "change": "Document X"}),
        )
    [cluster] = result["verified"]
    assert "remediation" not in cluster
    assert any(
        _REMEDIATED_TITLE in record.getMessage() and "docs/missing.md" in record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
    )


@pytest.mark.parametrize("absolute", [False, True], ids=["dot-dot", "absolute"])
def test_default_verify_fn_drops_a_remediation_path_outside_the_tree(tmp_path, absolute):
    root = _tree_with_guide(tmp_path)
    outside = tmp_path / "outside.md"
    outside.write_text("outside\n", encoding="utf-8")
    path = str(outside) if absolute else "../outside.md"
    result = _verify_one(root, _remediation_verdict({"path": path, "change": "Document X"}))
    [cluster] = result["verified"]
    assert "remediation" not in cluster


@pytest.mark.parametrize(
    "path_of",
    [
        pytest.param(lambda root: ".", id="dot"),
        pytest.param(lambda root: "./", id="dot-slash"),
        pytest.param(lambda root: "docs/..", id="up-to-root"),
        pytest.param(str, id="absolute"),
    ],
)
def test_default_verify_fn_drops_a_remediation_naming_the_tree_root(tmp_path, path_of):
    root = _tree_with_guide(tmp_path)
    reply = _remediation_verdict({"path": path_of(root), "change": "Document X"})
    [cluster] = _verify_one(root, reply)["verified"]
    assert "remediation" not in cluster


@pytest.mark.parametrize("path", ["docs/a\x00b.md", "x" * 5000], ids=["nul-byte", "too-long"])
def test_default_verify_fn_drops_an_unresolvable_remediation_path(tmp_path, path):
    result = _verify_one(
        _tree_with_guide(tmp_path), _remediation_verdict({"path": path, "change": "Document X"}),
    )
    [cluster] = result["verified"]
    assert "remediation" not in cluster


@pytest.mark.parametrize(
    "reply",
    [
        pytest.param(_remediation_verdict(None), id="null"),
        pytest.param(json.dumps({"verified": True, "reason": "r"}), id="absent"),
    ],
)
def test_default_verify_fn_without_remediation_still_verifies(tmp_path, reply):
    [cluster] = _verify_one(_tree_with_guide(tmp_path), reply)["verified"]
    assert "remediation" not in cluster


def test_default_verify_fn_never_attaches_a_remediation_to_a_rejection(tmp_path):
    reply = _remediation_verdict({"path": "docs/guide.md", "change": "Document X"}, verified=False)
    result = _verify_one(_tree_with_guide(tmp_path), reply)
    assert result["verified"] == []
    [cluster] = result["rejected"]
    assert "remediation" not in cluster


def test_default_verify_fn_does_not_mutate_the_input_cluster(tmp_path):
    cluster = {"title": _REMEDIATED_TITLE}
    reply = _remediation_verdict({"path": "docs/guide.md", "change": "Document X"})
    _verify_one(_tree_with_guide(tmp_path), reply, cluster=cluster)
    assert cluster == {"title": _REMEDIATED_TITLE}


def test_verify_prompt_requests_the_remediation_key(tmp_path):
    fake_invoke = make_fake_invoke(default=_verdict())
    census_verify.build_verify_fn(str(tmp_path), fake_invoke)(_clusters(1), model="sonnet")
    assert '"remediation"' in fake_invoke.calls[0]["prompt"]


# ---------------------------------------------------------------------------
# task 6401 step-7: RED — the default verifier parses every reply into a
# typed verdict.Verdict (plans/census-incremental-prd.md §4.4 C4) and logs
# each normalisation it needed.
# ---------------------------------------------------------------------------

_GUIDE_REMEDIATION = {"path": "docs/guide.md", "change": "Document X"}


def _c4_reply(**overrides):
    reply = {
        "verified": True,
        "reason": "r",
        "anchor": "docs/guide.md",
        "tags": ["h11"],
        "severity": "high",
        "severity_reason": "the next edit to the guide repeats it",
        "route": "mechanical",
        "remediation": _GUIDE_REMEDIATION,
    }
    reply.update(overrides)
    return json.dumps(reply)


def test_default_verify_fn_attaches_the_parsed_verdict_to_a_verified_cluster(tmp_path):
    [cluster] = _verify_one(_tree_with_guide(tmp_path), _c4_reply())["verified"]

    v = cluster["verdict"]
    assert isinstance(v, verdict.Verdict)
    assert v.severity == "high"
    assert v.tags == ("h11", "kind:confusion")
    assert v.anchor == "docs/guide.md"
    assert v.remediation is not None
    assert cluster["remediation"] == v.remediation.to_record()


def test_default_verify_fn_attaches_the_parsed_verdict_to_a_refuted_cluster(tmp_path):
    result = _verify_one(_tree_with_guide(tmp_path), _c4_reply(verified=False))

    assert result["verified"] == []
    [cluster] = result["rejected"]
    assert isinstance(cluster["verdict"], verdict.Verdict)
    assert cluster["verdict"].verified is False
    assert "remediation" not in cluster


def test_default_verify_fn_rejects_a_parsed_reply_that_is_no_verdict_without_a_probe(
    tmp_path, caplog,
):
    reply = json.dumps({"verified": "yes", "reason": REAL_CLI_CAP_MESSAGES[0]})
    probe = _make_recording_probe(census_headroom.HeadroomResult(ok=False, reason="must never be consulted"))
    verify_fn = census_verify.build_verify_fn(
        str(_tree_with_guide(tmp_path)), lambda prompt, model: reply, headroom_probe=probe,
    )

    with caplog.at_level(logging.WARNING, logger="legibility.census"):
        result = verify_fn([{"title": _REMEDIATED_TITLE}], model="sonnet")

    assert result["verified"] == []
    assert result["rejected"] == [{"title": _REMEDIATED_TITLE}]
    assert probe.calls == []
    assert any(
        _REMEDIATED_TITLE in record.getMessage() and "verified" in record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
    )


def test_default_verify_fn_logs_one_warning_per_verdict_normalisation(tmp_path, caplog):
    reply = _c4_reply(severity="critical", tags=["h11", "bogus"])

    with caplog.at_level(logging.WARNING, logger="legibility.census"):
        _verify_one(_tree_with_guide(tmp_path), reply)

    messages = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING and record.name == "legibility.census.verdicts"
    ]
    for kind, offered in (("tag_dropped", "'bogus'"), ("severity_substituted", "'critical'")):
        naming = [m for m in messages if _REMEDIATED_TITLE in m and kind in m and offered in m]
        assert len(naming) == 1, (kind, messages)


# ---------------------------------------------------------------------------
# task 6401 step-9: RED — both census prompts embed the quality definition
# rendered by code_quality.guidance() (docs/quality-findings-contract.md §10),
# computed here at test time, never pasted.
# ---------------------------------------------------------------------------

_C4_KEYS = (
    "verified", "reason", "anchor", "tags", "severity", "severity_reason", "route",
    "remediation",
)


def test_verify_prompt_embeds_the_quality_definition_and_asks_for_the_c4_verdict(tmp_path):
    fake_invoke = make_fake_invoke(default=_verdict())
    census_verify.build_verify_fn(str(tmp_path), fake_invoke)(_clusters(1), model="sonnet")
    prompt = fake_invoke.calls[0]["prompt"]

    assert quality_block_of(prompt) == code_quality.guidance()
    assert prompt.startswith("You are the periodic-census verifier")
    for key in _C4_KEYS:
        assert f'"{key}"' in prompt, key
    for severity in verdict.SEVERITIES:
        assert f'"{severity}"' in prompt, severity
    for route in verdict.Route:
        assert f'"{route.value}"' in prompt, route
    assert verdict.KIND_TAG not in prompt
    assert str(tmp_path) in prompt
