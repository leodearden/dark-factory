"""The census's verify seam: the default per-cluster verifier, and the Verdict every verified cluster carries."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

if (_SHARED_SRC := str(Path(__file__).resolve().parents[2] / "shared" / "src")) not in sys.path:
    sys.path.insert(0, _SHARED_SRC)
if (_ORCH_SRC := str(Path(__file__).resolve().parents[2] / "orchestrator" / "src")) not in sys.path:
    sys.path.insert(0, _ORCH_SRC)

from legibility import census_headroom, coder
from legibility import verdict as verdict_mod
from orchestrator.agents import code_quality
from shared.cap_markers import looks_like_blocking_banner

logger = logging.getLogger("legibility.census.verify")


_VERIFY_PROBE_EVERY = 5
"""How many clusters the default verifier adjudicates between backstop
headroom probes (``_build_default_verify_fn``'s third, weakest detector).

The trade-off this number sets: one extra trickle-tier probe per N
clusters, against up to N-1 clusters of misattributed verdicts in the one
residual case the cheaper detectors cannot see (a cap presenting as a
clean-but-wrong verdict). Verification is one agentic Sonnet call per
cluster, so a probe per cluster would be cheap but not free, and pointless
-- the two first-cluster-accurate detectors already cover the cases where
a cap announces itself."""


QUALITY_DEFINITION_MARKER = "=== QUALITY DEFINITION ===\n"


def _verify_prompt(cluster: dict, *, project_root: str, quality_block: str) -> str:
    """Prompt for the real Sonnet verify_fn: confirm-or-refute one novel
    cluster against *project_root*'s current main via targeted file reads,
    judged against *quality_block* (``code_quality.guidance``).

    The CLI subprocess this prompt is delivered to runs with its cwd set to
    *project_root* (``census_stage_specs`` binds it). That is load-bearing
    for a reason that is NOT relative-path resolution: the CLI
    SANDBOXES tool access to the cwd tree, so a verifier launched from
    anywhere else has every Read/Bash against this tree permission-denied,
    with no interactive prompt to approve (proven 2026-08-03). The
    absolute-paths instruction below is retained as belt-and-braces on top
    of that scoping, not as a substitute for it.

    Asks for an OBSERVATION, never a diagnosis (codebook lesson
    guards-assert-unverified-diagnoses)."""
    return (
        "You are the periodic-census verifier for the dark-factory "
        "agent-confusion codebook (plans/confusion-reduction-prd.md "
        "section 5.7). A trickle miner flagged the confusion cluster below "
        "as novel. Using targeted reads of " + str(project_root) + " "
        "(ABSOLUTE paths only), confirm whether this is still an "
        "OBSERVABLE fact about the CURRENT state of that tree -- never a "
        "diagnosis or a guess about root cause you cannot directly "
        "verify.\n\n"
        "Give a remediation ONLY when a specific EXISTING file or directory "
        "in this tree (never the tree root itself) produces, teaches, or fails to guard against the "
        "confusion; otherwise null -- including when the cause lies in "
        "tooling this tree does not contain.\n\n"
        "Judge the cost the confusion imposes on the next change against the "
        "quality definition below.\n\n"
        + QUALITY_DEFINITION_MARKER + quality_block
        + "\n=== RESPONSE ===\n" + verdict_mod.reply_instructions(str(project_root))
        + "\n=== CLUSTER ===\n" + json.dumps(cluster)
    )


def with_verdict(cluster: dict, found: verdict_mod.Verdict) -> dict:
    """*cluster* carrying its Verdict, its remediation exactly the Verdict's
    in-tree one when the claim is verified: a refuted claim, or a Verdict
    without a remediation, carries none whatever the cluster arrived with."""
    unremediated = {key: value for key, value in cluster.items() if key != "remediation"}
    with_verdict = {**unremediated, "verdict": found}
    if found.verified and found.remediation is not None:
        return {**with_verdict, "remediation": found.remediation.to_record()}
    return with_verdict


def _log_verdict_normalisations(title, found: verdict_mod.Verdict) -> None:
    for note in found.normalisations:
        logger.warning(
            "census: verdict for cluster %r normalised (%s): offered %s, used %r",
            title, note.kind.value, note.offered, note.used,
        )


def _verdict_of(cluster: dict, *, project_root: str) -> verdict_mod.Verdict:
    """*cluster*'s Verdict, else the verdict parser's reading of a verified
    reply stating only the cluster's own remediation: normalised, logged and
    counted like any other verdict."""
    found = cluster.get("verdict")
    if isinstance(found, verdict_mod.Verdict):
        return found
    defaulted = verdict_mod.parse_verdict(
        {"verified": True, "remediation": cluster.get("remediation")},
        title=cluster.get("title") or "",
        project_root=project_root,
    )
    _log_verdict_normalisations(cluster.get("title"), defaulted)
    return defaulted


def verified_with_verdicts(verified: list, *, project_root: str) -> list[dict]:
    """*verified*, every cluster carrying a Verdict (``_verdict_of``) and
    that Verdict's remediation (``_with_verdict``)."""
    verdictless = [
        cluster.get("title")
        for cluster in verified
        if not isinstance(cluster.get("verdict"), verdict_mod.Verdict)
    ]
    if verdictless:
        logger.warning(
            "census: %d verified cluster(s) arrived without a verdict; promoting "
            "them on the verdict parser's defaults: %s",
            len(verdictless), verdictless,
        )
    return [
        with_verdict(cluster, _verdict_of(cluster, project_root=project_root))
        for cluster in verified
    ]


def build_verify_fn(
    project_root: str,
    invoke,
    *,
    headroom_probe=None,
    probe_every: int = _VERIFY_PROBE_EVERY,
):
    """Build the real ``verify_fn(clusters, *, model)`` seam: one Sonnet
    call per cluster via *invoke* (in production the pooled runner's verify
    stage, ``census_stage_specs``), parsed via ``coder.parse_coder_output``
    and then ``verdict.parse_verdict``: verified and refuted clusters carry
    their parsed Verdict under ``"verdict"``. Any per-cluster failure (invocation error, unparseable output) rejects
    that cluster rather than crashing the whole census -- a conservative
    fail-closed default for an unverifiable claim. This default never
    reports a "fixed" entry (a stronger claim than a per-cluster verify
    prompt is designed to elicit).

    That fail-closed default is correct, and it is also what made the
    2026-08-03 sandbox gap SILENT: with the CLI subprocess rooted outside
    the censused tree, every permission-denied verifier read became an
    ordinary per-cluster rejection, so a whole census mass-rejected without
    a single error surfacing. What keeps the default honest rather than
    indiscriminate is ``census_stage_specs`` scoping the subprocess cwd to
    *project_root* -- a rejection then means the claim really could not be
    verified, not that the verifier could not see the tree.

    Scoping the cwd removed one CAUSE, not the class: a model that goes
    unreachable mid-run, a different permission denial, or persistently
    unparseable verdicts all still land here as ordinary rejections. So
    ``run_census`` additionally DETECTS the signature -- clusters offered,
    none verified -- and says so loudly rather than quietly reporting an
    empty census.

    **The cap case is carved out of the fail-closed default.** A usage cap
    is an infra failure, not a verdict about a claim, and must never be
    laundered into one -- that is the same silent-mass-rejection class the
    2026-08-03 incident above describes, arriving by a different route and
    invisible to the all-rejected detector whenever a cap lands partway
    through (some clusters verified, so the total-wipeout signature never
    fires). When *headroom_probe* is supplied, two detectors fire at the
    FIRST corrupted cluster:

    (a) a reply that FAILED TO PARSE as a verdict carries a blocking banner,
        and ONE re-probe then confirms no capacity. Banner-matching CONTENT
        may only TRIGGER a probe, never itself decide that the account is
        capped -- which makes (a) structurally identical to (b): content or
        exception triggers, the probe decides.

        The scan is confined to the parse-failure path because **a reply that
        parses into a verdict IS a verdict and must never be re-read as a
        banner.** The loose OR-substring marker list is safe on the ``"ping"``
        -> ``"pong"`` preflight probe precisely because that reply is fixed
        and contentless; it is unsafe the moment it is pointed at arbitrary
        model output, because a verify reply is a model-authored
        ``{"verified":..., "reason":...}`` whose reason legitimately QUOTES the
        cluster under adjudication -- and this repo's codebook is dominated by
        clusters ABOUT usage/weekly limits, so the markers match ordinary
        healthy content. Scanning pre-parse aborted the run on those clusters,
        and because the abort precedes ``census_state.advance`` the next run
        re-mined the same window and aborted again: a permanent census stall
        plus a stream of false "the account is capped" escalations. The
        probe-confirmation closes the residual half of that class, where the
        model ignores the prompt's STRICT-JSON instruction and answers in
        prose about a cap-themed cluster.

        Splitting on parse success loses no detection -- a real CLI cap banner
        is plain prose, and every entry of ``REAL_CLI_CAP_MESSAGES`` raises
        ``CoderParseError``, so all of them still reach the scan. That premise
        is load-bearing, so it is pinned directly rather than asserted here:
        ``test_every_real_cli_cap_message_fails_to_parse_as_a_verdict``
        parametrizes the whole corpus, and the detector's own test does too --
        a newly-observed phrasing that happened to embed a ``{...}`` fragment
        would parse, bypass this detector and go red there. The probe is
        reliable in BOTH directions here: ``preflight_headroom`` folds any
        probe exception into ``HeadroomResult(ok=False)``, so a genuinely
        capped account confirms whether its probe returns a banner or raises.
        Cost is bounded and paid only on a path that has ALREADY failed to
        parse AND matched a marker -- one trickle-tier probe, against a
        cap-themed cluster permanently stalling the census.
    (b) a per-cluster invocation exception triggers ONE re-probe. No
        headroom means ``CensusHeadroomExhausted``, and the hitting cluster
        is NOT recorded as rejected. Headroom still available means the
        claim really was unverifiable, and it rejects exactly as before --
        the fail-closed default is narrowed to non-cap causes, not removed.

    (c) every *probe_every* clusters, a backstop probe. Ordered LAST because
        it is the weakest: it is the only detector that can miss the first
        corrupted cluster, letting up to ``probe_every - 1`` clusters be
        fail-closed rejected before the cap is noticed -- which is the defect
        above reproduced in miniature. It earns its place by covering the one
        residual case neither (a) nor (b) can see: a cap presenting as a
        clean-but-wrong verdict, where the reply parses fine and nothing
        raises. It is a BACKSTOP for (a) and (b), never the primary, and a
        counter alone would not be an acceptable design.

        The cost trade-off, stated plainly: one extra trickle-tier probe per
        *probe_every* clusters, against up to ``probe_every - 1`` clusters of
        misattributed verdicts in that residual case. No probe is spent after
        the last cluster, where it would guard a stage already over.

    All three are no-ops without a *headroom_probe* -- literally, now that (a)
    confirms rather than assumes -- so every existing call site keeps its
    current behaviour exactly. Production reach is unaffected: ``main()``
    wires ``headroom_probe=functools.partial(preflight_headroom,
    verify_invoke, model=cfg.models.trickle)``, so every real run keeps full
    detection.

    Two properties of the probe seam itself, both owned by the ``_probe()``
    wrapper below rather than restated at the three call sites:

    * **It is counted.** The result dict carries ``headroom_probes`` -- how
      many probes this verifier actually spent -- which ``adjudicate`` folds
      into the report's cost note. Reported through the return value rather
      than a counter passed in, because the seam is wired by ``main()`` and a
      counter would need a second wiring site nothing forces anyone to
      connect. A caller that ignores the key loses only the count.
    * **It never raises.** *headroom_probe* is injected and carries no
      exception contract, so a raising seam is folded into
      ``HeadroomResult(ok=False)`` exactly as ``preflight_headroom`` folds an
      invocation error -- a deferral, not an exception escaping into
      ``main()``'s exit-1 path after the whole mining spend.
    """
    quality_block = code_quality.guidance()

    def _verify_fn(clusters, *, model):
        verified, rejected = [], []
        # Every probe THIS verifier spends, reported back in the result dict
        # so the report's cost note can state the run's real probe spend.
        # It rides the return value rather than a mutable counter passed in
        # because the probe seam is wired by main(), not by run_census: a
        # counter would need a second wiring site that nothing forces anyone
        # to connect, and an unwired counter is exactly how the cost note
        # came to under-report in the first place.
        probes = [0]

        def _probe():
            """Call the injected probe seam: counted, and never raising.

            *headroom_probe* is an injected seam with no exception contract.
            Production wires ``preflight_headroom``, which folds any failure
            into ``HeadroomResult(ok=False)`` -- but a seam that RAISES here
            would escape ``_verify_fn``, sail past ``adjudicate``'s
            ``except CensusHeadroomExhausted`` and land in ``main()``'s
            hard-failure path: exit 1 plus a traceback escalation, after the
            whole mining spend, on a path that has ALREADY failed once.
            Matching ``preflight_headroom``'s own fail-safe stance turns that
            into the deferral this guard exists to produce.
            """
            if headroom_probe is None:
                # Unreachable -- every call site below is guarded by
                # `headroom_probe is not None`. Stated rather than asserted so
                # a future caller that forgets the guard gets the DOCUMENTED
                # no-probe behaviour (the detectors are no-ops; nothing says
                # the account is capped) and spends no probe, instead of a
                # TypeError escaping mid-verification.
                return census_headroom.HeadroomResult(ok=True, reason=None)
            probes[0] += 1
            try:
                return headroom_probe()
            except Exception as exc:  # noqa: BLE001 - a raising probe defers, never crashes the census
                return census_headroom.HeadroomResult(ok=False, reason=f"headroom probe raised: {exc}")

        for index, cluster in enumerate(clusters):
            # The hitting cluster plus everything never attempted. Computed
            # identically at every raise site so the counts cannot disagree
            # about what "unverified" means. The `rejected` term is passed
            # the same way, from `len(rejected)` at every site, for the same
            # reason -- together with `verified` the three always sum to the
            # clusters offered (see CensusHeadroomExhausted's INVARIANT).
            remaining = len(clusters) - index
            prompt = _verify_prompt(
                cluster, project_root=project_root, quality_block=quality_block,
            )
            try:
                raw = invoke(prompt, model)
            except Exception as exc:  # noqa: BLE001 - an unverifiable claim rejects, never crashes
                # (b) One re-probe, only on a path that has already failed.
                if headroom_probe is not None:
                    probe = _probe()
                    if not probe.ok:
                        raise census_headroom.CensusHeadroomExhausted(
                            stage=census_headroom.CensusStage.VERIFY,
                            reason=(
                                "verify invocation failed and the headroom re-probe "
                                f"reports no capacity: {probe.reason or 'no headroom'}"
                            ),
                            verified=len(verified),
                            rejected=len(rejected),
                            unverified=remaining,
                        ) from exc
                logger.warning(
                    "census: verify failed for cluster %r: %s", cluster.get("title"), exc,
                )
                rejected.append(cluster)
                continue

            try:
                parsed = coder.parse_coder_output(raw)
            except Exception as exc:  # noqa: BLE001 - an unparseable verdict rejects, never crashes
                # (a) Only a reply that failed to parse as a verdict is
                # eligible for the banner scan. A parsed verdict is a verdict,
                # however cap-themed the cluster it adjudicates. And even
                # here the marker only TRIGGERS: a probe decides.
                marker = looks_like_blocking_banner(raw or "")
                if marker is not None and headroom_probe is not None:
                    probe = _probe()
                    if not probe.ok:
                        raise census_headroom.CensusHeadroomExhausted(
                            stage=census_headroom.CensusStage.VERIFY,
                            reason=(
                                "verify reply failed to parse and carries a banner "
                                f"marker {marker!r}; headroom probe confirms no "
                                f"capacity: {probe.reason or 'no headroom'}"
                            ),
                            verified=len(verified),
                            rejected=len(rejected),
                            unverified=remaining,
                        ) from exc
                    logger.warning(
                        "census: verify reply for cluster %r carries banner marker %r "
                        "but the headroom probe reports capacity available -- "
                        "treating it as an ordinary unparseable verdict",
                        cluster.get("title"),
                        marker,
                    )
                logger.warning(
                    "census: verify failed for cluster %r: %s", cluster.get("title"), exc,
                )
                rejected.append(cluster)
                continue
            try:
                found = verdict_mod.parse_verdict(
                    parsed, title=cluster.get("title") or "", project_root=project_root,
                )
            except verdict_mod.VerdictError as exc:
                logger.warning(
                    "census: verify reply for cluster %r is not a verdict: %s",
                    cluster.get("title"), exc,
                )
                rejected.append(cluster)
                continue
            _log_verdict_normalisations(cluster.get("title"), found)
            if found.verified:
                verified.append(with_verdict(cluster, found))
            else:
                rejected.append(with_verdict(cluster, found))

            # (c) The backstop. Guarded on `remaining > 1` so no probe is
            # spent after the last cluster, where it would guard a stage
            # already over.
            if (
                headroom_probe is not None
                and probe_every > 0
                and (index + 1) % probe_every == 0
                and remaining > 1
            ):
                probe = _probe()
                if not probe.ok:
                    raise census_headroom.CensusHeadroomExhausted(
                        stage=census_headroom.CensusStage.VERIFY,
                        reason=(
                            "headroom backstop probe after "
                            f"{index + 1} cluster(s) reports no capacity: "
                            f"{probe.reason or 'no headroom'}"
                        ),
                        verified=len(verified),
                        rejected=len(rejected),
                        unverified=remaining - 1,
                    )
        return {
            "verified": verified,
            "rejected": rejected,
            "fixed": [],
            "headroom_probes": probes[0],
        }

    return _verify_fn
