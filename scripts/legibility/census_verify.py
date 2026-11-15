"""The census's default per-cluster verifier."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

if (_SHARED_SRC := str(Path(__file__).resolve().parents[2] / "shared" / "src")) not in sys.path:
    sys.path.insert(0, _SHARED_SRC)
if (_ORCH_SRC := str(Path(__file__).resolve().parents[2] / "orchestrator" / "src")) not in sys.path:
    sys.path.insert(0, _ORCH_SRC)

from legibility import census_headroom, census_verdicts, coder
from legibility import verdict as verdict_mod
from orchestrator.agents import code_quality
from shared.cap_markers import looks_like_blocking_banner

logger = logging.getLogger("legibility.census.verify")


_VERIFY_PROBE_EVERY = 5
"""How many clusters the default verifier adjudicates between backstop
headroom probes (``build_verify_fn``'s third, weakest detector).

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
    *project_root* (``census.census_stage_specs`` binds it). That is load-bearing
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


def build_verify_fn(
    project_root: str,
    invoke,
    *,
    headroom_probe=None,
    probe_every: int = _VERIFY_PROBE_EVERY,
):
    """Build the real ``verify_fn(clusters, *, model)`` seam: one Sonnet
    call per cluster via *invoke* (in production the pooled runner's verify
    stage, ``census.census_stage_specs``), parsed via
    ``coder.parse_coder_output`` and then ``verdict.parse_verdict``: verified
    and refuted clusters carry their parsed Verdict under ``"verdict"``.

    Fail-closed per cluster: any per-cluster failure (an invocation error,
    unparseable output, a reply that is no verdict) rejects that cluster
    rather than crashing the census, and no "fixed" entry is ever reported.
    A rejection is evidence against the claim, not a sign the verifier could
    not see the tree, only because ``census.census_stage_specs`` roots the
    verifier's cwd in *project_root*; ``census_run.run_census`` flags the
    all-rejected signature any other infra cause leaves.

    A usage cap is carved out of that default: it is an infra failure, never
    a verdict. With *headroom_probe* supplied, three detectors raise
    ``census_headroom.CensusHeadroomExhausted`` (stage VERIFY; the hitting
    cluster is counted unverified, never rejected):

    (a) a reply that FAILED TO PARSE and carries a blocking-banner marker
        triggers one re-probe, and the probe, never the content, decides. A
        reply that parses into a verdict is never scanned: verify replies
        quote cap-themed clusters, so the loose marker list matches healthy
        content. Every ``REAL_CLI_CAP_MESSAGES`` entry fails to parse
        (``test_every_real_cli_cap_message_fails_to_parse_as_a_verdict``), so
        the split loses no real banner.
    (b) a per-cluster invocation exception triggers one re-probe; headroom
        still available means the cluster rejects as before.
    (c) every *probe_every* clusters, a backstop probe, never after the last
        cluster. It is the weakest detector -- up to ``probe_every - 1``
        clusters may reject before it notices -- and covers the one case (a)
        and (b) cannot see: a cap presenting as a clean-but-wrong verdict.

    All three are no-ops without a *headroom_probe*. Every probe goes through
    one wrapper that counts it (the result's ``headroom_probes``, which
    ``census_run.adjudicate`` folds into the report's cost note) and never
    raises: a raising seam reads as ``HeadroomResult(ok=False)``, a deferral
    rather than an exception escaping into ``census.main()``'s exit-1 path.
    """
    quality_block = code_quality.guidance()

    def _verify_fn(clusters, *, model):
        verified, rejected = [], []
        # Every probe THIS verifier spends, reported back in the result dict
        # so the report's cost note can state the run's real probe spend.
        # It rides the return value rather than a mutable counter passed in
        # because the probe seam is wired by census.main(), not by
        # census_run.run_census: a
        # counter would need a second wiring site that nothing forces anyone
        # to connect, and an unwired counter is exactly how the cost note
        # came to under-report in the first place.
        probes = [0]

        def _probe():
            """Call the injected probe seam: counted, and never raising.

            *headroom_probe* is an injected seam with no exception contract.
            Production wires ``census_headroom.preflight_headroom``, which
            folds any failure into ``HeadroomResult(ok=False)`` -- but a seam
            that RAISES here would escape ``_verify_fn``, sail past
            ``census_run.adjudicate``'s ``except CensusHeadroomExhausted`` and
            land in ``census.main()``'s hard-failure path: exit 1 plus a
            traceback escalation, after the whole mining spend, on a path that
            has ALREADY failed once. Matching ``preflight_headroom``'s own
            fail-safe stance turns that
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
            # clusters offered (census_headroom.CensusHeadroomExhausted's INVARIANT).
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
            census_verdicts.log_normalisations(cluster.get("title"), found)
            if found.verified:
                verified.append(census_verdicts.with_verdict(cluster, found))
            else:
                rejected.append(census_verdicts.with_verdict(cluster, found))

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
