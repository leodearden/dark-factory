"""A census cluster carrying the Verdict its verifier reached."""
from __future__ import annotations

import logging

from legibility import verdict as verdict_mod

logger = logging.getLogger("legibility.census.verdicts")


def with_verdict(cluster: dict, found: verdict_mod.Verdict) -> dict:
    """*cluster* carrying its Verdict, its remediation exactly the Verdict's
    in-tree one when the claim is verified: a refuted claim, or a Verdict
    without a remediation, carries none whatever the cluster arrived with."""
    unremediated = {key: value for key, value in cluster.items() if key != "remediation"}
    carried = {**unremediated, "verdict": found}
    if found.verified and found.remediation is not None:
        return {**carried, "remediation": found.remediation.to_record()}
    return carried


def log_normalisations(title, found: verdict_mod.Verdict) -> None:
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
    log_normalisations(cluster.get("title"), defaulted)
    return defaulted


def verified_with_verdicts(verified: list, *, project_root: str) -> list[dict]:
    """*verified*, every cluster carrying a Verdict (``_verdict_of``) and
    that Verdict's remediation (``with_verdict``)."""
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
