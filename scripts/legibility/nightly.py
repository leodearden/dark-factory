#!/usr/bin/env python3
"""scripts/legibility/nightly.py — nightly trickle pipeline assembly (PRD task ε).

Assembles the per-project nightly trickle: inventory+sample (β) -> digest
(α) -> code (δ) -> merge (γ) -> docs-only commit -> census trigger (ζ). See
plans/confusion-reduction-prd.md §5.5 (pipeline), §7.4 (per-project config),
decisions 7/8 (fail-loud contract, liveness probes git history never),
boundary test §8.8.

Every stage is reached behind a dependency-injection seam (``invoke`` for
the LLM, ``status_fetcher`` for the census, ``poster`` for escalation,
``committer`` for git) plus module-level functions a caller can monkeypatch
-- mirrors the established seam convention (coder.py's ``invoke`` override,
census_trigger.py's injected ``status_fetcher``). This is what the
systemd ``legibility-trickle@.service`` template runs nightly, and what
``install-trickle-timer.sh``/``check_trickle_liveness.sh`` install and probe.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

# Self-bootstrap for standalone `python scripts/legibility/nightly.py` runs
# (and the systemd ExecStart, which invokes this file directly) -- must run
# BEFORE the `legibility.*` imports below, since a direct script invocation
# puts only scripts/legibility/ (not scripts/) on sys.path. Skipped under
# pytest/normal package import: __name__ is 'legibility.nightly'. Mirrors
# sampling.py/census_trigger.py's identical guard.
if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from legibility import digest, inventory, sampling  # noqa: E402
from legibility.config import LegibilityConfig, load_config  # noqa: E402

# ---------------------------------------------------------------------------
# resolve_config_path — map a bare project_id to its legibility.yaml
# ---------------------------------------------------------------------------

def _default_search_roots() -> list[Path]:
    """Default search roots for :func:`resolve_config_path`.

    Env ``LEGIBILITY_SEARCH_ROOTS`` (``os.pathsep``-split) if set, else the
    dark-factory repo's own parent directory -- mirrors
    ``skills/factory-init/scripts/find_escalation_port.known_project_roots``'s
    sibling-repos-under-``/home/leo/src`` convention. Each returned path is a
    PARENT directory: :func:`resolve_config_path` globs one level down for
    candidate project roots, it is not itself a project root.
    """
    env_value = os.environ.get('LEGIBILITY_SEARCH_ROOTS')
    if env_value:
        return [Path(p) for p in env_value.split(os.pathsep) if p]
    repo_root = Path(__file__).resolve().parents[2]
    return [repo_root.parent]


def resolve_config_path(
    project_id: str, search_roots: Sequence[str | Path] | None = None,
) -> Path:
    """Resolve *project_id* to its ``docs/legibility/legibility.yaml`` path.

    *search_roots* (default: :func:`_default_search_roots`) is a list of
    PARENT directories; each is globbed one level down for candidate
    project-root directories, and each candidate's
    ``docs/legibility/legibility.yaml`` (if present and loadable) is matched
    against its own authoritative ``project_id`` field via
    :func:`legibility.config.load_config` -- never a directory-name guess.
    A candidate whose config fails to load (malformed YAML, schema error) is
    skipped rather than aborting the whole search. Raises
    ``FileNotFoundError`` if no candidate matches.
    """
    roots = (
        [Path(r) for r in search_roots] if search_roots is not None
        else _default_search_roots()
    )
    for root in roots:
        if not root.is_dir():
            continue
        for candidate_root in sorted(root.iterdir()):
            if not candidate_root.is_dir():
                continue
            config_path = candidate_root / 'docs' / 'legibility' / 'legibility.yaml'
            if not config_path.is_file():
                continue
            try:
                cfg = load_config(config_path)
            except Exception:
                continue
            if cfg.project_id == project_id:
                return config_path
    raise FileNotFoundError(
        f'no legibility.yaml found for project_id={project_id!r} under '
        f'search roots {[str(r) for r in roots]!r}'
    )


# ---------------------------------------------------------------------------
# select_scored_records / select_digest_sessions — inventory -> score ->
# classify -> stratified sample
# ---------------------------------------------------------------------------

def select_scored_records(
    cfg: LegibilityConfig, projects_root: Path | str, target_date: date,
) -> list[sampling.ScoredRecord]:
    """Enumerate *target_date*'s sessions for *cfg* and assemble a
    :class:`~legibility.sampling.ScoredRecord` per session.

    Reuses ``inventory.enumerate_sessions`` plus sampling's own private
    one-pass helpers (``_score_and_find_first_turn`` /
    ``_first_user_turn_text``) -- the EXACT loop ``sampling.main`` uses --
    rather than duplicating the score+first-turn pass or adding a new public
    function to the already-landed β module.
    """
    sessions = inventory.enumerate_sessions(projects_root, cfg.cwd_prefixes, target_date)

    scored: list[sampling.ScoredRecord] = []
    for session in sessions:
        counts, first_turn = sampling._score_and_find_first_turn(session.path)
        stratum = sampling.classify_agent_class(first_turn, session.path)
        scored.append(
            sampling.ScoredRecord(
                session=session,
                stratum=stratum,
                counts=counts,
                first_turn_text=sampling._first_user_turn_text(first_turn),
            )
        )
    return scored


def select_digest_sessions(
    cfg: LegibilityConfig, projects_root: Path | str, target_date: date,
) -> list[sampling.ScoredRecord]:
    """The budget-bounded, stratified subset of *target_date*'s sessions to
    digest -- :func:`select_scored_records` narrowed by
    ``sampling.stratified_sample``."""
    scored = select_scored_records(cfg, projects_root, target_date)
    return sampling.stratified_sample(scored, cfg).selected


# ---------------------------------------------------------------------------
# build_digests — render one digest per selected session, isolating crashes
# ---------------------------------------------------------------------------

DEFAULT_MAX_DIGEST_BYTES = 15360


def build_digests(
    selected: Sequence[sampling.ScoredRecord],
    *,
    max_bytes: int = DEFAULT_MAX_DIGEST_BYTES,
    build=digest.build_digest,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Render one confusion digest per *selected* record via *build*
    (default :func:`legibility.digest.build_digest`), passing beta's already
    -authoritative ``rec.stratum`` as ``agent_class_override`` -- alpha never
    re-guesses when the caller already knows.

    Any exception raised by *build* for a given record is isolated: it is
    captured as ``(session_basename, reason)`` in the returned
    ``extractor_failures`` list rather than propagated or fabricated into a
    placeholder digest, so a driving caller (:func:`run_nightly`) can treat
    a non-empty ``extractor_failures`` as the extractor-crash fail-loud
    trigger (PRD decision 8).
    """
    digests: list[str] = []
    extractor_failures: list[tuple[str, str]] = []

    for record in selected:
        try:
            rendered = build(
                record.path, agent_class_override=record.stratum, max_bytes=max_bytes,
            )
        except Exception as exc:  # noqa: BLE001 - isolate, never propagate/fabricate
            extractor_failures.append((record.path.name, str(exc)))
            continue
        digests.append(rendered)

    return digests, extractor_failures


if __name__ == '__main__':
    raise SystemExit(0)
