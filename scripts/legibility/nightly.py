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
from pathlib import Path

# Self-bootstrap for standalone `python scripts/legibility/nightly.py` runs
# (and the systemd ExecStart, which invokes this file directly) -- must run
# BEFORE the `legibility.*` imports below, since a direct script invocation
# puts only scripts/legibility/ (not scripts/) on sys.path. Skipped under
# pytest/normal package import: __name__ is 'legibility.nightly'. Mirrors
# sampling.py/census_trigger.py's identical guard.
if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from legibility.config import load_config  # noqa: E402

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


if __name__ == '__main__':
    raise SystemExit(0)
