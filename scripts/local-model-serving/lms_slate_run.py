"""Run the whole arm slate, one arm at a time, in a transient systemd --user unit.

PRD-MARKER:local-memory-models-eval serving

Task 4301 of `plans/local-memory-models-eval-prd.md`.

WHAT THIS IS.  The committed, reproducible driver for a full slate run: it
sweeps every arm the manifest commissions, writes one per-arm report part,
and then asks `lms_healthcheck --merge` to assemble
`verification/health-report.json` from them.

WHY IT EXISTS.  The 2026-08-06 slate run was driven BY HAND -- a person typed
the `lms_ctl` and `lms_healthcheck` invocations one arm at a time.  So no
compliant invocation for that run exists anywhere in the repo, and every
future re-measure re-derives the hazard-compliant form from scratch.  A
re-derivation is where `--collect` and `--working-directory` get dropped, and
where an operator reaches for `nohup ... &` because it is shorter.  A script
makes those errors impossible rather than merely fixed.

PRD hazard 11 is binding: "long runs in transient `systemd --user` units,
never bare background shells".  A full slate is ~30 minutes of model loads;
through a background shell it is unsupervised, unloggable, and dies with the
invoking session -- losing every arm measured so far.

TWO LAYERS, and the split is the point:

  * SUBMIT (the default).  `slate_argv` is a PURE function returning the
    transient-unit argv; `_submit` prints it and runs it.  Being pure, the
    compliant form is fully assertable offline -- the one property a live
    30-minute run can never conveniently prove.
  * IN-UNIT (`--in-unit`).  `run_slate` performs the sweep.  The flag is also
    the guard that stops a unit from recursively re-submitting itself.

The driver SHELLS OUT to `lms_ctl.py` and `lms_healthcheck.py` rather than
importing them.  Three reasons.  (1) `lms_ctl.start`'s LIBRARY default is
`exclusive=False` while its CLI default is `exclusive=True`; an import-based
driver one forgotten keyword from starting a second arm on a card that fits
one is exactly the hazard `start` exists to refuse.  Shelling out inherits the
safe default by construction.  (2) The driver then runs the same commands the
README documents an operator running by hand, so the script and the prose
cannot describe two different procedures.  (3) A wedged HTTP client or a
180s completion timeout stays in its own process instead of hanging the sweep.
"""
from __future__ import annotations

import sys
from pathlib import Path

import lms_fetch_weights
from lms_serve import REPO_ROOT

#: The transient unit the whole sweep runs in.  Quoted verbatim by the
#: README's `journalctl --user -u lms-slate-run -f` follow line, so a drift
#: here sends an operator to a unit that does not exist.
SLATE_UNIT_NAME = 'lms-slate-run'

#: This file, absolute.  The payload re-invokes it by absolute path: the unit
#: runs with a minimal PATH and none of the caller's venv, so neither a
#: relative path nor a bare module name resolves inside it.
MODULE_PATH = Path(__file__).resolve()

#: The slate artifact.  Written by `lms_healthcheck --merge`, never by this
#: module -- see the merge step for why that separation is load-bearing.
DEFAULT_ARTIFACT = (
    REPO_ROOT / 'scripts' / 'local-model-serving' / 'verification' / 'health-report.json'
)


def slate_argv(
    parts_dir: str | Path,
    output: str | Path,
    *,
    ready_timeout: float | None = None,
    force: bool = False,
    env: dict[str, str] | None = None,
) -> list[str]:
    """The transient-unit argv that runs the whole slate.

    Paths are resolved to ABSOLUTE here, in the submit layer, and passed
    explicitly in the payload.  Deriving them again inside the unit is how the
    two layers end up reading different directories and a resume silently does
    nothing: `systemd --user` propagates no caller environment, so an
    `XDG_RUNTIME_DIR`-derived path is not the same path on both sides.

    *env* is the caller environment the unit's `--setenv=` allowlist is read
    from, injectable so the allowlist is assertable offline.  It is accepted
    but not yet read; the allowlist itself lands in the next commit.
    """
    payload = [
        sys.executable, str(MODULE_PATH),
        '--in-unit',
        '--parts-dir', str(Path(parts_dir).resolve()),
        '--output', str(Path(output).resolve()),
    ]
    if ready_timeout is not None:
        payload += ['--ready-timeout', str(ready_timeout)]
    if force:
        payload.append('--force')

    return lms_fetch_weights.transient_unit_prefix(SLATE_UNIT_NAME, []) + payload
