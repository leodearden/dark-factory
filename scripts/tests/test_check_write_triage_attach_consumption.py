"""Tests for the write_triage flip gate's item 5 (task 4949).

The unit under test is ``scripts/check_write_triage_attach_consumption.py`` —
the CONSUMPTION probe. Item 1's probe proves the judge path can BIND a verdict
to a determinate candidate; nothing proved anything downstream CONSUMES that
binding, so a change that only widens the parse contract opened the gate while
the attach still landed on the band's top-1. This probe executes
``triage_write`` and asks whether the id the judge designated is the id the
attach ends up on.

The gate-side wiring for item 5 lives with the rest of the gate's end-to-end
suite in ``test_check_write_triage_flip_preconditions.py``; this file is about
the probe alone. They are separate files because that one is already 2300
lines, and a second probe's fixtures and suite would leave a file that no
longer makes internal sense in isolation.

The fixture ``write_triage`` modules are imported from
``write_triage_attach_fixtures`` rather than spelled here, because the gate
suite lays down the same modules into its hermetic repos and two copies would
be two things to keep in sync.

Assertions are on EXIT CODES and on the probe's own structured stdout
vocabulary — never on fixture prose. Pinning prose is the meta-test class this
repo deletes (task 3128 steps 23-25).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
# The repo-wide `--import-mode=importlib` addopts means pytest does NOT put a
# test file's own directory on sys.path, and scripts/tests/conftest.py inserts
# scripts/ but not scripts/tests/. Without this the sibling fixture module —
# which the gate suite imports too, and which is a plain module rather than a
# conftest fixture precisely so BOTH files can reach it — is unimportable.
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from write_triage_attach_fixtures import write_fake_triage  # noqa: E402

_REPO_ROOT = _HERE.parent.parent
_PROBE = _REPO_ROOT / 'scripts' / 'check_write_triage_attach_consumption.py'

# The probe's stable stdout vocabulary, as named constants so the two cannot
# drift apart silently and so it is obvious at a glance that nothing here pins
# a rendered prompt or a fixture's prose.
_PASS = 'PASS  the judge-bound candidate is CONSUMED by the attach'
_NOT_CONSUMED = 'FAIL  the judge-bound candidate is NOT CONSUMED by the attach'
_UNVERIFIABLE = 'UNVERIFIABLE'

#: The hoisted parent the probe's own fixture slate makes the band pick. It
#: belongs to NO candidate in the slate — that is what separates "the attach
#: followed the designation" from "the attach used the band's winner" — so the
#: NOT-CONSUMED report has to name it for an operator to read the finding.
_BAND_CANONICAL = 'parent-1'


def _run_probe(src_root: Path, *, extra_paths: tuple[Path, ...] = ()):
    """Drive the probe directly under this interpreter."""
    argv = [sys.executable, str(_PROBE), '--src-root', str(src_root)]
    for extra in extra_paths:
        argv += ['--extra-path', str(extra)]
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=120,
        env=dict(os.environ),
    )


class TestConsumptionProbe:
    """The probe decides the consumption invariant by EXECUTING the triage
    module it is pointed at, never by inspecting its source text."""

    def test_probe_exists_and_is_executable(self):
        # DeterministicRunner's _default_run_script executes the predicate
        # directly, not via `bash`, and the gate execs this probe the same way,
        # so the +x bit and the shebang are load-bearing.
        assert _PROBE.exists(), f'probe missing at {_PROBE}'
        assert os.access(_PROBE, os.X_OK), f'probe not executable: {_PROBE}'

    def test_band_top1_attach_is_not_consumption(self, tmp_path):
        """main's shape: the attach id is the BAND's top-1 whatever the judge said.

        The verdict has to NAME the measured band canonical. That id belongs to
        no candidate on the slate (it is the hoisted parent), so naming it is
        what tells the operator the attach tracked the band rather than the
        judge — a bare "not consumed" would leave them guessing what it did
        track.
        """
        src_root = write_fake_triage(tmp_path / 'src', variant='band_top1')
        proc = _run_probe(src_root)
        assert proc.returncode != 0, f'{proc.stdout}\n{proc.stderr}'
        assert _NOT_CONSUMED in proc.stdout, proc.stdout
        assert _BAND_CANONICAL in proc.stdout, proc.stdout
        assert _PASS not in proc.stdout, proc.stdout
        assert _UNVERIFIABLE not in proc.stdout, proc.stdout
