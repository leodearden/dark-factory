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

import pytest

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
#: Named on a PASS, so an operator can see WHICH wire shape the module speaks
#: rather than only that some shape worked.
_CHANNEL = 'designation channel'
#: The control spelling. It designates nothing, so it must never satisfy on its
#: own — a module that widened nothing would otherwise open the gate.
_BARE_STR_SPELLING = 'bare outcome str'
#: The finding a hard-coded attach position earns. It has to be distinguishable
#: from "no designation channel exists": the remedies are different.
_DID_NOT_TRACK = 'did not track the designated candidate'
#: A run in which the module RECORDED a fail-open is not evidence of
#: consumption — it is evidence the designation was swallowed. Detected
#: structurally, from the counter, and reported with its count.
_FAIL_OPEN = 'FAIL-OPEN'
_ONE_FAIL_OPEN = 'FAIL-OPEN (1 recorded)'
#: The SECOND way item 5's invariant can hold: under an option-(b)-shaped
#: remedy the CALLER picks the attach target and tells the judge, so the judge
#: names no candidate back and the judge-side swap cannot hold. Reported by
#: name on every run, satisfied or not, so a reader can see the branch was
#: evaluated rather than skipped.
_ANNOUNCED_BRANCH = 'announced-target branch'
#: The finding an announcement the write ignores earns. Distinguishable from
#: "the judge was told nothing": one remedy is to honour the announcement, the
#: other is to make one.
_ANNOUNCEMENT_IGNORED = 'the attach did not land on the announced target'

#: The option-(a) wire shapes a fix might land. None may be pinned by the
#: probe: option (a) has not landed, so a probe that required one spelling
#: would fail a correct fix that chose another.
_DESIGNATING_VARIANTS = (
    'consumes_designated_id',
    'consumes_designated_dict',
    'consumes_designated_object',
)

#: Emitted on the PASS path only — the run that authorises a production flag
#: flip — and saying what the PASS does NOT cover: the probe stops at
#: BandDecision.canonical_id and never executes tools.py::add_memory's stamp.
_SCOPE_NOTE = 'NOTE this gate asserts at BandDecision.canonical_id'
#: The probe's own degraded-measurement records. They go on STDOUT (the gate
#: drops stderr) and LAST (a tail-truncated report keeps its end).
_WARN = 'WARN'

#: Every way the probe can be pointed at a tree it cannot decide the invariant
#: on. Each must land on UNVERIFIABLE, never on a verdict and never on silence.
_UNVERIFIABLE_VARIANTS = (
    # No importable module at all.
    'missing',
    # Importable, but the write path cannot be executed.
    'raises_on_triage_write',
    # SystemExit is not an Exception; unguarded it exits the interpreter 0
    # having printed nothing, and a gate grepping for a marker reads that as a
    # PASS. Measured, not hypothetical.
    'exits_during_import',
    # Calling it returns a BandDecision rather than something to await, so
    # nothing measured came from executing the write path.
    'not_awaitable',
    # Reaches the judge and then returns a shape with no canonical_id, so every
    # attach id read is None — which is not the band top-1 either.
    'returns_non_decision',
)

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

    @pytest.mark.parametrize('variant', _DESIGNATING_VARIANTS)
    def test_a_module_that_consumes_the_designation_passes(self, tmp_path, variant):
        """Any of the three plausible option-(a) wire shapes satisfies item 5.

        Which spelling a fix picks is a MECHANISM. Pinning one would fail a
        correct fix that chose another — the false-FAIL class this gate family
        was rewritten to remove — so all three are accepted and the report says
        which one was found.
        """
        src_root = write_fake_triage(tmp_path / 'src', variant=variant)
        proc = _run_probe(src_root)
        assert proc.returncode == 0, f'{proc.stdout}\n{proc.stderr}'
        assert _PASS in proc.stdout, proc.stdout
        assert _CHANNEL in proc.stdout, proc.stdout

    def test_the_bare_str_spelling_alone_never_satisfies(self, tmp_path):
        """The control: widening nothing must not open the gate.

        A judge that returns a plain outcome word designates no candidate, so a
        module that still attaches to `decision.canonical_id` has changed
        nothing. The spelling is tried anyway — and reported as tried — because
        a control nobody exercises proves nothing.
        """
        src_root = write_fake_triage(tmp_path / 'src', variant='band_top1')
        proc = _run_probe(src_root)
        assert proc.returncode != 0, f'{proc.stdout}\n{proc.stderr}'
        assert _PASS not in proc.stdout, proc.stdout
        assert _BARE_STR_SPELLING in proc.stdout, proc.stdout

    def test_a_hardcoded_attach_position_is_not_consumption(self, tmp_path):
        """The positional bug, relocated to the consumption side.

        This variant DECODES the designation and then attaches to a fixed slot,
        so its attach id is neither the band's canonical nor anything the judge
        said. A single-shot "is the attach id different from the band
        canonical?" check blesses it — the same `candidates[0]` class item 1
        already had to defeat. Only requiring the attach to TRACK the
        designation rejects it, and the report must say so: "did not track" and
        "no designation channel exists" point at different remedies.
        """
        src_root = write_fake_triage(
            tmp_path / 'src', variant='hardcodes_last_candidate',
        )
        proc = _run_probe(src_root)
        assert proc.returncode != 0, f'{proc.stdout}\n{proc.stderr}'
        assert _PASS not in proc.stdout, proc.stdout
        assert _DID_NOT_TRACK in proc.stdout, proc.stdout

    def test_the_tracking_requirement_does_not_fail_a_correct_fix(self, tmp_path):
        """The converse control for the test above.

        A requirement that rejects the positional bug is worthless if it also
        rejects a module that genuinely threads the designation — that would
        re-block task 3169 against a correct fix, which is the false-FAIL class
        this gate family exists to remove.
        """
        src_root = write_fake_triage(
            tmp_path / 'src', variant='consumes_designated_id',
        )
        proc = _run_probe(src_root)
        assert proc.returncode == 0, f'{proc.stdout}\n{proc.stderr}'
        assert _PASS in proc.stdout, proc.stdout

    def test_a_fail_open_is_not_evidence_of_consumption(self, tmp_path):
        """The catastrophic false pass.

        This is what main does TODAY with a designating verdict: the payload is
        not in TRIAGE_OUTCOMES, a fail-open is recorded, and the write returns
        BandDecision(stored, None, ...). That canonical_id of None is not the
        band's top-1 either, so a naive "did the attach avoid the band
        canonical?" check reads it as CONSUMED — and authorises the production
        write_triage.enabled flip on a codebase where nothing changed at all.

        The diagnosis matters as much as the verdict: "the designation was
        swallowed" and "no designation channel exists" send an operator to
        different remedies, so the report must attribute this to the fail-open
        and name the count it measured.
        """
        src_root = write_fake_triage(
            tmp_path / 'src', variant='fail_opens_on_designation',
        )
        proc = _run_probe(src_root)
        assert proc.returncode != 0, f'{proc.stdout}\n{proc.stderr}'
        assert _PASS not in proc.stdout, proc.stdout
        assert _FAIL_OPEN in proc.stdout, proc.stdout
        assert _ONE_FAIL_OPEN in proc.stdout, proc.stdout

    def test_an_honoured_announcement_satisfies_item_5(self, tmp_path):
        """The option-(b)-shaped remedy, and why item 5 may not require the swap.

        Under option (b) the CALLER picks the attach target and announces it to
        the judge; the judge names no candidate back, so the judge-side swap
        cannot hold however correct the module is. Requiring the swap would
        therefore fail a correct fix and re-block task 3169 — the false-FAIL
        class this gate family was rewritten to remove. What item 5 asserts is
        the INVARIANT, not which remedy landed, so either branch satisfies it
        and the report says which one did.
        """
        src_root = write_fake_triage(
            tmp_path / 'src', variant='announces_attach_target',
        )
        proc = _run_probe(src_root)
        assert proc.returncode == 0, f'{proc.stdout}\n{proc.stderr}'
        assert _PASS in proc.stdout, proc.stdout
        assert _ANNOUNCED_BRANCH in proc.stdout, proc.stdout

    def test_the_announced_target_branch_is_not_vacuous(self, tmp_path):
        """main announces nothing, so the branch must not hold for main.

        main already hands the judge `decision`, whose `canonical_id` is the
        very id it already attaches to. A branch that read the announcement out
        of any of today's five kwargs would hold on a codebase where nothing
        changed at all — the same catastrophic false pass the fail-open control
        guards, reached by a different route. So the branch is asserted to be
        EVALUATED here (it is named) and to be UNSATISFIED.
        """
        src_root = write_fake_triage(tmp_path / 'src', variant='band_top1')
        proc = _run_probe(src_root)
        assert proc.returncode != 0, f'{proc.stdout}\n{proc.stderr}'
        assert _PASS not in proc.stdout, proc.stdout
        assert _ANNOUNCED_BRANCH in proc.stdout, proc.stdout

    def test_an_announcement_the_attach_ignores_is_not_consumption(self, tmp_path):
        """Announcing a target is not the same as attaching to it.

        This variant names one candidate in the prompt and files the verdict
        against the band's top-1 anyway — the option-(b)-shaped form of the
        defect item 5 exists to detect. Without it, "an announcement exists"
        would be indistinguishable from "the announcement was consumed", and
        the branch would bless the defect it was added to catch.
        """
        src_root = write_fake_triage(
            tmp_path / 'src', variant='announces_but_attaches_elsewhere',
        )
        proc = _run_probe(src_root)
        assert proc.returncode != 0, f'{proc.stdout}\n{proc.stderr}'
        assert _PASS not in proc.stdout, proc.stdout
        assert _ANNOUNCEMENT_IGNORED in proc.stdout, proc.stdout


class TestFailsClosed:
    """An unverifiable invariant is not a satisfied one.

    Item 5's PASS authorises flipping `write_triage.enabled` in production, so
    every way of not knowing has to be reported as not knowing. The failure
    mode that matters most is not a wrong verdict but SILENCE: the gate greps
    stdout for a marker, so a probe that dies without printing is read as
    whatever the shell's exit code says.
    """

    @pytest.mark.parametrize('variant', _UNVERIFIABLE_VARIANTS)
    def test_an_undecidable_ref_is_unverifiable_never_a_pass(self, tmp_path, variant):
        src_root = write_fake_triage(tmp_path / 'src', variant=variant)
        proc = _run_probe(src_root)
        assert proc.returncode != 0, f'{proc.stdout}\n{proc.stderr}'
        assert _UNVERIFIABLE in proc.stdout, proc.stdout
        assert _PASS not in proc.stdout, proc.stdout

    def test_a_sibling_root_that_merely_extends_the_name_is_not_inside_it(
        self, tmp_path,
    ):
        """Containment is decided on PATH COMPONENTS, not on characters.

        `str(origin).startswith(str(src_root))` also accepts `<root>-installed`,
        so the probe would report on a module the ref never shipped — and the
        gate would attribute the verdict to the ref it named. The extra path is
        the realistic route in: the gate passes one for the ref's `shared/src`.
        """
        src_root = write_fake_triage(tmp_path / 'src', variant='missing')
        sibling = write_fake_triage(
            tmp_path / 'src-installed', variant='consumes_designated_id',
        )
        proc = _run_probe(src_root, extra_paths=(sibling,))
        assert proc.returncode != 0, f'{proc.stdout}\n{proc.stderr}'
        assert _UNVERIFIABLE in proc.stdout, proc.stdout
        assert _PASS not in proc.stdout, proc.stdout

    def test_a_pass_carries_the_scope_note(self, tmp_path):
        """What the PASS does not cover, said on the run that acts on it.

        The probe stops at `BandDecision.canonical_id` — the value
        tools.py::add_memory consumes verbatim as `attached_to` — and does not
        execute the stamp, so a future edit to that function's target selection
        would pass this gate. An operator about to flip the flag is entitled to
        read that from the report rather than from the plan.

        On the PASS path only: a FAIL's report window belongs to the remedy.
        """
        src_root = write_fake_triage(
            tmp_path / 'src', variant='consumes_designated_id',
        )
        proc = _run_probe(src_root)
        assert proc.returncode == 0, f'{proc.stdout}\n{proc.stderr}'
        assert _SCOPE_NOTE in proc.stdout, proc.stdout

    def test_a_degraded_measurement_warns_on_stdout_and_last(self, tmp_path):
        """A ref whose fail-open counter is gone is still decidable — loudly.

        The probe falls back to a counting stand-in, so the invariant is still
        measured and this ref PASSes. But what it measured is no longer the
        ref's own accounting, and that has to reach the operator: on STDOUT
        because the gate drops stderr, and LAST because a report read through a
        tail keeps its end.
        """
        src_root = write_fake_triage(
            tmp_path / 'src', variant='counter_class_missing',
        )
        proc = _run_probe(src_root)
        assert proc.returncode == 0, f'{proc.stdout}\n{proc.stderr}'
        assert _PASS in proc.stdout, proc.stdout
        lines = [line for line in proc.stdout.splitlines() if line.strip()]
        assert _WARN in lines[-1], proc.stdout
