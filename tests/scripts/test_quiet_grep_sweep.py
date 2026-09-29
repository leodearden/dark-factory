"""The ONE quiet-grep sweep: no swept script pipes a producer into `grep -q`.

`producer | grep -q PAT` reports the PRODUCER's exit status under
`set -o pipefail`, not grep's verdict, so an `if` guarding on it can take the
else branch on output that plainly CONTAINS the pattern — either because the
producer exits non-zero after printing the match, or because `grep -q` closes
the pipe on its first match and the still-writing producer dies of SIGPIPE.

The behavioural suites (test_script_probe_pipelines.py,
test_setup_host_probe_pipelines.py) pin what each fixed site DOES. This module
pins that the defective CONSTRUCT does not come back, in every script listed in
`_SWEPT_SCRIPTS`. Its detector is the shared `grep_q_offenders` in
tests/scripts/shell_sections.py, guarded by
`test_the_grep_q_sweep_detects_a_planted_pipeline` below.
"""

from __future__ import annotations

import pytest
from shell_sections import REPO_ROOT, grep_q_offenders, sets_pipefail

# Every script in which the construct is forbidden, listed ONCE.
# check_write_triage_flip_preconditions.sh and memory-metadata-coverage-census.sh
# were already clean and are only READ here, which pins them without editing them.
_SWEPT_SCRIPTS = tuple(
    REPO_ROOT / "scripts" / name
    for name in (
        "setup-host.sh",
        "export-data.sh",
        "import-data.sh",
        "deploy-w5-recon-reliability.sh",
        "check_write_triage_flip_preconditions.sh",
        "memory-metadata-coverage-census.sh",
        "verify-migration.sh",
        "run_eval_matrix.sh",
    )
)


@pytest.mark.parametrize("script", _SWEPT_SCRIPTS, ids=lambda p: p.name)
def test_never_pipes_a_producer_into_grep_q(script):
    """No code line in a swept script may decide anything through `producer | grep --quiet PAT`.

    WHAT THE RULE DOES NOT FORBID. It is scoped to greps that EXIT ON FIRST
    MATCH — every spelling of that, short cluster or long `--quiet`/`--silent`,
    since they share one defect. A `| grep -F ... || true` inside a command
    substitution is a different, safe shape: a non-quiet grep drains its input
    rather than SIGPIPE-ing the producer. Neither is a `grep -q` reading a
    FILE swept in: with no producer upstream there is nothing for `pipefail`
    to conflate.

    And it mandates NO replacement spelling. These scripts happen to use
    `[[ ]]`, but `case` or a `<<<` here-string remain open to a future author
    — this forbids one known-defective construct, nothing more.
    """
    source = script.read_text(encoding="utf-8")

    # FIRST, because it is what makes the rule load-bearing: without pipefail
    # there is no defect here and the sweep below would be guarding nothing.
    assert sets_pipefail(source), (
        f"{script.name} no longer sets `-o pipefail`, so this sweep would pass "
        f"vacuously. Either restore it or retire this test deliberately."
    )

    offenders = grep_q_offenders(source)
    assert not offenders, f"producer piped into `grep -q` in {script.name}:\n" + "\n".join(
        f"  line {n}: {line.strip()}" for n, line in offenders
    )


def test_the_grep_q_sweep_detects_a_planted_pipeline():
    """Guard the guard: a detector that stops matching makes the sweep vacuous.

    Same discipline tests/scripts/test_check_dashboard_unit_parity.py::
    test_the_sweep_finds_every_known_parity_call_site applies to its own sweep.
    Passes on arrival — it pins the mechanism, not the product behaviour.

    Guards the SHARED detector in tests/scripts/shell_sections.py — both
    `grep_q_offenders` and the `sets_pipefail` precondition — on behalf of the
    one sweep directly above.
    """
    planted = (
        "if foo | grep -q BAR; then\n"
        "if foo | grep -qF BAR; then\n"
        "if foo | grep -Fq BAR; then\n"
        "if foo | grep -i -q BAR; then\n"
        # The long forms. `grep --quiet` reintroduces the sweep's exact defect
        # and reads as innocuous, so it is pinned by the same mechanism as the
        # short flags rather than left to a docstring claim.
        "if foo | grep --quiet BAR; then\n"
        "if foo | grep --silent BAR; then\n"
        # A flag carrying an argument in between must not hide the quiet one.
        "if foo | grep -e BAR --quiet; then\n"
    )
    assert len(grep_q_offenders(planted)) == 7, grep_q_offenders(planted)

    # A comment describing the construct is not the construct.
    assert grep_q_offenders("  # never write `foo | grep -q BAR` here\n") == []
    # Nor is a non-quiet grep, which drains its input instead of closing it.
    assert grep_q_offenders("out=\"$(foo | grep -F 'tag' || true)\"\n") == []
    # Nor is a `grep -q` over a FILE: no producer upstream, nothing to conflate.
    assert grep_q_offenders("if grep -q '^\\[Install\\]' \"$unit\"; then\n") == []
    # And a `-q` belonging to a LATER command on the line is not this grep's.
    assert grep_q_offenders("if foo | grep -F BAR; then bar -q; fi\n") == []
    # Nor is the trailing bar of an OR operator a pipe. `cmd || grep -q pat f`
    # runs grep over a FILE only when cmd failed: no pipeline, no producer, and
    # nothing for `pipefail` to conflate. None of the swept scripts writes this
    # today, so without a case here the false positive stays invisible until it
    # fails a future author's legitimate line.
    assert grep_q_offenders('cmd || grep -q pat "$f"\n') == []

    # The precondition accepts every spelling the swept scripts use...
    for spelling in (
        "set -euo pipefail\n",
        "set -uo pipefail\n",
        "set -o pipefail\n",
        "  set -uo pipefail\n",
    ):
        assert sets_pipefail(spelling), spelling
    # ...and neither a comment quoting it nor a `set` without it.
    assert not sets_pipefail("# never drop set -o pipefail\n")
    assert not sets_pipefail("set -eu\n")
