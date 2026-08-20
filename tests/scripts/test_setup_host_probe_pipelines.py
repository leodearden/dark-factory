"""Behavioural coverage for setup-host.sh's `producer | grep -q PAT` probes.

WHY THESE EXIST. `producer | grep -q PAT` reports the PRODUCER's exit status
under `set -o pipefail`, not grep's verdict, so an `if` guarding on it can take
the else branch on output that plainly CONTAINS the pattern. Two ways that
happens, both covered below per site:

  (a) the producer emits the match and then exits non-zero for its own reasons
      (a probe whose status reflects the whole run, not the one line asked
      about) — `pipefail` hands the `if` that non-zero and the match is lost;
  (b) SIGPIPE — `grep -q` exits the instant it matches and closes the read end,
      so a producer still writing dies of signal 13, the pipeline returns 141,
      and the `if` again reads "no match" from a matching reply.

Both are read through the REAL shipped text: each test slices its section out
of scripts/setup-host.sh by code anchors and runs it against PATH stubs, so a
test asserts on what the script DOES, never on how the fix is spelled.

The companion source-level sweep at the bottom forbids the construct itself.
"""

from __future__ import annotations

import os
import re

from setup_host_sections import (
    run_section,
    setup_host_text,
    slice_section,
    stub_bin_dir,
    write_stub,
)

# Trailing bytes a producer writes AFTER the matching line, to provoke (b).
#
# MEASURED, not chosen for roundness. Reproduction rate of the misread, 30
# trials per size against this exact stub shape:
#
#     65536 -> 26/30      131072 -> 30/30      262144 -> 30/30      1MiB -> 30/30
#
# Whether the producer is scheduled to write again BEFORE grep closes the read
# end is a race, so near the 64KiB pipe buffer the defect is intermittent — at
# 65536 the reply is small enough to sometimes land whole. 262144 is the first
# round size measured deterministic with margin. Do NOT lower this: a value in
# the flaky band leaves the SIGPIPE tests only probabilistically able to catch a
# reintroduced pipeline, and one below the buffer cannot catch it at all.
# (Consistent with the ~82KB flip point recorded at setup-host.sh's
# orchestrator gate — that gate's producer writes in one burst, this one does
# not, so the two thresholds are close but not the same number.)
#
# Note this only affects how reliably the tests are RED against the OLD form:
# the fixed form drains the producer through a command substitution, where
# there is no pipe to signal, so the tests below are deterministic once fixed.
BULK_BYTES = 262144

# --- section 2: the FalkorDB "wait for healthy" loop -----------------------
# Both anchors are CODE (not comment prose), are unique in the file, and
# survive the fix, which matters because the same anchors must slice the
# unfixed text and the fixed text.
_SECTION_2_START = 'docker compose -f "$COMPOSE_FILE" up -d falkordb qdrant'
_SECTION_2_END = "\ndone\n"


def _stub_bin(tmp_path):
    """The stub dir `run_section` will reuse, pre-loaded with a no-op `sleep`.

    `stub_bin_dir` is the harness's own accessor for the directory it prepends
    to PATH, so stubs written here land where `run_section` looks without this
    module re-deriving its path. The `sleep` stub is what keeps the
    30-iteration timeout case instant.
    """
    stub_bin = stub_bin_dir(tmp_path)
    write_stub(stub_bin, "sleep", "exit 0\n")
    return stub_bin


def _dispatch_stub_body(branches):
    """A `case "$*"` body running one of *branches* — (glob, text) pairs.

    `case` is the stub's LAST command, so the taken branch's own status becomes
    the stub's exit status. That is deliberate and load-bearing: the producer's
    status IS the thing these tests are about, and a trailing `exit 0` here
    would swallow it and make every test below vacuously green. The catch-all
    exits 0 so the invocations that are not under test stay silent.
    """
    arms = "".join(f"  {glob})\n{text}    ;;\n" for glob, text in branches)
    return 'case "$*" in\n' + arms + "  *)\n    exit 0\n    ;;\nesac\n"


def _run_probe(tmp_path, section_text, *, stub_name=None, stub_body=None, env_extra=None):
    """Run *section_text* in a tmp tree, with at most one scripted PATH stub.

    One scaffold for every probe site: build the stub dir, write the site's
    stub, hand `run_section` a tmp repo root. Sites differ only in the slice
    they pass, the stub they script, and whether they need $COMPOSE_FILE — so
    a new probe site is a wrapper, not another copy of this.

    *stub_name* None writes no stub at all, which is how the no-`claude` host
    is expressed.
    """
    stub_bin = _stub_bin(tmp_path)
    if stub_name is not None:
        write_stub(stub_bin, stub_name, stub_body)
    repo_root = tmp_path / "repo"
    repo_root.mkdir(exist_ok=True)
    return run_section(
        tmp_path,
        section_text,
        repo_root=repo_root,
        unit_dir=tmp_path / "units",
        env_extra=env_extra,
    )


def _compose_env(tmp_path):
    """The slices read $COMPOSE_FILE and the harness preamble does not set it.

    Under `set -u` an undefined one aborts the section before it probes
    anything. Same channel the existing parity sweeps use for $UV_PATH.
    """
    return {"COMPOSE_FILE": str(tmp_path / "docker-compose.yml")}


# Scenario bodies for the docker exec branch. Indented to sit inside `case`.
_REPLY_THEN_NONZERO = "    printf 'PONG\\n'\n    exit 1\n"
_REPLY_THEN_BULK = f"    printf 'PONG\\n'\n    head -c {BULK_BYTES} /dev/zero | tr '\\0' x\n"
_SILENT_FAILURE = "    exit 1\n"
_CLEAN_REPLY = "    printf 'PONG\\n'\n    exit 0\n"


def _docker_stub_body(exec_body):
    """A `docker` stub body whose `... exec ...` invocation runs *exec_body*.

    The `up -d` invocation falls to the catch-all and exits 0 silently; only
    the exec branch is under test.
    """
    return _dispatch_stub_body((('*" exec "*', exec_body),))


def _run_section_2(tmp_path, exec_body):
    """Slice the section-2 wait loop and run it against a scripted docker."""
    return _run_probe(
        tmp_path,
        slice_section(_SECTION_2_START, _SECTION_2_END),
        stub_name="docker",
        stub_body=_docker_stub_body(exec_body),
        env_extra=_compose_env(tmp_path),
    )


def test_section_2_reports_healthy_when_the_producer_exits_nonzero_after_the_reply(
    tmp_path,
):
    """A ping that ANSWERED PONG is healthy, whatever else the producer's status says.

    `docker compose exec` reports on the exec run as a whole; redis-cli having
    answered is a fact about the OUTPUT. Reading the verdict from the pipeline's
    status conflates the two and reports a live FalkorDB as never healthy.
    """
    result = _run_section_2(tmp_path, _REPLY_THEN_NONZERO)

    combined = result.stdout + result.stderr
    assert "OK FalkorDB healthy" in combined, combined
    assert "FAIL FalkorDB did not become healthy" not in combined, combined


def test_section_2_reports_healthy_when_the_producer_is_sigpiped_after_the_reply(
    tmp_path,
):
    """A producer still writing when grep matches dies of SIGPIPE; PONG was still said.

    `grep -q` closes the read end on its first match, so the producer takes
    signal 13, `pipefail` turns that into 141, and the `if` reads "not healthy"
    off a reply that began with PONG.
    """
    result = _run_section_2(tmp_path, _REPLY_THEN_BULK)

    combined = result.stdout + result.stderr
    assert "OK FalkorDB healthy" in combined, combined
    assert "FAIL FalkorDB did not become healthy" not in combined, combined


def test_section_2_still_times_out_when_the_producer_says_nothing(tmp_path):
    """No reply is still not-healthy — and the section must NOT abort getting there.

    The guard on the fix: a capture written as a bare `out="$(producer)"` makes
    the assignment a simple command, so `set -e` kills the whole bootstrap the
    moment docker is unavailable, where the old pipeline merely took the else
    branch. `returncode == 0` is what pins that difference.
    """
    result = _run_section_2(tmp_path, _SILENT_FAILURE)

    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert "FAIL FalkorDB did not become healthy in 30s" in combined, combined
    assert "OK FalkorDB healthy" not in combined, combined


def test_section_2_reports_healthy_on_a_clean_reply(tmp_path):
    """Characterization: the ordinary path answers PONG and exits 0."""
    result = _run_section_2(tmp_path, _CLEAN_REPLY)

    combined = result.stdout + result.stderr
    assert "OK FalkorDB healthy" in combined, combined
    assert "FAIL FalkorDB did not become healthy" not in combined, combined


# --- section 6: the jcodemunch MCP "already installed?" check --------------
# The slice deliberately starts one block EARLY, at the .jcodemunch.jsonc
# config write. The natural anchor — `if command -v claude &>/dev/null; then` —
# occurs TWICE in this file and `slice_section` takes the first, which is a
# different block entirely; and every narrower anchor is either comment prose
# or opens the slice mid-`if` and yields unbalanced bash. The extra block only
# writes $REPO_ROOT/.jcodemunch.jsonc into the tmp repo root, which is inert.
_JCODEMUNCH_START = 'if [ ! -f "$REPO_ROOT/.jcodemunch.jsonc" ]; then'
_JCODEMUNCH_END = 'ok "jcodemunch MCP added to user config"\n  fi\nfi'

# Printed by the `claude` stub when `mcp add` runs. Telling "already installed"
# from "installed it again" is the whole point: re-adding a server that IS
# registered is the operator-visible harm here, and that re-add runs under
# `set -e`, so a failing one takes the whole bootstrap down.
_ADD_SENTINEL = "STUB-CLAUDE-MCP-ADD-RAN"

_LISTING_NAMES_IT_THEN_NONZERO = (
    "    printf 'jcodemunch: uvx jcodemunch-mcp - Connected\\n'\n    exit 1\n"
)
_LISTING_NAMES_IT_THEN_BULK = (
    "    printf 'jcodemunch: uvx jcodemunch-mcp - Connected\\n'\n"
    f"    head -c {BULK_BYTES} /dev/zero | tr '\\0' x\n"
)
_LISTING_WITHOUT_IT = "    printf 'some-other-server: uvx other - Connected\\n'\n    exit 0\n"
_LISTING_UNREADABLE = "    exit 1\n"


def _run_jcodemunch(tmp_path, list_body):
    """Slice the jcodemunch MCP block and run it against a scripted `claude`."""
    return _run_probe(
        tmp_path,
        slice_section(_JCODEMUNCH_START, _JCODEMUNCH_END),
        stub_name="claude",
        stub_body=_dispatch_stub_body(
            (
                ('*"mcp add"*', f"    printf '{_ADD_SENTINEL}\\n'\n    exit 0\n"),
                ('*"mcp list"*', list_body),
            )
        ),
    )


def test_jcodemunch_sees_an_installed_server_when_the_listing_exits_nonzero(tmp_path):
    """A listing that NAMES jcodemunch means it is installed, whatever its status.

    `claude mcp list` reports on the whole probe — one unreachable server among
    several is enough for a non-zero status — so its status says nothing about
    whether jcodemunch appeared. Reading the verdict from the pipeline conflates
    the two and re-runs `claude mcp add` on an already-registered server.
    """
    result = _run_jcodemunch(tmp_path, _LISTING_NAMES_IT_THEN_NONZERO)

    combined = result.stdout + result.stderr
    assert "OK jcodemunch MCP already in user config" in combined, combined
    assert _ADD_SENTINEL not in combined, combined


def test_jcodemunch_sees_an_installed_server_when_the_listing_is_sigpiped(tmp_path):
    """A long listing dies of SIGPIPE the instant `grep -q` matches its first line."""
    result = _run_jcodemunch(tmp_path, _LISTING_NAMES_IT_THEN_BULK)

    combined = result.stdout + result.stderr
    assert "OK jcodemunch MCP already in user config" in combined, combined
    assert _ADD_SENTINEL not in combined, combined


def test_jcodemunch_adds_the_server_when_the_listing_does_not_name_it(tmp_path):
    """Guard: a listing without jcodemunch still installs it."""
    result = _run_jcodemunch(tmp_path, _LISTING_WITHOUT_IT)

    combined = result.stdout + result.stderr
    assert _ADD_SENTINEL in combined, combined
    assert "OK jcodemunch MCP added to user config" in combined, combined


def test_jcodemunch_adds_the_server_when_the_listing_cannot_be_read(tmp_path):
    """Guard: an unreadable listing installs, and the capture must not abort.

    `returncode == 0` is the pin against a bare `out="$(...)"` capture, which
    under `set -e` would kill the bootstrap on any host where `claude mcp list`
    fails rather than falling through to the add.
    """
    result = _run_jcodemunch(tmp_path, _LISTING_UNREADABLE)

    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert _ADD_SENTINEL in combined, combined


def _path_without_claude(stub_bin):
    """The stub dir, plus every inherited PATH entry that carries no `claude`.

    `run_section` PREPENDS its stub dir to the inherited PATH, so simply
    writing no `claude` stub is not enough: on a developer host Claude Code is
    installed and the section would probe that real one. Dropping only the
    directories that actually hold a `claude` executable keeps `mkdir` and
    `cat` — which the preamble and this slice both need — while making
    `command -v claude` fail deterministically rather than per-host.
    """
    kept = [
        entry
        for entry in os.environ.get("PATH", "").split(os.pathsep)
        if entry and not os.access(os.path.join(entry, "claude"), os.X_OK)
    ]
    return os.pathsep.join([str(stub_bin), *kept])


def test_jcodemunch_block_is_inert_on_a_host_with_no_claude(tmp_path):
    """A host without Claude Code installed skips the block: no failure, no claim.

    setup-host.sh keeps the `claude mcp list` capture INSIDE the
    `command -v claude` guard and says so in a comment; every other test here
    supplies a `claude` stub, so the case that placement was chosen for went
    unexercised. What this pins from the outside: the section exits 0 and
    claims neither "already in user config" nor "added to user config".

    That is the whole observable contract, and it is deliberately not
    overstated — a hoist that kept its `|| true` is invisible from out here,
    because a capture of a missing binary yields the same empty string. What
    it DOES catch is the two ways the block stops being inert: a capture
    hoisted without `|| true` (a failed simple command mid-bootstrap under
    `set -e`), and a guard dropped altogether, which reaches `claude mcp add`
    and exits 127.
    """
    result = _run_probe(
        tmp_path,
        slice_section(_JCODEMUNCH_START, _JCODEMUNCH_END),
        env_extra={"PATH": _path_without_claude(stub_bin_dir(tmp_path))},
    )

    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    # Positive first: three absence assertions over a slice that never ran are
    # a vacuous green, and the restricted PATH is exactly what could cause it.
    assert "Project config written" in combined, combined
    assert "jcodemunch MCP already in user config" not in combined, combined
    assert "jcodemunch MCP added to user config" not in combined, combined
    assert _ADD_SENTINEL not in combined, combined


# --- section 12: the FalkorDB health check ---------------------------------
# The twin of the section-2 wait loop, minus the retry. Both anchors are code,
# unique, and survive the fix.
_SECTION_12_START = 'info "Health checks"'
_SECTION_12_END = "\nfi\n"


def _run_section_12(tmp_path, exec_body):
    """Slice the section-12 FalkorDB health check and run it."""
    return _run_probe(
        tmp_path,
        slice_section(_SECTION_12_START, _SECTION_12_END),
        stub_name="docker",
        stub_body=_docker_stub_body(exec_body),
        env_extra=_compose_env(tmp_path),
    )


def test_section_12_reports_pong_when_the_producer_exits_nonzero_after_the_reply(
    tmp_path,
):
    """A health check that got PONG says PONG, whatever the producer's status was."""
    result = _run_section_12(tmp_path, _REPLY_THEN_NONZERO)

    combined = result.stdout + result.stderr
    assert "OK FalkorDB: PONG" in combined, combined
    assert "FAIL FalkorDB: not responding" not in combined, combined


def test_section_12_reports_pong_when_the_producer_is_sigpiped_after_the_reply(
    tmp_path,
):
    """The SIGPIPE misread, at the health check rather than the wait loop."""
    result = _run_section_12(tmp_path, _REPLY_THEN_BULK)

    combined = result.stdout + result.stderr
    assert "OK FalkorDB: PONG" in combined, combined
    assert "FAIL FalkorDB: not responding" not in combined, combined


def test_section_12_reports_not_responding_when_the_producer_says_nothing(tmp_path):
    """Guard: a silent producer is still not-responding, reached without aborting."""
    result = _run_section_12(tmp_path, _SILENT_FAILURE)

    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert "FAIL FalkorDB: not responding" in combined, combined
    assert "OK FalkorDB: PONG" not in combined, combined


# --- the file-scoped contract ----------------------------------------------
# A grep on the receiving end of a pipe, plus its arguments up to the end of
# THAT command: `[^|;&)]*` stops at the next pipeline stage, at a `;` or `&&`,
# and at the close of a command substitution, so a `-q` belonging to some later
# command on the same line is never read as this grep's.
_GREP_PIPE = re.compile(r"\|\s*grep\s+(?P<args>[^|;&)]*)")

# Every spelling of "exit on the first match and close the read end": the short
# clusters (`-q`, `-qF`, `-Fq`, `-iq`) and GNU's long forms. Matched against
# whole TOKENS rather than positionally, which is what lets a flag taking an
# argument sit in between — `grep -e PONG --quiet` is the same defect as
# `grep -q PONG` and the sweep must see both. Deliberately does NOT match a
# bare `| grep -F`, which reads its input to the end and cannot SIGPIPE the
# producer.
_QUIET_FLAG = re.compile(r"-[A-Za-z]*q[A-Za-z]*|--quiet|--silent")


def _pipes_into_quiet_grep(line):
    """True when *line* feeds a producer into a grep that exits on first match."""
    return any(
        any(_QUIET_FLAG.fullmatch(token) for token in match.group("args").split())
        for match in _GREP_PIPE.finditer(line)
    )


def _grep_q_offenders(source):
    """Every non-comment line of *source* piping a producer into a quiet grep."""
    return [
        (n, line)
        for n, line in enumerate(source.splitlines(), start=1)
        if not line.strip().startswith("#") and _pipes_into_quiet_grep(line)
    ]


def test_setup_host_never_pipes_a_producer_into_grep_q():
    """No code line may decide anything through `producer | grep --quiet PAT`.

    Generalises scripts/tests/test_lms_ctl.py::
    test_installer_never_pipes_systemctl_into_grep to this file. The `pipefail`
    assertion comes first because it is what makes the rule load-bearing:
    without it there is no defect here and the sweep below would be guarding
    nothing.

    Scoped to greps that EXIT ON FIRST MATCH — every spelling of that, short
    cluster or long flag, since they share one defect. `| grep -F ... || true`
    inside a command substitution is a different, already-guarded shape (the
    `|| true` is what makes it safe, and a non-quiet grep drains its input
    rather than SIGPIPE-ing the producer), so it is deliberately not swept in.
    Neither is a `grep -q` reading a FILE rather than a pipe: with no producer
    upstream there is nothing for `pipefail` to conflate.

    This forbids one known-defective construct and mandates no replacement
    spelling — `[[ ]]`, `case`, or a `<<<` here-string are all still open to a
    future author.
    """
    source = setup_host_text()

    assert "set -euo pipefail" in source

    offenders = _grep_q_offenders(source)
    assert not offenders, "producer piped into `grep -q`:\n" + "\n".join(
        f"  line {n}: {line.strip()}" for n, line in offenders
    )


def test_the_grep_q_sweep_detects_a_planted_pipeline():
    """Guard the guard: a detector that stops matching makes the sweep vacuous.

    Same discipline tests/scripts/test_check_dashboard_unit_parity.py::
    test_the_sweep_finds_every_known_parity_call_site applies to its own sweep.
    Passes on arrival — it pins the mechanism, not the product behaviour.
    """
    planted = (
        "if foo | grep -q BAR; then\n"
        "if foo | grep -qF BAR; then\n"
        "if foo | grep -Fq BAR; then\n"
        "if foo | grep -i -q BAR; then\n"
        # The long forms. `grep --quiet` reintroduces this task's exact defect
        # and reads as innocuous, so it is pinned by the same mechanism as the
        # short flags rather than left to a docstring claim.
        "if foo | grep --quiet BAR; then\n"
        "if foo | grep --silent BAR; then\n"
        # A flag carrying an argument in between must not hide the quiet one.
        "if foo | grep -e BAR --quiet; then\n"
    )
    assert len(_grep_q_offenders(planted)) == 7, _grep_q_offenders(planted)

    # A comment describing the construct is not the construct.
    assert _grep_q_offenders("  # never write `foo | grep -q BAR` here\n") == []
    # Nor is a non-quiet grep, which drains its input instead of closing it.
    assert _grep_q_offenders("out=\"$(foo | grep -F 'tag' || true)\"\n") == []
    # Nor is a `grep -q` over a FILE: no producer upstream, nothing to conflate.
    assert _grep_q_offenders("if grep -q '^\\[Install\\]' \"$unit\"; then\n") == []
    # And a `-q` belonging to a LATER command on the line is not this grep's.
    assert _grep_q_offenders("if foo | grep -F BAR; then bar -q; fi\n") == []
