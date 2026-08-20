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

import re

from setup_host_sections import run_section, setup_host_text, slice_section

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


def _write_stub(stub_bin, name, body):
    """Drop an executable bash stub *name* carrying *body* into *stub_bin*."""
    path = stub_bin / name
    path.write_text("#!/usr/bin/env bash\n" + body, encoding="utf-8")
    path.chmod(0o755)
    return path


def _stub_bin(tmp_path):
    """The stub dir `run_section` will reuse, pre-loaded with a no-op `sleep`.

    `run_section` does `stub_bin.mkdir(exist_ok=True)` on this exact path, so
    stubs written here survive and its own `systemctl` stub joins them. The
    `sleep` stub is what keeps the 30-iteration timeout case instant.
    """
    stub_bin = tmp_path / "stub-bin"
    stub_bin.mkdir(exist_ok=True)
    _write_stub(stub_bin, "sleep", "exit 0\n")
    return stub_bin


def _docker_stub(stub_bin, exec_body):
    """A `docker` stub whose `... exec ...` invocation runs *exec_body*.

    The `up -d` invocation exits 0 silently; only the exec branch is under
    test. `case` is the stub's LAST command, so the exec branch's own status
    becomes the stub's exit status. That is deliberate and load-bearing: the
    producer's status IS the thing these tests are about, and a trailing
    `exit 0` here would swallow it and make every test below vacuously green.
    """
    return _write_stub(
        stub_bin,
        "docker",
        'case "$*" in\n'
        '  *" exec "*)\n'
        f"{exec_body}"
        "    ;;\n"
        "  *)\n"
        "    exit 0\n"
        "    ;;\n"
        "esac\n",
    )


# Scenario bodies for the docker exec branch. Indented to sit inside `case`.
_REPLY_THEN_NONZERO = "    printf 'PONG\\n'\n    exit 1\n"
_REPLY_THEN_BULK = f"    printf 'PONG\\n'\n    head -c {BULK_BYTES} /dev/zero | tr '\\0' x\n"
_SILENT_FAILURE = "    exit 1\n"
_CLEAN_REPLY = "    printf 'PONG\\n'\n    exit 0\n"


def _run_section_2(tmp_path, exec_body):
    """Slice the section-2 wait loop and run it against a scripted docker."""
    stub_bin = _stub_bin(tmp_path)
    _docker_stub(stub_bin, exec_body)
    repo_root = tmp_path / "repo"
    repo_root.mkdir(exist_ok=True)
    return run_section(
        tmp_path,
        slice_section(_SECTION_2_START, _SECTION_2_END),
        repo_root=repo_root,
        unit_dir=tmp_path / "units",
        # The slice reads $COMPOSE_FILE and the harness preamble does not
        # define it, so under `set -u` it would abort without this.
        env_extra={"COMPOSE_FILE": str(tmp_path / "docker-compose.yml")},
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
    stub_bin = _stub_bin(tmp_path)
    _write_stub(
        stub_bin,
        "claude",
        'case "$*" in\n'
        '  *"mcp add"*)\n'
        f"    printf '{_ADD_SENTINEL}\\n'\n"
        "    exit 0\n"
        "    ;;\n"
        '  *"mcp list"*)\n'
        f"{list_body}"
        "    ;;\n"
        "  *)\n"
        "    exit 0\n"
        "    ;;\n"
        "esac\n",
    )
    repo_root = tmp_path / "repo"
    repo_root.mkdir(exist_ok=True)
    return run_section(
        tmp_path,
        slice_section(_JCODEMUNCH_START, _JCODEMUNCH_END),
        repo_root=repo_root,
        unit_dir=tmp_path / "units",
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


# --- section 12: the FalkorDB health check ---------------------------------
# The twin of the section-2 wait loop, minus the retry. Both anchors are code,
# unique, and survive the fix.
_SECTION_12_START = 'info "Health checks"'
_SECTION_12_END = "\nfi\n"


def _run_section_12(tmp_path, exec_body):
    """Slice the section-12 FalkorDB health check and run it."""
    stub_bin = _stub_bin(tmp_path)
    _docker_stub(stub_bin, exec_body)
    repo_root = tmp_path / "repo"
    repo_root.mkdir(exist_ok=True)
    return run_section(
        tmp_path,
        slice_section(_SECTION_12_START, _SECTION_12_END),
        repo_root=repo_root,
        unit_dir=tmp_path / "units",
        env_extra={"COMPOSE_FILE": str(tmp_path / "docker-compose.yml")},
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
# Any `-`-flag cluster containing `q` on the grep at the end of a pipe, so
# `-q`, `-qF`, `-Fq` and `-i -q` are all caught. Deliberately does NOT match a
# bare `| grep -F`.
_GREP_Q_PIPE = re.compile(r"\|\s*grep\s+(?:-[A-Za-z]+\s+)*-[A-Za-z]*q")


def _grep_q_offenders(source):
    """Every non-comment line of *source* piping a producer into `grep -q`."""
    return [
        (n, line)
        for n, line in enumerate(source.splitlines(), start=1)
        if not line.strip().startswith("#") and _GREP_Q_PIPE.search(line)
    ]


def test_setup_host_never_pipes_a_producer_into_grep_q():
    """No code line may decide anything through `producer | grep -q PAT`.

    Generalises scripts/tests/test_lms_ctl.py::
    test_installer_never_pipes_systemctl_into_grep to this file. The `pipefail`
    assertion comes first because it is what makes the rule load-bearing:
    without it there is no defect here and the sweep below would be guarding
    nothing.

    Scoped to `grep -q` ONLY. `| grep -F ... || true` inside a command
    substitution is a different, already-guarded shape (the `|| true` is what
    makes it safe) and is deliberately not swept in.

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
        'if foo | grep -q BAR; then\n'
        'if foo | grep -qF BAR; then\n'
        'if foo | grep -Fq BAR; then\n'
        'if foo | grep -i -q BAR; then\n'
    )
    assert len(_grep_q_offenders(planted)) == 4, _grep_q_offenders(planted)

    # A comment describing the construct is not the construct.
    assert _grep_q_offenders('  # never write `foo | grep -q BAR` here\n') == []
    # Nor is a non-`-q` grep, which cannot exit early on a match.
    assert _grep_q_offenders("out=\"$(foo | grep -F 'tag' || true)\"\n") == []
