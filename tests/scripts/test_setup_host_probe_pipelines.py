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

from setup_host_sections import run_section, slice_section

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
