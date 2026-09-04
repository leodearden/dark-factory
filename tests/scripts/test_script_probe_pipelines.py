"""Behavioural coverage for the `producer | grep -q PAT` probes in three scripts.

Covers scripts/export-data.sh, scripts/import-data.sh and
scripts/deploy-w5-recon-reliability.sh. The sibling suite
test_setup_host_probe_pipelines.py does the same job for scripts/setup-host.sh,
and the two share one slicer and one detector (tests/scripts/shell_sections.py).

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

EVERY ONE OF THESE SITES SITS INSIDE AN `if`, so the failure mode is a WRONG
ANSWER, not an abort: export-data.sh decides a running FalkorDB is not running
and skips the BGSAVE, import-data.sh decides live containers need no stopping
and then reports a healthy FalkorDB as not responding. Nothing crashes and
nothing is logged as a failure — the scripts just quietly do the wrong thing,
which is why these need behavioural tests rather than a lint.

Read through the REAL shipped text: each test slices its section out of the
script by CODE anchors and runs it against PATH stubs, so a test asserts on
what the script DOES, never on how the fix is spelled. The anchors were all
verified to survive the fix, which matters because the same anchors must slice
the unfixed text and the fixed text.

The companion source-level sweep at the bottom forbids the construct itself.
"""

from __future__ import annotations

import pathlib

from shell_sections import (
    REPO_ROOT,
    run_with_preamble,
    slice_section,
    stub_bin_dir,
    write_stub,
)

EXPORT_DATA_PATH = REPO_ROOT / "scripts" / "export-data.sh"
IMPORT_DATA_PATH = REPO_ROOT / "scripts" / "import-data.sh"
DEPLOY_W5_PATH = REPO_ROOT / "scripts" / "deploy-w5-recon-reliability.sh"

# Trailing bytes a producer writes AFTER the matching line, to provoke (b).
#
# MEASURED, not chosen for roundness — inherited from the reproduction sweep in
# test_setup_host_probe_pipelines.py, 30 trials per size:
#
#     65536 -> 26/30      131072 -> 30/30      262144 -> 30/30      1MiB -> 30/30
#
# Whether the producer is scheduled to write again BEFORE grep closes the read
# end is a race, so near the 64KiB pipe buffer the defect is intermittent. Do
# NOT lower this: a value in that flaky band leaves the SIGPIPE tests only
# probabilistically able to catch a reintroduced pipeline.
#
# This is TEST-SIZING guidance and nothing more. It is NOT a claim that smaller
# payloads are safe — measured under task 4981 on bash 5.2.21, a 270-BYTE reply
# with the marker on line 1 still missed 25 times in 4000 evaluations (0.6%,
# every miss rc=141), while the `[[ ]]` form missed 0 in 4000. A sub-buffer site
# is a low-rate FLAKE, not a non-site.
#
# Note the constant only affects how reliably the tests are RED against the OLD
# form: the fixed form drains the producer through a command substitution, where
# there is no pipe to signal, so the tests below are deterministic once fixed.
BULK_BYTES = 262144


# --- the shared scaffold ----------------------------------------------------
# export-data.sh and import-data.sh share `set -euo pipefail`, the same four
# logging shims (their lines 11-14) and the same $REPO_ROOT / $COMPOSE_FILE
# bindings, so one preamble serves every site in both.

# The four logging shims, reduced to PLAIN TEXT so assertions can match on
# prefixes without ANSI escapes. `fail` must still `exit 1` — the shipped one
# does, and a slice that reaches it must die the way the script would.
#
# Deliberately NOT a str.format() template: these bodies are bash brace groups,
# and every `{ printf ... }` in them would be read as a replacement field.
_PREAMBLE = (
    "set -euo pipefail\n"
    "info()  { printf '==> %s\\n' \"$*\"; }\n"
    "ok()    { printf 'OK %s\\n' \"$*\"; }\n"
    "warn()  { printf 'WARN %s\\n' \"$*\"; }\n"
    "fail()  { printf 'FAIL %s\\n' \"$*\"; exit 1; }\n"
)


def _preamble(tmp_path: pathlib.Path) -> str:
    """`_PREAMBLE` plus the two variables the slices read, pointed into tmp_path.

    Under `set -u` an unset $COMPOSE_FILE aborts a slice before it probes
    anything — a green-looking run that never reached the `if` under test.
    Bound here rather than passed through `env_extra` because both are
    script-owned variables, not environment knobs.
    """
    return _PREAMBLE + (
        f'REPO_ROOT="{tmp_path}"\n'
        f'COMPOSE_FILE="{tmp_path / "docker-compose.yml"}"\n'
    )


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


def _run_probe(tmp_path, section_text, *, docker_body):
    """Run *section_text* in a tmp tree against a scripted `docker`.

    One scaffold for every probe site in both scripts: three PATH stubs plus
    the shared preamble. Sites differ only in the slice they pass and the
    docker branch they script — so a new site is a wrapper, not another copy
    of this.

      `sleep`     — a no-op, which is what keeps the 30-iteration timeout
                    cases instant.
      `systemctl` — exits 0, so import-data.sh's section-1 slice can run its
                    `is-active` / `stop` calls without touching the host.
      `docker`    — the scripted producer, the thing actually under test.
    """
    stub_bin = stub_bin_dir(tmp_path)
    write_stub(stub_bin, "sleep", "exit 0\n")
    write_stub(stub_bin, "systemctl", "exit 0\n")
    write_stub(stub_bin, "docker", docker_body)
    return run_with_preamble(tmp_path, _preamble(tmp_path), section_text)


# Scenario bodies for a scripted docker branch, indented to sit inside `case`.
#
# Parameterized by the reply token rather than frozen as constants: the two
# `ps --status running` sites look for `falkordb` and the two `redis-cli ping`
# sites look for `PONG`, and eight near-identical constants is how the four
# sites drift apart.
def _match_then_nonzero(reply):
    """Producer emits the match, then exits non-zero for its own reasons — case (a)."""
    return f"    printf '{reply}\\n'\n    exit 1\n"


def _match_then_bulk(reply):
    """Producer emits the match, then keeps writing until grep closes the pipe — case (b)."""
    return f"    printf '{reply}\\n'\n    head -c {BULK_BYTES} /dev/zero | tr '\\0' x\n"


def _clean_match(reply):
    """Characterization: the ordinary path — the match, then exit 0."""
    return f"    printf '{reply}\\n'\n    exit 0\n"


# A producer that says NOTHING and fails. The honest verdict for every site is
# the negative branch, reached WITHOUT aborting — see each site's guard test.
_SILENT_FAILURE = "    exit 1\n"


# --- export-data.sh section 3: the FalkorDB BGSAVE flush --------------------
# Both anchors are CODE (not comment prose), are unique in the file, and
# survive the fix. Verified to yield the 18-line section-3 block.
_EXPORT_BGSAVE_START = 'info "Flushing FalkorDB to disk"'
_EXPORT_BGSAVE_END = "\nfi\n"


def _lastsave_counter(tmp_path):
    """A `redis-cli LASTSAVE` branch returning a CHANGING value on every call.

    The section reads LASTSAVE once as $BEFORE, fires BGSAVE, then polls
    LASTSAVE until it differs. A stub answering a constant would poll thirty
    times and warn — so the success path would never reach
    `ok "FalkorDB BGSAVE completed"` and test 1 below would assert on an
    absence rather than on a positive message.

    Exits 0: the LASTSAVE call is NOT the producer under test here (the
    `ps --status running` branch is), and its own `|| echo "0"` guard already
    covers a failing one.
    """
    counter = tmp_path / "lastsave-counter"
    return (
        f'    _n="$(cat {counter} 2>/dev/null || echo 0)"\n'
        f"    _n=$((_n + 1))\n"
        f'    printf \'%s\\n\' "$_n" > {counter}\n'
        f'    printf \'%s\\n\' "$_n"\n'
        f"    exit 0\n"
    )


def _run_export_bgsave(tmp_path, ps_body):
    """Slice export-data.sh's section-3 block and run it against a scripted docker.

    The `ps --status running` branch carries the scenario; the `exec` branch
    answers LASTSAVE. Order matters only in that neither glob matches the
    other's argv — `"$*"` for the listing contains `" ps "` and never `" exec "`.
    """
    return _run_probe(
        tmp_path,
        slice_section(EXPORT_DATA_PATH, _EXPORT_BGSAVE_START, _EXPORT_BGSAVE_END),
        docker_body=_dispatch_stub_body(
            (
                ('*" ps "*', ps_body),
                ('*" exec "*', _lastsave_counter(tmp_path)),
            )
        ),
    )


def test_export_flushes_falkordb_when_the_listing_exits_nonzero(tmp_path):
    """A listing that NAMED falkordb means the container is running, whatever its status.

    `docker compose ps` reports on the whole compose invocation; the container
    appearing in the listing is a fact about the OUTPUT. Conflating the two
    skips the BGSAVE on a live database and exports a stale dump.rdb — silent
    data loss, logged as a routine warning.
    """
    result = _run_export_bgsave(tmp_path, _match_then_nonzero("falkordb"))

    combined = result.stdout + result.stderr
    assert "OK FalkorDB BGSAVE completed" in combined, combined
    assert "WARN FalkorDB container not running" not in combined, combined


def test_export_flushes_falkordb_when_the_listing_is_sigpiped(tmp_path):
    """A producer still writing when grep matches dies of SIGPIPE; falkordb was still listed.

    `grep -q` closes the read end on its first match, so the producer takes
    signal 13, `pipefail` turns that into 141, and the `if` reads "not running"
    off a listing that began with the container's own name.
    """
    result = _run_export_bgsave(tmp_path, _match_then_bulk("falkordb"))

    combined = result.stdout + result.stderr
    assert "OK FalkorDB BGSAVE completed" in combined, combined
    assert "WARN FalkorDB container not running" not in combined, combined


def test_export_skips_the_flush_when_the_listing_says_nothing(tmp_path):
    """A silent listing is still "not running" — and the section must NOT abort getting there.

    The guard on the fix. A capture written as a bare `_running="$(producer)"`
    makes the assignment a SIMPLE COMMAND, so `set -e` kills the whole export
    the moment docker is unavailable, where the old pipeline merely took the
    else branch. `returncode == 0` is the only assertion in this file that
    distinguishes the correct fix from that plausible-looking regression.
    """
    result = _run_export_bgsave(tmp_path, _SILENT_FAILURE)

    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert "WARN FalkorDB container not running" in combined, combined
    assert "OK FalkorDB BGSAVE completed" not in combined, combined


def test_export_flushes_falkordb_on_a_clean_listing(tmp_path):
    """Characterization: the ordinary path lists falkordb and exits 0."""
    result = _run_export_bgsave(tmp_path, _clean_match("falkordb"))

    combined = result.stdout + result.stderr
    assert "OK FalkorDB BGSAVE completed" in combined, combined
    assert "WARN FalkorDB container not running" not in combined, combined


# --- import-data.sh section 1: stopping the backing stores ------------------
# The byte-identical `ps --status running | grep -q falkordb` construct fixed
# above, in a different executable.
#
# `end_after` IS REQUIRED HERE, and this was MEASURED rather than assumed:
# without it the slice ends at the `fi` closing the PRECEDING
# `systemctl --user is-active fused-memory` block and never reaches the docker
# site at all — a slice of the wrong region that runs cleanly and produces a
# vacuously green test. That is precisely the hazard slice_section's third
# anchor exists for. All three anchors occur exactly once, on non-comment
# lines, and survive the fix.
_IMPORT_STOP_START = 'info "Stopping services"'
_IMPORT_STOP_END = "\nfi\n"
_IMPORT_STOP_END_AFTER = 'ok "FalkorDB + Qdrant containers stopped"'

_STOPPED = "OK FalkorDB + Qdrant containers stopped"


def _run_import_stop(tmp_path, ps_body):
    """Slice import-data.sh's section-1 block and run it against a scripted docker.

    The slice also runs `systemctl --user is-active` and `systemctl --user stop`;
    the harness's PATH `systemctl` stub exits 0, so that branch is taken and is
    harmless. The `compose ... stop falkordb qdrant` call falls to the docker
    stub's catch-all — only the listing is under test.
    """
    return _run_probe(
        tmp_path,
        slice_section(
            IMPORT_DATA_PATH,
            _IMPORT_STOP_START,
            _IMPORT_STOP_END,
            end_after=_IMPORT_STOP_END_AFTER,
        ),
        docker_body=_dispatch_stub_body((('*" ps "*', ps_body),)),
    )


def test_import_stops_the_containers_when_the_listing_exits_nonzero(tmp_path):
    """A listing that NAMED falkordb means there are containers to stop.

    Misreading it leaves FalkorDB and Qdrant RUNNING while the next sections
    replace the data trees underneath them — the import's whole reason for
    stopping them first.
    """
    result = _run_import_stop(tmp_path, _match_then_nonzero("falkordb"))

    combined = result.stdout + result.stderr
    assert _STOPPED in combined, combined


def test_import_stops_the_containers_when_the_listing_is_sigpiped(tmp_path):
    """Same misread via SIGPIPE: `grep -q` closes the pipe, the producer dies, 141."""
    result = _run_import_stop(tmp_path, _match_then_bulk("falkordb"))

    combined = result.stdout + result.stderr
    assert _STOPPED in combined, combined


def test_import_stops_nothing_when_the_listing_says_nothing(tmp_path):
    """A silent listing means nothing to stop — reached WITHOUT aborting.

    The honest verdict for a producer that said nothing, and the guard against
    the bare-assignment spelling: `returncode == 0` pins that the fix must not
    turn an unavailable docker into a `set -e` abort of the whole import.
    """
    result = _run_import_stop(tmp_path, _SILENT_FAILURE)

    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert _STOPPED not in combined, combined


def test_import_stops_the_containers_on_a_clean_listing(tmp_path):
    """Characterization: the ordinary path lists falkordb and exits 0."""
    result = _run_import_stop(tmp_path, _clean_match("falkordb"))

    combined = result.stdout + result.stderr
    assert _STOPPED in combined, combined


# --- import-data.sh section 7: the FalkorDB "wait for healthy" loop ---------
# Anchor occurs exactly once on a non-comment line and survives the fix. The
# same anchor shape task 4204 used for setup-host.sh's section-2 wait loop,
# which this block is a literal copy of. Verified to yield the 11-line block.
_IMPORT_WAIT_START = 'docker compose -f "$COMPOSE_FILE" up -d falkordb qdrant'
_IMPORT_WAIT_END = "\ndone\n"

_HEALTHY = "OK FalkorDB healthy"
_NOT_HEALTHY = "WARN FalkorDB did not become healthy in 30s"


def _run_import_wait(tmp_path, exec_body):
    """Slice import-data.sh's section-7 wait loop and run it against a scripted docker.

    The `up -d` invocation falls to the stub's catch-all and exits 0 silently;
    only the `redis-cli ping` exec branch is under test. The no-op `sleep` stub
    is what makes the 30-iteration timeout case instant.
    """
    return _run_probe(
        tmp_path,
        slice_section(IMPORT_DATA_PATH, _IMPORT_WAIT_START, _IMPORT_WAIT_END),
        docker_body=_dispatch_stub_body((('*" exec "*', exec_body),)),
    )


def test_import_wait_reports_healthy_when_the_ping_exits_nonzero(tmp_path):
    """A ping that ANSWERED PONG is healthy, whatever else the producer's status says.

    `docker compose exec` reports on the exec run as a whole; redis-cli having
    answered is a fact about the OUTPUT. Reading the verdict from the
    pipeline's status conflates the two and polls a live FalkorDB for thirty
    seconds before declaring it never came up.
    """
    result = _run_import_wait(tmp_path, _match_then_nonzero("PONG"))

    combined = result.stdout + result.stderr
    assert _HEALTHY in combined, combined
    assert _NOT_HEALTHY not in combined, combined


def test_import_wait_reports_healthy_when_the_ping_is_sigpiped(tmp_path):
    """A producer still writing when grep matches dies of SIGPIPE; PONG was still said."""
    result = _run_import_wait(tmp_path, _match_then_bulk("PONG"))

    combined = result.stdout + result.stderr
    assert _HEALTHY in combined, combined
    assert _NOT_HEALTHY not in combined, combined


def test_import_wait_times_out_when_the_ping_says_nothing(tmp_path):
    """No reply is still not-healthy — and the loop must reach that verdict without aborting."""
    result = _run_import_wait(tmp_path, _SILENT_FAILURE)

    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert _NOT_HEALTHY in combined, combined
    assert _HEALTHY not in combined, combined


def test_import_wait_reports_healthy_on_a_clean_ping(tmp_path):
    """Characterization: the ordinary path answers PONG and exits 0."""
    result = _run_import_wait(tmp_path, _clean_match("PONG"))

    combined = result.stdout + result.stderr
    assert _HEALTHY in combined, combined
    assert _NOT_HEALTHY not in combined, combined
