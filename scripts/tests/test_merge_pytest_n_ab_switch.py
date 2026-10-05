"""Tests for merge-pytest-n-ab-switch.sh — drives the real script via
subprocess against a REAL loopback stateful escalation MCP on an ephemeral
port and a REAL temp git repo holding a temp dark-factory-orchestrator.yaml
(the test_deploy_w11_lane_lifecycle.py idiom), so step 1's YAML editor, step
2's commit idempotency and step 4's assertion are exercised in COMPOSITION —
which is where the crash-resume defect lives: file already at the value ->
nothing to commit -> the reload reports no verify_env change. The server, its
transport-fault catalogue and the temp-repo helpers live in
config_reload_script_fakes, which owns the measured wire model.

The reload travels over a real socket on purpose. A faked `curl` can only
ever return what the test author already believes the wire looks like, so it
cannot catch a wrong belief ABOUT the wire — and that is how a reload step
that never once reached the tool in production shipped green.

Nothing here touches a live orchestrator: the only server reachable is the
one the test itself owns, bound to an ephemeral port.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import types
from pathlib import Path

import pytest
from config_reload_script_fakes import (
    TRANSPORT_FAULTS,
    ClosedPort,
    FakeEscalationMcp,
    commit_config,
    git,
    head,
    init_repo,
    path_python3_shimmed_to,
    reload_report,
    system_python_without_the_transport,
    venvless_checkout_copy,
)

SCRIPT = Path(__file__).parent.parent / "merge-pytest-n-ab-switch.sh"
KEY = "PYTEST_XDIST_AUTO_NUM_WORKERS"


# ---------------------------------------------------------------------------
# Real temp git repo holding a temp dark-factory-orchestrator.yaml
# ---------------------------------------------------------------------------

def _make_repo(tmp_path, verify_env_value, *, marker=False):
    """A committed temp repo whose config's top-level `verify_env:` block pins
    KEY to *verify_env_value* -- the state every switch starts from.

    `marker=True` precedes the key with an `  # A/B arm: ...` line, the state
    a previously-switched config file is left in.

    Returns the yaml path.
    """
    marker_line = (
        f'  # A/B arm: {KEY} set to "{verify_env_value}" at '
        "2026-09-10T12:42:14Z by scripts/merge-pytest-n-ab-switch.sh\n"
    )
    return commit_config(
        init_repo(tmp_path),
        "project_root: /nowhere\n"
        "max_concurrent_tasks: 3\n"
        "\n"
        "verify_env:\n"
        + (marker_line if marker else "")
        + f'  {KEY}: "{verify_env_value}"\n'
        '  PYTHONWARNINGS: "ignore"\n'
        "\n"
        "review:\n"
        "  enabled: true\n",
    )


# ---------------------------------------------------------------------------
# Script driver
# ---------------------------------------------------------------------------

def _run(server, config_path, value, *, script=SCRIPT, env=None, cwd=None, flags=()):
    """Run the script for *value* against *config_path*, pointing its reload
    at *server*'s real ephemeral port.

    *script* and *env* are the seams the interpreter-resolution tests need: a
    COPY of the script in a checkout with no venv, and a PATH whose `python3`
    lacks the transport or cannot run at all. *cwd* is the seam the relative-config_path
    case needs (the client cwd is what a relative path is resolved against),
    and *flags* carries `--dry-run`.
    """
    return subprocess.run(
        ["bash", str(script), value, str(config_path), str(server.port), *flags],
        env=env, cwd=None if cwd is None else str(cwd),
        capture_output=True, text=True, timeout=60,
    )


def _verdict(proc):
    """Parse the script's last non-empty stdout line as the JSON verdict."""
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    assert lines, f"no stdout at all; rc={proc.returncode} stderr={proc.stderr}"
    return json.loads(lines[-1])


def _failure(proc):
    """The failure tag of a run that must have failed, read from its verdict."""
    assert proc.returncode != 0, f"stdout={proc.stdout} stderr={proc.stderr}"
    verdict = _verdict(proc)
    assert "outcome" not in verdict, verdict
    return verdict["failure"]


# ---------------------------------------------------------------------------
# The flip path: a real value change, hot-applied
# ---------------------------------------------------------------------------

def test_flip_reports_outcome_applied_and_commits(tmp_path):
    """A genuine 16 -> 8 switch edits the yaml, lands exactly one commit
    touching only that file, and reports `outcome: applied`."""
    config = _make_repo(tmp_path, "16")
    repo = config.parent
    before = head(repo)

    with FakeEscalationMcp(reload_report(
        reloaded=True,
        config_path=str(config),
        applied={"verify_env": {"old": {KEY: "16"}, "new": {KEY: "8"}}},
    )) as server:
        proc = _run(server, config, "8")
        methods = server.rpc_methods()

    assert proc.returncode == 0, f"stdout={proc.stdout} stderr={proc.stderr}"
    verdict = _verdict(proc)
    assert verdict["outcome"] == "applied"
    assert verdict["switched_to"] == "8"
    assert verdict["commit"] != "already-at-8"
    assert "initialize" in methods, (
        f"the script never handshook; the server saw {methods}"
    )

    landed = git(repo, "rev-list", f"{before}..HEAD").split()
    assert len(landed) == 1, f"expected exactly one new commit, got {landed}"
    touched = git(repo, "show", "--name-only", "--format=", "HEAD").split()
    assert touched == ["dark-factory-orchestrator.yaml"]
    assert f'{KEY}: "8"' in config.read_text()


# ---------------------------------------------------------------------------
# The converged path: a crash-resume onto a config that already carries the
# value, against an orchestrator whose live config already matches it
# ---------------------------------------------------------------------------

def test_converged_resume_exits_zero_when_reload_reports_no_verify_env_change(tmp_path):
    """Re-running the switch for a value the file and the running config
    BOTH already carry is success, not failure.

    The reload genuinely happens and genuinely applies other hot leaves, but
    verify_env is equal on both sides so it never appears under `applied` --
    `unchanged` is a bare int count, so absence is the only converged signal
    on the wire. Nothing to commit is likewise already success (step 2's
    `already-at-<value>` sentinel); step 4 must reach the same verdict.
    """
    config = _make_repo(tmp_path, "8", marker=True)
    repo = config.parent
    before_head = head(repo)
    before_bytes = config.read_bytes()

    with FakeEscalationMcp(reload_report(
        reloaded=True,
        error=None,
        config_path=str(config),
        applied={"max_turns.architect": {"old": 40, "new": 60}},
    )) as server:
        proc = _run(server, config, "8")

    assert proc.returncode == 0, f"stdout={proc.stdout} stderr={proc.stderr}"
    verdict = _verdict(proc)
    assert "failure" not in verdict, verdict
    assert verdict["outcome"] == "already_converged"
    assert verdict["switched_to"] == "8"
    assert verdict["commit"] == "already-at-8"

    assert head(repo) == before_head, "a converged re-run must commit nothing"
    assert config.read_bytes() == before_bytes, (
        "the idempotent rewrite must restore the prior A/B marker byte for "
        "byte, not stack a fresh one"
    )


def test_converged_resume_over_the_real_stateful_transport(tmp_path):
    """The converged verdict is reached THROUGH the session handshake.

    Its sibling above owns the git/yaml idempotence; this one owns the wire:
    the same converged report, asserted to have been fetched by an
    `initialize` followed by a `tools/call` naming `reload_config`. A
    session-less single-shot POST never gets that far -- it is rejected with
    a 400 before any tool runs -- so exiting 0 here cannot be luck.

    The sequence is asserted as "some tools/call AFTER the initialize", not
    as a fixed list: `post_mcp_envelope` is deliberately ADAPTIVE, sending
    the envelope session-less first and handshaking only on the 400, so the
    real wire also carries a rejected tools/call ahead of the handshake.

    The session it opened is asserted CLOSED for the same reason it is
    asserted opened: both are wire facts this script depends on and cannot
    see from its own exit code.
    """
    config = _make_repo(tmp_path, "8", marker=True)

    with FakeEscalationMcp(reload_report(
        reloaded=True,
        config_path=str(config),
        applied={"max_turns.architect": {"old": 40, "new": 60}},
    )) as server:
        proc = _run(server, config, "8")
        methods = server.rpc_methods()
        tools = server.called_tools()
        deletes = server.delete_count()

    assert proc.returncode == 0, f"stdout={proc.stdout} stderr={proc.stderr}"
    assert _verdict(proc)["outcome"] == "already_converged"
    assert "initialize" in methods, methods
    assert "tools/call" in methods[methods.index("initialize"):], methods
    assert set(tools) == {"reload_config"}, f"methods={methods}"
    assert deletes == 1, (
        f"the session was opened and not released, one leaked anyio task per "
        f"run in the long-lived escalation process; the server saw {methods} "
        f"and {deletes} DELETEs"
    )


# ---------------------------------------------------------------------------
# Exits that end the run before the orchestrator is ever contacted: the
# rehearsal, the refusal of a port no socket can reach, and step 1's refusal
# to edit a config it cannot find a verify_env block in.
# ---------------------------------------------------------------------------

def test_dry_run_neither_commits_nor_reloads(tmp_path):
    """`--dry-run` shows the operator the change and stops there.

    A whole separate exit path -- temp copy, `diff -u`, its own verdict shape
    -- so a regression in it would let a rehearsal diverge from the real run
    silently, which is the one thing a rehearsal must not do.
    """
    config = _make_repo(tmp_path, "16")
    repo = config.parent
    before_head = head(repo)
    before_bytes = config.read_bytes()

    with FakeEscalationMcp(reload_report(config_path=str(config))) as server:
        proc = _run(server, config, "8", flags=("--dry-run",))
        received = list(server.received)

    assert proc.returncode == 0, f"stdout={proc.stdout} stderr={proc.stderr}"
    assert _verdict(proc) == {"dry_run": True, "would_set": "8"}
    assert f'+  {KEY}: "8"' in proc.stdout, (
        f"a rehearsal that shows no diff rehearses nothing: {proc.stdout}"
    )
    assert head(repo) == before_head, "a rehearsal must not commit"
    assert config.read_bytes() == before_bytes, "a rehearsal must not edit in place"
    assert received == [], f"a rehearsal must not reach the orchestrator: {received}"


def test_a_config_with_no_verify_env_block_is_refused_before_the_commit(tmp_path):
    """Pointed at a config carrying no top-level `verify_env:`, the editor
    refuses rather than inventing one -- and refuses BEFORE the commit, so a
    config it cannot safely edit leaves no trace in git either.
    """
    repo = init_repo(tmp_path)
    config = commit_config(
        repo, "project_root: /nowhere\nreview:\n  enabled: true\n"
    )
    before_head = head(repo)
    before_bytes = config.read_bytes()

    with FakeEscalationMcp(reload_report(config_path=str(config))) as server:
        proc = _run(server, config, "8")
        received = list(server.received)

    assert _failure(proc) == "config_edit_refused"
    assert head(repo) == before_head, "the refusal must precede the commit"
    assert config.read_bytes() == before_bytes
    assert received == [], f"a refused edit must not reload anything: {received}"


@pytest.mark.parametrize("bad_port", ["0", "65536", "99999"])
def test_an_unconnectable_port_is_refused_before_the_commit(tmp_path, bad_port):
    """A port outside 1-65535 can never deliver the reload, so it is refused
    before a value that could never be hot-applied is committed."""
    config = _make_repo(tmp_path, "16")
    repo = config.parent
    before_head = head(repo)
    before_bytes = config.read_bytes()

    proc = _run(types.SimpleNamespace(port=bad_port), config, "8")

    assert _failure(proc) == "invalid_port"
    assert head(repo) == before_head, "the refusal must precede the commit"
    assert config.read_bytes() == before_bytes


# ---------------------------------------------------------------------------
# Responses that must NOT be read as success
#
# Absence of verify_env from `applied` is strictly WEAKER than "converged":
# the very same absence is produced by a reload that rolled every leaf back,
# and by a reload of a different orchestrator entirely (the port is a
# caller-supplied argument, so a wrong one reaches another project's MCP).
# This is a DEPLOY gate -- blessing an undeployed arm is worse than the
# over-strict assertion the converged branch removes. A PRESENT verify_env
# carrying our value is no stronger about OUR file: another orchestrator with
# its own pending change to the same value reports exactly that, so the same
# corroborators gate the flip.
#
# Each case pins the failure TAG it is about, not merely a non-zero exit: a
# transport rejection also exits non-zero, and a bare `returncode != 0` would
# let one stand in for all four and pass for the wrong reason.
# ---------------------------------------------------------------------------

def _verdict_lines(proc):
    """(line, verdict) for every stdout line that parses as a JSON object."""
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            yield line, parsed


def _converged_verdict_lines(proc):
    """Every stdout line that parses as a verdict claiming convergence."""
    return [
        line for line, verdict in _verdict_lines(proc)
        if verdict.get("outcome") == "already_converged"
    ]


def _outcome_verdict_lines(proc):
    """Every stdout line that parses as a verdict claiming any success outcome."""
    return [line for line, verdict in _verdict_lines(proc) if "outcome" in verdict]


def _other_projects_config(tmp_path):
    """Another project's dark-factory-orchestrator.yaml, outside this repo."""
    other = tmp_path / "other-project" / "dark-factory-orchestrator.yaml"
    other.parent.mkdir()
    other.write_text(f'verify_env:\n  {KEY}: "2"\n')
    return other


# id -> (builder taking (this repo's config, another project's config) and
# returning reload-report fields that fail to show THIS config file was
# re-read and committed, the failure tag naming which corroborator is missing).
# Shared by the converged and flip shapes, which must hold the reload to the
# same corroborators.
_CORROBORATOR_BREACHES = {
    # apply_reload rolled every leaf back, so the live config is untouched.
    "reload_failed_silently": (
        lambda config, other: {"reloaded": False, "error": None, "config_path": str(config)},
        "reload_not_committed",
    ),
    # We reloaded something that is not the file we just edited, so its
    # verify_env says nothing about ours.
    "different_config_file": (
        lambda config, other: {"reloaded": True, "config_path": str(other)},
        "different_config_file",
    ),
    "no_config_path": (
        lambda config, other: {"reloaded": True, "config_path": None},
        "different_config_file",
    ),
    # A RELATIVE config_path is the reporting orchestrator's own
    # ORCH_CONFIG_PATH, resolved by ITS cwd -- resolving it against ours makes
    # any unit configured with the bare filename match us whenever the script
    # runs from the config's checkout, which is exactly how it is run (and how
    # this group runs it below). Uncomparable, therefore not corroborating.
    "relative_config_path": (
        lambda config, other: {"reloaded": True, "config_path": "dark-factory-orchestrator.yaml"},
        "different_config_file",
    ),
}

# The converged shape's cases: every corroborator breach, plus two that only
# an ABSENT verify_env can carry.
_UNCORROBORATED_ABSENCE = {
    # Control: the existing error branch already rejects this one, so a green
    # here proves the group's harness really drives the script.
    "reload_failed_loudly": (
        lambda config, other: {
            "reloaded": False, "error": "load_config: while parsing a block mapping",
            "config_path": str(config),
        },
        "reload_error",
    ),
    **_CORROBORATOR_BREACHES,
    # verify_env DID change, but was bucketed as restart-required instead of
    # hot-applied, so the arm is committed and NOT live -- and it is absent
    # from `applied` exactly like a converged one. Latent while verify_env
    # stays in RELOADABLE_FIELDS; a fail-open the moment it does not.
    "verify_env_is_restart_required": (
        lambda config, other: {
            "reloaded": True, "config_path": str(config),
            "restart_required": {"verify_env": {"old": {KEY: "16"}, "new": {KEY: "8"}}},
        },
        "restart_required",
    ),
}


@pytest.mark.parametrize(
    ("report_fields", "failure"),
    list(_UNCORROBORATED_ABSENCE.values()),
    ids=list(_UNCORROBORATED_ABSENCE),
)
def test_uncorroborated_absence_is_not_convergence(tmp_path, report_fields, failure):
    """An absent verify_env that is NOT corroborated by a committed reload of
    THIS config file must fail, naming which corroborator was missing."""
    config = _make_repo(tmp_path, "8", marker=True)
    other = _other_projects_config(tmp_path)

    with FakeEscalationMcp(reload_report(**report_fields(config, other))) as server:
        # From the config's own checkout, as an operator runs it -- and the
        # only cwd from which a relative reported config_path could realpath
        # onto ours and falsely corroborate.
        proc = _run(server, config, "8", cwd=config.parent)

    assert _failure(proc) == failure
    assert not _converged_verdict_lines(proc)


@pytest.mark.parametrize(
    ("report_fields", "failure"),
    list(_CORROBORATOR_BREACHES.values()),
    ids=list(_CORROBORATOR_BREACHES),
)
def test_a_flip_reported_by_an_uncorroborated_reload_is_not_applied(
    tmp_path, report_fields, failure,
):
    """A verify_env flip to our value that is NOT corroborated by a committed
    reload of THIS config file must fail too. The flip commit has landed, by
    design, so the failure verdict names it."""
    config = _make_repo(tmp_path, "16")
    repo = config.parent
    other = _other_projects_config(tmp_path)

    with FakeEscalationMcp(reload_report(
        **report_fields(config, other),
        applied={"verify_env": {"old": {KEY: "16"}, "new": {KEY: "8"}}},
    )) as server:
        proc = _run(server, config, "8", cwd=config.parent)

    assert _failure(proc) == failure
    assert not _outcome_verdict_lines(proc)
    assert _verdict(proc)["commit"] == head(repo, short=True)


def test_applied_verify_env_carrying_a_different_value_still_fails(tmp_path):
    """The strict branch stays strict: a PRESENT verify_env carrying some
    other value is a contradiction, never convergence."""
    config = _make_repo(tmp_path, "8", marker=True)

    with FakeEscalationMcp(reload_report(
        reloaded=True,
        config_path=str(config),
        applied={"verify_env": {"old": {KEY: "8"}, "new": {KEY: "16"}}},
    )) as server:
        proc = _run(server, config, "8")

    assert _failure(proc) == "value_mismatch"
    assert not _converged_verdict_lines(proc)


# ---------------------------------------------------------------------------
# The reload never reached the tool
#
# A transport rejection must never be readable as a rolled-back reload. The
# defect this file exists to close was not that the script passed when it
# should have failed -- it was that it failed while naming the WRONG cause:
# every session-less POST got a 400 before any tool ran, and the script
# reported `reloaded=None` and blamed an in-orchestrator rollback for it.
# ---------------------------------------------------------------------------

def _assert_failed_closed_on_the_transport(proc):
    """The one verdict every transport fault must reach: a failure that names
    the TRANSPORT, and never the in-orchestrator rollback it is not."""
    assert _failure(proc) == "no_reload_report"
    assert not _converged_verdict_lines(proc)


@pytest.mark.parametrize(
    "fault", list(TRANSPORT_FAULTS.values()), ids=list(TRANSPORT_FAULTS)
)
def test_a_reload_that_never_reached_the_tool_fails_closed(tmp_path, fault):
    """Every way the transport can refuse to deliver reload_config."""
    config = _make_repo(tmp_path, "8", marker=True)

    with FakeEscalationMcp(reload_report(config_path=str(config)), **fault) as server:
        proc = _run(server, config, "8")

    _assert_failed_closed_on_the_transport(proc)


def test_a_dead_socket_fails_closed(tmp_path):
    """Nothing listening at all -- httpx's own transport exception, which is
    neither of the two census_trigger raises and must fail the same way."""
    config = _make_repo(tmp_path, "8", marker=True)

    proc = _run(ClosedPort(), config, "8")

    _assert_failed_closed_on_the_transport(proc)


def test_the_transport_diagnostic_keeps_the_committed_shas_remedy(tmp_path):
    """On the FLIP path the commit HAS landed, so both the failure verdict and
    the operator's diagnostic name the sha that carries the value.

    That is what the deleted `curl ... || die "... (committed as ${SHA}; the
    value lands at the next restart)"` carried; losing it when curl went away
    would leave an operator with a failed deploy and no next step.
    """
    config = _make_repo(tmp_path, "16")
    repo = config.parent

    with FakeEscalationMcp(
        initialize_reply=FakeEscalationMcp.MISSING_SESSION_REPLY,
    ) as server:
        proc = _run(server, config, "8")

    _assert_failed_closed_on_the_transport(proc)
    sha = head(repo, short=True)
    assert _verdict(proc)["commit"] == sha
    assert sha in proc.stderr, f"stderr={proc.stderr}"


# ---------------------------------------------------------------------------
# Which interpreter runs the script's python steps
#
# The transport is not stdlib: it needs httpx, plus pydantic via
# census_trigger's module-level `legibility.config` import. Measured here on
# 2026-09-12: the system interpreter (/usr/bin/python3, 3.12.3) has neither
# (it does have yaml), while every venv in this tree has all three. So a
# reload step that inherits whatever `python3` the caller's shell offers is a
# deploy gate an operator cannot reach from a login shell -- the same
# unreachable-gate outcome as the transport defect above, with a different
# cause. Step 1's editor shares the reload's resolved interpreter, so neither
# step reaches PATH's `python3` while the checkout venv exists.
# ---------------------------------------------------------------------------

CHECKOUT_VENV_PYTHON = SCRIPT.parent.parent / ".venv" / "bin" / "python3"


def test_no_step_runs_under_path_python3_when_the_checkout_venv_exists(tmp_path):
    """PATH's `python3` cannot run at all; the flip still lands.

    That can only pass if BOTH the YAML edit and the reload resolved the
    SCRIPT's own checkout venv rather than inheriting PATH.
    """
    if not CHECKOUT_VENV_PYTHON.exists():
        pytest.skip(f"{CHECKOUT_VENV_PYTHON} is absent (an un-synced worktree)")

    config = _make_repo(tmp_path, "16")
    env = path_python3_shimmed_to(tmp_path, shutil.which("false"))

    with FakeEscalationMcp(reload_report(
        config_path=str(config),
        applied={"verify_env": {"old": {KEY: "16"}, "new": {KEY: "8"}}},
    )) as server:
        proc = _run(server, config, "8", env=env)

    assert proc.returncode == 0, f"stdout={proc.stdout} stderr={proc.stderr}"
    assert _verdict(proc)["outcome"] == "applied"


def test_an_interpreter_without_the_transport_fails_loud_with_a_remedy(tmp_path):
    """No venv to fall back on: fail naming the interpreter and the remedy.

    The script is copied into a checkout carrying its real reload step but NO
    `.venv`, so the bare-`python3` fallback leg is taken. A raw ImportError
    traceback would leave an operator with no next step.
    """
    system_python, why = system_python_without_the_transport()
    if system_python is None:
        pytest.skip(why)

    script = venvless_checkout_copy(tmp_path, SCRIPT)

    config = _make_repo(tmp_path, "8", marker=True)
    env = path_python3_shimmed_to(tmp_path, system_python)
    tried = subprocess.run(
        [system_python, "-c", "import sys; print(sys.executable)"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()

    with FakeEscalationMcp(reload_report(config_path=str(config))) as server:
        proc = _run(server, config, "8",
                    script=script, env=env)

    assert _failure(proc) == "transport_not_importable"
    assert not _converged_verdict_lines(proc)
    assert tried in proc.stderr, f"stderr={proc.stderr}"
