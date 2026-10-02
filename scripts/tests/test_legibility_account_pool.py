"""Tests for scripts/legibility/account_pool.py — the builder of the fleet's
shared multi-account ``UsageGate`` for a scripts/ process (tasks 5488, 6042).

WHAT THIS MODULE IS FOR, and therefore what these tests pin: the trickle
used to ride whatever login ``~/.claude`` happened to hold, so one capped
account deferred an entire night while six other accounts sat idle.
``build_pool`` builds the gate from the roster and the unit's ``.env`` alone,
never adopts the operator's own login, and never raises. Rotation, cap and
auth handling are the shared runner's, pinned in
test_legibility_session_runner.py.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import account_pool as mod
import pytest

from shared import usage_gate as usage_gate_mod


@pytest.fixture(autouse=True)
def _restore_environ():
    """Snapshot and restore ``os.environ`` around every test in this module.

    Every ``build_pool`` call pops ``ANTHROPIC_API_KEY`` and its
    ``load_dotenv`` sets variables, both directly on ``os.environ``;
    ``monkeypatch`` reverses neither, so without this one test's effects
    would reach every later test in the process.
    """
    before = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(before)


# ---------------------------------------------------------------------------
# step-15: build_pool() — a REAL UsageGate from nothing but an accounts file.
#
# The trickle's original excuse for riding ~/.claude was that the
# orchestrator config is unreachable from its interpreter. Only the
# orchestrator YAML is: UsageCapConfig(accounts_file=...) needs nothing else,
# which is the same two-liner fused_memory/config/schema.py already uses.
# ---------------------------------------------------------------------------

_ROSTER_YAML = """\
accounts:
  - name: max-b
    oauth_token_env: CLAUDE_OAUTH_TOKEN_B
  - name: max-c
    oauth_token_env: CLAUDE_OAUTH_TOKEN_C
  - name: max-d
    oauth_token_env: CLAUDE_OAUTH_TOKEN_D
"""


@pytest.fixture
def roster_file(tmp_path):
    path = tmp_path / "usage-accounts.yaml"
    path.write_text(_ROSTER_YAML)
    return path


@pytest.fixture
def empty_env_file(tmp_path):
    """A guaranteed-empty ``.env`` for every ``build_pool`` call that is not
    itself testing dotenv loading.

    ``build_pool``'s default ``env_file=None`` resolves to
    ``_REPO_ROOT / ".env"`` -- inert in a worktree (no such file) but, run
    from the main checkout, a file carrying real ``CLAUDE_OAUTH_TOKEN_*``
    and ``ANTHROPIC_API_KEY``. Passing this fixture instead of leaving
    ``env_file`` unset is what keeps a `build_pool()` call hermetic to which
    checkout the suite happens to run from.
    """
    path = tmp_path / "empty.env"
    path.write_text("")
    return path


def _set_pool_tokens(monkeypatch, *letters):
    for letter in letters:
        monkeypatch.setenv(f"CLAUDE_OAUTH_TOKEN_{letter}", f"tok-{letter.lower()}")


def test_build_pool_resolves_the_roster_in_file_order(
    roster_file, monkeypatch, empty_env_file,
):
    _set_pool_tokens(monkeypatch, "B", "C", "D")

    gate = mod.build_pool(accounts_file=str(roster_file), env_file=str(empty_env_file))

    assert gate.account_count == 3
    assert [a.name for a in gate._accounts] == ["max-b", "max-c", "max-d"], (
        "order is the failover order — config/usage-accounts.yaml says so in "
        "its own header"
    )


def test_default_accounts_file_resolves_relative_to_this_checkout():
    """Never a hardcoded absolute: a copy of this script running from a
    worktree must read ITS OWN roster, the same reason coder.py resolves
    `shared` __file__-relatively (tasks 2881/2882/3329)."""
    repo_root = Path(mod.__file__).resolve().parents[2]

    assert mod.default_accounts_file() == repo_root / "config" / "usage-accounts.yaml"
    assert mod.default_accounts_file().exists(), (
        "the default must point at a roster that actually exists in this "
        "checkout — a missing accounts_file degrades to an EMPTY pool with "
        "only a warning, which is the silent ~/.claude fallback again"
    )


def test_build_pool_actually_uses_the_default_accounts_file_when_nothing_else_is_set(
    monkeypatch, tmp_path, empty_env_file,
):
    """With neither ``accounts_file`` nor ``USAGE_ACCOUNTS_FILE`` set,
    ``build_pool`` reads ``default_accounts_file()`` -- the test above never
    calls ``build_pool``. The roster's one account exists in no other roster, so
    only a pool built from ``default_accounts_file()`` can resolve it -- a
    ``max-*`` name would also resolve from the real
    ``config/usage-accounts.yaml`` and prove nothing.
    """
    roster = tmp_path / "default-roster.yaml"
    roster.write_text(
        "accounts:\n"
        "  - name: only-in-the-default-roster\n"
        "    oauth_token_env: CLAUDE_OAUTH_TOKEN_DEFAULT_ROSTER_PROBE\n"
    )
    monkeypatch.delenv("USAGE_ACCOUNTS_FILE", raising=False)
    monkeypatch.setattr(mod, "default_accounts_file", lambda: roster)
    monkeypatch.setenv("CLAUDE_OAUTH_TOKEN_DEFAULT_ROSTER_PROBE", "tok-probe")

    gate = mod.build_pool(env_file=str(empty_env_file))

    assert [a.name for a in gate._accounts] == ["only-in-the-default-roster"], (
        "build_pool must resolve the roster through default_accounts_file(), "
        "the only source left once accounts_file and USAGE_ACCOUNTS_FILE are "
        "both unset"
    )


def test_build_pool_honours_the_USAGE_ACCOUNTS_FILE_override(
    roster_file, monkeypatch, empty_env_file,
):
    """The fleet convention, shared with fused_memory/config/schema.py and
    scripts/run_vllm_eval.py."""
    _set_pool_tokens(monkeypatch, "B", "C", "D")
    monkeypatch.setenv("USAGE_ACCOUNTS_FILE", str(roster_file))

    gate = mod.build_pool(env_file=str(empty_env_file))

    assert [a.name for a in gate._accounts] == ["max-b", "max-c", "max-d"]


def test_build_pool_hands_the_validator_an_ABSOLUTE_path(
    roster_file, monkeypatch, empty_env_file,
):
    """A relative accounts_file is `.resolve()`d against the CWD by
    UsageCapConfig's validator, and a path that misses degrades to an empty
    pool with only a warning. The trickle's CWD is the systemd unit's, not
    the repo's, so a relative path would resolve somewhere arbitrary."""
    _set_pool_tokens(monkeypatch, "B", "C", "D")

    assert mod.default_accounts_file().is_absolute()

    gate = mod.build_pool(accounts_file=str(roster_file), env_file=str(empty_env_file))
    assert gate.account_count == 3


def test_build_pool_loads_dotenv_before_building_the_gate(tmp_path, monkeypatch):
    """ORDER IS THE WHOLE POINT: _init_accounts reads os.environ EAGERLY at
    construction, so a .env loaded afterwards resolves nothing and the pool
    silently falls back to ~/.claude — today's broken behaviour."""
    roster = tmp_path / "roster.yaml"
    roster.write_text(_ROSTER_YAML)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "CLAUDE_OAUTH_TOKEN_B=tok-from-dotenv-b\n"
        "CLAUDE_OAUTH_TOKEN_C=tok-from-dotenv-c\n"
        "CLAUDE_OAUTH_TOKEN_D=tok-from-dotenv-d\n"
    )
    for letter in ("B", "C", "D"):
        monkeypatch.delenv(f"CLAUDE_OAUTH_TOKEN_{letter}", raising=False)

    gate = mod.build_pool(accounts_file=str(roster), env_file=str(env_file))

    assert gate.account_count == 3
    assert gate._accounts[0].token == "tok-from-dotenv-b", (
        "the token must have come from the .env — if the gate were built "
        "first, every account would have resolved token-less"
    )


def test_build_pool_does_not_let_the_dotenv_undo_the_units_api_key_strip(
    tmp_path, monkeypatch,
):
    """The .env that carries the POOL also carries ANTHROPIC_API_KEY.

    `load_dotenv` sets any variable not already present, and "not present"
    is exactly the state `UnsetEnvironment=ANTHROPIC_API_KEY` leaves the
    trickle in — so the load above would hand the key straight back and
    silently defeat the unit's own directive. The strip belongs HERE, at the
    one point that re-introduces it; the shared runner's own strip
    (``shared.cli_invoke._invoke_claude``) cannot cover this, because it only
    shapes the children it spawns, never the census process that inherits
    this environment.
    """
    roster = tmp_path / "roster.yaml"
    roster.write_text(_ROSTER_YAML)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "ANTHROPIC_API_KEY=sk-ant-must-not-come-back\n"
        "CLAUDE_OAUTH_TOKEN_B=tok-from-dotenv-b\n"
        "CLAUDE_OAUTH_TOKEN_C=tok-from-dotenv-c\n"
        "CLAUDE_OAUTH_TOKEN_D=tok-from-dotenv-d\n"
    )
    # delenv rather than a bare assertion, and it is what makes the test safe
    # to run anywhere: monkeypatch restores whatever the ambient value was,
    # including after build_pool pops it for real.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    for letter in ("B", "C", "D"):
        monkeypatch.delenv(f"CLAUDE_OAUTH_TOKEN_{letter}", raising=False)

    gate = mod.build_pool(accounts_file=str(roster), env_file=str(env_file))

    assert "ANTHROPIC_API_KEY" not in os.environ, (
        "the .env load put the API key back into this process — every child "
        "that INHERITS this environment (the census launch) would carry the "
        "key's identity while the pool's failover still looked like it worked"
    )
    assert gate.account_count == 3, (
        "the strip must remove ONE variable, not disable the dotenv load the "
        "whole pool depends on"
    )


def test_build_pool_logs_the_resolved_roster_but_never_a_token(
    roster_file, monkeypatch, caplog, empty_env_file,
):
    _set_pool_tokens(monkeypatch, "B", "C", "D")

    with caplog.at_level("INFO", logger="legibility.account_pool"):
        mod.build_pool(accounts_file=str(roster_file), env_file=str(empty_env_file))

    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "max-b" in logged and "max-d" in logged, logged
    assert "3" in logged, logged
    assert "tok-b" not in logged and "tok-d" not in logged, (
        f"a token must NEVER reach the journal; got {logged!r}"
    )


def _module_warnings(caplog) -> list[str]:
    """WARNING records THIS module actually emitted.

    ``caplog.at_level(logger=...)`` only raises that logger's level; per the
    pytest docs it does NOT scope capture to that logger. ``UsageGate.
    _init_accounts`` logs its own ``Account 'max-b': env var
    CLAUDE_OAUTH_TOKEN_B not set — skipping`` WARNING for every token-less
    account, which alone would satisfy an unfiltered ``caplog.records``
    assertion regardless of whether ``build_pool``'s own warning fired.
    Filtering by ``r.name`` is what proves the warning came from
    ``account_pool.build_pool`` and not merely from the gate underneath it.
    """
    return [
        r.getMessage() for r in caplog.records
        if r.levelname == "WARNING" and r.name == "legibility.account_pool"
    ]


def test_build_pool_warns_LOUDLY_when_it_resolves_no_accounts(
    tmp_path, monkeypatch, caplog, empty_env_file,
):
    """The degradation that must never be silent: no token env var resolves
    AND no ``~/.claude/.credentials.json`` fallback exists either, so the
    gate ends up with literally zero accounts. It has to be VISIBLE, or this
    task's fix silently un-does itself the day a token env var is dropped
    from the unit.

    ``CREDENTIALS_PATH`` is pinned to a path that does not exist so this
    arm's premise (zero accounts, not the OTHER zero-usable-account arm
    below) does not depend on whether the host running the suite happens to
    hold real Claude credentials.
    """
    roster = tmp_path / "roster.yaml"
    roster.write_text(_ROSTER_YAML)
    for letter in ("B", "C", "D"):
        monkeypatch.delenv(f"CLAUDE_OAUTH_TOKEN_{letter}", raising=False)
    monkeypatch.setattr(
        usage_gate_mod, "CREDENTIALS_PATH", tmp_path / "no-such-credentials.json",
    )

    with caplog.at_level("WARNING", logger="legibility.account_pool"):
        gate = mod.build_pool(accounts_file=str(roster), env_file=str(empty_env_file))

    assert gate.account_count == 0, (
        f"this arm's premise: no token env vars AND no default credential "
        f"on disk; got {gate.account_count} accounts"
    )
    warnings = _module_warnings(caplog)
    assert warnings, (
        "a pool that resolved no real accounts must SAY so — silently "
        "returning the ~/.claude fallback is the defect this task removes"
    )
    assert any("resolved NO usable accounts" in w for w in warnings), (
        f"the module's own distinctive phrase must be present -- the gate "
        f"logs its own per-account skip warning too, which this assertion "
        f"must not be satisfied by; got {warnings}"
    )
    assert any("max-b" in w for w in warnings), (
        f"name the accounts it could not resolve, so an operator knows which "
        f"env var is missing; got {warnings}"
    )


def test_build_pool_never_adopts_the_operator_login_and_warns_LOUDLY(
    tmp_path, monkeypatch, caplog, empty_env_file,
):
    """The exact pre-5488 defect, now closed at the gate (task 6042): every
    ``CLAUDE_OAUTH_TOKEN_*`` is missing but ``~/.claude/.credentials.json``
    exists. The gate is built with that fallback disabled, so the pool is
    EMPTY rather than one account named 'default' riding the operator's own
    login -- and the emptiness is as loud as the no-credentials arm above.
    """
    roster = tmp_path / "roster.yaml"
    roster.write_text(_ROSTER_YAML)
    for letter in ("B", "C", "D"):
        monkeypatch.delenv(f"CLAUDE_OAUTH_TOKEN_{letter}", raising=False)
    cred_file = tmp_path / "credentials.json"
    cred_file.write_text(json.dumps({"accessToken": "tok-default"}))
    monkeypatch.setattr(usage_gate_mod, "CREDENTIALS_PATH", cred_file)

    with caplog.at_level("WARNING", logger="legibility.account_pool"):
        gate = mod.build_pool(accounts_file=str(roster), env_file=str(empty_env_file))

    assert gate.account_count == 0
    warnings = _module_warnings(caplog)
    assert any("resolved NO usable accounts" in w for w in warnings), warnings
    assert any("max-b" in w for w in warnings), warnings


# ---------------------------------------------------------------------------
# task 5635 hole 1: a roster that cannot be loaded is a loud empty pool, never
# a raise. build_pool runs while the night's runner is being built; an escaped
# exception there used to take the whole run down unrecorded.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "roster_text", ["accounts: [unclosed\n", "- name: max-b\n"], ids=["not-yaml", "a-list"],
)
def test_a_roster_that_cannot_be_loaded_is_a_loud_empty_pool_never_a_raise(
    tmp_path, caplog, empty_env_file, roster_text,
):
    roster = tmp_path / "corrupt-roster.yaml"
    roster.write_text(roster_text)

    with caplog.at_level("WARNING", logger="legibility.account_pool"):
        gate = mod.build_pool(accounts_file=str(roster), env_file=str(empty_env_file))

    assert gate.account_count == 0
    warnings = _module_warnings(caplog)
    assert any(str(roster) in w for w in warnings), (
        f"name the roster the operator has to fix; got {warnings}"
    )
    assert any("Error" in w for w in warnings), (
        f"name what was wrong with it; got {warnings}"
    )
