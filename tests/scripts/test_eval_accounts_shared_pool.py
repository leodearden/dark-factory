"""Eval launchers must roster the SHARED fleet pool — no interactive-account injection.

Task 4945, discharging the follow-up filed from the 2026-08-30 ruling
(task 4741).

WHAT THE RULING SETTLED. Account A ("max-a") is Leo's own INTERACTIVE
account. His sessions exhaust its weekly cap most weeks, so it never was
the uncapped private reserve the eval launchers treated it as. It is
deliberately absent from ``config/usage-accounts.yaml`` so orchestrator
automation cannot spend it — and an eval run that appends it back to its
own roster defeats exactly that. Evals have NO dedicated account: they
draw on the shared pool like the rest of the fleet and tolerate cap/429
events via ``invoke_with_cap_retry``'s 48h patience plus session
``--resume`` (``orchestrator.evals.runner``).

WHY THIS GUARD LIVES HERE AND NOT ONLY IN THE LAUNCHER'S UNIT TESTS.
``scripts/rerun_cap_tainted.sh`` duplicated the injection INLINE, as a
python heredoc inside a command substitution — a second copy of the same
premise in a language with no unit-test harness of its own. Fixing only
the Python launcher would have left the shell copy live and unguarded,
which is how the premise got duplicated in the first place. This file is
the one place that sweeps BOTH launchers plus the roster they point at,
so re-adding A to any of the three goes red rather than going unnoticed.

The ``USAGE_ACCOUNTS_FILE`` override itself is NOT under indictment and
is asserted PRESENT below: it is the established cross-project seam for
"which roster does this run use" (``shared.config_models.UsageCapConfig``
reads it, and so does reify's orchestrator config). The defect was the
roster it pointed at, not the pointing.

ASSERTED ON EXECUTABLE CONTENT, NOT ON PROSE. The Python launcher is
read through ``ast`` (so comments are absent by construction and
docstrings are excluded explicitly) and the shell script has its
whole-line comments stripped. A comment may therefore still NAME account
A to explain why it is excluded — which the launchers do — while any
attempt to actually roster it fails. A guard that banned the string
outright would forbid the explanation along with the defect, and the
explanation is the thing most likely to stop the next reinstatement.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).parents[2]

SHELL_LAUNCHER = REPO_ROOT / "scripts" / "rerun_cap_tainted.sh"
PY_LAUNCHER = REPO_ROOT / "scripts" / "run_vllm_eval.py"
SHARED_ROSTER = REPO_ROOT / "config" / "usage-accounts.yaml"

# The interactive account, by both of the names a launcher could reach it by:
# the roster entry name and the env var holding its OAuth token.
INTERACTIVE_ACCOUNT_LITERALS = ("max-a", "CLAUDE_OAUTH_TOKEN_A")

_SHARED_ROSTER_SUFFIX = ("config", "usage-accounts.yaml")


def _shell_code(path: pathlib.Path) -> str:
    """The script body with whole-line comments removed.

    Only WHOLE-LINE comments are stripped, never a mid-line ``#``: a bare
    split on ``#`` would cut inside string literals and command
    substitutions and could silently delete the very code under test.
    Nothing this guard looks for hides behind a trailing comment.
    """
    return "\n".join(
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    )


def _py_string_constants(path: pathlib.Path) -> list[str]:
    """Every string literal in the module EXCEPT docstrings.

    Comments never enter the AST at all, so they are excluded for free.
    Docstrings are excluded deliberately — see the module docstring on why
    prose must stay free to name the account it is warning about.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))

    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            body = getattr(node, "body", None)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstrings.add(id(body[0].value))

    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


def _exported_usage_accounts_file(shell_code: str) -> str:
    """The value of the script's ``export USAGE_ACCOUNTS_FILE=...`` line."""
    matches = re.findall(
        r"^\s*export\s+USAGE_ACCOUNTS_FILE=(.+)$", shell_code, flags=re.MULTILINE
    )
    assert len(matches) == 1, (
        f"{SHELL_LAUNCHER.name} must export USAGE_ACCOUNTS_FILE exactly once "
        f"(found {len(matches)}: {matches!r}). That export is how an eval run "
        "selects its account roster; zero exports silently falls back to the "
        "orchestrator config's default, which is a HARDCODED absolute path "
        "into the MAIN checkout and therefore wrong for a worktree-launched "
        "run, and more than one makes the effective roster order-dependent."
    )
    return matches[0].strip().strip('"').strip("'")


class TestShellLauncherRostersTheSharedPool:
    """``scripts/rerun_cap_tainted.sh`` must point at the shared roster."""

    def test_exports_the_shared_roster_path(self) -> None:
        value = _exported_usage_accounts_file(_shell_code(SHELL_LAUNCHER))

        assert pathlib.PurePosixPath(value).parts[-2:] == _SHARED_ROSTER_SUFFIX, (
            f"{SHELL_LAUNCHER.name} exports USAGE_ACCOUNTS_FILE={value!r}, which "
            "does not resolve to config/usage-accounts.yaml. Task 4945: this "
            "used to be a command substitution generating a tempfile of "
            "'shared pool + max-a'; the roster must now be the shared pool "
            "file itself."
        )

    def test_generates_no_roster_of_its_own(self) -> None:
        code = _shell_code(SHELL_LAUNCHER)

        for marker in ("mkstemp", "tempfile", "yaml.safe_dump"):
            assert marker not in code, (
                f"{SHELL_LAUNCHER.name} still contains {marker!r}, i.e. it is "
                "still SYNTHESISING an accounts roster rather than pointing at "
                "the shared one. The inline python heredoc that did this was "
                "the shell copy of the retired account-A injection (task 4945)."
            )

    def test_names_no_interactive_account_in_executable_code(self) -> None:
        code = _shell_code(SHELL_LAUNCHER)

        for literal in INTERACTIVE_ACCOUNT_LITERALS:
            assert literal not in code, (
                f"{SHELL_LAUNCHER.name} references {literal!r} in executable "
                "code. Account A is reserved for INTERACTIVE use only (ruling "
                "2026-08-30, task 4741) — an eval run that rosters it, or "
                "bootstraps from its token, spends the one account Leo's own "
                "sessions depend on. Comments may still name it to explain the "
                "exclusion; only executable code is checked here."
            )


class TestPythonLauncherRostersTheSharedPool:
    """``scripts/run_vllm_eval.py`` must not reintroduce the injection.

    The behavioural end of this — that ``build_eval_env()`` resolves to the
    shared roster verbatim — is covered by
    ``orchestrator/tests/test_run_vllm_eval.py::TestBuildEvalEnvSharedPool``,
    which calls the function. This is the cheap companion assertion that
    the interactive account is not reachable from the module's code at all,
    swept together with its shell sibling so both launchers are held to one
    contract in one place.
    """

    def test_names_no_interactive_account_in_executable_code(self) -> None:
        constants = _py_string_constants(PY_LAUNCHER)

        for literal in INTERACTIVE_ACCOUNT_LITERALS:
            offenders = [c for c in constants if literal in c]
            assert not offenders, (
                f"{PY_LAUNCHER.name} carries the string literal {literal!r} "
                f"outside a docstring ({offenders!r}). Account A is reserved "
                "for INTERACTIVE use only (ruling 2026-08-30, task 4741): the "
                "launcher must neither roster it nor seed "
                "CLAUDE_CODE_OAUTH_TOKEN from it."
            )


class TestSharedRosterExcludesTheInteractiveAccount:
    """The roster both launchers now point at must stay account-A-free.

    Without this, the guards above would be satisfiable while the defect
    was fully restored one file over: pointing at the shared roster only
    protects the interactive account for as long as the shared roster does
    not list it.
    """

    def test_roster_parses_and_is_non_empty(self) -> None:
        assert SHARED_ROSTER.is_file(), (
            f"{SHARED_ROSTER} does not exist — both eval launchers now export "
            "USAGE_ACCOUNTS_FILE pointing at it, so a missing file leaves "
            "every eval run rostered against nothing"
        )
        loaded = yaml.safe_load(SHARED_ROSTER.read_text(encoding="utf-8"))
        accounts = (loaded or {}).get("accounts")
        assert accounts, (
            f"{SHARED_ROSTER} declares no accounts, so every assertion below "
            f"would pass VACUOUSLY and every eval run would have no pool to "
            f"draw on: {loaded!r}"
        )

    @pytest.mark.parametrize("field", ["name", "oauth_token_env"])
    def test_roster_excludes_the_interactive_account(self, field: str) -> None:
        loaded = yaml.safe_load(SHARED_ROSTER.read_text(encoding="utf-8"))
        values = [a.get(field) for a in (loaded or {}).get("accounts", [])]

        for literal in INTERACTIVE_ACCOUNT_LITERALS:
            assert literal not in values, (
                f"{SHARED_ROSTER.name} lists {literal!r} under {field!r}. "
                "Account A is reserved for INTERACTIVE use only (ruling "
                "2026-08-30, task 4741) and must not be in the shared pool: "
                "orchestrator automation and eval runs both draw on this file, "
                f"so an entry here spends it fleet-wide. Rostered: {values!r}"
            )


class TestGuardReadsExecutableCodeOnly:
    """The two extractors above must tolerate prose and catch code.

    Without these, the comment-stripping and docstring-exclusion are
    UNVERIFIED claims: neither launcher happens to spell ``max-a`` in
    prose at every moment, so a broken extractor — one returning nothing,
    say — would leave every assertion above passing vacuously while
    checking nothing at all. Pinned on fixtures rather than on the real
    launchers so the guarantee holds no matter how those files' comments
    are later reworded.
    """

    def test_shell_prose_is_tolerated_but_code_is_not(self, tmp_path) -> None:
        script = tmp_path / "s.sh"
        script.write_text(
            "#!/bin/bash\n"
            "# max-a is excluded here on purpose (task 4741).\n"
            "  # indented prose may name CLAUDE_OAUTH_TOKEN_A too\n"
            'export USAGE_ACCOUNTS_FILE=/repo/config/usage-accounts.yaml\n'
        )
        code = _shell_code(script)

        assert "max-a" not in code and "CLAUDE_OAUTH_TOKEN_A" not in code, (
            "whole-line comments, indented ones included, must be stripped so "
            f"the guard cannot be tripped by its own explanation: {code!r}"
        )
        assert "export USAGE_ACCOUNTS_FILE" in code, (
            f"stripping must not eat executable lines: {code!r}"
        )

        script.write_text("accounts.append({'name': 'max-a'})\n")
        assert "max-a" in _shell_code(script), (
            "an inline roster append is executable code and must survive "
            "stripping — otherwise the shell guard is blind to the defect"
        )

    def test_py_docstrings_are_tolerated_but_code_is_not(self, tmp_path) -> None:
        module = tmp_path / "m.py"
        module.write_text(
            '"""Module prose naming max-a and CLAUDE_OAUTH_TOKEN_A."""\n'
            "\n"
            "\n"
            "def f():\n"
            '    """Function prose naming max-a."""\n'
            '    return "shared-pool"\n'
        )

        assert _py_string_constants(module) == ["shared-pool"], (
            "docstrings at module and function scope must be excluded while "
            "ordinary string literals survive; got "
            f"{_py_string_constants(module)!r}"
        )

        module.write_text('ACCOUNTS = [{"name": "max-a"}]\n')
        assert "max-a" in _py_string_constants(module), (
            "a rostered account name is an ordinary string constant and must "
            "be reported — otherwise the launcher guard is blind to the defect"
        )
