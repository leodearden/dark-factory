"""Guard: opted-in modules' type gates report a suppression that suppresses nothing.

WHAT IT ASSERTS (task 5086). For every module named in
``OPTED_IN_MODULE_PREFIXES``, pyright's ``reportUnnecessaryTypeIgnoreComment``
is live, at ERROR severity, in the ``[tool.pyright]`` table that module's
declared ``type_check_command`` resolves. It is a BEHAVIOURAL probe, not a
config-key read: a two-line file carrying one vestigial ``# type: ignore`` and
one vestigial ``# pyright: ignore[reportArgumentType]`` is checked with
``-p <that table's directory>``, and both lines must come back as this rule's
errors. A key assertion would stay green if the gate stopped reading that
table or pyright stopped honouring the key; the probe would not. The directory
is derived from the declared command (its ``--directory`` wrapper argument),
so the probe follows the gate if the gate's spelling changes. Both spellings
are probed because the rule does not depend on the rule code a suppression
names. Error, never warning: the declared gates fail on errors only.

OPT-IN CRITERION. A table is opted in only when its declared gate measures
clean after a small cleanup AND none of its flagged sites is an import pragma
whose resolution depends on venv contents — a cross-member import that
resolves only because verify runs ``uv sync --all-packages``, or an optional
third-party lazy import. Such a pragma is vestigial in the gate env and
load-bearing in a narrower one, so with the rule on no spelling of that line is
right in both.

WHY EACH OTHER TABLE STAYS OFF. The dated per-table counts are in task 5086's
record, not here, because they go stale with the next edit to any member.

* root — ``scripts/source_measures.py::_import_complexipy`` and the radon
  import beside it are lazy by design, so env-dependent; and the root table
  also governs every ad-hoc root-scoped pyright run over member files, which
  would then surface their sites.
  ``tests/scripts/test_no_vestigial_import_pragmas.py`` guards the import class
  under this table instead.
* shared — env-dependent cross-member imports in
  ``shared/tests/test_task_statuses.py::TestCrossPackageDriftGuardPlaceholder``.
* escalation — env-dependent ``orchestrator.*`` imports.
* fused-memory — volume alone; its import sites resolve via its own
  ``extraPaths``, so it is the natural next candidate.
* orchestrator — volume alone.

GOTCHA. pyright treats ``# type: ignore`` / ``# pyright: ignore`` appearing
anywhere after a ``#`` in a COMMENT as a live pragma, so prose mentioning one
in a comment is reported. Docstring mentions are string tokens and are not.

TO RE-MEASURE OR EXTEND. In a worktree with its own synced ``.venv``, under
``env -u VIRTUAL_ENV``, temporarily add
``reportUnnecessaryTypeIgnoreComment = "error"`` to the table, run the module's
declared ``type_check_command`` with ``--outputjson`` and count that rule.
Delete the vestigial sites (never narrow or carve out), then add the prefix to
``OPTED_IN_MODULE_PREFIXES``.

PLACEMENT. ``tests/scripts/`` carries its own module config, so a guard here
cannot be silenced by editing the command it asserts about — the reasoning in
``tests/scripts/test_module_type_check_invocation.py``'s PLACEMENT paragraph.
"""
from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Any

import pytest
from pyright_json import pyright_json_report
from verify_command_invariants import PYRIGHT, anchor_split, required_segment

if TYPE_CHECKING:
    from collections.abc import Callable

    from orchestrator.config import ModuleConfig

REPO_ROOT = pathlib.Path(__file__).parents[2]

OPTED_IN_MODULE_PREFIXES = ("cockpit", "dashboard", "sampler")

RULE = "reportUnnecessaryTypeIgnoreComment"

_PROBE_SOURCE = (
    "checked: int = 1  # type: ignore\n"
    "also_checked: int = 2  # pyright: ignore[reportArgumentType]\n"
)
_EXPECTED_ERRORS = [(1, RULE), (2, RULE)]


def _pyright_config_dir(command: str, *, label: str) -> pathlib.Path:
    """The directory whose ``pyproject.toml`` the type gate *command* resolves.

    ``uv run --directory <x> pyright ...`` (or ``--directory=<x>``) runs
    pyright from ``<x>``; a gate with no ``--directory`` runs from the repo
    root (e.g. scripts' ``uv run --project shared pyright scripts/``, where
    ``--project`` selects only the environment). Only the wrapper's tokens are
    read: after the anchor, the same spelling would be pyright's own argument.
    """
    segment = required_segment(command, PYRIGHT, label=label)
    pre, _ = anchor_split(segment, PYRIGHT, label=label)
    for index, token in enumerate(pre):
        if token == "--directory":
            return REPO_ROOT / pre[index + 1]
        if token.startswith("--directory="):
            return REPO_ROOT / token.removeprefix("--directory=")
    return REPO_ROOT


def _error_rules_by_line(payload: dict[str, Any]) -> list[tuple[int, str]]:
    """Sorted ``(1-based line, rule)`` per error-severity diagnostic."""
    return sorted(
        (int(item["range"]["start"]["line"]) + 1, item.get("rule", "<no rule>"))
        for item in payload.get("generalDiagnostics", [])
        if item.get("severity") == "error"
    )


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("uv run --directory dashboard pyright src/ tests/", REPO_ROOT / "dashboard"),
        ("uv run --directory=dashboard pyright src/ tests/", REPO_ROOT / "dashboard"),
        ("uv run --project shared pyright scripts/", REPO_ROOT),
    ],
    ids=["space-separated", "equals-joined", "no-directory"],
)
def test_config_dir_follows_either_directory_spelling(
    command: str, expected: pathlib.Path
) -> None:
    assert _pyright_config_dir(command, label="probe") == expected


@pytest.mark.parametrize("prefix", OPTED_IN_MODULE_PREFIXES, ids=OPTED_IN_MODULE_PREFIXES)
def test_opted_in_type_gate_reports_a_vestigial_suppression(
    prefix: str,
    tmp_path: pathlib.Path,
    discover_module_configs: Callable[[], dict[str, ModuleConfig]],
) -> None:
    discovered = discover_module_configs()
    assert prefix in discovered, (
        f"opted-in module {prefix!r} has no discovered module config, so its "
        f"type gate cannot be probed; discovered: {sorted(discovered)}"
    )
    label = f"{prefix} type_check_command"
    command = discovered[prefix].type_check_command
    assert command, f"{label} is not declared, so there is no gate to probe"
    config_dir = _pyright_config_dir(command, label=label)
    probe = tmp_path / "probe.py"
    probe.write_text(_PROBE_SOURCE, encoding="utf-8")

    payload = pyright_json_report([probe], project_dir=config_dir)

    analyzed = payload.get("summary", {}).get("filesAnalyzed")
    assert analyzed == 1, (
        f"pyright analysed {analyzed} file(s) for the one-file probe under "
        f"{config_dir} — any other count makes the verdict below vacuous"
    )
    errors = _error_rules_by_line(payload)
    assert errors == _EXPECTED_ERRORS, (
        f"{prefix}'s declared type gate does not report vestigial suppressions "
        f"as errors: got {errors}, expected {_EXPECTED_ERRORS}. Add "
        f'`{RULE} = "error"` to the [tool.pyright] table in '
        f"{config_dir / 'pyproject.toml'}; never weaken it to warning, because "
        f"the declared gates fail on errors only."
    )
