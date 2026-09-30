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

CENSUS (task 5086, pyright 1.1.408, vestigial sites / files per table). The
full census was measured 2026-09-28 by adding the rule to all eight tables and
running all nine declared module gates; the opted-in tables were re-measured
2026-09-30 before their cleanup. Opted in: cockpit 0; sampler 1; dashboard 9 in
5 files. Staying OFF:

* root, 15 (scripts 6 + tests/scripts 9) — ``scripts/merge_lane_metrics.py::
  _import_complexipy`` and the radon import beside it are lazy by design, so
  env-dependent; and the root table also governs every ad-hoc root-scoped
  pyright run over member files, which would then surface their sites.
  ``tests/scripts/test_no_vestigial_import_pragmas.py`` guards the import class
  under this table instead.
* shared, 39 — 3 env-dependent cross-member imports in
  ``shared/tests/test_task_statuses.py::TestCrossPackageDriftGuardPlaceholder``.
* escalation, 184 — ~115 env-dependent ``orchestrator.*`` imports.
* fused-memory, 307 in 76 files — count alone; its import sites resolve via its
  own ``extraPaths``, so it is the natural next candidate.
* orchestrator, 1547 in 173 files — count alone.

GOTCHA. pyright treats ``# type: ignore`` / ``# pyright: ignore`` appearing
anywhere after a ``#`` in a COMMENT as a live pragma, so prose mentioning one
in a comment is reported (5 such sites were measured). Docstring mentions are
string tokens and are not.

TO RE-MEASURE OR EXTEND. In a worktree with its own synced ``.venv``, under
``env -u VIRTUAL_ENV``, temporarily add
``reportUnnecessaryTypeIgnoreComment = "error"`` to the table, run the module's
declared ``type_check_command`` with ``--outputjson`` and count that rule.
Delete the vestigial sites (never narrow or carve out), then add the prefix to
``OPTED_IN_MODULE_PREFIXES``.

PLACEMENT. ``tests/scripts/`` carries its own module config, so a guard here
cannot be silenced by editing the command it asserts about — the reasoning in
``tests/scripts/test_module_type_check_invocation.py``'s PLACEMENT paragraph.
The pyright subprocess discipline is replicated from
``tests/scripts/test_root_py_type_gate.py::_pyright_report``, not imported: this
directory bans importing a sibling test file.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
from typing import TYPE_CHECKING, Any

import pytest
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


def _pyright_config_dir(mc: ModuleConfig) -> pathlib.Path:
    """The directory whose ``pyproject.toml`` *mc*'s declared type gate resolves.

    ``uv run --directory <x> pyright ...`` runs pyright from ``<x>``; a gate
    with no ``--directory`` runs from the repo root (e.g. scripts'
    ``uv run --project shared pyright scripts/``, where ``--project`` selects
    only the environment).
    """
    label = f"{mc.prefix} type_check_command"
    assert mc.type_check_command, f"{label} is not declared, so there is no gate to probe"
    segment = required_segment(mc.type_check_command, PYRIGHT, label=label)
    pre, _ = anchor_split(segment, PYRIGHT, label=label)
    if "--directory" in pre:
        return REPO_ROOT / pre[pre.index("--directory") + 1]
    return REPO_ROOT


def _probe_report(config_dir: pathlib.Path, probe: pathlib.Path) -> dict[str, Any]:
    """The parsed ``pyright --outputjson -p <config_dir> <probe>`` payload."""
    proc = subprocess.run(
        [
            sys.executable, "-m", "pyright", "--outputjson",
            "-p", str(config_dir), str(probe),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    # No try/except-and-skip: a missing or broken pyright must FAIL this guard.
    assert proc.returncode in (0, 1), (
        f"`pyright --outputjson` exited {proc.returncode} — expected 0 (clean) "
        f"or 1 (diagnostics). A missing pyright module or a bad invocation must "
        f"fail this guard rather than skip it; stderr: {proc.stderr.strip()!r}"
    )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise AssertionError(
            f"could not parse `pyright --outputjson` output: {exc}; "
            f"stdout: {proc.stdout[:500]!r}; stderr: {proc.stderr.strip()!r}"
        ) from exc


def _error_rules_by_line(payload: dict[str, Any]) -> list[tuple[int, str]]:
    """Sorted ``(1-based line, rule)`` per error-severity diagnostic."""
    return sorted(
        (int(item["range"]["start"]["line"]) + 1, item.get("rule", "<no rule>"))
        for item in payload.get("generalDiagnostics", [])
        if item.get("severity") == "error"
    )


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
    config_dir = _pyright_config_dir(discovered[prefix])
    probe = tmp_path / "probe.py"
    probe.write_text(_PROBE_SOURCE, encoding="utf-8")

    payload = _probe_report(config_dir, probe)

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
