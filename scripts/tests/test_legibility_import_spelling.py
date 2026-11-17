"""Every scripts/legibility module is reached under one name, ``legibility.<mod>``.

A bare sibling name (``import codebook``) resolves only because a direct run
puts scripts/legibility at sys.path[0]. It then builds a second module object
for a file already loaded as ``legibility.<mod>``, so the two copies' classes
are different classes and an ``except`` written against one misses the other.
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest
from cli_subprocess_timeout import cli_timeout_from_env
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

LEGIBILITY = Path(__file__).resolve().parents[1] / "legibility"

STEMS = frozenset(path.stem for path in LEGIBILITY.glob("*.py"))

PROBE_TIMEOUT = cli_timeout_from_env("LEGIBILITY_IMPORT_PROBE_TIMEOUT_SECS")

PROBE = """
import importlib, json, sys
from pathlib import Path
legibility = Path(sys.argv[1])
sys.path[0:0] = [str(legibility.parent), str(legibility)]
stems = {path.stem for path in legibility.glob("*.py")}
for stem in sorted(stems):
    importlib.import_module(f"legibility.{stem}")
print(json.dumps(sorted(stems & set(sys.modules))))
"""


def _second_spellings(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bare = any(alias.name.split(".")[0] in STEMS for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            bare = node.level > 0 or (node.module or "").split(".")[0] in STEMS
        else:
            continue
        if bare:
            offenders.append(f"{path.relative_to(LEGIBILITY.parent)}:{node.lineno}: {ast.unparse(node)}")
    return offenders


def test_every_legibility_module_imports_its_siblings_through_the_package():
    offenders = [line for path in sorted(LEGIBILITY.glob("*.py")) for line in _second_spellings(path)]
    assert offenders == [], (
        "one module object per source file: import a legibility sibling as "
        "`from legibility import <mod>`, never by its bare name or relatively. A bare "
        "name resolves only because a direct run puts scripts/legibility at sys.path[0], "
        "and it builds a second copy of the module:\n" + "\n".join(offenders)
    )


def test_a_direct_run_layout_loads_each_legibility_module_once(tmp_path):
    probe = subprocess.run(
        [sys.executable, "-I", "-c", PROBE, str(LEGIBILITY)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=PROBE_TIMEOUT,
    )
    assert probe.returncode == 0, probe.stderr
    assert json.loads(probe.stdout) == [], (
        "these legibility modules were also loaded under their bare name, "
        "a second module object beside legibility.<mod>"
    )
