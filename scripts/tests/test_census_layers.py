"""The census modules form layers: each imports only census modules on a strictly lower layer."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

LEGIBILITY = Path(__file__).resolve().parents[1] / "legibility"

CENSUS_MODULES = sorted(path.stem for path in LEGIBILITY.glob("census*.py"))

LAYERS = {
    "census_findings": 0,
    "census_headroom": 0,
    "census_mining": 0,
    "census_state": 0,
    "census_verdicts": 0,
    "census_window": 0,
    "census_identity": 1,
    "census_report": 1,
    "census_verify": 1,
    "census_filing": 2,
    "census_trigger": 2,
    "census_run": 3,
    "census": 4,
}


def _census_imports(module: str) -> set[str]:
    tree = ast.parse((LEGIBILITY / f"{module}.py").read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                imported.add(parts[1] if parts[0] == "legibility" and len(parts) > 1 else parts[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            parts = node.module.split(".")
            if parts == ["legibility"]:
                imported.update(alias.name for alias in node.names)
            elif parts[0] == "legibility":
                imported.add(parts[1])
            else:
                imported.add(parts[0])
    return imported & set(CENSUS_MODULES)


def test_every_census_module_has_a_layer():
    assert sorted(LAYERS) == CENSUS_MODULES


@pytest.mark.parametrize("module", CENSUS_MODULES)
def test_census_layer_imports_only_lower_layers(module):
    assert module in LAYERS, f"{module} has no layer in LAYERS"
    for imported in sorted(_census_imports(module)):
        assert LAYERS[imported] < LAYERS[module], (
            f"{module} (layer {LAYERS[module]}) imports {imported} (layer {LAYERS[imported]}); "
            f"a census module may import only census modules on a strictly lower layer"
        )
