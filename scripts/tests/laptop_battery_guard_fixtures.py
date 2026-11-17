"""Shared fakes for the laptop battery guard suites (task 9051):
test_laptop_battery_guard.py and test_install_laptop_battery_guard.py.

Every fake power_supply tree carries ACAD online=1 and a USB-C source node
reading online=1, the readings that stayed up while the battery ran flat on
2026-11-08 and 2026-11-10. So each WARN and POWEROFF case built on it also
proves that AC presence cannot suppress the guard.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path


def _write_node(node: Path, files: Mapping[str, str]) -> None:
    node.mkdir(parents=True)
    for name, value in files.items():
        (node / name).write_text(f"{value}\n")


def power_supply(
    tmp_path: Path,
    batteries: Mapping[str, tuple[str, int | str]],
    extra: Iterable[tuple[str, Mapping[str, str]]] = (),
) -> Path:
    """A fake /sys/class/power_supply under *tmp_path*: AC always online,
    plus *batteries*.

    *batteries* maps a BAT* name to (status, capacity). *extra* is a sequence
    of (node name, files) pairs for further non-BAT* nodes.
    """
    root = tmp_path / "power_supply"
    _write_node(root / "ACAD", {"type": "Mains", "online": "1"})
    _write_node(root / "ucsi-source-psy-USBC000:003", {"type": "USB", "online": "1"})
    for name, files in extra:
        _write_node(root / name, files)
    for name, (status, capacity) in batteries.items():
        _write_node(
            root / name, {"type": "Battery", "status": status, "capacity": str(capacity)},
        )
    return root


def write_shims(bin_dir: Path, shims: Mapping[str, str]) -> None:
    """Write each bash *shims* body as an executable named for its key."""
    bin_dir.mkdir(exist_ok=True)
    for name, body in shims.items():
        shim = bin_dir / name
        shim.write_text("#!/usr/bin/env bash\n" + body)
        shim.chmod(0o755)
