#!/usr/bin/env python3
"""Find an unused escalation MCP port for a new dark-factory project.

All projects share the fused-memory server on 8002; only the escalation port is
unique per project. This script gathers the ports already claimed by existing
project configs, reserves the well-known ports (8002 fused-memory, 8103 shared
reconciliation queue), checks what's actually bound, and proposes the lowest
free port >= 8100.

It prints a human-readable evidence table to stderr and the chosen port alone on
stdout (so callers can capture it). Read-only: it never edits anything.

Usage:
    python3 find_escalation_port.py [--df-root PATH] [--exclude-root PATH] [--base 8100]
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import socket
import subprocess
import sys
from pathlib import Path

# Ports that must never be handed out as an escalation port.
RESERVED = {
    8002: "fused-memory HTTP server (shared by all projects)",
    8103: "shared reconciliation escalation queue",
}

# Candidate config filenames, in the order projects have historically used them.
CONFIG_NAMES = (
    "orchestrator.yaml",
    "orchestrator-config.yaml",
    "config.yaml",
    "orchestrator/config.yaml",
)

PORT_RE = re.compile(r"^\s*port:\s*(\d+)", re.MULTILINE)


def known_project_roots(df_root: Path) -> list[Path]:
    """Project roots from the live fused-memory unit's env plus /home/leo/src/*."""
    roots: list[Path] = []

    # 1. DASHBOARD_KNOWN_PROJECT_ROOTS from the live systemd unit, if reachable.
    try:
        out = subprocess.run(
            ["systemctl", "--user", "show", "fused-memory.service", "-p", "Environment"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        m = re.search(r"DASHBOARD_KNOWN_PROJECT_ROOTS=([^\s]+)", out)
        if m:
            roots += [Path(p) for p in m.group(1).split(",") if p]
    except Exception:
        pass

    # 2. Fallback: sibling repos next to the dark-factory checkout.
    roots += [Path(p) for p in glob.glob(str(df_root.parent / "*")) if Path(p).is_dir()]

    # Dedup, keep existing dirs only.
    seen: set[str] = set()
    uniq: list[Path] = []
    for r in roots:
        rp = str(r.resolve())
        if rp not in seen and r.is_dir():
            seen.add(rp)
            uniq.append(r)
    return uniq


def find_config(root: Path) -> Path | None:
    for name in CONFIG_NAMES:
        cand = root / name
        if cand.is_file():
            return cand
    return None


def escalation_port(cfg: Path) -> int | None:
    """Parse the port inside the top-level `escalation:` block (no yaml dep)."""
    try:
        text = cfg.read_text()
    except Exception:
        return None
    # Slice from the `escalation:` line to the next top-level (column-0) key.
    lines = text.splitlines()
    block: list[str] = []
    in_block = False
    for ln in lines:
        if re.match(r"^escalation:\s*$", ln):
            in_block = True
            continue
        if in_block:
            if ln and not ln[0].isspace() and not ln.lstrip().startswith("#"):
                break  # next top-level key
            block.append(ln)
    m = PORT_RE.search("\n".join(block))
    return int(m.group(1)) if m else None


def is_bound(port: int) -> bool:
    """True if something is already listening on 127.0.0.1:port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
            return False
        except OSError:
            return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--df-root", default=os.environ.get("DARK_FACTORY_ROOT", "/home/leo/src/dark-factory"))
    ap.add_argument("--exclude-root", default=None, help="target repo to ignore (it may already have a stale config)")
    ap.add_argument("--base", type=int, default=8100, help="lowest port to consider")
    args = ap.parse_args()

    df_root = Path(args.df_root).resolve()
    exclude = Path(args.exclude_root).resolve() if args.exclude_root else None

    used: dict[int, str] = {}
    for root in known_project_roots(df_root):
        if exclude and root.resolve() == exclude:
            continue
        cfg = find_config(root)
        if not cfg:
            continue
        port = escalation_port(cfg)
        if port is not None:
            used[port] = f"{root.name} ({cfg.relative_to(root)})"

    print("Escalation port survey", file=sys.stderr)
    print("-" * 60, file=sys.stderr)
    print("Reserved:", file=sys.stderr)
    for p, why in sorted(RESERVED.items()):
        print(f"  {p}  {why}", file=sys.stderr)
    print("Claimed by existing project configs:", file=sys.stderr)
    for p in sorted(used):
        print(f"  {p}  {used[p]}", file=sys.stderr)
    if not used:
        print("  (none found)", file=sys.stderr)

    blocked = set(RESERVED) | set(used)
    chosen = None
    for cand in range(args.base, args.base + 100):
        if cand in blocked:
            continue
        if is_bound(cand):
            print(f"  note: {cand} is free in config but currently BOUND — skipping", file=sys.stderr)
            continue
        chosen = cand
        break

    print("-" * 60, file=sys.stderr)
    if chosen is None:
        print("ERROR: no free escalation port found", file=sys.stderr)
        return 1
    print(f"CHOSEN escalation port: {chosen}", file=sys.stderr)
    print(chosen)  # stdout: machine-readable
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
