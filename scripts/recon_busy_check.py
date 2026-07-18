#!/usr/bin/env python3
"""STDLIB-ONLY gate helper for the cycle-aware restart of fused-memory.

Reads the fused-memory /health JSON body from stdin and classifies whether a
full reconciliation cycle is in flight, so restart-fused-memory.sh's default
path can defer a restart while a 9-30min cycle runs and proceed between
cycles (task 2703 δ). Deliberately stdlib-only + stdin-driven, mirroring
scripts/drain_check.py: the pure classify() gets pytest coverage, the CLI
gets subprocess coverage, and the bash-script test drives the REAL gate
logic through a fake curl with no HTTP server. python3 is guaranteed present
(fm's own runtime) whereas jq is not universally installed.

The /health body shape consumed (additive `recon_busy` field, task 2703):
    {"status": "ok", "graphiti": true, "mem0": true,
     "recon_busy": [{"project_id":.., "run_id":.., "stage":.., "started_at":..}, ...]}

CLI usage (called by restart-fused-memory.sh's recon gate):
    curl -s --max-time N http://localhost:8002/health | python3 recon_busy_check.py
Prints the classification word (busy/idle/unreachable) on line 1; when busy,
one `recon_busy_cycle ...` detail line per in-flight cycle follows. Always
exits 0 — the caller decides defer-vs-proceed from the first line.
"""

from __future__ import annotations

import argparse
import json
import sys


def classify(health: object) -> str:
    """Classify a parsed /health body into busy / idle / unreachable.

    - unreachable iff *health* is not a JSON object (None from an
      unreachable/blank/malformed endpoint, or a non-dict). Fail-safe: the
      caller PROCEEDS rather than deferring on a body it cannot read.
    - busy iff ``recon_busy`` is a non-empty list (a full cycle is in flight).
    - idle otherwise (``recon_busy`` empty, absent, or not a list).
    """
    if not isinstance(health, dict):
        return "unreachable"
    recon_busy = health.get("recon_busy")
    if isinstance(recon_busy, list) and len(recon_busy) > 0:
        return "busy"
    return "idle"


def format_cycle_lines(health: object) -> list[str]:
    """Return one greppable ``recon_busy_cycle ...`` detail line per in-flight
    cycle. Empty unless *health* is a dict with a non-empty ``recon_busy``
    list (i.e. ``classify(health) == 'busy'``)."""
    if not isinstance(health, dict):
        return []
    recon_busy = health.get("recon_busy")
    if not isinstance(recon_busy, list):
        return []
    lines: list[str] = []
    for entry in recon_busy:
        if not isinstance(entry, dict):
            continue
        lines.append(
            "recon_busy_cycle "
            f"project_id={entry.get('project_id')} "
            f"run_id={entry.get('run_id')} "
            f"stage={entry.get('stage')} "
            f"started_at={entry.get('started_at')}"
        )
    return lines


def parse_health(text: str | None) -> object | None:
    """Parse a /health body defensively. Blank/whitespace-only input or
    invalid JSON returns None (→ classify 'unreachable')."""
    if text is None or text.strip() == "":
        return None
    try:
        return json.loads(text)
    except ValueError:
        return None


def _build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description=(
            "Read a fused-memory /health JSON body from stdin and print "
            "busy/idle/unreachable (line 1) plus per-cycle detail lines when "
            "busy. Always exits 0."
        ),
    )


def main(argv: list[str] | None = None) -> int:
    _build_parser().parse_args(argv)
    health = parse_health(sys.stdin.read())
    verdict = classify(health)
    print(verdict)
    if verdict == "busy":
        for line in format_cycle_lines(health):
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
