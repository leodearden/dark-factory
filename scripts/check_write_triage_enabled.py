#!/usr/bin/env python3
"""Sub-check for gate φ of plans/write-triage-link-healing-prd.md: is the write-triage flip live?

Exit 0 iff the config file sets `write_triage.enabled` to the YAML boolean true. Anything else
exits 1: a missing or unparseable file, a missing section or key, or any other value, so the
gate fails closed. It reads the committed file, not the running server's config.

Run it under the project's interpreter, as gate φ's `--subcheck` does:
  uv run --frozen --project fused-memory python scripts/check_write_triage_enabled.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path('fused-memory/config/config.yaml')


def flag_state(config_path: Path) -> tuple[bool, str]:
    try:
        doc = yaml.safe_load(config_path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        return False, f'config unreadable: {exc}'
    section = doc.get('write_triage') if isinstance(doc, dict) else None
    if not isinstance(section, dict) or 'enabled' not in section:
        return False, 'write_triage.enabled absent'
    value = section['enabled']
    return value is True, f'write_triage.enabled = {value!r}'


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    enabled, detail = flag_state(args.config)
    print(f'{"PASS" if enabled else "FAIL"}  {args.config}: {detail}')
    return 0 if enabled else 1


if __name__ == '__main__':
    sys.exit(main())
