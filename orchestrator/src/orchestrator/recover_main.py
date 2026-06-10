"""recover_main — enforce-safe break-glass recovery ref-move CLI.

CLI usage (JSON to stdout):
  python -m orchestrator.recover_main \\
      --project-root <watched-project-root> \\
      --config      <watched-orchestrator.yaml> \\
      --target-sha  <good-sha> \\
      --expected-main <current-bad-sha>

Prints a JSON object:
  {"result": "rewound"|"cas_failed"|"error", "target_sha": "<sha>"}

On unexpected exceptions (bad config path, inaccessible repo, etc.), the JSON
contract is still honored:
  {"result": "error", "detail": "<exception message>", "target_sha": "<sha>"}

Exit codes:
  0  — rewound (success)
  1  — cas_failed (another writer moved the ref first), error, or exception

Mirrors b3_gate's cross-project invocation pattern: loads the WATCHED
project's GitConfig (where main_gate_mark_command is set), constructs
GitOps, and invokes recover_red_main for a single sanctioned CAS move.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from orchestrator.config import load_config
from orchestrator.git_ops import GitOps


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='python -m orchestrator.recover_main',
        description=(
            'Enforce-safe break-glass recovery: move refs/heads/main to a '
            'known-good SHA in one atomic sanctioned CAS update-ref.'
        ),
    )
    p.add_argument('--project-root', required=True,
                   help='Root of the WATCHED project repo (where main lives)')
    p.add_argument('--config', required=True,
                   help='Path to the WATCHED project\'s orchestrator YAML')
    p.add_argument('--target-sha', required=True,
                   help='Good SHA to restore main to')
    p.add_argument('--expected-main', required=True,
                   help='Current (bad) value of refs/heads/main (CAS old-value)')
    return p


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns 0 on rewound, 1 on cas_failed / error / exception."""
    p = _build_parser()
    args = p.parse_args(argv)

    try:
        cfg = load_config(Path(args.config))
        git_ops = GitOps(cfg.git, Path(args.project_root))
        result = asyncio.run(git_ops.recover_red_main(args.target_sha, args.expected_main))
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({
            'result': 'error',
            'detail': str(exc),
            'target_sha': args.target_sha,
        }))
        return 1

    print(json.dumps({'result': result, 'target_sha': args.target_sha}))
    return 0 if result == 'rewound' else 1


if __name__ == '__main__':
    sys.exit(main())
