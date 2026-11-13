"""An operator home with a git identity and excludes, shared by the redirected-HOME backend tests.

Each builder writes a fresh tree, so no test can mutate another's copy.
"""

from __future__ import annotations

from pathlib import Path

from orchestrator.agents.agent_run_home import OPERATOR_TOOL_HOMES

OPERATOR_EMAIL = 'op@example.test'
OPERATOR_IGNORED_FILE = 'dropped.operator-ignored'
OPERATOR_ENV_VARS = (*(name for name, _ in OPERATOR_TOOL_HOMES), 'GIT_CONFIG_GLOBAL')


def write_operator_home(root: Path, *, with_gitconfig: bool = True) -> Path:
    """An operator home whose git identity is OPERATOR_EMAIL and whose excludes drop OPERATOR_IGNORED_FILE."""
    root.mkdir()
    if with_gitconfig:
        (root / '.gitconfig').write_text(
            f'[user]\n\tname = Op Erator\n\temail = {OPERATOR_EMAIL}\n',
        )
    ignore = root / '.config' / 'git' / 'ignore'
    ignore.parent.mkdir(parents=True)
    ignore.write_text(f'{OPERATOR_IGNORED_FILE}\n')
    return root
