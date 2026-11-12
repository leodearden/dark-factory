"""A private, removed-on-exit HOME for one third-party agent CLI invocation, and the child env that redirects HOME into it while keeping the operator's git identity and tool homes."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from shared.secret_env import agent_subprocess_env, is_secret_env_name

logger = logging.getLogger(__name__)

# HOME-derived tool locations the agent's commands keep at the operator's values; see plans/zcode-harness-probe-2026-11.md.
OPERATOR_TOOL_HOMES: tuple[tuple[str, str], ...] = (
    ('XDG_CONFIG_HOME', '.config'),
    ('XDG_CACHE_HOME', '.cache'),
    ('XDG_DATA_HOME', '.local/share'),
    ('XDG_STATE_HOME', '.local/state'),
    ('CARGO_HOME', '.cargo'),
    ('RUSTUP_HOME', '.rustup'),
)


def require_operator_home(base_env: Mapping[str, str], *, backend: str) -> Path:
    home = (base_env.get('HOME') or '').strip()
    if not home:
        raise ValueError(
            f'the {backend} backend needs HOME in its environment: without the operator home it '
            "cannot reproduce its control's tool environment under the redirected run home",
        )
    return Path(home)


@contextmanager
def private_run_home(*, backend: str, operator_home: Path) -> Iterator[Path]:
    """A fresh 0700 directory holding tmp/ and a .gitconfig including the operator's; removed on exit."""
    root = Path(tempfile.mkdtemp(prefix=f'{backend}-home-'))
    try:
        (root / '.gitconfig').write_text(_git_include(operator_home / '.gitconfig'))
        (root / 'tmp').mkdir()
        yield root
    finally:
        _remove_run_home(root, backend=backend)


def _git_include(path: Path) -> str:
    quoted = str(path).replace('\\', '\\\\').replace('"', '\\"')
    return f'[include]\n\tpath = "{quoted}"\n'


def write_owner_only_json(path: Path, document: Mapping[str, Any]) -> None:
    """Created 0600 before any byte lands, so a key in *document* is never readable by others."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as handle:
        json.dump(document, handle)


def _remove_run_home(root: Path, *, backend: str) -> None:
    try:
        shutil.rmtree(root)
    except OSError:
        logger.warning(
            '%s backend: could not remove run home %s, which holds an endpoint key',
            backend, root, exc_info=True,
        )


def redirected_home_env(
    base_env: Mapping[str, str], *, root: Path, operator_home: Path,
) -> dict[str, str]:
    inherited = {
        name: value for name, value in agent_subprocess_env(base_env).items()
        if not is_secret_env_name(name)
    }
    return {
        **inherited,
        **_operator_tool_homes(base_env, operator_home),
        'HOME': str(root),
        'USERPROFILE': str(root),
        'TMPDIR': str(root / 'tmp'),
    }


def _operator_tool_homes(base_env: Mapping[str, str], operator_home: Path) -> dict[str, str]:
    overlay: dict[str, str] = {}
    for name, default in OPERATOR_TOOL_HOMES:
        inherited = base_env.get(name, '')
        overlay[name] = inherited if inherited.strip() else str(operator_home / default)
    return overlay
