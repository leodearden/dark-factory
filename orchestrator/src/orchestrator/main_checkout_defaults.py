"""Defaults that point into the main checkout, each with an override the operator can name.

runs.db, tickets.db and the escalation queue are gitignored data that live in the
MAIN checkout, never a task worktree. A default for one is derived per call from
shared.project_layout.default_project_root, so importing a caller never needs a
checkout, and a host where none can be derived is told which override to use.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import click
from shared.project_layout import CheckoutRootError, default_project_root


def env_or_main_checkout(
    env_var: str | None, derive: Callable[[Path], Path], *, flag: str | None = None,
) -> Path:
    """``$env_var`` when set, else *derive* applied to the main checkout.

    Raises CheckoutRootError naming *flag* and *env_var* when neither resolves.
    """
    override = os.environ.get(env_var) if env_var else None
    if override:
        return Path(override)
    try:
        root = default_project_root()
    except CheckoutRootError as exc:
        raise _unavailable(flag, env_var, exc) from exc
    return derive(root)


def _unavailable(flag: str | None, env_var: str | None, cause: CheckoutRootError) -> CheckoutRootError:
    overrides = []
    if flag:
        overrides.append(f'pass {flag}')
    if env_var:
        overrides.append(f'set ${env_var}')
    return CheckoutRootError(
        f'{" or ".join(overrides)} to name the path: its default lives in the main '
        f'checkout, which cannot be derived here ({cause})'
    )


def cli_default(flag: str, derive: Callable[[Path], Path], *, env_var: str | None = None) -> Path:
    """:func:`env_or_main_checkout` for an omitted click *flag*, failing as a usage error."""
    try:
        return env_or_main_checkout(env_var, derive, flag=flag)
    except CheckoutRootError as exc:
        raise click.UsageError(str(exc)) from exc
