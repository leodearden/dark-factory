"""Resolves the checkouts eval code runs in and reads fixtures from, from shared.project_layout (plans/review-all-s18-evals-measurement-prd.md D11)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from shared.project_id import normalize_project_id, project_id_for_root
from shared.project_layout import (
    KNOWN_PROJECT_ROOTS_ENV,
    CheckoutRootError,
    default_project_root,
    known_project_roots,
)

__all__ = ['ProjectCheckoutNotFound', 'main_checkout', 'project_checkout']


class ProjectCheckoutNotFound(CheckoutRootError):
    """No checkout of a project is this repository or registered among the known roots."""

    def __init__(self, project_id: str, main_checkout: Path, known_roots: tuple[Path, ...]) -> None:
        self.project_id = project_id
        self.main_checkout = main_checkout
        self.known_roots = known_roots
        registered = ', '.join(str(root) for root in known_roots) or 'unset'
        super().__init__(
            f'no checkout of project {project_id!r}: this repository is '
            f'{project_id_for_root(main_checkout)!r} at {main_checkout}, and '
            f'${KNOWN_PROJECT_ROOTS_ENV} ({registered}) names no root with that id; '
            f'register the project\'s root in ${KNOWN_PROJECT_ROOTS_ENV}'
        )


def main_checkout() -> Path:
    """The eval package's name for shared.project_layout.default_project_root.

    Raises CheckoutRootError when none can be derived. Eval CLI flag defaults reach the
    same root through orchestrator/src/orchestrator/main_checkout_defaults.py::cli_default.
    """
    return default_project_root()


def project_checkout(project_id: str, environ: Mapping[str, str] = os.environ) -> Path:
    """This repository's main checkout when it is *project_id*, else its registered known root."""
    wanted = normalize_project_id(project_id)
    main = main_checkout()
    if project_id_for_root(main) == wanted:
        return main
    roots = known_project_roots(environ)
    for root in roots:
        if _registered_project_id(root) == wanted:
            return root
    raise ProjectCheckoutNotFound(wanted, main, roots)


def _registered_project_id(root: Path) -> str | None:
    try:
        return project_id_for_root(root)
    except ValueError:
        return None
