"""Which of a capability-manifest sidecar's task labels its human-readable .md twin never mentions.

Unit-tested by ``shared/tests/test_capability_manifest_twins.py`` and consumed by the corpus
guard ``tests/repo_guards/test_checked_in_capability_manifest_twins.py``. Importable bare
because both suites' conftests put this ``cross_suite_helpers/`` directory on sys.path.

A label is mentioned when the twin contains it as a token bounded by the label alphabet ``[\\w′-]``, verbatim or, for an
ASCII Greek name with optional digits (``gamma1``), as its Greek letter in either case.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from shared.capability_manifest import load_capability_manifest

_GREEK_NAMES = (
    'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta',
    'iota', 'kappa', 'lambda', 'mu', 'nu', 'xi', 'omicron', 'pi',
    'rho', 'sigma', 'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega',
)
_GREEK_LETTER_BY_NAME: dict[str, str] = dict(zip(_GREEK_NAMES, 'αβγδεζηθικλμνξοπρστυφχψω', strict=True))
_TRANSLITERATED_LABEL = re.compile('(' + '|'.join(_GREEK_NAMES) + r')(\d*)')
_LABEL_CHAR = r'[\w′-]'


def md_twin_path(sidecar: Path) -> Path:
    return sidecar.with_suffix('.md')


def _label_spellings(label: str) -> frozenset[str]:
    match = _TRANSLITERATED_LABEL.fullmatch(label)
    if match is None:
        return frozenset({label})
    letter = _GREEK_LETTER_BY_NAME[match.group(1)]
    digits = match.group(2)
    return frozenset({label, letter + digits, letter.upper() + digits})


def _occurs_as_token(spelling: str, text: str) -> bool:
    pattern = f'(?<!{_LABEL_CHAR}){re.escape(spelling)}(?!{_LABEL_CHAR})'
    return re.search(pattern, text) is not None


def unmentioned_labels(labels: Iterable[str], md_text: str) -> tuple[str, ...]:
    return tuple(
        label
        for label in labels
        if not any(_occurs_as_token(s, md_text) for s in _label_spellings(label))
    )


def _display(path: Path, repo_root: Path) -> Path:
    try:
        return path.relative_to(repo_root)
    except ValueError:
        return path


def check_md_twin(sidecar: Path, repo_root: Path) -> str | None:
    labels = [task.label for task in load_capability_manifest(sidecar).tasks]
    twin = md_twin_path(sidecar)
    sidecar_rel, twin_rel = _display(sidecar, repo_root), _display(twin, repo_root)
    if not twin.is_file():
        return (
            f'{sidecar_rel}: no .md twin at {twin_rel} — '
            'the human-readable manifest must sit beside its sidecar'
        )
    missing = unmentioned_labels(labels, twin.read_text(encoding='utf-8'))
    if not missing:
        return None
    return (
        f'{twin_rel}: never mentions sidecar label(s) {", ".join(map(repr, missing))} '
        f'declared in {sidecar_rel} — add a section or table row naming each '
        '(verbatim, or its Greek letter for an ASCII-spelled name)'
    )
