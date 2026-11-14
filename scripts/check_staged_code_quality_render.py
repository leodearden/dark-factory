#!/usr/bin/env python3
"""Refuse a commit whose staged code-quality doc cannot render.

``orchestrator/src/orchestrator/agents/roles.py`` renders
``orchestrator/src/orchestrator/agents/code_quality.md`` through
``orchestrator/src/orchestrator/agents/code_quality.py::render`` at import, so a
doc that cannot render makes every orchestrator raise on its next restart. A
docs-only commit straight to main never meets the merge gate's
``orchestrator/tests/test_code_quality.py``. So this renders what the commit
STAGES: the staged doc with the staged renderer, the pair that lands together.

Exit 0 when the pair renders, 1 when the commit is refused. Called by
``hooks/project-checks`` whenever a commit stages either file.
"""
from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable

import _checkout as _checkout

DOC_RELPATH = 'orchestrator/src/orchestrator/agents/code_quality.md'
RENDERER_RELPATH = 'orchestrator/src/orchestrator/agents/code_quality.py'


class NotStaged(Exception):
    """A file the render needs is absent from the index."""


def _staged_text(relpath: str) -> str:
    shown = subprocess.run(
        ['git', 'show', f':{relpath}'],
        cwd=_checkout.ROOT,
        capture_output=True,
        encoding='utf-8',
        check=False,
    )
    if shown.returncode != 0:
        raise NotStaged(
            f'code-quality render check: {relpath} is not in the index, so this '
            'commit would leave roles.py unable to render the code-quality block. '
            f'git said: {shown.stderr.strip()}'
        )
    return shown.stdout


def _staged_render() -> Callable[[str], object]:
    namespace: dict[str, object] = {'__name__': 'staged_code_quality'}
    exec(compile(_staged_text(RENDERER_RELPATH), RENDERER_RELPATH, 'exec'), namespace)
    render = namespace.get('render')
    if not callable(render):
        raise TypeError(f'the staged {RENDERER_RELPATH} defines no callable render()')
    return render


def main() -> int:
    try:
        doc = _staged_text(DOC_RELPATH)
        render = _staged_render()
    except NotStaged as exc:
        print(exc, file=sys.stderr)
        return 1
    try:
        render(doc)
    except ValueError as exc:
        print(f'code-quality render check: {exc}', file=sys.stderr)
        print(
            'roles.py renders this doc at import, so this commit would make every '
            'orchestrator raise on its next restart.',
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
