"""cockpit.panes.spawn_bar — the `n` spawn-bar (Fleet Cockpit C5b, PRD §9).

build_spawn_argv is pure argv construction only: it renders the verified
spawn-claude.sh positional contract as a plain, subprocess-ready list[str].
Launching the process itself is the app's job (an injected spawn_runner),
so this module stays import-clean of subprocess and fast/deterministic to
unit test. The SpawnScreen(ModalScreen) project/role/prompt picker widget
lands in a later C5b step.
"""

from __future__ import annotations

import os


def build_spawn_argv(
    script: os.PathLike[str] | str,
    cwd: str,
    title: str,
    prompt: str,
    *,
    skip_perms: bool = False,
) -> list[str]:
    """Build the argv for spawn-claude.sh's verified positional contract.

    ``spawn-claude.sh <cwd> <skip_perms:true|false> <title|""> <prompt>``
    -- see skills/spawn/spawn-claude.sh's usage comment. *script* may be a
    Path or str; it's rendered with str() so the result is a plain
    list[str] ready to hand straight to a subprocess runner. *skip_perms*
    maps to the literal 'true'/'false' strings the shell script's `[ ]`
    comparisons expect (not Python's True/False). An empty *title* is
    passed through unchanged -- spawn-claude.sh treats "" as "no title".
    """
    return [str(script), cwd, 'true' if skip_perms else 'false', title, prompt]
