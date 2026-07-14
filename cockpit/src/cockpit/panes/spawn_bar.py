"""cockpit.panes.spawn_bar — the `n` spawn-bar (Fleet Cockpit C5b, PRD §9).

build_spawn_argv is pure argv construction only: it renders the verified
spawn-claude.sh positional contract as a plain, subprocess-ready list[str].
Launching the process itself is the app's job (an injected spawn_runner),
so this module stays import-clean of subprocess and fast/deterministic to
unit test. SpawnScreen(ModalScreen) is the project/role/prompt/skip-perms
picker widget itself -- it only ever calls its injected on_submit callback
(app.spawn_session in production) with (project_root, role, prompt,
skip_perms); it never builds argv or launches a process directly.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, Label

_TRUTHY_SKIP_PERMS_TOKENS = frozenset({'true', '1', 'yes', 'on'})


def default_skip_perms() -> bool:
    """Config/env default for the cockpit spawn-bar's skip-perms toggle.

    Reads $CLAUDE_SPAWN_SKIP_PERMS (stripped, case-insensitive); a value in
    {'true', '1', 'yes', 'on'} resolves to True, everything else (including
    unset) resolves to False. Replaces the old hardcoded skip_perms=false:
    the operator who runs with bypass-perms sets this env var to inherit
    that mode for cockpit-spawned sessions (AFK-durable), and/or flips the
    per-spawn SpawnScreen toggle seeded from this default.
    """
    value = os.environ.get('CLAUDE_SPAWN_SKIP_PERMS')
    if value is None:
        return False
    return value.strip().lower() in _TRUTHY_SKIP_PERMS_TOKENS


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


class SpawnScreen(ModalScreen[None]):
    """The 'n' spawn-bar's project/role/prompt/skip-perms picker (PRD §9 C5b).

    A minimal modal form -- a project-root field (seeded from
    known_project_roots' first candidate, when any), a role field (feeds
    the spawned session's title, mirroring session_table's own
    "role:project#task" title shape), a free-text prompt field, and a
    skip-perms Checkbox (F9 fix, task 2518) seeded from the injected
    *default_skip_perms* (the operator's env-configured bypass-perms
    default -- see spawn_bar.default_skip_perms), overridable per spawn.
    Submitting (the button, or Enter in the prompt field) calls the
    injected *on_submit* callback with (project_root, role, prompt,
    skip_perms) and dismisses -- this widget never builds argv or launches
    a process itself; that's app.spawn_session's job (see
    cockpit.app.CockpitApp). Escape dismisses without submitting.
    """

    DEFAULT_CSS = """
    SpawnScreen {
        align: center middle;
    }

    SpawnScreen > Vertical {
        width: 60;
        height: auto;
        border: round $accent;
        padding: 1 2;
        background: $surface;
    }
    """

    BINDINGS = [
        Binding('escape', 'cancel', 'Cancel', show=False),
    ]

    def __init__(
        self,
        project_roots: Sequence[str],
        on_submit: Callable[[str, str, str, bool], None],
        *args,
        default_skip_perms: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._project_roots = list(project_roots)
        self._on_submit = on_submit
        self._default_skip_perms = default_skip_perms

    def compose(self) -> ComposeResult:
        default_project = self._project_roots[0] if self._project_roots else ''
        with Vertical():
            yield Label('Spawn a new session')
            yield Input(value=default_project, placeholder='project root', id='spawn-project')
            yield Input(value='unblock', placeholder='role', id='spawn-role')
            yield Input(placeholder='prompt', id='spawn-prompt')
            yield Checkbox(
                'skip permissions (--dangerously-skip-permissions)',
                value=self._default_skip_perms,
                id='spawn-skip-perms',
            )
            yield Button('Spawn', id='spawn-submit', variant='primary')

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == 'spawn-submit':
            self._submit()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._submit()

    def _submit(self) -> None:
        project_root = self.query_one('#spawn-project', Input).value
        role = self.query_one('#spawn-role', Input).value
        prompt = self.query_one('#spawn-prompt', Input).value
        skip_perms = self.query_one('#spawn-skip-perms', Checkbox).value
        self.dismiss()
        self._on_submit(project_root, role, prompt, skip_perms)

    def action_cancel(self) -> None:
        self.dismiss()
