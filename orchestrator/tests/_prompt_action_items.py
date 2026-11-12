"""The numbered ``## Action`` items of an architect prompt, for the briefing test files."""

from __future__ import annotations

import re


def action_items(prompt: str) -> list[str]:
    """The numbered items of *prompt*'s ``## Action`` section, continuation lines included."""
    section = prompt.split('## Action', 1)[1].split('\n#', 1)[0]
    return [item for item in re.split(r'\n(?=\d+\. )', section.strip()) if re.match(r'\d+\. ', item)]
