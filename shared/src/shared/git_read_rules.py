"""Claude CLI permission rules that grant git read subcommands to a dontAsk run.

A prefix allow rule ``Bash(git log:*)`` admits any flag after the subcommand,
including the flags through which that subcommand writes a file. Each such
flag therefore gets a deny rule ``Bash(git log *--output*)``; a ``*`` that is
not a trailing ``:*`` is a wildcard, so the deny matches the flag anywhere.

These denies refuse the plain spelling with a clear message. They are not a
write boundary: the CLI matches rules against the raw command text, so a
shell-quoted spelling such as ``--outp''ut=F`` passes them (measured on CLI
2.1.316). The boundary is the run's Landlock confinement
(``orchestrator/src/orchestrator/agents/sandbox_dispatch.py::wrap_command``).
The git facts the table rests on are pinned against real git by
``shared/tests/test_git_read_rules.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

GIT_FILE_WRITING_FLAGS: Mapping[str, tuple[str, ...]] = MappingProxyType({
    'log': ('--output',),
    'show': ('--output',),
    'diff': ('--output',),
    'status': (),
    'rev-parse': (),
    'ls-files': (),
})


@dataclass(frozen=True)
class GitReadRules:
    allow: tuple[str, ...]
    deny: tuple[str, ...]


def git_read_rules(*subcommands: str) -> GitReadRules:
    unknown = [s for s in subcommands if s not in GIT_FILE_WRITING_FLAGS]
    if unknown:
        raise ValueError(
            f'not a git read subcommand: {unknown!r}; known: {sorted(GIT_FILE_WRITING_FLAGS)!r}',
        )
    return GitReadRules(
        allow=tuple(f'Bash(git {s}:*)' for s in subcommands),
        deny=tuple(
            f'Bash(git {s} *{flag}*)' for s in subcommands for flag in GIT_FILE_WRITING_FLAGS[s]
        ),
    )
