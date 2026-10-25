"""Real child processes for tests: start them, and always kill and reap them."""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Iterator
from contextlib import contextmanager


@contextmanager
def spawned_children() -> Iterator[Callable[..., subprocess.Popen]]:
    """Yield a spawner; on exit kill and reap every child it started, even one already reaped."""
    children: list[subprocess.Popen] = []

    def spawn(argv: list[str], **popen_kwargs) -> subprocess.Popen:
        child = subprocess.Popen(argv, **popen_kwargs)
        children.append(child)
        return child

    try:
        yield spawn
    finally:
        for child in children:
            child.kill()
            child.wait()
            for stream in (child.stdin, child.stdout, child.stderr):
                if stream is not None:
                    stream.close()
