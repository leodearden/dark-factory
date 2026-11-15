"""The shell's logical-line grammar, for guards that read a shell script's commands.

This module holds NO test functions of its own.  It exists because more than
one guard reads the commands a shell script will run, and two copies of the
continuation/comment rule drift until one silently mis-reads a script.

Importable from tests/scripts/test_*.py only because tests/scripts/conftest.py
puts this directory on sys.path — pytest's --import-mode=importlib (set in
pyproject.toml addopts) deliberately does not.  Same arrangement as
tests/scripts/systemd_unit_invariants.py.
"""


def logical_lines(script: str) -> list[str]:
    """*script*'s lines with backslash continuations joined into one each.

    A line-oriented scan reads ``"$UV" sync \\`` / ``  --all-packages`` as two
    fragments and finds no invocation at all — a guard then reports "found 0",
    which is loud but names the wrong defect.  The shell joins those two
    physical lines into one command, and so must anything claiming to read the
    arguments the shell will see.

    A trailing backslash on a COMMENT line does not continue it in shell (the
    comment ends at the newline), so such a line is never joined onto the next.
    Joining it would swallow a real command into a comment and drop it from the
    scan silently — the one direction this helper must refuse.

    Two consumers: tests/scripts/test_sync_orchestrator_env.py::
    _uv_sync_invocations and tests/scripts/test_uv_run_venv_isolation.py::
    shell_uv_run_sites.

    Not merged with its unit-file counterpart,
    tests/scripts/systemd_unit_invariants.py::logical_exec_start: this is a
    SHELL grammar over a whole script, bound by the shell's comment rule
    above, and that module holds unit-file semantics only.
    """
    joined: list[str] = []
    pending = ""
    for raw in script.splitlines():
        line = raw.strip()
        if pending:
            line = f"{pending} {line}"
            pending = ""
        elif line.startswith("#"):
            joined.append(line)
            continue
        if line.endswith("\\"):
            pending = line[:-1].strip()
            continue
        joined.append(line)
    if pending:
        joined.append(pending)
    return joined
