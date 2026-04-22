"""Landlock-restrict-self wrapper: apply a ruleset, then execvp the inner command.

Analogous to ``bwrap <args> -- <inner_cmd>`` — a standalone executable that an
orchestrator-style command constructor can wrap around any child process. Unlike
bwrap, this does not use user namespaces, so it sidesteps the Bun v1.3.13 +
kernel 6.17 self-init segfault.

Usage::

    python -m orchestrator.agents.landlock_exec \\
        --writable /path/mod_a \\
        --writable /path/mod_b \\
        [--writable ...] \\
        -- <inner cmd> [args...]

The ``/`` filesystem is made read-only (exec + read_file + read_dir). Each
``--writable`` path is granted full v1 access. ``~/.claude`` is always
writable (OAuth/session state). ``/dev`` gets WRITE_FILE + RO (for /dev/null
etc.). Nothing else can be written.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys

# x86_64 syscall numbers — landlock landed in 5.13 on this arch
SYS_landlock_create_ruleset = 444
SYS_landlock_add_rule = 445
SYS_landlock_restrict_self = 446
PR_SET_NO_NEW_PRIVS = 38

# Landlock v1 filesystem access bits
FS_EXECUTE = 1 << 0
FS_WRITE_FILE = 1 << 1
FS_READ_FILE = 1 << 2
FS_READ_DIR = 1 << 3
FS_REMOVE_DIR = 1 << 4
FS_REMOVE_FILE = 1 << 5
FS_MAKE_CHAR = 1 << 6
FS_MAKE_DIR = 1 << 7
FS_MAKE_REG = 1 << 8
FS_MAKE_SOCK = 1 << 9
FS_MAKE_FIFO = 1 << 10
FS_MAKE_BLOCK = 1 << 11
FS_MAKE_SYM = 1 << 12

FS_V1_ALL = 0
for _b in range(13):
    FS_V1_ALL |= 1 << _b

FS_RO = FS_EXECUTE | FS_READ_FILE | FS_READ_DIR

LANDLOCK_RULE_PATH_BENEATH = 1


class _RulesetAttr(ctypes.Structure):
    _fields_ = [
        ('handled_access_fs', ctypes.c_uint64),
        ('handled_access_net', ctypes.c_uint64),
        ('scoped', ctypes.c_uint64),
    ]


class _PathBeneathAttr(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ('allowed_access', ctypes.c_uint64),
        ('parent_fd', ctypes.c_int32),
    ]


def _die(label: str) -> None:
    err = ctypes.get_errno()
    print(
        f'landlock_exec: {label} failed: errno={err} ({os.strerror(err)})',
        file=sys.stderr,
    )
    sys.exit(2)


def _add_path(libc, ruleset_fd: int, path: str, allowed: int) -> None:
    if not os.path.isdir(path):
        return
    fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
    try:
        rule = _PathBeneathAttr()
        rule.allowed_access = allowed
        rule.parent_fd = fd
        rc = libc.syscall(
            SYS_landlock_add_rule,
            ruleset_fd,
            LANDLOCK_RULE_PATH_BENEATH,
            ctypes.byref(rule),
            0,
        )
        if rc != 0:
            _die(f'add_rule({path})')
    finally:
        os.close(fd)


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    if '--' not in argv:
        print('landlock_exec: missing "--" separator before inner command', file=sys.stderr)
        sys.exit(64)
    split = argv.index('--')
    head, inner = argv[:split], argv[split + 1 :]
    if not inner:
        print('landlock_exec: empty inner command', file=sys.stderr)
        sys.exit(64)
    parser = argparse.ArgumentParser(prog='landlock_exec', add_help=False)
    parser.add_argument('--writable', action='append', default=[])
    parser.add_argument('-h', '--help', action='store_true')
    ns = parser.parse_args(head)
    if ns.help:
        print(__doc__, file=sys.stderr)
        sys.exit(0)
    return ns, inner


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    ns, inner = _parse_args(argv)

    libc = ctypes.CDLL('libc.so.6', use_errno=True)

    # Create ruleset covering all v1 fs operations
    attr = _RulesetAttr()
    attr.handled_access_fs = FS_V1_ALL
    ruleset_fd = libc.syscall(
        SYS_landlock_create_ruleset,
        ctypes.byref(attr),
        ctypes.sizeof(attr),
        0,
    )
    if ruleset_fd < 0:
        _die('create_ruleset')

    # Read-only + execute everywhere
    _add_path(libc, ruleset_fd, '/', FS_RO)

    # /dev needs WRITE_FILE for /dev/null, /dev/stderr etc.
    _add_path(libc, ruleset_fd, '/dev', FS_WRITE_FILE | FS_RO)

    # Agent scratch (temp files, MCP configs, sysprompt files).
    # /tmp only — avoid /var/tmp so worktrees placed there stay restricted.
    _add_path(libc, ruleset_fd, '/tmp', FS_V1_ALL)

    # Claude CLI OAuth + session state
    _add_path(libc, ruleset_fd, os.path.expanduser('~/.claude'), FS_V1_ALL)

    # Per-invocation writable paths (locked modules, .task, extras)
    for path in ns.writable:
        _add_path(libc, ruleset_fd, path, FS_V1_ALL)

    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        _die('prctl(NO_NEW_PRIVS)')

    if libc.syscall(SYS_landlock_restrict_self, ruleset_fd, 0) != 0:
        _die('restrict_self')

    os.close(ruleset_fd)

    try:
        os.execvp(inner[0], inner)
    except FileNotFoundError:
        print(f'landlock_exec: inner command not found: {inner[0]}', file=sys.stderr)
        return 127


if __name__ == '__main__':
    sys.exit(main() or 0)
