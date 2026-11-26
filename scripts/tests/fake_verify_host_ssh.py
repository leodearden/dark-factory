"""A fake `ssh` that runs the remote command locally, for driving the verify-host scripts.

Keeps ssh's own semantics: options are skipped, the first operand is the host,
and the remaining operands are joined by spaces and run by the "remote" shell
with stdin passed through. Each call's argv is appended to $FAKE_SSH_LOG as a
JSON array. FAKE_SSH_UNREACHABLE=1 fails every call with ssh's own 255, and
FAKE_SSH_UNPINNED=1 fails any call that demands StrictHostKeyChecking=yes.
"""

import sys
from pathlib import Path

SSH_EXIT_ERROR = 255
OPTIONS_TAKING_A_VALUE = 'BbcDEeFIiJLlmOopQRSWw'

_FAKE_SSH = '''\
import json
import os
import subprocess
import sys

args = sys.argv[1:]
log = os.environ.get('FAKE_SSH_LOG')
if log:
    with open(log, 'a') as handle:
        handle.write(json.dumps(args) + '\\n')
options = []
while args and args[0].startswith('-'):
    flag = args.pop(0)
    if len(flag) == 2 and flag[1] in {options_taking_a_value!r} and args:
        options.append(args.pop(0))
host = args.pop(0)
if os.environ.get('FAKE_SSH_UNREACHABLE') == '1':
    sys.exit({ssh_exit_error})
if os.environ.get('FAKE_SSH_UNPINNED') == '1' and 'StrictHostKeyChecking=yes' in options:
    print(f'Host key verification failed for {{host}}.', file=sys.stderr)
    sys.exit({ssh_exit_error})
sys.exit(subprocess.run(['bash', '-c', ' '.join(args)]).returncode)
'''


def write_fake_ssh(directory: Path) -> Path:
    """Write the fake ssh executable into *directory* and return its path."""
    path = directory / 'fake-ssh'
    body = _FAKE_SSH.format(options_taking_a_value=OPTIONS_TAKING_A_VALUE, ssh_exit_error=SSH_EXIT_ERROR)
    path.write_text(f'#!{sys.executable}\n{body}')
    path.chmod(0o755)
    return path
