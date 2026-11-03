"""The installed toolchain versions as one JSON object, to confirm that two
pods carry identical tools.

    python3 -I toolchain_versions.py [<python of the DF venv>]
"""

import json
import subprocess
import sys

TOOLS = {
    'uv': ['uv', '--version'],
    'node': ['node', '--version'],
    'npm': ['npm', '--version'],
    'rustc': ['rustc', '--version'],
    'cargo': ['cargo', '--version'],
    'python3': ['python3', '--version'],
}


def version(argv: list[str]) -> str | None:
    try:
        done = subprocess.run(argv, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return (done.stdout or done.stderr).strip()


if __name__ == '__main__':
    found = {name: version(argv) for name, argv in TOOLS.items()}
    if len(sys.argv) > 1:
        found['df_venv_python'] = version([sys.argv[1], '--version'])
    print(json.dumps(found))
