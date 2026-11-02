"""Write the merge-gate spec a project's live config ships to a verify host.

    .venv/bin/python make_spec.py <dark-factory-orchestrator.yaml> <out.json> [<only module>]

Works for any target project: dark-factory (per-module commands) or reify,
which registers no modules, so its spec carries the global test/lint commands.
<only module> keeps one module's commands, for a smoke run; naming a module the
config does not have is an error rather than an empty gate.
"""

import sys
from pathlib import Path

from orchestrator.config import load_config
from orchestrator.verify_plan import effective_merge_module_configs
from orchestrator.verify_runner import build_merge_verify_spec, spec_to_json


def main(config_path: str, out_path: str, only: str | None = None) -> None:
    config = load_config(Path(config_path))
    modules = effective_merge_module_configs(config, list(config.module_configs_or_empty))
    if only is not None:
        modules = [m for m in modules if m.prefix == only]
        if not modules:
            sys.exit(f'make_spec.py: {config_path} has no module {only!r}')
    Path(out_path).write_text(spec_to_json(build_merge_verify_spec(config, modules, None)))


if __name__ == '__main__':
    main(*sys.argv[1:])
