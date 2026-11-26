"""Parser of the private verify-host roster file (plans/verify-host-roster-and-rented-trial-prd.md §6.3).

The roster names every remote verify host on this workstation and, per host,
the projects it verifies. Schema only lives here: remote path rules belong to
orchestrator.config.VerifyRunnerConfig, which every consumer validates through.
"""

import os
import re
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError, field_validator

ROSTER_FILE_ENV = 'ORCH_VERIFY_HOSTS_FILE'

Trust = Literal['probation', 'trusted']

_SHELL_NAME = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')


class RosterFileError(ValueError):
    """The roster file cannot be read, or names no such host; the message names the file."""


def roster_file_path(env: Mapping[str, str]) -> Path:
    override = env.get(ROSTER_FILE_ENV)
    if override:
        return Path(override)
    config_home = env.get('XDG_CONFIG_HOME') or os.path.join(env['HOME'], '.config')
    return Path(config_home) / 'orchestrator' / 'verify-hosts.yaml'


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')


class RosterProject(_Frozen):
    git_remote: str
    config_path: str | None = None
    task_slots: int | None = Field(default=None, ge=0)
    trust: Trust = 'probation'
    disk_footprint_gib: int | None = Field(default=None, ge=0)
    host_config: dict[str, Any] = {}


class RosterHost(_Frozen):
    ssh_host: str
    df_checkout_path: str
    gate_checkouts_path: str | None = None
    enabled: bool = True
    wrapper_env: dict[str, str] = {}
    projects: dict[str, RosterProject]

    @field_validator('wrapper_env')
    @classmethod
    def _names_are_shell_identifiers(cls, wrapper_env: dict[str, str]) -> dict[str, str]:
        bad = sorted(name for name in wrapper_env if not _SHELL_NAME.fullmatch(name))
        if bad:
            raise ValueError(f'wrapper_env names must be shell identifiers, got {bad}')
        return wrapper_env


@dataclass(frozen=True)
class RosterRunner:
    """One host's runner for one project; field names are VerifyRunnerConfig's plus the live pair."""

    name: str
    ssh_host: str
    git_remote: str
    config_path: str | None
    df_checkout_path: str
    gate_checkouts_path: str | None
    enabled: bool
    task_slots: int | None
    trust: Trust


class Roster(_Frozen):
    hosts: dict[str, RosterHost] = {}
    _source: Path | None = PrivateAttr(default=None)

    @classmethod
    def from_document(cls, document: object, source: Path) -> 'Roster':
        roster = cls.model_validate(document)
        roster._source = source
        return roster

    def runners_for_project(self, project_id: str) -> tuple[RosterRunner, ...]:
        return tuple(
            _runner(name, host, host.projects[project_id])
            for name, host in self.hosts.items()
            if project_id in host.projects
        )

    def host(self, name: str) -> RosterHost:
        try:
            return self.hosts[name]
        except KeyError:
            raise RosterFileError(
                f'{self._source}: no host {name!r} in the roster (hosts: {sorted(self.hosts)})'
            ) from None

    def host_configs(self, name: str) -> Mapping[str, Mapping[str, Any]]:
        return {project_id: project.host_config for project_id, project in self.host(name).projects.items()}


def _runner(name: str, host: RosterHost, project: RosterProject) -> RosterRunner:
    return RosterRunner(
        name=name,
        ssh_host=host.ssh_host,
        git_remote=project.git_remote,
        config_path=project.config_path,
        df_checkout_path=host.df_checkout_path,
        gate_checkouts_path=host.gate_checkouts_path,
        enabled=host.enabled,
        task_slots=project.task_slots,
        trust=project.trust,
    )


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that refuses a repeated mapping key instead of keeping the last one."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Hashable, Any]:
        seen: set[Hashable] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if isinstance(key, Hashable) and key in seen:
                raise yaml.constructor.ConstructorError(
                    'while constructing a mapping', node.start_mark,
                    f'found duplicate key {key!r}', key_node.start_mark,
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def load_roster(path: Path) -> Roster:
    """Parse the roster at *path*; an absent file is an empty roster, anything unreadable raises."""
    try:
        text = path.read_text()
    except FileNotFoundError:
        return Roster.from_document({}, path)
    except OSError as exc:
        raise RosterFileError(f'{path}: cannot read the roster file: {exc}') from exc
    try:
        document = yaml.load(text, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise RosterFileError(f'{path}: not valid YAML: {exc}') from exc
    if not document:
        raise RosterFileError(f'{path}: the roster file is empty; delete it to declare no hosts')
    if not isinstance(document, dict):
        raise RosterFileError(f'{path}: the top level must be a mapping, got {type(document).__name__}')
    try:
        return Roster.from_document(document, path)
    except ValidationError as exc:
        raise RosterFileError(f'{path}: {_render(exc)}') from exc


def _render(exc: ValidationError) -> str:
    return '; '.join(
        f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}" for error in exc.errors()
    )
