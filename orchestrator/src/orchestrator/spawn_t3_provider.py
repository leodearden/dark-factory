"""orchestrator.spawn_t3_provider — which T3 provider instance a /spawn thread runs on.

``t3 thread start`` takes a model only together with a provider instance id,
and refuses an instance the project has disabled. The instance is resolved per
launch, in this order:

1. ``explicit`` — the caller named one (``CLAUDE_SPAWN_T3_PROVIDER``);
2. ``project`` — the project's live Claude instance: its default model
   selection's instance when that is an enabled Claude instance, else the only
   enabled Claude instance there is;
3. ``fallback`` — ``claudeAgent``, T3's built-in Claude instance.

Enablement and the default selection are read from the server's settings,
``$T3CODE_HOME/userdata/settings.json``, by the rules of t3code v0.1.6
``packages/contracts/src/settings.ts::resolveProjectProviderInstanceEnabled``
and ``packages/shared/src/projectSettings.ts::resolveProjectOverrides`` (folded
project settings: per-project overrides live under ``projectSettingsOverrides``).
Unreadable settings verify nothing, and neither do settings still carrying the
pre-instance ``providers`` blob, which T3 migrates at load
(``apps/server/src/serverSettings.ts::migrateLegacyProviderSettings``): the
choice is then ``enabled`` None, and ``t3`` itself remains the last word.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

CLAUDE_DRIVER = 'claudeAgent'
FALLBACK_INSTANCE = 'claudeAgent'
_ENABLED_WHEN_UNCONFIGURED = frozenset({'codex', 'claudeAgent'})
_DISABLED_BY_DEFAULT_DRIVERS = frozenset(
    {'cursor', 'grok', 'fx', 'muse', 'pi', 'opencode', 'antigravity'}
)


class ProviderSource(StrEnum):
    EXPLICIT = 'explicit'
    PROJECT = 'project'
    FALLBACK = 'fallback'


@dataclass(frozen=True)
class ProviderChoice:
    instance_id: str
    source: ProviderSource
    enabled: bool | None
    """None when the settings could not be read, so nothing was verified."""


@dataclass(frozen=True)
class ProviderSettings:
    instances: Mapping[str, Mapping[str, Any]]
    project_overrides: Mapping[str, Mapping[str, Any]]
    default_instance: str | None

    @classmethod
    def read(cls, userdata: Path) -> ProviderSettings | None:
        try:
            settings = json.loads((userdata / 'settings.json').read_text())
        except (OSError, ValueError):
            return None
        if not isinstance(settings, dict) or 'providers' in settings:
            return None
        return cls(
            instances=_mapping_of_mappings(settings.get('providerInstances')),
            project_overrides=_mapping_of_mappings(settings.get('projectSettingsOverrides')),
            default_instance=_selection_instance(settings.get('defaultModelSelection')),
        )

    def enabled(self, project_id: str | None, instance_id: str) -> bool:
        override = self._project_instance_flag(project_id, instance_id)
        if override is not None:
            return override
        instance = self.instances.get(instance_id)
        if instance is None:
            return instance_id in _ENABLED_WHEN_UNCONFIGURED
        instance_flag = _bool_or_none(instance.get('enabled'))
        config_flag = _config_flag(instance.get('config'))
        if instance_flag is False or config_flag is False:
            return False
        if instance_flag is not None:
            return instance_flag
        if config_flag is not None:
            return config_flag
        return self._driver(instance_id) not in _DISABLED_BY_DEFAULT_DRIVERS

    def project_claude_instance(self, project_id: str | None) -> str | None:
        usable = [
            instance_id
            for instance_id in self._default_instances(project_id)
            if self._is_claude(instance_id) and self.enabled(project_id, instance_id)
        ]
        if usable:
            return usable[0]
        enabled_claude = [
            instance_id
            for instance_id in self._known_instances()
            if self._is_claude(instance_id) and self.enabled(project_id, instance_id)
        ]
        return enabled_claude[0] if len(enabled_claude) == 1 else None

    def _default_instances(self, project_id: str | None) -> list[str]:
        project = self.project_overrides.get(project_id or '', {})
        candidates = [_selection_instance(project.get('defaultModelSelection')), self.default_instance]
        return [instance_id for instance_id in candidates if instance_id]

    def _known_instances(self) -> list[str]:
        return [*self.instances, *(i for i in _ENABLED_WHEN_UNCONFIGURED if i not in self.instances)]

    def _driver(self, instance_id: str) -> str:
        instance = self.instances.get(instance_id)
        return str(instance.get('driver')) if instance is not None else instance_id

    def _is_claude(self, instance_id: str) -> bool:
        return self._driver(instance_id) == CLAUDE_DRIVER

    def _project_instance_flag(self, project_id: str | None, instance_id: str) -> bool | None:
        if project_id is None:
            return None
        instances = _mapping_of_mappings(
            self.project_overrides.get(project_id, {}).get('providerInstances')
        )
        return _bool_or_none(instances.get(instance_id, {}).get('enabled'))


def resolve_instance(
    userdata: Path, project_id: str | None, explicit: str | None
) -> ProviderChoice:
    settings = ProviderSettings.read(userdata)
    if explicit:
        enabled = settings.enabled(project_id, explicit) if settings else None
        return ProviderChoice(explicit, ProviderSource.EXPLICIT, enabled)
    if settings is None:
        return ProviderChoice(FALLBACK_INSTANCE, ProviderSource.FALLBACK, None)
    project_instance = settings.project_claude_instance(project_id)
    if project_instance is not None:
        return ProviderChoice(project_instance, ProviderSource.PROJECT, True)
    return ProviderChoice(
        FALLBACK_INSTANCE, ProviderSource.FALLBACK, settings.enabled(project_id, FALLBACK_INSTANCE)
    )


def _mapping_of_mappings(value: object) -> Mapping[str, Mapping[str, Any]]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items() if isinstance(item, dict)}


def _selection_instance(selection: object) -> str | None:
    instance_id = selection.get('instanceId') if isinstance(selection, dict) else None
    return instance_id if isinstance(instance_id, str) and instance_id else None


def _config_flag(config: object) -> bool | None:
    return _bool_or_none(config.get('enabled')) if isinstance(config, dict) else None


def _bool_or_none(value: object) -> bool | None:
    return value if isinstance(value, bool) else None
