"""Config shapes shared by the config_fields and config_reload tests: a role table and a submodel."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class Role(StrEnum):
    ARCHITECT = 'architect'
    REVIEWER = 'reviewer'
    UNBLOCK_AUTO = 'unblock_auto'


class Row(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    model: str = 'sonnet'
    budget_usd: float = 1.0
    max_turns: int = 10


ROW_KNOBS = tuple(Row.model_fields)


class Sub(BaseModel):
    alpha: int = 1
    beta: str = 'b'


def assign_item(container, key, value) -> None:
    container[key] = value
