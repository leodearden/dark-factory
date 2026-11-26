"""FrozenModel: the one pydantic base every arm-harness record and spec shares.

Unknown fields are rejected and instances are immutable, so a record read back from
disk is exactly the shape that was written and cannot be edited after validation.
"""

from pydantic import BaseModel, ConfigDict


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
