"""Collection-only HTTP input."""

from __future__ import annotations

import re
from typing import Annotated

from pydantic import Field, model_validator

from sim_engine.config import StrictModel


class StatusRequest(StrictModel):
    request_ids: Annotated[list[str], Field(min_length=1, max_length=256)]

    @model_validator(mode="after")
    def valid_ids(self):
        if any(not re.fullmatch(r"[a-f0-9]{64}", v) for v in self.request_ids):
            raise ValueError("invalid collection identity")
        return self
