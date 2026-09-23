"""Shared response validation policy and explicit application failures."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class ContractModel(BaseModel):
    """Reject undeclared fields and non-finite numbers at public boundaries."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ErrorResponse(ContractModel):
    """An expected application failure; code is stable, error explains context."""

    success: Literal[False]
    error: str
    code: str
