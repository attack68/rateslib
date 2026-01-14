# SPDX-License-Identifier: LicenseRef-Rateslib-Dual
#
# Copyright (c) 2026 Siffrorna Technology Limited
#
# Dual-licensed: Free Educational Licence or Paid Commercial Licence (commercial/professional use)
# Source-available, not open source.
#
# See LICENSE and https://rateslib.com/py/en/latest/i_licence.html for details,
# and/or contact info (at) rateslib (dot) com
####################################################################################################


from __future__ import annotations

from enum import Enum
from typing import Any, Generic, NoReturn, TypeAlias, TypeVar

T = TypeVar("T")


class Err:
    """
    Standard result class indicating **failure** and containing some *Exception* type.
    """

    _exception: Exception

    def __init__(self, exception: Exception) -> None:
        self._exception = exception

    def __repr__(self) -> str:
        return f"<rl.DelayedException at {hex(id(self))}>"

    @property
    def is_err(self) -> bool:
        return True

    @property
    def is_ok(self) -> bool:
        return False

    def unwrap(self) -> NoReturn:
        raise self._exception


class Ok(Generic[T]):
    """Standard result class indicating **success** and containing some value."""

    _value: T

    def __init__(self, value: T) -> None:
        self._value = value

    def __repr__(self) -> str:
        return f"<rl.Result {self._value.__repr__()}>"

    @property
    def is_err(self) -> bool:
        return False

    @property
    def is_ok(self) -> bool:
        return True

    def unwrap(self) -> T:
        return self._value


Result: TypeAlias = Ok[T] | Err


class NoInput(Enum):
    """
    Enumerable type to handle setting default values.

    See :ref:`default values <defaults-doc>`.
    """

    blank = 0
    inherit = 1
    negate = -1


def _validate_obj_not_no_input(obj: T | NoInput, expected: str) -> T:
    if isinstance(obj, NoInput):
        raise ValueError(f"Object of type `{expected}` must be supplied. Got NoInput.")
    return obj


def _try_validate_obj_not_no_input(obj: T | NoInput, expected: str) -> Result[T]:
    if isinstance(obj, NoInput):
        return Err(ValueError(f"Object of type `{expected}` must be supplied. Got NoInput."))
    else:
        return Ok(obj)


def _drb(default: Any, possible_ni: Any | NoInput) -> Any:
    """(D)efault (r)eplaces (b)lank"""
    return default if isinstance(possible_ni, NoInput) else possible_ni
