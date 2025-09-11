from __future__ import annotations

from enum import Enum
from typing import Any, Generic, NoReturn, TypeAlias, TypeVar

T = TypeVar("T")


class Err:
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


def _drb(default: Any, possible_ni: Any | NoInput) -> Any:
    """(D)efault (r)eplaces (b)lank"""
    return default if isinstance(possible_ni, NoInput) else possible_ni
