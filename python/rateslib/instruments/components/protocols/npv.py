from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from rateslib.typing import (
        _BaseLeg,
    )


class _WithNPV(Protocol):
    """
    Protocol to establish value of any *Leg* type.

    """

    _legs: list[_BaseLeg]

    @property
    def legs(self) -> list[_BaseLeg]:
        return self._legs

    def __repr__(self) -> str:
        return f"<rl.{type(self).__name__} at {hex(id(self))}>"
