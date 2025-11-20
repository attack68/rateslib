from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        FixedLeg,
        FloatLeg,
        datetime,
    )


class _WithExDiv(Protocol):
    """
    Protocol to determine the *yield-to-maturity* of a bond type *Instrument*.
    """

    @property
    def leg1(self) -> FixedLeg | FloatLeg: ...

    def ex_div(self, settlement: datetime) -> bool:
        return self.leg1.ex_div(settlement)
