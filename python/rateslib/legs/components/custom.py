from __future__ import annotations

from typing import TYPE_CHECKING

from rateslib.legs.components.protocols import _BaseLeg
from rateslib.periods.components import _BasePeriod

if TYPE_CHECKING:
    pass


class CustomLeg(_BaseLeg):
    """
    Create a leg contained of user specified ``Periods``.

    Useful for crafting amortising swaps with custom notional and date schedules.

    Parameters
    ----------
    periods : iterable of _BasePeriod
        A sequence of *Periods* to attach to the leg.

    Examples
    --------

    .. ipython:: python
       :suppress:

       from rateslib.legs.components import CustomLeg
       from rateslib.periods.components import FixedPeriod

    .. ipython:: python

       fp1 = FixedPeriod(
           start=dt(2021,1,1),
           end=dt(2021,7,1),
           payment=dt(2021,7,2),
           frequency="Q",
           notional=1e6,
           convention="Act365F",
           fixed_rate=2.10
       )
       fp2 = FixedPeriod(
           start=dt(2021,3,7),
           end=dt(2021,9,7),
           payment=dt(2021,9,8),
           frequency="Q",
           notional=-5e6,
           convention="Act365F",
           fixed_rate=3.10
       )
       custom_leg = CustomLeg(periods=[fp1, fp2])
       custom_leg.cashflows()

    """  # noqa: E501

    @property
    def periods(self) -> list[_BasePeriod]:
        """Combine all period collection types into an ordered list."""
        return self._periods

    def __init__(self, periods: list[_BasePeriod]) -> None:
        if not all(isinstance(p, _BasePeriod) for p in periods):
            raise ValueError(
                "Each object in `periods` must be an instance of `_BasePeriod`.",
            )
        self._periods = periods
