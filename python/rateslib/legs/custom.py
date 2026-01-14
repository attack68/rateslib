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

from typing import TYPE_CHECKING

from rateslib.legs.protocols import _BaseLeg
from rateslib.periods.protocols import _BasePeriod

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        Any,
        DualTypes,
        Sequence,
    )


class CustomLeg(_BaseLeg):
    """
    A *Leg* containing user specified :class:`~rateslib.periods._BasePeriod`.

    .. rubric:: Examples

    .. ipython:: python
       :suppress:

       from rateslib.legs import CustomLeg
       from rateslib.periods import FixedPeriod

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

    Parameters
    ----------
    periods : iterable of _BasePeriod
        A sequence of *Periods* to attach to the leg.

    """  # noqa: E501

    @property
    def periods(self) -> Sequence[_BasePeriod]:
        """Combine all period collection types into an ordered list."""
        return self._periods

    def __init__(self, periods: Sequence[_BasePeriod]) -> None:
        if not all(isinstance(p, _BasePeriod) for p in periods):
            raise ValueError(
                "Each object in `periods` must be an instance of `_BasePeriod`.",
            )
        self._periods = periods

    def spread(self, *args: Any, **kwargs: Any) -> DualTypes:
        return super().spread(*args, **kwargs)  # type: ignore[safe-super]
