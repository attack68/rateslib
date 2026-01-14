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

from typing import TYPE_CHECKING, Protocol

from rateslib.dual import Dual, Dual2, gradient
from rateslib.dual.utils import _dual_float

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        Any,
        DualTypes,
        FixedLeg,
        FloatLeg,
        datetime,
    )


class _WithDuration(Protocol):
    """
    Protocol to determine the *yield-to-maturity* of a bond type *Instrument*.
    """

    def price(self, *args: Any, **kwargs: Any) -> DualTypes: ...

    @property
    def leg1(self) -> FixedLeg | FloatLeg: ...

    def duration(self, ytm: DualTypes, settlement: datetime, metric: str = "risk") -> float:
        """
        Return the (negated) derivative of ``price`` w.r.t. ``ytm``.

        Parameters
        ----------
        ytm : float
            The yield-to-maturity for the bond.
        settlement : datetime
            The settlement date of the bond.
        metric : str
            The specific duration calculation to return. See notes.

        Returns
        -------
        float

        Notes
        -----
        The available metrics are:

        - *"risk"*: the derivative of price w.r.t. ytm, scaled to -1bp.

          .. math::

             risk = - \\frac{\\partial P }{\\partial y}

        - *"modified"*: the modified duration which is *risk* divided by dirty price.

          .. math::

             mod \\; duration = \\frac{risk}{P} = - \\frac{1}{P} \\frac{\\partial P }{\\partial y}

        - *"duration"* (or *"macaulay"*): the duration which is modified duration reverse modified.

          .. math::

             duration = mod \\; duration \\times (1 + y / f)

        Examples
        --------
        .. ipython:: python

           gilt = FixedRateBond(
               effective=dt(1998, 12, 7),
               termination=dt(2015, 12, 7),
               frequency="S",
               calendar="ldn",
               currency="gbp",
               convention="ActActICMA",
               ex_div=7,
               fixed_rate=8.0
           )
           gilt.duration(4.445, dt(1999, 5, 27), "risk")
           gilt.duration(4.445, dt(1999, 5, 27), "modified")
           gilt.duration(4.445, dt(1999, 5, 27), "duration")

        This result is interpreted as cents. If the yield is increased by 1bp the price
        will fall by 14.65 cents.

        .. ipython:: python

           gilt.price(4.445, dt(1999, 5, 27))
           gilt.price(4.455, dt(1999, 5, 27))
        """
        # TODO: this is not AD safe: returns only float
        ytm_: float = _dual_float(ytm)
        if metric == "risk":
            price_dual: Dual = self.price(Dual(ytm_, ["__y__§"], []), settlement)  # type: ignore[assignment]
            _: float = -gradient(price_dual, ["__y__§"])[0]
        elif metric == "modified":
            price_dual = -self.price(Dual(ytm_, ["__y__§"], []), settlement, dirty=True)  # type: ignore[assignment]
            _ = -gradient(price_dual, ["__y__§"])[0] / float(price_dual) * 100
        elif metric == "duration" or metric == "macaulay":
            price_dual = self.price(Dual(ytm_, ["__y__§"], []), settlement, dirty=True)  # type: ignore[assignment]
            f = self.leg1.schedule.periods_per_annum
            v = 1 + ytm_ / (100 * f)
            _ = -gradient(price_dual, ["__y__§"])[0] / float(price_dual) * v * 100
        return _

    def convexity(self, ytm: DualTypes, settlement: datetime, metric: str = "risk") -> float:
        """
        Return the second derivative of ``price`` w.r.t. ``ytm``.

        Parameters
        ----------
        ytm : float
            The yield-to-maturity for the bond.
        settlement : datetime
            The settlement date of the bond.
        metric: str, optional

        Returns
        -------
        float

        Notes
        ------
        The default metric is similar to the :meth:`duration` method and is *'risk'* based,
        but the traditional calculation is available.

        - *"risk"*: the second derivative of price w.r.t. ytm, scaled to -1bp.

          .. math::

             risk = \\frac{\\partial^2 P }{\\partial y^2}

        - *"convexity"*: the standard formula for convexity which is the above scaled by price.

          .. math::

             convexity = \\frac{1}{P} \\frac{\\partial P^2 }{\\partial y^2}

        Examples
        --------
        .. ipython:: python
           :suppress:

           from rateslib import FixedRateBond

        .. ipython:: python

           gilt = FixedRateBond(
               effective=dt(1998, 12, 7),
               termination=dt(2015, 12, 7),
               frequency="S",
               calendar="ldn",
               currency="gbp",
               convention="ActActICMA",
               ex_div=7,
               fixed_rate=8.0
           )
           gilt.convexity(4.445, dt(1999, 5, 27))

        This number is interpreted as hundredths of a cent. For a 1bp increase in
        yield the duration will decrease by 2 hundredths of a cent.

        .. ipython:: python

           gilt.duration(4.445, dt(1999, 5, 27))
           gilt.duration(4.455, dt(1999, 5, 27))
        """
        # TODO: method is not AD safe: returns float
        ytm_: float = _dual_float(ytm)
        d = self.price(Dual2(ytm_, ["_ytm__§"], [], []), settlement, dirty=True)
        ret: float = gradient(d, ["_ytm__§"], 2)[0][0]

        if metric == "risk":
            return ret
        elif metric == "convexity":
            return ret * 100.0 / _dual_float(d)
        return ret
