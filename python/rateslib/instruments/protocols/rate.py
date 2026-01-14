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

from rateslib.enums.generics import NoInput

if TYPE_CHECKING:
    from rateslib.typing import (
        CurvesT_,
        DualTypes,
        FXForwards_,
        Solver_,
        VolT_,
        datetime_,
        str_,
    )


class _WithRate(Protocol):
    """
    Protocol to establish a *rate* pricing metric of any *Instrument* type.
    """

    _rate_scalar: float

    def rate(
        self,
        *,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: VolT_ = NoInput(0),
        base: str_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
        metric: str_ = NoInput(0),
    ) -> DualTypes:
        # Overloaded rate docs are for: IndexFixedRateBond
        """
        Calculate some pricing rate metric for the *Instrument*.

        .. rubric:: Examples

        The default metric for an :class:`~rateslib.instruments.irs.IRS` is its fixed *'rate'*.

        .. ipython:: python
           :suppress:

           from rateslib import dt, Curve, IRS

        .. ipython:: python

           curve = Curve({dt(2000, 1, 1): 1.0, dt(2010, 1, 1): 0.75})
           irs = IRS(dt(2000, 1, 1), "3Y", spec="usd_irs", curves=[curve], fixed_rate=2.0)
           irs.rate()       # <- `fixed_rate` on fixed leg to equate value with float leg

        Parameters
        ----------
        curves: _Curves, :green:`optional`
            Pricing objects. See **Pricing** on each *Instrument* for details of allowed inputs.
        solver: Solver, :green:`optional`
            A :class:`~rateslib.solver.Solver` object containing *Curve*, *Smile*, *Surface*, or
            *Cube* mappings for pricing.
        fx: FXForwards, :green:`optional`
            The :class:`~rateslib.fx.FXForwards` object used for forecasting FX rates, if necessary.
        vol: _Vol, :green:`optional`
            Pricing objects. See **Pricing** on each *Instrument* for details of allowed inputs.
        base: str, :green:`optional (set to settlement currency)`
            The currency to convert the *local settlement* NPV to.
        local: bool, :green:`optional (set as False)`
            An override flag to return a dict of NPV values indexed by string currency.
        settlement: datetime, :green:`optional`
            The assumed settlement date of the *PV* determination. Used only to evaluate
            *ex-dividend* status.
        forward: datetime, :green:`optional`
            The future date to project the *PV* to using the ``disc_curve``.
        metric: str, :green:`optional`
            The specific calculation to perform and the value to return.
            See **Pricing** on each *Instrument* for details of allowed inputs.

        Returns
        -------
        float, Dual, Dual2, Variable
        """
        raise NotImplementedError(f"`rate` must be implemented for type: {type(self).__name__}")

    def spread(
        self,
        *,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: VolT_ = NoInput(0),
        base: str_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> DualTypes:
        """
        Calculate some pricing spread metric for the *Instrument*.

        This calculation may be an alias for :meth:`~rateslib.instruments.protocols._WithRate.rate`
        with a specific `metric` and is designated at an *Instrument* level.

        .. rubric:: Examples

        The *'spread'* on an :class:`~rateslib.instruments.irs.IRS` is the float leg spread to
        equate value with the fixed leg.

        .. ipython:: python
           :suppress:

           from rateslib import dt, Curve, IRS

        .. ipython:: python

           curve = Curve({dt(2000, 1, 1): 1.0, dt(2010, 1, 1): 0.75})
           irs = IRS(dt(2000, 1, 1), "3Y", spec="usd_irs", curves=[curve], fixed_rate=2.0)
           irs.spread()       # <- `spread` on float leg to equate value with fixed leg

        Parameters
        ----------
        curves: _Curves, :green:`optional`
            Pricing objects. See **Pricing** on each *Instrument* for details of allowed inputs.
        solver: Solver, :green:`optional`
            A :class:`~rateslib.solver.Solver` object containing *Curve*, *Smile*, *Surface*, or
            *Cube* mappings for pricing.
        fx: FXForwards, :green:`optional`
            The :class:`~rateslib.fx.FXForwards` object used for forecasting FX rates, if necessary.
        vol: _Vol, :green:`optional`
            Pricing objects. See **Pricing** on each *Instrument* for details of allowed inputs.
        base: str, :green:`optional (set to settlement currency)`
            The currency to convert the *local settlement* NPV to.
        local: bool, :green:`optional (set as False)`
            An override flag to return a dict of NPV values indexed by string currency.
        settlement: datetime, :green:`optional`
            The assumed settlement date of the *PV* determination. Used only to evaluate
            *ex-dividend* status.
        forward: datetime, :green:`optional`
            The future date to project the *PV* to using the ``disc_curve``.

        Returns
        -------
        float, Dual, Dual2, Variable
        """
        raise NotImplementedError(f"`spread` is not implemented for type: {type(self).__name__}")

    @property
    def rate_scalar(self) -> float:
        """
        A scaling quantity associated with the :class:`~rateslib.solver.Solver` risk calculations.
        """
        return self._rate_scalar
