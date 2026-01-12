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
from rateslib.instruments.protocols.pricing import (
    _get_fx_forwards_maybe_from_solver,
    _maybe_get_curve_maybe_from_solver,
    _maybe_get_curve_or_dict_maybe_from_solver,
    _maybe_get_fx_vol_maybe_from_solver,
    _Vol,
    _WithPricingObjs,
)

if TYPE_CHECKING:
    from rateslib.typing import (
        CurvesT_,
        DualTypes,
        FXForwards_,
        Solver_,
        VolT_,
        _Curves,
        _KWArgs,
        datetime_,
        str_,
    )


class _WithAnalyticDelta(_WithPricingObjs, Protocol):
    """
    Protocol to determine the *analytic rate delta* of a particular *Leg* of an *Instrument*.
    """

    @property
    def kwargs(self) -> _KWArgs: ...

    def analytic_delta(
        self,
        *,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: VolT_ = NoInput(0),
        base: str_ = NoInput(0),
        local: bool = False,
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
        leg: int = 1,
    ) -> DualTypes | dict[str, DualTypes]:
        """
        Calculate the analytic rate delta of a *Leg* of the *Instrument*.

        .. rubric:: Examples

        .. ipython:: python
           :suppress:

           from rateslib import dt, Curve, IRS

        .. ipython:: python

           curve = Curve({dt(2000, 1, 1): 1.0, dt(2010, 1, 1): 0.75})
           irs = IRS(dt(2000, 1, 1), "3Y", spec="usd_irs", fixed_rate=1.0, curves=[curve])
           irs.analytic_delta()
           irs.analytic_delta(local=True)

        .. role:: red

        .. role:: green

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
        leg: int, :green:`optional (set as 1)`
            The *Leg* over which to calculate the analytic rate delta.

        Returns
        -------
        float, Dual, Dual2, Variable or dict of such indexed by string currency.
        """
        _curves: _Curves = self._parse_curves(curves)
        _vol: _Vol = self._parse_vol(vol)
        _curves_meta: _Curves = self.kwargs.meta["curves"]
        _vol_meta: _Vol = self.kwargs.meta["vol"]

        prefix = "" if leg == 1 else "leg2_"

        if hasattr(self, "legs"):
            value: DualTypes | dict[str, DualTypes] = self.legs[leg - 1].analytic_delta(
                rate_curve=_maybe_get_curve_or_dict_maybe_from_solver(
                    _curves_meta, _curves, f"{prefix}rate_curve", solver
                ),
                disc_curve=_maybe_get_curve_maybe_from_solver(
                    _curves_meta, _curves, f"{prefix}disc_curve", solver
                ),
                index_curve=_maybe_get_curve_maybe_from_solver(
                    _curves_meta, _curves, f"{prefix}index_curve", solver
                ),
                fx_vol=_maybe_get_fx_vol_maybe_from_solver(_vol_meta, _vol, solver),
                fx=_get_fx_forwards_maybe_from_solver(fx=fx, solver=solver),
                base=base,
                local=local,
                settlement=settlement,
                forward=forward,
            )
        else:
            raise NotImplementedError("`analytic_delta` can only called on Leg based Instruments.")

        return value
