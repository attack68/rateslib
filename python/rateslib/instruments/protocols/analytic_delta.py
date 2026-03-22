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
    _get_curve,
    _get_fx_forwards_maybe_from_solver,
    _get_fx_vol,
    _parse_curves,
    _parse_vol,
    _WithPricingObjs,
)

if TYPE_CHECKING:
    from rateslib.local_types import (
        CurvesT_,
        DualTypes,
        FXForwards_,
        Solver_,
        VolT_,
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
        c = _parse_curves(self, curves, solver)  # type: ignore[arg-type]
        v = _parse_vol(self, vol, solver, False)  # type: ignore[call-overload, misc]

        prefix = "" if leg == 1 else "leg2_"

        if hasattr(self, "legs"):
            rate_curve = _get_curve(f"{prefix}rate_curve", True, True, *c)
            disc_curve = _get_curve(f"{prefix}disc_curve", False, True, *c)
            index_curve = _get_curve(f"{prefix}index_curve", False, True, *c)
            value: DualTypes | dict[str, DualTypes] = self.legs[leg - 1].analytic_delta(
                rate_curve=rate_curve,
                disc_curve=disc_curve,
                index_curve=index_curve,
                fx_vol=_get_fx_vol(True, True, *v),
                fx=_get_fx_forwards_maybe_from_solver(fx=fx, solver=solver),
                base=base,
                local=local,
                settlement=settlement,
                forward=forward,
            )
        else:
            raise NotImplementedError("`analytic_delta` can only called on Leg based Instruments.")

        return value
