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
from rateslib.fx import FXForwards, FXRates
from rateslib.instruments.protocols.npv import _WithNPV
from rateslib.instruments.protocols.pricing import (
    _get_fx_forwards_maybe_from_solver,
)

if TYPE_CHECKING:
    from rateslib.typing import (
        CurvesT_,
        DataFrame,
        Dual,
        Dual2,
        FXForwards_,
        NoInput,
        Solver_,
        VolT_,
        datetime_,
        str_,
    )


class _WithSensitivities(_WithNPV, Protocol):
    """
    Protocol to establish **delta** and **gamma** calculations using a
    :class:`~rateslib.solver.Solver` of any *Instrument* type.
    """

    def delta(
        self,
        *,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: VolT_ = NoInput(0),
        base: str_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> DataFrame:
        """
        Calculate delta risk of an *Instrument* against the calibrating instruments in a
        :class:`~rateslib.solver.Solver`.

        .. rubric:: Examples

        .. ipython:: python
           :suppress:

           from rateslib import IRS, Curve, Solver, dt

        .. ipython:: python

           curve = Curve({dt(2000, 1, 1): 1.0, dt(2002, 1, 1): 0.85, dt(2010, 1, 1): 0.75})
           solver = Solver(
               curves=[curve],
               instruments=[
                   IRS(dt(2000, 1, 1), "2Y", spec="usd_irs", curves=[curve]),
                   IRS(dt(2000, 1, 1), "5Y", spec="usd_irs", curves=[curve]),
               ],
               s=[2.0, 2.25],
               instrument_labels=["2Y", "5Y"],
               id="US_RATES"
           )
           irs = IRS(dt(2000, 1, 1), "3Y", spec="usd_irs", curves=[curve])
           irs.delta(solver=solver)

        Parameters
        ----------
        curves: _Curves, :green:`optional`
            Pricing objects. See **Pricing** on each *Instrument* for details of allowed inputs.
        solver: Solver, :red:`required`
            A :class:`~rateslib.solver.Solver` object containing *Curve*, *Smile*, *Surface*, or
            *Cube* mappings for pricing.
        fx: FXForwards, :green:`optional`
            The :class:`~rateslib.fx.FXForwards` object used for forecasting FX rates, if necessary.
        vol: _Vol, :green:`optional`
            Pricing objects. See **Pricing** on each *Instrument* for details of allowed inputs.
        base: str, :green:`optional (set to settlement currency)`
            The currency to convert the *local settlement* NPV to.
        settlement: datetime, :green:`optional`
            The assumed settlement date of the *PV* determination. Used only to evaluate
            *ex-dividend* status.
        forward: datetime, :green:`optional`
            The future date to project the *PV* to using the ``disc_curve``.

        Returns
        -------
        DataFrame

        Notes
        -----
        **Delta** measures the sensitivity of the *PV* to a change in any of the calibrating
        instruments of the given :class:`~rateslib.solver.Solver`. Values are returned
        according to the ``rate_scalar`` quantity at an *Instrument* level and according to the
        ``metric`` used to derive the :meth:`~rateslib.instruments.protocols._WithRate.rate`
        method of each *Instrument*.

        """
        if isinstance(solver, NoInput):
            raise ValueError("`solver` is required for delta/gamma methods.")
        npv: dict[str, Dual] = self.npv(  # type: ignore[assignment]
            curves=curves,
            solver=solver,
            fx=fx,
            vol=vol,
            base=base,
            forward=forward,
            settlement=settlement,
            local=True,
        )
        return solver.delta(
            npv=npv, base=base, fx=_get_fx_forwards_maybe_from_solver(fx=fx, solver=solver)
        )

    def exo_delta(
        self,
        *,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: VolT_ = NoInput(0),
        base: str_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
        vars: list[str],  # noqa: A002
        vars_scalar: list[float] | NoInput = NoInput(0),
        vars_labels: list[str] | NoInput = NoInput(0),
    ) -> DataFrame:
        """
        Calculate delta risk of an *Instrument* against some exogenous user created *Variables*,
        via a :class:`~rateslib.solver.Solver`.

        See :ref:`What are exogenous variables? <cook-exogenous-doc>` in the cookbook.

        .. rubric:: Examples

        This example calculates the risk of the fixed rate increasing by 1bp and the notional
        increasing by 1mm. Mathematically this should be equivalent to the `npv` and the
        `analytic delta` (although the calculation is based on AD and is completely independent
        of the solver).

        .. ipython:: python
           :suppress:

           from rateslib import IRS, Curve, Solver, dt, Variable

        .. ipython:: python

           curve = Curve({dt(2000, 1, 1): 1.0, dt(2002, 1, 1): 0.85, dt(2010, 1, 1): 0.75})
           solver = Solver(
               curves=[curve],
               instruments=[
                   IRS(dt(2000, 1, 1), "2Y", spec="usd_irs", curves=[curve]),
                   IRS(dt(2000, 1, 1), "5Y", spec="usd_irs", curves=[curve]),
               ],
               s=[2.0, 2.25],
               instrument_labels=["2Y", "5Y"],
               id="US_RATES"
           )
           irs = IRS(dt(2000, 1, 1), "3Y", spec="usd_irs", fixed_rate=Variable(3.0, ["R"]), notional=Variable(1e6, ["N"]), curves=[curve])
           irs.exo_delta(solver=solver, vars=["R", "N"], vars_scalar=[1e-2, 1e6])
           irs.analytic_delta()
           irs.npv()

        Parameters
        ----------
        curves: _Curves, :green:`optional`
            Pricing objects. See **Pricing** on each *Instrument* for details of allowed inputs.
        solver: Solver, :red:`required`
            A :class:`~rateslib.solver.Solver` object containing *Curve*, *Smile*, *Surface*, or
            *Cube* mappings for pricing.
        fx: FXForwards, :green:`optional`
            The :class:`~rateslib.fx.FXForwards` object used for forecasting FX rates, if necessary.
        vol: _Vol, :green:`optional`
            Pricing objects. See **Pricing** on each *Instrument* for details of allowed inputs.
        base: str, :green:`optional (set to settlement currency)`
            The currency to convert the *local settlement* NPV to.
        settlement: datetime, :green:`optional`
            The assumed settlement date of the *PV* determination. Used only to evaluate
            *ex-dividend* status.
        forward: datetime, :green:`optional`
            The future date to project the *PV* to using the ``disc_curve``.
        vars : list[str], :red:`required`
            The variable tags which to determine sensitivities for.
        vars_scalar : list[float], :green:`optional`
            Scaling factors for each variable, for example converting rates to basis point etc.
            Defaults to ones.
        vars_labels : list[str], :green:`optional`
            Alternative names to relabel variables in DataFrames.

        Returns
        -------
        DataFrame
        """  # noqa: E501
        if isinstance(solver, NoInput):
            raise ValueError("`solver` is required for delta/gamma methods.")
        npv: dict[str, Dual | Dual2] = self.npv(  # type: ignore[assignment]
            curves=curves,
            solver=solver,
            fx=fx,
            vol=vol,
            base=base,
            forward=forward,
            settlement=settlement,
            local=True,
        )
        return solver.exo_delta(
            npv=npv,
            vars=vars,
            base=base,
            fx=_get_fx_forwards_maybe_from_solver(fx=fx, solver=solver),
            vars_scalar=vars_scalar,
            vars_labels=vars_labels,
        )

    def gamma(
        self,
        *,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: VolT_ = NoInput(0),
        base: str_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> DataFrame:
        """
        Calculate cross-gamma risk of an *Instrument* against the calibrating instruments of a
        :class:`~rateslib.solver.Solver`.

        .. rubric:: Examples

        .. ipython:: python
           :suppress:

           from rateslib import IRS, Curve, Solver, dt

        .. ipython:: python

           curve = Curve({dt(2000, 1, 1): 1.0, dt(2002, 1, 1): 0.85, dt(2010, 1, 1): 0.75})
           solver = Solver(
               curves=[curve],
               instruments=[
                   IRS(dt(2000, 1, 1), "2Y", spec="usd_irs", curves=[curve]),
                   IRS(dt(2000, 1, 1), "5Y", spec="usd_irs", curves=[curve]),
               ],
               s=[2.0, 2.25],
               instrument_labels=["2Y", "5Y"],
               id="US_RATES"
           )
           irs = IRS(dt(2000, 1, 1), "3Y", spec="usd_irs", curves=[curve])
           irs.gamma(solver=solver)

        Parameters
        ----------
        curves: _Curves, :green:`optional`
            Pricing objects. See **Pricing** on each *Instrument* for details of allowed inputs.
        solver: Solver, :red:`required`
            A :class:`~rateslib.solver.Solver` object containing *Curve*, *Smile*, *Surface*, or
            *Cube* mappings for pricing.
        fx: FXForwards, :green:`optional`
            The :class:`~rateslib.fx.FXForwards` object used for forecasting FX rates, if necessary.
        vol: _Vol, :green:`optional`
            Pricing objects. See **Pricing** on each *Instrument* for details of allowed inputs.
        base: str, :green:`optional (set to settlement currency)`
            The currency to convert the *local settlement* NPV to.
        settlement: datetime, :green:`optional`
            The assumed settlement date of the *PV* determination. Used only to evaluate
            *ex-dividend* status.
        forward: datetime, :green:`optional`
            The future date to project the *PV* to using the ``disc_curve``.

        Returns
        -------
        DataFrame

        Notes
        -----
        **Gamma** measures the second order cross-sensitivity of the *PV* to a change in any
        of the calibrating instruments of the given :class:`~rateslib.solver.Solver`. Values are
        returned according to the ``rate_scalar`` quantity at an *Instrument* level and according
        to the ``metric`` used to derive the :meth:`~rateslib.instruments.protocols._WithRate.rate`
        method of each *Instrument*.
        """
        if isinstance(solver, NoInput):
            raise ValueError("`solver` is required for delta/gamma methods.")

        fx_ = _get_fx_forwards_maybe_from_solver(fx=fx, solver=solver)
        # store original order
        if id(solver.fx) != id(fx_) and isinstance(fx_, FXRates | FXForwards):
            # then the fx_ object is available on solver but that is not being used.
            _ad_fx = fx_._ad
            fx_._set_ad_order(2)

        _ad_svr = solver._ad
        solver._set_ad_order(2)

        npv: dict[str, Dual2] = self.npv(  # type: ignore[assignment]
            curves=curves,
            solver=solver,
            fx=fx_,
            vol=vol,
            base=NoInput(0),  # local override
            settlement=settlement,
            forward=forward,
            local=True,
        )
        grad_s_sT_P: DataFrame = solver.gamma(npv, base, fx_)

        # reset original order
        if id(solver.fx) != id(fx_) and isinstance(fx_, FXRates | FXForwards):
            fx_._set_ad_order(_ad_fx)
        solver._set_ad_order(_ad_svr)

        return grad_s_sT_P
