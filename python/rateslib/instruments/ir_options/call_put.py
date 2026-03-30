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

from abc import ABCMeta
from datetime import datetime
from typing import TYPE_CHECKING, NoReturn

from rateslib import defaults
from rateslib.curves._parsers import _validate_obj_not_no_input
from rateslib.default import plot
from rateslib.dual.utils import _dual_float
from rateslib.enums.generics import NoInput, _drb
from rateslib.enums.parameters import (
    IROptionMetric,
    SwaptionSettlementMethod,
    _get_ir_option_metric,
)
from rateslib.instruments.irs import IRS
from rateslib.instruments.protocols import _BaseInstrument, _KWArgs
from rateslib.instruments.protocols.pricing import (
    _Curves,
    _get_curve,
    _maybe_get_ir_vol_maybe_from_solver,
    _parse_curves,
    _Vol,
)
from rateslib.legs import CustomLeg
from rateslib.periods import Cashflow, IRSCallPeriod, IRSPutPeriod
from rateslib.periods.utils import (
    _get_ir_vol_value_and_forward_maybe_from_obj,
)
from rateslib.volatility.fx import FXVolObj
from rateslib.volatility.ir import _BaseIRSmile
from rateslib.volatility.ir.utils import _get_ir_expiry_and_payment

if TYPE_CHECKING:
    from rateslib.local_types import (  # pragma: no cover
        Any,
        Arr1dF64,
        CurveOption_,
        CurvesT_,
        DataFrame,
        DualTypes,
        DualTypes_,
        FXForwards_,
        IRSSeries,
        PlotOutput,
        Sequence,
        Solver_,
        VolT_,
        _BaseCurve_,
        _BaseIRSOptionPeriod,
        _BaseLeg,
        _IRVolOption_,
        _IRVolPricingParams,
        bool_,
        datetime_,
        float_,
        str_,
    )


class _BaseIRSOption(_BaseInstrument, metaclass=ABCMeta):
    """
    Abstract base class for implementing *IR Swaptions*.

    See :class:`~rateslib.instruments.IRSCall` and
    :class:`~rateslib.instruments.IRSPut`.
    """

    _pricing: _IRVolPricingParams

    def analytic_greeks(
        self,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: VolT_ = NoInput(0),
    ) -> dict[str, Any]:
        """
        Return various pricing metrics of the *FX Option*.

        .. rubric:: Examples

        .. ipython:: python
           :suppress:

           from rateslib import Curve, FXCall, dt, FXForwards, FXRates, FXDeltaVolSmile

        .. ipython:: python

           eur = Curve({dt(2020, 1, 1): 1.0, dt(2021, 1, 1): 0.98})
           usd = Curve({dt(2020, 1, 1): 1.0, dt(2021, 1, 1): 0.96})
           fxf = FXForwards(
               fx_rates=FXRates({"eurusd": 1.10}, settlement=dt(2020, 1, 3)),
               fx_curves={"eureur": eur, "eurusd": eur, "usdusd": usd},
           )
           fxvs = FXDeltaVolSmile(
               nodes={0.25: 11.0, 0.5: 9.8, 0.75: 10.7},
               delta_type="forward",
               eval_date=dt(2020, 1, 1),
               expiry=dt(2020, 4, 1)
           )
           fxc = FXCall(
               expiry="3m",
               strike=1.10,
               eval_date=dt(2020, 1, 1),
               spec="eurusd_call",
           )
           fxc.analytic_greeks(fx=fxf, curves=[eur, usd], vol=fxvs)

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

        Returns
        -------
        dict
        """
        return self._analytic_greeks_set_metrics(
            curves=curves,
            solver=solver,
            fx=fx,
            vol=vol,
            set_metrics=True,
        )

    def _analytic_greeks_set_metrics(
        self,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: VolT_ = NoInput(0),
        set_metrics: bool_ = True,
    ) -> dict[str, Any]:
        """
        Return various pricing metrics of the *FX Option*.

        Returns
        -------
        float, Dual, Dual2
        """
        c = _parse_curves(self, curves, solver)
        rate_curve = _get_curve("rate_curve", True, False, *c)
        disc_curve = _get_curve("disc_curve", False, False, *c)
        index_curve = _get_curve("index_curve", False, False, *c)

        _vol = self._parse_vol(vol)

        ir_vol = _maybe_get_ir_vol_maybe_from_solver(
            vol=_vol, vol_meta=self.kwargs.meta["vol"], solver=solver
        )

        if set_metrics:
            self._set_strike_and_vol(
                rate_curve=rate_curve, disc_curve=disc_curve, index_curve=index_curve, vol=ir_vol
            )
            # self._set_premium(curves, fx)

        return self._option.analytic_greeks(
            rate_curve=rate_curve,
            disc_curve=disc_curve,
            index_curve=index_curve,
            ir_vol=ir_vol,
            premium=NoInput(0),
            premium_payment=NoInput(0),
        )

    def local_analytic_rate_fixings(
        self,
        *,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: VolT_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> DataFrame:
        raise NotImplementedError(
            "`local_analytic_rate_fixings` is not implemented for `_BaseIRSOption` types."
        )

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
        raise NotImplementedError("`spread` is not implemented for `_BaseIRSOption` types.")

    @property
    def _rate_scalar(self) -> float:  # type: ignore[override]
        if type(self.kwargs.meta["metric"]) in [
            IROptionMetric.BlackVolShift,
            IROptionMetric.NormalVol,
        ]:
            return 100.0
        else:
            return 1.0

    @property
    def leg1(self) -> CustomLeg:
        """The :class:`~rateslib.legs.CustomLeg` of the *Instrument* containing the
        :class:`~rateslib.periods.IROptionPeriod`."""
        return self._leg1

    @property
    def leg2(self) -> CustomLeg:
        """The :class:`~rateslib.legs.CustomLeg` of the *Instrument* containing the
        premium :class:`~rateslib.periods.Cashflow`."""
        return self._leg2

    @property
    def legs(self) -> Sequence[_BaseLeg]:
        """A list of the *Legs* of the *Instrument*."""
        return self._legs

    @property
    def _option(self) -> _BaseIRSOptionPeriod:
        return self.leg1.periods[0]  # type: ignore[return-value]

    @property
    def _irs(self) -> IRS:
        return self._option.ir_option_params.option_fixing.irs

    @property
    def _premium(self) -> Cashflow:
        return self.leg2.periods[0]  # type: ignore[return-value]

    @classmethod
    def _parse_curves(cls, curves: CurvesT_) -> _Curves:
        """
        A Swaption has 3 curve requirements. See **Pricing**.
        """
        if isinstance(curves, NoInput):
            return _Curves()
        elif isinstance(curves, list | tuple):
            if len(curves) == 1:
                return _Curves(
                    rate_curve=curves[0],
                    index_curve=curves[0],
                    disc_curve=curves[0],
                    leg2_disc_curve=curves[0],
                )
            elif len(curves) == 2:
                return _Curves(
                    rate_curve=curves[0],
                    disc_curve=curves[1],
                    index_curve=curves[1],
                    leg2_disc_curve=curves[1],
                )
            elif len(curves) == 3:
                return _Curves(
                    rate_curve=curves[0],
                    disc_curve=curves[1],
                    index_curve=curves[2],
                    leg2_disc_curve=curves[1],
                )
            else:
                raise ValueError(
                    f"{type(cls).__name__} requires only 2 curve types. Got {len(curves)}."
                )
        elif isinstance(curves, dict):
            return _Curves(
                rate_curve=curves.get("rate_curve", NoInput(0)),
                disc_curve=curves.get("disc_curve", NoInput(0)),
                index_curve=curves.get("index_curve", NoInput(0)),
                leg2_disc_curve=_drb(
                    curves.get("disc_curve", NoInput(0)),
                    curves.get("leg2_disc_curve", NoInput(0)),
                ),
            )
        elif isinstance(curves, _Curves):
            return curves
        else:  # `curves` is just a single input which is copied across all curves
            return _Curves(
                rate_curve=curves,  # type: ignore[arg-type]
                disc_curve=curves,  # type: ignore[arg-type]
                index_curve=curves,  # type: ignore[arg-type]
                leg2_disc_curve=curves,  # type: ignore[arg-type]
            )

    @classmethod
    def _parse_vol(cls, vol: VolT_) -> _Vol:
        """
        IR options requires only a single IRVolObj or a scalar.
        """
        if isinstance(vol, _Vol):
            return vol
        elif isinstance(vol, FXVolObj):
            raise TypeError("`vol` cannot be an FX type vol object and must be IR type vol object.")
        else:
            return _Vol(ir_vol=vol)

    def __init__(
        self,
        expiry: datetime | str,
        tenor: datetime | str,
        strike: DualTypes | str,
        irs_series: IRSSeries | str,
        *,
        notional: DualTypes_ = NoInput(0),
        eval_date: datetime | NoInput = NoInput(0),
        premium: DualTypes_ = NoInput(0),
        payment_lag: str | datetime_ = NoInput(0),
        option_fixings: DualTypes_ = NoInput(0),
        settlement_method: SwaptionSettlementMethod | str_ = NoInput(0),
        metric: IROptionMetric | str_ = NoInput(0),
        curves: CurvesT_ = NoInput(0),
        vol: VolT_ = NoInput(0),
        spec: str_ = NoInput(0),
        call: bool = True,
    ):
        user_args = dict(
            tenor=tenor,
            expiry=expiry,
            notional=notional,
            strike=strike,
            irs_series=irs_series,
            option_fixings=option_fixings,
            settlement_method=settlement_method,
            leg2_payment_lag=payment_lag,
            leg2_premium=premium,
            metric=metric,
            curves=self._parse_curves(curves),
            vol=self._parse_vol(vol),
        )
        # instrument_args: dict[str, Any] = dict()
        default_args = dict(
            notional=defaults.notional,
            metric=defaults.ir_option_metric,
            settlement_method=defaults.ir_option_settlement,
        )
        self._kwargs = _KWArgs(
            user_args=user_args,
            default_args=default_args,
            spec=spec,
            meta_args=["curves", "vol", "metric"],
        )

        # determine the `expiry` and `delivery` as datetimes if derived from other combinations
        (self.kwargs.leg1["expiry"], self.kwargs.leg2["payment"]) = _get_ir_expiry_and_payment(
            eval_date=eval_date,
            expiry=self.kwargs.leg1["expiry"],
            irs_series=self.kwargs.leg1["irs_series"],
            payment_lag=self.kwargs.leg2["payment_lag"],
        )

        # sanitise
        self.kwargs.meta["metric"] = _get_ir_option_metric(self.kwargs.meta["metric"])

        self._leg1 = CustomLeg(
            [
                IRSCallPeriod(  # type: ignore[abstract]
                    expiry=self.kwargs.leg1["expiry"],
                    tenor=self.kwargs.leg1["tenor"],
                    irs_series=self.kwargs.leg1["irs_series"],
                    strike=NoInput(0)
                    if isinstance(self.kwargs.leg1["strike"], str)
                    else self.kwargs.leg1["strike"],
                    notional=self.kwargs.leg1["notional"],
                    option_fixings=self.kwargs.leg1["option_fixings"],
                    metric=self.kwargs.meta["metric"],
                    settlement_method=self.kwargs.leg1["settlement_method"],
                )
                if call
                else IRSPutPeriod(  # type: ignore[abstract]
                    expiry=self.kwargs.leg1["expiry"],
                    tenor=self.kwargs.leg1["tenor"],
                    irs_series=self.kwargs.leg1["irs_series"],
                    strike=NoInput(0)
                    if isinstance(self.kwargs.leg1["strike"], str)
                    else self.kwargs.leg1["strike"],
                    notional=self.kwargs.leg1["notional"],
                    option_fixings=self.kwargs.leg1["option_fixings"],
                    metric=self.kwargs.meta["metric"],
                    settlement_method=self.kwargs.leg1["settlement_method"],
                )
            ]
        )
        self._leg2 = CustomLeg(
            [
                Cashflow(
                    notional=_drb(0.0, self.kwargs.leg2["premium"]),
                    payment=self.kwargs.leg2["payment"],
                    currency=self._leg1.settlement_params.currency,
                ),
            ]
        )
        self._legs = [self._leg1, self._leg2]

    def __repr__(self) -> str:
        return f"<rl.{type(self).__name__} at {hex(id(self))}>"

    def _set_strike_and_vol(
        self,
        rate_curve: CurveOption_,
        disc_curve: _BaseCurve_,
        index_curve: _BaseCurve_,
        vol: _IRVolOption_,
    ) -> None:
        """
        Set the strike, if necessary, and determine pricing metrics from the volatility objects.

        The strike for the *OptionPeriod* is either; string or numeric.

        If it is string, then a numeric strike must be determined with an associated vol.

        If it is numeric then the volatility must be determined for the given strike.

        Pricing elements are captured and cached so they can be used later by subsequent methods.
        """
        if isinstance(vol, _BaseIRSmile):  # TODO _BaseIRCube
            eval_date = vol.meta.eval_date
        else:
            _ = _validate_obj_not_no_input(disc_curve, "disc_curve")
            eval_date = _.nodes.initial

        _pricing = _get_ir_vol_value_and_forward_maybe_from_obj(
            rate_curve=rate_curve,
            index_curve=index_curve,
            strike=self.kwargs.leg1["strike"],
            ir_vol=vol,
            irs=self._irs,
            tenor=self._option.ir_option_params.option_fixing.termination,
            expiry=self._option.ir_option_params.expiry,
            t_e=self._option.ir_option_params.time_to_expiry(eval_date),
        )

        # Review section in book regarding Hyper-parameters and Solver interaction
        self._option.ir_option_params.strike = _pricing.k
        self._pricing = _pricing
        # self._option_periods[0].strike = _dual_float(self._pricing.k)

    def _set_premium(
        self,
        rate_curve: CurveOption_,
        disc_curve: _BaseCurve_,
        index_curve: _BaseCurve_,
        pricing: _IRVolPricingParams,
    ) -> None:
        """
        Set an unspecified premium on the Option to be equal to the mid-market premium.
        """
        if isinstance(self.kwargs.leg2["premium"], NoInput):
            # then set the CashFlow to mid-market
            npv: DualTypes = self._option.npv(  # type: ignore[assignment]
                rate_curve=rate_curve,
                disc_curve=disc_curve,
                index_curve=index_curve,
                ir_vol=pricing,
                local=False,
                forward=self.kwargs.leg2["payment"],
            )
            self._premium.settlement_params._notional = _dual_float(npv)

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
        metric: IROptionMetric | str_ = NoInput(0),
    ) -> DualTypes:
        c = _parse_curves(self, curves, solver)
        rate_curve = _get_curve("rate_curve", True, False, *c)
        disc_curve = _get_curve("disc_curve", False, False, *c)
        index_curve = _get_curve("index_curve", False, False, *c)

        _vol = self._parse_vol(vol)
        del vol

        ir_vol = _maybe_get_ir_vol_maybe_from_solver(
            vol=_vol, vol_meta=self.kwargs.meta["vol"], solver=solver
        )
        self._set_strike_and_vol(
            rate_curve=rate_curve, disc_curve=disc_curve, index_curve=index_curve, vol=ir_vol
        )

        # Premium is not required for rate and also sets as float
        # Review section: "Hyper-parameters and Solver interaction" before enabling.
        # self._set_premium(curves, fx)

        metric_ = _get_ir_option_metric(_drb(self.kwargs.meta["metric"], metric))
        del metric

        value = self._option.rate(
            rate_curve=rate_curve,
            disc_curve=disc_curve,
            index_curve=index_curve,
            ir_vol=self._pricing,
            metric=metric_,
        )
        if (
            metric_ in [IROptionMetric.Premium(), IROptionMetric.PercentNotional()]
            and self.leg2.settlement_params.payment != self.leg1.settlement_params.payment
        ):
            return (
                value
                * disc_curve[self.leg2.settlement_params.payment]
                / disc_curve[self.leg1.settlement_params.payment]
            )
        else:
            return value

    def npv(
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
    ) -> DualTypes | dict[str, DualTypes]:
        c = _parse_curves(self, curves, solver)
        rate_curve = _get_curve("rate_curve", True, True, *c)
        disc_curve = _get_curve("disc_curve", False, True, *c)
        index_curve = _get_curve("index_curve", False, True, *c)

        _vol = self._parse_vol(vol)
        del vol

        ir_vol = _maybe_get_ir_vol_maybe_from_solver(
            vol=_vol, vol_meta=self.kwargs.meta["vol"], solver=solver
        )
        self._set_strike_and_vol(
            rate_curve=rate_curve, disc_curve=disc_curve, index_curve=index_curve, vol=ir_vol
        )

        self._set_premium(
            rate_curve=rate_curve,
            disc_curve=disc_curve,
            index_curve=index_curve,
            pricing=self._pricing,
        )

        if not local:
            base_ = _drb(self.legs[0].settlement_params.currency, base)
        else:
            base_ = base

        opt_npv = self._option.npv(
            rate_curve=rate_curve,  # _validate_obj_not_no_input(rate_curve, "rate curve"),
            disc_curve=disc_curve,
            index_curve=index_curve,
            fx=fx,
            base=base_,
            local=local,
            ir_vol=self._pricing,
            settlement=settlement,
            forward=forward,
        )
        prem_npv = self._premium.npv(
            disc_curve=_get_curve("leg2_disc_curve", False, True, *c),
            fx=fx,
            base=base_,
            local=local,
            settlement=settlement,
            forward=forward,
        )
        if local:
            return {k: opt_npv.get(k, 0) + prem_npv.get(k, 0) for k in set(opt_npv) | set(prem_npv)}  # type:ignore[union-attr, arg-type]
        else:
            return opt_npv + prem_npv  # type: ignore[operator]

    def cashflows(
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
        c = _parse_curves(self, curves, solver)
        rate_curve = _get_curve("rate_curve", True, True, *c)
        disc_curve = _get_curve("disc_curve", False, True, *c)
        index_curve = _get_curve("index_curve", False, True, *c)

        _vol = self._parse_vol(vol)
        del vol

        try:
            ir_vol = _maybe_get_ir_vol_maybe_from_solver(
                vol=_vol, vol_meta=self.kwargs.meta["vol"], solver=solver
            )
            self._set_strike_and_vol(
                rate_curve=rate_curve,
                disc_curve=disc_curve,
                index_curve=index_curve,
                vol=ir_vol,
            )
            self._set_premium(
                rate_curve=rate_curve,
                disc_curve=disc_curve,
                index_curve=index_curve,
                pricing=self._pricing,
            )
        except Exception:  # noqa: S110
            pass  # `cashflows` proceed without pricing determined values

        return self._cashflows_from_legs(
            curves=c[0],
            solver=solver,
            fx=fx,
            base=base,
            settlement=settlement,
            forward=forward,
            vol=_vol,
        )

    def analytic_delta(self, *args: Any, leg: int = 1, **kwargs: Any) -> NoReturn:
        """Not implemented for Option types.
        Use :meth:`~rateslib.instruments._BaseFXOption.analytic_greeks`.
        """
        raise NotImplementedError("For Option types use `analytic_greeks`.")

    def _plot_payoff(
        self,
        window: tuple[float, float] | NoInput = NoInput(0),
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: VolT_ = NoInput(0),
    ) -> tuple[Arr1dF64, Arr1dF64]:
        """
        Mechanics to determine (x,y) coordinates for payoff at expiry plot.
        """
        c = _parse_curves(self, curves, solver)
        rate_curve = _get_curve("rate_curve", True, True, *c)
        disc_curve = _get_curve("disc_curve", False, False, *c)
        index_curve = _get_curve("index_curve", False, False, *c)

        _vol = self._parse_vol(vol)
        del vol

        ir_vol = _maybe_get_ir_vol_maybe_from_solver(
            vol=_vol, vol_meta=self.kwargs.meta["vol"], solver=solver
        )
        self._set_strike_and_vol(
            rate_curve=rate_curve, disc_curve=disc_curve, index_curve=index_curve, vol=ir_vol
        )

        # self._set_premium(curves, fx)
        x, y = self._option._payoff_at_expiry(window)
        return x, y

    def plot_payoff(
        self,
        range: tuple[float, float] | NoInput = NoInput(0),  # noqa: A002
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        base: str_ = NoInput(0),
        vol: float_ = NoInput(0),
    ) -> PlotOutput:
        """
        Return a plot of the payoff at expiry, indexed by the *FXFixing* value.

        Parameters
        ----------
        range: list of float, :green:`optional`
            A range of values for the *FXFixing* value at expiry to use as the x-axis.
        curves: _Curves, :green:`optional`
            Pricing objects. See **Pricing** on each *Instrument* for details of allowed inputs.
        solver: Solver, :green:`optional`
            A :class:`~rateslib.solver.Solver` object containing *Curve*, *Smile*, *Surface*, or
            *Cube* mappings for pricing.
        fx: FXForwards, :green:`optional`
            The :class:`~rateslib.fx.FXForwards` object used for forecasting FX rates, if necessary.
        vol: _Vol, :green:`optional`
            Pricing objects. See **Pricing** on each *Instrument* for details of allowed inputs.

        Returns
        -------
        (Figure, Axes, list[Lines2D])
        """

        x, y = self._plot_payoff(window=range, curves=curves, solver=solver, fx=fx, vol=vol)
        return plot([x], [y])  # type: ignore

    #
    # def local_analytic_rate_fixings(
    #     self,
    #     *,
    #     curves: CurvesT_ = NoInput(0),
    #     solver: Solver_ = NoInput(0),
    #     fx: FXForwards_ = NoInput(0),
    #     vol: VolT_ = NoInput(0),
    #     settlement: datetime_ = NoInput(0),
    #     forward: datetime_ = NoInput(0),
    # ) -> DataFrame:
    #     return DataFrame()
    #
    # def spread(
    #     self,
    #     *,
    #     curves: CurvesT_ = NoInput(0),
    #     solver: Solver_ = NoInput(0),
    #     fx: FXForwards_ = NoInput(0),
    #     vol: VolT_ = NoInput(0),
    #     base: str_ = NoInput(0),
    #     settlement: datetime_ = NoInput(0),
    #     forward: datetime_ = NoInput(0),
    # ) -> DualTypes:
    #     """
    #     Not implemented for Option types. Use :meth:`~rateslib.instruments._BaseFXOption.rate`.
    #     """
    #     raise NotImplementedError(f"`spread` is not implemented for type: {type(self).__name__}")


class IRSCall(_BaseIRSOption):
    """
    An *IR Payer Swaption*.

    .. warning::

       *Swaptions* and *IR Volatility* are in Beta status introduced in v2.7.0

    .. rubric:: Examples

    .. ipython:: python
       :suppress:

       from rateslib import dt, Curve, IRSCall

    .. ipython:: python

       iro = IRSCall(
           expiry=dt(2027, 2, 16),
           tenor="6m",
           strike=3.02,
           notional=100e6,
           irs_series="usd_irs",
           premium=10000.0,
       )
       # iro.cashflows()

    .. rubric:: Pricing

    A *Swaption* requires from one to three *Curves*;

    - a ``rate_curve`` used to forecast the rates on the :class:`~rateslib.legs.FloatLeg` of the
      underlying :class:`~rateslib.instruments.IRS`.
    - a ``disc_curve`` used to discount the value of the *Swaption* and the premium under the
      terms of its bilateral collateral agreement.
    - an ``index_curve`` used as the price alignment index rate for the discounting of the
      underlying :class:`~rateslib.instruments.IRS`. This does not necessarily need to equal the
      ``disc_curve``.

    Allowable inputs are:

    .. code-block:: python

       curves = rate_curve | [rate_curve] #  one curve is used as all curves
       curves = [rate_curve, disc_curve]  #  two curves are applied in the given order, index_curve is set equal to disc_curve
       curves = [rate_curve, disc_curve, index_curve]  # three curves applied in the given order
       curves = {
           "rate_curve": rate_curve,
           "disc_curve": disc_curve
           "index_curve": index_curve
       }  # dict form is explicit

    A *Swaption* also requires an *IRVolatility* object or numeric value for the ``vol`` argument.
    If a numeric value is given it is assumed to be a Black (log-normal) volatility without shift.
    Allowed inputs are:

    .. code-block:: python

       vol = 12.0     # a specific Black (log-normal) calendar-day annualized vol until expiry
       vol = vol_obj  # an explicit volatility object, e.g. IRSabrSmile

    The following pricing ``metric`` are available, with examples:

    .. ipython:: python

       curve = Curve(
           nodes={dt(2026, 2, 16): 1.0, dt(2028, 2, 16): 0.941024343401225}, calendar="nyc"
       )

    - **"BlackVolShift(_)"**:
      The *rate* method will make the necessary conversions between the different volatility
      representations.

      .. ipython:: python

          iro.rate(curves=[curve], vol=25.16, metric="BlackVolShift_0")
          iro.rate(curves=[curve], vol=25.16, metric="BlackVolShift_100")
          iro.rate(curves=[curve], vol=25.16, metric="BlackVolShift_200")
          iro.rate(curves=[curve], vol=25.16, metric="BlackVolShift_300")

    - **"NormalVol"**: the equivalent number of basis point volatility used in the Bachelier
      formula:

      .. ipython:: python

          iro.rate(curves=[curve], vol=25.16, metric="NormalVol")

    - **"Premium"**: the cash premium amount applicable to the 'payment' date, expressed in the
      premium currency.

      .. ipython:: python

          iro.rate(curves=[curve], vol=25.16, metric="Premium")

    - **"PercentNotional"**: the cash premium amount expressed as a percentage of the
      notional.

      .. ipython:: python

          iro.rate(curves=[curve], vol=25.16, metric="PercentNotional")

    .. role:: red

    .. role:: green

    Parameters
    ----------
    .

        .. note::

           The following define **ir option** and generalised **settlement** parameters.

    expiry: datetime, str, :red:`required`
        The expiry of the option. If given in string tenor format, e.g. "1M" requires an
        ``eval_date``. See **Notes**.
    tenor: datetime, str, :red:`required`
        The parameter defining the maturity of the underlying :class:`~rateslib.instruments.IRS`.
    irs_series: IRSSeries, str, :red:`required`
        The standard conventions applied to the underlying :class:`~rateslib.instruments.IRS`.
    strike: float, Variable, str, :red:`required`
        The strike value of the option.
        If str, there are two possibilities; {"atm", "{}bps"}. "atm" will produce a strike equal
        to the mid-market *IRS* rate, whilst "20bps" or "-50bps" will yield a strike that number
        of basis points different to the mid-market rate.
    notional: float, :green:`optional (set by 'defaults')`
        The notional amount expressed in units of ``currency`` fo the ``irs_series``.
    eval_date: datetime, :green:`optional`
        Only required if ``expiry`` is given as string tenor.
        Should be entered as today (also called horizon) and **not** spot.
    payment_lag: int or datetime, :green:`optional (set as IRS effective)`
        The number of business days after expiry to pay premium. If a *datetime* is given this will
        set the premium date explicitly.
    settlement_method: SwaptionSettlementMethod, str, :green:`optional (set by 'default')`
        The method for deriving the settlement cashflow or underlying value.

        .. note::

           The following define additional **rate** parameters.

    premium: float, :green:`optional`
        The amount paid for the option. If not given assumes an unpriced *Option* and sets this as
        mid-market premium during pricing.
    option_fixings: float, Dual, Dual2, Variable, Series, str, :green:`optional`
        The value of the option :class:`~rateslib.data.fixings.IRSFixing`. If a scalar, is used
        directly. If a string identifier, links to the central ``fixings`` object and data loader.

        .. note::

           The following are **meta parameters**.

    metric: IROptionMetric, str, :green:`optional` (set by 'default')`
        The metric used by default in the
        :meth:`~rateslib.instruments._BaseIRSOption.rate` method. See **Pricing**.
    curves : _BaseCurve, str, dict, _Curves, Sequence, :green:`optional`
        Pricing objects passed directly to the *Instrument's* methods' ``curves`` argument. See
        **Pricing**.
    vol: str, Smile, Surface, float, Dual, Dual2, Variable
        Pricing objects passed directly to the *Instrument's* methods' ``vol`` argument. See
        **Pricing**.
    spec : str, optional
        An identifier to pre-populate many field with conventional values. See
        :ref:`here<defaults-doc>` for more info and available values.

    """  # noqa: E501

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, call=True, **kwargs)


class IRSPut(_BaseIRSOption):
    """
    An *IR Receiver Swaption*.

    For parameters and examples see :class:`~rateslib.instruments.IRSCall`.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, call=False, **kwargs)
