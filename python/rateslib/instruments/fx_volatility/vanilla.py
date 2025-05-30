from __future__ import annotations

from abc import ABCMeta
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from pandas import DataFrame

from rateslib import FXDeltaVolSmile, FXDeltaVolSurface, defaults
from rateslib.calendars import _get_fx_expiry_and_delivery, get_calendar
from rateslib.curves import Curve
from rateslib.curves._parsers import _validate_obj_not_no_input
from rateslib.default import NoInput, PlotOutput, _drb, plot
from rateslib.dual.utils import _dual_float
from rateslib.fx_volatility import FXSabrSmile, FXSabrSurface
from rateslib.instruments.base import Metrics
from rateslib.instruments.sensitivities import Sensitivities
from rateslib.instruments.utils import (
    _get_fxvol_curves_fx_and_base_maybe_from_solver,
    _push,
    _update_with_defaults,
)
from rateslib.periods import Cashflow, FXCallPeriod, FXPutPeriod
from rateslib.periods.utils import _validate_fx_as_forwards

if TYPE_CHECKING:
    from typing import NoReturn

    import numpy as np

    from rateslib.typing import (
        FX_,
        NPV,
        Any,
        CalInput,
        Curves_,
        Curves_DiscTuple,
        DualTypes,
        DualTypes_,
        FXVol_,
        FXVolOption_,
        Solver_,
        bool_,
        datetime_,
        float_,
        int_,
        str_,
    )


@dataclass
class _PricingMetrics:
    """None elements are used as flags to indicate an element is not yet set."""

    vol: FXVolOption_ | None
    k: DualTypes | None
    delta_index: DualTypes | None
    spot: datetime
    t_e: DualTypes | None
    f_d: DualTypes


class FXOption(Sensitivities, Metrics, metaclass=ABCMeta):
    """
    Create an *FX Option*.

    Parameters
    ----------
    pair: str
        The currency pair for the FX rate which settles the option, in 3-digit codes, e.g. "eurusd".
    expiry: datetime, str
        The expiry of the option. If given in string tenor format, e.g. "1M" requires an
        ``eval_date``. See notes.
    notional: float, optional (defaults.notional)
        The amount in ccy1 (left side of `pair`) on which the option is based.
    strike: float, Dual, Dual2, str in {"atm_forward", "atm_spot", "atm_delta", "[float]d"}
        The strike value of the option.
        If str, there are four possibilities as above. If giving a specific delta should end
        with a 'd' for delta e.g. "-25d". Put deltas should be input including negative sign.
    eval_date: datetime, optional
        Only required if ``expiry`` is given as string tenor.
        Should be entered as today (also called horizon) and **not** spot. Spot is derived
        from ``delivery_lag`` and ``calendar``.
    modifier : str, optional (defaults.modifier)
        The modification rule, in {"F", "MF", "P", "MP"} for date evaluation.
    eom: bool, optional (defaults.eom_fx)
        Whether to use end-of-month rolls when expiry is given as a month or year tenor.
    calendar : calendar or str, optional
        The holiday calendar object to use. If str, looks up named calendar from
        static data.
    delivery_lag: int, optional (defaults.fx_delivery_lag)
        The number of business days after expiry that the physical settlement of the FX
        exchange occurs.
    payment_lag: int or datetime, optional (defaults.payment_lag)
        The number of business days after expiry to pay premium. If a *datetime* is given this will
        set the premium date explicitly.
    premium: float, optional
        The amount paid for the option. If not given assumes an unpriced *Option* and sets this as
        mid-market premium during pricing.
    premium_ccy: str, optional (RHS)
        The currency in which the premium is paid. Can *only* be one of the two currencies
        in `pair`.
    option_fixing: float
        The value determined at expiry to set the moneyness of the option.
    delta_type: str in {"spot", "forward"}, optional (defaults.fx_delta_type)
        When deriving strike from delta use the equation associated with spot or forward delta.
        If premium currency is ccy1 (left side of `pair`) then this will produce
        **premium adjusted**
        delta values. If the `premium_ccy` is ccy2 (right side of `pair`) then delta values are
        **unadjusted**.
    metric: str in {"pips_or_%", "vol", "premium"}, optional ("pips_or_%")
        The pricing metric returned by the ``rate`` method.
    curves : Curve, LineCurve, str or list of such, optional
        For *FXOptions* curves should be expressed as a list with the discount curves
        entered either as *Curve* or str for discounting cashflows in the appropriate currency
        with a consistent collateral on each side. E.g. *[None, "eurusd", None, "usdusd"]*.
        Forecasting curves are not relevant.
    vol: str, Smile, Surface, float, Dual, Dual2, Variable
        An attached pricing metric used to value the *FXOption* in the absence of volatility
        values supplied at price-time.
    spec : str, optional
        An identifier to pre-populate many field with conventional values. See
        :ref:`here<defaults-doc>` for more info and available values.

    Notes
    ------

    Date calculations for *FXOption* products are very specific. See *'Expiry and Delivery Rules'*
    in *FX Option Pricing* by I. Clark. *Rateslib* uses calendars with associated settlement
    calendars and the recognised market convention rules to derive dates.

    .. ipython:: python
       :suppress:

       from rateslib import dt, FXCall

    .. ipython:: python

       fxc = FXCall(
           pair="eursek",
           expiry="2M",
           eval_date=dt(2024, 6, 19),  # <- Wednesday
           strike=11.0,
           modifier="mf",
           calendar="tgt,stk|fed",
           delivery_lag=2,
           payment_lag=2,
       )
       fxc.kwargs["delivery"]  # <- '2M' out of spot: Monday 24 Jun 2024
       fxc.kwargs["expiry"]    # <- '2b' before 'delivery'
       fxc.kwargs["payment"]  # <- '2b' after 'expiry'

    """

    style: str = "european"
    _rate_scalar: float = 1.0
    _pricing: _PricingMetrics

    _option_periods: tuple[FXPutPeriod | FXCallPeriod]
    _cashflow_periods: tuple[Cashflow]
    curves: Curves_

    def __init__(
        self,
        pair: str,
        expiry: datetime | str,
        notional: DualTypes_ = NoInput(0),
        eval_date: datetime | NoInput = NoInput(0),
        calendar: CalInput = NoInput(0),
        modifier: str_ = NoInput(0),
        eom: bool_ = NoInput(0),
        delivery_lag: int_ = NoInput(0),
        strike: DualTypes | str_ = NoInput(0),
        premium: DualTypes_ = NoInput(0),
        premium_ccy: str_ = NoInput(0),
        payment_lag: str | datetime_ = NoInput(0),
        option_fixing: DualTypes_ = NoInput(0),
        delta_type: str_ = NoInput(0),
        metric: str_ = NoInput(0),
        curves: Curves_ = NoInput(0),
        vol: FXVol_ = NoInput(0),
        spec: str_ = NoInput(0),
    ):
        self.kwargs: dict[str, Any] = dict(
            pair=pair,
            expiry=expiry,
            notional=notional,
            strike=strike,
            premium=premium,
            premium_ccy=premium_ccy,
            option_fixing=option_fixing,
            payment_lag=payment_lag,
            delivery_lag=delivery_lag,
            calendar=calendar,
            eom=eom,
            modifier=modifier,
            delta_type=delta_type,
            metric=metric,
        )
        self.kwargs = _push(spec, self.kwargs)

        self.kwargs = _update_with_defaults(
            self.kwargs,
            {
                "delta_type": defaults.fx_delta_type,
                "notional": defaults.notional,
                "modifier": defaults.modifier,
                "metric": "pips_or_%",
                "delivery_lag": defaults.fx_delivery_lag,
                "payment_lag": defaults.payment_lag,
                "premium_ccy": self.kwargs["pair"][3:],
                "eom": defaults.eom_fx,
            },
        )

        self.kwargs["expiry"], self.kwargs["delivery"] = _get_fx_expiry_and_delivery(
            eval_date,
            self.kwargs["expiry"],
            self.kwargs["delivery_lag"],
            self.kwargs["calendar"],
            self.kwargs["modifier"],
            self.kwargs["eom"],
        )

        if isinstance(self.kwargs["payment_lag"], datetime):
            self.kwargs["payment"] = self.kwargs["payment_lag"]
        else:
            self.kwargs["payment"] = get_calendar(self.kwargs["calendar"]).lag(
                self.kwargs["expiry"],
                self.kwargs["payment_lag"],
                True,
            )

        if self.kwargs["premium_ccy"] not in [
            self.kwargs["pair"][:3],
            self.kwargs["pair"][3:],
        ]:
            raise ValueError(
                f"`premium_ccy`: '{self.kwargs['premium_ccy']}' must be one of option "
                f"currency pair: '{self.kwargs['pair']}'.",
            )
        elif self.kwargs["premium_ccy"] == self.kwargs["pair"][3:]:
            self.kwargs["metric_period"] = "pips"
            self.kwargs["delta_adjustment"] = ""
        else:
            self.kwargs["metric_period"] = "percent"
            self.kwargs["delta_adjustment"] = "_pa"

        # nothing to inherit or negate.
        # self.kwargs = _inherit_or_negate(self.kwargs)  # inherit or negate the complete arg list

        self._validate_strike_and_premiums()

        self.vol = vol
        self.curves = curves
        self.spec = spec

        self._cashflow_periods = (
            Cashflow(
                notional=self.kwargs["premium"],
                payment=self.kwargs["payment"],
                currency=self.kwargs["premium_ccy"],
                stub_type="Premium",
            ),
        )

    @property
    def periods(self) -> tuple[FXCallPeriod | FXPutPeriod, Cashflow]:
        return (self._option_periods[0], self._cashflow_periods[0])

    def __repr__(self) -> str:
        return f"<rl.{type(self).__name__} at {hex(id(self))}>"

    def _validate_strike_and_premiums(self) -> None:
        if isinstance(self.kwargs["strike"], NoInput):
            raise ValueError("`strike` for FXOption must be set to numeric or string value.")
        if isinstance(self.kwargs["strike"], str) and not isinstance(
            self.kwargs["premium"], NoInput
        ):
            raise ValueError(
                "FXOption with string delta as `strike` cannot be initialised with a known "
                "`premium`.\n"
                "Either set `strike` as a defined numeric value, or remove the `premium`.",
            )

    def _set_strike_and_vol(
        self,
        curves: Curves_DiscTuple,
        fx: FX_,
        vol: FXVolOption_,
    ) -> None:
        """
        Set the strike, if necessary, and determine pricing metrics from the volatility objects.

        The strike for the *OptionPeriod* is either; string or numeric.

        If it is string, then a numeric strike must be determined with an associated vol.

        If it is numeric then the volatility must be determined for the given strike.

        Pricing elements are captured and cached so they can be used later by subsequent methods.
        """
        fx_ = _validate_fx_as_forwards(fx)
        # vol_: FXVolOption = _validate_obj_not_no_input(vol, "vol")  # type: ignore[assignment]
        vol_ = vol
        curves_3: Curve = _validate_obj_not_no_input(curves[3], "curves[3]")
        curves_1: Curve = _validate_obj_not_no_input(curves[1], "curves[1]")

        self._pricing = _PricingMetrics(
            vol=None,
            k=None,
            delta_index=None,
            spot=fx_.pairs_settlement[self.kwargs["pair"]],
            t_e=None,
            f_d=fx_.rate(self.kwargs["pair"], self.kwargs["delivery"]),
        )

        if isinstance(vol_, FXDeltaVolSmile | FXDeltaVolSurface | FXSabrSmile | FXSabrSurface):
            eval_date = vol_.meta.eval_date
        else:
            eval_date = curves_3.nodes.initial
            self._pricing.vol = vol_  # Not a vol model so set directly
        self._pricing.t_e = self._option_periods[0]._t_to_expiry(eval_date)

        w_deli = curves_1[self.kwargs["delivery"]]
        w_spot = curves_1[self._pricing.spot]

        # determine if _PricingMetrics.k can be set directly to a numeric value
        if isinstance(self.kwargs["strike"], str) and self.kwargs["strike"].lower() in [
            "atm_forward",
            "atm_spot",
        ]:
            # then strike can be set directly as a calculated value
            method: str = self.kwargs["strike"].lower()
            if method == "atm_forward":
                self._pricing.k = fx_.rate(self.kwargs["pair"], self.kwargs["delivery"])
            elif method == "atm_spot":
                self._pricing.k = fx_.rate(self.kwargs["pair"], self._pricing.spot)
        elif not isinstance(self.kwargs["strike"], str):
            self._pricing.k = self.kwargs["strike"]

        if self._pricing.k is not None:
            if self._pricing.vol is None:
                # vol is only None if vol_ is a VolObj so can be safely type ignored
                # then an explicit strike is set so determine the vol from strike, set and return.
                self._pricing.delta_index, self._pricing.vol, _ = vol_.get_from_strike(  # type: ignore[union-attr]
                    k=self._pricing.k,
                    f=self._pricing.f_d if not isinstance(vol_, FXSabrSurface) else fx_,  # type: ignore[arg-type]
                    expiry=self.kwargs["expiry"],
                    w_deli=w_deli,
                    w_spot=w_spot,
                )
        else:
            # will determine the strike from % delta or ATM-delta string
            method = self.kwargs["strike"].lower()
            if method == "atm_delta":
                self._pricing.delta_index, self._pricing.vol, self._pricing.k = (
                    self._option_periods[0]._index_vol_and_strike_from_atm(
                        delta_type=self._option_periods[0].delta_type,
                        vol=_validate_obj_not_no_input(vol_, "vol"),  # type: ignore[arg-type]
                        w_deli=w_deli,
                        w_spot=w_spot,
                        f=fx_ if isinstance(vol_, FXSabrSurface) else self._pricing.f_d,
                        t_e=self._pricing.t_e,
                    )
                )

            elif method[-1] == "d":  # representing delta
                # then strike is commanded by delta
                (
                    self._pricing.delta_index,
                    self._pricing.vol,
                    self._pricing.k,
                ) = self._option_periods[0]._index_vol_and_strike_from_delta(
                    delta=float(self.kwargs["strike"][:-1]) / 100.0,
                    delta_type=self.kwargs["delta_type"] + self.kwargs["delta_adjustment"],
                    vol=_validate_obj_not_no_input(vol_, "vol"),  # type: ignore[arg-type]
                    w_deli=w_deli,
                    w_spot=w_spot,
                    f=fx_ if isinstance(vol_, FXSabrSurface) else self._pricing.f_d,
                    t_e=self._pricing.t_e,
                )

        # _PricingMetrics.k is completely specified
        assert self._pricing.k is not None  # noqa: S101
        # Review section in book regarding Hyper-parameters and Solver interaction
        self.periods[0].strike = self._pricing.k
        # self._option_periods[0].strike = _dual_float(self._pricing.k)

    def _set_premium(self, curves: Curves_DiscTuple, fx: FX_ = NoInput(0)) -> None:
        if isinstance(self.kwargs["premium"], NoInput):
            # then set the CashFlow to mid-market
            curves_3: Curve = _validate_obj_not_no_input(curves[3], "curves[3]")
            try:
                npv: DualTypes = self._option_periods[0].npv(  # type: ignore[assignment]
                    _validate_obj_not_no_input(curves[1], "curves[1]"),
                    curves_3,
                    fx,
                    vol=self._pricing.vol,  # type: ignore[arg-type]
                    local=False,
                )
            except AttributeError:
                raise ValueError(
                    "`premium` has not been configured for the specified FXOption.\nThis is "
                    "normally determined at mid-market from the given `curves` and `vol` but "
                    "in this case these values do not provide a valid calculation. "
                    "If not required, initialise the "
                    "FXOption with a `premium` of 0.0, and this will be avoided.",
                )
            m_p = self.kwargs["payment"]
            if self.kwargs["premium_ccy"] == self.kwargs["pair"][:3]:
                fx_ = _validate_fx_as_forwards(fx)
                premium = npv / (curves_3[m_p] * fx_.rate("eurusd", m_p))
            else:
                premium = npv / curves_3[m_p]

            self._cashflow_periods[0].notional = _dual_float(premium)

    def rate(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
        vol: FXVol_ = NoInput(0),
        metric: str_ = NoInput(0),
    ) -> DualTypes:
        """
        Return various pricing metrics of the *FX Option*.

        Parameters
        ----------
        curves : list of Curve
            Curves for discounting cashflows. List follows the structure used by IRDs and
            should be given as:
            `[None, Curve for domestic ccy, None, Curve for foreign ccy]`
        solver : Solver, optional
            The numerical :class:`Solver` that constructs *Curves*, *Smiles* or *Surfaces* from
            calibrating instruments.
        fx: FXForwards
            The object to project the relevant forward and spot FX rates.
        base: str, optional
            3-digit currency to express values in (not used by the `rate` method).
        vol: float, Dual, Dual2, FXDeltaVolSmile or FXDeltaVolSurface
            The volatility used in calculation.
        metric: str in {"pips_or_%", "vol", "premium"}, optional
            The pricing metric type to return. See notes.

        Returns
        -------
        float, Dual, Dual2

        Notes
        -----
        The available choices for the pricing ``metric`` that can be used are:

        - *"pips_or_%"*: if the ``premium_ccy`` is the foreign (RHS) currency then *pips* will
          be returned, else
          if the premium is the domestic (LHS) currency then % of notional will be returned.

        - *"vol"*: the volatility used to price the option at that strike / delta is returned.

        - *"premium"*: the monetary amount in ``premium_ccy`` payable at the payment date is
          returned.

        If calculating the *rate* of an *FXOptionStrat* then the attributes ``rate_weight``
        and ``rate_weight_vol``
        will be used to combine the output for each individual *FXOption* within the strategy.

        *FXStrangle* and *FXBrokerFly* have the additional ``metric`` *'single_vol'* which is a
        more complex and
        integrated calculation.
        """
        curves_, fx_, base_, vol_ = _get_fxvol_curves_fx_and_base_maybe_from_solver(
            self.curves,
            self.vol,
            solver,
            curves,
            fx,
            base,
            vol,
            self.kwargs["pair"][3:],
        )
        self._set_strike_and_vol(curves_, fx_, vol_)

        # Premium is not required for rate and also sets as float
        # Review section: "Hyper-parameters and Solver interaction" before enabling.
        # self._set_premium(curves, fx)

        metric = _drb(self.kwargs["metric"], metric)
        if metric in ["vol", "single_vol"]:
            return _validate_obj_not_no_input(self._pricing.vol, "vol")  # type: ignore[return-value]

        _: DualTypes = self._option_periods[0].rate(
            disc_curve=_validate_obj_not_no_input(curves_[1], "curve"),
            disc_curve_ccy2=_validate_obj_not_no_input(curves_[3], "curve"),
            fx=fx_,
            base=NoInput(0),
            vol=self._pricing.vol,  # type: ignore[arg-type]
        )
        if metric == "premium":
            if self.periods[0].metric == "pips":
                _ *= self._option_periods[0].notional / 10000
            else:  # == "percent"
                _ *= self._option_periods[0].notional / 100
        return _

    def npv(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
        local: bool = False,
        vol: FXVol_ = NoInput(0),
    ) -> NPV:
        """
        Return the NPV of the *Option*.

        Parameters
        ----------
        curves : list of Curve
            Curves for discounting cashflows. List follows the structure used by IRDs and
            should be given as:
            `[None, Curve for domestic ccy, None, Curve for foreign ccy]`
        solver : Solver, optional
            The numerical :class:`Solver` that constructs *Curves*, *Smiles* or *Surfaces*
            from calibrating instruments.
        fx: FXForwards
            The object to project the relevant forward and spot FX rates.
        base : str, optional
            The base currency to convert cashflows into (3-digit code).
            If not given defaults to ``fx.base``.
        local : bool, optional
            If `True` will return a dict identifying NPV by local currencies on each
            period.
        vol: float, Dual, Dual2, FXDeltaVolSmile or FXDeltaVolSurface
            The volatility used in calculation.

        Returns
        -------
        float, Dual, Dual2 or dict of such.
        """
        curves_, fx_, base_, vol_ = _get_fxvol_curves_fx_and_base_maybe_from_solver(
            curves_attr=self.curves,
            vol_attr=self.vol,
            solver=solver,
            curves=curves,
            fx=fx,
            base=base,
            vol=vol,
            local_ccy=self.kwargs["pair"][3:],
        )
        self._set_strike_and_vol(curves_, fx_, vol_)
        self._set_premium(curves_, fx_)

        opt_npv = self._option_periods[0].npv(
            disc_curve=_validate_obj_not_no_input(curves_[1], "curve_[1]"),
            disc_curve_ccy2=_validate_obj_not_no_input(curves_[3], "curve_[3]"),
            fx=fx_,
            base=base_,
            local=local,
            vol=self._pricing.vol,  # type: ignore[arg-type]
        )
        if self.kwargs["premium_ccy"] == self.kwargs["pair"][:3]:
            disc_curve = curves_[1]
        else:
            disc_curve = curves_[3]
        prem_npv = self._cashflow_periods[0].npv(NoInput(0), disc_curve, fx, base, local)
        if local:
            return {k: opt_npv.get(k, 0) + prem_npv.get(k, 0) for k in set(opt_npv) | set(prem_npv)}  # type:ignore[union-attr, arg-type]
        else:
            return opt_npv + prem_npv  # type: ignore[operator]

    def cashflows(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
        vol: FXVol_ = NoInput(0),
    ) -> DataFrame:
        """
        Return the properties of all periods used in calculating cashflows.

        Parameters
        ----------
        curves : list of Curve
            Curves for discounting cashflows. List follows the structure used by IRDs and
            should be given as:
            `[None, Curve for domestic ccy, None, Curve for foreign ccy]`
        solver : Solver, optional
            The numerical :class:`Solver` that constructs ``Curves`` from calibrating
            instruments.
        fx: FXForwards
            The object to project the relevant forward and spot FX rates.
        base: str, optional
            Not used by `rate`.
        vol: float, Dual, Dual2, FXDeltaVolSmile or FXDeltaVolSurface
            The volatility used in calculation.

        Returns
        -------
        DataFrame

        """
        curves_, fx_, base_, vol_ = _get_fxvol_curves_fx_and_base_maybe_from_solver(
            curves_attr=self.curves,
            vol_attr=self.vol,
            solver=solver,
            curves=curves,
            fx=fx,
            base=base,
            vol=vol,
            local_ccy=self.kwargs["pair"][3:],
        )
        self._set_strike_and_vol(curves_, fx_, vol_)
        self._set_premium(curves_, fx_)

        seq = [
            self._option_periods[0].cashflows(
                disc_curve=_validate_obj_not_no_input(curves_[1], "curves_[1]"),
                disc_curve_ccy2=_validate_obj_not_no_input(curves_[3], "curves_[3]"),
                fx=fx_,
                base=base_,
                vol=vol_,
            ),
            self._cashflow_periods[0].cashflows(curves_[1], curves_[3], fx_, base_),
        ]
        return DataFrame.from_records(seq)

    def analytic_greeks(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
        vol: FXVol_ = NoInput(0),
    ) -> dict[str, Any]:
        """
        Return various pricing metrics of the *FX Option*.

        Parameters
        ----------
        curves : list of Curve
            Curves for discounting cashflows. List follows the structure used by IRDs and
            should be given as:
            `[None, Curve for domestic ccy, None, Curve for foreign ccy]`
        solver : Solver, optional
            The numerical :class:`Solver` that constructs ``Curves`` from calibrating
            instruments.
        fx: FXForwards
            The object to project the relevant forward and spot FX rates.
        base: str, optional
            Not used by `analytic_greeks`.
        vol: float, Dual, Dual2, FXDeltaVolSmile or FXDeltaVolSurface
            The volatility used in calculation.

        Returns
        -------
        float, Dual, Dual2
        """
        curves_, fx_, base_, vol_ = _get_fxvol_curves_fx_and_base_maybe_from_solver(
            curves_attr=self.curves,
            vol_attr=self.vol,
            solver=solver,
            curves=curves,
            fx=fx,
            base=base,
            vol=vol,
            local_ccy=self.kwargs["pair"][3:],
        )
        self._set_strike_and_vol(curves_, fx_, vol_)
        # self._set_premium(curves, fx)

        return self._option_periods[0]._analytic_greeks(
            disc_curve=_validate_obj_not_no_input(curves_[1], "curves_[1]"),
            disc_curve_ccy2=_validate_obj_not_no_input(curves_[3], "curves_[3]"),
            fx=_validate_fx_as_forwards(fx_),
            base=base_,
            vol=vol_,
            premium=NoInput(0),
            _reduced=False,
        )

    def _analytic_greeks_reduced(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
        vol: FXVol_ = NoInput(0),
    ) -> dict[str, Any]:
        """
        Return various pricing metrics of the *FX Option*.
        """
        curves_, fx_, base_, vol_ = _get_fxvol_curves_fx_and_base_maybe_from_solver(
            curves_attr=self.curves,
            vol_attr=self.vol,
            solver=solver,
            curves=curves,
            fx=fx,
            base=base,
            vol=vol,
            local_ccy=self.kwargs["pair"][3:],
        )
        self._set_strike_and_vol(curves_, fx_, vol_)
        # self._set_premium(curves, fx)

        return self._option_periods[0]._analytic_greeks(
            disc_curve=_validate_obj_not_no_input(curves_[1], "curves_[1]"),
            disc_curve_ccy2=_validate_obj_not_no_input(curves_[3], "curves_[3]"),
            fx=_validate_fx_as_forwards(fx_),
            base=base_,
            vol=self._pricing.vol,  # type: ignore[arg-type]
            premium=NoInput(0),
            _reduced=True,
        )  # none of the reduced greeks need a VolObj - faster to reuse from _pricing.vol

    def analytic_delta(self, *args: Any, leg: int = 1, **kwargs: Any) -> NoReturn:
        """Not implemented for Option types.
        Use :class:`~rateslib.instruments.FXOption.analytic_greeks`.
        """
        raise NotImplementedError("For Option types use `analytic_greeks`.")

    def _plot_payoff(
        self,
        window: list[float] | NoInput = NoInput(0),
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
        local: bool = False,
        vol: FXVol_ = NoInput(0),
    ) -> tuple[
        np.ndarray[tuple[int], np.dtype[np.float64]], np.ndarray[tuple[int], np.dtype[np.float64]]
    ]:
        """
        Mechanics to determine (x,y) coordinates for payoff at expiry plot.
        """
        curves_, fx_, base_, vol_ = _get_fxvol_curves_fx_and_base_maybe_from_solver(
            curves_attr=self.curves,
            vol_attr=self.vol,
            solver=solver,
            curves=curves,
            fx=fx,
            base=base,
            vol=vol,
            local_ccy=self.kwargs["pair"][3:],
        )
        self._set_strike_and_vol(curves_, fx_, vol_)
        # self._set_premium(curves, fx)

        x, y = self._option_periods[0]._payoff_at_expiry(window)
        return x, y

    def plot_payoff(
        self,
        range: list[float] | NoInput = NoInput(0),  # noqa: A002
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
        local: bool = False,
        vol: float_ = NoInput(0),
    ) -> PlotOutput:
        x, y = self._plot_payoff(range, curves, solver, fx, base, local, vol)
        return plot([x], [y])  # type: ignore


class FXCall(FXOption):
    """
    Create an *FX Call* option.

    For parameters see :class:`~rateslib.instruments.FXOption`.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._option_periods = (
            FXCallPeriod(
                pair=self.kwargs["pair"],
                expiry=self.kwargs["expiry"],
                delivery=self.kwargs["delivery"],
                payment=self.kwargs["payment"],
                strike=(
                    NoInput(0) if isinstance(self.kwargs["strike"], str) else self.kwargs["strike"]
                ),
                notional=self.kwargs["notional"],
                option_fixing=self.kwargs["option_fixing"],
                delta_type=self.kwargs["delta_type"] + self.kwargs["delta_adjustment"],
                metric=self.kwargs["metric_period"],
            ),
        )


class FXPut(FXOption):
    """
    Create an *FX Put* option.

    For parameters see :class:`~rateslib.instruments.FXOption`.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._option_periods = (
            FXPutPeriod(
                pair=self.kwargs["pair"],
                expiry=self.kwargs["expiry"],
                delivery=self.kwargs["delivery"],
                payment=self.kwargs["payment"],
                strike=(
                    NoInput(0) if isinstance(self.kwargs["strike"], str) else self.kwargs["strike"]
                ),
                notional=self.kwargs["notional"],
                option_fixing=self.kwargs["option_fixing"],
                delta_type=self.kwargs["delta_type"] + self.kwargs["delta_adjustment"],
                metric=self.kwargs["metric_period"],
            ),
        )
