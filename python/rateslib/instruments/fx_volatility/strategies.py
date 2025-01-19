from __future__ import annotations

from typing import TYPE_CHECKING

from pandas import DataFrame

from rateslib.default import NoInput, _drb
from rateslib.dual import dual_log
from rateslib.fx_volatility import FXDeltaVolSurface, FXVolObj
from rateslib.instruments.fx_volatility.vanilla import FXCall, FXOption, FXPut
from rateslib.instruments.utils import (
    _get_curves_fx_and_base_maybe_from_solver,
    _get_fxvol_maybe_from_solver,
)
from rateslib.splines import evaluate

if TYPE_CHECKING:
    from rateslib.typing import (
        FX_,
        NPV,
        Any,
        Curves_,
        DualTypes,
        Solver_,
        str_,
    )


class FXOptionStrat:
    """
    Create a custom option strategy composed of a list of :class:`~rateslib.instruments.FXOption`.

    Parameters
    ----------
    options: list
        The *FXOptions* which make up the strategy.
    rate_weight: list
        The multiplier for the *'pips_or_%'* metric that sums the options to a final *rate*.
    rate_weight_vol: list
        The multiplier for the *'vol'* metric that sums the options to a final *rate*.
    """

    _greeks: dict[str, Any] = {}
    _strat_elements: tuple[FXOption | FXOptionStrat, ...]
    periods: list[FXOption]

    def __init__(
        self,
        options: list[FXOption | FXOptionStrat],
        rate_weight: list[float],
        rate_weight_vol: list[float],
    ):
        self._strat_elements = tuple(options)
        self.rate_weight = rate_weight
        self.rate_weight_vol = rate_weight_vol
        if len(self.periods) != len(self.rate_weight) or len(self.periods) != len(
            self.rate_weight_vol,
        ):
            raise ValueError(
                "`rate_weight` and `rate_weight_vol` must have same length as `options`.",
            )

    @property
    def periods(self) -> list[FXOption | FXOptionStrat]:
        return list(self._strat_elements)

    def __repr__(self) -> str:
        return f"<rl.{type(self).__name__} at {hex(id(self))}>"

    def _vol_as_list(self, vol, solver):
        """Standardise a vol input over the list of periods"""
        if not isinstance(vol, list | tuple):
            vol = [vol] * len(self.periods)
        return [_get_fxvol_maybe_from_solver(self.vol, _, solver) for _ in vol]

    def rate(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
        vol: list[float] | float = NoInput(0),
        metric: str_ = NoInput(0),  # "pips_or_%",
    ) -> DualTypes:
        """
        Return the mid-market rate of an option strategy.

        See :meth:`~rateslib.instruments.FXOption.rate`.
        """
        curves, fx, base = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            fx,
            base,
            self.kwargs["pair"][3:],
        )
        vol = self._vol_as_list(vol, solver)

        metric = metric if metric is not NoInput.blank else self.kwargs["metric"]
        map_ = {
            "pips_or_%": self.rate_weight,
            "vol": self.rate_weight_vol,
            "premium": [1.0] * len(self.periods),
            "single_vol": self.rate_weight_vol,
        }
        weights = map_[metric]

        _ = 0.0
        for option, vol_, weight in zip(self.periods, vol, weights, strict=False):
            _ += option.rate(curves, solver, fx, base, vol_, metric) * weight
        return _

    def npv(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
        local: bool = False,
        vol: list[float] | float = NoInput(0),
    ) -> NPV:
        if not isinstance(vol, list):
            vol = [vol] * len(self.periods)

        results = [
            option.npv(curves, solver, fx, base, local, vol_)
            for (option, vol_) in zip(self.periods, vol, strict=False)
        ]

        if local:
            _ = DataFrame(results).fillna(0.0)
            _ = _.sum()
            _ = _.to_dict()
        else:
            _ = sum(results)
        return _

    def _plot_payoff(
        self,
        window: list[float] | NoInput = NoInput(0),
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
        local: bool = False,
        vol: list[float] | float = NoInput(0),
    ):
        if not isinstance(vol, list):
            vol = [vol] * len(self.periods)

        y = None
        for option, vol_ in zip(self.periods, vol, strict=False):
            x, y_ = option._plot_payoff(window, curves, solver, fx, base, local, vol_)
            if y is None:
                y = y_
            else:
                y += y_

        return x, y

    def _set_notionals(self, notional):
        """
        Set the notionals on each option period. Mainly used by Brokerfly for vega neutral
        strangle and straddle.
        """
        for option in self.periods:
            option.periods[0].notional = notional

    def analytic_greeks(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
        local: bool = False,
        vol: float = NoInput(0),
    ):
        """
        Return various pricing metrics of the *FX Option*.

        Parameters
        ----------
        curves : list of Curve
            Curves for discounting cashflows. List follows the structure used by IRDs and should
            be given as:
            `[None, Curve for domestic ccy, None, Curve for foreign ccy]`
        solver : Solver, optional
            The numerical :class:`Solver` that constructs ``Curves`` from calibrating
            instruments.
        fx : float, FXRates, FXForwards, optional
            The immediate settlement FX rate that will be used to convert values
            into another currency. A given `float` is used directly. If giving a
            ``FXRates`` or ``FXForwards`` object, converts from local currency
            into ``base``.
        base : str, optional
            The base currency to convert cashflows into (3-digit code), set by default.
            Only used if ``fx`` is an ``FXRates`` or ``FXForwards`` object.


        Returns
        -------
        float, Dual, Dual2

        Notes
        ------

        """

        # implicitly call set_pricing_mid for unpriced parameters
        # this is important for Strategies whose options are
        # dependent upon each other, e.g. Strangle. (RR and Straddle do not have
        # interdependent options)
        self.rate(curves, solver, fx, base, vol)

        curves, fx, base = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            fx,
            base,
            self.kwargs["pair"][3:],
        )
        vol = self._vol_as_list(vol, solver)

        gks = []
        for option, _vol in zip(self.periods, vol, strict=False):
            # by calling on the OptionPeriod directly the strike is maintained from rate call.
            gks.append(
                option.periods[0].analytic_greeks(
                    curves[1],
                    curves[3],
                    fx,
                    base,
                    local,
                    _vol,
                    option.kwargs["premium"],
                ),
            )

        _unit_attrs = ["delta", "gamma", "vega", "vomma", "vanna", "_kega", "_kappa", "__bs76"]
        _ = {}
        for attr in _unit_attrs:
            _[attr] = sum(gk[attr] * self.rate_weight[i] for i, gk in enumerate(gks))

        _notional_attrs = [
            f"delta_{self.kwargs['pair'][:3]}",
            f"gamma_{self.kwargs['pair'][:3]}_1%",
            f"vega_{self.kwargs['pair'][3:]}",
        ]
        for attr in _notional_attrs:
            _[attr] = sum(gk[attr] * self.rate_weight[i] for i, gk in enumerate(gks))

        _.update(
            {
                "__class": "FXOptionStrat",
                "__options": gks,
                "__delta_type": gks[0]["__delta_type"],
                "__notional": self.kwargs["notional"],
            },
        )
        return _


class FXRiskReversal(FXOptionStrat, FXOption):
    """
    Create an *FX Risk Reversal* option strategy.

    For additional arguments see :class:`~rateslib.instruments.FXOption`.

    Parameters
    ----------
    args: tuple
        Positional arguments to :class:`~rateslib.instruments.FXOption`.
    strike: 2-element sequence
        The first element is applied to the lower strike put and the
        second element applied to the higher strike call, e.g. `["-25d", "25d"]`.
    premium: 2-element sequence, optional
        The premiums associated with each option of the risk reversal.
    metric: str, optional
        The default metric to apply in the method :meth:`~rateslib.instruments.FXOptionStrat.rate`
    kwargs: tuple
        Keyword arguments to :class:`~rateslib.instruments.FXOption`.

    Notes
    -----
    When supplying ``strike`` as a string delta the strike will be determined at price time from
    the provided volatility.

    Buying a *Risk Reversal* equates to selling a lower strike :class:`~rateslib.instruments.FXPut`
    and buying a higher strike :class:`~rateslib.instruments.FXCall`.

    This class is essentially an alias constructor for an
    :class:`~rateslib.instruments.FXOptionStrat` where the number
    of options and their definitions and nominals have been specifically set.
    """

    rate_weight = [-1.0, 1.0]
    rate_weight_vol = [-1.0, 1.0]
    _rate_scalar = 100.0

    def __init__(
        self,
        *args,
        strike=(NoInput(0), NoInput(0)),
        premium=(NoInput(0), NoInput(0)),
        metric: str = "vol",
        **kwargs,
    ):
        super(FXOptionStrat, self).__init__(
            *args,
            strike=list(strike),
            premium=list(premium),
            **kwargs,
        )
        self.kwargs["metric"] = metric
        self._strat_elements = [
            FXPut(
                pair=self.kwargs["pair"],
                expiry=self.kwargs["expiry"],
                delivery_lag=self.kwargs["delivery"],
                payment_lag=self.kwargs["payment"],
                calendar=self.kwargs["calendar"],
                modifier=self.kwargs["modifier"],
                strike=self.kwargs["strike"][0],
                notional=-self.kwargs["notional"],
                option_fixing=self.kwargs["option_fixing"],
                delta_type=self.kwargs["delta_type"],
                premium=self.kwargs["premium"][0],
                premium_ccy=self.kwargs["premium_ccy"],
                curves=self.curves,
                vol=self.vol,
            ),
            FXCall(
                pair=self.kwargs["pair"],
                expiry=self.kwargs["expiry"],
                delivery_lag=self.kwargs["delivery"],
                payment_lag=self.kwargs["payment"],
                calendar=self.kwargs["calendar"],
                modifier=self.kwargs["modifier"],
                strike=self.kwargs["strike"][1],
                notional=self.kwargs["notional"],
                option_fixing=self.kwargs["option_fixing"],
                delta_type=self.kwargs["delta_type"],
                premium=self.kwargs["premium"][1],
                premium_ccy=self.kwargs["premium_ccy"],
                curves=self.curves,
                vol=self.vol,
            ),
        ]

    def _validate_strike_and_premiums(self):
        """called as part of init, specific validation rules for straddle"""
        if any(_ is NoInput.blank for _ in self.kwargs["strike"]):
            raise ValueError(
                "`strike` for FXRiskReversal must be set to list of 2 numeric or string values.",
            )
        for k, p in zip(self.kwargs["strike"], self.kwargs["premium"], strict=False):
            if isinstance(k, str) and p != NoInput.blank:
                raise ValueError(
                    "FXRiskReversal with string delta as `strike` cannot be initialised with a "
                    "known `premium`.\n"
                    "Either set `strike` as a defined numeric value, or remove the `premium`.",
                )


class FXStraddle(FXOptionStrat, FXOption):
    """
    Create an *FX Straddle* option strategy.

    For additional arguments see :class:`~rateslib.instruments.FXOption`.

    Parameters
    ----------
    args: tuple
        Positional arguments to :class:`~rateslib.instruments.FXOption`.
    premium: 2-element sequence, optional
        The premiums associated with each option of the straddle.
    metric: str, optional
        The default metric to apply in the method :meth:`~rateslib.instruments.FXOptionStrat.rate`
    kwargs: tuple
        Keyword arguments to :class:`~rateslib.instruments.FXOption`.

    Notes
    -----
    When supplying ``strike`` as a string delta the strike will be determined at price time from
    the provided volatility and FX forward market.

    Buying a *Straddle* equates to buying an :class:`~rateslib.instruments.FXCall`
    and an :class:`~rateslib.instruments.FXPut` at the same strike.

    This class is essentially an alias constructor for an
    :class:`~rateslib.instruments.FXOptionStrat` where the number
    of options and their definitions and nominals have been specifically set.
    """

    rate_weight = [1.0, 1.0]
    rate_weight_vol = [0.5, 0.5]
    _rate_scalar = 100.0

    def __init__(self, *args, premium=(NoInput(0), NoInput(0)), metric="vol", **kwargs):
        super(FXOptionStrat, self).__init__(*args, premium=list(premium), **kwargs)
        self.kwargs["metric"] = metric
        self._strat_elements = [
            FXPut(
                pair=self.kwargs["pair"],
                expiry=self.kwargs["expiry"],
                delivery_lag=self.kwargs["delivery"],
                payment_lag=self.kwargs["payment"],
                calendar=self.kwargs["calendar"],
                modifier=self.kwargs["modifier"],
                strike=self.kwargs["strike"],
                notional=self.kwargs["notional"],
                option_fixing=self.kwargs["option_fixing"],
                delta_type=self.kwargs["delta_type"],
                premium=self.kwargs["premium"][0],
                premium_ccy=self.kwargs["premium_ccy"],
                curves=self.curves,
                vol=self.vol,
            ),
            FXCall(
                pair=self.kwargs["pair"],
                expiry=self.kwargs["expiry"],
                delivery_lag=self.kwargs["delivery"],
                payment_lag=self.kwargs["payment"],
                calendar=self.kwargs["calendar"],
                modifier=self.kwargs["modifier"],
                strike=self.kwargs["strike"],
                notional=self.kwargs["notional"],
                option_fixing=self.kwargs["option_fixing"],
                delta_type=self.kwargs["delta_type"],
                premium=self.kwargs["premium"][1],
                premium_ccy=self.kwargs["premium_ccy"],
                curves=self.curves,
                vol=self.vol,
            ),
        ]

    def _validate_strike_and_premiums(self):
        """called as part of init, specific validation rules for straddle"""
        if self.kwargs["strike"] is NoInput.blank:
            raise ValueError("`strike` for FXStraddle must be set to numeric or string value.")
        if isinstance(self.kwargs["strike"], str) and self.kwargs["premium"] != [
            NoInput.blank,
            NoInput.blank,
        ]:
            raise ValueError(
                "FXStraddle with string delta as `strike` cannot be initialised with a known "
                "`premium`.\nEither set `strike` as a defined numeric value, or remove "
                "the `premium`.",
            )


class FXStrangle(FXOptionStrat, FXOption):
    """
    Create an *FX Strangle* option strategy.

    For additional arguments see :class:`~rateslib.instruments.FXOption`.

    Parameters
    ----------
    args: tuple
        Positional arguments to :class:`~rateslib.instruments.FXOption`.
    strike: 2-element sequence
        The first element is applied to the lower strike put and the
        second element applied to the higher strike call, e.g. `["-25d", "25d"]`.
    premium: 2-element sequence, optional
        The premiums associated with each option of the strangle.
    metric: str, optional
        The default metric to apply in the method :meth:`~rateslib.instruments.FXOptionStrat.rate`
    kwargs: tuple
        Keyword arguments to :class:`~rateslib.instruments.FXOption`.

    Notes
    -----
    When supplying ``strike`` as a string delta the strike will be determined at price time from
    the provided volatility.

    Buying a *Strangle* equates to buying a lower strike :class:`~rateslib.instruments.FXPut`
    and buying a higher strike :class:`~rateslib.instruments.FXCall`.

    This class is essentially an alias constructor for an
    :class:`~rateslib.instruments.FXOptionStrat` where the number
    of options and their definitions and nominals have been specifically set.

    .. warning::

       The default ``metric`` for an *FXStraddle* is *'single_vol'*, which requires an iterative
       algorithm to solve.
       For defined strikes it is usually very accurate but for strikes defined by delta it
       will return a solution within 0.1 pips. This means it is both slower than other instruments
       and inexact.

    """

    rate_weight = [1.0, 1.0]
    rate_weight_vol = [0.5, 0.5]
    _rate_scalar = 100.0

    def __init__(
        self,
        *args,
        strike=(NoInput(0), NoInput(0)),
        premium=(NoInput(0), NoInput(0)),
        metric="single_vol",
        **kwargs,
    ):
        super(FXOptionStrat, self).__init__(
            *args, strike=list(strike), premium=list(premium), **kwargs
        )
        self.kwargs["metric"] = metric
        self._is_fixed_delta = [
            isinstance(self.kwargs["strike"][0], str)
            and self.kwargs["strike"][0][-1].lower() == "d"
            and self.kwargs["strike"][0] != "atm_forward",
            isinstance(self.kwargs["strike"][1], str)
            and self.kwargs["strike"][1][-1].lower() == "d"
            and self.kwargs["strike"][1] != "atm_forward",
        ]
        self._strat_elements = [
            FXPut(
                pair=self.kwargs["pair"],
                expiry=self.kwargs["expiry"],
                delivery_lag=self.kwargs["delivery"],
                payment_lag=self.kwargs["payment"],
                calendar=self.kwargs["calendar"],
                modifier=self.kwargs["modifier"],
                strike=self.kwargs["strike"][0],
                notional=self.kwargs["notional"],
                option_fixing=self.kwargs["option_fixing"],
                delta_type=self.kwargs["delta_type"],
                premium=self.kwargs["premium"][0],
                premium_ccy=self.kwargs["premium_ccy"],
                curves=self.curves,
                vol=self.vol,
            ),
            FXCall(
                pair=self.kwargs["pair"],
                expiry=self.kwargs["expiry"],
                delivery_lag=self.kwargs["delivery"],
                payment_lag=self.kwargs["payment"],
                calendar=self.kwargs["calendar"],
                modifier=self.kwargs["modifier"],
                strike=self.kwargs["strike"][1],
                notional=self.kwargs["notional"],
                option_fixing=self.kwargs["option_fixing"],
                delta_type=self.kwargs["delta_type"],
                premium=self.kwargs["premium"][1],
                premium_ccy=self.kwargs["premium_ccy"],
                curves=self.curves,
                vol=self.vol,
            ),
        ]

    def _validate_strike_and_premiums(self):
        """called as part of init, specific validation rules for strangle"""
        if any(_ is NoInput.blank for _ in self.kwargs["strike"]):
            raise ValueError(
                "`strike` for FXStrangle must be set to list of 2 numeric or string values.",
            )
        for k, p in zip(self.kwargs["strike"], self.kwargs["premium"], strict=False):
            if isinstance(k, str) and p != NoInput.blank:
                raise ValueError(
                    "FXStrangle with string delta as `strike` cannot be initialised with a "
                    "known `premium`.\n"
                    "Either set `strike` as a defined numeric value, or remove the `premium`.",
                )

    def rate(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
        vol: list[float] | float = NoInput(0),
        metric: str_ = NoInput(0),  # "pips_or_%",
    ):
        """
        Returns the rate of the *FXStraddle* according to a pricing metric.

        Notes
        ------

        .. warning::

           The default ``metric`` for an *FXStraddle* is *'single_vol'*, which requires an
           iterative algorithm to solve.
           For defined strikes it is usually very accurate but for strikes defined by delta it
           will return a solution within 0.01 pips. This means it is both slower than other
           instruments and inexact.

        For parameters see :meth:`~rateslib.instruments.FXOption.rate`.

        The ``metric`` *'vol'* is not sensible to use with an *FXStraddle*, although it will
        return the arithmetic
        average volatility across both options, *'single_vol'* is the more standardised choice.
        """
        return self._rate(curves, solver, fx, base, vol, metric)

    def _rate(self, curves, solver, fx, base, vol, metric, record_greeks=False):
        metric = _drb(self.kwargs["metric"], metric).lower()
        if metric != "single_vol" and not any(self._is_fixed_delta):
            # the strikes are explicitly defined and independent across options.
            # can evaluate separately
            return super().rate(curves, solver, fx, base, vol, metric)
        else:
            # must perform single vol evaluation to determine mkt convention strikes
            single_vol = self._rate_single_vol(curves, solver, fx, base, vol, record_greeks)
            if metric == "single_vol":
                return single_vol
            else:
                # return the premiums using the single_vol as the volatility
                return super().rate(curves, solver, fx, base, vol=single_vol, metric=metric)

    def _rate_single_vol(self, curves, solver, fx, base, vol, record_greeks):
        """
        Solve the single vol rate metric for a strangle using iterative market convergence routine.
        """
        # Get curves and vol
        curves, fx, base = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            fx,
            base,
            self.kwargs["pair"][3:],
        )
        vol = self._vol_as_list(vol, solver)
        vol = [
            _ if not isinstance(_, FXDeltaVolSurface) else _.get_smile(self.kwargs["expiry"])
            for _ in vol
        ]

        spot = fx.pairs_settlement[self.kwargs["pair"]]
        w_spot, w_deli = curves[1][spot], curves[1][self.kwargs["delivery"]]
        f_d, f_t = (
            fx.rate(self.kwargs["pair"], self.kwargs["delivery"]),
            fx.rate(self.kwargs["pair"], spot),
        )
        z_w_0 = 1.0 if "forward" in self.kwargs["delta_type"] else w_deli / w_spot
        f_0 = f_d if "forward" in self.kwargs["delta_type"] else f_t
        eta1 = None
        if isinstance(vol[0], FXVolObj):
            eta1 = -0.5 if "_pa" in vol[0].delta_type else 0.5
            z_w_1 = 1.0 if "forward" in vol[0].delta_type else w_deli / w_spot
            fzw1zw0 = f_0 * z_w_1 / z_w_0

        # first start by evaluating the individual swaptions given their
        # strikes with the smile - delta or fixed
        gks = [
            self.periods[0].analytic_greeks(curves, solver, fx, base, vol=vol[0]),
            self.periods[1].analytic_greeks(curves, solver, fx, base, vol=vol[1]),
        ]

        def d_wrt_sigma1(period_index, greeks, smile_greeks, vol, eta1):
            """
            Obtain derivatives with respect to tgt vol.

            This function was tested by adding AD to the tgt_vol as a variable e.g.:
            tgt_vol = Dual(float(tgt_vol), ["tgt_vol"], [100.0]) # note scaled to 100
            Then the options defined by fixed delta should not have a strike set to float, i.e.
            self.periods[0].strike = float(self._pricing["k"]) ->
            self.periods[0].strike = self._pricing["k"]
            Then evaluate, for example: smile_greeks[i]["_delta_index"] with respect to "tgt_vol".
            That value calculated with AD aligns with the analyical method here.

            To speed up this function AD could be used, but it requires careful management of
            whether the strike above is set to float or is left in AD format which has other
            implications for the calculation of risk sensitivities.
            """
            i, sg, g = period_index, smile_greeks, greeks
            fixed_delta, vol = self._is_fixed_delta[i], vol[i]
            if not fixed_delta:
                return g[i]["vega"], 0.0
            elif not isinstance(vol, FXVolObj):
                return (
                    g[i]["_kappa"] * g[i]["_kega"] + g[i]["vega"],
                    sg[i]["_kappa"] * g[i]["_kega"],
                )
            else:
                dvol_ddeltaidx = evaluate(vol.spline, sg[i]["_delta_index"], 1) * 0.01
                ddeltaidx_dvol1 = sg[i]["gamma"] * fzw1zw0
                if eta1 < 0:  # premium adjusted vol smile
                    ddeltaidx_dvol1 += sg[i]["_delta_index"]
                ddeltaidx_dvol1 *= g[i]["_kega"] / sg[i]["__strike"]

                _ = dual_log(sg[i]["__strike"] / f_d) / sg[i]["__vol"]
                _ += eta1 * sg[i]["__vol"] * sg[i]["__sqrt_t"] ** 2
                _ *= dvol_ddeltaidx * sg[i]["gamma"] * fzw1zw0
                ddeltaidx_dvol1 /= 1 + _

                return (
                    g[i]["_kappa"] * g[i]["_kega"] + g[i]["vega"],
                    sg[i]["_kappa"] * g[i]["_kega"]
                    + sg[i]["vega"] * dvol_ddeltaidx * ddeltaidx_dvol1,
                )

        tgt_vol = (gks[0]["__vol"] * gks[0]["vega"] + gks[1]["__vol"] * gks[1]["vega"]) * 100.0
        tgt_vol /= gks[0]["vega"] + gks[1]["vega"]
        f0, iters = 100e6, 1
        while abs(f0) > 1e-6 and iters < 10:
            # Determine the strikes at the current tgt_vol
            # Also determine the greeks of these options measure with tgt_vol
            gks = [
                self.periods[0].analytic_greeks(curves, solver, fx, base, vol=tgt_vol),
                self.periods[1].analytic_greeks(curves, solver, fx, base, vol=tgt_vol),
            ]
            # Also determine the greeks of these options measured with the market smile vol.
            # (note the strikes have been set by previous call, call OptionPeriods direct
            # to avoid re-determination)
            smile_gks = [
                self.periods[0]
                .periods[0]
                .analytic_greeks(curves[1], curves[3], fx, base, vol=vol[0]),
                self.periods[1]
                .periods[0]
                .analytic_greeks(curves[1], curves[3], fx, base, vol=vol[1]),
            ]

            # The value of the root function is derived from the 4 previous calculated prices
            f0 = (
                smile_gks[0]["__bs76"]
                + smile_gks[1]["__bs76"]
                - gks[0]["__bs76"]
                - gks[1]["__bs76"]
            )

            dc1_dvol1_0, dcmkt_dvol1_0 = d_wrt_sigma1(0, gks, smile_gks, vol, eta1)
            dc1_dvol1_1, dcmkt_dvol1_1 = d_wrt_sigma1(1, gks, smile_gks, vol, eta1)
            f1 = dcmkt_dvol1_0 + dcmkt_dvol1_1 - dc1_dvol1_0 - dc1_dvol1_1

            tgt_vol = tgt_vol - (f0 / f1) * 100.0  # Newton-Raphson step
            iters += 1

        if record_greeks:  # this needs to be explicitly called since it degrades performance
            self._greeks["strangle"] = {
                "single_vol": {
                    "FXPut": self.periods[0].analytic_greeks(curves, solver, fx, base, vol=tgt_vol),
                    "FXCall": self.periods[1].analytic_greeks(
                        curves,
                        solver,
                        fx,
                        base,
                        vol=tgt_vol,
                    ),
                },
                "market_vol": {
                    "FXPut": self.periods[0]
                    .periods[0]
                    .analytic_greeks(curves[1], curves[3], fx, base, vol=vol[0]),
                    "FXCall": self.periods[1]
                    .periods[0]
                    .analytic_greeks(curves[1], curves[3], fx, base, vol=vol[1]),
                },
            }

        return tgt_vol

    # def _single_vol_rate_known_strikes(
    #     self,
    #     imm_prem,
    #     f_d,
    #     t_e,
    #     v_deli,
    #     g0,
    # ):
    #     k1 = self.kwargs["strike"][0]
    #     k2 = self.kwargs["strike"][1]
    #     sqrt_t = t_e ** 0.5
    #
    #     def root(g, imm_prem, k1, k2, f_d, sqrt_t, v_deli):
    #         vol_sqrt_t = g * sqrt_t
    #         d_plus_1 = _d_plus_min(k1, f_d, vol_sqrt_t, 0.5)
    #         d_min_1 = _d_plus_min(k1, f_d, vol_sqrt_t, -0.5)
    #         d_plus_2 = _d_plus_min(k2, f_d, vol_sqrt_t, 0.5)
    #         d_min_2 = _d_plus_min(k2, f_d, vol_sqrt_t, -0.5)
    #         f0 = -(f_d * dual_norm_cdf(-d_plus_1) - k1 * dual_norm_cdf(-d_min_1))
    #         f0 += (f_d * dual_norm_cdf(d_plus_2) - k2 * dual_norm_cdf(d_min_2))
    #         f0 = f0 * v_deli - imm_prem
    #         f1 = v_deli * f_d * sqrt_t * (dual_norm_pdf(-d_plus_1) + dual_norm_pdf(d_plus_2))
    #         return f0, f1
    #
    #     result = newton_1dim(root, g0=g0, args=(imm_prem, k1, k2, f_d, sqrt_t, v_deli))
    #     return result["g"]


class FXBrokerFly(FXOptionStrat, FXOption):
    """
    Create an *FX BrokerFly* option strategy.

    For additional arguments see :class:`~rateslib.instruments.FXOption`.

    Parameters
    ----------
    args: tuple
        Positional arguments to :class:`~rateslib.instruments.FXOption`.
    strike: 3-element sequence
        The first element is applied to the lower strike put, the
        second element to the straddle strike and the third element to the higher strike
        call, e.g. `["-25d", "atm_delta", "25d"]`.
    premium: 4-element sequence, optional
        The premiums associated with each option of the strategy; lower strike put, straddle put,
        straddle call, higher strike call.
    notional: 2-element sequence, optional
        The first element is the notional associated with the *Strangle*. If the second element
        is *None*, it will be implied in a vega neutral sense.
    metric: str, optional
        The default metric to apply in the method :meth:`~rateslib.instruments.FXOptionStrat.rate`
    kwargs: tuple
        Keyword arguments to :class:`~rateslib.instruments.FXOption`.

    Notes
    -----
    When supplying ``strike`` as a string delta the strike will be determined at price time from
    the provided volatility.

    Buying a *BrokerFly* equates to buying an :class:`~rateslib.instruments.FXStrangle` and
    selling a :class:`~rateslib.instruments.FXStraddle`, where the convention is to set the
    notional on the *Straddle* such that the entire strategy is *vega* neutral at inception.

    .. warning::

       The default ``metric`` for an *FXBrokerFly* is *'single_vol'*, which requires an iterative
       algorithm to solve.
       For defined strikes it is usually very accurate but for strikes defined by delta it
       will return a solution within 0.1 pips. This means it is both slower than other instruments
       and inexact.

    """

    rate_weight = [1.0, 1.0]
    rate_weight_vol = [1.0, -1.0]
    _rate_scalar = 100.0

    def __init__(
        self,
        *args,
        strike=(NoInput(0), NoInput(0), NoInput(0)),
        premium=(NoInput(0), NoInput(0), NoInput(0), NoInput(0)),
        notional=(NoInput(0), NoInput(0)),
        metric="single_vol",
        **kwargs,
    ):
        super(FXOptionStrat, self).__init__(
            *args,
            premium=list(premium),
            strike=list(strike),
            notional=list(notional),
            **kwargs,
        )
        self.kwargs["notional"][1] = (
            NoInput(0) if self.kwargs["notional"][1] is None else self.kwargs["notional"][1]
        )
        self.kwargs["metric"] = metric
        self._strat_elements = (
            FXStrangle(
                pair=self.kwargs["pair"],
                expiry=self.kwargs["expiry"],
                delivery_lag=self.kwargs["delivery"],
                payment_lag=self.kwargs["payment"],
                calendar=self.kwargs["calendar"],
                modifier=self.kwargs["modifier"],
                strike=[self.kwargs["strike"][0], self.kwargs["strike"][2]],
                notional=self.kwargs["notional"][0],
                option_fixing=self.kwargs["option_fixing"],
                delta_type=self.kwargs["delta_type"],
                premium=[self.kwargs["premium"][0], self.kwargs["premium"][3]],
                premium_ccy=self.kwargs["premium_ccy"],
                metric=self.kwargs["metric"],
                curves=self.curves,
                vol=self.vol,
            ),
            FXStraddle(
                pair=self.kwargs["pair"],
                expiry=self.kwargs["expiry"],
                delivery_lag=self.kwargs["delivery"],
                payment_lag=self.kwargs["payment"],
                calendar=self.kwargs["calendar"],
                modifier=self.kwargs["modifier"],
                strike=self.kwargs["strike"][1],
                notional=self.kwargs["notional"][1],
                option_fixing=self.kwargs["option_fixing"],
                delta_type=self.kwargs["delta_type"],
                premium=self.kwargs["premium"][1:3],
                premium_ccy=self.kwargs["premium_ccy"],
                metric="vol" if self.kwargs["metric"] == "single_vol" else self.kwargs["metric"],
                curves=self.curves,
                vol=self.vol,
            ),
        )

    def _maybe_set_vega_neutral_notional(self, curves, solver, fx, base, vol, metric):
        if self.kwargs["notional"][1] is NoInput.blank and metric in ["pips_or_%", "premium"]:
            self.periods[0]._rate(
                curves,
                solver,
                fx,
                base,
                vol=vol[0],
                metric="single_vol",
                record_greeks=True,
            )
            self._greeks["straddle"] = self.periods[1].analytic_greeks(
                curves,
                solver,
                fx,
                base,
                vol=vol[1],
            )
            strangle_vega = self._greeks["strangle"]["market_vol"]["FXPut"]["vega"]
            strangle_vega += self._greeks["strangle"]["market_vol"]["FXCall"]["vega"]
            straddle_vega = self._greeks["straddle"]["vega"]
            scalar = strangle_vega / straddle_vega
            self.periods[1].kwargs["notional"] = float(
                self.periods[0].periods[0].periods[0].notional * -scalar,
            )
            self.periods[1]._set_notionals(self.periods[1].kwargs["notional"])
            # BrokerFly -> Strangle -> FXPut -> FXPutPeriod

    def rate(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
        vol: list[float] | float = NoInput(0),
        metric: str_ = NoInput(0),
    ):
        """
        Return the mid-market rate of an option strategy.

        Parameters
        ----------
        curves
        solver
        fx
        base
        vol
        metric

        Returns
        -------
        float, Dual, Dual2

        Notes
        -----

        The different types of ``metric`` return different quotation conventions.

        - *'single_vol'*: the default type for a :class:`~rateslib.instruments.FXStrangle`

        - *'vol'*: sums the mid-market volatilities of each option multiplied by their
          respective ``rate_weight_vol``
          parameter. For example this is the default pricing convention for
          a :class:`~rateslib.instruments.FXRiskReversal` where the price is the vol of the call
          minus the vol of the
          put and the ``rate_weight_vol`` parameters are [-1.0, 1.0].

        - *'pips_or_%'*: sums the mid-market pips or percent price of each option multiplied by
          their respective
          ``rate_weight`` parameter. For example for a :class:`~rateslib.instruments.FXStraddle`
          the total premium
          is the sum of two premiums and the ``rate_weight`` parameters are [1.0, 1.0].
        """
        if not isinstance(vol, list):
            vol = [[vol, vol], vol]
        else:
            vol = [
                [vol[0], vol[2]],
                vol[1],
            ]  # restructure to pass to Strangle and Straddle separately

        temp_metric = _drb(self.kwargs["metric"], metric)
        self._maybe_set_vega_neutral_notional(curves, solver, fx, base, vol, temp_metric.lower())

        if temp_metric == "pips_or_%":
            straddle_scalar = (
                self.periods[1].periods[0].periods[0].notional
                / self.periods[0].periods[0].periods[0].notional
            )
            weights = [1.0, straddle_scalar]
        elif temp_metric == "premium":
            weights = self.rate_weight
        else:
            weights = self.rate_weight_vol
        _ = 0.0
        for option_strat, vol_, weight in zip(self.periods, vol, weights, strict=False):
            _ += option_strat.rate(curves, solver, fx, base, vol_, metric) * weight
        return _

    def analytic_greeks(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
        local: bool = False,
        vol: float = NoInput(0),
    ):
        """ """
        # implicitly call set_pricing_mid for unpriced parameters
        self.rate(curves, solver, fx, base, vol, metric="pips_or_%")
        # curves, fx, base = _get_curves_fx_and_base_maybe_from_solver(
        #     self.curves, solver, curves, fx, base, self.kwargs["pair"][3:]
        # )
        if not isinstance(vol, list):
            vol = [[vol, vol], vol]
        else:
            vol = [[vol[0], vol[2]], vol[1]]  # restructure for strangle / straddle

        # TODO: this meth can be optimised because it calculates greeks at multiple times in frames
        g_grks = self.periods[0].analytic_greeks(curves, solver, fx, base, local, vol[0])
        d_grks = self.periods[1].analytic_greeks(curves, solver, fx, base, local, vol[1])
        sclr = abs(
            self.periods[1].periods[0].periods[0].notional
            / self.periods[0].periods[0].periods[0].notional,
        )

        _unit_attrs = ["delta", "gamma", "vega", "vomma", "vanna", "_kega", "_kappa", "__bs76"]
        _ = {}
        for attr in _unit_attrs:
            _[attr] = g_grks[attr] - sclr * d_grks[attr]

        _notional_attrs = [
            f"delta_{self.kwargs['pair'][:3]}",
            f"gamma_{self.kwargs['pair'][:3]}_1%",
            f"vega_{self.kwargs['pair'][3:]}",
        ]
        for attr in _notional_attrs:
            _[attr] = g_grks[attr] - d_grks[attr]

        _.update(
            {
                "__class": "FXOptionStrat",
                "__strategies": {"FXStrangle": g_grks, "FXStraddle": d_grks},
                "__delta_type": g_grks["__delta_type"],
                "__notional": self.kwargs["notional"],
            },
        )
        return _

    def _plot_payoff(
        self,
        range: list[float] | NoInput = NoInput(0),  # noqa: A002
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
        local: bool = False,
        vol: list[float] | float = NoInput(0),
    ):
        vol = self._vol_as_list(vol, solver)
        self._maybe_set_vega_neutral_notional(curves, solver, fx, base, vol, metric="pips_or_%")
        return super()._plot_payoff(range, curves, solver, fx, base, local, vol)
