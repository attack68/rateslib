from __future__ import annotations

from typing import TYPE_CHECKING

from rateslib import defaults
from rateslib.dual.utils import _dual_float
from rateslib.enums.generics import NoInput, _drb
from rateslib.instruments.fx_options.risk_reversal import FXOptionStrat
from rateslib.instruments.fx_options.straddle import FXStraddle
from rateslib.instruments.fx_options.strangle import FXStrangle

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        Any,
        CalInput,
        CurvesT_,
        DualTypes,
        DualTypes_,
        FXForwards_,
        FXVolStrat_,
        Sequence,
        Solver_,
        VolT_,
        bool_,
        datetime,
        datetime_,
        int_,
        str_,
    )


class FXBrokerFly(FXOptionStrat):
    """
    An *FX Strangle* :class:`~rateslib.instruments.FXOptionStrat`.

    A *Straddle* is composed of a lower strike :class:`~rateslib.instruments.FXPut`
    and a higher strike :class:`~rateslib.instruments.FXCall`.

    .. rubric:: Examples

    .. ipython:: python
       :suppress:

       from rateslib.instruments import FXStraddle
       from datetime import datetime as dt

    .. ipython:: python

       fxc = FXRiskReversal(
           expiry="3m",
           strike="atm_delta",
           eval_date=dt(2020, 1, 1),
           spec="eurusd_call",
       )
       fxc.cashflows()

    .. rubric:: Pricing

    The pricing mirrors that for an :class:`~rateslib.instruments.FXCall`.
    Allowable inputs are:

    .. code-block:: python

       curves = [rate_curve, disc_curve]  #  two curves are applied in the given order
       curves = {"rate_curve": rate_curve, "disc_curve": disc_curve}  # dict form is explicit

    Any *FXOption* also requires an :class:`~rateslib.fx.FXForwards` as input to the ``fx``
    argument.

    A ``vol`` argument must be provided to each *Instrument*. This can either be a single
    value universally used for all, or an individual item as part of a sequence. Allowed
    inputs are:

    .. code-block:: python

       vol = 12.0 | vol_obj  # a single item universally applied
       vol = [12.0, 12.0]  # values for the Put and Call respectively

    The pricing ``metric`` will return the following calculations:

    - *'vol'*: the implied volatility value of the option from a volatility object.
    - *'premium'*: the cash premium amount applicable to the 'payment' date.
    - *'pips_or_%'*: if the premium currency is LHS of ``pair`` this is a % of notional, whilst if
      the premium currency is RHS this gives a number of pips of the FX rate.

    .. role:: red

    .. role:: green

    Parameters
    ----------
    .

        .. note::

           The following define **fx option** and generalised **settlement** parameters.

    expiry: datetime, str, :red:`required`
        The expiry of the option. If given in string tenor format, e.g. "1M" requires an
        ``eval_date``. See **Notes**.
    strike: 2-tuple of float, Variable, str, :red:`required`
        The strikes of the put and the call in order.
    pair: str, :red:`required`
        The currency pair for the FX rate which settles the option, in 3-digit codes, e.g. "eurusd".
        May be included as part of ``spec``.
    notional: float, :green:`optional (set by 'defaults')`
        The notional amount of each option expressed in units of LHS of ``pair``.
    eval_date: datetime, :green:`optional`
        Only required if ``expiry`` is given as string tenor.
        Should be entered as today (also called horizon) and **not** spot. Spot is derived
        from ``delivery_lag`` and ``calendar``.
    modifier : str, :green:`optional (set by 'defaults')`
        The modification rule, in {"F", "MF", "P", "MP"} for date evaluation.
    eom: bool, :green:`optional (set by 'defaults')`
        Whether to use end-of-month rolls when expiry is given as a month or year tenor.
    calendar : calendar or str, :green:`optional`
        The holiday calendar object to use. If str, looks up named calendar from
        static data.
    delivery_lag: int, :green:`optional (set by 'defaults')`
        The number of business days after expiry that the physical settlement of the FX
        exchange occurs.
    payment_lag: int or datetime, :green:`optional (set by 'defaults')`
        The number of business days after expiry to pay premium. If a *datetime* is given this will
        set the premium date explicitly.
    premium_ccy: str, :green:`optional (set as RHS of 'pair')`
        The currency in which the premium is paid. Can *only* be one of the two currencies
        in `pair`.
    delta_type: FXDeltaMethod, str, :green:`optional (set by 'defaults')`
        When deriving strike from delta use the equation associated with *'spot'* or *'forward'*
        delta. If premium currency is LHS of ``pair`` then this will produce
        **premium adjusted** delta values. If the `premium_ccy` is RHS of ``pair`` then delta values
        are **unadjusted**.

        .. note::

           The following define additional **rate** parameters.

    premium: 2-tuple of float, :green:`optional`
        The amount paid for the put and call in order. If not given assumes unpriced
        *Options* and sets this as mid-market premium during pricing.
    option_fixings: 2-tuple of float, Dual, Dual2, Variable, Series, str, :green:`optional`
        The value of each option's :class:`~rateslib.data.fixings.FXFixing`. If a scalar, is used
        directly. If a string identifier, links to the central ``fixings`` object and data loader.

        .. note::

           The following are **meta parameters**.

    metric : str, :green:`optional (set as "pips_or_%")`
        The pricing metric returned by the ``rate`` method. See **Pricing**.
    curves : _BaseCurve, str, dict, _Curves, Sequence, :green:`optional`
        Pricing objects passed directly to the *Instrument's* methods' ``curves`` argument. See
        **Pricing**.
    vol: str, Smile, Surface, float, Dual, Dual2, Variable, Sequence
        Pricing objects passed directly to the *Instrument's* methods' ``vol`` argument. See
        **Pricing**.
    spec : str, optional
        An identifier to pre-populate many field with conventional values. See
        :ref:`here<defaults-doc>` for more info and available values.

    Notes
    -----
    Buying a *Straddle* equates to buying a :class:`~rateslib.instruments.FXPut`
    and buying a :class:`~rateslib.instruments.FXCall` with the same strike. The ``notional`` of
    each are the same, and should be entered as a single value.

    When supplying ``strike`` as a string delta the strike will be determined at price time from
    the provided volatility.

    This class is an alias constructor for an
    :class:`~rateslib.instruments.FXOptionStrat` where the number
    of options and their definitions and nominals have been specifically overloaded for
    convenience.
    """

    _rate_scalar = 100.0

    def __init__(
        self,
        expiry: datetime | str,
        strike: tuple[tuple[DualTypes | str, DualTypes | str], DualTypes | str],
        pair: str_ = NoInput(0),
        *,
        notional: tuple[DualTypes_, DualTypes_] = NoInput(0),
        eval_date: datetime | NoInput = NoInput(0),
        calendar: CalInput = NoInput(0),
        modifier: str_ = NoInput(0),
        eom: bool_ = NoInput(0),
        delivery_lag: int_ = NoInput(0),
        premium: tuple[tuple[DualTypes_, DualTypes_], tuple[DualTypes_, DualTypes_]] = (
            (NoInput(0), NoInput(0)),
            (NoInput(0), NoInput(0)),
        ),
        premium_ccy: str_ = NoInput(0),
        payment_lag: str | datetime_ = NoInput(0),
        option_fixings: DualTypes_ = NoInput(0),
        delta_type: str_ = NoInput(0),
        metric: str_ = NoInput(0),
        curves: CurvesT_ = NoInput(0),
        vol: VolT_ = NoInput(0),
        spec: str_ = NoInput(0),
    ) -> None:
        vol_ = self._parse_vol(vol)
        if isinstance(notional, NoInput):
            notional_ = (defaults.notional, NoInput(0))
        elif isinstance(notional, tuple | list):
            notional_ = notional
            notional_[1] = NoInput(0) if notional_[1] is None else notional_[1]
        else:
            raise ValueError("FXBrokerFly `notional` must be a 2 element sequence if given.")
        strategies = [
            FXStrangle(
                pair=pair,
                expiry=expiry,
                delivery_lag=delivery_lag,
                payment_lag=payment_lag,
                calendar=calendar,
                modifier=modifier,
                eom=eom,
                eval_date=eval_date,
                strike=strike[0],
                notional=notional_[0],
                option_fixings=option_fixings[0]
                if isinstance(option_fixings, tuple | list)
                else option_fixings,
                delta_type=delta_type,
                premium=premium[0],
                premium_ccy=premium_ccy,
                curves=curves,
                vol=vol_[0],  # type: ignore[arg-type, index]
                metric=NoInput(0),
                spec=spec,
            ),
            FXStraddle(
                pair=pair,
                expiry=expiry,
                delivery_lag=delivery_lag,
                payment_lag=payment_lag,
                calendar=calendar,
                modifier=modifier,
                eom=eom,
                eval_date=eval_date,
                strike=strike[1],
                notional=notional_[1],
                option_fixings=option_fixings[1]
                if isinstance(option_fixings, tuple | list)
                else option_fixings,
                delta_type=delta_type,
                premium=premium[1],
                premium_ccy=premium_ccy,
                curves=curves,
                vol=vol_[1],  # type: ignore[arg-type, index]
                metric=NoInput(0),
                spec=spec,
            ),
        ]
        super().__init__(
            options=strategies,
            rate_weight=[1.0, 1.0],
            rate_weight_vol=[1.0, -1.0],
            metric=_drb("single_vol", metric),
            curves=curves,
            vol=vol_,
        )
        self.kwargs.leg1["notional"] = notional_

    @classmethod
    def _parse_vol(cls, vol: FXVolStrat_) -> tuple[FXVolStrat_, FXVolStrat_]:  # type: ignore[override]
        if not isinstance(vol, list | tuple):
            vol = (vol, vol)
        return (FXStrangle._parse_vol(vol[0]), FXStrangle._parse_vol(vol[1]))

    def _maybe_set_vega_neutral_notional(
        self,
        curves: CurvesT_,
        solver: Solver_,
        fx: FXForwards_,
        base: str_,
        vol: FXVolStrat_,
        metric: str_,
    ) -> None:
        """
        Calculate the vega of the strangle and then set the notional on the straddle
        to yield a vega neutral strategy.

        Notional is set as a fixed quantity, collapsing any AD sensitivities in accordance
        with the general principle for determining risk sensitivities of unpriced instruments.

        This is only applied if ``metric`` is a cash based quantity, {"pips_or_%", "premium"}
        """
        if isinstance(self.kwargs.leg1["notional"][1], NoInput) and metric in [
            "pips_or_%",
            "premium",
        ]:
            self.instruments[0]._rate(  # type: ignore[union-attr]
                curves,
                solver,
                fx,
                base,
                vol=vol[0],
                metric="single_vol",
                record_greeks=True,
                forward=NoInput(0),
                settlement=NoInput(0),
            )
            self._greeks["straddle"] = self.instruments[1].analytic_greeks(
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
            self.instruments[1].kwargs.leg1["notional"] = _dual_float(
                self.instruments[0].kwargs.leg1["notional"] * -scalar,
            )
            self.instruments[1]._set_notionals(self.instruments[1].kwargs.leg1["notional"])  # type: ignore[union-attr]
            # BrokerFly -> Strangle -> FXPut -> FXPutPeriod

    def rate(
        self,
        *,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: FXVolStrat_ = NoInput(0),
        base: str_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
        metric: str_ = NoInput(0),
    ) -> DualTypes:
        """
        Returns the rate of the *FXBrokerFly* according to a pricing metric.

        For parameters see :meth:`FXOptionStrat.rate <rateslib.instruments.FXOptionStrat.rate>`.

        Notes
        ------

        .. warning::

           The default ``metric`` for an *FXBrokerFly* is *'single_vol'*, which requires an
           iterative algorithm to solve.
           For defined strikes it is usually very accurate but for strikes defined by delta it
           will return a solution within 0.01 pips. This means it is both slower than other
           instruments and inexact.

           The ``metric`` *'vol'* is not sensible to use with an *FXBrokerFly*, although it will
           return the arithmetic average volatility across both strategies, *'single_vol'* is the
           more standardised choice.
        """
        # Get curves and vol
        vol_ = [
            _drb(d, b) for (d, b) in zip(self.kwargs.meta["vol"], self._parse_vol(vol), strict=True)
        ]
        _curves = self._parse_curves(curves)

        metric_ = _drb(self.kwargs.meta["metric"], metric).lower()
        self._maybe_set_vega_neutral_notional(_curves, solver, fx, base, vol_, metric_)

        if metric_ == "pips_or_%":
            straddle_scalar = (
                self.instruments[1].instruments[0]._option.settlement_params.notional  # type: ignore[union-attr]
                / self.instruments[0].instruments[0]._option.settlement_params.notional  # type: ignore[union-attr]
            )
            weights: Sequence[DualTypes] = [1.0, straddle_scalar]
        elif metric_ == "premium":
            weights = self.kwargs.meta["rate_weight"]
        else:
            weights = self.kwargs.meta["rate_weight_vol"]
        _: DualTypes = 0.0
        for option_strat, vol__, weight in zip(self.instruments, vol_, weights, strict=False):
            _ += (
                option_strat.rate(
                    curves=_curves,
                    solver=solver,
                    fx=fx,
                    base=base,
                    vol=vol__,
                    metric=metric_,
                    forward=forward,
                    settlement=settlement,
                )
                * weight
            )
        return _

    def analytic_greeks(
        self,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        base: str_ = NoInput(0),
        vol: FXVolStrat_ = NoInput(0),
    ) -> dict[str, Any]:
        # implicitly call set_pricing_mid for unpriced parameters
        self.rate(curves=curves, solver=solver, fx=fx, base=base, vol=vol, metric="pips_or_%")

        vol_ = self._parse_vol(vol)

        # TODO: this meth can be optimised because it calculates greeks at multiple times in frames
        g_grks = self.instruments[0].analytic_greeks(curves, solver, fx, base, vol_[0])  # type: ignore[index, arg-type]
        d_grks = self.instruments[1].analytic_greeks(curves, solver, fx, base, vol_[1])  # type: ignore[index, arg-type]
        sclr = abs(
            self.instruments[1].instruments[0]._option.settlement_params.notional  # type: ignore[union-attr]
            / self.instruments[0].instruments[0]._option.settlement_params.notional,  # type: ignore[union-attr]
        )

        _unit_attrs = ["delta", "gamma", "vega", "vomma", "vanna", "_kega", "_kappa", "__bs76"]
        _: dict[str, Any] = {}
        for attr in _unit_attrs:
            _[attr] = g_grks[attr] - sclr * d_grks[attr]

        _notional_attrs = [
            f"delta_{self.kwargs.leg1['pair'][:3]}",
            f"gamma_{self.kwargs.leg1['pair'][:3]}_1%",
            f"vega_{self.kwargs.leg1['pair'][3:]}",
        ]
        for attr in _notional_attrs:
            _[attr] = g_grks[attr] - d_grks[attr]

        _.update(
            {
                "__class": "FXOptionStrat",
                "__strategies": {"FXStrangle": g_grks, "FXStraddle": d_grks},
                "__delta_type": g_grks["__delta_type"],
                "__notional": self.kwargs.leg1["notional"],
            },
        )
        return _

    def _plot_payoff(
        self,
        window: list[float] | NoInput = NoInput(0),  # noqa: A002
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        base: str_ = NoInput(0),
        local: bool = False,
        vol: FXVolStrat_ = NoInput(0),
    ) -> tuple[Any, Any]:
        vol_ = self._parse_vol(vol)
        self._maybe_set_vega_neutral_notional(curves, solver, fx, base, vol_, metric="pips_or_%")
        return super()._plot_payoff(window, curves, solver, fx, base, local, vol_)


#
#
# class FXBrokerFly(FXOptionStrat):
#     """
#     Create an *FX BrokerFly* option strategy.
#
#     An *FXBrokerFly* is composed of an :class:`~rateslib.instruments.FXStrangle` and an
#     :class:`~rateslib.instruments.FXStraddle`, in that order.
#
#     For additional arguments see :class:`~rateslib.instruments.FXOption`.
#
#     Parameters
#     ----------
#     args: tuple
#         Positional arguments to :class:`~rateslib.instruments.FXOption`.
#     strike: 2-element sequence
#         The first element should be a 2-element sequence of strikes of the *FXStrangle*.
#         The second element should be a single element for the strike of the *FXStraddle*.
#         call, e.g. `[["-25d", "25d"], "atm_delta"]`.
#     premium: 2-element sequence, optional
#         The premiums associated with each option of the strategy;
#         The first element contains 2 values for the premiums of each *FXOption* in the *Strangle*.
#         The second element contains 2 values for the premiums of each *FXOption* in the *Straddle*.
#     notional: 2-element sequence, optional
#         The first element is the notional associated with the *Strangle*. If the second element
#         is *None*, it will be implied in a vega neutral sense at price time.
#     metric: str, optional
#         The default metric to apply in the method :meth:`~rateslib.instruments.FXOptionStrat.rate`
#     kwargs: tuple
#         Keyword arguments to :class:`~rateslib.instruments.FXOption`.
#
#     Notes
#     -----
#     Buying a *BrokerFly* equates to buying an :class:`~rateslib.instruments.FXStrangle` and
#     selling a :class:`~rateslib.instruments.FXStraddle`, where the convention is to set the
#     notional on the *Straddle* such that the entire strategy is *vega* neutral at inception.
#
#     When supplying ``strike`` as a string delta the strike will be determined at price time from
#     the provided volatility.
#
#     .. warning::
#
#        The default ``metric`` for an *FXBrokerFly* is *'single_vol'*, which requires an iterative
#        algorithm to solve.
#        For defined strikes it is accurate but for strikes defined by delta it
#        will return a solution within 0.1 pips. This means it is both slower than other instruments
#        and inexact.
#
#     """
#
#     rate_weight = [1.0, 1.0]
#     rate_weight_vol = [1.0, -1.0]
#     _rate_scalar = 100.0
#
#     periods: list[FXOptionStrat]  # type: ignore[assignment]
#     vol: FXVolStrat_
#
#     def __init__(
#         self,
#         *args: Any,
#         strike: tuple[tuple[DualTypes | str_, DualTypes | str_], DualTypes | str_] = (
#             (NoInput(0), NoInput(0)),
#             NoInput(0),
#         ),
#         premium: tuple[tuple[DualTypes_, DualTypes_], tuple[DualTypes_, DualTypes_]] = (
#             (NoInput(0), NoInput(0)),
#             (NoInput(0), NoInput(0)),
#         ),
#         notional: tuple[DualTypes_, DualTypes_] = (NoInput(0), NoInput(0)),
#         metric: str = "single_vol",
#         **kwargs: Any,
#     ) -> None:
#         super(FXOptionStrat, self).__init__(  # type: ignore[misc]
#             *args,
#             premium=list(premium),  # type: ignore[arg-type]
#             strike=list(strike),  # type: ignore[arg-type]
#             notional=list(notional),  # type: ignore[arg-type]
#             **kwargs,
#         )
#         self.kwargs["notional"][1] = (
#             NoInput(0) if self.kwargs["notional"][1] is None else self.kwargs["notional"][1]
#         )
#         self.kwargs["metric"] = metric
#         self._strat_elements = (
#             FXStrangle(
#                 pair=self.kwargs["pair"],
#                 expiry=self.kwargs["expiry"],
#                 delivery_lag=self.kwargs["delivery"],
#                 payment_lag=self.kwargs["payment"],
#                 calendar=self.kwargs["calendar"],
#                 modifier=self.kwargs["modifier"],
#                 strike=self.kwargs["strike"][0],
#                 notional=self.kwargs["notional"][0],
#                 option_fixing=self.kwargs["option_fixing"],
#                 delta_type=self.kwargs["delta_type"],
#                 premium=self.kwargs["premium"][0],
#                 premium_ccy=self.kwargs["premium_ccy"],
#                 metric=self.kwargs["metric"],
#                 curves=self.curves,
#                 vol=self.vol,
#             ),
#             FXStraddle(
#                 pair=self.kwargs["pair"],
#                 expiry=self.kwargs["expiry"],
#                 delivery_lag=self.kwargs["delivery"],
#                 payment_lag=self.kwargs["payment"],
#                 calendar=self.kwargs["calendar"],
#                 modifier=self.kwargs["modifier"],
#                 strike=self.kwargs["strike"][1],
#                 notional=self.kwargs["notional"][1],
#                 option_fixing=self.kwargs["option_fixing"],
#                 delta_type=self.kwargs["delta_type"],
#                 premium=self.kwargs["premium"][1],
#                 premium_ccy=self.kwargs["premium_ccy"],
#                 metric="vol" if self.kwargs["metric"] == "single_vol" else self.kwargs["metric"],
#                 curves=self.curves,
#                 vol=self.vol,
#             ),
#         )
