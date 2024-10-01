from __future__ import annotations

from abc import ABCMeta
from datetime import datetime

from pandas import DataFrame

from rateslib import defaults
from rateslib.calendars import CalInput, _get_fx_expiry_and_delivery, get_calendar
from rateslib.curves import Curve
from rateslib.default import NoInput, _drb, plot
from rateslib.dual import DualTypes, dual_log
from rateslib.fx import FXForwards, FXRates
from rateslib.fx_volatility import FXDeltaVolSmile, FXDeltaVolSurface, FXVolObj, FXVols
from rateslib.instruments.core import (
    Sensitivities,
    _get_curves_fx_and_base_maybe_from_solver,
    _get_vol_maybe_from_solver,
    _push,
    _update_with_defaults,
)
from rateslib.periods import Cashflow, FXCallPeriod, FXPutPeriod
from rateslib.solver import Solver
from rateslib.splines import evaluate


class FXOption(Sensitivities, metaclass=ABCMeta):
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

    style = "european"
    _pricing = None
    _rate_scalar = 1.0

    def __init__(
        self,
        pair: str,
        expiry: datetime | str,
        notional: float = NoInput(0),
        eval_date: datetime | NoInput = NoInput(0),
        calendar: CalInput = NoInput(0),
        modifier: str | NoInput = NoInput(0),
        eom: bool | NoInput = NoInput(0),
        delivery_lag: int | NoInput = NoInput(0),
        strike: DualTypes | str | NoInput = NoInput(0),
        premium: float | NoInput = NoInput(0),
        premium_ccy: str | NoInput = NoInput(0),
        payment_lag: str | datetime | NoInput = NoInput(0),
        option_fixing: float | NoInput = NoInput(0),
        delta_type: float | NoInput = NoInput(0),
        metric: str | NoInput = NoInput(0),
        curves: list | str | Curve | NoInput = NoInput(0),
        vol: str | FXVols | NoInput = NoInput(0),
        spec: str | NoInput = NoInput(0),
    ):
        self.kwargs = dict(
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

    def __repr__(self):
        return f"<rl.{type(self).__name__} at {hex(id(self))}>"

    def _validate_strike_and_premiums(self):
        if self.kwargs["strike"] is NoInput.blank:
            raise ValueError("`strike` for FXOption must be set to numeric or string value.")
        if isinstance(self.kwargs["strike"], str) and self.kwargs["premium"] is not NoInput.blank:
            raise ValueError(
                "FXOption with string delta as `strike` cannot be initialised with a known "
                "`premium`.\n"
                "Either set `strike` as a defined numeric value, or remove the `premium`.",
            )

    def _set_strike_and_vol(
        self,
        curves: Curve | str | list | NoInput = NoInput(0),
        fx: FXForwards | NoInput = NoInput(0),
        vol: float = NoInput(0),
    ):
        # If the strike for the option is not set directly it must be inferred
        # and some of the pricing elements associated with this strike definition must
        # be captured for use in subsequent formulae.

        if fx is NoInput.blank:
            raise ValueError(
                "An FXForwards object for `fx` is required for FXOption pricing.\n"
                "If this instrument is part of a Solver, have you omitted the `fx` input?",
            )

        self._pricing = {
            "vol": vol,
            "k": self.kwargs["strike"],
            "delta_index": None,
            "spot": fx.pairs_settlement[self.kwargs["pair"]],
            "t_e": self.periods[0]._t_to_expiry(curves[3].node_dates[0]),
            "f_d": fx.rate(self.kwargs["pair"], self.kwargs["delivery"]),
        }
        w_deli = curves[1][self.kwargs["delivery"]]
        w_spot = curves[1][self._pricing["spot"]]

        if isinstance(self.kwargs["strike"], str):
            method = self.kwargs["strike"].lower()

            if method == "atm_forward":
                self._pricing["k"] = fx.rate(self.kwargs["pair"], self.kwargs["delivery"])

            elif method == "atm_spot":
                self._pricing["k"] = fx.rate(self.kwargs["pair"], self._pricing["spot"])

            elif method == "atm_delta":
                self._pricing["k"], self._pricing["delta_index"] = self.periods[
                    0
                ]._strike_and_index_from_atm(
                    delta_type=self.periods[0].delta_type,
                    vol=vol,
                    w_deli=w_deli,
                    w_spot=w_spot,
                    f=self._pricing["f_d"],
                    t_e=self._pricing["t_e"],
                )

            elif method[-1] == "d":  # representing delta
                # then strike is commanded by delta
                self._pricing["k"], self._pricing["delta_index"] = self.periods[
                    0
                ]._strike_and_index_from_delta(
                    delta=float(self.kwargs["strike"][:-1]) / 100.0,
                    delta_type=self.kwargs["delta_type"] + self.kwargs["delta_adjustment"],
                    vol=vol,
                    w_deli=w_deli,
                    w_spot=w_spot,
                    f=self._pricing["f_d"],
                    t_e=self._pricing["t_e"],
                )

            # TODO: this may affect solvers dependent upon sensitivity to vol for changing strikes.
            # set the strike as a float without any sensitivity. Trade definition is a fixed
            # quantity at this stage. Similar to setting a fixed rate as a float on an unpriced
            # IRS for mid-market.

            # self.periods[0].strike = self._pricing["k"]
            self.periods[0].strike = float(self._pricing["k"])

        if isinstance(vol, FXVolObj):
            if self._pricing["delta_index"] is None:
                self._pricing["delta_index"], self._pricing["vol"], _ = vol.get_from_strike(
                    k=self._pricing["k"],
                    f=self._pricing["f_d"],
                    w_deli=w_deli,
                    w_spot=w_spot,
                    expiry=self.kwargs["expiry"],
                )
            else:
                self._pricing["vol"] = vol._get_index(
                    self._pricing["delta_index"],
                    self.kwargs["expiry"],
                )

    def _set_premium(
        self,
        curves: Curve | str | list | NoInput = NoInput(0),
        fx: FXForwards | NoInput = NoInput(0),
    ):
        if self.kwargs["premium"] is NoInput.blank:
            # then set the CashFlow to mid-market
            try:
                npv = self.periods[0].npv(curves[1], curves[3], fx, vol=self._pricing["vol"])
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
                premium = npv / (curves[3][m_p] * fx.rate("eurusd", m_p))
            else:
                premium = npv / curves[3][m_p]

            self.periods[1].notional = float(premium)

    def _get_vol_curves_fx_and_base_maybe_from_solver(self, solver, curves, fx, base, vol):
        """
        Parses the inputs including the instrument's attributes and also validates them
        """
        curves, fx, base = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            fx,
            base,
            self.kwargs["pair"][3:],
        )
        vol = _get_vol_maybe_from_solver(self.vol, vol, solver)
        if isinstance(vol, FXDeltaVolSmile) and vol.eval_date != curves[1].node_dates[0]:
            raise ValueError(
                "The `eval_date` on the FXDeltaVolSmile and the Curve do not align.\n"
                "Aborting calculation to avoid pricing errors.",
            )
        return curves, fx, base, vol

    def rate(
        self,
        curves: Curve | str | list | NoInput = NoInput(0),
        solver: Solver | NoInput = NoInput(0),
        fx: FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
        vol: float | FXVols | NoInput = NoInput(0),
        metric: str | NoInput = NoInput(0),
    ):
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
        curves, fx, _base, vol = self._get_vol_curves_fx_and_base_maybe_from_solver(
            solver,
            curves,
            fx,
            base,
            vol,
        )
        self._set_strike_and_vol(curves, fx, vol)
        # self._set_premium(curves, fx)

        metric = _drb(self.kwargs["metric"], metric)
        if metric in ["vol", "single_vol"]:
            return self._pricing["vol"]

        _ = self.periods[0].rate(curves[1], curves[3], fx, NoInput(0), False, self._pricing["vol"])
        if metric == "premium":
            if self.periods[0].metric == "pips":
                _ *= self.periods[0].notional / 10000
            else:  # == "percent"
                _ *= self.periods[0].notional / 100
        return _

    def npv(
        self,
        curves: Curve | str | list | NoInput = NoInput(0),
        solver: Solver | NoInput = NoInput(0),
        fx: FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
        local: bool = False,
        vol: float = NoInput(0),
    ):
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

        Returns
        -------
        float, Dual, Dual2 or dict of such.
        """
        curves, fx, base, vol = self._get_vol_curves_fx_and_base_maybe_from_solver(
            solver,
            curves,
            fx,
            base,
            vol,
        )
        self._set_strike_and_vol(curves, fx, vol)
        self._set_premium(curves, fx)

        opt_npv = self.periods[0].npv(curves[1], curves[3], fx, base, local, vol)
        if self.kwargs["premium_ccy"] == self.kwargs["pair"][:3]:
            disc_curve = curves[1]
        else:
            disc_curve = curves[3]
        prem_npv = self.periods[1].npv(NoInput(0), disc_curve, fx, base, local)
        if local:
            return {k: opt_npv.get(k, 0) + prem_npv.get(k, 0) for k in set(opt_npv) | set(prem_npv)}
        else:
            return opt_npv + prem_npv

    def cashflows(
        self,
        curves: Curve | str | list | NoInput = NoInput(0),
        solver: Solver | NoInput = NoInput(0),
        fx: float | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
        vol: float = NoInput(0),
    ):
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
        vol: float, Dual, Dual2 or FXDeltaVolSmile
            The volatility used in calculation.

        Returns
        -------
        DataFrame

        """
        curves, fx, base, vol = self._get_vol_curves_fx_and_base_maybe_from_solver(
            solver,
            curves,
            fx,
            base,
            vol,
        )
        self._set_strike_and_vol(curves, fx, vol)
        self._set_premium(curves, fx)

        seq = [
            self.periods[0].cashflows(curves[1], curves[3], fx, base, vol=vol),
            self.periods[1].cashflows(curves[1], curves[3], fx, base),
        ]
        return DataFrame.from_records(seq)

    def analytic_greeks(
        self,
        curves: Curve | str | list | NoInput = NoInput(0),
        solver: Solver | NoInput = NoInput(0),
        fx: FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
        local: bool = False,
        vol: float = NoInput(0),
    ):
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
        local: bool,
            Not used by `analytic_greeks`.
        vol: float, or FXDeltaVolSmile
            The volatility used in calculation.

        Returns
        -------
        float, Dual, Dual2
        """
        curves, fx, base, vol = self._get_vol_curves_fx_and_base_maybe_from_solver(
            solver,
            curves,
            fx,
            base,
            vol,
        )
        self._set_strike_and_vol(curves, fx, vol)
        # self._set_premium(curves, fx)

        return self.periods[0].analytic_greeks(
            curves[1],
            curves[3],
            fx,
            base,
            local,
            vol,
            premium=NoInput(0),
        )

    def _plot_payoff(
        self,
        range: list[float] | NoInput = NoInput(0),
        curves: Curve | str | list | NoInput = NoInput(0),
        solver: Solver | NoInput = NoInput(0),
        fx: FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
        local: bool = False,
        vol: float = NoInput(0),
    ):
        """
        Mechanics to determine (x,y) coordinates for payoff at expiry plot.
        """
        curves, fx, base, vol = self._get_vol_curves_fx_and_base_maybe_from_solver(
            solver,
            curves,
            fx,
            base,
            vol,
        )
        self._set_strike_and_vol(curves, fx, vol)
        # self._set_premium(curves, fx)

        x, y = self.periods[0]._payoff_at_expiry(range)
        return x, y

    def plot_payoff(
        self,
        range: list[float] | NoInput = NoInput(0),
        curves: Curve | str | list | NoInput = NoInput(0),
        solver: Solver | NoInput = NoInput(0),
        fx: FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
        local: bool = False,
        vol: float = NoInput(0),
    ):
        x, y = self._plot_payoff(range, curves, solver, fx, base, local, vol)
        return plot(x, [y])


class FXCall(FXOption):
    """
    Create an *FX Call* option.

    For parameters see :class:`~rateslib.instruments.FXOption`.
    """

    style = "european"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.periods = [
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
            Cashflow(
                notional=self.kwargs["premium"],
                payment=self.kwargs["payment"],
                currency=self.kwargs["premium_ccy"],
                stub_type="Premium",
            ),
        ]


class FXPut(FXOption):
    """
    Create an *FX Put* option.

    For parameters see :class:`~rateslib.instruments.FXOption`.
    """

    style = "european"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.periods = [
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
            Cashflow(
                notional=self.kwargs["premium"],
                payment=self.kwargs["payment"],
                currency=self.kwargs["premium_ccy"],
                stub_type="Premium",
            ),
        ]


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

    _pricing = {}

    def __init__(
        self,
        options: list[FXOption],
        rate_weight: list[float],
        rate_weight_vol: list[float],
    ):
        self.periods = options
        self.rate_weight = rate_weight
        self.rate_weight_vol = rate_weight_vol
        if len(self.periods) != len(self.rate_weight) or len(self.periods) != len(
            self.rate_weight_vol,
        ):
            raise ValueError(
                "`rate_weight` and `rate_weight_vol` must have same length as `options`.",
            )

    def __repr__(self):
        return f"<rl.{type(self).__name__} at {hex(id(self))}>"

    def _vol_as_list(self, vol, solver):
        """Standardise a vol input over the list of periods"""
        if not isinstance(vol, (list, tuple)):
            vol = [vol] * len(self.periods)
        return [_get_vol_maybe_from_solver(self.vol, _, solver) for _ in vol]

    def rate(
        self,
        curves: Curve | str | list | NoInput = NoInput(0),
        solver: Solver | NoInput = NoInput(0),
        fx: float | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
        vol: list[float] | float = NoInput(0),
        metric: str | NoInput = NoInput(0),  # "pips_or_%",
    ):
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
        for option, vol_, weight in zip(self.periods, vol, weights):
            _ += option.rate(curves, solver, fx, base, vol_, metric) * weight
        return _

    def npv(
        self,
        curves: Curve | str | list | NoInput = NoInput(0),
        solver: Solver | NoInput = NoInput(0),
        fx: float | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
        local: bool = False,
        vol: list[float] | float = NoInput(0),
    ):
        if not isinstance(vol, list):
            vol = [vol] * len(self.periods)

        results = [
            option.npv(curves, solver, fx, base, local, vol_)
            for (option, vol_) in zip(self.periods, vol)
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
        range: list[float] | NoInput = NoInput(0),
        curves: Curve | str | list | NoInput = NoInput(0),
        solver: Solver | NoInput = NoInput(0),
        fx: FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
        local: bool = False,
        vol: list[float] | float = NoInput(0),
    ):
        if not isinstance(vol, list):
            vol = [vol] * len(self.periods)

        y = None
        for option, vol_ in zip(self.periods, vol):
            x, y_ = option._plot_payoff(range, curves, solver, fx, base, local, vol_)
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
        curves: Curve | str | list | NoInput = NoInput(0),
        solver: Solver | NoInput = NoInput(0),
        fx: FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
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
        for option, _vol in zip(self.periods, vol):
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
        self.periods = [
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
        for k, p in zip(self.kwargs["strike"], self.kwargs["premium"]):
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
        self.periods = [
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
        self.periods = [
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
        for k, p in zip(self.kwargs["strike"], self.kwargs["premium"]):
            if isinstance(k, str) and p != NoInput.blank:
                raise ValueError(
                    "FXStrangle with string delta as `strike` cannot be initialised with a "
                    "known `premium`.\n"
                    "Either set `strike` as a defined numeric value, or remove the `premium`.",
                )

    def rate(
        self,
        curves: Curve | str | list | NoInput = NoInput(0),
        solver: Solver | NoInput = NoInput(0),
        fx: float | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
        vol: list[float] | float = NoInput(0),
        metric: str | NoInput = NoInput(0),  # "pips_or_%",
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
            self._pricing["strangle_greeks"] = {
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
        self.periods = [
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
        ]

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
            self._pricing["straddle_greeks"] = self.periods[1].analytic_greeks(
                curves,
                solver,
                fx,
                base,
                vol=vol[1],
            )
            strangle_vega = self._pricing["strangle_greeks"]["market_vol"]["FXPut"]["vega"]
            strangle_vega += self._pricing["strangle_greeks"]["market_vol"]["FXCall"]["vega"]
            straddle_vega = self._pricing["straddle_greeks"]["vega"]
            scalar = strangle_vega / straddle_vega
            self.periods[1].kwargs["notional"] = float(
                self.periods[0].periods[0].periods[0].notional * -scalar,
            )
            self.periods[1]._set_notionals(self.periods[1].kwargs["notional"])
            # BrokerFly -> Strangle -> FXPut -> FXPutPeriod

    def rate(
        self,
        curves: Curve | str | list | NoInput = NoInput(0),
        solver: Solver | NoInput = NoInput(0),
        fx: float | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
        vol: list[float] | float = NoInput(0),
        metric: str | NoInput = NoInput(0),
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
        for option_strat, vol_, weight in zip(self.periods, vol, weights):
            _ += option_strat.rate(curves, solver, fx, base, vol_, metric) * weight
        return _

    def analytic_greeks(
        self,
        curves: Curve | str | list | NoInput = NoInput(0),
        solver: Solver | NoInput = NoInput(0),
        fx: FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
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
        range: list[float] | NoInput = NoInput(0),
        curves: Curve | str | list | NoInput = NoInput(0),
        solver: Solver | NoInput = NoInput(0),
        fx: FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
        local: bool = False,
        vol: list[float] | float = NoInput(0),
    ):
        vol = self._vol_as_list(vol, solver)
        self._maybe_set_vega_neutral_notional(curves, solver, fx, base, vol, metric="pips_or_%")
        return super()._plot_payoff(range, curves, solver, fx, base, local, vol)
