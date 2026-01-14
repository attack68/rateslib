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

from typing import TYPE_CHECKING

from rateslib import defaults
from rateslib.curves._parsers import _validate_obj_not_no_input
from rateslib.dual import dual_log, newton_1dim
from rateslib.dual.utils import _set_ad_order_objects
from rateslib.enums.generics import NoInput, _drb
from rateslib.enums.parameters import FXDeltaMethod
from rateslib.fx_volatility import FXDeltaVolSmile, FXDeltaVolSurface, FXSabrSmile, FXSabrSurface
from rateslib.instruments.fx_options.call_put import FXCall, FXPut
from rateslib.instruments.fx_options.risk_reversal import _BaseFXOptionStrat
from rateslib.instruments.protocols.pricing import (
    _get_fx_forwards_maybe_from_solver,
    _maybe_get_curve_maybe_from_solver,
    _maybe_get_fx_vol_maybe_from_solver,
    _Vol,
)
from rateslib.periods.utils import _validate_fx_as_forwards
from rateslib.splines import evaluate

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        Any,
        CalInput,
        CurvesT_,
        DualTypes,
        DualTypes_,
        FXForwards,
        FXForwards_,
        FXVolStrat_,
        Solver_,
        VolT_,
        _BaseFXOptionPeriod,
        _FXVolOption,
        _Vol,
        bool_,
        datetime,
        datetime_,
        int_,
        str_,
    )


class FXStrangle(_BaseFXOptionStrat):
    """
    An *FX Strangle* :class:`~rateslib.instruments._BaseFXOptionStrat`.

    A *Strangle* is composed of a lower strike :class:`~rateslib.instruments.FXPut`
    and a higher strike :class:`~rateslib.instruments.FXCall`.

    .. rubric:: Examples

    .. ipython:: python
       :suppress:

       from rateslib import FXStrangle, Curve, FXForwards, FXRates, FXDeltaVolSmile, dt

    .. ipython:: python

       fxs = FXStrangle(
           expiry="3m",
           strike=["-10d", "10d"],
           eval_date=dt(2020, 1, 1),
           spec="eurusd_call",
           notional=1000000,
       )
       fxs.cashflows()

    .. rubric:: Pricing

    The pricing mirrors that for an :class:`~rateslib.instruments.FXCall`. All options use the
    same ``curves``.
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

    *FXStrangles* have peculiar market conventions. If the strikes are given as delta percentages
    then numeric values will first be derived using the *'single_vol'* approach. Any *'premium'*
    or *'pips_or_%'* values can then be calculated using those strikes and this volatility.
    The following pricing ``metric`` are available, with examples:

    .. ipython:: python

       eur = Curve({dt(2020, 1, 1): 1.0, dt(2021, 1, 1): 0.98})
       usd = Curve({dt(2020, 1, 1): 1.0, dt(2021, 1, 1): 0.96})
       fxf = FXForwards(
           fx_rates=FXRates({"eurusd": 1.10}, settlement=dt(2020, 1, 3)),
           fx_curves={"eureur": eur, "eurusd": eur, "usdusd": usd},
       )
       fxvs = FXDeltaVolSmile(
           nodes={0.25: 11.0, 0.5: 9.8, 0.75: 10.7},
           expiry=dt(2020, 4, 1),
           eval_date=dt(2020, 1, 1),
           delta_type="forward",
       )

    - **'single_vol'**: the singular volatility value that when applied to each option separately
      yields a summed premium amount equal to the summed premium when each option is valued with
      the appropriate volatility from an object (with the strikes determined by the single vol).
      **'vol'** is an alias for single vol and returns the same value.

      .. ipython:: python

         fxs.rate(vol=fxvs, curves=[eur, usd], fx=fxf, metric="single_vol")
         fxs.rate(vol=[12.163490, 12.163490], curves=[eur, usd], fx=fxf, metric="premium")

      This requires an iterative calculation for which the tolerance is set to 1e-6 with a
      maximum allowed number of iterations of 10.

    - **'premium'**: the summed cash premium amount, of both options, applicable to the 'payment'
      date. If strikes are given as delta percentages then they are first determined using the
      *'single_vol'*.

      .. ipython:: python

         fxs.rate(vol=fxvs, curves=[eur, usd], fx=fxf, metric="premium")

    - **'pips_or_%'**: if the premium currency is LHS of ``pair`` this is a % of notional, whilst if
      the premium currency is RHS this gives a number of pips of the FX rate. Summed over both
      options. For strikes set with delta percentages these are first determined using the
      'single_vol'.

      .. ipython:: python

         fxs.rate(vol=fxvs, curves=[eur, usd], fx=fxf, metric="pips_or_%")

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

    metric : str, :green:`optional (set as "single_vol")`
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

    """

    _rate_scalar = 100.0

    def __init__(
        self,
        expiry: datetime | str,
        strike: tuple[DualTypes | str, DualTypes | str],
        pair: str_ = NoInput(0),
        *,
        notional: DualTypes_ = NoInput(0),
        eval_date: datetime | NoInput = NoInput(0),
        calendar: CalInput = NoInput(0),
        modifier: str_ = NoInput(0),
        eom: bool_ = NoInput(0),
        delivery_lag: int_ = NoInput(0),
        premium: tuple[DualTypes_, DualTypes_] = (NoInput(0), NoInput(0)),
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
        notional_ = _drb(defaults.notional, notional)
        options = [
            FXPut(
                pair=pair,
                expiry=expiry,
                delivery_lag=delivery_lag,
                payment_lag=payment_lag,
                calendar=calendar,
                modifier=modifier,
                eom=eom,
                eval_date=eval_date,
                strike=strike[0],
                notional=notional_,
                option_fixings=option_fixings[0]
                if isinstance(option_fixings, tuple | list)
                else option_fixings,
                delta_type=delta_type,
                premium=premium[0],
                premium_ccy=premium_ccy,
                curves=curves,
                vol=vol_[0],
                metric=NoInput(0),
                spec=spec,
            ),
            FXCall(
                pair=pair,
                expiry=expiry,
                delivery_lag=delivery_lag,
                payment_lag=payment_lag,
                calendar=calendar,
                modifier=modifier,
                eom=eom,
                eval_date=eval_date,
                strike=strike[1],
                notional=notional_,
                option_fixings=option_fixings[1]
                if isinstance(option_fixings, tuple | list)
                else option_fixings,
                delta_type=delta_type,
                premium=premium[1],
                premium_ccy=premium_ccy,
                curves=curves,
                vol=vol_[1],
                metric=NoInput(0),
                spec=spec,
            ),
        ]
        super().__init__(
            options=options,
            rate_weight=[1.0, 1.0],
            rate_weight_vol=[0.5, 0.5],
            metric=_drb("single_vol", metric),
            curves=curves,
            vol=vol_,
        )
        self.kwargs.leg1["notional"] = notional_
        self.kwargs.meta["fixed_delta"] = [
            isinstance(strike[0], str)
            and strike[0][-1].lower() == "d"
            and strike[0].lower() != "atm_forward",
            isinstance(strike[1], str)
            and strike[1][-1].lower() == "d"
            and strike[1].lower() != "atm_forward",
        ]
        self.kwargs.leg1["delivery"] = self.instruments[0].kwargs.leg1["delivery"]
        self.kwargs.leg1["delta_type"] = self.instruments[0].kwargs.leg1["delta_type"]
        self.kwargs.leg1["expiry"] = self.instruments[0].kwargs.leg1["expiry"]

    @classmethod
    def _parse_vol(cls, vol: FXVolStrat_) -> tuple[_Vol, _Vol]:  # type: ignore[override]
        if not isinstance(vol, list | tuple):
            vol = (vol,) * 2
        return FXPut._parse_vol(vol[0]), FXCall._parse_vol(vol[1])

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
        return self._rate(
            curves=curves,
            solver=solver,
            fx=fx,
            base=base,
            vol=vol,
            metric=metric,
            forward=forward,
            settlement=settlement,
        )

    def _rate(
        self,
        curves: CurvesT_,
        solver: Solver_,
        fx: FXForwards_,
        base: str_,
        vol: FXVolStrat_,
        metric: str_,
        settlement: datetime_,
        forward: datetime_,
        record_greeks: bool = False,
    ) -> DualTypes:
        metric_: str = _drb(self.kwargs.meta["metric"], metric).lower()
        if metric_ != "single_vol" and not any(self.kwargs.meta["fixed_delta"]):
            # the strikes are explicitly defined and independent across options.
            # can evaluate separately, therefore the default method will suffice.
            return super().rate(
                curves=curves, solver=solver, fx=fx, base=base, vol=vol, metric=metric_
            )
        else:
            # must perform single vol evaluation to determine mkt convention strikes
            single_vol = self._rate_single_vol(
                curves=curves, solver=solver, fx=fx, base=base, vol=vol, record_greeks=record_greeks
            )
            if metric_ == "single_vol":
                return single_vol
            elif metric_ in ["premium", "pips_or_%"]:
                # return the premiums using the single_vol as the volatility
                return super().rate(
                    curves=curves, solver=solver, fx=fx, vol=single_vol, metric=metric_
                )
            elif metric_ == "vol":
                # this will return the same value as the single_vol, since the `vol` is
                # directly specified
                # return super().rate(
                #     curves=curves, solver=solver, fx=fx, vol=single_vol, metric=metric_
                # )
                return single_vol
            else:
                raise ValueError(
                    f"Metric {metric_} must be in {{'single_vol', 'premium', 'pips_or_%', 'vol'}}."
                )

    def _rate_single_vol(
        self,
        curves: CurvesT_,
        solver: Solver_,
        fx: FXForwards_,
        base: str_,
        vol: FXVolStrat_,
        record_greeks: bool,
    ) -> DualTypes:
        """
        Solve the single vol rate metric for a strangle using iterative market convergence routine.
        """
        # Get curves and vol
        _curves = self._parse_curves(curves)
        _vol = self._parse_vol(vol)
        fxf = _validate_fx_as_forwards(_get_fx_forwards_maybe_from_solver(solver=solver, fx=fx))
        rate_curve = _validate_obj_not_no_input(
            _maybe_get_curve_maybe_from_solver(
                curves_meta=self.kwargs.meta["curves"],
                curves=_curves,
                name="rate_curve",
                solver=solver,
            ),
            "rate_curve",
        )
        disc_curve = _validate_obj_not_no_input(
            _maybe_get_curve_maybe_from_solver(
                curves_meta=self.kwargs.meta["curves"],
                curves=_curves,
                name="disc_curve",
                solver=solver,
            ),
            "disc_curve",
        )
        vol_0: _FXVolOption = _validate_obj_not_no_input(  # type: ignore[assignment]
            _maybe_get_fx_vol_maybe_from_solver(
                vol_meta=self.kwargs.meta["vol"][0],
                vol=_vol[0],
                solver=solver,
            ),
            "`vol` at index [0]",
        )
        vol_1: _FXVolOption = _validate_obj_not_no_input(  # type: ignore[assignment]
            _maybe_get_fx_vol_maybe_from_solver(
                vol_meta=self.kwargs.meta["vol"][1],
                vol=_vol[1],
                solver=solver,
            ),
            "`vol` at index [1]",
        )

        # Get initial data from objects in their native AD order
        spot: datetime = fxf.pairs_settlement[self.kwargs.leg1["pair"].pair]
        w_spot: DualTypes = rate_curve[spot]
        w_deli: DualTypes = rate_curve[self.kwargs.leg1["delivery"]]
        f_d: DualTypes = fxf.rate(self.kwargs.leg1["pair"], self.kwargs.leg1["delivery"])
        f_t: DualTypes = fxf.rate(self.kwargs.leg1["pair"], spot)
        z_w_0 = (
            1.0
            if self.kwargs.leg1["delta_type"]
            in [FXDeltaMethod.ForwardPremiumAdjusted, FXDeltaMethod.Forward]
            else w_deli / w_spot
        )
        f_0 = (
            f_d
            if self.kwargs.leg1["delta_type"]
            in [FXDeltaMethod.ForwardPremiumAdjusted, FXDeltaMethod.Forward]
            else f_t
        )

        eta1 = None
        fzw1zw0: DualTypes = 0.0
        if isinstance(
            vol_0, FXDeltaVolSurface | FXDeltaVolSmile
        ):  # multiple Vol objects cannot be used, will derive conventions from the first one found.
            eta1 = (
                -0.5
                if vol_0.meta.delta_type
                in [FXDeltaMethod.ForwardPremiumAdjusted, FXDeltaMethod.SpotPremiumAdjusted]
                else 0.5
            )
            z_w_1 = (
                1.0
                if vol_0.meta.delta_type
                in [FXDeltaMethod.ForwardPremiumAdjusted, FXDeltaMethod.Forward]
                else w_deli / w_spot
            )
            fzw1zw0 = f_0 * z_w_1 / z_w_0

        # Determine the initial guess for Newton type iterations

        _ad = _set_ad_order_objects([0] * 5, [vol_0, vol_1, rate_curve, disc_curve, fxf])
        gks: list[dict[str, Any]] = [
            self.instruments[0]._analytic_greeks_reduced(
                curves=[rate_curve, disc_curve],
                solver=NoInput(0),
                fx=fxf,
                base=base,
                vol=vol_0,
            ),
            self.instruments[1]._analytic_greeks_reduced(
                curves=[rate_curve, disc_curve],
                solver=NoInput(0),
                fx=fxf,
                base=base,
                vol=vol_1,
            ),
        ]

        g0: DualTypes = gks[0]["__vol"] * gks[0]["vega"] + gks[1]["__vol"] * gks[1]["vega"]
        g0 /= gks[0]["vega"] + gks[1]["vega"]

        put_op_period: _BaseFXOptionPeriod = self.instruments[0]._option
        call_op_period: _BaseFXOptionPeriod = self.instruments[1]._option

        def root1d(
            tgt_vol: DualTypes, fzw1zw0: DualTypes, as_float: bool
        ) -> tuple[DualTypes, DualTypes]:
            if not as_float:
                # reset objects to their original order and perform final iterations
                _set_ad_order_objects(_ad, [vol_0, vol_1, rate_curve, disc_curve, fxf])

            # Determine the greeks of the options with the current tgt_vol iterate
            gks = [
                self.instruments[0]._analytic_greeks_reduced(
                    curves=[rate_curve, disc_curve],
                    solver=NoInput(0),
                    fx=fxf,
                    base=base,
                    vol=tgt_vol * 100.0,
                ),
                self.instruments[1]._analytic_greeks_reduced(
                    curves=[rate_curve, disc_curve],
                    solver=NoInput(0),
                    fx=fxf,
                    base=base,
                    vol=tgt_vol * 100.0,
                ),
            ]

            # Also determine the greeks of these options measured with the market smile vol.
            # (note the strikes have been set by previous call, call OptionPeriods direct
            # to avoid re-determination)
            s_gks = [
                put_op_period._base_analytic_greeks(
                    rate_curve=rate_curve,
                    disc_curve=disc_curve,
                    fx=fxf,
                    fx_vol=vol_0,
                    _reduced=True,
                ),
                call_op_period._base_analytic_greeks(
                    rate_curve=rate_curve,
                    disc_curve=disc_curve,
                    fx=fxf,
                    fx_vol=vol_1,
                    _reduced=True,
                ),
            ]

            # The value of the root function is derived from the 4 previous calculated prices
            f0 = s_gks[0]["__bs76"] + s_gks[1]["__bs76"] - gks[0]["__bs76"] - gks[1]["__bs76"]

            dc1_dvol1_0 = _d_c_hat_d_sigma_hat(gks[0], self.kwargs.meta["fixed_delta"][0])
            dcmkt_dvol1_0 = _d_c_mkt_d_sigma_hat(
                gks[0],
                s_gks[0],
                self.kwargs.leg1["expiry"],
                vol_0,
                eta1,
                self.kwargs.meta["fixed_delta"][0],
                fzw1zw0,
                fxf,
            )
            dc1_dvol1_1 = _d_c_hat_d_sigma_hat(gks[1], self.kwargs.meta["fixed_delta"][1])
            dcmkt_dvol1_1 = _d_c_mkt_d_sigma_hat(
                gks[1],
                s_gks[1],
                self.kwargs.leg1["expiry"],
                vol_1,
                eta1,
                self.kwargs.meta["fixed_delta"][1],
                fzw1zw0,
                fxf,
            )
            f1 = dcmkt_dvol1_0 + dcmkt_dvol1_1 - dc1_dvol1_0 - dc1_dvol1_1

            return f0, f1

        root_solver = newton_1dim(
            root1d,
            g0,
            args=(fzw1zw0,),
            pre_args=(True,),  # solve `as_float` in iterations
            final_args=(False,),  # capture AD in final iterations
            raise_on_fail=True,
            max_iter=10,
            func_tol=1e-6,
        )
        tgt_vol: DualTypes = root_solver["g"] * 100.0

        if record_greeks:  # this needs to be explicitly called since it degrades performance
            self._greeks["strangle"] = {
                "single_vol": {
                    "FXPut": self.instruments[0].analytic_greeks(curves, solver, fxf, tgt_vol),
                    "FXCall": self.instruments[1].analytic_greeks(curves, solver, fxf, tgt_vol),
                },
                "market_vol": {
                    "FXPut": put_op_period.analytic_greeks(rate_curve, disc_curve, fxf, vol_0),
                    "FXCall": call_op_period.analytic_greeks(rate_curve, disc_curve, fxf, vol_1),
                },
            }

        return tgt_vol


# Calculations related to Strange:single_vol


def _d_c_hat_d_sigma_hat(
    g: dict[str, Any],  # greeks
    fixed_delta: bool,
) -> DualTypes:
    """
    Return the total derivative of option priced with single vol with respect to single
    vol.

    Parameters
    ----------
    g: dict
        The dict of greeks for the given option period measured against the tgt, single vol.
    fixed_delta: bool
        Whether the given FXOption is defined by fixed delta or an explicit strike.

    Returns
    -------
    DualTypes
    """
    if not fixed_delta:
        # kega is 0.0
        return g["vega"]  # type: ignore[no-any-return]
    else:
        return g["_kappa"] * g["_kega"] + g["vega"]  # type: ignore[no-any-return]


def _d_c_mkt_d_sigma_hat(
    g: dict[str, Any],  # greeks
    sg: dict[str, Any],  # smile_greeks
    expiry: datetime,
    vol: _FXVolOption,
    eta1: float | None,
    fixed_delta: bool,
    fzw1zw0: DualTypes | None,
    fxf: FXForwards,
) -> DualTypes:
    """
    Return the total derivative of option priced with mkt vol with respect to single
    vol.

    Parameters
    ----------
    g: dict
        The dict of greeks for the given option period measured against the tgt, single vol.
    sg: dict
        The dict of greeks for the given option period measured against the smile.
    expiry: datetime
        The expiry of the Option.
    vol: VolObj
        The smile object.
    eta1: float | None
        The delta type of the Smile if available
    fixed_delta: bool
        Whether the option is defined by fixed delta or an explicit strike.
    fxf: FXForwards,
        Used by SabrSurface to forecast multiple forward rates for cross-sectional smiles before
        interpolation.

    Returns
    -------
    DualTypes
    """
    if not fixed_delta:
        return 0.0  # kega is zero and the mkt vol has no sensitivity to vol_hat.
    else:
        if isinstance(vol, FXDeltaVolSurface | FXDeltaVolSmile):
            if isinstance(vol, FXDeltaVolSurface):
                vol = vol.get_smile(expiry)

            dvol_ddeltaidx = evaluate(vol.nodes.spline.spline, sg["_delta_index"], 1) * 0.01

            ddeltaidx_dvol1 = sg["gamma"] * fzw1zw0
            if eta1 < 0:  # type: ignore[operator]
                # premium adjusted vol smile
                ddeltaidx_dvol1 += sg["_delta_index"]
            ddeltaidx_dvol1 *= g["_kega"] / sg["__strike"]

            _ = dual_log(sg["__strike"] / sg["__forward"]) / sg["__vol"]
            _ += eta1 * sg["__vol"] * sg["__sqrt_t"] ** 2
            _ *= dvol_ddeltaidx * sg["gamma"] * fzw1zw0
            ddeltaidx_dvol1 /= 1 + _

            dvol_dvol1: DualTypes = dvol_ddeltaidx * ddeltaidx_dvol1
        elif isinstance(vol, FXSabrSmile):
            dvol_dk = vol._d_sabr_d_k_or_f(
                k=sg["__strike"],
                f=sg["__forward"],
                expiry=expiry,
                as_float=False,
                derivative=1,
            )[1]
            dvol_dvol1 = dvol_dk * g["_kega"]
        elif isinstance(vol, FXSabrSurface):
            dvol_dk = vol._d_sabr_d_k_or_f(
                k=sg["__strike"],
                f=fxf,
                expiry=expiry,
                as_float=False,
                derivative=1,
            )[1]
            dvol_dvol1 = dvol_dk * g["_kega"]
        else:
            dvol_dvol1 = 0.0

        return sg["_kappa"] * g["_kega"] + sg["vega"] * dvol_dvol1  # type: ignore[no-any-return]
