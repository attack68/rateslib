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
from rateslib.enums.generics import NoInput, _drb
from rateslib.instruments.fx_options.call_put import FXCall, FXPut
from rateslib.instruments.fx_options.risk_reversal import _BaseFXOptionStrat

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        CalInput,
        CurvesT_,
        DualTypes,
        DualTypes_,
        FXVolStrat_,
        VolT_,
        _Vol,
        bool_,
        datetime,
        datetime_,
        int_,
        str_,
    )


class FXStraddle(_BaseFXOptionStrat):
    """
    An *FX Straddle* :class:`~rateslib.instruments._BaseFXOptionStrat`.

    A *Straddle* is composed of a :class:`~rateslib.instruments.FXPut`
    and :class:`~rateslib.instruments.FXCall` with the same strike.

    .. rubric:: Examples

    .. ipython:: python
       :suppress:

       from rateslib import FXStraddle, FXForwards, FXRates, FXDeltaVolSmile, Curve, dt

    .. ipython:: python

       fxs = FXStraddle(
           expiry="3m",
           strike=1.10,  # <- "atm_delta" is also a common input
           eval_date=dt(2020, 1, 1),
           spec="eurusd_call",
           notional=1000000,
       )
       fxs.cashflows()

    .. rubric:: Pricing

    The pricing mirrors that for an :class:`~rateslib.instruments.FXCall`. All options use the
    same ``curves``. Allowable inputs are:

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

    - **'vol'**: the implied volatility value of the straddle from a volatility object.
      **'single_vol'** is also an alias for this, since both options assume the same volatility.

      .. ipython:: python

         fxs.rate(vol=fxvs, curves=[eur, usd], fx=fxf, metric="vol")

    - **'premium'**: the summed cash premium amount, of both options, applicable to the 'payment'
      date.

      .. ipython:: python

         fxs.rate(vol=fxvs, curves=[eur, usd], fx=fxf, metric="premium")
         fxs.instruments[0].rate(vol=fxvs, curves=[eur, usd], fx=fxf, metric="premium")
         fxs.instruments[1].rate(vol=fxvs, curves=[eur, usd], fx=fxf, metric="premium")

    - **'pips_or_%'**: if the premium currency is LHS of ``pair`` this is a % of notional, whilst if
      the premium currency is RHS this gives a number of pips of the FX rate. Summed over both
      options.

      .. ipython:: python

         fxs.rate(vol=fxvs, curves=[eur, usd], fx=fxf, metric="pips_or_%")
         fxs.instruments[0].rate(vol=fxvs, curves=[eur, usd], fx=fxf, metric="pips_or_%")
         fxs.instruments[1].rate(vol=fxvs, curves=[eur, usd], fx=fxf, metric="pips_or_%")

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
    strike: float, Variable, str, :red:`required`
        The strike of the put and the call.
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

    """

    _rate_scalar = 100.0

    def __init__(
        self,
        expiry: datetime | str,
        strike: DualTypes | str,
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
                strike=strike,
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
                strike=strike,
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
            metric=metric,
            curves=curves,
            vol=vol_,
        )
        self.kwargs.leg1["notional"] = notional_
        self.kwargs.leg2["premium_ccy"] = self.instruments[0].kwargs.leg2["premium_ccy"]

    @classmethod
    def _parse_vol(cls, vol: FXVolStrat_) -> tuple[_Vol, _Vol]:  # type: ignore[override]
        if not isinstance(vol, list | tuple):
            vol = (vol,) * 2
        return FXPut._parse_vol(vol[0]), FXCall._parse_vol(vol[1])

    def _set_notionals(self, notional: DualTypes) -> None:
        """
        Set the notionals on each option period. Mainly used by Brokerfly for vega neutral
        strangle and straddle.
        """
        for option in self.instruments:
            option.kwargs.leg1["notional"] = notional
            option._option.settlement_params._notional = notional
