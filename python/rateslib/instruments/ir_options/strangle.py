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
from rateslib.enums.parameters import IROptionMetric
from rateslib.instruments.ir_options.call_put import IRSCall, IRSPut
from rateslib.instruments.ir_options.straddle import _BaseIRSOptionStrat

if TYPE_CHECKING:
    from rateslib.local_types import (  # pragma: no cover
        CurvesT_,
        DualTypes,
        DualTypes_,
        IRSSeries,
        SwaptionSettlementMethod,
        VolStrat_,
        VolT_,
        _Vol,
        datetime,
        datetime_,
        str_,
    )


class IRSStrangle(_BaseIRSOptionStrat):
    """
    An *IR Strangle* :class:`~rateslib.instruments._BaseIRSOptionStrat`.

    .. warning::

       *Swaptions* and *IR Volatility* are in Beta status introduced in v2.7.0

    A *Strangle* is composed of a lower strike :class:`~rateslib.instruments.IRSPut`
    and a higher strike :class:`~rateslib.instruments.IRSCall` with the same expiry and tenor.

    .. rubric:: Examples

    .. ipython:: python
       :suppress:

       from rateslib import IRSStrangle, Curve, dt

    .. ipython:: python

       irstr = IRSStrangle(
           eval_date=dt(2020, 1, 1),
           expiry="3m",
           tenor="1Y",
           strike=("-20bps", "+20bps"),
           irs_series="usd_irs",
           notional=1000000,
       )
       irstr.cashflows()

    .. rubric:: Pricing

    The pricing mirrors that for an :class:`~rateslib.instruments.IRSCall`. All options use the
    same ``curves``. Allowable inputs are:

    .. code-block:: python

       curves = rate_curve | [rate_curve] #  one curve is used as all curves
       curves = [rate_curve, disc_curve]  #  two curves are applied in the given order, index_curve is set equal to disc_curve
       curves = [rate_curve, disc_curve, index_curve]  # three curves applied in the given order
       curves = {
           "rate_curve": rate_curve,
           "disc_curve": disc_curve
           "index_curve": index_curve
       }  # dict form is explicit

    A ``vol`` argument must be provided to each *Instrument*. This can either be a single
    value universally used for all, or an individual item as part of a sequence. Allowed
    inputs are:

    .. code-block:: python

       vol = 12.0 | vol_obj  # a single item universally applied
       vol = [12.0, 12.0]  # values for the Put and Call respectively

    The following pricing ``metric`` are available, with examples:

    TODO

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
    strike: 2-tuple of float, Variable, str, :red:`required`
        The strike values of each option.
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

    """  # noqa: E501

    _rate_scalar = 100.0

    def __init__(
        self,
        expiry: datetime | str,
        tenor: datetime | str,
        strike: tuple[DualTypes | str, DualTypes | str],
        irs_series: IRSSeries | str,
        *,
        notional: DualTypes_ = NoInput(0),
        eval_date: datetime | NoInput = NoInput(0),
        premium: tuple[DualTypes_, DualTypes_] = (NoInput(0), NoInput(0)),
        payment_lag: str | datetime_ = NoInput(0),
        option_fixings: DualTypes_ = NoInput(0),
        settlement_method: SwaptionSettlementMethod | str_ = NoInput(0),
        metric: IROptionMetric | str_ = NoInput(0),
        curves: CurvesT_ = NoInput(0),
        vol: VolT_ = NoInput(0),
        spec: str_ = NoInput(0),
    ) -> None:
        vol_ = self._parse_vol(vol)
        notional_ = _drb(defaults.notional, notional)
        options = [
            IRSPut(
                irs_series=irs_series,
                expiry=expiry,
                payment_lag=payment_lag,
                eval_date=eval_date,
                tenor=tenor,
                strike=strike[0],
                notional=notional_,
                option_fixings=option_fixings[0]
                if isinstance(option_fixings, tuple | list)
                else option_fixings,
                settlement_method=settlement_method,
                premium=premium[0],
                curves=curves,
                vol=vol_[0],
                metric=NoInput(0),
                spec=spec,
            ),
            IRSCall(
                irs_series=irs_series,
                expiry=expiry,
                payment_lag=payment_lag,
                eval_date=eval_date,
                tenor=tenor,
                strike=strike[1],
                notional=notional_,
                option_fixings=option_fixings[1]
                if isinstance(option_fixings, tuple | list)
                else option_fixings,
                settlement_method=settlement_method,
                premium=premium[1],
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

    @classmethod
    def _parse_vol(cls, vol: VolStrat_) -> tuple[_Vol, _Vol]:  # type: ignore[override]
        if not isinstance(vol, list | tuple):
            vol = (vol,) * 2
        return IRSPut._parse_vol(vol[0]), IRSCall._parse_vol(vol[1])

    def _set_notionals(self, notional: DualTypes) -> None:
        """
        Set the notionals on each option period. Mainly used by Brokerfly for vega neutral
        strangle and straddle.
        """
        for option in self.instruments:
            option.kwargs.leg1["notional"] = notional
            option._option.settlement_params._notional = notional
