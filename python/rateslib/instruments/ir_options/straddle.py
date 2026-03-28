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

from pandas import DataFrame

from rateslib import defaults
from rateslib.enums.generics import NoInput, _drb
from rateslib.enums.parameters import IROptionMetric, _get_ir_option_metric
from rateslib.instruments.ir_options.call_put import IRSCall, IRSPut, _BaseIRSOption
from rateslib.instruments.protocols import _KWArgs

if TYPE_CHECKING:
    from rateslib.local_types import (  # pragma: no cover
        Any,
        CurvesT_,
        DualTypes,
        DualTypes_,
        FXForwards_,
        IRSSeries,
        Sequence,
        Solver_,
        SwaptionSettlementMethod,
        VolStrat_,
        VolT_,
        _Vol,
        datetime,
        datetime_,
        str_,
    )


class _BaseIRSOptionStrat(_BaseIRSOption):
    """
    A custom option strategy composed of a list of :class:`~rateslib.instruments._BaseIRSOption`,
    or other :class:`~rateslib.instruments._BaseIRSOptionStrat` objects, of the same
    :class:`~rateslib.data.fixings.IRSSeries`.

    .. warning::

       *Swaptions* and *IR Volatility* are in Beta status introduced in v2.7.0

    Parameters
    ----------
    options: list
        The *IROptions* or *IROptionStrats* which make up the strategy.
    rate_weight: list
        The multiplier for non-vol type metrics that sums the options to a final *rate*.
        E.g. A *RiskReversal* uses [-1.0, 1.0] for a sale and a purchase.
        E.g. A *Straddle* uses [1.0, 1.0] for summing two premium purchases.
    rate_weight_vol: list
        The multiplier for the *'vol'* metric that sums the options to a final *rate*.
        E.g. A *RiskReversal* uses [-1.0, 1.0] to obtain the vol difference between two options.
        E.g. A *Straddle* uses [0.5, 0.5] to obtain the volatility at the strike of each option.
    """

    _greeks: dict[str, Any] = {}
    _strat_elements: tuple[_BaseIRSOption | _BaseIRSOptionStrat, ...]

    @property
    def kwargs(self) -> _KWArgs:
        """The :class:`~rateslib.instruments.protocols._KWArgs` of the *Instrument*."""
        return self._kwargs

    def __init__(
        self,
        options: Sequence[_BaseIRSOption | _BaseIRSOptionStrat],
        rate_weight: list[float],
        rate_weight_vol: list[float],
        metric: IROptionMetric | str_ = NoInput(0),
        curves: CurvesT_ = NoInput(0),
        vol: VolStrat_ = NoInput(0),
    ):
        self._n = len(options)
        if self._n != len(rate_weight) or self._n != len(rate_weight_vol):
            raise ValueError(
                "`rate_weight` and `rate_weight_vol` must have same length as `options`.",
            )
        self._kwargs = _KWArgs(
            spec=NoInput(0),
            user_args=dict(
                rate_weight=rate_weight,
                rate_weight_vol=rate_weight_vol,
                instruments=tuple(options),
                metric=metric,
                irs_series=options[0].kwargs.leg1["irs_series"],
                curves=NoInput(0),
                vol=vol,
            ),
            default_args=dict(
                metric=defaults.ir_option_metric,
            ),
            meta_args=["metric", "vol", "curves", "instruments", "rate_weight", "rate_weight_vol"],
        )
        self.kwargs.meta["curves"] = self._parse_curves(curves)

    @classmethod
    def _parse_vol(cls, vol: VolStrat_) -> VolStrat_:  # type: ignore[override]
        raise NotImplementedError(f"{type(cls).__name__} must implement `_parse_vol`.")

    @property
    def instruments(self) -> tuple[_BaseIRSOption | _BaseIRSOptionStrat, ...]:
        return self.kwargs.meta["instruments"]  # type: ignore[no-any-return]

    def __repr__(self) -> str:
        return f"<rl.{type(self).__name__} at {hex(id(self))}>"

    def rate(
        self,
        *,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: VolStrat_ = NoInput(0),
        base: str_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
        metric: IROptionMetric | str_ = NoInput(0),
    ) -> DualTypes:
        vol_: VolStrat_ = self._parse_vol(vol)
        metric_: IROptionMetric = _get_ir_option_metric(_drb(self.kwargs.meta["metric"], metric))
        match type(metric_):
            case IROptionMetric.NormalVol | IROptionMetric.BlackVolShift:
                weights = self.kwargs.meta["rate_weight_vol"]
            case IROptionMetric.Premium | IROptionMetric.PercentNotional:
                weights = self.kwargs.meta["rate_weight"]

        _: DualTypes = 0.0
        for option, vol__, weight in zip(self.instruments, vol_, weights, strict=True):  # type: ignore[misc, arg-type]
            _ += (
                option.rate(
                    curves=curves,
                    solver=solver,
                    fx=fx,
                    base=base,
                    vol=vol__,  # type: ignore[arg-type]
                    metric=metric_,
                    settlement=settlement,
                    forward=forward,
                )
                * weight
            )
        return _

    def npv(
        self,
        *,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: VolStrat_ = NoInput(0),
        base: str_ = NoInput(0),
        local: bool = False,
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> DualTypes | dict[str, DualTypes]:
        vol_ = self._parse_vol(vol)

        results = [
            option.npv(
                curves=curves,
                solver=solver,
                fx=fx,
                base=base,
                local=local,
                vol=vol__,  # type: ignore[arg-type]
                forward=forward,
                settlement=settlement,
            )
            for (option, vol__) in zip(self.instruments, vol_, strict=True)  # type: ignore[arg-type]
        ]

        if local:
            df = DataFrame(results).fillna(0.0)
            df_sum = df.sum()
            _: DualTypes | dict[str, DualTypes] = df_sum.to_dict()  # type: ignore[assignment]
        else:
            _ = sum(results)  # type: ignore[arg-type]
        return _

    def cashflows(
        self,
        *,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: VolStrat_ = NoInput(0),
        base: str_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> DataFrame:
        return self._cashflows_from_instruments(
            curves=curves,
            solver=solver,
            fx=fx,
            vol=vol,
            settlement=settlement,
            forward=forward,
            base=base,
        )

    def _plot_payoff(
        self,
        window: tuple[float, float] | NoInput = NoInput(0),
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: VolStrat_ = NoInput(0),
    ) -> tuple[Any, Any]:
        vol_ = self._parse_vol(vol)

        y = None
        for inst, vol__ in zip(self.instruments, vol_, strict=True):  # type: ignore[arg-type]
            x, y_ = inst._plot_payoff(
                window=window,
                curves=curves,
                solver=solver,
                fx=fx,
                vol=vol__,  # type: ignore[arg-type]
            )
            if y is None:
                y = y_
            else:
                y += y_

        return x, y

    def analytic_greeks(
        self,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: VolStrat_ = NoInput(0),
    ) -> dict[str, Any]:
        # implicitly call set_pricing_mid for unpriced parameters
        # this may be important for Strategies whose options are
        # dependent upon each other, (RR and Straddle do not have interdependent options)
        self.rate(curves=curves, solver=solver, fx=fx, vol=vol)

        vol_: VolStrat_ = self._parse_vol(vol=vol)
        gks = []
        for inst, vol_i in zip(self.instruments, vol_, strict=True):  # type: ignore[misc, arg-type]
            if isinstance(inst, _BaseIRSOptionStrat):
                gks.append(
                    inst.analytic_greeks(
                        curves=curves,
                        solver=solver,
                        fx=fx,
                        vol=vol_i,
                    )
                )
            else:  # option is _BaseIRSOption
                gks.append(
                    inst._analytic_greeks_set_metrics(
                        curves=curves,
                        solver=solver,
                        fx=fx,
                        vol=vol_i,  # type: ignore[arg-type]
                        set_metrics=False,  # already done in the rate call above
                    )
                )

        _unit_attrs = ["delta", "gamma", "vega", "vomma", "vanna", "__bs76", "__bachelier"]
        _: dict[str, Any] = {}
        for attr in _unit_attrs:
            tally = 0.0
            for i, gk in enumerate(gks):
                if attr not in gk:
                    continue
                tally += gk[attr] * self.kwargs.meta["rate_weight"][i]
            _[attr] = tally

        _notional_attrs = [
            f"delta_{self.settlement_params.currency}",
            f"gamma_{self.settlement_params.currency}",
            f"vega_{self.settlement_params.currency}",
        ]
        for attr in _notional_attrs:
            _[attr] = sum(gk[attr] * self.kwargs.meta["rate_weight"][i] for i, gk in enumerate(gks))

        _.update(
            {
                "__class": "IROptionStrat",
                "__options": gks,
                "__notional": self.kwargs.leg1["notional"],
            },
        )
        return _


class IRSStraddle(_BaseIRSOptionStrat):
    """
    An *IR Straddle* :class:`~rateslib.instruments._BaseIRSOptionStrat`.

    .. warning::

       *Swaptions* and *IR Volatility* are in Beta status introduced in v2.7.0

    A *Straddle* is composed of a :class:`~rateslib.instruments.IRSPut`
    and :class:`~rateslib.instruments.IRSCall` with the same strike, expiry and tenor.

    .. rubric:: Examples

    .. ipython:: python
       :suppress:

       from rateslib import IRSStraddle, Curve, dt

    .. ipython:: python

       irstr = IRSStraddle(
           eval_date=dt(2020, 1, 1),
           expiry="3m",
           tenor="1Y",
           strike="atm",
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
        strike: DualTypes | str,
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
                strike=strike,
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
                strike=strike,
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
