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

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING

import rateslib.errors as err
from rateslib import defaults
from rateslib.curves._parsers import (
    _disc_required_maybe_from_curve,
    _validate_curve_not_no_input,
)
from rateslib.data.fixings import _get_irs_series
from rateslib.dual import ift_1dim
from rateslib.dual.utils import _dual_float
from rateslib.enums.generics import Err, NoInput, Ok, _drb
from rateslib.enums.parameters import (
    IROptionMetric,
    OptionType,
    SwaptionSettlementMethod,
    _get_ir_option_metric,
)
from rateslib.instruments.protocols.pricing import _Curves
from rateslib.periods.parameters import (
    _IndexParams,
    _IROptionParams,
    _NonDeliverableParams,
    _SettlementParams,
)
from rateslib.periods.protocols import _BasePeriodStatic, _WithAnalyticIROptionGreeks
from rateslib.periods.utils import (
    _get_ir_vol_value_and_forward_maybe_from_obj,
)
from rateslib.volatility.ir.utils import _IRVolPricingParams
from rateslib.volatility.utils import (
    _OptionModelBachelier,
    _OptionModelBlack76,
)

if TYPE_CHECKING:
    from rateslib.local_types import (  # pragma: no cover
        Any,
        CurveOption,
        CurveOption_,
        DualTypes,
        DualTypes_,
        FXForwards_,
        IRSSeries,
        Result,
        Series,
        _BaseCurve,
        _BaseCurve_,
        _IRVolOption_,
        datetime,
        datetime_,
        str_,
    )


class _BaseIRSOptionPeriod(_BasePeriodStatic, _WithAnalyticIROptionGreeks, metaclass=ABCMeta):
    r"""
    Abstract base class for *IROptionPeriods* types.

    **See Also**: :class:`~rateslib.periods.IRCallPeriod`,
    :class:`~rateslib.periods.IRPutPeriod`

    """

    def analytic_greeks(
        self,
        rate_curve: CurveOption,
        disc_curve: _BaseCurve,
        index_curve: _BaseCurve,
        fx: FXForwards_ = NoInput(0),
        ir_vol: _IRVolOption_ = NoInput(0),
        premium: DualTypes_ = NoInput(0),  # expressed in the payment currency
        premium_payment: datetime_ = NoInput(0),
    ) -> dict[str, Any]:
        return super()._base_analytic_greeks(
            rate_curve=rate_curve,
            disc_curve=disc_curve,
            index_curve=index_curve,
            ir_vol=ir_vol,
            fx=fx,
            premium=premium,
            premium_payment=premium_payment,
        )

    @property
    def period_params(self) -> None:
        """This *Period* type has no
        :class:`~rateslib.periods.parameters._PeriodParams`."""
        return self._period_params  # type: ignore[return-value]  # pragma: no cover

    @property
    def settlement_params(self) -> _SettlementParams:
        """The :class:`~rateslib.periods.parameters._SettlementParams`
        of the *Period*."""
        return self._settlement_params

    @property
    def index_params(self) -> _IndexParams | None:
        """The :class:`~rateslib.periods.parameters._IndexParams` of
        the *Period*, if any."""
        return self._index_params

    @property
    def non_deliverable_params(self) -> _NonDeliverableParams | None:
        """The :class:`~rateslib.periods.parameters._NonDeliverableParams` of the
        *Period*., if any."""
        return self._non_deliverable_params

    @property
    def rate_params(self) -> None:
        """This *Period* type has no rate parameters."""
        return self._rate_params  # type: ignore[return-value]  # pragma: no cover

    @property
    def ir_option_params(self) -> _IROptionParams:
        """The :class:`~rateslib.periods.parameters._IROptionParams` of the
        *Period*."""
        return self._ir_option_params

    @abstractmethod
    def __init__(
        self,
        *,
        # option params:
        direction: OptionType,
        expiry: datetime,
        tenor: datetime | str,
        irs_series: IRSSeries | str,
        strike: DualTypes_ = NoInput(0),
        notional: DualTypes_ = NoInput(0),
        metric: IROptionMetric | str_ = NoInput(0),
        option_fixings: DualTypes | Series[DualTypes] | str_ = NoInput(0),  # type: ignore[type-var]
        # currency args:
        settlement_method: SwaptionSettlementMethod | str_ = NoInput(0),
        ex_dividend: datetime_ = NoInput(0),
    ) -> None:
        self._index_params = None
        self._rate_params = None
        self._period_params = None

        self._ir_option_params = _IROptionParams(
            _direction=direction,
            _expiry=expiry,
            _tenor=tenor,
            _irs_series=_get_irs_series(irs_series),
            _strike=strike,
            _metric=_drb(defaults.ir_option_metric, metric),
            _option_fixings=option_fixings,
            _settlement_method=_drb(defaults.ir_option_settlement, settlement_method),
        )

        nd_pair = NoInput(0)
        if isinstance(nd_pair, NoInput):
            # then option is directly deliverable
            self._non_deliverable_params: _NonDeliverableParams | None = None
            self._settlement_params = _SettlementParams(
                _notional=_drb(defaults.notional, notional),
                _payment=self.ir_option_params.option_fixing.effective,
                _currency=self.ir_option_params.option_fixing.irs_series.currency,
                _notional_currency=self.ir_option_params.option_fixing.irs_series.currency,
                _ex_dividend=ex_dividend,
            )
        else:
            raise NotImplementedError("ND IR Options not implement")  # pragma: no cover

    def __repr__(self) -> str:
        return f"<rl.{type(self).__name__} at {hex(id(self))}>"

    def _unindexed_reference_cashflow_elements(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        ir_vol: _IRVolOption_ = NoInput(0),
    ) -> tuple[DualTypes, DualTypes | None, _IRVolPricingParams | None]:
        """
        Perform the unindexed_reference_cashflow calculations but return calculation
        components.

        Returns
        -------
        (cashflow, analytic_delta, pricing params)
        """
        # The unindexed_reference_cashflow is the value of the IRS after expiry.
        # This may be based on number numerous different settlement methods: physical / cash etc.
        # Currently we only offer 1 form of valuation which is "physical or physical simulation".
        if isinstance(self.ir_option_params.strike, NoInput):
            raise ValueError(err.VE_NEEDS_STRIKE)
        k = self.ir_option_params.strike
        r = self.ir_option_params.option_fixing.value
        if not isinstance(r, NoInput):
            # the presence of fixing value here is used purely as an indicator of exercise status.

            phi: OptionType = self.ir_option_params.direction

            if (phi == OptionType.Call and k < r) or (phi == OptionType.Put and k > r):
                if self.ir_option_params.settlement_method is SwaptionSettlementMethod.Physical:
                    local_npv_pay_dt: DualTypes = self.ir_option_params.option_fixing.irs.npv(  # type: ignore[assignment]
                        curves=_Curves(
                            disc_curve=index_curve,
                            leg2_rate_curve=rate_curve,
                            leg2_disc_curve=index_curve,
                        ),
                        forward=self.settlement_params.payment,
                        local=False,
                    )
                    value = (
                        local_npv_pay_dt
                        * self.settlement_params.notional
                        / 1e6
                        * self.ir_option_params.direction.value
                    )
                    return value, None, None
                else:
                    # in [
                    #   SwaptionSettlementMethod.CashParTenor,
                    #   SwaptionSettlementMethod.CashCollateralized
                    # ]
                    index_curve_ = _validate_curve_not_no_input(index_curve)
                    del index_curve
                    a_r = self.ir_option_params.option_fixing.annuity(
                        settlement_method=self.ir_option_params.settlement_method,
                        rate_curve=rate_curve,
                        index_curve=index_curve_,
                    )
                    value = (
                        (r - k)
                        * 100.0
                        * a_r
                        * self.settlement_params.notional
                        / 1e6
                        * self.ir_option_params.direction.value
                    )
                    return value, None, None
            else:
                # no exercise
                return 0.0, None, None

        else:
            disc_curve_ = _disc_required_maybe_from_curve(curve=rate_curve, disc_curve=disc_curve)
            del disc_curve
            index_curve_ = _validate_curve_not_no_input(index_curve)
            del index_curve

            pricing_ = _get_ir_vol_value_and_forward_maybe_from_obj(
                ir_vol=ir_vol,
                index_curve=index_curve_,
                rate_curve=rate_curve,
                strike=k,
                irs=self.ir_option_params.option_fixing.irs,
                expiry=self.ir_option_params.expiry,
                tenor=self.ir_option_params.option_fixing.termination,
            )

            t_e = self.ir_option_params.time_to_expiry(disc_curve_.nodes.initial)  # time to expiry
            expected = (
                _OptionModelBlack76._value(
                    F=pricing_.f + pricing_.shift,
                    K=pricing_.k + pricing_.shift,
                    t_e=t_e,
                    v2=1.0,  # not required
                    vol=pricing_.vol / 100.0,
                    phi=self.ir_option_params.direction.value,  # controls calls or put price
                )
                * 100.0
            )  # bps
            a_r = self.ir_option_params.option_fixing.annuity(
                settlement_method=self.ir_option_params.settlement_method,
                rate_curve=rate_curve,
                index_curve=index_curve_,
            )
            return (
                expected * self.settlement_params.notional / 1e6 * a_r,
                a_r,
                pricing_,
            )

    def unindexed_reference_cashflow(  # type: ignore[override]
        self,
        *,
        rate_curve: _BaseCurve_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        ir_vol: _IRVolOption_ = NoInput(0),
        **kwargs: Any,
    ) -> DualTypes:
        return self._unindexed_reference_cashflow_elements(
            rate_curve=rate_curve,
            disc_curve=disc_curve,
            index_curve=index_curve,
            ir_vol=ir_vol,
        )[0]

    def try_rate(
        self,
        rate_curve: CurveOption_,
        disc_curve: _BaseCurve,
        index_curve: _BaseCurve,
        fx: FXForwards_ = NoInput(0),
        ir_vol: _IRVolOption_ = NoInput(0),
        metric: IROptionMetric | str_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> Result[DualTypes]:
        """
        Return the pricing metric of the *FXOption*, with lazy error handling.

        See :meth:`~rateslib.periods.FXOptionPeriod.rate`.
        """
        try:
            return Ok(
                self.rate(
                    rate_curve=rate_curve,
                    disc_curve=disc_curve,
                    index_curve=index_curve,
                    fx=fx,
                    ir_vol=ir_vol,
                    metric=metric,
                    forward=forward,
                )
            )
        except Exception as e:
            return Err(e)

    def rate(
        self,
        *,
        rate_curve: CurveOption_,
        disc_curve: _BaseCurve,
        index_curve: _BaseCurve,
        fx: FXForwards_ = NoInput(0),
        ir_vol: _IRVolOption_ = NoInput(0),
        metric: IROptionMetric | str_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> DualTypes:
        """
        Return the pricing metric of the *IRSOption*.

        This is priced according to the ``payment`` date of the *OptionPeriod*.

        Parameters
        ----------
        rate_curve: Curve
            The curve used for forecasting rates on the underlying
            :class:`~rateslib.instruments.IRS`.
        disc_curve: Curve
            The discount *Curve* according to the collateral agreement of the option.
        index_curve: Curve
            The curve used for price alignment indexing according to the
            :class:`~rateslib.enums.SwaptionSettlementMethod`. I.e. the discount curve used on the
            underlying :class:`~rateslib.instruments.IRS`.
        fx: float, FXRates, FXForwards, optional
            The object to project the currency pair FX rate at delivery.
        ir_vol: IRSabrSmile, float, Dual, Dual2
            The volatility object to price the option. If given as numeric, it is assumed to be
            Black (log-normal) volatility with zero shift.
        metric: IROptionMetric,
            The metric to return. See examples.
        forward: datetime, optional (set as payment date of option)
            Not currently used by IRSOptionPeriod.rate.

        Returns
        -------
        float, Dual, Dual2 or dict of such.
        """
        if not isinstance(metric, NoInput):
            metric_ = _get_ir_option_metric(metric)
        else:  # use metric associated with self
            metric_ = self.ir_option_params.metric
        del metric

        cash, anal_delta, pricing = self._unindexed_reference_cashflow_elements(
            rate_curve=rate_curve,
            disc_curve=disc_curve,
            index_curve=index_curve,
            ir_vol=ir_vol,
        )

        if metric_ == IROptionMetric.Cash:
            return cash
        elif metric_ == IROptionMetric.PercentNotional:
            return cash / self.settlement_params.notional * 100.0

        disc_curve_ = _disc_required_maybe_from_curve(curve=rate_curve, disc_curve=disc_curve)
        del disc_curve

        if pricing is None:
            pricing_ = _get_ir_vol_value_and_forward_maybe_from_obj(
                ir_vol=ir_vol,
                index_curve=index_curve,
                rate_curve=rate_curve,
                strike=self.ir_option_params.strike,  # type: ignore[arg-type]
                irs=self.ir_option_params.option_fixing.irs,
                expiry=self.ir_option_params.expiry,
                tenor=self.ir_option_params.option_fixing.termination,
            )
        else:
            pricing_ = pricing
        del pricing

        if metric_ == IROptionMetric.NormalVol:
            # use a root finder to reverse engineer the _Bachelier model.
            if anal_delta is None:
                anal_delta_: DualTypes = self.ir_option_params.option_fixing.irs.analytic_delta(  # type: ignore[assignment]
                    curves=_Curves(disc_curve=disc_curve_),
                    forward=self.settlement_params.payment,
                    local=False,
                )
            else:
                anal_delta_ = anal_delta
            del anal_delta

            def s(g: DualTypes) -> DualTypes:
                return _OptionModelBachelier._value(
                    F=pricing_.f,
                    K=pricing_.k,
                    t_e=self.ir_option_params.time_to_expiry(disc_curve_.nodes.initial),
                    v2=1.0,
                    vol=g,
                    phi=self.ir_option_params.direction.value,
                )

            result = ift_1dim(
                s=s,
                s_tgt=1e4 * cash / (anal_delta_ * self.settlement_params.notional),
                h="modified_brent",
                ini_h_args=(0.0001, 10.0),
            )
            g: DualTypes = result["g"]
            return g * 100.0

        else:  # metric_ in [BlackVol types]
            # might need to resolve a volatility value depending upon the required shift
            # and the expected shift

            expected_shift = {
                IROptionMetric.LogNormalVol: 0,
                IROptionMetric.BlackVol: 0,
                IROptionMetric.BlackVolShift100: 100,
                IROptionMetric.BlackVolShift200: 200,
                IROptionMetric.BlackVolShift300: 300,
            }
            required_shift = expected_shift[metric_]
            provided_shift = int(_dual_float(pricing_.shift))
            if required_shift == provided_shift:
                return pricing_.vol
            else:
                # use a root finder to reverse engineer the shifted vol value.
                if anal_delta is None:
                    anal_delta_ = self.ir_option_params.option_fixing.irs.analytic_delta(  # type: ignore[assignment]
                        curves=_Curves(disc_curve=disc_curve_),
                        forward=self.settlement_params.payment,
                        local=False,
                    )
                else:
                    anal_delta_ = anal_delta
                del anal_delta

                def s(g: DualTypes) -> DualTypes:
                    return _OptionModelBlack76._value(
                        F=pricing_.f + float(required_shift) / 100.0,
                        K=pricing_.k + float(required_shift) / 100.0,
                        t_e=self.ir_option_params.time_to_expiry(disc_curve_.nodes.initial),
                        v2=1.0,
                        vol=g,
                        phi=self.ir_option_params.direction.value,
                    )

                result = ift_1dim(
                    s=s,
                    s_tgt=1e4 * cash / (anal_delta_ * self.settlement_params.notional),
                    h="modified_brent",
                    ini_h_args=(0.0001, 10.0),
                )
                g = result["g"]
                return g * 100.0

    #
    # def implied_vol(
    #     self,
    #     rate_curve: _BaseCurve,
    #     disc_curve: _BaseCurve,
    #     fx: FXForwards,
    #     premium: DualTypes,
    #     metric: FXOptionMetric | str_ = NoInput(0),
    # ) -> Number:
    #     """
    #     Calculate the implied volatility of the FX option.
    #
    #     Parameters
    #     ----------
    #     rate_curve: Curve
    #         Not used by `implied_vol`.
    #     disc_curve: Curve
    #         The discount *Curve* for the RHS currency.
    #     fx: FXForwards
    #         The object to project the currency pair FX rate at delivery.
    #     premium: float, Dual, Dual2
    #         The premium value of the option paid at the appropriate payment date. Expressed
    #         either in *'pips'* or *'percent'* of notional. Must align with ``metric``.
    #     metric: str in {"pips", "percent"}, optional
    #         The manner in which the premium is expressed.
    #
    #     Returns
    #     -------
    #     float, Dual or Dual2
    #     """
    #     if isinstance(self.ir_option_params.strike, NoInput):
    #         raise ValueError(err.VE_NEEDS_STRIKE)
    #     k = self.ir_option_params.strike
    #     phi = self.ir_option_params.direction
    #     metric_ = _get_ir_option_metric(_drb(self.ir_option_params.metric, metric))
    #
    #     # This function uses newton_1d and is AD safe.
    #
    #     # convert the premium to a standardised immediate pips value.
    #     if metric_ == FXOptionMetric.Percent:
    #         # convert premium to pips form
    #         premium = (
    #             premium
    #             * fx.rate(self.ir_option_params.pair, self.settlement_params.payment)
    #             * 100.0
    #         )
    #     # convert to immediate pips form
    #     imm_premium = premium * disc_curve[self.settlement_params.payment]
    #
    #     t_e = self.ir_option_params.time_to_expiry(disc_curve.nodes.initial)
    #     v2 = disc_curve[self.ir_option_params.delivery]
    #     f_d = fx.rate(self.ir_option_params.pair, self.ir_option_params.delivery)
    #
    #     def root(
    #         vol: DualTypes, f_d: DualTypes, k: DualTypes, t_e: float, v2: DualTypes, phi: float
    #     ) -> tuple[DualTypes, DualTypes]:
    #         f0 = _OptionModelBlack76._value(f_d, k, t_e, NoInput(0), v2, vol, phi) * 10000.0 - imm_premium
    #         sqrt_t = t_e**0.5
    #         d_plus = _d_plus_min_u(k / f_d, vol * sqrt_t, 0.5)
    #         f1 = v2 * dual_norm_pdf(phi * d_plus) * f_d * sqrt_t * 10000.0
    #         return f0, f1
    #
    #     result = newton_1dim(root, 0.10, args=(f_d, k, t_e, v2, phi))
    #     _: Number = result["g"] * 100.0
    #     return _
    #
    # def _payoff_at_expiry(
    #     self, rng: tuple[float, float] | NoInput = NoInput(0)
    # ) -> tuple[Arr1dF64, Arr1dF64]:
    #     # used by plotting methods
    #     if isinstance(self.ir_option_params.strike, NoInput):
    #         raise ValueError(
    #             "Cannot return payoff for option without a specified `strike`.",
    #         )  # pragma: no cover
    #     if isinstance(rng, NoInput):
    #         x = np.linspace(0, 20, 1001)
    #     else:
    #         x = np.linspace(rng[0], rng[1], 1001)
    #     k: float = _dual_float(self.ir_option_params.strike)
    #     _ = (x - k) * self.ir_option_params.direction
    #     __ = np.zeros(1001)
    #     if self.ir_option_params.direction > 0:  # call
    #         y = np.where(x < k, __, _) * self.settlement_params.notional
    #     else:  # put
    #         y = np.where(x > k, __, _) * self.settlement_params.notional
    #     return x, y


class IRSCallPeriod(_BaseIRSOptionPeriod):
    r"""
    A *Period* defined by a European call option on an IRS.

    The expected unindexed reference cashflow is given by,

    .. math::

       \mathbb{E^Q}[\bar{C}_t] = \left \{ \begin{matrix} \max(f_d - K, 0) & \text{after expiry} \\ B76(f_d, K, t, \sigma) & \text{before expiry} \end{matrix} \right .

    where :math:`B76(.)` is the Black-76 option pricing formula, using log-normal volatility
    calculations with calendar day time reference.

    .. rubric:: Examples

    .. ipython:: python
       :suppress:

       from rateslib.periods import FXCallPeriod
       from datetime import datetime as dt

    .. ipython:: python

       fxo = FXCallPeriod(
           delivery=dt(2000, 3, 1),
           pair="eurusd",
           expiry=dt(2000, 2, 28),
           strike=1.10,
           delta_type="forward",
       )
       fxo.cashflows()

    .. role:: red

    .. role:: green

    Parameters
    ----------
    .
        .. note::

           The following define **ir option** and generalised **settlement** parameters.

    expiry: datetime, :red:`required`
        The expiry date of the option, when the option fixing is determined.
    irs_series: IRSSeries, str :red:`required`
        This defines the conventions of the underlying :class:`~rateslib.instruments.IRS`.
    tenor: datetime, str :red:`required`
        The tenor of the underlying :class:`~rateslib.instruments.IRS`.
    strike: float, Dual, Dual2, Variable, :green:`optional`
        The strike fixed rate of the option. Can be set after initialisation.
    notional: float, Dual, Dual2, Variable, :green:`optional (set by 'defaults')`
        The notional of the option expressed in reference currency.
    metric: IROptionMetric, str, :green:`optional` (set by 'default')`
        The metric used by default in the
        :meth:`~rateslib.periods.ir_volatility._BaseIRSOptionPeriod.rate` method.
    option_fixings: float, Dual, Dual2, Variable, Series, str, :green:`optional`
        The value of the option :class:`~rateslib.data.fixings.IRSFixing`. If a scalar, is used
        directly. If a string identifier, links to the central ``fixings`` object and data loader.
        See :ref:`fixings <fixings-doc>`.
    settlement_method: SwaptionSettlementMethod, str, :green:`optional` (set by 'default')`
        The method for deriving the settlement cashflow or underlying value.
    ex_dividend: datetime, :green:`optional (set as 'delivery')`
        The ex-dividend date of the settled cashflow.

        .. note::

           This *Period* type has not implemented **indexation** or **non-deliverability**.

    """  # noqa: E501

    def __init__(
        self,
        *,
        # option params:
        expiry: datetime,
        tenor: datetime | str,
        irs_series: IRSSeries | str,
        strike: DualTypes_ = NoInput(0),
        notional: DualTypes_ = NoInput(0),
        metric: IROptionMetric | str_ = NoInput(0),
        option_fixings: DualTypes | Series[DualTypes] | str_ = NoInput(0),  # type: ignore[type-var]
        # currency args:
        settlement_method: SwaptionSettlementMethod | str_ = NoInput(0),
        ex_dividend: datetime_ = NoInput(0),
    ) -> None:
        super().__init__(
            direction=OptionType.Call,
            tenor=tenor,
            irs_series=irs_series,
            expiry=expiry,
            strike=strike,
            notional=notional,
            metric=metric,
            option_fixings=option_fixings,
            settlement_method=settlement_method,
            ex_dividend=ex_dividend,
        )


class IRSPutPeriod(_BaseIRSOptionPeriod):
    r"""
    A *Period* defined by a European FX put option.

    The expected unindexed reference cashflow is given by,

    .. math::

       \mathbb{E^Q}[\bar{C}_t] = \left \{ \begin{matrix} \max(K - f_d, 0) & \text{after expiry} \\ B76(f_d, K, t, \sigma) & \text{before expiry} \end{matrix} \right .

    where :math:`B76(.)` is the Black-76 option pricing formula, using log-normal volatility
    calculations with calendar day time reference.

    .. rubric:: Examples

    .. ipython:: python
       :suppress:

       from rateslib.periods import FXPutPeriod
       from datetime import datetime as dt

    .. ipython:: python

       fxo = FXPutPeriod(
           delivery=dt(2000, 3, 1),
           pair="eurusd",
           expiry=dt(2000, 2, 28),
           strike=1.10,
           delta_type="forward",
       )
       fxo.cashflows()

    .. role:: red

    .. role:: green

    Parameters
    ----------
    .
        .. note::

           The following define **fx option** and generalised **settlement** parameters.

    delivery: datetime, :red:`required`
        The settlement date of the underlying FX rate of the option. Also used as the implied
        payment date of the cashflow valuation date.
    pair: str, :red:`required`
        The currency pair of the :class:`~rateslib.data.fixings.FXFixing` against which the option
        will settle.
    expiry: datetime, :red:`required`
        The expiry date of the option, when the option fixing is determined.
    strike: float, Dual, Dual2, Variable, :green:`optional`
        The strike price of the option. Can be set after initialisation.
    notional: float, Dual, Dual2, Variable, :green:`optional (set by 'defaults')`
        The notional of the option expressed in units of LHS currency of `pair`.
    delta_type: FXDeltaMethod, str, :green:`optional (set by 'default')`
        The definition of the delta for the option.
    metric: FXDeltaMethod, str, :green:`optional` (set by 'default')`
        The metric used by default in the
        :meth:`~rateslib.periods.fx_volatility.FXOptionPeriod.rate` method.
    option_fixings: float, Dual, Dual2, Variable, Series, str, :green:`optional`
        The value of the option :class:`~rateslib.data.fixings.FXFixing`. If a scalar, is used
        directly. If a string identifier, links to the central ``fixings`` object and data loader.
        See :ref:`fixings <fixings-doc>`.
    settlement_method: SwaptionSettlementMethod, str, :green:`optional` (set by 'default')`
        The method for deriving the settlement cashflow or underlying value.
    ex_dividend: datetime, :green:`optional (set as 'delivery')`
        The ex-dividend date of the settled cashflow.

        .. note::

           This *Period* type has not implemented **indexation** or **non-deliverability**.

    """  # noqa: E501

    def __init__(
        self,
        *,
        # option params:
        expiry: datetime,
        tenor: datetime | str,
        irs_series: IRSSeries | str,
        strike: DualTypes_ = NoInput(0),
        notional: DualTypes_ = NoInput(0),
        metric: IROptionMetric | str_ = NoInput(0),
        option_fixings: DualTypes | Series[DualTypes] | str_ = NoInput(0),  # type: ignore[type-var]
        # currency args:
        settlement_method: SwaptionSettlementMethod | str_ = NoInput(0),
        ex_dividend: datetime_ = NoInput(0),
    ) -> None:
        super().__init__(
            direction=OptionType.Put,
            tenor=tenor,
            irs_series=irs_series,
            expiry=expiry,
            strike=strike,
            notional=notional,
            metric=metric,
            option_fixings=option_fixings,
            settlement_method=settlement_method,
            ex_dividend=ex_dividend,
        )
