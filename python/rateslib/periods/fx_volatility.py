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

import numpy as np
from pytz import UTC

import rateslib.errors as err
from rateslib import defaults
from rateslib.curves._parsers import _validate_obj_not_no_input
from rateslib.data.fixings import _get_fx_index
from rateslib.dual import dual_exp, dual_log, dual_norm_cdf, dual_norm_pdf, newton_1dim
from rateslib.dual.utils import _dual_float
from rateslib.enums.generics import NoInput, Ok, Result, _drb
from rateslib.enums.parameters import (
    FXDeltaMethod,
    FXOptionMetric,
    OptionType,
    _get_fx_delta_type,
    _get_fx_option_metric,
)
from rateslib.fx import FXForwards
from rateslib.fx_volatility import (
    FXDeltaVolSmile,
    FXDeltaVolSurface,
    FXSabrSmile,
    FXSabrSurface,
    FXVolObj,
)
from rateslib.fx_volatility.delta_vol import (
    _moneyness_from_atm_delta_one_dimensional,
    _moneyness_from_atm_delta_two_dimensional,
    _moneyness_from_delta_one_dimensional,
    _moneyness_from_delta_two_dimensional,
)
from rateslib.fx_volatility.utils import (
    _black76,
    _d_plus_min_u,
    _delta_type_constants,
    _moneyness_from_atm_delta_closed_form,
    _moneyness_from_delta_closed_form,
    _surface_index_left,
)
from rateslib.periods.parameters import (
    _FXOptionParams,
    _IndexParams,
    _NonDeliverableParams,
    _SettlementParams,
)
from rateslib.periods.protocols import _BasePeriodStatic, _WithAnalyticFXOptionGreeks
from rateslib.periods.utils import (
    _get_vol_delta_type,
    _get_vol_maybe_from_obj,
    _get_vol_smile_or_raise,
    _get_vol_smile_or_value,
    _validate_fx_as_forwards,
)

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        Any,
        Arr1dF64,
        DualTypes,
        DualTypes_,
        FXForwards_,
        FXIndex,
        Number,
        Series,
        _BaseCurve,
        _BaseCurve_,
        _FXVolOption,
        _FXVolOption_,
        datetime,
        datetime_,
        str_,
    )


class _BaseFXOptionPeriod(_BasePeriodStatic, _WithAnalyticFXOptionGreeks, metaclass=ABCMeta):
    r"""
    Abstract base class for *FXOptionPeriods* types.

    **See Also**: :class:`~rateslib.periods.FXCallPeriod`,
    :class:`~rateslib.periods.FXPutPeriod`

    """

    def analytic_greeks(
        self,
        rate_curve: _BaseCurve,
        disc_curve: _BaseCurve,
        fx: FXForwards,
        fx_vol: _FXVolOption_ = NoInput(0),
        premium: DualTypes_ = NoInput(0),  # expressed in the payment currency
        premium_payment: datetime_ = NoInput(0),
    ) -> dict[str, Any]:
        return super()._base_analytic_greeks(
            rate_curve=rate_curve,
            disc_curve=disc_curve,
            fx=fx,
            fx_vol=fx_vol,
            premium=premium,
            premium_payment=premium_payment,
        )

    @property
    def period_params(self) -> None:
        """This *Period* type has no
        :class:`~rateslib.periods.parameters._PeriodParams`."""
        return self._period_params

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
        return self._rate_params

    @property
    def fx_option_params(self) -> _FXOptionParams:
        """The :class:`~rateslib.periods.parameters._FXOptionParams` of the
        *Period*."""
        return self._fx_option_params

    @abstractmethod
    def __init__(
        self,
        *,
        # option params:
        direction: OptionType,
        delivery: datetime,  # otherwise termed the 'payment' of the period
        pair: FXIndex | str,
        expiry: datetime,
        strike: DualTypes_ = NoInput(0),
        notional: DualTypes_ = NoInput(0),
        delta_type: FXDeltaMethod | str_ = NoInput(0),
        metric: FXOptionMetric | str_ = NoInput(0),
        option_fixings: DualTypes | Series[DualTypes] | str_ = NoInput(0),  # type: ignore[type-var]
        # currency args:
        ex_dividend: datetime_ = NoInput(0),
        # # non-deliverable args:
        # nd_pair: str_ = NoInput(0),
        # fx_fixings: DualTypes | Series[DualTypes] | str_ = NoInput(0),  # type: ignore[type-var]
        # # index-args:
        # index_base: DualTypes_ = NoInput(0),
        # index_lag: int_ = NoInput(0),
        # index_method: IndexMethod | str_ = NoInput(0),
        # index_fixings: DualTypes | Series[DualTypes] | str_ = NoInput(0),  #type: ignore[type-var]
        # index_only: bool_ = NoInput(0),
        # index_base_date: datetime_ = NoInput(0),
        # index_reference_date: datetime_ = NoInput(0),
    ) -> None:
        # self._index_params = _init_or_none_IndexParams(
        #     _index_base=index_base,
        #     _index_lag=index_lag,
        #     _index_method=index_method,
        #     _index_fixings=index_fixings,
        #     _index_only=index_only,
        #     _index_base_date=index_base_date,
        #     _index_reference_date=_drb(delivery, index_reference_date),
        # )
        self._index_params = None
        self._fx_option_params = _FXOptionParams(
            _direction=direction,
            _expiry=expiry,
            _delivery=delivery,
            _delta_type=_get_fx_delta_type(_drb(defaults.fx_delta_type, delta_type)),
            _fx_index=_get_fx_index(pair),
            _strike=strike,
            _metric=_drb(defaults.fx_option_metric, metric),
            _option_fixings=option_fixings,
        )
        self._rate_params = None
        self._period_params = None

        nd_pair = NoInput(0)
        if isinstance(nd_pair, NoInput):
            # then option is directly deliverable
            self._non_deliverable_params: _NonDeliverableParams | None = None
            self._settlement_params = _SettlementParams(
                _notional=_drb(defaults.notional, notional),
                _payment=delivery,
                _currency=self.fx_option_params.fx_index.pair[3:],
                _notional_currency=self.fx_option_params.fx_index.pair[:3],
                _ex_dividend=ex_dividend,
            )
        else:
            pass
            # fx_ccy1, fx_ccy2 = self.fx_option_params.pair[:3], self.fx_option_params.pair[3:]
            # nd_ccy1, nd_ccy2 = nd_pair.lower()[:3], nd_pair.lower()[3:]
            #
            # if nd_ccy1 != fx_ccy1 and nd_ccy1 != fx_ccy2:
            #     raise ValueError(
            #         err.VE_MISMATCHED_FX_PAIR_ND_PAIR.format(nd_ccy1, self.fx_option_params.pair)
            #     )
            # elif nd_ccy2 != fx_ccy1 and nd_ccy2 != fx_ccy2:
            #     raise ValueError(
            #         err.VE_MISMATCHED_FX_PAIR_ND_PAIR.format(nd_ccy2, self.fx_option_params.pair)
            #     )
            #
            # self._non_deliverable_params = _NonDeliverableParams(
            #     _currency=fx_ccy1,
            #     _pair=nd_pair,
            #     _delivery=delivery,
            #     _fx_fixings=fx_fixings,
            # )
            # self._settlement_params = _SettlementParams(
            #     _notional=_drb(defaults.notional, notional),
            #     _payment=delivery,
            #     _currency=fx_ccy1,
            #     _notional_currency=fx_ccy1,
            #     _ex_dividend=ex_dividend,
            # )

    def __repr__(self) -> str:
        return f"<rl.{type(self).__name__} at {hex(id(self))}>"

    def unindexed_reference_cashflow(  # type: ignore[override]
        self,
        *,
        rate_curve: _BaseCurve_ = NoInput(0),  # w(.) variety
        disc_curve: _BaseCurve_ = NoInput(0),  # v(.) variety
        # index_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: _FXVolOption_ = NoInput(0),
        **kwargs: Any,
    ) -> DualTypes:
        # The unindexed_reference_cashflow does not require a discount curve.
        # A curve may only be required to determine an evaluation date, which in turn is used to
        # derive 'time_to_expiry'. The cashflow is expressed in reference currency on the delivery
        # date of the FX forward, i.e. the 'forward FX date'.
        if isinstance(self.fx_option_params.strike, NoInput):
            raise ValueError(err.VE_NEEDS_STRIKE)
        k = self.fx_option_params.strike

        if not isinstance(self.fx_option_params.option_fixing.value, NoInput):
            # then the cashflow amount is defined by a known fixing
            fix: DualTypes = self.fx_option_params.option_fixing.value
            phi: OptionType = self.fx_option_params.direction

            if phi == OptionType.Call and k < fix:
                return (fix - k) * self.settlement_params.notional
            elif phi == OptionType.Put and k > fix:
                return (k - fix) * self.settlement_params.notional
            else:
                return 0.0

        else:
            # value is expressed in reference currency (i.e. pair[3:])
            fx_ = _validate_fx_as_forwards(fx)

            vol_ = _get_vol_maybe_from_obj(
                fx_vol=fx_vol,
                fx=fx_,
                rate_curve=rate_curve,
                strike=k,
                pair=self.fx_option_params.pair,
                delivery=self.fx_option_params.delivery,
                expiry=self.fx_option_params.expiry,
            )

            # Get time to expiry from some object
            if not isinstance(disc_curve, NoInput):
                t_e = self.fx_option_params.time_to_expiry(disc_curve.nodes.initial)
            elif isinstance(fx_vol, FXVolObj):
                t_e = self.fx_option_params.time_to_expiry(fx_vol.meta.eval_date)
            elif not isinstance(rate_curve, NoInput):
                t_e = self.fx_option_params.time_to_expiry(rate_curve.nodes.initial)
            else:
                raise ValueError(
                    "Object required to define evaluation date and time to expiry.\n"
                    "Use one of `disc_curve`, `fx_vol`, or `rate_curve`."
                )

            expected = _black76(
                F=fx_.rate(self.fx_option_params.pair, self.fx_option_params.delivery),
                K=k,
                t_e=t_e,
                v1=NoInput(0),  # not required
                v2=1.0,  # disc_curve_[delivery] / disc_curve_[payment],
                vol=vol_ / 100.0,
                phi=self.fx_option_params.direction.value,  # controls calls or put price
            )
            return expected * self.settlement_params.notional

    def try_rate(
        self,
        rate_curve: _BaseCurve,
        disc_curve: _BaseCurve,
        fx: FXForwards_ = NoInput(0),
        fx_vol: _FXVolOption_ = NoInput(0),
        metric: FXOptionMetric | str_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> Result[DualTypes]:
        """
        Return the pricing metric of the *FXOption*, with lazy error handling.

        See :meth:`~rateslib.periods.FXOptionPeriod.rate`.
        """
        if not isinstance(metric, NoInput):
            metric_ = _get_fx_option_metric(metric)
        else:  # use metric associated with self
            metric_ = self.fx_option_params.metric

        cash_res = self.try_unindexed_reference_cashflow(
            rate_curve=rate_curve,
            disc_curve=disc_curve,
            fx=fx,
            fx_vol=fx_vol,
        )
        if cash_res.is_err:
            return cash_res
        cash: DualTypes = cash_res.unwrap()

        if metric_ == FXOptionMetric.Pips:
            points_premium = cash / self.settlement_params.notional
            if isinstance(forward, NoInput):
                return Ok(points_premium * 10000.0)
            else:
                return Ok(
                    points_premium
                    * 10000.0
                    * disc_curve[self.settlement_params.payment]
                    / disc_curve[forward]
                )
        else:  # metric_ == FXOptionMetric.Percent:
            fx_ = _validate_fx_as_forwards(fx)
            currency_premium = cash / fx_.rate(
                self.fx_option_params.pair, self.settlement_params.payment
            )
            if isinstance(forward, NoInput):
                return Ok(currency_premium / self.settlement_params.notional * 100)
            else:
                currency_premium *= rate_curve[self.settlement_params.payment] / rate_curve[forward]
                return Ok(currency_premium / self.settlement_params.notional * 100)

    def rate(
        self,
        *,
        rate_curve: _BaseCurve,
        disc_curve: _BaseCurve,
        fx: FXForwards_ = NoInput(0),
        fx_vol: _FXVolOption_ = NoInput(0),
        metric: str_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> DualTypes:
        """
        Return the pricing metric of the *FXOption*.

        This is priced according to the ``payment`` date of the *OptionPeriod*.

        Parameters
        ----------
        rate_curve: Curve
            The discount *Curve* for the LHS currency. (Not used).
        disc_curve: Curve
            The discount *Curve* for the RHS currency.
        fx: float, FXRates, FXForwards, optional
            The object to project the currency pair FX rate at delivery.
        base: str, optional
            Not used by `rate`.
        fx_vol: float, Dual, Dual2
            The percentage log-normal volatility to price the option.
        metric: str in {"pips", "percent"}
            The metric to return. If "pips" assumes the premium is in foreign (rhs)
            currency. If "percent", the premium is assumed to be domestic (lhs).
        forward: datetime, optional (set as payment date of option)
            The date to project the cashflow value to using the ``disc_curve`` if RHS ("pips") or
            using ``rate_curve`` if LHS ("percent").

        Returns
        -------
        float, Dual, Dual2 or dict of such.
        """
        return self.try_rate(
            rate_curve=rate_curve,
            disc_curve=disc_curve,
            fx=fx,
            fx_vol=fx_vol,
            metric=metric,
            forward=forward,
        ).unwrap()

    def implied_vol(
        self,
        rate_curve: _BaseCurve,
        disc_curve: _BaseCurve,
        fx: FXForwards,
        premium: DualTypes,
        metric: FXOptionMetric | str_ = NoInput(0),
    ) -> Number:
        """
        Calculate the implied volatility of the FX option.

        Parameters
        ----------
        rate_curve: Curve
            Not used by `implied_vol`.
        disc_curve: Curve
            The discount *Curve* for the RHS currency.
        fx: FXForwards
            The object to project the currency pair FX rate at delivery.
        premium: float, Dual, Dual2
            The premium value of the option paid at the appropriate payment date. Expressed
            either in *'pips'* or *'percent'* of notional. Must align with ``metric``.
        metric: str in {"pips", "percent"}, optional
            The manner in which the premium is expressed.

        Returns
        -------
        float, Dual or Dual2
        """
        if isinstance(self.fx_option_params.strike, NoInput):
            raise ValueError(err.VE_NEEDS_STRIKE)
        k = self.fx_option_params.strike
        phi = self.fx_option_params.direction
        metric_ = _get_fx_option_metric(_drb(self.fx_option_params.metric, metric))

        # This function uses newton_1d and is AD safe.

        # convert the premium to a standardised immediate pips value.
        if metric_ == FXOptionMetric.Percent:
            # convert premium to pips form
            premium = (
                premium
                * fx.rate(self.fx_option_params.pair, self.settlement_params.payment)
                * 100.0
            )
        # convert to immediate pips form
        imm_premium = premium * disc_curve[self.settlement_params.payment]

        t_e = self.fx_option_params.time_to_expiry(disc_curve.nodes.initial)
        v2 = disc_curve[self.fx_option_params.delivery]
        f_d = fx.rate(self.fx_option_params.pair, self.fx_option_params.delivery)

        def root(
            vol: DualTypes, f_d: DualTypes, k: DualTypes, t_e: float, v2: DualTypes, phi: float
        ) -> tuple[DualTypes, DualTypes]:
            f0 = _black76(f_d, k, t_e, NoInput(0), v2, vol, phi) * 10000.0 - imm_premium
            sqrt_t = t_e**0.5
            d_plus = _d_plus_min_u(k / f_d, vol * sqrt_t, 0.5)
            f1 = v2 * dual_norm_pdf(phi * d_plus) * f_d * sqrt_t * 10000.0
            return f0, f1

        result = newton_1dim(root, 0.10, args=(f_d, k, t_e, v2, phi))
        _: Number = result["g"] * 100.0
        return _

    # Volatility determinations

    def _index_vol_and_strike_from_atm(
        self,
        delta_type: FXDeltaMethod,
        vol: _FXVolOption,
        w_deli: DualTypes,
        w_spot: DualTypes,
        f: DualTypes | FXForwards,
        t_e: DualTypes,
    ) -> tuple[DualTypes | None, DualTypes, DualTypes]:
        """
        This function returns strike and vol, where available, a delta index for an option period
        defined by ATM delta.

        Parameters
        ----------
        delta_type: FXDeltaMethod
            The delta type of the option period.
        vol: DualTypes | Smile | Surface
            The volatility used, either specifici value or a Smile/Surface.
        w_deli: DualTypes
            The relevant discount factor at delivery.
        w_spot: DualTypes
            The relevant discount factor at spot.
        f: DualTypes, FXForwards
            The forward FX rate for delivery. FXForwards is used when a SabrSurface is present.
        t_e: DualTypes
            The time to expiry

        Returns
        -------
        (delta_index, vol, strike)
        """
        # TODO this method branches depending upon eta0 and eta1, but depending upon the
        # type of vol these maybe automatically set equal to each other. Refactoring this would
        # make eliminate repeated type checking for the vol argument.
        vol_delta_type = _get_vol_delta_type(vol, delta_type)

        z_w = w_deli / w_spot
        eta_0, z_w_0, _ = _delta_type_constants(delta_type, z_w, 0.0)  # u: unused
        eta_1, z_w_1, _ = _delta_type_constants(vol_delta_type, z_w, 0.0)  #  u: unused

        if isinstance(vol, FXSabrSmile | FXSabrSurface):
            return self._index_vol_and_strike_from_atm_sabr(f, eta_0, vol)
        else:  # DualTypes | FXDeltaVolSmile | FXDeltaVolSurface
            f_: DualTypes = f  # type: ignore[assignment]
            vol_: DualTypes | FXDeltaVolSmile | FXDeltaVolSurface = vol  # type: ignore[assignment]
            return self._index_vol_and_strike_from_atm_dv(
                f_,
                eta_0,
                eta_1,
                z_w_0,
                z_w_1,
                vol_,
                t_e,
                delta_type,
                vol_delta_type,
                z_w,
            )

    def _index_vol_and_strike_from_atm_sabr(
        self,
        f: DualTypes | FXForwards,
        eta_0: float,
        vol: FXSabrSmile | FXSabrSurface,
    ) -> tuple[DualTypes | None, DualTypes, DualTypes]:
        """Get vol and strike from ATM delta specification under a SABR model."""
        t_e = (self.fx_option_params.expiry - vol.meta.eval_date).days / 365.0
        if isinstance(f, FXForwards):
            f_d: DualTypes = f.rate(self.fx_option_params.pair, self.fx_option_params.delivery)
            # _ad = _set_ad_order_objects([0], [f])  # GH755
        else:
            f_d = f  # type: ignore[assignment]

        def root1d(
            k: DualTypes, f_d: DualTypes, fx: DualTypes | FXForwards, as_float: bool
        ) -> tuple[DualTypes, DualTypes]:
            # if not as_float and isinstance(fx, FXForwards):
            #     _set_ad_order_objects(_ad, [fx])
            dsigma_dk: Number
            sigma, dsigma_dk = vol._d_sabr_d_k_or_f(  # type: ignore[assignment]
                k=k, f=fx, expiry=self.fx_option_params.expiry, as_float=as_float, derivative=1
            )
            f0 = -dual_log(k / f_d) + eta_0 * sigma**2 * t_e
            f1 = -1 / k + eta_0 * 2 * sigma * dsigma_dk * t_e
            return f0, f1

        if isinstance(vol, FXSabrSmile):
            alpha = vol.nodes.alpha
        else:  # FXSabrSurface
            vol_: FXSabrSurface = vol  # type: ignore[assignment]
            expiry_posix = self.fx_option_params.expiry.replace(tzinfo=UTC).timestamp()
            e_idx, _ = _surface_index_left(vol_.meta.expiries_posix, expiry_posix)
            alpha = vol_.smiles[e_idx].nodes.alpha

        root_solver = newton_1dim(
            root1d,
            f_d * dual_exp(eta_0 * alpha**2 * t_e),
            args=(f_d, f),
            pre_args=(True,),  # solve `as_float` in iterations
            final_args=(False,),  # capture AD in final iterations
            raise_on_fail=True,
        )

        k: DualTypes = root_solver["g"]
        v_ = vol.get_from_strike(k, f, self.fx_option_params.expiry)[1]
        return None, v_, k

    def _index_vol_and_strike_from_atm_dv(  # DeltaVol type models
        self,
        f: DualTypes,
        eta_0: float,
        eta_1: float,
        z_w_0: DualTypes,
        z_w_1: DualTypes,
        vol: DualTypes | FXDeltaVolSmile | FXDeltaVolSurface,
        t_e: DualTypes,
        delta_type: FXDeltaMethod,
        vol_delta_type: FXDeltaMethod,
        z_w: DualTypes,
    ) -> tuple[DualTypes | None, DualTypes, DualTypes]:
        """Determine strike from ATM delta specification with DeltaVol models or fixed volatility"""
        if eta_0 == 0.5:  # then delta type is unadjusted
            if eta_1 == 0.5:  # then smile delta type matches: closed form eqn available
                if isinstance(vol, FXDeltaVolSmile | FXDeltaVolSurface):
                    d_i: DualTypes = z_w_1 / 2.0
                    vol_value: DualTypes = _get_vol_smile_or_raise(
                        vol, self.fx_option_params.expiry
                    )[d_i]
                    delta_idx: DualTypes | None = d_i
                else:
                    vol_value = _validate_obj_not_no_input(vol, "vol")  # type: ignore[assignment]
                    delta_idx = None
                u = _moneyness_from_atm_delta_closed_form(vol_value, t_e)
                return delta_idx, vol_value, u * f
            else:  # then smile delta type unmatched: 2-d solver required
                delta: DualTypes = z_w_0 * self.fx_option_params.direction.value / 2.0
                u, delta_idx = _moneyness_from_delta_two_dimensional(
                    delta,
                    delta_type,
                    _get_vol_smile_or_raise(vol, self.fx_option_params.expiry),
                    t_e,
                    z_w,
                    self.fx_option_params.direction.value,
                )
        else:  # then delta type is adjusted,
            if eta_1 == -0.5:  # then smile type matches: use 1-d solver
                u = _moneyness_from_atm_delta_one_dimensional(
                    delta_type,
                    vol_delta_type,
                    _get_vol_smile_or_value(vol, self.fx_option_params.expiry),
                    t_e,
                    z_w,
                    self.fx_option_params.direction,
                )
                delta_idx = z_w_1 * u * 0.5
            else:  # smile delta type unmatched: 2-d solver required
                u, delta_idx = _moneyness_from_atm_delta_two_dimensional(
                    delta_type,
                    _get_vol_smile_or_raise(vol, self.fx_option_params.expiry),
                    t_e,
                    z_w,
                    self.fx_option_params.direction,
                )

        if isinstance(vol, FXDeltaVolSmile | FXDeltaVolSurface):
            vol_value = _get_vol_smile_or_raise(vol, self.fx_option_params.expiry)[delta_idx]
        else:
            vol_value = _validate_obj_not_no_input(vol, "vol")  # type: ignore[assignment]

        # u, delta_idx, delta =
        # self._moneyness_from_delta_three_dimensional(delta_type, vol, t_e, z_w)
        return delta_idx, vol_value, u * f

    def _index_vol_and_strike_from_delta(
        self,
        delta: float,
        delta_type: FXDeltaMethod,
        vol: _FXVolOption,
        w_deli: DualTypes,
        w_spot: DualTypes,
        f: DualTypes | FXForwards,
        t_e: DualTypes,
    ) -> tuple[DualTypes | None, DualTypes, DualTypes]:
        """
        This function returns strike and, where available, a delta index for an option period
        defined by a fixed delta percentage.

        Parameters
        ----------
        delta: float
           The delta percent, e.g 0.25.
        delta_type: FXDeltaMethod
           The delta type of the option period.
        vol: DualTypes | Smile | Surface
           The volatility used, either a specific value or a Smile/Surface.
        w_deli: DualTypes
           The relevant discount factor at delivery.
        w_spot: DualTypes
           The relevant discount factor at spot.
        f: DualTypes | FXForwards
           The forward FX rate for delivery. When using a *SabrSurface* this is required in
           *FXForwards* form.
        t_e: DualTypes
           The time to expiry

        Returns
        -------
        (DualTypes, DualTypes)
        """
        vol_delta_type = _get_vol_delta_type(vol, delta_type)
        z_w = w_deli / w_spot

        if isinstance(vol, FXSabrSmile | FXSabrSurface):
            return self._index_vol_and_strike_from_delta_sabr(delta, delta_type, vol, z_w, f)
        else:  # DualTypes | FXDeltaVolSmile | FXDeltaVolSurface
            f_: DualTypes = f  # type: ignore[assignment]
            vol_: DualTypes | FXDeltaVolSmile = vol  # type: ignore[assignment]
            return self._index_vol_and_strike_from_delta_dv(
                f_,
                delta,
                vol_,
                t_e,
                delta_type,
                vol_delta_type,
                z_w,
            )

    def _index_vol_and_strike_from_delta_dv(
        self,
        f: DualTypes,
        delta: float,
        vol: DualTypes | FXDeltaVolSmile | FXDeltaVolSurface,
        t_e: DualTypes,
        delta_type: FXDeltaMethod,
        vol_delta_type: FXDeltaMethod,
        z_w: DualTypes,
    ) -> tuple[DualTypes | None, DualTypes, DualTypes]:
        """Determine strike and delta index for an option by delta % for DeltaVol type models
        or constant volatility"""
        eta_0, z_w_0, _ = _delta_type_constants(delta_type, z_w, 0.0)  # u: unused
        eta_1, z_w_1, _ = _delta_type_constants(vol_delta_type, z_w, 0.0)  # u: unused
        # then delta types are both unadjusted, used closed form.
        if eta_0 == eta_1 and eta_0 == 0.5:
            if isinstance(vol, FXDeltaVolSmile | FXDeltaVolSurface):
                d_i: DualTypes = (-z_w_1 / z_w_0) * (
                    delta - 0.5 * z_w_0 * (self.fx_option_params.direction + 1.0)
                )
                vol_value: DualTypes = _get_vol_smile_or_raise(vol, self.fx_option_params.expiry)[
                    d_i
                ]
                delta_idx: DualTypes | None = d_i
            else:
                vol_value = _validate_obj_not_no_input(vol, "vol")  # type: ignore[assignment]
                delta_idx = None
            u: DualTypes = _moneyness_from_delta_closed_form(
                delta, vol_value, t_e, z_w_0, self.fx_option_params.direction
            )
            return delta_idx, vol_value, u * f
        # then delta types are both adjusted, use 1-d solver.
        elif eta_0 == eta_1 and eta_0 == -0.5:
            u = _moneyness_from_delta_one_dimensional(
                delta,
                delta_type,
                vol_delta_type,
                _get_vol_smile_or_value(vol, self.fx_option_params.expiry),
                t_e,
                z_w,
                self.fx_option_params.direction,
            )
            delta_idx = (-z_w_1 / z_w_0) * (
                delta - z_w_0 * u * (self.fx_option_params.direction + 1.0) * 0.5
            )
        else:  # delta adjustment types are different, use 2-d solver.
            u, delta_idx = _moneyness_from_delta_two_dimensional(
                delta,
                delta_type,
                _get_vol_smile_or_raise(vol, self.fx_option_params.expiry),
                t_e,
                z_w,
                self.fx_option_params.direction,
            )

        _1: DualTypes | None = delta_idx
        _2: DualTypes = u * f
        if isinstance(vol, FXDeltaVolSmile | FXDeltaVolSurface):
            vol_value = _get_vol_smile_or_raise(vol, self.fx_option_params.expiry)[delta_idx]
        else:
            vol_value = _validate_obj_not_no_input(vol, "vol")  # type: ignore[assignment]
        return _1, vol_value, _2

    def _index_vol_and_strike_from_delta_sabr(
        self,
        delta: float,
        delta_type: FXDeltaMethod,
        vol: FXSabrSmile | FXSabrSurface,
        z_w: DualTypes,
        f: DualTypes | FXForwards,
    ) -> tuple[DualTypes | None, DualTypes, DualTypes]:
        eta_0, z_w_0, _ = _delta_type_constants(delta_type, z_w, 0.0)  # u: unused
        t_e = (self.fx_option_params.expiry - vol.meta.eval_date).days / 365.0
        sqrt_t = t_e**0.5
        if isinstance(f, FXForwards):
            f_d: DualTypes = f.rate(self.fx_option_params.pair, self.fx_option_params.delivery)
            # _ad = _set_ad_order_objects([0], [f])  # GH755
        else:
            f_d = f  # type: ignore[assignment]

        def root1d(
            k: DualTypes,
            f_d: DualTypes,
            fx: FXForwards | DualTypes,
            z_w_0: DualTypes,
            delta: float,
            as_float: bool,
        ) -> tuple[DualTypes, DualTypes]:
            # if not as_float and isinstance(fx, FXForwards):
            #     _set_ad_order_objects(_ad, [fx])

            sigma, dsigma_dk = vol._d_sabr_d_k_or_f(
                k=k, f=fx, expiry=self.fx_option_params.expiry, as_float=as_float, derivative=1
            )
            dn0 = -dual_log(k / f_d) / (sigma * sqrt_t) + eta_0 * sigma * sqrt_t
            Phi = dual_norm_cdf(self.fx_option_params.direction * dn0)

            if eta_0 == -0.5:
                z_u_0, dz_u_dk = k / f_d, 1 / f_d
                d_1 = -dz_u_dk * z_w_0 * self.fx_option_params.direction * Phi
            else:
                z_u_0, dz_u_dk = 1.0, 0.0
                d_1 = 0.0

            ddn_dk = (dual_log(k / f_d) / (sigma**2 * sqrt_t) + eta_0 * sqrt_t) * dsigma_dk - 1 / (
                k * sigma * sqrt_t
            )
            d_2 = -z_u_0 * z_w_0 * dual_norm_pdf(self.fx_option_params.direction * dn0) * ddn_dk

            f0 = delta - z_w_0 * z_u_0 * self.fx_option_params.direction * Phi
            f1 = d_1 + d_2
            return f0, f1

        g01 = delta if self.fx_option_params.direction > 0 else max(delta, -0.75)
        if isinstance(vol, FXSabrSmile):
            alpha = vol.nodes.alpha
        else:  # FXSabrSurface
            vol_: FXSabrSurface = vol  # type: ignore[assignment]
            expiry_posix = self.fx_option_params.expiry.replace(tzinfo=UTC).timestamp()
            e_idx, _ = _surface_index_left(vol_.meta.expiries_posix, expiry_posix)
            alpha = vol_.smiles[e_idx].nodes.alpha

        g0 = (
            _moneyness_from_delta_closed_form(
                g01, alpha * 100.0, t_e, z_w_0, self.fx_option_params.direction
            )
            * f_d
        )

        root_solver = newton_1dim(
            root1d,
            g0,
            args=(f_d, f, z_w_0, delta),
            pre_args=(True,),  # solve iterations `as_float`
            final_args=(False,),  # solve final iteration with AD
            raise_on_fail=True,
        )

        k: DualTypes = root_solver["g"]
        v_ = vol.get_from_strike(k, f, self.fx_option_params.expiry)[1]
        return None, v_, k

    def _payoff_at_expiry(
        self, rng: tuple[float, float] | NoInput = NoInput(0)
    ) -> tuple[Arr1dF64, Arr1dF64]:
        # used by plotting methods
        if isinstance(self.fx_option_params.strike, NoInput):
            raise ValueError(
                "Cannot return payoff for option without a specified `strike`.",
            )  # pragma: no cover
        if isinstance(rng, NoInput):
            x = np.linspace(0, 20, 1001)
        else:
            x = np.linspace(rng[0], rng[1], 1001)
        k: float = _dual_float(self.fx_option_params.strike)
        _ = (x - k) * self.fx_option_params.direction
        __ = np.zeros(1001)
        if self.fx_option_params.direction > 0:  # call
            y = np.where(x < k, __, _) * self.settlement_params.notional
        else:  # put
            y = np.where(x > k, __, _) * self.settlement_params.notional
        return x, y


class FXCallPeriod(_BaseFXOptionPeriod):
    r"""
    A *Period* defined by a European FX call option.

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
    ex_dividend: datetime, :green:`optional (set as 'delivery')`
        The ex-dividend date of the settled cashflow.

        .. note::

           This *Period* type has not implemented **indexation** or **non-deliverability**.

    """  # noqa: E501

    def __init__(
        self,
        *,
        # option params:
        delivery: datetime,  # otherwise termed the 'payment' of the period
        pair: str,
        expiry: datetime,
        strike: DualTypes_ = NoInput(0),
        notional: DualTypes_ = NoInput(0),
        delta_type: FXDeltaMethod | str_ = NoInput(0),
        metric: FXOptionMetric | str_ = NoInput(0),
        option_fixings: DualTypes | Series[DualTypes] | str_ = NoInput(0),  # type: ignore[type-var]
        # currency args:
        ex_dividend: datetime_ = NoInput(0),
    ) -> None:
        super().__init__(
            direction=OptionType.Call,
            delivery=delivery,
            pair=pair,
            expiry=expiry,
            strike=strike,
            notional=notional,
            delta_type=delta_type,
            metric=metric,
            option_fixings=option_fixings,
            ex_dividend=ex_dividend,
        )


class FXPutPeriod(_BaseFXOptionPeriod):
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
    ex_dividend: datetime, :green:`optional (set as 'delivery')`
        The ex-dividend date of the settled cashflow.

        .. note::

           This *Period* type has not implemented **indexation** or **non-deliverability**.

    """  # noqa: E501

    def __init__(
        self,
        *,
        # option params:
        delivery: datetime,  # otherwise termed the 'payment' of the period
        pair: str,
        expiry: datetime,
        strike: DualTypes_ = NoInput(0),
        notional: DualTypes_ = NoInput(0),
        delta_type: FXDeltaMethod | str_ = NoInput(0),
        metric: FXOptionMetric | str_ = NoInput(0),
        option_fixings: DualTypes | Series[DualTypes] | str_ = NoInput(0),  # type: ignore[type-var]
        # currency args:
        ex_dividend: datetime_ = NoInput(0),
    ) -> None:
        super().__init__(
            direction=OptionType.Put,
            delivery=delivery,
            pair=pair,
            expiry=expiry,
            strike=strike,
            notional=notional,
            delta_type=delta_type,
            metric=metric,
            option_fixings=option_fixings,
            ex_dividend=ex_dividend,
        )
