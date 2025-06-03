from __future__ import annotations

import abc
import warnings
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from pandas import DataFrame, Series

from rateslib import defaults
from rateslib.calendars import add_tenor, dcf
from rateslib.curves import Curve, LineCurve, average_rate, index_left, index_value
from rateslib.curves._parsers import (
    _disc_required_maybe_from_curve,
    _validate_curve_is_not_dict,
    _validate_curve_not_no_input,
)
from rateslib.default import NoInput, _drb
from rateslib.dual import Dual, Dual2, gradient, ift_1dim, newton_1dim, quadratic_eqn
from rateslib.dual.utils import _dual_float
from rateslib.instruments.base import Metrics
from rateslib.instruments.bonds.conventions import (
    BILL_MODE_MAP,
    BOND_MODE_MAP,
    BillCalcMode,
    BondCalcMode,
    _get_calc_mode_for_class,
)
from rateslib.instruments.sensitivities import Sensitivities

# from scipy.optimize import brentq
from rateslib.instruments.utils import (
    _get,
    _get_curves_fx_and_base_maybe_from_solver,
    _push,
    _update_with_defaults,
)
from rateslib.legs import FixedLeg, FloatLeg, IndexFixedLeg
from rateslib.periods import (
    FloatPeriod,
)
from rateslib.periods.utils import _maybe_local

if TYPE_CHECKING:
    from rateslib.instruments.bonds.conventions.accrued import AccrualFunction
    from rateslib.instruments.bonds.conventions.discounting import (
        CashflowFunction,
        YtmDiscountFunction,
    )
    from rateslib.typing import (
        FX_,
        NPV,
        Any,
        CalInput,
        Callable,
        Cashflow,
        Curve_,
        CurveOption,
        CurveOption_,
        Curves_,
        DualTypes,
        DualTypes_,
        FixedPeriod,
        FixingsRates_,
        IndexCashflow,
        IndexFixedPeriod,
        Number,
        Solver_,
        bool_,
        datetime_,
        int_,
        str_,
    )


class BondMixin:
    """Inheritable class to provide basic functionality."""

    leg1: FixedLeg | FloatLeg | IndexFixedLeg
    kwargs: dict[str, Any]
    calc_mode: BondCalcMode
    curves: Curves_
    rate: Callable[..., DualTypes]

    def _period_index(self, settlement: datetime) -> int:
        """
        Get the coupon period index for that which the settlement date fall within.
        Uses unadjusted dates.
        """
        _: int = index_left(
            self.leg1.schedule.uschedule,
            len(self.leg1.schedule.uschedule),
            settlement,
        )
        return _

    @abc.abstractmethod
    def _period_cashflow(
        self,
        period: Cashflow | FixedPeriod | FloatPeriod | IndexCashflow | IndexFixedPeriod,
        curve: CurveOption_,
    ) -> DualTypes:
        pass  # pragma: no cover

    # def _accrued_fraction(self, settlement: datetime, calc_mode: str_, acc_idx: int):
    #     """
    #     Return the accrual fraction of period between last coupon and settlement and
    #     coupon period left index.
    #
    #     Branches to a calculation based on the bond `calc_mode`.
    #     """
    #     try:
    #         func = getattr(self, f"_{calc_mode}")["accrual"]
    #         # func = getattr(self, self._acc_frac_mode_map[calc_mode])
    #         return func(settlement, acc_idx)
    #     except KeyError:
    #         raise ValueError(f"Cannot calculate for `calc_mode`: {calc_mode}")

    def _set_base_index_if_none(self, curve: CurveOption_) -> None:
        if type(self) is IndexFixedRateBond and isinstance(self.index_base, NoInput):
            curve_ = _validate_curve_not_no_input(_validate_curve_is_not_dict(curve))
            self.leg1.index_base = curve_.index_value(
                self.leg1.schedule.effective,
                self.leg1.index_lag,
                self.leg1.index_method,
            )

    def ex_div(self, settlement: datetime) -> bool:
        """
        Return a boolean whether the security is ex-div at the given settlement.

        Parameters
        ----------
        settlement : datetime
            The settlement date to test.

        Returns
        -------
        bool

        Notes
        -----
        By default uses the UK DMO convention of returning *False* if ``settlement``
        **is on or before** the ex-div date.

        Some ``calc_mode`` options return *True* if ``settlement`` **is on** the ex-div date.

        Ex-div dates are determined as measured by the number of ``ex_div`` business days prior
        to the unadjusted coupon end date.

        With an ``ex_div`` of 1, a ``settlement`` that occurs on the coupon payment date will be
        classified as ex-dividend and not receive that coupon.

        With an ``ex_div`` of 0, a ``settlement`` that occurs on the coupon payment date will
        **not** be classified as ex-dividend and will receive that coupon (in the default
        calculation mode).
        """
        prev_a_idx = index_left(
            self.leg1.schedule.uschedule,
            len(self.leg1.schedule.uschedule),
            settlement,
        )
        ex_div_date = self.leg1.schedule.calendar.lag(
            self.leg1.schedule.uschedule[prev_a_idx + 1],
            -self.kwargs["ex_div"],
            True,
        )
        if self.calc_mode in []:  # currently no identified calc_modes
            return settlement >= ex_div_date  # pragma: no cover
        else:
            return settlement > ex_div_date

    def _accrued(self, settlement: datetime, func: AccrualFunction) -> DualTypes:
        """func is the specific accrued function associated with the bond ``calc_mode``"""
        acc_idx = self._period_index(settlement)
        frac = func(self, settlement, acc_idx)
        if self.ex_div(settlement):
            frac = frac - 1  # accrued is negative in ex-div period
        _: DualTypes = self._period_cashflow(self.leg1._regular_periods[acc_idx], NoInput(0))
        return frac * _ / -self.leg1.notional * 100

    def _ytm(
        self,
        price: DualTypes,
        settlement: datetime,
        curve: CurveOption_,
        dirty: bool,
    ) -> Number:
        """
        Calculate the yield-to-maturity of the security given its price.

        Parameters
        ----------
        price : float, Dual, Dual2
            The price, per 100 nominal, against which to determine the yield.
        settlement : datetime
            The settlement date on which to determine the price.
        dirty : bool, optional
            If `True` will assume the
            :meth:`~rateslib.instruments.FixedRateBond.accrued` is included in the price.

        Returns
        -------
        float, Dual, Dual2

        Notes
        -----
        If ``price`` is given as :class:`~rateslib.dual.Dual` or
        :class:`~rateslib.dual.Dual2` input the result of the yield will be output
        as the same type with the variables passed through accordingly.

        """  # noqa: E501

        def s(g: DualTypes) -> DualTypes:
            return self._price_from_ytm(
                ytm=g, settlement=settlement, calc_mode=self.calc_mode, dirty=dirty, curve=curve
            )

        result = ift_1dim(
            s,
            s_tgt=price,
            h="ytm_quadratic",
            ini_h_args=(-3.0, 2.0, 12.0),
            func_tol=1e-9,
            conv_tol=1e-9,
            raise_on_fail=True,
        )
        return result["g"]  # type: ignore[no-any-return]

    def _price_from_ytm(
        self,
        ytm: DualTypes,
        settlement: datetime,
        calc_mode: str | BondCalcMode | NoInput,
        dirty: bool,
        curve: CurveOption_,
    ) -> DualTypes:
        """
        Loop through all future cashflows and discount them with ``ytm`` to achieve
        correct price.
        """
        calc_mode_ = _drb(self.calc_mode, calc_mode)
        if isinstance(calc_mode_, str):
            calc_mode_ = BOND_MODE_MAP[calc_mode_]
        try:
            return self._generic_price_from_ytm(
                ytm=ytm,
                settlement=settlement,
                dirty=dirty,
                f1=calc_mode_._v1,
                f2=calc_mode_._v2,
                f3=calc_mode_._v3,
                c1=calc_mode_._c1,
                ci=calc_mode_._ci,
                cn=calc_mode_._cn,
                accrual=calc_mode_._ytm_accrual,
                curve=curve,
            )
        except KeyError:
            raise ValueError(f"Cannot calculate with `calc_mode`: {calc_mode}")

    def _generic_price_from_ytm(
        self,
        ytm: DualTypes,
        settlement: datetime,
        dirty: bool,
        f1: YtmDiscountFunction,
        f2: YtmDiscountFunction,
        f3: YtmDiscountFunction,
        c1: CashflowFunction,
        ci: CashflowFunction,
        cn: CashflowFunction,
        accrual: AccrualFunction,
        curve: CurveOption_,
    ) -> DualTypes:
        """
        Refer to supplementary material.

        Note: `curve` is only needed for FloatRate Periods on `_period_cashflow`
        """
        f: int = 12 / defaults.frequency_months[self.leg1.schedule.frequency]  # type: ignore[assignment]
        acc_idx: int = self._period_index(settlement)
        _is_ex_div: bool = self.ex_div(settlement)
        if settlement == self.leg1.schedule.uschedule[acc_idx + 1]:
            # then settlement aligns with a cashflow: manually adjust to next period
            _is_ex_div = False
            acc_idx += 1

        v2 = f2(self, ytm, f, settlement, acc_idx, None, accrual, -100000)
        v1 = f1(self, ytm, f, settlement, acc_idx, v2, accrual, acc_idx)
        v3 = f3(
            self,
            ytm,
            f,
            settlement,
            self.leg1.schedule.n_periods - 1,
            v2,
            accrual,
            self.leg1.schedule.n_periods - 1,
        )

        # Sum up the coupon cashflows discounted by the calculated factors
        d: DualTypes = 0.0
        n = self.leg1.schedule.n_periods
        for i, p_idx in enumerate(range(acc_idx, n)):
            if i == 0 and _is_ex_div:
                # no coupon cashflow is received so no addition to the sum
                continue
            elif i == 0:
                # then this is the first period: c1 and v1 are used
                cf1 = c1(self, ytm, f, acc_idx, p_idx, n, curve)
                d += cf1 * v1
            elif p_idx == (self.leg1.schedule.n_periods - 1):
                # then this is last period, but it is not the first (i>0).
                # cn and v3 are relevant, but v1 is also used, and if i > 1 then v2 is also used.
                cfn = cn(self, ytm, f, acc_idx, p_idx, n, curve)
                d += cfn * v2 ** (i - 1) * v3 * v1
            else:
                # this is not the first and not the last period.
                # ci and v2i are relevant, but v1 is also required and v2 may also be used if i > 1.
                # v2i allows for a per-period adjustment to the v2 discount factor, e.g. BTPs.
                cfi = ci(self, ytm, f, acc_idx, p_idx, n, curve)
                v2i = f2(self, ytm, f, settlement, acc_idx, v2, accrual, p_idx)
                d += cfi * v2 ** (i - 1) * v2i * v1

        # Add the redemption payment discounted by relevant factors
        redemption: Cashflow | IndexCashflow = self.leg1._exchange_periods[1]  # type: ignore[assignment]
        if i == 0:  # only looped 1 period, only use the last discount
            d += self._period_cashflow(redemption, curve) * v1
        elif i == 1:  # only looped 2 periods, no need for v2
            d += self._period_cashflow(redemption, curve) * v3 * v1
        else:  # looped more than 2 periods, regular formula applied
            d += self._period_cashflow(redemption, curve) * v2 ** (i - 1) * v3 * v1

        # discount all by the first period factor and scaled to price
        p = d / -self.leg1.notional * 100

        return p if dirty else p - self._accrued(settlement, accrual)

    def fwd_from_repo(
        self,
        price: DualTypes,
        settlement: datetime,
        forward_settlement: datetime,
        repo_rate: DualTypes,
        convention: str_ = NoInput(0),
        dirty: bool = False,
        method: str = "proceeds",
    ) -> DualTypes:
        """
        Return a forward price implied by a given repo rate.

        Parameters
        ----------
        price : float, Dual, or Dual2
            The initial price of the security at ``settlement``.
        settlement : datetime
            The settlement date of the bond
        forward_settlement : datetime
            The forward date for which to calculate the forward price.
        repo_rate : float, Dual or Dual2
            The rate which is used to calculate values.
        convention : str, optional
            The day count convention applied to the rate. If not given uses default
            values.
        dirty : bool, optional
            Whether the input and output price are specified including accrued interest.
        method : str in {"proceeds", "compounded"}, optional
            The method for determining the forward price.

        Returns
        -------
        float, Dual or Dual2

        Notes
        -----
        Any intermediate (non ex-dividend) cashflows between ``settlement`` and
        ``forward_settlement`` will also be assumed to accrue at ``repo_rate``.
        """
        convention_ = _drb(defaults.convention, convention)
        dcf_ = dcf(settlement, forward_settlement, convention_)
        if not dirty:
            d_price = price + self._accrued(settlement, self.calc_mode._settle_accrual)
        else:
            d_price = price
        if self.leg1.amortization != 0:
            raise NotImplementedError(
                "method for forward price not available with amortization",
            )  # pragma: no cover
        total_rtn = d_price * (1 + repo_rate * dcf_ / 100) * -self.leg1.notional / 100

        # now systematically deduct coupons paid between settle and forward settle
        settlement_idx = index_left(
            self.leg1.schedule.aschedule,
            self.leg1.schedule.n_periods + 1,
            settlement,
        )
        fwd_settlement_idx = index_left(
            self.leg1.schedule.aschedule,
            self.leg1.schedule.n_periods + 1,
            forward_settlement,
        )

        # do not accrue a coupon not received
        settlement_idx += 1 if self.ex_div(settlement) else 0
        # deduct final coupon if received within period
        fwd_settlement_idx += 1 if self.ex_div(forward_settlement) else 0

        for p_idx in range(settlement_idx, fwd_settlement_idx):
            # deduct accrued coupon from dirty price
            c_period = self.leg1._regular_periods[p_idx]
            c_cashflow: DualTypes = c_period.cashflow  # type: ignore[assignment]
            # TODO handle FloatPeriod cashflow fetch if need a curve.
            if method.lower() == "proceeds":
                dcf_ = dcf(c_period.payment, forward_settlement, convention_)
                accrued_coup = c_cashflow * (1 + dcf_ * repo_rate / 100)
                total_rtn -= accrued_coup
            elif method.lower() == "compounded":
                r_bar, d, _ = average_rate(
                    settlement, forward_settlement, convention_, repo_rate, dcf_
                )
                n = (forward_settlement - c_period.payment).days
                accrued_coup = c_cashflow * (1 + d * r_bar / 100) ** n
                total_rtn -= accrued_coup
            else:
                raise ValueError("`method` must be in {'proceeds', 'compounded'}.")

        forward_price: DualTypes = total_rtn / -self.leg1.notional * 100
        if dirty:
            return forward_price
        else:
            return forward_price - self._accrued(forward_settlement, self.calc_mode._settle_accrual)

    def repo_from_fwd(
        self,
        price: DualTypes,
        settlement: datetime,
        forward_settlement: datetime,
        forward_price: DualTypes,
        convention: str_ = NoInput(0),
        dirty: bool = False,
    ) -> DualTypes:
        """
        Return an implied repo rate from a forward price.

        Parameters
        ----------
        price : float, Dual, or Dual2
            The initial price of the security at ``settlement``.
        settlement : datetime
            The settlement date of the bond
        forward_settlement : datetime
            The forward date for which to calculate the forward price.
        forward_price : float, Dual or Dual2
            The forward price which iplies the repo rate
        convention : str, optional
            The day count convention applied to the rate. If not given uses default
            values.
        dirty : bool, optional
            Whether the input and output price are specified including accrued interest.

        Returns
        -------
        float, Dual or Dual2

        Notes
        -----
        Any intermediate (non ex-dividend) cashflows between ``settlement`` and
        ``forward_settlement`` will also be assumed to accrue at ``repo_rate``.
        """
        convention_ = _drb(defaults.convention, convention)
        # forward price from repo is linear in repo_rate so reverse calculate with AD
        if not dirty:
            p_t = forward_price + self._accrued(forward_settlement, self.calc_mode._settle_accrual)
            p_0 = price + self._accrued(settlement, self.calc_mode._settle_accrual)
        else:
            p_t, p_0 = forward_price, price

        dcf_ = dcf(settlement, forward_settlement, convention_)
        numerator = p_t - p_0
        denominator = p_0 * dcf_

        # now systematically deduct coupons paid between settle and forward settle
        settlement_idx = index_left(
            self.leg1.schedule.aschedule,
            self.leg1.schedule.n_periods + 1,
            settlement,
        )
        fwd_settlement_idx = index_left(
            self.leg1.schedule.aschedule,
            self.leg1.schedule.n_periods + 1,
            forward_settlement,
        )

        # do not accrue a coupon not received
        settlement_idx += 1 if self.ex_div(settlement) else 0
        # deduct final coupon if received within period
        fwd_settlement_idx += 1 if self.ex_div(forward_settlement) else 0

        for p_idx in range(settlement_idx, fwd_settlement_idx):
            # deduct accrued coupon from dirty price
            c_period = self.leg1._regular_periods[p_idx]
            c_cashflow: DualTypes = c_period.cashflow  # type: ignore[assignment]
            # TODO handle FloatPeriod if it needs a Curve to forecast cashflow
            dcf_ = dcf(c_period.payment, forward_settlement, convention_)
            numerator += 100 * c_cashflow / -self.leg1.notional
            denominator -= 100 * dcf_ * c_cashflow / -self.leg1.notional

        return numerator / denominator * 100

    def _npv_local(
        self,
        curve: CurveOption_,
        disc_curve: Curve,
        settlement: datetime,
        projection: datetime_,
    ) -> DualTypes:
        """
        Return the NPV (local) of the security by summing cashflow valuations.

        Parameters
        ----------
        curve : Curve or LineCurve
            A curve used for projecting cashflows of floating rates.
        disc_curve : Curve, str or list of such
            A single :class:`Curve` for discounting cashflows.
        fx : float, FXRates, FXForwards, optional
            The immediate settlement FX rate that will be used to convert values
            into another currency. A given `float` is used directly. If giving a
            ``FXRates`` or ``FXForwards`` object, converts from local currency
            into ``base``.
        base : str, optional
            The base currency to convert cashflows into (3-digit code), set by default.
            Only used if ``fx`` is an ``FXRates`` or ``FXForwards`` object.
        settlement : datetime
            The date of settlement of the bond which declares which cashflows are
            unpaid and therefore valid for the calculation.
        projection : datetime, optional
           Curves discount cashflows to the initial node of the Curve. This parameter
           allows the NPV to be projected forward to a future date under the appropriate
           discounting mechanism. If *None* is not projected forward.

        Returns
        -------
        float, Dual, Dual2

        Notes
        -----
        The cashflows for determination (excluding an ``ex_div`` cashflow) are
        evaluated by ``settlement``.

        The date for which the PV is returned is by ``projection``, and not the
        initial node date of the ``disc_curve``.
        """
        self._set_base_index_if_none(curve)
        npv: DualTypes = self.leg1.npv(curve, disc_curve, NoInput(0), NoInput(0), local=False)  # type: ignore[assignment]

        # now must systematically deduct any cashflow between the initial node date
        # and the settlement date, including the cashflow after settlement if ex_div.
        initial_idx = index_left(
            self.leg1.schedule.aschedule,
            self.leg1.schedule.n_periods + 1,
            disc_curve.nodes.initial,
        )
        settle_idx = index_left(
            self.leg1.schedule.aschedule,
            self.leg1.schedule.n_periods + 1,
            settlement,
        )

        for period_idx in range(initial_idx, settle_idx):
            # deduct coupon period
            npv -= self.leg1.periods[period_idx].npv(  # type: ignore[operator]
                curve, disc_curve, NoInput(0), NoInput(0), local=False
            )

        if self.ex_div(settlement):
            # deduct coupon after settlement which is also unpaid
            npv -= self.leg1.periods[settle_idx].npv(  # type: ignore[operator]
                curve, disc_curve, NoInput(0), NoInput(0), local=False
            )

        if isinstance(projection, NoInput):
            return npv
        else:
            return npv / disc_curve[projection]

    def npv(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
        local: bool = False,
    ) -> NPV:
        """
        Return the NPV of the security by summing cashflow valuations.

        Parameters
        ----------
        curves : Curve, str or list of such
            A single :class:`Curve` or id or a list of such. A list defines the
            following curves in the order:

              - Forecasting :class:`Curve` for ``leg1``.
              - Discounting :class:`Curve` for ``leg1``.
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
        local : bool, optional
            If `True` will ignore the ``base`` request and return a dict identifying
            local currency NPV.

        Returns
        -------
        float, Dual, Dual2 or dict of such

        Notes
        -----
        The ``settlement`` date of the bond is inferred from the objects ``settle``
        days parameter and the initial date of the supplied ``curves``.
        The NPV returned is for immediate settlement.

        If **only one curve** is given this is used as all four curves.

        If **two curves** are given the forecasting curve is used as the forecasting
        curve on both legs and the discounting curve is used as the discounting
        curve for both legs.
        """
        curves_, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            fx,
            base,
            self.leg1.currency,
        )
        curves_1 = _validate_curve_not_no_input(curves_[1])
        settlement = self.leg1.schedule.calendar.lag(
            curves_1.nodes.initial,
            self.kwargs["settle"],
            True,
        )
        npv = self._npv_local(curves_[0], curves_1, settlement, NoInput(0))
        return _maybe_local(npv, local, self.leg1.currency, fx_, base_)

    def analytic_delta(
        self,
        curve: CurveOption_ = NoInput(0),
        disc_curve: Curve_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
    ) -> DualTypes:
        """
        Return the analytic delta of the security via summing all periods.

        For arguments see :meth:`~rateslib.periods.BasePeriod.analytic_delta`.
        """
        disc_curve_ = _disc_required_maybe_from_curve(curve, disc_curve)
        settlement = self.leg1.schedule.calendar.lag(
            disc_curve_.nodes.initial,
            self.kwargs["settle"],
            True,
        )
        a_delta = self.leg1.analytic_delta(curve, disc_curve_, fx, base)
        if self.ex_div(settlement):
            # deduct the next coupon which has otherwise been included in valuation
            current_period = index_left(
                self.leg1.schedule.aschedule,
                self.leg1.schedule.n_periods + 1,
                settlement,
            )
            a_delta -= self.leg1.periods[current_period].analytic_delta(
                curve,  # type: ignore[arg-type]
                disc_curve_,
                fx,
                base,
            )
        return a_delta

    def cashflows(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
    ) -> DataFrame:
        """
        Return the properties of the security used in calculating cashflows.

        Parameters
        ----------
        curves : Curve, str or list of such
            A single :class:`Curve` or id or a list of such. A list defines the
            following curves in the order:

              - Forecasting :class:`Curve` for ``leg1``.
              - Discounting :class:`Curve` for ``leg1``.
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
            Only used if ``fx_rate`` is an ``FXRates`` or ``FXForwards`` object.
        settlement : datetime, optional
            The settlement date of the security. If *None* adds the regular ``settle``
            time to the initial node date of the given discount ``curves``.

        Returns
        -------
        DataFrame
        """
        curves_, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            fx,
            base,
            self.leg1.currency,
        )
        self._set_base_index_if_none(curves_[0])

        if isinstance(settlement, NoInput):
            if isinstance(curves_[1], NoInput):
                settlement_ = self.leg1.schedule.effective
            else:
                settlement_ = self.leg1.schedule.calendar.lag(
                    curves_[1].nodes.initial,
                    self.kwargs["settle"],
                    True,
                )
        else:
            settlement_ = settlement
        cashflows = self.leg1.cashflows(curves_[0], curves_[1], fx_, base_)
        if self.ex_div(settlement_):
            # deduct the next coupon which has otherwise been included in valuation
            current_period = index_left(
                self.leg1.schedule.aschedule,
                self.leg1.schedule.n_periods + 1,
                settlement_,
            )
            cashflows.loc[current_period, defaults.headers["npv"]] = 0
            cashflows.loc[current_period, defaults.headers["npv_fx"]] = 0
        return cashflows

    def oaspread(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
        price: DualTypes_ = NoInput(0),
        dirty: bool = False,
    ) -> DualTypes:
        """
        The option adjusted spread added to the discounting *Curve* to value the security
        at ``price``.

        Parameters
        ----------
        curves : Curve, str or list of such
            A single :class:`Curve` or id or a list of such. A list defines the
            following curves in the order:

              - Forecasting :class:`Curve` for ``leg1``.
              - Discounting :class:`Curve` for ``leg1``.
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
        price : float, Dual, Dual2
            The price of the bond to match.
        dirty : bool
            Whether the price is given clean or dirty.

        Returns
        -------
        float, Dual, Dual2
        """
        if isinstance(price, NoInput):
            raise ValueError("`price` must be supplied in order to derive the `oaspread`.")

        curves_, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            fx,
            base,
            self.leg1.currency,
        )
        metric = "dirty_price" if dirty else "clean_price"

        return self._oaspread_algorithm(
            curves_[0], _validate_curve_not_no_input(curves_[1]), metric, _dual_float(price)
        )

    def _oaspread_algorithm(
        self, curve: CurveOption_, disc_curve: Curve, metric: str, price: float
    ) -> float:
        """
        Perform the algorithm as specified in "Coding Interest Rates" to derive an OAS spread
        balancing performance of the iteration with accuracy.

        Not AD safe, returns a float.

        Parameters
        ----------
        curve
        disc_curve
        metric
        price

        Returns
        -------
        float
        """

        def _copy_curve(curve: CurveOption_) -> CurveOption_:
            if isinstance(curve, NoInput) or curve is None:
                return NoInput(0)
            elif isinstance(curve, dict):
                return {k: v.copy() for k, v in curve.items()}
            else:
                return curve.copy()

        def _set_ad_order_of_forecasting_curve(curve: CurveOption_, order: int) -> None:
            if isinstance(curve, NoInput):
                pass
            elif isinstance(curve, dict):
                for _k, v in curve.items():
                    v._set_ad_order(order)
            else:
                curve._set_ad_order(order)

        # attach "z_spread" sensitivity to an AD order 1 curve.
        disc_curve_ = disc_curve._shift(
            Dual(0.0, ["z_spread"], []),
            composite=False,
            _no_validation=True,
        )
        curve_ = _copy_curve(curve)
        _set_ad_order_of_forecasting_curve(curve_, 0)

        # find a first order approximation of z, z_hat, using a Dual approach:
        npv_price: Dual | Dual2 = self.rate(curves=[curve_, disc_curve_], metric=metric)  # type: ignore[assignment]
        b: float = gradient(npv_price, ["z_spread"], 1)[0]
        c: float = _dual_float(npv_price) - price
        z_hat: float = -c / b

        # shift the curve to the first order approximation and fine tune with 2nd order approxim.
        disc_curve_ = disc_curve._shift(
            Dual2(z_hat, ["z_spread"], [], []),
            composite=False,
            _no_validation=True,
        )
        npv_price = self.rate(curves=[curve_, disc_curve_], metric=metric)  # type: ignore[assignment]
        coeffs: tuple[float, float, float] = (
            0.5 * gradient(npv_price, ["z_spread"], 2)[0][0],
            gradient(npv_price, ["z_spread"], 1)[0],
            _dual_float(npv_price) - price,
        )
        z_hat2: float = quadratic_eqn(*coeffs, x0=-c / b)["g"]

        # perform one final approximation albeit the additional price calculation slows calc time
        disc_curve_ = disc_curve._shift(
            z_hat + z_hat2,
            composite=False,
            _no_validation=True,
        )
        disc_curve_._set_ad_order(0)
        _set_ad_order_of_forecasting_curve(curve_, 0)
        npv_price_: float = self.rate(curves=[curve_, disc_curve_], metric=metric)  # type: ignore[assignment]
        b = coeffs[1] + 2 * coeffs[0] * z_hat2  # forecast the new gradient
        c = npv_price_ - price
        z_hat3: float = -c / b

        z = z_hat + z_hat2 + z_hat3
        return z

    # TODO: unit tests for the oaspread_newton algo, and derive the analytics to keep this AD safe
    def _oaspread_newton_algorithm(
        self, curve: CurveOption_, disc_curve: Curve, metric: str, price: float
    ) -> DualTypes:
        """
        NOT FULLY CHECKED or TESTED: DO NOT USE

        Perform the algorithm as specified in "Coding Interest Rates" to derive an OAS spread
        balancing performance of the iteration with accuracy.

        Not AD safe, returns a float.

        Parameters
        ----------
        curve
        disc_curve
        metric
        price

        Returns
        -------
        float
        """

        def _copy_curve(curve: CurveOption_) -> CurveOption_:
            if isinstance(curve, NoInput) or curve is None:
                return NoInput(0)
            elif isinstance(curve, dict):
                return {k: v.copy() for k, v in curve.items()}
            else:
                return curve.copy()

        curve_ = _copy_curve(curve)

        def root(z: DualTypes, P_tgt: DualTypes) -> tuple[DualTypes, float]:
            if isinstance(z, Dual):
                vars_ = [_ for _ in z.vars if _ != "__z_spd__§"]
                z_: DualTypes = Dual(float(z), vars=vars_, dual=gradient(z, vars=vars_, order=1))  # type: ignore[arg-type]
                z_ += Dual(0.0, ["__z_spd__§"], [])
            else:
                z_ = z + Dual(0.0, ["__z_spd__§"], [])

            shifted_curve = disc_curve.shift(z_, composite=False)
            P_iter: Dual | Dual2 = self.rate(curves=[curve_, shifted_curve], metric=metric)  # type: ignore[assignment]
            f_0 = P_tgt - P_iter
            f_1 = -gradient(P_iter, vars=["__z_spd__§"], order=1)[0]
            return f_0, f_1

        soln = newton_1dim(root, 0.0, 10, 1e-7, 1e-5, (price,), raise_on_fail=False)
        _: DualTypes = soln["g"]
        return _


class FixedRateBond(Sensitivities, BondMixin, Metrics):  # type: ignore[misc]
    # TODO (mid) ensure calculations work for amortizing bonds.
    """
    Create a fixed rate bond security.

    Parameters
    ----------
    effective : datetime
        The adjusted or unadjusted effective date.
    termination : datetime or str
        The adjusted or unadjusted termination date. If a string, then a tenor must be
        given expressed in days (`"D"`), months (`"M"`) or years (`"Y"`), e.g. `"48M"`.
    frequency : str in {"M", "B", "Q", "T", "S", "A"}, optional
        The frequency of the schedule. "Z" is **not** permitted. For zero-coupon-bonds use a
        ``fixed_rate`` of zero and set the frequency according to the yield-to-maturity
        convention required.
    stub : str combining {"SHORT", "LONG"} with {"FRONT", "BACK"}, optional
        The stub type to enact on the swap. Can provide two types, for
        example "SHORTFRONTLONGBACK".
    front_stub : datetime, optional
        An adjusted or unadjusted date for the first stub period.
    back_stub : datetime, optional
        An adjusted or unadjusted date for the back stub period.
        See notes for combining ``stub``, ``front_stub`` and ``back_stub``
        and any automatic stub inference.
    roll : int in [1, 31] or str in {"eom", "imm", "som"}, optional
        The roll day of the schedule. Inferred if not given.
    eom : bool, optional
        Use an end of month preference rather than regular rolls for inference. Set by
        default. Not required if ``roll`` is specified.
    modifier : str, optional
        The modification rule, in {"F", "MF", "P", "MP"}
    calendar : calendar or str, optional
        The holiday calendar object to use. If str, looks up named calendar from
        static data.
    payment_lag : int, optional
        The number of business days to lag payments by.
    notional : float, optional
        The leg notional, which is applied to each period.
    currency : str, optional
        The currency of the leg (3-digit code).
    amortization: float, optional
        The amount by which to adjust the notional each successive period. Should have
        sign equal to that of notional if the notional is to reduce towards zero.
    convention: str, optional
        The day count convention applied to calculations of period accrual dates.
        See :meth:`~rateslib.calendars.dcf`.
    fixed_rate : float, optional
        The **coupon** rate applied to determine cashflows. Can be set
        to `None` and designated
        later, perhaps after a mid-market rate for all periods has been calculated.
    ex_div : int
        The number of days prior to a cashflow during which the bond is considered
        ex-dividend.
    settle : int
        The number of business days for regular settlement time, i.e, 1 is T+1.
    calc_mode : str or BondCalcMode
        A calculation mode for dealing with bonds under different conventions. See notes.
    curves : CurveType, str or list of such, optional
        A single *Curve* or string id or a list of such.

        A list defines the following curves in the order:

        - Forecasting *Curve* for ``leg1``.
        - Discounting :class:`~rateslib.curves.Curve` for ``leg1``.
    spec : str, optional
        An identifier to pre-populate many field with conventional values. See
        :ref:`here<defaults-doc>` for more info and available values.
    metric: str, optional
        The pricing metric returned by the ``rate`` method of the *Instrument*.

    Attributes
    ----------
    ex_div_days : int
    settle : int
    curves : str, list, CurveType
    leg1 : FixedLeg

    Notes
    -----

    **Calculation Modes**

    The ``calc_mode`` parameter allows the calculation for **yield-to-maturity** and
    **accrued interest** to branch depending upon the particular convention of different bonds.
    See the documentation for the :class:`~rateslib.instruments.BondCalcMode` class.

    Calculation modes that have been preconfigured, and are available, can be
    found at :ref:`Securities Defaults <defaults-securities-input>`.

    Examples
    --------
    This example is taken from the UK debt management office (DMO) website. A copy of
    which is available :download:`here<_static/ukdmoyldconv.pdf>`.

    We demonstrate the use of **analogue methods** which do not need *Curves* or
    *Solvers*,
    :meth:`~rateslib.instruments.FixedRateBond.price`,
    :meth:`~rateslib.instruments.FixedRateBond.ytm`,
    :meth:`~rateslib.instruments.FixedRateBond.ex_div`,
    :meth:`~rateslib.instruments.FixedRateBond.accrued`,
    :meth:`~rateslib.instruments.FixedRateBond.repo_from_fwd`
    :meth:`~rateslib.instruments.FixedRateBond.fwd_from_repo`
    :meth:`~rateslib.instruments.FixedRateBond.duration`,
    :meth:`~rateslib.instruments.FixedRateBond.convexity`.

    .. ipython:: python

       gilt = FixedRateBond(
           effective=dt(1998, 12, 7),
           termination=dt(2015, 12, 7),
           frequency="S",
           calendar="ldn",
           currency="gbp",
           convention="ActActICMA",
           ex_div=7,
           settle=1,
           fixed_rate=8.0,
           notional=-1e6,  # negative notional receives fixed, i.e. buys a bond
           curves="gilt_curve",
       )
       gilt.ex_div(dt(1999, 5, 27))
       gilt.price(ytm=4.445, settlement=dt(1999, 5, 27), dirty=True)
       gilt.ytm(price=141.070132, settlement=dt(1999, 5, 27), dirty=True)
       gilt.accrued(dt(1999, 5, 27))
       gilt.fwd_from_repo(
           price=141.070132,
           settlement=dt(1999, 5, 27),
           forward_settlement=dt(2000, 2, 27),
           repo_rate=4.5,
           convention="Act365F",
           dirty=True,
       )
       gilt.repo_from_fwd(
           price=141.070132,
           settlement=dt(1999, 5, 27),
           forward_settlement=dt(2000, 2, 27),
           forward_price=141.829943,
           convention="Act365F",
           dirty=True,
       )
       gilt.duration(settlement=dt(1999, 5, 27), ytm=4.445, metric="risk")
       gilt.duration(settlement=dt(1999, 5, 27), ytm=4.445, metric="modified")
       gilt.convexity(settlement=dt(1999, 5, 27), ytm=4.445)


    The following **digital methods** consistent with the library's ecosystem are
    also available,
    :meth:`~rateslib.instruments.FixedRateBond.analytic_delta`,
    :meth:`~rateslib.instruments.FixedRateBond.rate`,
    :meth:`~rateslib.instruments.FixedRateBond.npv`,
    :meth:`~rateslib.instruments.FixedRateBond.cashflows`,
    :meth:`~rateslib.instruments.FixedRateBond.delta`,
    :meth:`~rateslib.instruments.FixedRateBond.gamma`.

    .. ipython:: python

       gilt_curve = Curve({dt(1999, 5, 26): 1.0, dt(2019, 5, 26): 1.0}, id="gilt_curve")
       instruments = [
           (gilt, (), {"metric": "ytm"}),
       ]
       solver = Solver(
           curves=[gilt_curve],
           instruments=instruments,
           s=[4.445],
           instrument_labels=["8% Dec15"],
           id="gilt_solver",
       )
       gilt.npv(solver=solver)
       gilt.analytic_delta(disc_curve=gilt_curve)
       gilt.rate(solver=solver, metric="clean_price")

    The sensitivities are also available. In this case the *Solver* is calibrated
    with *instruments* priced in yield terms so sensitivities are measured in basis
    points (bps).

    .. ipython:: python

       gilt.delta(solver=solver)
       gilt.gamma(solver=solver)

    The DataFrame of cashflows.

    .. ipython:: python

       gilt.cashflows(solver=solver)

    """  # noqa: E501

    _fixed_rate_mixin = True
    fixed_rate: DualTypes
    leg1: FixedLeg

    def _period_cashflow(self, period: Cashflow | FixedPeriod, curve: Curve_) -> DualTypes:  # type: ignore[override]
        """Nominal fixed rate bonds use the known "cashflow" attribute on the *Period*."""
        return period.cashflow  # type: ignore[return-value]  # FixedRate on bond cannot be NoInput

    def __init__(
        self,
        effective: datetime_ = NoInput(0),
        termination: datetime | str_ = NoInput(0),
        frequency: str_ = NoInput(0),
        stub: str_ = NoInput(0),
        front_stub: datetime_ = NoInput(0),
        back_stub: datetime_ = NoInput(0),
        roll: str | int_ = NoInput(0),
        eom: bool_ = NoInput(0),
        modifier: str_ = NoInput(0),
        calendar: CalInput = NoInput(0),
        payment_lag: int_ = NoInput(0),
        notional: DualTypes_ = NoInput(0),
        currency: str_ = NoInput(0),
        amortization: DualTypes_ = NoInput(0),
        convention: str_ = NoInput(0),
        fixed_rate: DualTypes_ = NoInput(0),
        ex_div: int_ = NoInput(0),
        settle: int_ = NoInput(0),
        calc_mode: BondCalcMode | BillCalcMode | str_ = NoInput(0),
        curves: Curves_ = NoInput(0),
        spec: str_ = NoInput(0),
        metric: str = "clean_price",
    ) -> None:
        self.kwargs = dict(
            effective=effective,
            termination=termination,
            frequency=frequency,
            stub=stub,
            front_stub=front_stub,
            back_stub=back_stub,
            roll=roll,
            eom=eom,
            modifier=modifier,
            calendar=calendar,
            payment_lag=payment_lag,
            notional=notional,
            currency=currency,
            amortization=amortization,
            convention=convention,
            fixed_rate=fixed_rate,
            initial_exchange=NoInput(0),
            final_exchange=NoInput(0),
            ex_div=ex_div,
            settle=settle,
            calc_mode=calc_mode,
            metric=metric,
        )
        self.kwargs = _push(spec, self.kwargs)

        # set defaults for missing values
        default_kwargs = dict(
            calc_mode=defaults.calc_mode[type(self).__name__],
            initial_exchange=False,
            final_exchange=True,
            payment_lag=defaults.payment_lag_specific[type(self).__name__],
            ex_div=defaults.ex_div,
            settle=defaults.settle,
        )
        self.kwargs = _update_with_defaults(self.kwargs, default_kwargs)

        if isinstance(self.kwargs["frequency"], NoInput):
            raise ValueError("`frequency` must be provided for Bond.")
        if isinstance(self.kwargs["fixed_rate"], NoInput):
            raise ValueError("`fixed_rate` must be provided for Bond.")
        # elif self.kwargs["frequency"].lower() == "z":
        #     raise ValueError("FixedRateBond `frequency` must be in {M, B, Q, T, S, A}.")

        self.calc_mode = _get_calc_mode_for_class(self, self.kwargs["calc_mode"])  # type: ignore[assignment]

        self.curves = curves
        self.spec = spec

        self._fixed_rate = fixed_rate
        self.leg1 = FixedLeg(
            **_get(self.kwargs, leg=1, filter=("ex_div", "settle", "calc_mode", "metric"))
        )

        if self.leg1.amortization != 0:
            # Note if amortization is added to FixedRateBonds must systematically
            # go through and update all methods. Many rely on the quantity
            # self.notional which is currently assumed to be a fixed quantity
            raise NotImplementedError("`amortization` for FixedRateBond must be zero.")

    # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
    # Commercial use of this code, and/or copying and redistribution is prohibited.
    # Contact rateslib at gmail.com if this code is observed outside its intended sphere.

    def accrued(self, settlement: datetime) -> DualTypes:
        """
        Calculate the accrued amount per nominal par value of 100.

        Parameters
        ----------
        settlement : datetime
            The settlement date which to measure accrued interest against.

        Notes
        -----
        Fractionally apportions the coupon payment based on calendar days.

        .. math::

           \\text{Accrued} = \\text{Coupon} \\times \\frac{\\text{Settle - Last Coupon}}{\\text{Next Coupon - Last Coupon}}

        """  # noqa: E501
        return self._accrued(settlement, self.calc_mode._settle_accrual)

    def rate(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
        metric: str_ = NoInput(0),
        forward_settlement: datetime_ = NoInput(0),
    ) -> DualTypes:
        """
        Return various pricing metrics of the security calculated from
        :class:`~rateslib.curves.Curve` s.

        Parameters
        ----------
        curves : Curve, str or list of such
            A single :class:`Curve` or id or a list of such. A list defines the
            following curves in the order:

              - Forecasting :class:`Curve` for ``leg1``.
              - Discounting :class:`Curve` for ``leg1``.
        solver : Solver, optional
            The numerical :class:`Solver` that constructs ``Curves`` from calibrating
            instruments.
        fx : float, FXRates, FXForwards, optional
            Not used by *FixedRateBond* rate. Output is in local currency.
        base : str, optional
            Not used by *FixedRateBond* rate. Output is in local currency.
        metric : str, optional
            Metric returned by the method. Available options are {"clean_price",
            "dirty_price", "ytm"}. Uses the *Instrument* default if not given.
        forward_settlement : datetime, optional
            The forward settlement date. If not given the settlement date is inferred from the
            discount *Curve* and the ``settle`` attribute.

        Returns
        -------
        float, Dual, Dual2
        """
        curves_, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            fx,
            base,
            self.leg1.currency,
        )
        curves_1 = _validate_curve_not_no_input(curves_[1])
        metric = _drb(self.kwargs["metric"], metric).lower()
        if metric in ["clean_price", "dirty_price", "ytm"]:
            if isinstance(forward_settlement, NoInput):
                settlement = self.leg1.schedule.calendar.lag(
                    curves_1.nodes.initial,
                    self.kwargs["settle"],
                    True,
                )
            else:
                settlement = forward_settlement
            npv = self._npv_local(curves_[0], curves_1, settlement, settlement)
            # scale price to par 100 (npv is already projected forward to settlement)
            dirty_price = npv * 100 / -self.leg1.notional

            if metric == "dirty_price":
                return dirty_price
            elif metric == "clean_price":
                return dirty_price - self.accrued(settlement)
            elif metric == "ytm":
                return self.ytm(dirty_price, settlement, True)

        raise ValueError("`metric` must be in {'dirty_price', 'clean_price', 'ytm'}.")

    # def par_spread(self, *args, price, settlement, dirty, **kwargs):
    #     """
    #     The spread to the fixed rate added to value the security at par valued from
    #     the given :class:`~rateslib.curves.Curve` s.
    #
    #     Parameters
    #     ----------
    #     args: tuple
    #         Positional arguments to :meth:`~rateslib.periods.BasePeriod.npv`.
    #     price: float
    #         The price of the security.
    #     settlement : datetime
    #         The settlement date.
    #     dirty : bool
    #         Whether the price given includes accrued interest.
    #     kwargs : dict
    #         Keyword arguments to :meth:`~rateslib.periods.BasePeriod.npv`.
    #
    #     Returns
    #     -------
    #     float, Dual, Dual2
    #     """
    #     TODO: calculate this par_spread formula.
    #     return (self.notional - self.npv(*args, **kwargs)) / self.analytic_delta(*args, **kwargs)

    def ytm(self, price: DualTypes, settlement: datetime, dirty: bool = False) -> Number:
        """
        Calculate the yield-to-maturity of the security given its price.

        Parameters
        ----------
        price : float, Dual, Dual2
            The price, per 100 nominal, against which to determine the yield.
        settlement : datetime
            The settlement date on which to determine the price.
        dirty : bool, optional
            If `True` will assume the
            :meth:`~rateslib.instruments.FixedRateBond.accrued` is included in the price.

        Returns
        -------
        float, Dual, Dual2

        Notes
        -----
        If ``price`` is given as :class:`~rateslib.dual.Dual` or
        :class:`~rateslib.dual.Dual2` input the result of the yield will be output
        as the same type with the variables passed through accordingly.

        Examples
        --------
        .. ipython:: python

           gilt = FixedRateBond(
               effective=dt(1998, 12, 7),
               termination=dt(2015, 12, 7),
               frequency="S",
               calendar="ldn",
               currency="gbp",
               convention="ActActICMA",
               ex_div=7,
               fixed_rate=8.0
           )
           gilt.ytm(
               price=141.0701315,
               settlement=dt(1999,5,27),
               dirty=True
           )
           gilt.ytm(Dual(141.0701315, ["price", "a", "b"], [1, -0.5, 2]), dt(1999, 5, 27), True)
           gilt.ytm(Dual2(141.0701315, ["price", "a", "b"], [1, -0.5, 2], []), dt(1999, 5, 27), True)

        """  # noqa: E501
        return self._ytm(price=price, settlement=settlement, dirty=dirty, curve=NoInput(0))

    def duration(self, ytm: DualTypes, settlement: datetime, metric: str = "risk") -> float:
        """
        Return the (negated) derivative of ``price`` w.r.t. ``ytm``.

        Parameters
        ----------
        ytm : float
            The yield-to-maturity for the bond.
        settlement : datetime
            The settlement date of the bond.
        metric : str
            The specific duration calculation to return. See notes.

        Returns
        -------
        float

        Notes
        -----
        The available metrics are:

        - *"risk"*: the derivative of price w.r.t. ytm, scaled to -1bp.

          .. math::

             risk = - \\frac{\\partial P }{\\partial y}

        - *"modified"*: the modified duration which is *risk* divided by price.

          .. math::

             mduration = \\frac{risk}{P} = - \\frac{1}{P} \\frac{\\partial P }{\\partial y}

        - *"duration"*: the duration which is modified duration reverse modified.

          .. math::

             duration = mduration \\times (1 + y / f)

        Examples
        --------
        .. ipython:: python

           gilt = FixedRateBond(
               effective=dt(1998, 12, 7),
               termination=dt(2015, 12, 7),
               frequency="S",
               calendar="ldn",
               currency="gbp",
               convention="ActActICMA",
               ex_div=7,
               fixed_rate=8.0
           )
           gilt.duration(4.445, dt(1999, 5, 27), "risk")
           gilt.duration(4.445, dt(1999, 5, 27), "modified")
           gilt.duration(4.445, dt(1999, 5, 27), "duration")

        This result is interpreted as cents. If the yield is increased by 1bp the price
        will fall by 14.65 cents.

        .. ipython:: python

           gilt.price(4.445, dt(1999, 5, 27))
           gilt.price(4.455, dt(1999, 5, 27))
        """
        # TODO: this is not AD safe: returns only float
        ytm_: float = _dual_float(ytm)
        if metric == "risk":
            price_dual: Dual = self.price(Dual(ytm_, ["y"], []), settlement)  # type: ignore[assignment]
            _: float = -gradient(price_dual, ["y"])[0]
        elif metric == "modified":
            price_dual = -self.price(Dual(ytm_, ["y"], []), settlement, dirty=True)  # type: ignore[assignment]
            _ = -gradient(price_dual, ["y"])[0] / float(price_dual) * 100
        elif metric == "duration":
            price_dual = self.price(Dual(ytm_, ["y"], []), settlement, dirty=True)  # type: ignore[assignment]
            f = 12 / defaults.frequency_months[self.kwargs["frequency"].upper()]
            v = 1 + ytm_ / (100 * f)
            _ = -gradient(price_dual, ["y"])[0] / float(price_dual) * v * 100
        return _

    def convexity(self, ytm: DualTypes, settlement: datetime) -> float:
        """
        Return the second derivative of ``price`` w.r.t. ``ytm``.

        Parameters
        ----------
        ytm : float
            The yield-to-maturity for the bond.
        settlement : datetime
            The settlement date of the bond.

        Returns
        -------
        float

        Examples
        --------
        .. ipython:: python

           gilt = FixedRateBond(
               effective=dt(1998, 12, 7),
               termination=dt(2015, 12, 7),
               frequency="S",
               calendar="ldn",
               currency="gbp",
               convention="ActActICMA",
               ex_div=7,
               fixed_rate=8.0
           )
           gilt.convexity(4.445, dt(1999, 5, 27))

        This number is interpreted as hundredths of a cent. For a 1bp increase in
        yield the duration will decrease by 2 hundredths of a cent.

        .. ipython:: python

           gilt.duration(4.445, dt(1999, 5, 27))
           gilt.duration(4.455, dt(1999, 5, 27))
        """
        # TODO: method is not AD safe: returns float
        ytm_: float = _dual_float(ytm)
        _ = self.price(Dual2(ytm_, ["_ytm__§"], [], []), settlement)
        return gradient(_, ["_ytm__§"], 2)[0][0]  # type: ignore[no-any-return, arg-type]

    def price(self, ytm: DualTypes, settlement: datetime, dirty: bool = False) -> DualTypes:
        """
        Calculate the price of the security per nominal value of 100, given
        yield-to-maturity.

        Parameters
        ----------
        ytm : float
            The yield-to-maturity against which to determine the price.
        settlement : datetime
            The settlement date on which to determine the price.
        dirty : bool, optional
            If `True` will include the
            :meth:`rateslib.instruments.FixedRateBond.accrued` in the price.

        Returns
        -------
        float, Dual, Dual2

        Examples
        --------
        This example is taken from the UK debt management office website.
        The result should be `141.070132` and the bond is ex-div.

        .. ipython:: python

           gilt = FixedRateBond(
               effective=dt(1998, 12, 7),
               termination=dt(2015, 12, 7),
               frequency="S",
               calendar="ldn",
               currency="gbp",
               convention="ActActICMA",
               ex_div=7,
               fixed_rate=8.0
           )
           gilt.ex_div(dt(1999, 5, 27))
           gilt.price(
               ytm=4.445,
               settlement=dt(1999, 5, 27),
               dirty=True
           )

        This example is taken from the Swedish national debt office website.
        The result of accrued should, apparently, be `0.210417` and the clean
        price should be `99.334778`.

        .. ipython:: python

           bond = FixedRateBond(
               effective=dt(2017, 5, 12),
               termination=dt(2028, 5, 12),
               frequency="A",
               calendar="stk",
               currency="sek",
               convention="ActActICMA",
               ex_div=5,
               fixed_rate=0.75
           )
           bond.ex_div(dt(2017, 8, 23))
           bond.accrued(dt(2017, 8, 23))
           bond.price(
               ytm=0.815,
               settlement=dt(2017, 8, 23),
               dirty=False
           )

        """
        return self._price_from_ytm(
            ytm=ytm, settlement=settlement, calc_mode=self.calc_mode, dirty=dirty, curve=NoInput(0)
        )

    def delta(self, *args: Any, **kwargs: Any) -> DataFrame:
        """
        Calculate the delta of the *Instrument*.

        For arguments see :meth:`Sensitivities.delta()<rateslib.instruments.Sensitivities.delta>`.
        """
        return super().delta(*args, **kwargs)

    def gamma(self, *args: Any, **kwargs: Any) -> DataFrame:
        """
        Calculate the gamma of the *Instrument*.

        For arguments see :meth:`Sensitivities.gamma()<rateslib.instruments.Sensitivities.gamma>`.
        """
        return super().gamma(*args, **kwargs)


class IndexFixedRateBond(FixedRateBond):
    # TODO (mid) ensure calculations work for amortizing bonds.
    """
    Create an indexed fixed rate bond security.

    For more information see the :ref:`Cookbook Article:<cookbook-doc>` *"Using Curves with an
    Index and Inflation Instruments"*.

    Parameters
    ----------
    args : tuple
        Required positional args for :class:`~rateslib.instruments.FixedRateBond`.
    index_base : float or None, optional
        The base index applied to all periods.
    index_fixings : float, or Series, optional
        If a float scalar, will be applied as the index fixing for the first
        period.
        If a list of *n* fixings will be used as the index fixings for the first *n*
        periods.
        If a datetime indexed ``Series`` will use the fixings that are available in
        that object, and derive the rest from the ``curve``.
    index_method : str
        Whether the indexing uses a daily measure for settlement or the most recently
        monthly data taken from the first day of month.
    index_lag : int, optional
        The number of months by which the index value is lagged. Used to ensure
        consistency between curves and forecast values. Defined by default.
    kwargs : dict
        Required keyword args for :class:`~rateslib.instruments.FixedRateBond`.

    Examples
    --------
    See :class:`~rateslib.instruments.FixedRateBond` for similar.
    """

    _fixed_rate_mixin = True
    _index_base_mixin = True

    leg1: IndexFixedLeg  # type: ignore[assignment]

    def _period_cashflow(
        self,
        period: IndexCashflow | IndexFixedPeriod,  # type: ignore[override]
        curve: CurveOption_,
    ) -> DualTypes:
        """Indexed bonds use the known "real_cashflow" attribute on the *Period*."""
        return period.real_cashflow  # type: ignore[return-value]

    def __init__(
        self,
        effective: datetime_ = NoInput(0),
        termination: datetime | str_ = NoInput(0),
        frequency: int_ = NoInput(0),
        stub: str_ = NoInput(0),
        front_stub: datetime_ = NoInput(0),
        back_stub: datetime_ = NoInput(0),
        roll: str | int_ = NoInput(0),
        eom: bool_ = NoInput(0),
        modifier: str_ = NoInput(0),
        calendar: CalInput = NoInput(0),
        payment_lag: int_ = NoInput(0),
        notional: DualTypes_ = NoInput(0),
        currency: str_ = NoInput(0),
        amortization: DualTypes_ = NoInput(0),
        convention: str_ = NoInput(0),
        fixed_rate: DualTypes_ = NoInput(0),
        index_base: DualTypes_ | Series[DualTypes] = NoInput(0),  # type: ignore[type-var]
        index_fixings: DualTypes_ | Series[DualTypes] = NoInput(0),  # type: ignore[type-var]
        index_method: str_ = NoInput(0),
        index_lag: int_ = NoInput(0),
        ex_div: int_ = NoInput(0),
        settle: int_ = NoInput(0),
        calc_mode: str_ | BondCalcMode = NoInput(0),
        curves: Curves_ = NoInput(0),
        spec: str_ = NoInput(0),
        metric: str = "clean_price",
    ):
        self.kwargs = dict(
            effective=effective,
            termination=termination,
            frequency=frequency,
            stub=stub,
            front_stub=front_stub,
            back_stub=back_stub,
            roll=roll,
            eom=eom,
            modifier=modifier,
            calendar=calendar,
            payment_lag=payment_lag,
            notional=notional,
            currency=currency,
            amortization=amortization,
            convention=convention,
            fixed_rate=fixed_rate,
            initial_exchange=NoInput(0),
            final_exchange=NoInput(0),
            ex_div=ex_div,
            settle=settle,
            calc_mode=calc_mode,
            index_base=index_base,
            index_method=index_method,
            index_lag=index_lag,
            index_fixings=index_fixings,
            metric=metric,
        )
        self.kwargs = _push(spec, self.kwargs)

        # set defaults for missing values
        default_kwargs = dict(
            calc_mode=defaults.calc_mode[type(self).__name__],
            initial_exchange=False,
            final_exchange=True,
            payment_lag=defaults.payment_lag_specific[type(self).__name__],
            ex_div=defaults.ex_div,
            settle=defaults.settle,
            index_method=defaults.index_method,
            index_lag=defaults.index_lag,
        )
        self.kwargs = _update_with_defaults(self.kwargs, default_kwargs)

        if isinstance(self.kwargs["frequency"], NoInput):
            raise ValueError("`frequency` must be provided for Bond.")
        # elif self.kwargs["frequency"].lower() == "z":
        #     raise ValueError("FixedRateBond `frequency` must be in {M, B, Q, T, S, A}.")

        self.calc_mode = _get_calc_mode_for_class(self, self.kwargs["calc_mode"])  # type: ignore[assignment]

        self.curves = curves
        self.spec = spec

        self._fixed_rate = fixed_rate
        self._index_base = index_base  # type: ignore[assignment]

        self.leg1 = IndexFixedLeg(
            **_get(self.kwargs, leg=1, filter=("ex_div", "settle", "calc_mode", "metric")),
        )
        if self.leg1.amortization != 0:
            # Note if amortization is added to IndexFixedRateBonds must systematically
            # go through and update all methods. Many rely on the quantity
            # self.notional which is currently assumed to be a fixed quantity
            raise NotImplementedError("`amortization` for IndexFixedRateBond must be zero.")

    def index_ratio(self, settlement: datetime, curve: Curve_) -> DualTypes:
        """
        Return the index ratio assigned to an *IndexFixedRateBond* for a given settlement.

        Parameters
        ----------
        settlement:
            The settlement date of the bond.
        curve: Curve
            A curve capable of forecasting index values.

        Returns
        -------
        float, Dual, Dual2, Variable
        """
        # TODO: this indexing of periods assumes no amortization
        index_val: DualTypes = index_value(  # type: ignore[assignment]
            index_fixings=self.leg1.index_fixings,
            index_curve=curve,
            index_lag=self.leg1.index_lag,
            index_method=self.leg1.index_method,
            index_date=settlement,
        )
        index_base: DualTypes = index_value(  # type: ignore[assignment]
            index_fixings=self.index_base,
            index_date=self.leg1.schedule.effective,
            index_lag=self.leg1.index_lag,
            index_method=self.leg1.index_method,
            index_curve=curve,
        )
        return index_val / index_base

    def rate(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
        metric: str_ = NoInput(0),
        forward_settlement: datetime_ = NoInput(0),
    ) -> DualTypes:
        """
        Return various pricing metrics of the security calculated from
        :class:`~rateslib.curves.Curve` s.

        Parameters
        ----------
        curves : Curve, str or list of such
            A single :class:`Curve` or id or a list of such. A list defines the
            following curves in the order:

              - Forecasting :class:`Curve` for ``leg1``.
              - Discounting :class:`Curve` for ``leg1``.
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
        metric : str, optional
            Metric returned by the method. Available options are {"clean_price",
            "dirty_price", "ytm", "index_clean_price", "index_dirty_price"}
        forward_settlement : datetime, optional
            The forward settlement date. If not given uses the discount *Curve* and the ``settle``
            attribute of the bond.

        Returns
        -------
        float, Dual, Dual2
        """

        curves_, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            fx,
            base,
            self.leg1.currency,
        )
        curves_1 = _validate_curve_not_no_input(curves_[1])
        metric = _drb(self.kwargs["metric"], metric).lower()
        if metric in [
            "clean_price",
            "dirty_price",
            "index_clean_price",
            "ytm",
            "index_dirty_price",
        ]:
            if isinstance(forward_settlement, NoInput):
                settlement = self.leg1.schedule.calendar.lag(
                    curves_1.nodes.initial,
                    self.kwargs["settle"],
                    True,
                )
            else:
                settlement = forward_settlement
            npv = self._npv_local(curves_[0], curves_1, settlement, settlement)
            # scale price to par 100 (npv is already projected forward to settlement)
            index_dirty_price = npv * 100 / -self.leg1.notional
            index_ratio = self.index_ratio(settlement, _validate_curve_is_not_dict(curves_[0]))
            dirty_price = index_dirty_price / index_ratio

            if metric == "dirty_price":
                return dirty_price
            elif metric == "clean_price":
                return dirty_price - self.accrued(settlement)
            elif metric == "ytm":
                return self.ytm(dirty_price, settlement, True)
            elif metric == "index_dirty_price":
                return index_dirty_price
            elif metric == "index_clean_price":
                return index_dirty_price - self.accrued(settlement) * index_ratio

        raise ValueError(
            "`metric` must be in {'dirty_price', 'clean_price', 'ytm', "
            "'index_dirty_price', 'index_clean_price'}.",
        )


class Bill(FixedRateBond):
    """
    Create a discount security.

    Parameters
    ----------
    effective : datetime
        The adjusted or unadjusted effective date.
    termination : datetime or str
        The adjusted or unadjusted termination date. If a string, then a tenor must be
        given expressed in days (`"D"`), months (`"M"`) or years (`"Y"`), e.g. `"7M"`.
    frequency : str in {"M", "B", "Q", "T", "S", "A"}, optional
        The frequency used only by the :meth:`~rateslib.instruments.Bill.ytm` method.
        All *Bills* have an implicit frequency of "Z" for schedule construction.
    modifier : str, optional
        The modification rule, in {"F", "MF", "P", "MP"}
    calendar : calendar or str, optional
        The holiday calendar object to use. If str, looks up named calendar from
        static data.
    payment_lag : int, optional
        The number of business days to lag payments by.
    notional : float, optional
        The leg notional, which is applied to each period.
    currency : str, optional
        The currency of the leg (3-digit code).
    convention: str, optional
        The day count convention applied to calculations of period accrual dates.
        See :meth:`~rateslib.calendars.dcf`.
    settle : int
        The number of business days for regular settlement time, i.e, 1 is T+1.
    calc_mode : str, optional (defaults.calc_mode["Bill"])
        A calculation mode for dealing with bonds that are in short stub or accrual
        periods. All modes give the same value for YTM at issue date for regular
        bonds but differ slightly for bonds with stubs or with accrued.
    curves : CurveType, str or list of such, optional
        A single *Curve* or string id or a list of such.

        A list defines the following curves in the order:

        - Forecasting *Curve* for ``leg1``.
        - Discounting :class:`~rateslib.curves.Curve` for ``leg1``.
    spec : str, optional
        An identifier to pre-populate many field with conventional values. See
        :ref:`here<defaults-doc>` for more info and available values.
    metric: str, optional
        The pricing metric returned by the ``rate`` method of the *Instrument*.

    Examples
    --------
    This example is taken from the US Treasury Federal website. A copy of
    which is available :download:`here<_static/ofcalc6decTbill.pdf>`.

    We demonstrate the use of **analogue methods** which do not need *Curves* or
    *Solvers*,
    :meth:`~rateslib.instruments.Bill.price`,
    :meth:`~rateslib.instruments.Bill.simple_rate`,
    :meth:`~rateslib.instruments.Bill.discount_rate`,
    :meth:`~rateslib.instruments.FixedRateBond.ytm`,
    :meth:`~rateslib.instruments.FixedRateBond.ex_div`,
    :meth:`~rateslib.instruments.FixedRateBond.accrued`,
    :meth:`~rateslib.instruments.FixedRateBond.repo_from_fwd`
    :meth:`~rateslib.instruments.FixedRateBond.fwd_from_repo`
    :meth:`~rateslib.instruments.FixedRateBond.duration`,
    :meth:`~rateslib.instruments.FixedRateBond.convexity`.

    .. ipython:: python

       bill = Bill(
           effective=dt(2004, 1, 22),
           termination=dt(2004, 2, 19),
           calendar="nyc",
           modifier="NONE",
           currency="usd",
           convention="Act360",
           settle=1,
           notional=-1e6,  # negative notional receives fixed, i.e. buys a bill
           curves="bill_curve",
           calc_mode="us_gbb",
       )
       bill.ex_div(dt(2004, 1, 22))
       bill.price(rate=0.80, settlement=dt(2004, 1, 22))
       bill.simple_rate(price=99.937778, settlement=dt(2004, 1, 22))
       bill.discount_rate(price=99.937778, settlement=dt(2004, 1, 22))
       bill.ytm(price=99.937778, settlement=dt(2004, 1, 22))
       bill.accrued(dt(2004, 1, 22))
       bill.fwd_from_repo(
           price=99.937778,
           settlement=dt(2004, 1, 22),
           forward_settlement=dt(2004, 2, 19),
           repo_rate=0.8005,
           convention="Act360",
       )
       bill.repo_from_fwd(
           price=99.937778,
           settlement=dt(2004, 1, 22),
           forward_settlement=dt(2004, 2, 19),
           forward_price=100.00,
           convention="Act360",
       )
       bill.duration(settlement=dt(2004, 1, 22), ytm=0.8005, metric="risk")
       bill.duration(settlement=dt(2004, 1, 22), ytm=0.8005, metric="modified")
       bill.convexity(settlement=dt(2004, 1, 22), ytm=0.8005)


    The following **digital methods** consistent with the library's ecosystem are
    also available,
    :meth:`~rateslib.instruments.Bill.rate`,
    :meth:`~rateslib.instruments.FixedRateBond.npv`,
    :meth:`~rateslib.instruments.FixedRateBond.analytic_delta`,
    :meth:`~rateslib.instruments.FixedRateBond.cashflows`,
    :meth:`~rateslib.instruments.FixedRateBond.delta`,
    :meth:`~rateslib.instruments.FixedRateBond.gamma`,

    .. ipython:: python

       bill_curve = Curve({dt(2004, 1, 21): 1.0, dt(2004, 3, 21): 1.0}, id="bill_curve")
       instruments = [
           (bill, (), {"metric": "ytm"}),
       ]
       solver = Solver(
           curves=[bill_curve],
           instruments=instruments,
           s=[0.8005],
           instrument_labels=["Feb04 Tbill"],
           id="bill_solver",
       )
       bill.npv(solver=solver)
       bill.analytic_delta(disc_curve=bill_curve)
       bill.rate(solver=solver, metric="price")

    The sensitivities are also available. In this case the *Solver* is calibrated
    with *instruments* priced in yield terms so sensitivities are measured in basis
    points (bps).

    .. ipython:: python

       bill.delta(solver=solver)
       bill.gamma(solver=solver)

    The DataFrame of cashflows.

    .. ipython:: python

       bill.cashflows(solver=solver)

    """

    calc_mode: BillCalcMode  # type: ignore[assignment]

    def __init__(
        self,
        effective: datetime_ = NoInput(0),
        termination: datetime | str_ = NoInput(0),
        frequency: str_ = NoInput(0),
        modifier: str_ = NoInput(0),
        calendar: CalInput = NoInput(0),
        payment_lag: int_ = NoInput(0),
        notional: DualTypes_ = NoInput(0),
        currency: str_ = NoInput(0),
        convention: str_ = NoInput(0),
        settle: int_ = NoInput(0),
        calc_mode: BillCalcMode | str_ = NoInput(0),
        curves: Curves_ = NoInput(0),
        spec: str_ = NoInput(0),
        metric: str = "price",
    ):
        super().__init__(
            effective=effective,
            termination=termination,
            frequency="z",
            stub=NoInput(0),
            front_stub=NoInput(0),
            back_stub=NoInput(0),
            roll=NoInput(0),
            eom=NoInput(0),
            modifier=modifier,
            calendar=calendar,
            payment_lag=payment_lag,
            notional=notional,
            currency=currency,
            amortization=NoInput(0),
            convention=convention,
            fixed_rate=0,
            ex_div=0,
            settle=settle,
            curves=curves,
            calc_mode=calc_mode,
            spec=spec,
            metric=metric,
        )
        self.kwargs["frequency"] = _drb(
            self.calc_mode._ytm_clone_kwargs["frequency"],
            frequency,
        )

    @property
    def dcf(self) -> float:
        # bills will typically have 1 period since they are configured with frequency "z".
        d = 0.0
        for i in range(self.leg1.schedule.n_periods):
            d += self.leg1._regular_periods[i].dcf
        return d

    def rate(  # type: ignore[override]
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
        metric: str_ = NoInput(0),
    ) -> DualTypes:
        """
        Return various pricing metrics of the security calculated from
        :class:`~rateslib.curves.Curve` s.

        Parameters
        ----------
        curves : Curve, str or list of such
            A single :class:`Curve` or id or a list of such. A list defines the
            following curves in the order:

              - Forecasting :class:`Curve` for ``leg1``.
              - Discounting :class:`Curve` for ``leg1``.
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
        metric : str in {"price", "discount_rate", "ytm", "simple_rate"}
            Metric returned by the method. Uses the *Instrument* default if not given.

        Returns
        -------
        float, Dual, Dual2
        """
        curves_, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            fx,
            base,
            self.leg1.currency,
        )
        curves_1 = _validate_curve_not_no_input(curves_[1])
        settlement = self.leg1.schedule.calendar.lag(
            curves_1.nodes.initial,
            self.kwargs["settle"],
            True,
        )
        # scale price to par 100 and make a fwd adjustment according to curve
        price = (
            self.npv(curves, solver, fx_, base_, local=False)  # type: ignore[operator]
            * 100
            / (-self.leg1.notional * curves_1[settlement])
        )
        metric = _drb(self.kwargs["metric"], metric).lower()
        if metric in ["price", "clean_price", "dirty_price"]:
            return price
        elif metric == "discount_rate":
            return self.discount_rate(price, settlement)
        elif metric == "simple_rate":
            return self.simple_rate(price, settlement)
        elif metric == "ytm":
            return self.ytm(price, settlement, NoInput(0))
        raise ValueError("`metric` must be in {'price', 'discount_rate', 'ytm', 'simple_rate'}")

    def simple_rate(self, price: DualTypes, settlement: datetime) -> DualTypes:
        """
        Return the simple rate of the security from its ``price``.

        Parameters
        ----------
        price : float, Dual, or Dual2
            The price of the security.
        settlement : datetime
            The settlement date of the security.

        Returns
        -------
        float, Dual, or Dual2
        """
        acc_frac = self.calc_mode._settle_accrual(self, settlement, 0)
        dcf = (1 - acc_frac) * self.dcf
        return ((100 / price - 1) / dcf) * 100

    def discount_rate(self, price: DualTypes, settlement: datetime) -> DualTypes:
        """
        Return the discount rate of the security from its ``price``.

        Parameters
        ----------
        price : float, Dual, or Dual2
            The price of the security.
        settlement : datetime
            The settlement date of the security.

        Returns
        -------
        float, Dual, or Dual2
        """
        acc_frac = self.calc_mode._settle_accrual(self, settlement, 0)
        dcf = (1 - acc_frac) * self.dcf
        rate = ((1 - price / 100) / dcf) * 100
        return rate

    def price(
        self,
        rate: DualTypes,
        settlement: datetime,
        dirty: bool = False,
        calc_mode: str_ = NoInput(0),
    ) -> DualTypes:
        """
        Return the price of the bill given the ``discount_rate``.

        Parameters
        ----------
        rate : float
            The rate used by the pricing formula.
        settlement : datetime
            The settlement date.
        dirty : bool, not required
            Discount securities have no coupon, the concept of clean or dirty is not
            relevant. Argument is included for signature consistency with
            :meth:`FixedRateBond.price<rateslib.instruments.FixedRateBond.price>`.
        calc_mode : str, optional
            A calculation mode to force, which is used instead of that attributed the
            *Bill* instance.

        Returns
        -------
        float, Dual, Dual2
        """
        if isinstance(calc_mode, NoInput):
            calc_mode_ = self.calc_mode
        else:
            if isinstance(calc_mode, str):
                calc_mode_ = BILL_MODE_MAP[calc_mode.lower()]
            else:
                calc_mode_ = calc_mode

        price_func = getattr(self, f"_price_{calc_mode_._price_type}")
        return price_func(rate, settlement)  # type: ignore[no-any-return]

    def _price_discount(self, rate: DualTypes, settlement: datetime) -> DualTypes:
        acc_frac = self.calc_mode._settle_accrual(self, settlement, 0)
        dcf = (1 - acc_frac) * self.dcf
        return 100 - rate * dcf

    def _price_simple(self, rate: DualTypes, settlement: datetime) -> DualTypes:
        acc_frac = self.calc_mode._settle_accrual(self, settlement, 0)
        dcf = (1 - acc_frac) * self.dcf
        return 100 / (1 + rate * dcf / 100)

    def ytm(  # type: ignore[override]
        self,
        price: DualTypes,
        settlement: datetime,
        calc_mode: BillCalcMode | str_ = NoInput(0),
    ) -> Number:
        """
        Calculate the yield-to-maturity on an equivalent bond with a coupon of 0%.

        Parameters
        ----------
        price: float, Dual, Dual2
            The price of the *Bill*.
        settlement: datetime
            The settlement date of the *Bill*.
        calc_mode : str, optional
            A calculation mode to force, which is used instead of that attributed the
            *Bill* instance.

        Notes
        -----
        Maps the following *Bill* ``calc_mode`` to the following *Bond* specifications:

        - *NoInput* -> "ust"
        - *"ustb"* -> "ust"
        - *"uktb"* -> "ukt"
        - *"sgbb"* -> "sgb"

        This method calculates by constructing a :class:`~rateslib.instruments.FixedRateBond`
        with a regular 0% coupon measured from the termination date of the bill.
        """

        if isinstance(calc_mode, NoInput):
            calc_mode = self.calc_mode
            # kwargs["frequency"] is populated as the ytm_clone frequency at __init__
            freq = self.kwargs["frequency"]
        else:
            if isinstance(calc_mode, str):
                calc_mode = BILL_MODE_MAP[calc_mode.lower()]
            freq = calc_mode._ytm_clone_kwargs["frequency"]

        frequency_months = defaults.frequency_months[freq.upper()]
        quasi_start = self.leg1.schedule.termination
        while quasi_start > settlement:
            quasi_start = add_tenor(
                quasi_start,
                f"-{frequency_months}M",
                "NONE",
                NoInput(0),
                NoInput(0),
            )
        equiv_bond = FixedRateBond(
            effective=quasi_start,
            termination=self.leg1.schedule.termination,
            fixed_rate=0.0,
            **_get(
                calc_mode._ytm_clone_kwargs,
                leg=1,
                filter=("initial_exchange", "final_exchange", "payment_lag_exchange"),
            ),
        )
        return equiv_bond.ytm(price, settlement)

    def duration(self, *args: Any, **kwargs: Any) -> float:
        """
        Return the duration of the *Bill*. See :class:`~rateslib.instruments.FixedRateBond.duration` for arguments.

        Notes
        ------

        .. warning::

           This function returns a *duration* that is consistent with a
           *FixedRateBond* yield-to-maturity definition. It currently does not use the
           specified ``convention`` of the *Bill*, and can be sensitive to the
           ``frequency`` of the representative *FixedRateBond* equivalent.

        .. ipython:: python

           bill = Bill(effective=dt(2024, 2, 29), termination=dt(2024, 8, 29), spec="us_gbb")
           bill.duration(settlement=dt(2024, 5, 30), ytm=5.2525, metric="duration")

           bill = Bill(effective=dt(2024, 2, 29), termination=dt(2024, 8, 29), spec="us_gbb", frequency="A")
           bill.duration(settlement=dt(2024, 5, 30), ytm=5.2525, metric="duration")

        """  # noqa: E501
        return super().duration(*args, **kwargs)


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


class FloatRateNote(Sensitivities, BondMixin, Metrics):  # type: ignore[misc]
    """
    Create a floating rate note (FRN) security.

    Parameters
    ----------
    effective : datetime
        The adjusted or unadjusted effective date.
    termination : datetime or str
        The adjusted or unadjusted termination date. If a string, then a tenor must be
        given expressed in days (`"D"`), months (`"M"`) or years (`"Y"`), e.g. `"48M"`.
    frequency : str in {"M", "B", "Q", "T", "S", "A"}, optional
        The frequency of the schedule. "Z" is not permitted.
    stub : str combining {"SHORT", "LONG"} with {"FRONT", "BACK"}, optional
        The stub type to enact on the swap. Can provide two types, for
        example "SHORTFRONTLONGBACK".
    front_stub : datetime, optional
        An adjusted or unadjusted date for the first stub period.
    back_stub : datetime, optional
        An adjusted or unadjusted date for the back stub period.
        See notes for combining ``stub``, ``front_stub`` and ``back_stub``
        and any automatic stub inference.
    roll : int in [1, 31] or str in {"eom", "imm", "som"}, optional
        The roll day of the schedule. Inferred if not given.
    eom : bool, optional
        Use an end of month preference rather than regular rolls for inference. Set by
        default. Not required if ``roll`` is specified.
    modifier : str, optional
        The modification rule, in {"F", "MF", "P", "MP"}
    calendar : calendar or str, optional
        The holiday calendar object to use. If str, looks up named calendar from
        static data.
    payment_lag : int, optional
        The number of business days to lag payments by.
    notional : float, optional
        The leg notional, which is applied to each period.
    currency : str, optional
        The currency of the leg (3-digit code).
    amortization: float, optional
        The amount by which to adjust the notional each successive period. Should have
        sign equal to that of notional if the notional is to reduce towards zero.
    convention: str, optional
        The day count convention applied to calculations of period accrual dates.
        See :meth:`~rateslib.calendars.dcf`.
    float_spread : float, optional
        The spread applied to determine cashflows. Can be set to `None` and designated
        later, perhaps after a mid-market spread for all periods has been calculated.
    spread_compound_method : str, optional
        The method to use for adding a floating spread to compounded rates. Available
        options are `{"none_simple", "isda_compounding", "isda_flat_compounding"}`.
    fixings : float or list, optional
        If a float scalar, will be applied as the determined fixing for the first
        period. If a list of *n* fixings will be used as the fixings for the first *n*
        periods. If any sublist of length *m* is given as the first *m* RFR fixings
        within individual curve and composed into the overall rate.
    fixing_method : str, optional
        The method by which floating rates are determined, set by default. See notes.
    method_param : int, optional
        A parameter that is used for the various ``fixing_method`` s. See notes.
    ex_div : int
        The number of days prior to a cashflow during which the bond is considered
        ex-dividend.
    settle : int
        The number of business days for regular settlement time, i.e, 1 is T+1.
    calc_mode : str
        A calculation mode for dealing with bonds under different conventions. See notes.
    curves : CurveType, str or list of such, optional
        A single *Curve* or string id or a list of such.

        A list defines the following curves in the order:

        - Forecasting *Curve* for ``leg1``.
        - Discounting :class:`~rateslib.curves.Curve` for ``leg1``.
    spec : str, optional
        An identifier to pre-populate many field with conventional values. See
        :ref:`here<defaults-doc>` for more info and available values.
    metric: str, optional
        The pricing metric returned by the ``rate`` method of the *Instrument*.

    Notes
    -----
    .. warning::

       FRNs based on RFR rates which have ex-div days must ensure that fixings are
       available to define the entire period. This means that `ex_div` days must be less
       than the `fixing_method` `method_param` lag minus the time to settlement time.

        That is, a bond with a `method_param` of 5 and a settlement time of 2 days
        can have an `ex_div` period of at maximum 3.

        A bond with a `method_param` of 2 and a settlement time of 1 day cnan have an
        `ex_div` period of at maximum 1.

    Attributes
    ----------
    ex_div_days : int
    leg1 : FloatLeg
    """

    _float_spread_mixin = True

    leg1: FloatLeg

    def __init__(
        self,
        effective: datetime_ = NoInput(0),
        termination: datetime | str_ = NoInput(0),
        frequency: int_ = NoInput(0),
        stub: str_ = NoInput(0),
        front_stub: datetime_ = NoInput(0),
        back_stub: datetime_ = NoInput(0),
        roll: str | int_ = NoInput(0),
        eom: bool | NoInput = NoInput(0),
        modifier: str_ = NoInput(0),
        calendar: CalInput = NoInput(0),
        payment_lag: int_ = NoInput(0),
        notional: DualTypes_ = NoInput(0),
        currency: str_ = NoInput(0),
        amortization: DualTypes_ = NoInput(0),
        convention: str_ = NoInput(0),
        float_spread: DualTypes_ = NoInput(0),
        fixings: FixingsRates_ = NoInput(0),  # type: ignore[type-var]
        fixing_method: str_ = NoInput(0),
        method_param: int_ = NoInput(0),
        spread_compound_method: str_ = NoInput(0),
        ex_div: int_ = NoInput(0),
        settle: int_ = NoInput(0),
        calc_mode: str_ = NoInput(0),
        curves: Curves_ = NoInput(0),
        spec: str_ = NoInput(0),
        metric: str = "clean_price",
    ):
        self.kwargs = dict(
            effective=effective,
            termination=termination,
            frequency=frequency,
            stub=stub,
            front_stub=front_stub,
            back_stub=back_stub,
            roll=roll,
            eom=eom,
            modifier=modifier,
            calendar=calendar,
            payment_lag=payment_lag,
            notional=notional,
            currency=currency,
            amortization=amortization,
            convention=convention,
            float_spread=float_spread,
            fixings=fixings,
            fixing_method=fixing_method,
            method_param=method_param,
            spread_compound_method=spread_compound_method,
            initial_exchange=NoInput(0),
            final_exchange=NoInput(0),
            ex_div=ex_div,
            settle=settle,
            calc_mode=calc_mode,
            metric=metric,
        )
        self.kwargs = _push(spec, self.kwargs)

        # set defaults for missing values
        default_kwargs = dict(
            calc_mode=defaults.calc_mode[type(self).__name__],
            initial_exchange=False,
            final_exchange=True,
            payment_lag=defaults.payment_lag_specific[type(self).__name__],
            ex_div=defaults.ex_div,
            settle=defaults.settle,
        )
        self.kwargs = _update_with_defaults(self.kwargs, default_kwargs)

        if isinstance(self.kwargs["frequency"], NoInput):
            raise ValueError("`frequency` must be provided for Bond.")
        elif self.kwargs["frequency"].lower() == "z":
            raise ValueError("FloatRateNote `frequency` must be in {M, B, Q, T, S, A}.")

        if isinstance(self.kwargs["calc_mode"], str):
            map_ = {
                "FloatRateNote": BOND_MODE_MAP,
            }
            self.calc_mode = map_[type(self).__name__][self.kwargs["calc_mode"].lower()]
        else:
            self.calc_mode = self.kwargs["calc_mode"]

        self.curves = curves
        self.spec = spec

        self._float_spread = float_spread
        self.leg1 = FloatLeg(
            **_get(self.kwargs, leg=1, filter=("ex_div", "settle", "calc_mode", "metric"))
        )

        if "rfr" in self.leg1.fixing_method and self.kwargs["ex_div"] > (
            self.leg1.method_param + 1
        ):
            raise ValueError(
                "For RFR FRNs `ex_div` must be less than or equal to (`method_param` + 1) "
                "otherwise negative accrued payments cannot be explicitly "
                "determined due to unknown fixings.",
            )

        if self.leg1.amortization != 0:
            # Note if amortization is added to FloatRateNote must systematically
            # go through and update all methods. Many rely on the quantity
            # self.notional which is currently assumed to be a fixed quantity
            raise NotImplementedError("`amortization` for FloatRateNote must be zero.")

    def _period_cashflow(self, period: Cashflow | FloatPeriod, curve: Curve_) -> DualTypes:  # type: ignore[override]
        """FloatRateNotes must forecast cashflows with a *Curve* on the *Period*."""
        if isinstance(period, FloatPeriod):
            _: DualTypes = period.cashflow(curve)  # type: ignore[assignment]
        else:
            _ = period.cashflow
        return _

    def _accrual_rate(
        self, pseudo_period: FloatPeriod, curve: CurveOption_, method_param: int
    ) -> float:
        """
        Take a period and try to forecast the rate which determines the accrual,
        either from known fixings, a curve or forward filling historical fixings.

        This method is required to handle the case where a curve is not provided and
        fixings are enough.
        """
        # TODO: make this AD safe and not return a float.
        if pseudo_period.dcf < 1e-10:
            return 0.0  # there are no fixings in the period.

        if not isinstance(curve, NoInput):
            curve_ = curve
        else:
            # Test to see if any missing fixings are required:
            # The fixings calendar and convention are taken from Curve so the pseudo curve
            # can only get them from the instrument and assume that they align. Otherwise
            # it is best practice to supply a forecast curve when calculating accrued interest.
            pseudo_curve = Curve(
                {},
                calendar=pseudo_period.calendar,
                convention=pseudo_period.convention,
                modifier="F",
            )
            try:
                _: DualTypes = pseudo_period.rate(pseudo_curve)
                return _  # type: ignore[return-value]
            except IndexError:
                # the pseudo_curve has no nodes so when it needs to calculate a rate it cannot
                # be indexed.
                # Try to revert back to using the last fixing as forward projection.
                try:
                    if isinstance(pseudo_period.fixings, Series):
                        last_fixing = pseudo_period.fixings.iloc[-1]
                    else:
                        last_fixing = pseudo_period.fixings[-1]  # type: ignore[index, assignment]
                    warnings.warn(
                        "A `Curve` was not supplied. Residual required fixings not yet "
                        "published are forecast from the last known fixing.",
                        UserWarning,
                    )
                    # For negative accr in ex-div we need to forecast unpublished rates.
                    # Build a curve which replicates the last fixing value from fixings.
                except TypeError:
                    # then rfr fixing cannot be fetched from attribute
                    if pseudo_period.dcf < 1e-10:
                        # then settlement is same as period.start so no rate necessary
                        # create a dummy curve
                        last_fixing = 0.0
                    else:
                        raise TypeError(
                            "`fixings` or `curve` are not available for RFR float period. If"
                            "supplying `fixings` must be a Series or list, "
                            f"got: {pseudo_period.fixings}",
                        )
                curve_ = LineCurve(
                    {
                        pseudo_period.start
                        - timedelta(
                            days=0 if isinstance(method_param, NoInput) else method_param,
                        ): last_fixing,
                        pseudo_period.end: last_fixing,
                    },
                    convention=pseudo_period.convention,
                    calendar=pseudo_period.calendar,
                )

        # Otherwise rate to settle is determined fully by known fixings.
        _ = _dual_float(pseudo_period.rate(curve_))
        return _

    def accrued(
        self,
        settlement: datetime,
        curve: CurveOption_ = NoInput(0),
    ) -> DualTypes:
        """
        Calculate the accrued amount per nominal par value of 100.

        Parameters
        ----------
        settlement : datetime
            The settlement date which to measure accrued interest against.
        curve : Curve, optional
            If ``forecast`` is *True* and fixings are future based then must provide
            a forecast curve.

        Notes
        -----
        The settlement of an FRN will always be a definite amount. The
        ``fixing_method``, ``method_param`` and ``ex_div`` will contain a
        valid combination of parameters such that when payments need to be
        cleared these definitive amounts can be calculated
        via previously published fixings.

        If the coupon is IBOR based then the accrued
        fractionally apportions the coupon payment based on calendar days, including
        negative accrued during ex div periods. This rarely poses a problem since
        IBOR is fixed well in advance of settlement.

        .. math::

           \\text{Accrued} = \\text{Coupon} \\times \\frac{\\text{Settle - Last Coupon}}{\\text{Next Coupon - Last Coupon}}

        With RFR rates, however, and since ``settlement`` typically occurs
        in the future, e.g. T+2, it may be
        possible, particularly if the bond is *ex-div* that some fixings are not known
        today, but they will be known by ``settlement``. This is also true if we
        wish to calculate the forward dirty price of a bond and need to forecast
        the accrued amount (and also for a forecast IBOR period).

        Thus, there are two options:

        - In the analogue mode where very few fixings might be missing, and we require
          these values to calculate negative accrued in an ex-div period we do not
          require a ``curve`` but repeat the last historic fixing.
        - In the digital mode where the ``settlement`` is likely in the future we
          use a ``curve`` to forecast rates,

        Examples
        --------
        An RFR based FRN where the fixings are known up to the end of period.

        .. ipython:: python

           fixings = Series(2.0, index=date_range(dt(1999, 12, 1), dt(2000, 6, 2)))
           frn = FloatRateNote(
               effective=dt(1998, 12, 7),
               termination=dt(2015, 12, 7),
               frequency="S",
               currency="gbp",
               convention="Act365F",
               ex_div=3,
               fixings=fixings,
               fixing_method="rfr_observation_shift",
               method_param=5,
           )
           frn.accrued(dt(2000, 3, 27))
           frn.accrued(dt(2000, 6, 4))


        An IBOR based FRN where the coupon is known in advance.

        .. ipython:: python

           fixings = Series(2.0, index=[dt(1999, 12, 5)])
           frn = FloatRateNote(
               effective=dt(1998, 12, 7),
               termination=dt(2015, 12, 7),
               frequency="S",
               currency="gbp",
               convention="Act365F",
               ex_div=7,
               fixings=fixings,
               fixing_method="ibor",
               method_param=2,
           )
           frn.accrued(dt(2000, 3, 27))
           frn.accrued(dt(2000, 6, 4))
        """  # noqa: E501
        if self.leg1.fixing_method == "ibor":
            acc_idx = self._period_index(settlement)
            frac = self.calc_mode._settle_accrual(self, settlement, acc_idx)
            if self.ex_div(settlement):
                frac = frac - 1  # accrued is negative in ex-div period

            if isinstance(curve, NoInput):
                curve_: CurveOption = Curve(
                    {  # create a dummy curve. rate() will return the fixing
                        self.leg1._regular_periods[acc_idx].start: 1.0,
                        self.leg1._regular_periods[acc_idx].end: 1.0,
                    },
                )
            else:
                curve_ = curve

            rate = self.leg1._regular_periods[acc_idx].rate(curve_)

            cashflow = (
                -self.leg1._regular_periods[acc_idx].notional
                * self.leg1._regular_periods[acc_idx].dcf
                * rate
                / 100
            )
            return frac * cashflow / -self.leg1.notional * 100
        else:  # is "rfr"
            acc_idx = index_left(
                self.leg1.schedule.aschedule,
                len(self.leg1.schedule.aschedule),
                settlement,
            )
            p = FloatPeriod(
                start=self.leg1.schedule.aschedule[acc_idx],
                end=settlement,
                payment=settlement,
                frequency=self.leg1.schedule.frequency,
                notional=-100,
                currency=self.leg1.currency,
                convention=self.leg1.convention,
                termination=self.leg1.schedule.aschedule[acc_idx + 1],
                stub=True,
                float_spread=self.float_spread,
                fixing_method=self.leg1.fixing_method,
                fixings=self.leg1.fixings[acc_idx],
                method_param=self.leg1.method_param,
                spread_compound_method=self.leg1.spread_compound_method,
                roll=self.leg1.schedule.roll,
                calendar=self.leg1.schedule.calendar,
            )
            rate_to_settle = self._accrual_rate(p, curve, self.leg1.method_param)
            accrued_to_settle = 100 * p.dcf * rate_to_settle / 100

            if self.ex_div(settlement):
                rate_to_end = self._accrual_rate(
                    self.leg1._regular_periods[acc_idx],
                    curve,
                    self.leg1.method_param,
                )
                accrued_to_end = 100 * self.leg1._regular_periods[acc_idx].dcf * rate_to_end / 100
                return accrued_to_settle - accrued_to_end
            else:
                return accrued_to_settle

    def rate(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
        metric: str_ = NoInput(0),
        forward_settlement: datetime_ = NoInput(0),
    ) -> DualTypes:
        """
        Return various pricing metrics of the security calculated from
        :class:`~rateslib.curves.Curve` s.

        Parameters
        ----------
        curves : Curve, str or list of such
            A single :class:`Curve` or id or a list of such. A list defines the
            following curves in the order:

              - Forecasting :class:`Curve` for ``leg1``.
              - Discounting :class:`Curve` for ``leg1``.
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
        metric : str, optional
            Metric returned by the method. Available options are {"clean_price",
            "dirty_price", "spread", "ytm"}. Uses the *Instrument* default if not given.
        forward_settlement : datetime, optional
            The forward settlement date. If not give uses the discount *Curve* and the bond's
            ``settle`` attribute.}.

        Returns
        -------
        float, Dual, Dual2

        """
        curves_, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            fx,
            base,
            self.leg1.currency,
        )
        metric = _drb(self.kwargs["metric"], metric).lower()
        if metric in ["clean_price", "dirty_price", "spread", "ytm"]:
            curves_1 = _validate_curve_not_no_input(curves_[1])
            if isinstance(forward_settlement, NoInput):
                settlement = self.leg1.schedule.calendar.lag(
                    curves_1.nodes.initial,  # discount curve
                    self.kwargs["settle"],
                    True,
                )
            else:
                settlement = forward_settlement
            npv = self._npv_local(curves_[0], curves_1, settlement, settlement)
            # scale price to par 100 (npv is already projected forward to settlement)
            dirty_price = npv * 100 / -self.leg1.notional

            if metric == "dirty_price":
                return dirty_price
            elif metric == "clean_price":
                return dirty_price - self.accrued(settlement, curve=curves_[0])
            elif metric == "spread":
                _: DualTypes = self.leg1._spread(-(npv + self.leg1.notional), curves_[0], curves_1)
                z: DualTypes = _drb(0.0, self.float_spread)
                return _ + z
            elif metric == "ytm":
                return self.ytm(
                    price=dirty_price, settlement=settlement, dirty=True, curve=curves_[0]
                )

        raise ValueError("`metric` must be in {'dirty_price', 'clean_price', 'spread', 'ytm'}.")

    def delta(self, *args: Any, **kwargs: Any) -> DataFrame:
        """
        Calculate the delta of the *Instrument*.

        For arguments see :meth:`Sensitivities.delta()<rateslib.instruments.Sensitivities.delta>`.
        """
        return super().delta(*args, **kwargs)

    def gamma(self, *args: Any, **kwargs: Any) -> DataFrame:
        """
        Calculate the gamma of the *Instrument*.

        For arguments see :meth:`Sensitivities.gamma()<rateslib.instruments.Sensitivities.gamma>`.
        """
        return super().gamma(*args, **kwargs)

    def fixings_table(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
        approximate: bool = False,
        right: datetime_ = NoInput(0),
    ) -> DataFrame:
        """
        Return a DataFrame of fixing exposures on the :class:`~rateslib.legs.FloatLeg`.

        Parameters
        ----------
        curves : Curve, str or list of such
            A single :class:`~rateslib.curves.Curve` or id or a list of such.
            A list defines the following curves in the order:

            - Forecasting :class:`~rateslib.curves.Curve` for floating leg.
            - Discounting :class:`~rateslib.curves.Curve` for both legs.

        solver : Solver, optional
            The numerical :class:`~rateslib.solver.Solver` that constructs
            :class:`~rateslib.curves.Curve` from calibrating instruments.

            .. note::

               The arguments ``fx`` and ``base`` are unused by single currency
               *Instruments* rates calculations.

        approximate : bool, optional
            Perform a calculation that is broadly 10x faster but potentially loses
            precision upto 0.1%.
        right : datetime, optional
            Only calculate fixing exposures upto and including this date.

        Returns
        -------
        DataFrame
        """
        curves_, _, _ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            NoInput(0),
            NoInput(0),
            self.leg1.currency,
        )
        df = self.leg1.fixings_table(
            curve=curves_[0], approximate=approximate, disc_curve=curves_[1], right=right
        )
        return df

    def ytm(
        self,
        price: DualTypes,
        settlement: datetime,
        curve: CurveOption_ = NoInput(0),
        dirty: bool = False,
    ) -> Number:
        """
        Calculate the yield-to-maturity of the security given its price.

        Parameters
        ----------
        price : float, Dual, Dual2
            The price, per 100 nominal, against which to determine the yield.
        settlement : datetime
            The settlement date on which to determine the price.
        curve : Curve, dict[str, Curve]
            Curve used to forecast cashflows.
        dirty : bool, optional
            If `True` will assume the
            :meth:`~rateslib.instruments.FixedRateBond.accrued` is included in the price.

        Returns
        -------
        float, Dual, Dual2

        Notes
        -----
        If ``price`` is given as :class:`~rateslib.dual.Dual` or
        :class:`~rateslib.dual.Dual2` input the result of the yield will be output
        as the same type with the variables passed through accordingly.

        """  # noqa: E501
        return self._ytm(price=price, settlement=settlement, dirty=dirty, curve=curve)
