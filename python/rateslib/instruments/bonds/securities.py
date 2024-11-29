from __future__ import annotations

import warnings
from datetime import datetime, timedelta
from functools import partial

import numpy as np
from pandas import DataFrame, Series

from rateslib import defaults
from rateslib.calendars import CalInput, add_tenor, dcf
from rateslib.curves import Curve, IndexCurve, LineCurve, average_rate, index_left
from rateslib.default import NoInput, _drb
from rateslib.dual import Dual, Dual2, DualTypes, gradient
from rateslib.fx import FXForwards, FXRates
from rateslib.instruments.bonds.conventions import (
    BILL_MODE_MAP,
    BOND_MODE_MAP,
    BillCalcMode,
    BondCalcMode,
    _get_calc_mode_for_class,
)

# from scipy.optimize import brentq
from rateslib.instruments.core import (
    BaseMixin,
    Sensitivities,
    _get,
    _get_curves_fx_and_base_maybe_from_solver,
    _push,
    _update_with_defaults,
)
from rateslib.legs import FixedLeg, FloatLeg, IndexFixedLeg, IndexMixin
from rateslib.periods import FloatPeriod, _disc_maybe_from_curve, _maybe_local
from rateslib.solver import Solver, quadratic_eqn


class BondMixin:
    def _period_index(self, settlement: datetime):
        """
        Get the coupon period index for that which the settlement date fall within.
        Uses unadjusted dates.
        """
        _ = index_left(
            self.leg1.schedule.uschedule,
            len(self.leg1.schedule.uschedule),
            settlement,
        )
        return _

    # def _accrued_fraction(self, settlement: datetime, calc_mode: str | NoInput, acc_idx: int):
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

    def _set_base_index_if_none(self, curve: IndexCurve):
        if self._index_base_mixin and self.index_base is NoInput.blank:
            self.leg1.index_base = curve.index_value(
                self.leg1.schedule.effective,
                self.leg1.index_method,
            )

    def ex_div(self, settlement: datetime):
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

    def _accrued(self, settlement: datetime, func: callable):
        """func is the specific accrued function associated with the bond ``calc_mode``"""
        acc_idx = self._period_index(settlement)
        frac = func(self, settlement, acc_idx)
        if self.ex_div(settlement):
            frac = frac - 1  # accrued is negative in ex-div period
        _ = getattr(self.leg1.periods[acc_idx], self._ytm_attribute)
        return frac * _ / -self.leg1.notional * 100

    def _generic_ytm(
        self,
        ytm: DualTypes,
        settlement: datetime,
        dirty: bool,
        f1: callable,
        f2: callable,
        f3: callable,
        accrual: callable,
    ):
        """
        Refer to supplementary material.
        """
        f = 12 / defaults.frequency_months[self.leg1.schedule.frequency]
        acc_idx = self._period_index(settlement)

        v2 = f2(self, ytm, f, settlement, acc_idx)
        v1 = f1(self, ytm, f, settlement, acc_idx, v2, accrual)
        v3 = f3(self, ytm, f, settlement, self.leg1.schedule.n_periods - 1, v2, accrual)

        # Sum up the coupon cashflows discounted by the calculated factors
        d = 0
        for i, p_idx in enumerate(range(acc_idx, self.leg1.schedule.n_periods)):
            if i == 0 and self.ex_div(settlement):
                # no coupon cashflow is receiveable so no addition to the sum
                continue
            elif i == 0 and p_idx == (self.leg1.schedule.n_periods - 1):
                # the last period is the first period so discounting handled only by v1
                d += getattr(self.leg1.periods[p_idx], self._ytm_attribute) * v1
            elif p_idx == (self.leg1.schedule.n_periods - 1):
                # this is last period, but it is not the first (i>0). Tag on v3 at end.
                d += (
                    getattr(self.leg1.periods[p_idx], self._ytm_attribute) * v2 ** (i - 1) * v3 * v1
                )
            else:
                # this is not the first and not the last period. Discount only with v1 and v2.
                d += getattr(self.leg1.periods[p_idx], self._ytm_attribute) * v2**i * v1

        # Add the redemption payment discounted by relevant factors
        if i == 0:  # only looped 1 period, only use the last discount
            d += getattr(self.leg1.periods[-1], self._ytm_attribute) * v1
        elif i == 1:  # only looped 2 periods, no need for v2
            d += getattr(self.leg1.periods[-1], self._ytm_attribute) * v3 * v1
        else:  # looped more than 2 periods, regular formula applied
            d += getattr(self.leg1.periods[-1], self._ytm_attribute) * v2 ** (i - 1) * v3 * v1

        # discount all by the first period factor and scaled to price
        p = d / -self.leg1.notional * 100

        return p if dirty else p - self._accrued(settlement, accrual)

    def _price_from_ytm(
        self,
        ytm: float,
        settlement: datetime,
        calc_mode: str | BondCalcMode | NoInput,
        dirty: bool = False,
    ):
        """
        Loop through all future cashflows and discount them with ``ytm`` to achieve
        correct price.
        """
        calc_mode_ = _drb(self.calc_mode, calc_mode)
        if isinstance(calc_mode_, str):
            calc_mode_ = BOND_MODE_MAP[calc_mode_]
        try:
            func = partial(
                self._generic_ytm,
                f1=calc_mode_._v1,
                f2=calc_mode_._v2,
                f3=calc_mode_._v3,
                accrual=calc_mode_._ytm_acc_frac_func,
            )
            return func(ytm, settlement, dirty)
        except KeyError:
            raise ValueError(f"Cannot calculate with `calc_mode`: {calc_mode}")

    def fwd_from_repo(
        self,
        price: float | Dual | Dual2,
        settlement: datetime,
        forward_settlement: datetime,
        repo_rate: float | Dual | Dual2,
        convention: str | NoInput = NoInput(0),
        dirty: bool = False,
        method: str = "proceeds",
    ):
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
        convention = defaults.convention if convention is NoInput.blank else convention
        dcf_ = dcf(settlement, forward_settlement, convention)
        if not dirty:
            d_price = price + self.accrued(settlement)
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
            if method.lower() == "proceeds":
                dcf_ = dcf(self.leg1.periods[p_idx].payment, forward_settlement, convention)
                accrued_coup = self.leg1.periods[p_idx].cashflow * (1 + dcf_ * repo_rate / 100)
                total_rtn -= accrued_coup
            elif method.lower() == "compounded":
                r_bar, d, _ = average_rate(settlement, forward_settlement, convention, repo_rate)
                n = (forward_settlement - self.leg1.periods[p_idx].payment).days
                accrued_coup = self.leg1.periods[p_idx].cashflow * (1 + d * r_bar / 100) ** n
                total_rtn -= accrued_coup
            else:
                raise ValueError("`method` must be in {'proceeds', 'compounded'}.")

        forward_price = total_rtn / -self.leg1.notional * 100
        if dirty:
            return forward_price
        else:
            return forward_price - self.accrued(forward_settlement)

    def repo_from_fwd(
        self,
        price: float | Dual | Dual2,
        settlement: datetime,
        forward_settlement: datetime,
        forward_price: float | Dual | Dual2,
        convention: str | NoInput = NoInput(0),
        dirty: bool = False,
    ):
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
        convention = defaults.convention if convention is NoInput.blank else convention
        # forward price from repo is linear in repo_rate so reverse calculate with AD
        if not dirty:
            p_t = forward_price + self.accrued(forward_settlement)
            p_0 = price + self.accrued(settlement)
        else:
            p_t, p_0 = forward_price, price

        dcf_ = dcf(settlement, forward_settlement, convention)
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
            dcf_ = dcf(self.leg1.periods[p_idx].payment, forward_settlement, convention)
            numerator += 100 * self.leg1.periods[p_idx].cashflow / -self.leg1.notional
            denominator -= 100 * dcf_ * self.leg1.periods[p_idx].cashflow / -self.leg1.notional

        return numerator / denominator * 100

    def _npv_local(
        self,
        curve: Curve | LineCurve,
        disc_curve: Curve,
        settlement: datetime,
        projection: datetime,
    ):
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
        npv = self.leg1.npv(curve, disc_curve, NoInput(0), NoInput(0))

        # now must systematically deduct any cashflow between the initial node date
        # and the settlement date, including the cashflow after settlement if ex_div.
        initial_idx = index_left(
            self.leg1.schedule.aschedule,
            self.leg1.schedule.n_periods + 1,
            disc_curve.node_dates[0],
        )
        settle_idx = index_left(
            self.leg1.schedule.aschedule,
            self.leg1.schedule.n_periods + 1,
            settlement,
        )

        for period_idx in range(initial_idx, settle_idx):
            # deduct coupon period
            npv -= self.leg1.periods[period_idx].npv(curve, disc_curve, NoInput(0), NoInput(0))

        if self.ex_div(settlement):
            # deduct coupon after settlement which is also unpaid
            npv -= self.leg1.periods[settle_idx].npv(curve, disc_curve, NoInput(0), NoInput(0))

        if projection is NoInput.blank:
            return npv
        else:
            return npv / disc_curve[projection]

    def npv(
        self,
        curves: Curve | str | list | NoInput = NoInput(0),
        solver: Solver | NoInput = NoInput(0),
        fx: float | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
        local: bool = False,
    ):
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
        curves, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            fx,
            base,
            self.leg1.currency,
        )
        settlement = self.leg1.schedule.calendar.lag(
            curves[1].node_dates[0],
            self.kwargs["settle"],
            True,
        )
        npv = self._npv_local(curves[0], curves[1], settlement, NoInput(0))
        return _maybe_local(npv, local, self.leg1.currency, fx_, base_)

    def analytic_delta(
        self,
        curve: Curve | NoInput = NoInput(0),
        disc_curve: Curve | NoInput = NoInput(0),
        fx: float | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
    ):
        """
        Return the analytic delta of the security via summing all periods.

        For arguments see :meth:`~rateslib.periods.BasePeriod.analytic_delta`.
        """
        disc_curve_: Curve | NoInput = _disc_maybe_from_curve(curve, disc_curve)
        settlement = self.leg1.schedule.calendar.lag(
            disc_curve_.node_dates[0],
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
                curve,
                disc_curve_,
                fx,
                base,
            )
        return a_delta

    def cashflows(
        self,
        curves: Curve | str | list | NoInput = NoInput(0),
        solver: Solver | NoInput = NoInput(0),
        fx: float | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
        settlement: datetime | NoInput = NoInput(0),
    ):
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
        curves, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            fx,
            base,
            self.leg1.currency,
        )
        self._set_base_index_if_none(curves[0])

        if settlement is NoInput.blank and curves[1] is NoInput.blank:
            settlement = self.leg1.schedule.effective
        elif settlement is NoInput.blank:
            settlement = self.leg1.schedule.calendar.lag(
                curves[1].node_dates[0],
                self.kwargs["settle"],
                True,
            )
        cashflows = self.leg1.cashflows(curves[0], curves[1], fx_, base_)
        if self.ex_div(settlement):
            # deduct the next coupon which has otherwise been included in valuation
            current_period = index_left(
                self.leg1.schedule.aschedule,
                self.leg1.schedule.n_periods + 1,
                settlement,
            )
            cashflows.loc[current_period, defaults.headers["npv"]] = 0
            cashflows.loc[current_period, defaults.headers["npv_fx"]] = 0
        return cashflows

    def oaspread(
        self,
        curves: Curve | str | list | NoInput = NoInput(0),
        solver: Solver | NoInput = NoInput(0),
        fx: float | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
        price: DualTypes = NoInput(0),
        dirty: bool = False,
    ):
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
        curves, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            fx,
            base,
            self.leg1.currency,
        )
        metric = "dirty_price" if dirty else "clean_price"

        # Create a discounting curve with ADOrder:1 exposure to z_spread
        disc_curve = curves[1].shift(Dual(0, ["z_spread"], []), composite=False)

        # Get forecasting curve
        if type(self).__name__ in ["FloatRateNote", "IndexFixedRateBond"]:
            fore_curve = curves[0].copy()
            fore_curve._set_ad_order(1)
        elif type(self).__name__ in ["FixedRateBond", "Bill"]:
            fore_curve = None
        else:
            raise TypeError("Method `oaspread` can only be called on Bond type securities.")

        npv_price = self.rate(curves=[fore_curve, disc_curve], metric=metric)
        # find a first order approximation of z
        b = gradient(npv_price, ["z_spread"], 1)[0]
        c = float(npv_price) - float(price)
        z_hat = -c / b

        # shift the curve to the first order approximation and fine tune with 2nd order approxim.
        disc_curve = curves[1].shift(Dual2(z_hat, ["z_spread"], [], []), composite=False)
        if fore_curve is not None:
            fore_curve._set_ad_order(2)
        npv_price = self.rate(curves=[fore_curve, disc_curve], metric=metric)
        a, b, c = (
            0.5 * gradient(npv_price, ["z_spread"], 2)[0][0],
            gradient(npv_price, ["z_spread"], 1)[0],
            float(npv_price) - float(price),
        )
        z_hat2 = quadratic_eqn(a, b, c, x0=-c / b)["g"]

        # perform one final approximation albeit the additional price calculation slows calc time
        disc_curve = curves[1].shift(z_hat + z_hat2, composite=False)
        disc_curve._set_ad_order(0)
        if fore_curve is not None:
            fore_curve._set_ad_order(0)
        npv_price = self.rate(curves=[fore_curve, disc_curve], metric=metric)
        b = b + 2 * a * z_hat2  # forecast the new gradient
        c = float(npv_price) - float(price)
        z_hat3 = -c / b

        z = z_hat + z_hat2 + z_hat3
        return z


class FixedRateBond(Sensitivities, BondMixin, BaseMixin):
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

    Attributes
    ----------
    ex_div_days : int
    settle : int
    curves : str, list, CurveType
    leg1 : FixedLeg

    Notes
    -----

    **Calculation Modes**

    The ``calc_mode`` parameter allows the calculation for **yield-to-maturity** and **accrued interest**
    to branch depending upon the particular convention of different bonds.

    The following modes are currently available with a brief description of its particular
    action:

    - *"us_gb"*: US Treasury Bond Street convention  (deprecated alias *"ust"*)
    - *"us_gb_tsy"*: US Treasury Bond Treasury convention. (deprecated alias *"ust_31bii"*)
    - *"uk_gb"*: UK Gilt DMO method. (deprecated alias *"ukg"*)
    - *"se_gb"*: Swedish Government Bond DMO convention. (deprecated alias *"sgb"*)
    - *"ca_gb"*: Canadian Government Bond DMO convention. (deprecated alias *"cadgb"*)
    - *"de_gb"*: German Government Bond (Bunds/Bobls) ICMA convention.
    - *"fr_gb"*: French Government Bond (OAT) ICMA convention.
    - *"it_gb"*: Italian Government Bond (BTP) ICMA convention.
    - *"nl_gb"*: Dutch Government Bond ICMA convention.
    - *"no_gb"*: Norwegian Government Bond DMO convention.

    More details available in supplementary materials. The table below
    outlines the *rateslib* price result relative to the calculation examples provided
    from official sources.

    .. ipython:: python
       :suppress:

       sgb = FixedRateBond(
           effective=dt(2022, 3, 30), termination=dt(2039, 3, 30),
           frequency="A", convention="ActActICMA", calc_mode="se_gb",
           fixed_rate=3.5, calendar="stk"
       )
       s1c = sgb.price(ytm=2.261, settlement=dt(2023, 3, 15), dirty=False)
       s1d = sgb.price(ytm=2.261, settlement=dt(2023, 3, 15), dirty=True)

       uk1 = FixedRateBond(
           effective=dt(1995, 1, 1), termination=dt(2015, 12, 7),
           frequency="S", convention="ActActICMA", calc_mode="uk_gb",
           fixed_rate=8.0, calendar="ldn", ex_div=7,
       )
       uk11c = uk1.price(ytm=4.445, settlement=dt(1999, 5, 24), dirty=False)
       uk11d = uk1.price(ytm=4.445, settlement=dt(1999, 5, 24), dirty=True)
       uk12c = uk1.price(ytm=4.445, settlement=dt(1999, 5, 26), dirty=False)
       uk12d = uk1.price(ytm=4.445, settlement=dt(1999, 5, 26), dirty=True)
       uk13c = uk1.price(ytm=4.445, settlement=dt(1999, 5, 27), dirty=False)
       uk13d = uk1.price(ytm=4.445, settlement=dt(1999, 5, 27), dirty=True)
       uk14c = uk1.price(ytm=4.445, settlement=dt(1999, 6, 7), dirty=False)
       uk14d = uk1.price(ytm=4.445, settlement=dt(1999, 6, 7), dirty=True)

       uk2 = FixedRateBond(
           effective=dt(1998, 11, 26), termination=dt(2004, 11, 26),
           frequency="S", convention="ActActICMA", calc_mode="uk_gb",
           fixed_rate=6.75, calendar="ldn", ex_div=7,
       )
       uk21c = uk2.price(ytm=4.634, settlement=dt(1999, 5, 10), dirty=False)
       uk21d = uk2.price(ytm=4.634, settlement=dt(1999, 5, 10), dirty=True)
       uk22c = uk2.price(ytm=4.634, settlement=dt(1999, 5, 17), dirty=False)
       uk22d = uk2.price(ytm=4.634, settlement=dt(1999, 5, 17), dirty=True)
       uk23c = uk2.price(ytm=4.634, settlement=dt(1999, 5, 18), dirty=False)
       uk23d = uk2.price(ytm=4.634, settlement=dt(1999, 5, 18), dirty=True)
       uk24c = uk2.price(ytm=4.634, settlement=dt(1999, 5, 26), dirty=False)
       uk24d = uk2.price(ytm=4.634, settlement=dt(1999, 5, 26), dirty=True)

       usA = FixedRateBond(
           effective=dt(1990, 5, 15), termination=dt(2020, 5, 15),
           frequency="S", convention="ActActICMA", calc_mode="us_gb_tsy",
           fixed_rate=8.75, calendar="nyc", ex_div=1, modifier="none",
       )

       usAc = usA.price(ytm=8.84, settlement=dt(1990, 5, 15), dirty=False)
       usAd = usA.price(ytm=8.84, settlement=dt(1990, 5, 15), dirty=True)

       usB = FixedRateBond(
           effective=dt(1990, 4, 2), termination=dt(1992, 3, 31),
           frequency="S", convention="ActActICMA", calc_mode="us_gb_tsy",
           fixed_rate=8.5, calendar="nyc", ex_div=1, modifier="none",
       )

       usBc = usB.price(ytm=8.59, settlement=dt(1990, 4, 2), dirty=False)
       usBd = usB.price(ytm=8.59, settlement=dt(1990, 4, 2), dirty=True)

       usC = FixedRateBond(
           effective=dt(1990, 3, 1), termination=dt(1995, 5, 15),
           front_stub=dt(1990, 11, 15),
           frequency="S", convention="ActActICMA", calc_mode="us_gb_tsy",
           fixed_rate=8.5, calendar="nyc", ex_div=1, modifier="none",
       )

       usCc = usC.price(ytm=8.53, settlement=dt(1990, 3, 1), dirty=False)
       usCd = usC.price(ytm=8.53, settlement=dt(1990, 3, 1), dirty=True)

       usD = FixedRateBond(
           effective=dt(1985, 11, 15), termination=dt(1995, 11, 15),
           frequency="S", convention="ActActICMA", calc_mode="us_gb_tsy",
           fixed_rate=9.5, calendar="nyc", ex_div=1, modifier="none",
       )

       usDc = usD.price(ytm=9.54, settlement=dt(1985, 11, 29), dirty=False)
       usDd = usD.price(ytm=9.54, settlement=dt(1985, 11, 29), dirty=True)

       usE = FixedRateBond(
           effective=dt(1985, 7, 2), termination=dt(2005, 8, 15),
           front_stub=dt(1986, 2, 15),
           frequency="S", convention="ActActICMA", calc_mode="us_gb_tsy",
           fixed_rate=10.75, calendar="nyc", ex_div=1, modifier="none",
       )

       usEc = usE.price(ytm=10.47, settlement=dt(1985, 11, 4), dirty=False)
       usEd = usE.price(ytm=10.47, settlement=dt(1985, 11, 4), dirty=True)

       usF = FixedRateBond(
           effective=dt(1983, 5, 16), termination=dt(1991, 5, 15), roll=15,
           frequency="S", convention="ActActICMA", calc_mode="us_gb_tsy",
           fixed_rate=10.50, calendar="nyc", ex_div=1, modifier="none",
       )

       usFc = usF.price(ytm=10.53, settlement=dt(1983, 8, 15), dirty=False)
       usFd = usF.price(ytm=10.53, settlement=dt(1983, 8, 15), dirty=True)

       usG = FixedRateBond(
           effective=dt(1988, 10, 15), termination=dt(1994, 12, 15),
           front_stub=dt(1989, 6, 15),
           frequency="S", convention="ActActICMA", calc_mode="us_gb_tsy",
           fixed_rate=9.75, calendar="nyc", ex_div=1, modifier="none",
       )

       usGc = usG.price(ytm=9.79, settlement=dt(1988, 11, 15), dirty=False)
       usGd = usG.price(ytm=9.79, settlement=dt(1988, 11, 15), dirty=True)

       data = DataFrame(data=[
               ["Riksgalden Website", "Nominal Bond", 116.514000, 119.868393, "se_gb", s1c, s1d],
               ["UK DMO Website", "Ex 1, Scen 1", None, 145.012268, "uk_gb", uk11c, uk11d],
               ["UK DMO Website", "Ex 1, Scen 2", None, 145.047301, "uk_gb", uk12c, uk12d],
               ["UK DMO Website", "Ex 1, Scen 3", None, 141.070132, "uk_gb", uk13c, uk13d],
               ["UK DMO Website", "Ex 1, Scen 4", None, 141.257676, "uk_gb", uk14c, uk14d],
               ["UK DMO Website", "Ex 2, Scen 1", None, 113.315543, "uk_gb", uk21c, uk21d],
               ["UK DMO Website", "Ex 2, Scen 2", None, 113.415969, "uk_gb", uk22c, uk22d],
               ["UK DMO Website", "Ex 2, Scen 3", None, 110.058738, "uk_gb", uk23c, uk23d],
               ["UK DMO Website", "Ex 2, Scen 4", None, 110.170218, "uk_gb", uk24c, uk24d],
               ["Title-31 Subtitle-B II", "Ex A (reg)",99.057893, 99.057893, "us_gb_tsy", usAc, usAd],
               ["Title-31 Subtitle-B II", "Ex B (stub)", 99.838183, 99.838183, "us_gb_tsy", usBc, usBd],
               ["Title-31 Subtitle-B II", "Ex C (stub)", 99.805118, 99.805118, "us_gb_tsy", usCc, usCd],
               ["Title-31 Subtitle-B II", "Ex D (reg)", 99.730918, 100.098321, "us_gb_tsy", usDc, usDd],
               ["Title-31 Subtitle-B II", "Ex E (stub)", 102.214586, 105.887384, "us_gb_tsy", usEc, usEd],
               ["Title-31 Subtitle-B II", "Ex F (stub)", 99.777074, 102.373541, "us_gb_tsy", usFc, usFd],
               ["Title-31 Subtitle-B II", "Ex G (stub)", 99.738045, 100.563865, "us_gb_tsy", usGc, usGd],
           ],
           columns=["Source", "Example", "Expected clean", "Expected dirty", "Calc mode", "Rateslib clean", "Rateslib dirty"],
       )

    .. ipython:: python

       from pandas import option_context
       with option_context("display.float_format", lambda x: '%.6f' % x):
           print(data)

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
    _ytm_attribute = "cashflow"  # nominal bonds use cashflows in YTM calculation

    def __init__(
        self,
        effective: datetime | NoInput = NoInput(0),
        termination: datetime | str | NoInput = NoInput(0),
        frequency: int | NoInput = NoInput(0),
        stub: str | NoInput = NoInput(0),
        front_stub: datetime | NoInput = NoInput(0),
        back_stub: datetime | NoInput = NoInput(0),
        roll: str | int | NoInput = NoInput(0),
        eom: bool | NoInput = NoInput(0),
        modifier: str | None | NoInput = NoInput(0),
        calendar: CalInput = NoInput(0),
        payment_lag: int | NoInput = NoInput(0),
        notional: float | NoInput = NoInput(0),
        currency: str | NoInput = NoInput(0),
        amortization: float | NoInput = NoInput(0),
        convention: str | NoInput = NoInput(0),
        fixed_rate: float | NoInput = NoInput(0),
        ex_div: int | NoInput = NoInput(0),
        settle: int | NoInput = NoInput(0),
        calc_mode: str | BondCalcMode | NoInput = NoInput(0),
        curves: list | str | Curve | NoInput = NoInput(0),
        spec: str | NoInput = NoInput(0),
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

        if self.kwargs["frequency"] is NoInput.blank:
            raise ValueError("`frequency` must be provided for Bond.")
        # elif self.kwargs["frequency"].lower() == "z":
        #     raise ValueError("FixedRateBond `frequency` must be in {M, B, Q, T, S, A}.")

        self.calc_mode = _get_calc_mode_for_class(self, self.kwargs["calc_mode"])

        self.curves = curves
        self.spec = spec

        self._fixed_rate = fixed_rate
        self.leg1 = FixedLeg(**_get(self.kwargs, leg=1, filter=["ex_div", "settle", "calc_mode"]))

        if self.leg1.amortization != 0:
            # Note if amortization is added to FixedRateBonds must systematically
            # go through and update all methods. Many rely on the quantity
            # self.notional which is currently assumed to be a fixed quantity
            raise NotImplementedError("`amortization` for FixedRateBond must be zero.")

    # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
    # Commercial use of this code, and/or copying and redistribution is prohibited.
    # Contact rateslib at gmail.com if this code is observed outside its intended sphere.

    def accrued(self, settlement: datetime):
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
        return self._accrued(settlement, self.calc_mode._settle_acc_frac_func)

    def rate(
        self,
        curves: Curve | str | list | NoInput = NoInput(0),
        solver: Solver | NoInput = NoInput(0),
        fx: float | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
        metric: str = "clean_price",
        forward_settlement: datetime | NoInput = NoInput(0),
    ):
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
            "dirty_price", "ytm"}
        forward_settlement : datetime, optional
            The forward settlement date. If not given the settlement date is inferred from the
            discount *Curve* and the ``settle`` attribute.

        Returns
        -------
        float, Dual, Dual2
        """
        curves, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            fx,
            base,
            self.leg1.currency,
        )

        metric = metric.lower()
        if metric in ["clean_price", "dirty_price", "ytm"]:
            if forward_settlement is NoInput.blank:
                settlement = self.leg1.schedule.calendar.lag(
                    curves[1].node_dates[0],
                    self.kwargs["settle"],
                    True,
                )
            else:
                settlement = forward_settlement
            npv = self._npv_local(curves[0], curves[1], settlement, settlement)
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

    def ytm(self, price: DualTypes, settlement: datetime, dirty: bool = False) -> DualTypes:
        """
        Calculate the yield-to-maturity of the security given its price.

        Parameters
        ----------
        price : float
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

        def root(y):
            # we set this to work in float arithmetic for efficiency. Dual is added
            # back below, see PR GH3
            return self._price_from_ytm(y, settlement, self.calc_mode, dirty) - float(price)

        # x = brentq(root, -99, 10000)  # remove dependence to scipy.optimize.brentq
        # x, iters = _brents(root, -99, 10000)  # use own local brents code
        x = _ytm_quadratic_converger2(root, -3.0, 2.0, 12.0)  # use special quad interp

        if isinstance(price, Dual):
            # use the inverse function theorem to express x as a Dual
            p = self._price_from_ytm(Dual(x, ["y"], []), settlement, self.calc_mode, dirty)
            return Dual(x, price.vars, 1 / gradient(p, ["y"])[0] * price.dual)
        elif isinstance(price, Dual2):
            # use the IFT in 2nd order to express x as a Dual2
            p = self._price_from_ytm(Dual2(x, ["y"], [], []), settlement, self.calc_mode, dirty)
            dydP = 1 / gradient(p, ["y"])[0]
            d2ydP2 = -gradient(p, ["y"], order=2)[0][0] * gradient(p, ["y"])[0] ** -3
            dual = dydP * price.dual
            dual2 = 0.5 * (
                dydP * gradient(price, price.vars, order=2)
                + d2ydP2 * np.matmul(price.dual[:, None], price.dual[None, :])
            )

            return Dual2(x, price.vars, dual.tolist(), list(dual2.flat))
        else:
            return x

    def duration(self, ytm: float, settlement: datetime, metric: str = "risk"):
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
        if metric == "risk":
            _ = -gradient(self.price(Dual(float(ytm), ["y"], []), settlement), ["y"])[0]
        elif metric == "modified":
            price = -self.price(Dual(float(ytm), ["y"], []), settlement, dirty=True)
            _ = -gradient(price, ["y"])[0] / float(price) * 100
        elif metric == "duration":
            price = self.price(Dual(float(ytm), ["y"], []), settlement, dirty=True)
            f = 12 / defaults.frequency_months[self.kwargs["frequency"].upper()]
            v = 1 + float(ytm) / (100 * f)
            _ = -gradient(price, ["y"])[0] / float(price) * v * 100
        return _

    def convexity(self, ytm: float, settlement: datetime):
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
        _ = self.price(Dual2(float(ytm), ["y"], [], []), settlement)
        return gradient(_, ["y"], 2)[0][0]

    def price(self, ytm: float, settlement: datetime, dirty: bool = False):
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
        return self._price_from_ytm(ytm, settlement, self.calc_mode, dirty)

    def delta(self, *args, **kwargs):
        """
        Calculate the delta of the *Instrument*.

        For arguments see :meth:`Sensitivities.delta()<rateslib.instruments.Sensitivities.delta>`.
        """
        return super().delta(*args, **kwargs)

    def gamma(self, *args, **kwargs):
        """
        Calculate the gamma of the *Instrument*.

        For arguments see :meth:`Sensitivities.gamma()<rateslib.instruments.Sensitivities.gamma>`.
        """
        return super().gamma(*args, **kwargs)


class IndexFixedRateBond(FixedRateBond):
    # TODO (mid) ensure calculations work for amortizing bonds.
    """
    Create an indexed fixed rate bond security.

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
    _ytm_attribute = "real_cashflow"  # index linked bonds use real cashflows
    _index_base_mixin = True

    def __init__(
        self,
        effective: datetime | NoInput = NoInput(0),
        termination: datetime | str | NoInput = NoInput(0),
        frequency: int | NoInput = NoInput(0),
        stub: str | NoInput = NoInput(0),
        front_stub: datetime | NoInput = NoInput(0),
        back_stub: datetime | NoInput = NoInput(0),
        roll: str | int | NoInput = NoInput(0),
        eom: bool | NoInput = NoInput(0),
        modifier: str | None | NoInput = NoInput(0),
        calendar: CalInput = NoInput(0),
        payment_lag: int | NoInput = NoInput(0),
        notional: float | NoInput = NoInput(0),
        currency: str | NoInput = NoInput(0),
        amortization: float | NoInput = NoInput(0),
        convention: str | NoInput = NoInput(0),
        fixed_rate: float | NoInput = NoInput(0),
        index_base: float | Series | NoInput = NoInput(0),
        index_fixings: float | Series | NoInput = NoInput(0),
        index_method: str | NoInput = NoInput(0),
        index_lag: int | NoInput = NoInput(0),
        ex_div: int | NoInput = NoInput(0),
        settle: int | NoInput = NoInput(0),
        calc_mode: str | BondCalcMode | NoInput = NoInput(0),
        curves: list | str | Curve | NoInput = NoInput(0),
        spec: str | NoInput = NoInput(0),
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

        if self.kwargs["frequency"] is NoInput.blank:
            raise ValueError("`frequency` must be provided for Bond.")
        # elif self.kwargs["frequency"].lower() == "z":
        #     raise ValueError("FixedRateBond `frequency` must be in {M, B, Q, T, S, A}.")

        self.calc_mode = _get_calc_mode_for_class(self, self.kwargs["calc_mode"])

        self.curves = curves
        self.spec = spec

        self._fixed_rate = fixed_rate
        self._index_base = index_base

        self.leg1 = IndexFixedLeg(
            **_get(self.kwargs, leg=1, filter=["ex_div", "settle", "calc_mode"]),
        )
        if self.leg1.amortization != 0:
            # Note if amortization is added to IndexFixedRateBonds must systematically
            # go through and update all methods. Many rely on the quantity
            # self.notional which is currently assumed to be a fixed quantity
            raise NotImplementedError("`amortization` for IndexFixedRateBond must be zero.")

    def index_ratio(self, settlement: datetime, curve: IndexCurve | NoInput):
        if self.leg1.index_fixings is not NoInput.blank and not isinstance(
            self.leg1.index_fixings,
            Series,
        ):
            raise ValueError(
                "Must provide `index_fixings` as a Series for inter-period settlement.",
            )
        # TODO: this indexing of periods assumes no amortization
        index_val = IndexMixin._index_value(
            i_fixings=self.leg1.index_fixings,
            i_curve=curve,
            i_lag=self.leg1.index_lag,
            i_method=self.leg1.index_method,
            i_date=settlement,
        )
        index_base = IndexMixin._index_value(
            i_fixings=self.index_base,
            i_date=self.leg1.schedule.effective,
            i_lag=self.leg1.index_lag,
            i_method=self.leg1.index_method,
            i_curve=curve,
        )
        return index_val / index_base

    def rate(
        self,
        curves: Curve | str | list | NoInput = NoInput(0),
        solver: Solver | NoInput = NoInput(0),
        fx: float | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
        metric: str = "clean_price",
        forward_settlement: datetime | NoInput = NoInput(0),
    ):
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

        curves, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            fx,
            base,
            self.leg1.currency,
        )

        metric = metric.lower()
        if metric in [
            "clean_price",
            "dirty_price",
            "index_clean_price",
            "ytm",
            "index_dirty_price",
        ]:
            if forward_settlement is NoInput.blank:
                settlement = self.leg1.schedule.calendar.lag(
                    curves[1].node_dates[0],
                    self.kwargs["settle"],
                    True,
                )
            else:
                settlement = forward_settlement
            npv = self._npv_local(curves[0], curves[1], settlement, settlement)
            # scale price to par 100 (npv is already projected forward to settlement)
            index_dirty_price = npv * 100 / -self.leg1.notional
            index_ratio = self.index_ratio(settlement, curves[0])
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

    def __init__(
        self,
        effective: datetime | NoInput = NoInput(0),
        termination: datetime | str | NoInput = NoInput(0),
        frequency: str | NoInput = NoInput(0),
        modifier: str | None | NoInput = NoInput(0),
        calendar: CalInput = NoInput(0),
        payment_lag: int | NoInput = NoInput(0),
        notional: float | NoInput = NoInput(0),
        currency: str | NoInput = NoInput(0),
        convention: str | NoInput = NoInput(0),
        settle: str | NoInput = NoInput(0),
        calc_mode: str | BillCalcMode | NoInput = NoInput(0),
        curves: list | str | Curve | NoInput = NoInput(0),
        spec: str | NoInput = NoInput(0),
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
        )
        self.kwargs["frequency"] = _drb(
            self.calc_mode._ytm_clone_kwargs["frequency"],
            frequency,
        )

    @property
    def dcf(self):
        # bills will typically have 1 period since they are configured with frequency "z".
        d = 0.0
        for i in range(self.leg1.schedule.n_periods):
            d += self.leg1.periods[i].dcf
        return d

    def rate(
        self,
        curves: Curve | str | list | NoInput = NoInput(0),
        solver: Solver | NoInput = NoInput(0),
        fx: float | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
        metric="price",
    ):
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
            Metric returned by the method.

        Returns
        -------
        float, Dual, Dual2
        """
        curves, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            fx,
            base,
            self.leg1.currency,
        )
        settlement = self.leg1.schedule.calendar.lag(
            curves[1].node_dates[0],
            self.kwargs["settle"],
            True,
        )
        # scale price to par 100 and make a fwd adjustment according to curve
        price = (
            self.npv(curves, solver, fx_, base_)
            * 100
            / (-self.leg1.notional * curves[1][settlement])
        )
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
        acc_frac = self.calc_mode._settle_acc_frac_func(self, settlement, 0)
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
        acc_frac = self.calc_mode._settle_acc_frac_func(self, settlement, 0)
        dcf = (1 - acc_frac) * self.dcf
        rate = ((1 - price / 100) / dcf) * 100
        return rate

    def price(
        self,
        rate: DualTypes,
        settlement: datetime,
        dirty: bool = False,
        calc_mode: str | NoInput = NoInput(0),
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
        if not isinstance(calc_mode, str):
            calc_mode = self.calc_mode
        price_func = getattr(self, f"_price_{self.calc_mode._price_type}")
        return price_func(rate, settlement)

    def _price_discount(self, rate: DualTypes, settlement: datetime):
        acc_frac = self.calc_mode._settle_acc_frac_func(self, settlement, 0)
        dcf = (1 - acc_frac) * self.dcf
        return 100 - rate * dcf

    def _price_simple(self, rate: DualTypes, settlement: datetime):
        acc_frac = self.calc_mode._settle_acc_frac_func(self, settlement, 0)
        dcf = (1 - acc_frac) * self.dcf
        return 100 / (1 + rate * dcf / 100)

    def ytm(
        self,
        price: DualTypes,
        settlement: datetime,
        calc_mode: str | BillCalcMode | NoInput = NoInput(0),
    ):
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

        if calc_mode is NoInput.blank:
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
                filter=["initial_exchange", "final_exchange", "payment_lag_exchange"],
            ),
        )
        return equiv_bond.ytm(price, settlement)

    def duration(self, *args, **kwargs):
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

           bill = Bill(effective=dt(2024, 2, 29), termination=dt(2024, 8, 29), spec="ustb")
           bill.duration(settlement=dt(2024, 5, 30), ytm=5.2525, metric="duration")

           bill = Bill(effective=dt(2024, 2, 29), termination=dt(2024, 8, 29), spec="ustb", frequency="A")
           bill.duration(settlement=dt(2024, 5, 30), ytm=5.2525, metric="duration")

        """  # noqa: E501
        return super().duration(*args, **kwargs)


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


class FloatRateNote(Sensitivities, BondMixin, BaseMixin):
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

    def __init__(
        self,
        effective: datetime | NoInput = NoInput(0),
        termination: datetime | str | NoInput = NoInput(0),
        frequency: int | NoInput = NoInput(0),
        stub: str | NoInput = NoInput(0),
        front_stub: datetime | NoInput = NoInput(0),
        back_stub: datetime | NoInput = NoInput(0),
        roll: str | int | NoInput = NoInput(0),
        eom: bool | NoInput = NoInput(0),
        modifier: str | None | NoInput = NoInput(0),
        calendar: CalInput = NoInput(0),
        payment_lag: int | NoInput = NoInput(0),
        notional: float | NoInput = NoInput(0),
        currency: str | NoInput = NoInput(0),
        amortization: float | NoInput = NoInput(0),
        convention: str | NoInput = NoInput(0),
        float_spread: float = NoInput(0),
        fixings: float | list | NoInput = NoInput(0),
        fixing_method: str | NoInput = NoInput(0),
        method_param: int | NoInput = NoInput(0),
        spread_compound_method: str | NoInput = NoInput(0),
        ex_div: int | NoInput = NoInput(0),
        settle: int | NoInput = NoInput(0),
        calc_mode: str | NoInput = NoInput(0),
        curves: list | str | Curve | NoInput = NoInput(0),
        spec: str | NoInput = NoInput(0),
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

        if self.kwargs["frequency"] is NoInput.blank:
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
        self.leg1 = FloatLeg(**_get(self.kwargs, leg=1, filter=["ex_div", "settle", "calc_mode"]))

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

    def _accrual_rate(self, pseudo_period, curve, method_param):
        """
        Take a period and try to forecast the rate which determines the accrual,
        either from known fixings, a curve or forward filling historical fixings.

        This method is required to handle the case where a curve is not provided and
        fixings are enough.
        """
        if pseudo_period.dcf < 1e-10:
            return 0.0  # there are no fixings in the period.

        if curve is not NoInput.blank:
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
                _ = pseudo_period.rate(pseudo_curve)
                return _
            except IndexError:
                # the pseudo_curve has no nodes so when it needs to calculate a rate it cannot
                # be indexed.
                # Try to revert back to using the last fixing as forward projection.
                try:
                    if isinstance(pseudo_period.fixings, Series):
                        last_fixing = pseudo_period.fixings.iloc[-1]
                    else:
                        last_fixing = pseudo_period.fixings[-1]
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
        _ = float(pseudo_period.rate(curve_))
        return _

    def accrued(
        self,
        settlement: datetime,
        curve: Curve | NoInput = NoInput(0),
    ):
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
            frac = self.calc_mode._settle_acc_frac_func(self, settlement, acc_idx)
            if self.ex_div(settlement):
                frac = frac - 1  # accrued is negative in ex-div period

            if curve is not NoInput.blank:
                curve_ = curve
            else:
                curve_ = Curve(
                    {  # create a dummy curve. rate() will return the fixing
                        self.leg1.periods[acc_idx].start: 1.0,
                        self.leg1.periods[acc_idx].end: 1.0,
                    },
                )
            rate = self.leg1.periods[acc_idx].rate(curve_)

            cashflow = (
                -self.leg1.periods[acc_idx].notional * self.leg1.periods[acc_idx].dcf * rate / 100
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
                    self.leg1.periods[acc_idx],
                    curve,
                    self.leg1.method_param,
                )
                accrued_to_end = 100 * self.leg1.periods[acc_idx].dcf * rate_to_end / 100
                return accrued_to_settle - accrued_to_end
            else:
                return accrued_to_settle

    def rate(
        self,
        curves: Curve | str | list | NoInput = NoInput(0),
        solver: Solver | NoInput = NoInput(0),
        fx: float | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
        metric="clean_price",
        forward_settlement: datetime | NoInput = NoInput(0),
    ):
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
            "dirty_price", "spread"}
        forward_settlement : datetime, optional
            The forward settlement date. If not give uses the discount *Curve* and the bond's
            ``settle`` attribute.}.

        Returns
        -------
        float, Dual, Dual2

        """
        curves, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            fx,
            base,
            self.leg1.currency,
        )

        metric = metric.lower()
        if metric in ["clean_price", "dirty_price", "spread"]:
            if forward_settlement is NoInput.blank:
                settlement = self.leg1.schedule.calendar.lag(
                    curves[1].node_dates[0],
                    self.kwargs["settle"],
                    True,
                )
            else:
                settlement = forward_settlement
            npv = self._npv_local(curves[0], curves[1], settlement, settlement)
            # scale price to par 100 (npv is already projected forward to settlement)
            dirty_price = npv * 100 / -self.leg1.notional

            if metric == "dirty_price":
                return dirty_price
            elif metric == "clean_price":
                return dirty_price - self.accrued(settlement, curve=curves[0])
            elif metric == "spread":
                _ = self.leg1._spread(-(npv + self.leg1.notional), curves[0], curves[1])
                z = 0.0 if self.float_spread is NoInput.blank else self.float_spread
                return _ + z

        raise ValueError("`metric` must be in {'dirty_price', 'clean_price', 'spread'}.")

    def delta(self, *args, **kwargs):
        """
        Calculate the delta of the *Instrument*.

        For arguments see :meth:`Sensitivities.delta()<rateslib.instruments.Sensitivities.delta>`.
        """
        return super().delta(*args, **kwargs)

    def gamma(self, *args, **kwargs):
        """
        Calculate the gamma of the *Instrument*.

        For arguments see :meth:`Sensitivities.gamma()<rateslib.instruments.Sensitivities.gamma>`.
        """
        return super().gamma(*args, **kwargs)

    def fixings_table(
        self,
        curves: Curve | str | list | NoInput = NoInput(0),
        solver: Solver | NoInput = NoInput(0),
        fx: float | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
        approximate: bool = False,
        right: datetime | NoInput = NoInput(0),
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
        curves, _, _ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            NoInput(0),
            NoInput(0),
            self.leg1.currency,
        )
        df = self.leg1.fixings_table(
            curve=curves[0], approximate=approximate, disc_curve=curves[1], right=right
        )
        return df


def _ytm_quadratic_converger2(f, y0, y1, y2, f0=None, f1=None, f2=None, tol=1e-9):
    """
    Convert a price from yield function `f` into a quadratic approximation and
    determine the root, yield, which matches the target price.
    """
    # allow function values to be passed recursively to avoid re-calculation
    f0 = f0 if f0 is not None else f(y0)
    f1 = f1 if f1 is not None else f(y1)
    f2 = f2 if f2 is not None else f(y2)

    if f0 < 0 and f1 < 0 and f2 < 0:
        # reassess initial values
        return _ytm_quadratic_converger2(f, 2 * y0 - y2, y1 - y2 + y0, y0, None, None, f0, tol)
    elif f0 > 0 and f1 > 0 and f2 > 0:
        return _ytm_quadratic_converger2(
            f,
            y2,
            y1 + 1 * (y2 - y0),
            y2 + 2 * (y2 - y0),
            f2,
            None,
            None,
            tol,
        )

    _b = np.array([y0, y1, y2])[:, None]
    # check tolerance from previous recursive estimations
    for i, f_ in enumerate([f0, f1, f2]):
        if abs(f_) < tol:
            return _b[i, 0]

    _A = np.array([[f0**2, f0, 1], [f1**2, f1, 1], [f2**2, f2, 1]])
    c = np.linalg.solve(_A, _b)
    y = c[2, 0]
    f_ = f(y)
    if abs(f_) < tol:
        return y

    pad = min(tol * 1e8, 0.0001, abs(f_ * 1e4))  # avoids singular matrix error
    if y <= y0:
        # line not hit due to reassessment of initial vars?
        return _ytm_quadratic_converger2(
            f,
            2 * y - y0 - pad,
            y,
            y0 + pad,
            None,
            f_,
            None,
            tol,
        )  # pragma: no cover
    elif y0 < y <= y1:
        if (y - y0) < (y1 - y):
            return _ytm_quadratic_converger2(f, y0, y, 2 * y - y0 + pad, f0, f_, None, tol)
        else:
            return _ytm_quadratic_converger2(f, 2 * y - y1 - pad, y, y1, None, f_, f1, tol)
    elif y1 < y <= y2:
        if (y - y1) < (y2 - y):
            return _ytm_quadratic_converger2(f, y1, y, 2 * y - y1 + pad, f1, f_, None, tol)
        else:
            return _ytm_quadratic_converger2(f, 2 * y - y2 - pad, y, y2, None, f_, f2, tol)
    else:  # y2 < y:
        # line not hit due to reassessment of initial vars?
        return _ytm_quadratic_converger2(
            f,
            y2 - pad,
            y,
            2 * y - y2 + pad,
            None,
            f_,
            None,
            tol,
        )  # pragma: no cover


# def _ytm_quadratic_converger(f, y0, y1, y2, tol=1e-9):
#     """
#     Convert a price from yield function `f` into a quadratic approximation and
#     determine the root, yield, which matches the target price.
#     """
#     _A = np.array([[y0**2, y0, 1], [y1**2, y1, 1], [y2**2, y2, 1]])
#     _b = np.array([f(y0), f(y1), f(y2)])[:, None]
#
#     # check tolerance from previous recursive estimations
#     if abs(_b[1, 0]) < tol:
#         return y1
#
#     c = np.linalg.solve(_A, _b)
#
#     yield1 = ((-c[1] + sqrt(c[1]**2 - 4 * c[0] * c[2])) / (2 * c[0]))[0]
#     yield2 = ((-c[1] - sqrt(c[1]**2 - 4 * c[0] * c[2])) / (2 * c[0]))[0]
#     z1, z2 = f(yield1), f(yield2)
#
#     # make a linear guess at new quadratic solution
#     approx_yield = yield1 - (yield2 - yield1) * z1 / (z2 - z1)
#     if abs(z1) < abs(z2):
#         soln_yield = yield1
#         if abs(z1) < tol:
#             return soln_yield
#     else:
#         soln_yield = yield2
#         if abs(z2) < tol:
#             return soln_yield
#     return _ytm_quadratic_converger(
#         f,
#         approx_yield - max(10 * (approx_yield - soln_yield), 0.001),
#         approx_yield,
#         approx_yield + max(10 * (approx_yield - soln_yield), 0.001),
#         tol
#     )
