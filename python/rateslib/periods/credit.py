from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from typing import TYPE_CHECKING

from rateslib import defaults
from rateslib.default import NoInput, _drb
from rateslib.dual import Dual, Dual2, Variable, gradient
from rateslib.dual.utils import _dual_float
from rateslib.periods.base import BasePeriod
from rateslib.periods.utils import (
    _get_fx_and_base,
    _maybe_fx_converted,
    _maybe_local,
    _validate_credit_curves,
)

if TYPE_CHECKING:
    from rateslib.typing import (
        FX_,
        Any,
        Curve,
        Curve_,
        CurveOption_,
        DualTypes,
        DualTypes_,
        datetime,
        str_,
    )


class CreditPremiumPeriod(BasePeriod):
    """
    Create a credit premium period defined by a credit spread.

    Parameters
    ----------
    args : dict
        Required positional args to :class:`BasePeriod`.
    fixed_rate : float or None, optional
        The rate applied to determine the cashflow. If `None`, can be set later,
        typically after a mid-market rate for all periods has been calculated.
        Entered in percentage points, e.g 50bps is 0.50.
    premium_accrued : bool, optional
        Whether the premium is accrued within the period to default.
    kwargs : dict
        Required keyword arguments to :class:`BasePeriod`.

    Notes
    -----
    The ``cashflow`` is defined as follows;

    .. math::

       C = -NdS

    The NPV of the full cashflow is defined as;

    .. math::

       P_c = Cv(m_{payment})Q(m_{end})

    If ``premium_accrued`` is permitted then an additional component equivalent to the following
    is calculated using an approximation of the inter-period default rate,

    .. math::

       P_a = Cv(m_{payment}) \\left ( Q(m_{start}) - Q(m_{end}) \\right ) \\frac{(n+r)}{2n}

    where *r* is the number of days after the *start* that *today* is for an on-going period, zero otherwise, and
    :math:`Q(m_{start})` is equal to one for an on-going period.

    The :meth:`~rateslib.periods.BasePeriod.npv` is defined as;

    .. math::

       P = P_c + I_{pa} P_a

    where :math:`I_{pa}` is an indicator function if the *Period* allows ``premium_accrued`` or not.

    The :meth:`~rateslib.periods.BasePeriod.analytic_delta` is defined as;

    .. math::

       A = - \\frac{\\partial P}{\\partial S} = Ndv(m) \\left ( Q(m_{end}) + I_{pa} (Q(m_{start}) - Q(m_{end}) \\frac{(n+r)}{2n}  \\right )
    """  # noqa: E501

    def __init__(
        self,
        *args: Any,
        fixed_rate: DualTypes_ = NoInput(0),
        premium_accrued: bool | NoInput = NoInput(0),
        **kwargs: Any,
    ) -> None:
        self.premium_accrued: bool = _drb(defaults.cds_premium_accrued, premium_accrued)
        self.fixed_rate: DualTypes_ = fixed_rate
        super().__init__(*args, **kwargs)

    @property
    def cashflow(self) -> DualTypes | None:
        """
        float, Dual or Dual2 : The calculated value from rate, dcf and notional.
        """
        if isinstance(self.fixed_rate, NoInput):
            return None
        else:
            _: DualTypes = -self.notional * self.dcf * self.fixed_rate * 0.01
            return _

    def accrued(self, settlement: datetime) -> DualTypes | None:
        """
        Calculate the amount of premium accrued until a specific date within the *Period*.

        Parameters
        ----------
        settlement: datetime
            The date against which accrued is measured.

        Returns
        -------
        float
        """
        if self.cashflow is None:  # self.fixed_rate is NoInput
            return None
        else:
            if settlement <= self.start or settlement >= self.end:
                return 0.0
            return self.cashflow * (settlement - self.start).days / (self.end - self.start).days

    def npv(
        self,
        curve: CurveOption_ = NoInput(0),
        disc_curve: CurveOption_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str | NoInput = NoInput(0),
        local: bool = False,
    ) -> DualTypes | dict[str, DualTypes]:
        """
        Return the NPV of the *CreditPremiumPeriod*.
        See :meth:`BasePeriod.npv()<rateslib.periods.BasePeriod.npv>`
        """
        curve_, disc_curve_ = _validate_credit_curves(curve, disc_curve)

        if isinstance(self.fixed_rate, NoInput):
            raise ValueError("`fixed_rate` must be set as a value to return a valid NPV.")
        v_payment = disc_curve_[self.payment]
        q_end = curve_[self.end]
        _ = 0.0
        if self.premium_accrued:
            v_end = disc_curve_[self.end]
            n = _dual_float((self.end - self.start).days)

            if self.start < curve_.nodes.initial:
                # then mid-period valuation
                r: float = _dual_float((curve_.nodes.initial - self.start).days)
                q_start: DualTypes = 1.0
                _v_start: DualTypes = 1.0
            else:
                r, q_start, _v_start = 0.0, curve_[self.start], disc_curve_[self.start]

            # method 1:
            _ = 0.5 * (1 + r / n)
            _ *= q_start - q_end
            _ *= v_end

            # # method 4 EXACT
            # _ = 0.0
            # for i in range(1, int(s)):
            #     m_i, m_i2 = m_today + timedelta(days=i-1), m_today + timedelta(days=i)
            #     _ += (
            #     (i + r) / n * disc_curve[m_today + timedelta(days=i)] * (curve[m_i] - curve[m_i2])
            #     )

        return _maybe_local(self.cashflow * (q_end * v_payment + _), local, self.currency, fx, base)

    def analytic_delta(
        self,
        curve: CurveOption_ = NoInput(0),
        disc_curve: CurveOption_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
    ) -> DualTypes:
        """
        Return the analytic delta of the *CreditPremiumPeriod*.
        See
        :meth:`BasePeriod.analytic_delta()<rateslib.periods.BasePeriod.analytic_delta>`
        """
        curve_, disc_curve_ = _validate_credit_curves(curve, disc_curve)

        v_payment = disc_curve_[self.payment]
        q_end = curve_[self.end]
        _ = 0.0
        if self.premium_accrued:
            v_end = disc_curve_[self.end]
            n = _dual_float((self.end - self.start).days)

            if self.start < curve_.nodes.initial:
                # then mid-period valuation
                r: float = _dual_float((curve_.nodes.initial - self.start).days)
                q_start: DualTypes = 1.0
                _v_start: DualTypes = 1.0
            else:
                r = 0.0
                q_start = curve_[self.start]
                _v_start = disc_curve_[self.start]

            # method 1:
            _ = 0.5 * (1 + r / n)
            _ *= q_start - q_end
            _ *= v_end

        return _maybe_fx_converted(
            0.0001 * self.notional * self.dcf * (q_end * v_payment + _),
            self.currency,
            fx,
            base,
        )

    def cashflows(
        self,
        curve: Curve_ = NoInput(0),  # type: ignore[override]
        disc_curve: Curve_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
    ) -> dict[str, Any]:
        """
        Return the cashflows of the *CreditPremiumPeriod*.
        See
        :meth:`BasePeriod.cashflows()<rateslib.periods.BasePeriod.cashflows>`
        """
        fx, base = _get_fx_and_base(self.currency, fx, base)

        if not isinstance(curve, NoInput) and not isinstance(disc_curve, NoInput):
            npv_: DualTypes = self.npv(curve, disc_curve)  # type: ignore[assignment]
            npv = _dual_float(npv_)
            npv_fx = npv * _dual_float(fx)
            survival = _dual_float(curve[self.end])
        else:
            npv, npv_fx, survival = None, None, None

        return {
            **super().cashflows(curve, disc_curve, fx, base),
            defaults.headers["rate"]: None
            if isinstance(self.fixed_rate, NoInput)
            else _dual_float(self.fixed_rate),
            defaults.headers["survival"]: survival,
            defaults.headers["cashflow"]: None
            if self.cashflow is None
            else _dual_float(self.cashflow),
            defaults.headers["npv"]: npv,
            defaults.headers["fx"]: _dual_float(fx),
            defaults.headers["npv_fx"]: npv_fx,
        }


class CreditProtectionPeriod(BasePeriod):
    """
    Create a credit protection period defined by a recovery rate.

    Parameters
    ----------
    args : dict
        Required positional args to :class:`BasePeriod`.
    recovery_rate : float, Dual, Dual2, optional
        The assumed recovery rate that defines payment on credit default. Set by ``defaults``.
    kwargs : dict
        Required keyword arguments to :class:`BasePeriod`.

    Notes
    -----
    The ``cashflow``, paid on a credit event, is defined as follows;

    .. math::

       C = -N(1-R)

    where *R* is the recovery rate.

    The :meth:`~rateslib.periods.BasePeriod.npv` is defined as a discretized sum of inter-period blocks whose
    probability of default and protection payment sum to give an expected payment;

    .. math::

       j &= [n/discretization] \\\\
       P &= C \\sum_{i=1}^{j} \\frac{1}{2} \\left ( v(m_{i-1}) + v_(m_{i}) \\right ) \\left ( Q(m_{i-1}) - Q(m_{i}) \\right ) \\\\

    The *start* and *end* of the period are restricted by the *Curve* if the *Period* is current (i.e. *today* is
    later than *start*)

    The :meth:`~rateslib.periods.BasePeriod.analytic_delta` is defined as;

    .. math::

       A = 0
    """  # noqa: E501

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

    def cashflow(self, curve: Curve) -> DualTypes:
        """
        float, Dual or Dual2 : The calculated protection amount determined from notional
        and recovery rate.
        """
        return -self.notional * (1 - curve.meta.credit_recovery_rate)

    def npv(
        self,
        curve: CurveOption_ = NoInput(0),
        disc_curve: CurveOption_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str | NoInput = NoInput(0),
        local: bool = False,
    ) -> DualTypes | dict[str, DualTypes]:
        """
        Return the NPV of the *CreditProtectionPeriod*.
        See :meth:`BasePeriod.npv()<rateslib.periods.BasePeriod.npv>`
        """
        curve_, disc_curve_ = _validate_credit_curves(curve, disc_curve)
        discretization = curve_.meta.credit_discretization

        if self.start < curve_.nodes.initial:
            s2 = curve_.nodes.initial
        else:
            s2 = self.start

        value: DualTypes = 0.0
        q2: DualTypes = curve_[s2]
        v2: DualTypes = disc_curve_[s2]
        while s2 < self.end:
            q1, v1 = q2, v2
            s2 = s2 + timedelta(days=discretization)
            if s2 > self.end:
                s2 = self.end
            q2, v2 = curve_[s2], disc_curve_[s2]
            value += 0.5 * (v1 + v2) * (q1 - q2)
            # value += v2 * (q1 - q2)

        value *= self.cashflow(curve_)
        return _maybe_local(value, local, self.currency, fx, base)

    def analytic_delta(
        self,
        curve: CurveOption_ = NoInput(0),
        disc_curve: CurveOption_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
    ) -> DualTypes:
        """
        Return the analytic delta of the *CreditProtectionPeriod*.
        See
        :meth:`BasePeriod.analytic_delta()<rateslib.periods.BasePeriod.analytic_delta>`
        """
        return 0.0

    def cashflows(
        self,
        curve: Curve_ = NoInput(0),  # type: ignore[override]
        disc_curve: Curve_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
    ) -> dict[str, Any]:
        """
        Return the cashflows of the *CreditProtectionPeriod*.
        See
        :meth:`BasePeriod.cashflows()<rateslib.periods.BasePeriod.cashflows>`
        """
        fx, base = _get_fx_and_base(self.currency, fx, base)

        if not isinstance(curve, NoInput) and not isinstance(disc_curve, NoInput):
            npv_: DualTypes = self.npv(curve, disc_curve)  # type: ignore[assignment]
            npv = _dual_float(npv_)
            npv_fx = npv * _dual_float(fx)
            survival = _dual_float(curve[self.end])
            rec = _dual_float(curve.meta.credit_recovery_rate)
            cashf = _dual_float(self.cashflow(curve))
        else:
            rec, npv, npv_fx, survival, cashf = None, None, None, None, None

        return {
            **super().cashflows(curve, disc_curve, fx, base),
            defaults.headers["recovery"]: rec,
            defaults.headers["survival"]: survival,
            defaults.headers["cashflow"]: cashf,
            defaults.headers["npv"]: npv,
            defaults.headers["fx"]: _dual_float(fx),
            defaults.headers["npv_fx"]: npv_fx,
        }

    def analytic_rec_risk(
        self,
        curve: Curve_ = NoInput(0),
        disc_curve: Curve_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
    ) -> float:
        """
        Calculate the exposure of the NPV to a change in recovery rate.

        For parameters see
        :meth:`BasePeriod.analytic_delta()<rateslib.periods.BasePeriod.analytic_delta>`

        Returns
        -------
        float
        """
        curve_, disc_curve_ = _validate_credit_curves(curve, disc_curve)
        haz_curve = curve_.copy()
        haz_curve._meta = replace(
            curve_.meta,
            _credit_recovery_rate=Variable(
                _dual_float(curve_.meta.credit_recovery_rate), ["__rec_rate__"], []
            ),
        )
        pv: Dual | Dual2 | Variable = self.npv(haz_curve, disc_curve_, fx, base, False)  # type: ignore[assignment]
        _: float = _dual_float(gradient(pv, ["__rec_rate__"], order=1)[0])
        return _ * 0.01
