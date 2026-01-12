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

from dataclasses import replace
from datetime import timedelta
from typing import TYPE_CHECKING

import rateslib.errors as err
from rateslib import defaults
from rateslib.dual import Variable, gradient
from rateslib.dual.utils import _dual_float
from rateslib.enums.generics import Err, NoInput, Ok, Result, _drb
from rateslib.periods.parameters import (
    _CreditParams,
    _FixedRateParams,
    _PeriodParams,
    _SettlementParams,
)
from rateslib.periods.protocols import _BasePeriod
from rateslib.periods.utils import _try_validate_base_curve, _validate_credit_curves
from rateslib.scheduling import Frequency, get_calendar
from rateslib.scheduling.adjuster import _get_adjuster
from rateslib.scheduling.convention import _get_convention
from rateslib.scheduling.frequency import _get_frequency

if TYPE_CHECKING:  # pragma: no cover
    from rateslib.typing import (
        FX_,
        Adjuster,
        CalInput,
        CurveOption_,
        DualTypes,
        DualTypes_,
        FXForwards_,
        FXRevised_,
        RollDay,
        _BaseCurve,
        _BaseCurve_,
        _FXVolOption_,
        bool_,
        datetime,
        datetime_,
        str_,
    )


class CreditPremiumPeriod(_BasePeriod):
    r"""
    A *Period* defined by a fixed interest rate and contingent credit event.

    The immediate expected valuation of the *Period* cashflow is defined as;

    .. math::

       \mathbb{E^Q} [V(m_T)C_T] = -N S d (Q(m_{a.s}) v(m_t) + V_{I_{pa}} )

    where,

    .. math::

       V_{I_{pa}} = C_t I_{pa} v(m_{a.e}) \times \left \{ \begin{matrix}  \frac{1}{2} \left ( Q(m_{a.s}) - Q(m_{a.e}) \right ) & m_{a.s} >= m_{today} \\ \frac{\tilde{n}+r}{2\tilde{n}} \left ( 1 - Q(m_{a.e}) \right ) & m_{a.s} < m_{today} \\ \end{matrix} \right .

    For *analytic delta* purposes the :math:`\xi=-S`.

    .. rubric:: Examples

    .. ipython:: python
       :suppress:

       from rateslib.periods import CreditPremiumPeriod
       from datetime import datetime as dt

    .. ipython:: python

       cp = CreditPremiumPeriod(
           start=dt(2000, 3, 20),
           end=dt(2000, 6, 20),
           payment=dt(2000, 6, 20),
           frequency="Q",
           fixed_rate=1.00,
       )
       cp.cashflows()

    .. role:: red

    .. role:: green

    Parameters
    ----------
    .
        .. note::

           The following define generalised **settlement** parameters.

    currency: str, :green:`optional (set by 'defaults')`
        The physical *settlement currency* of the *Period*.
    notional: float, Dual, Dual2, Variable, :green:`optional (set by 'defaults')`
        The notional amount of the *Period* expressed in ``notional currency``.
    payment: datetime, :red:`required`
        The payment date of the *Period* cashflow.
    ex_dividend: datetime, :green:`optional (set as 'payment')`
        The ex-dividend date of the *Period*. Settlements occurring **after** this date
        are assumed to be non-receivable.

        .. note::

           The following parameters are scheduling **period** parameters

    start: datetime, :red:`required`
        The identified start date of the *Period*.
    end: datetime, :red:`required`
        The identified end date of the *Period*.
    frequency: Frequency, str, :red:`required`
        The :class:`~rateslib.scheduling.Frequency` associated with the *Period*.
    convention: Convention, str, :green:`optional` (set by 'defaults')
        The day count :class:`~rateslib.scheduling.Convention` associated with the *Period*.
    termination: datetime, :green:`optional`
        The termination date of an external :class:`~rateslib.scheduling.Schedule`.
    calendar: Calendar, :green:`optional`
         The calendar associated with the *Period*.
    stub: bool, str, :green:`optional (set as False)`
        Whether the *Period* is defined as a stub according to some external
        :class:`~rateslib.scheduling.Schedule`.
    adjuster: Adjuster, :green:`optional`
        The date :class:`~rateslib.scheduling.Adjuster` applied to unadjusted dates in the
        external :class:`~rateslib.scheduling.Schedule` to arrive at adjusted accrual dates.

        .. note::

           The following define **fixed rate** parameters.

    fixed_rate: float, Dual, Dual2, Variable, :green:`optional`
        The fixed rate to determine the *Period* cashflow.

        .. note::

           The following parameters define **credit specific** elements.

    premium_accrued: bool, :green:`optional (set by 'defaults')`
        Whether an accrued premium is paid on the event of mid-period credit default.

    """  # noqa: E501

    @property
    def credit_params(self) -> _CreditParams:
        """The :class:`~rateslib.periods.parameters._CreditParams` of the *Period*."""
        return self._credit_params

    @property
    def rate_params(self) -> _FixedRateParams:
        """The :class:`~rateslib.periods.parameters._FixedRateParams` of the *Period*."""
        return self._rate_params

    @property
    def period_params(self) -> _PeriodParams:
        """The :class:`~rateslib.periods.parameters._PeriodParams` of the *Period*."""
        return self._period_params

    def __init__(
        self,
        *,
        # currency args:
        payment: datetime,
        notional: DualTypes_ = NoInput(0),
        currency: str_ = NoInput(0),
        ex_dividend: datetime_ = NoInput(0),
        # period params
        start: datetime,
        end: datetime,
        frequency: Frequency | str,
        convention: str_ = NoInput(0),
        termination: datetime_ = NoInput(0),
        stub: bool = False,
        roll: RollDay | int | str_ = NoInput(0),
        calendar: CalInput = NoInput(0),
        adjuster: Adjuster | str_ = NoInput(0),
        # specific params
        fixed_rate: DualTypes_ = NoInput(0),
        premium_accrued: bool_ = NoInput(0),
    ) -> None:
        self._settlement_params = _SettlementParams(
            _currency=_drb(defaults.base_currency, currency).lower(),
            _notional_currency=_drb(defaults.base_currency, currency).lower(),
            _payment=payment,
            _notional=_drb(defaults.notional, notional),
            _ex_dividend=_drb(payment, ex_dividend),
        )
        self._rate_params = _FixedRateParams(
            _fixed_rate=fixed_rate,
        )
        self._credit_params = _CreditParams(
            _premium_accrued=_drb(defaults.cds_premium_accrued, premium_accrued),
        )
        self._period_params = _PeriodParams(
            _start=start,
            _end=end,
            _frequency=_get_frequency(frequency, roll, calendar),
            _calendar=get_calendar(calendar),
            _adjuster=NoInput(0) if isinstance(adjuster, NoInput) else _get_adjuster(adjuster),
            _stub=stub,
            _convention=_get_convention(_drb(defaults.convention, convention)),
            _termination=termination,
        )

    def immediate_local_npv(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: _FXVolOption_ = NoInput(0),
    ) -> DualTypes:
        rate_curve_, disc_curve_ = _validate_credit_curves(rate_curve, disc_curve).unwrap()

        cf = self.cashflow()
        return cf * self._probability_adjusted_df(rate_curve_, disc_curve_)

    def try_immediate_local_analytic_delta(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        fx: FXRevised_ = NoInput(0),
        fx_vol: _FXVolOption_ = NoInput(0),
    ) -> Result[DualTypes]:
        c = 0.0001 * self.period_params.dcf * self.settlement_params.notional

        c_res = _validate_credit_curves(rate_curve, disc_curve)
        if isinstance(c_res, Err):
            return c_res
        else:
            rate_curve_, disc_curve_ = c_res.unwrap()

        return Ok(c * self._probability_adjusted_df(rate_curve_, disc_curve_))

    def cashflow(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: _FXVolOption_ = NoInput(0),
    ) -> DualTypes:
        if isinstance(self.rate_params.fixed_rate, NoInput):
            raise ValueError(err.VE_NEEDS_FIXEDRATE)
        return (
            -self.rate_params.fixed_rate
            * 0.01
            * self.period_params.dcf
            * self.settlement_params.notional
        )

    def try_cashflow(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: _FXVolOption_ = NoInput(0),
    ) -> Result[DualTypes]:
        r"""
        Replicate :meth:`~rateslib.periods.protocols._WithNPVStatic.cashflow`
        with lazy exception handling.

        Returns
        -------
        Result[float, Dual, Dual2, Variable]
        """
        try:
            v = self.cashflow(
                rate_curve=rate_curve,
                index_curve=index_curve,
                disc_curve=disc_curve,
                fx_vol=fx_vol,
                fx=fx,
            )
        except Exception as e:
            return Err(e)
        else:
            return Ok(v)

    def _probability_adjusted_df(self, rate_curve: _BaseCurve, disc_curve: _BaseCurve) -> DualTypes:
        v_payment = disc_curve[self.settlement_params.payment]
        q_end = rate_curve[self.period_params.end]
        if self.credit_params.premium_accrued:
            v_end = disc_curve[self.period_params.end]
            n = _dual_float((self.period_params.end - self.period_params.start).days)

            if self.period_params.start < disc_curve.nodes.initial:
                # then mid-period valuation
                r: float = _dual_float((disc_curve.nodes.initial - self.period_params.start).days)
                q_start: DualTypes = 1.0
                _v_start: DualTypes = 1.0
            else:
                r = 0.0
                q_start = rate_curve[self.period_params.start]
                _v_start = disc_curve[self.period_params.start]

            # method 1:
            accrued_: DualTypes = 0.5 * (1 + r / n)
            accrued_ *= q_start - q_end
            accrued_ *= v_end

            # # method 4 EXACT
            # _ = 0.0
            # for i in range(1, int(s)):
            #     m_i, m_i2 = m_today + timedelta(days=i-1), m_today + timedelta(days=i)
            #     _ += (
            #     (i + r) / n * disc_curve[m_today + timedelta(days=i)] * (curve[m_i] - curve[m_i2])
            #     )
        else:
            accrued_ = 0.0
        return q_end * v_payment + accrued_

    def try_accrued(self, settlement: datetime) -> Result[DualTypes]:
        """
        Calculate the amount of premium accrued until a specific date within the *Period*, with
        lazy error raising.

        Parameters
        ----------
        settlement: datetime
            The date against which accrued is measured.

        Returns
        -------
        Result[float]
        """
        if isinstance(self.rate_params.fixed_rate, NoInput):
            return Err(ValueError(err.VE_NEEDS_FIXEDRATE))

        c = (
            -self.rate_params.fixed_rate
            * 0.01
            * self.period_params.dcf
            * self.settlement_params.notional
        )
        start, end = self.period_params.start, self.period_params.end
        if settlement <= start or settlement >= end:
            return Ok(0.0)
        return Ok(c * (settlement - start).days / (end - start).days)

    def accrued(self, settlement: datetime) -> DualTypes:
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
        return self.try_accrued(settlement).unwrap()


class CreditProtectionPeriod(_BasePeriod):
    r"""
    A *Period* defined by a credit event and contingent notional payment.

    The immediate expected valuation of the *Period* cashflow is defined as;

    .. math::

       \mathbb{E^Q}[V(m_T)C_T] = -N(1-RR) \int_{max(m_{a.s}, m_{today})}^{m_{a.e}} w_{loc:col}(m_s) Q(m_s) \lambda(s) ds

    where the integral is numerically determined.

    There is no *analytical delta* for this *Period* type and hence :math:`\xi` is not defined.

    .. rubric:: Examples

    .. ipython:: python
       :suppress:

       from rateslib.periods import CreditProtectionPeriod
       from datetime import datetime as dt

    .. ipython:: python

       cp = CreditProtectionPeriod(
           start=dt(2000, 3, 20),
           end=dt(2000, 6, 20),
           payment=dt(2000, 6, 20),
           frequency="Q",
       )
       cp.cashflows()

    .. role:: red

    .. role:: green

    Parameters
    ----------
    .
        .. note::

           The following define generalised **settlement** parameters.

    currency: str, :green:`optional (set by 'defaults')`
        The physical *settlement currency* of the *Period*.
    notional: float, Dual, Dual2, Variable, :green:`optional (set by 'defaults')`
        The notional amount of the *Period* expressed in ``notional currency``.
    payment: datetime, :red:`required`
        The payment date of the *Period* cashflow.
    ex_dividend: datetime, :green:`optional (set as 'payment')`
        The ex-dividend date of the *Period*. Settlements occurring **after** this date
        are assumed to be non-receivable.

        .. note::

           The following parameters are scheduling **period** parameters

    start: datetime, :red:`required`
        The identified start date of the *Period*.
    end: datetime, :red:`required`
        The identified end date of the *Period*.
    frequency: Frequency, str, :red:`required`
        The :class:`~rateslib.scheduling.Frequency` associated with the *Period*.
    convention: Convention, str, :green:`optional` (set by 'defaults')
        The day count :class:`~rateslib.scheduling.Convention` associated with the *Period*.
    termination: datetime, :green:`optional`
        The termination date of an external :class:`~rateslib.scheduling.Schedule`.
    calendar: Calendar, :green:`optional`
         The calendar associated with the *Period*.
    stub: bool, str, :green:`optional (set as False)`
        Whether the *Period* is defined as a stub according to some external
        :class:`~rateslib.scheduling.Schedule`.
    adjuster: Adjuster, :green:`optional`
        The date :class:`~rateslib.scheduling.Adjuster` applied to unadjusted dates in the
        external :class:`~rateslib.scheduling.Schedule` to arrive at adjusted accrual dates.

    """  # noqa: E501

    @property
    def credit_params(self) -> _CreditParams:
        """The :class:`~rateslib.periods.parameters._CreditParams` of the *Period*."""
        return self._credit_params

    @property
    def period_params(self) -> _PeriodParams:
        """The :class:`~rateslib.periods.parameters._PeriodParams` of the *Period*."""
        return self._period_params

    def __init__(
        self,
        *,
        # currency args:
        payment: datetime,
        notional: DualTypes_ = NoInput(0),
        currency: str_ = NoInput(0),
        ex_dividend: datetime_ = NoInput(0),
        # period params
        start: datetime,
        end: datetime,
        frequency: Frequency | str,
        convention: str_ = NoInput(0),
        termination: datetime_ = NoInput(0),
        stub: bool = False,
        roll: RollDay | int | str_ = NoInput(0),
        calendar: CalInput = NoInput(0),
        adjuster: Adjuster | str_ = NoInput(0),
    ) -> None:
        self._settlement_params = _SettlementParams(
            _currency=_drb(defaults.base_currency, currency).lower(),
            _notional_currency=_drb(defaults.base_currency, currency).lower(),
            _payment=payment,
            _notional=_drb(defaults.notional, notional),
            _ex_dividend=_drb(payment, ex_dividend),
        )
        self._credit_params = _CreditParams(
            _premium_accrued=True
        )  # arg irrelevant for Period type.
        self._period_params = _PeriodParams(
            _start=start,
            _end=end,
            _frequency=_get_frequency(frequency, roll, calendar),
            _calendar=get_calendar(calendar),
            _adjuster=NoInput(0) if isinstance(adjuster, NoInput) else _get_adjuster(adjuster),
            _stub=stub,
            _convention=_get_convention(_drb(defaults.convention, convention)),
            _termination=termination,
        )

    def cashflow(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: _FXVolOption_ = NoInput(0),
    ) -> DualTypes:
        rate_curve_ = _try_validate_base_curve(rate_curve).unwrap()
        return -self.settlement_params.notional * (1 - rate_curve_.meta.credit_recovery_rate)

    def try_cashflow(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: _FXVolOption_ = NoInput(0),
    ) -> Result[DualTypes]:
        r"""
        Replicate :meth:`~rateslib.periods.protocols._WithNPVStatic.cashflow`
        with lazy exception handling.

        Returns
        -------
        Result[float, Dual, Dual2, Variable]
        """
        try:
            v = self.cashflow(
                rate_curve=rate_curve,
                index_curve=index_curve,
                disc_curve=disc_curve,
                fx_vol=fx_vol,
                fx=fx,
            )
        except Exception as e:
            return Err(e)
        else:
            return Ok(v)

    def immediate_local_npv(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: _FXVolOption_ = NoInput(0),
    ) -> DualTypes:
        rate_curve_, disc_curve_ = _validate_credit_curves(rate_curve, disc_curve).unwrap()

        discretization = rate_curve_.meta.credit_discretization

        if self.period_params.start < rate_curve_.nodes.initial:
            s2 = rate_curve_.nodes.initial
        else:
            s2 = self.period_params.start

        value: DualTypes = 0.0
        q2: DualTypes = rate_curve_[s2]
        v2: DualTypes = disc_curve_[s2]
        while s2 < self.period_params.end:
            q1, v1 = q2, v2
            s2 = s2 + timedelta(days=discretization)
            if s2 > self.period_params.end:
                s2 = self.period_params.end
            q2, v2 = rate_curve_[s2], disc_curve_[s2]
            value += 0.5 * (v1 + v2) * (q1 - q2)
            # value += v2 * (q1 - q2)

        # curves are pre-validated so will not error
        cf = self.cashflow(rate_curve=rate_curve)
        return value * cf

    def try_immediate_local_analytic_delta(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        fx: FXRevised_ = NoInput(0),
        fx_vol: _FXVolOption_ = NoInput(0),
    ) -> Result[DualTypes]:
        return Ok(0.0)

    def analytic_rec_risk(
        self,
        rate_curve: _BaseCurve_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
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
        c_res = _validate_credit_curves(rate_curve, disc_curve)
        if isinstance(c_res, Err):
            c_res.unwrap()
        else:
            rate_curve_, disc_curve_ = c_res.unwrap()

        haz_curve = rate_curve_.copy()
        haz_curve._meta = replace(  # type: ignore[misc]
            rate_curve_.meta,
            _credit_recovery_rate=Variable(
                _dual_float(rate_curve_.meta.credit_recovery_rate), ["__rec_rate__"], []
            ),
        )
        pv: DualTypes = self.npv(  # type: ignore[assignment]
            rate_curve=haz_curve,
            disc_curve=disc_curve_,
            fx=fx,  # type: ignore[arg-type]
            base=base,
            local=False,
        )
        _: float = _dual_float(gradient(pv, ["__rec_rate__"], order=1)[0])
        return _ * 0.01
