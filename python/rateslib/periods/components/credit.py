from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from typing import TYPE_CHECKING

import rateslib.errors as err
from rateslib import defaults
from rateslib.dual import Variable, gradient
from rateslib.dual.utils import _dual_float
from rateslib.enums.generics import Err, NoInput, Ok, Result, _drb
from rateslib.periods.components.parameters import (
    _CreditParams,
    _FixedRateParams,
    _PeriodParams,
    _SettlementParams,
)
from rateslib.periods.components.protocols import _WithAnalyticDelta, _WithNPVCashflows
from rateslib.periods.components.utils import _validate_credit_curve, _validate_credit_curves
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
        FXVolOption_,
        RollDay,
        _BaseCurve,
        _BaseCurve_,
        bool_,
        datetime,
        datetime_,
        str_,
    )


class CreditPremiumPeriod(_WithNPVCashflows, _WithAnalyticDelta):
    r"""
    A *Period* defined by a fixed interest rate and contingent credit event.

    The immediate expected valuation of the *Period* cashflow is defined as;

    .. math::

       \mathbb{E^Q} [V(m_T)C_T] = -N S d (Q(m_{a.s}) v(m_t) + V_{I_{pa}} )

    where,

    .. math::

       V_{I_{pa}} = C_t I_{pa} v(m_{a.e}) \times \left \{ \begin{matrix}  \frac{1}{2} \left ( Q(m_{a.s}) - Q(m_{a.e}) \right ) & m_{a.s} >= m_{today} \\ \frac{\tilde{n}+r}{2\tilde{n}} \left ( 1 - Q(m_{a.e}) \right ) & m_{a.s} < m_{today} \\ \end{matrix} \right .


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
        self.settlement_params = _SettlementParams(
            _currency=_drb(defaults.base_currency, currency).lower(),
            _notional_currency=_drb(defaults.base_currency, currency).lower(),
            _payment=payment,
            _notional=_drb(defaults.notional, notional),
            _ex_dividend=_drb(payment, ex_dividend),
        )
        self.rate_params = _FixedRateParams(
            _fixed_rate=fixed_rate,
        )
        self.credit_params = _CreditParams(
            _premium_accrued=_drb(defaults.cds_premium_accrued, premium_accrued),
        )
        self.non_deliverable_params = None
        self.index_params = None
        self.period_params = _PeriodParams(
            _start=start,
            _end=end,
            _frequency=_get_frequency(frequency, roll, calendar),
            _calendar=get_calendar(calendar),
            _adjuster=NoInput(0) if isinstance(adjuster, NoInput) else _get_adjuster(adjuster),
            _stub=stub,
            _convention=_get_convention(_drb(defaults.convention, convention)),
            _termination=termination,
        )

    def try_local_npv(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: FXVolOption_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> Result[DualTypes]:
        c_res = _validate_credit_curves(rate_curve, disc_curve)
        if isinstance(c_res, Err):
            return c_res
        else:
            rate_curve_, disc_curve_ = c_res.unwrap()

        cf_res = self.try_cashflow()
        if isinstance(cf_res, Err):
            return cf_res

        pv0 = cf_res.unwrap() * self._probability_adjusted_df(rate_curve_, disc_curve_)
        return Ok(
            self._screen_ex_div_and_forward(
                local_npv=pv0, disc_curve=disc_curve_, settlement=settlement, forward=forward
            )
        )

    def try_analytic_delta(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        fx: FXRevised_ = NoInput(0),
        base: str_ = NoInput(0),
    ) -> Result[DualTypes]:
        c = 0.0001 * self.period_params.dcf * self.settlement_params.notional

        c_res = _validate_credit_curves(rate_curve, disc_curve)
        if isinstance(c_res, Err):
            return c_res
        else:
            rate_curve_, disc_curve_ = c_res.unwrap()

        return Ok(c * self._probability_adjusted_df(rate_curve_, disc_curve_))

    def try_cashflow(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: FXVolOption_ = NoInput(0),
    ) -> Result[DualTypes]:
        if isinstance(self.rate_params.fixed_rate, NoInput):
            return Err(ValueError(err.VE_NEEDS_FIXEDRATE))
        return Ok(
            -self.rate_params.fixed_rate
            * 0.01
            * self.period_params.dcf
            * self.settlement_params.notional
        )

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


class CreditProtectionPeriod(_WithNPVCashflows, _WithAnalyticDelta):
    """
    A *Period* defined by a credit event and contingent notional payment.

    The immediate expected valuation of the *Period* cashflow is defined as;

    [TODO: NEEDS INPUT]

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
        self.settlement_params = _SettlementParams(
            _currency=_drb(defaults.base_currency, currency).lower(),
            _notional_currency=_drb(defaults.base_currency, currency).lower(),
            _payment=payment,
            _notional=_drb(defaults.notional, notional),
            _ex_dividend=_drb(payment, ex_dividend),
        )
        self.credit_params = _CreditParams(_premium_accrued=True)  # arg irrelevant for Period type.
        self.rate_params = None
        self.non_deliverable_params = None
        self.index_params = None
        self.period_params = _PeriodParams(
            _start=start,
            _end=end,
            _frequency=_get_frequency(frequency, roll, calendar),
            _calendar=get_calendar(calendar),
            _adjuster=NoInput(0) if isinstance(adjuster, NoInput) else _get_adjuster(adjuster),
            _stub=stub,
            _convention=_get_convention(_drb(defaults.convention, convention)),
            _termination=termination,
        )

    def try_cashflow(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: FXVolOption_ = NoInput(0),
    ) -> Result[DualTypes]:
        rc_res = _validate_credit_curve(rate_curve)
        if isinstance(rc_res, Err):
            return rc_res

        return Ok(
            -self.settlement_params.notional * (1 - rc_res.unwrap().meta.credit_recovery_rate)
        )

    def try_local_npv(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: FXVolOption_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> Result[DualTypes]:
        r"""
        Calculate the NPV of the *Period* in local settlement currency.

        Parameters
        ----------
        rate_curve: _BaseCurve or dict of such indexed by string tenor, optional
            Used to forecast floating period rates, if necessary.
        index_curve: _BaseCurve, optional
            Used to forecast index values for indexation, if necessary.
        disc_curve: _BaseCurve, optional
            Used to discount cashflows.
        fx: FXForwards, optional
            The :class:`~rateslib.fx.FXForward` object used for forecasting the
            ``fx_fixing`` for deliverable cashflows, if necessary. Or, an
            class:`~rateslib.fx.FXRates` object purely for immediate currency conversion.
        fx_vol: FXDeltaVolSmile, FXSabrSmile, FXDeltaVolSurface, FXSabrSurface, optional
            The FX volatility *Smile* or *Surface* object used for determining Black calendar
            day implied volatility values.
        settlement: datetime, optional
            The assumed settlement date of the *PV* determination. Used only to evaluate
            *ex-dividend* status.
        forward: datetime, optional
            The future date to project the *PV* to using the ``disc_curve``.

        Returns
        -------
        Result of float, Dual, Dual2, Variable

        Notes
        -----

        Is a generalised function for determining the ex-dividend adjusted, forward projected
        *NPV* of any *Period's* modelled cashflow, expressed in local *settlement currency* units.

        .. math::

           P(m_s, m_f) = \mathbb{I}(m_s) \frac{1}{v(m_f)} \mathbb{E^Q} [v(m_T) C(m_T) ],  \qquad \; \mathbb{I}(m_s) = \left \{ \begin{matrix} 0 & m_s > m_{ex} \\ 1 & m_s \leq m_{ex} \end{matrix} \right .

        for forward, :math:`m_f`, settlement, :math:`m_s`, and ex-dividend, :math:`m_{ex}`.
        """  # noqa: E501
        c_res = _validate_credit_curves(rate_curve, disc_curve)
        if isinstance(c_res, Err):
            return c_res
        else:
            rate_curve_, disc_curve_ = c_res.unwrap()

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
        cf = self.try_cashflow(rate_curve=rate_curve).unwrap()
        return Ok(
            self._screen_ex_div_and_forward(
                local_npv=value * cf,
                disc_curve=disc_curve_,
                forward=forward,
                settlement=settlement,
            )
        )

    def try_analytic_delta(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        fx: FXRevised_ = NoInput(0),
        base: str_ = NoInput(0),
    ) -> Result[DualTypes]:
        """
        Calculate the analytic rate delta of a *Period* expressed in a base currency.

        Parameters
        ----------
        rate_curve: _BaseCurve or dict of such indexed by string tenor, optional
            Used to forecast floating period rates, if necessary.
        index_curve: _BaseCurve, optional
            Used to forecast index values for indexation, if necessary.
        disc_curve: _BaseCurve, optional
            Used to discount cashflows.
        fx: FXForwards, optional
            The :class:`~rateslib.fx.FXForward` object used for forecasting the
            ``fx_fixing`` for deliverable cashflows, if necessary. Or, an
            class:`~rateslib.fx.FXRates` object purely for immediate currency conversion.

        Returns
        -------
        Result[DualTypes]
        """
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
