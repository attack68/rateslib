from __future__ import annotations

from typing import TYPE_CHECKING

import rateslib.errors as err
from rateslib import defaults
from rateslib.curves.curves import _BaseCurve
from rateslib.dual.utils import _dual_float
from rateslib.enums.generics import Err, NoInput, Ok, Result, _drb
from rateslib.periods.components.parameters import (
    _CreditParams,
    _FixedRateParams,
    _PeriodParams,
    _SettlementParams,
)
from rateslib.periods.components.protocols import _WithAnalyticDelta, _WithNPVCashflows
from rateslib.scheduling import Frequency, get_calendar
from rateslib.scheduling.adjuster import _get_adjuster
from rateslib.scheduling.convention import _get_convention
from rateslib.scheduling.frequency import _get_frequency

if TYPE_CHECKING:
    from rateslib.typing import (
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

        cf_res = self.try_cashflow()
        if isinstance(cf_res, Err):
            return cf_res

        return Ok(cf_res.unwrap() * self._probability_adjusted_df(rate_curve_, disc_curve_))

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
        Calculate the amount of premium accrued until a specific date within the *Period*.

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


def _validate_credit_curves(
    rate_curve: CurveOption_, disc_curve: CurveOption_
) -> Result[tuple[_BaseCurve, _BaseCurve]]:
    # used by Credit type Periods to narrow inputs
    if not isinstance(rate_curve, _BaseCurve):
        return Err(
            TypeError(
                "`curves` have not been supplied correctly.\n"
                "`curve`for a CreditPremiumPeriod must be supplied as a Curve type."
            )
        )
    if not isinstance(disc_curve, _BaseCurve):
        return Err(
            TypeError(
                "`curves` have not been supplied correctly.\n"
                "`disc_curve` for a CreditPremiumPeriod must be supplied as a Curve type."
            )
        )
    return Ok((rate_curve, disc_curve))
