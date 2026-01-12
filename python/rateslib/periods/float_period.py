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

from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from pandas import DataFrame, Index, MultiIndex, Series, concat, merge

import rateslib.errors as err
from rateslib import defaults
from rateslib.curves import _BaseCurve
from rateslib.curves.utils import average_rate
from rateslib.data.fixings import (
    FloatRateSeries,
    _leg_fixings_to_list,
    _maybe_get_fx_index,
    _RFRRate,
)
from rateslib.data.loader import _find_neighbouring_tenors
from rateslib.dual import Variable, gradient
from rateslib.dual.utils import _dual_float
from rateslib.enums.generics import Err, NoInput, Ok, _drb
from rateslib.enums.parameters import FloatFixingMethod, IndexMethod, SpreadCompoundMethod
from rateslib.periods.float_rate import (
    try_rate_value,
)
from rateslib.periods.parameters import (
    _init_FloatRateParams,
    _init_or_none_IndexParams,
    _init_or_none_NonDeliverableParams,
    _init_SettlementParams_with_fx_pair,
    _PeriodParams,
)
from rateslib.periods.protocols import _BasePeriodStatic
from rateslib.periods.utils import _get_rfr_curve_from_dict
from rateslib.scheduling import Adjuster, Frequency, dcf, get_calendar
from rateslib.scheduling.adjuster import _get_adjuster
from rateslib.scheduling.convention import _get_convention
from rateslib.scheduling.frequency import _get_frequency, _get_tenor_from_frequency

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        Any,
        Arr1dObj,
        CalInput,
        Convention,
        CurveOption_,
        DualTypes,
        DualTypes_,
        Frequency,
        FXForwards_,
        FXIndex,
        Result,
        RFRFixing,
        RollDay,
        Schedule,
        Series,
        _BaseCurve_,
        _FloatRateParams,
        _FXVolOption_,
        bool_,
        datetime,
        datetime_,
        int_,
        str_,
    )


class FloatPeriod(_BasePeriodStatic):
    r"""
    A *Period* defined by a floating interest rate.

    The expected unindexed reference cashflow under the risk neutral distribution is defined as,

    .. math::

       \mathbb{E^Q} [\bar{C}_t] = -N d r(\mathbf{C}, z, R_i)

    For *analytic delta* purposes the :math:`\xi=-z`.

    .. role:: red

    .. role:: green

    .. rubric:: Examples

    .. ipython:: python
       :suppress:

       from rateslib.periods import FloatPeriod
       from rateslib import fixings
       from datetime import datetime as dt
       from pandas import Series

    .. ipython:: python

       fixings.add("MY_RATE_INDEX_6M", Series(index=[dt(2000, 1, 1)], data=[2.66]))
       period = FloatPeriod(
           start=dt(2000, 1, 1),
           end=dt(2000, 7, 1),
           payment=dt(2000, 7, 1),
           notional=1e6,
           convention="Act360",
           frequency="S",
           fixing_method="ibor",
           method_param=0,
           rate_fixings="MY_RATE_INDEX"
       )
       period.cashflows()

    .. ipython:: python
       :suppress:

       fixings.pop("MY_RATE_INDEX_6M")

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

           The following define **floating rate** parameters.

    fixing_method: FloatFixingMethod, str, :green:`optional (set by 'defaults')`
        The :class:`~rateslib.enums.parameters.FloatFixingMethod` describing the determination
        of the floating rate for the period.
    method_param: int, :green:`optional (set by 'defaults')`
        A specific parameter that is used by the specific ``fixing_method``.
    fixing_frequency: Frequency, str, :green:`optional (set by 'frequency' or '1B')`
        The :class:`~rateslib.scheduling.Frequency` as a component of the
        :class:`~rateslib.data.fixings.FloatRateIndex`. If not given is assumed to match the
        frequency of the period for an IBOR type ``fixing_method`` or '1B' if RFR type.
    fixing_series: FloatRateSeries, str, :green:`optional (implied by other parameters)`
        The :class:`~rateslib.data.fixings.FloatRateSeries` as a component of the
        :class:`~rateslib.data.fixings.FloatRateIndex`. If not given inherits attributes given
        such as the ``calendar``, ``convention``, ``method_param`` etc.
    float_spread: float, Dual, Dual2, Variable, :green:`optional (set as 0.0)`
        The amount (in bps) added to the rate in the period rate determination. If not given is
        set to zero.
    spread_compound_method: SpreadCompoundMethod, str, :green:`optional (set by 'defaults')`
        The :class:`~rateslib.enums.parameters.SpreadCompoundMethod` used in the calculation
        of the period rate when combining a ``float_spread``. Used **only** with RFR type
        ``fixing_method``.
    rate_fixings: float, Dual, Dual2, Variable, Series, str, :green:`optional`
        The value of the rate fixing. If a scalar, is used directly. If a string identifier, links
        to the central ``fixings`` object and data loader.

        .. note::

           The following parameters define **non-deliverability**. If the *Period* is directly
           deliverable do not supply these parameters.

    pair: str, :green:`optional`
        The currency pair of the :class:`~rateslib.data.fixings.FXFixing` that determines
        settlement. The *reference currency* is implied from ``pair``. Must include ``currency``.
    fx_fixings: float, Dual, Dual2, Variable, Series, str, :green:`optional`
        The value of the :class:`~rateslib.data.fixings.FXFixing`. If a scalar is used directly.
        If a string identifier will link to the central ``fixings`` object and data loader.
    delivery: datetime, :green:`optional (set as 'payment')`
        The settlement delivery date of the :class:`~rateslib.data.fixings.FXFixing`.

        .. note::

           The following parameters define **indexation**. The *Period* will be considered
           indexed if any of ``index_method``, ``index_lag``, ``index_base``, ``index_fixings``
           are given.

    index_method : IndexMethod, str, :green:`optional (set by 'defaults')`
        The interpolation method, or otherwise, to determine index values from reference dates.
    index_lag: int, :green:`optional (set by 'defaults')`
        The indexation lag, in months, applied to the determination of index values.
    index_base: float, Dual, Dual2, Variable, :green:`optional`
        The specific value set of the base index value.
        If not given and ``index_fixings`` is a str fixings identifier that will be
        used to determine the base index value.
    index_fixings: float, Dual, Dual2, Variable, Series, str, :green:`optional`
        The index value for the reference date.
        If a scalar value this is used directly. If a string identifier will link to the
        central ``fixings`` object and data loader.
    index_base_date: datetime, :green:`optional`
        The reference date for determining the base index value. Not required if ``_index_base``
        value is given directly.
    index_reference_date: datetime, :green:`optional (set as 'end')`
        The reference date for determining the index value. Not required if ``_index_fixings``
        is given as a scalar value.
    index_only: bool, :green:`optional (set as False)`
        A flag which determines non-payment of notional on supported *Periods*.


    ..  Examples
        --------

        A typical RFR type :class:`~rateslib.periods.FloatPeriod`.

        .. ipython:: python
           :supress:

           from rateslib.periods import FloatPeriod
           from rateslib.data.fixings import FloatRateIndex
           from datetime import datetime as dt

        .. ipython:: python

           period = FloatPeriod(
               start=dt(2025, 9, 22),
               end=dt(2025, 10, 20),
               payment=dt(2025, 10, 22),
               frequency="1M",
           )

        A typical IBOR tenor type :class:`~rateslib.periods.FloatPeriod`.

        .. ipython:: python

           period = FloatPeriod(
               start=dt(2025, 9, 22),
               end=dt(2025, 10, 22),
               payment=dt(2025, 10, 22),
               frequency="1M",
               currency="eur",
               fixing_method="IBOR",
               fixing_series="eur_IBOR",
           )

    """

    @property
    def rate_params(self) -> _FloatRateParams:
        """The :class:`~rateslib.periods.parameters._FloatRateParams` of the *Period*."""
        return self._rate_params

    @property
    def period_params(self) -> _PeriodParams:
        return self._period_params

    def __init__(
        self,
        *,
        float_spread: DualTypes_ = NoInput(0),
        rate_fixings: DualTypes | Series[DualTypes] | str_ = NoInput(0),  # type: ignore[type-var]
        fixing_method: FloatFixingMethod | str_ = NoInput(0),
        method_param: int_ = NoInput(0),
        spread_compound_method: SpreadCompoundMethod | str_ = NoInput(0),
        fixing_frequency: Frequency | str_ = NoInput(0),
        fixing_series: FloatRateSeries | str_ = NoInput(0),
        # currency args:
        payment: datetime,
        notional: DualTypes_ = NoInput(0),
        currency: str_ = NoInput(0),
        ex_dividend: datetime_ = NoInput(0),
        # period params
        start: datetime,
        end: datetime,
        frequency: Frequency | str,
        convention: Convention | str_ = NoInput(0),
        termination: datetime_ = NoInput(0),
        stub: bool = False,
        roll: RollDay | int | str_ = NoInput(0),
        calendar: CalInput = NoInput(0),
        adjuster: Adjuster | str_ = NoInput(0),
        # non-deliverable args:
        pair: FXIndex | str_ = NoInput(0),
        fx_fixings: DualTypes | Series[DualTypes] | str_ = NoInput(0),  # type: ignore[type-var]
        delivery: datetime_ = NoInput(0),
        # index-args:
        index_base: DualTypes_ = NoInput(0),
        index_lag: int_ = NoInput(0),
        index_method: IndexMethod | str_ = NoInput(0),
        index_fixings: DualTypes | Series[DualTypes] | str_ = NoInput(0),  # type: ignore[type-var]
        index_only: bool_ = NoInput(0),
        index_base_date: datetime_ = NoInput(0),
        index_reference_date: datetime_ = NoInput(0),
    ) -> None:
        self._settlement_params = _init_SettlementParams_with_fx_pair(
            _currency=_drb(defaults.base_currency, currency).lower(),
            _payment=payment,
            _notional=_drb(defaults.notional, notional),
            _ex_dividend=_drb(payment, ex_dividend),
            _fx_pair=_maybe_get_fx_index(pair),
        )
        self._non_deliverable_params = _init_or_none_NonDeliverableParams(
            _currency=self.settlement_params.currency,
            _fx_index=pair,
            _delivery=_drb(self.settlement_params.payment, delivery),
            _fx_fixings=fx_fixings,
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
        self._index_params = _init_or_none_IndexParams(
            _index_base=index_base,
            _index_lag=index_lag,
            _index_method=index_method,
            _index_fixings=index_fixings,
            _index_only=index_only,
            _index_base_date=index_base_date,
            _index_reference_date=_drb(self.period_params.end, index_reference_date),
        )
        self._rate_params = _init_FloatRateParams(
            _method_param=method_param,
            _float_spread=float_spread,
            _spread_compound_method=spread_compound_method,
            _fixing_method=fixing_method,
            _fixing_series=fixing_series,
            _fixing_frequency=fixing_frequency,
            _rate_fixings=rate_fixings,
            _accrual_start=self.period_params.start,
            _accrual_end=self.period_params.end,
            _period_calendar=self.period_params.calendar,
            _period_convention=self.period_params.convention,
            _period_adjuster=self.period_params.adjuster,
            _period_frequency=self.period_params.frequency,
            _period_stub=self.period_params.stub,
        )

    def unindexed_reference_cashflow(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        **kwargs: Any,
    ) -> DualTypes:
        r = self.rate(rate_curve)
        return -self.settlement_params.notional * r * 0.01 * self.period_params.dcf

    def try_unindexed_reference_cashflow_analytic_delta(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
    ) -> Result[DualTypes]:
        """
        Calculate the analytic rate delta of a *Period* expressed in ``reference_currency``
        without indexation.

        Parameters
        ----------
        rate_curve: _BaseCurve or dict of such indexed by string tenor, optional
            Used to forecast floating period rates, if necessary.
        disc_curve: _BaseCurve, optional
            Used to discount cashflows.

        Returns
        -------
        float, Dual, Dual2, Variable
        """
        if (
            self.rate_params.spread_compound_method == SpreadCompoundMethod.NoneSimple
            or self.rate_params.float_spread == 0
        ):
            # then analytic_delta is not impacted by float_spread compounding
            dr_dz: float = 1.0
        else:
            _ = self.rate_params.float_spread
            self.rate_params.float_spread = Variable(_dual_float(_), ["z_float_spread"])
            rate: Result[DualTypes] = self.try_rate(rate_curve)
            if rate.is_err:
                return rate
            dr_dz = gradient(rate.unwrap(), ["z_float_spread"])[0] * 100
            self.rate_params.float_spread = _

        return Ok(self.settlement_params.notional * 0.0001 * dr_dz * self.period_params.dcf)

    def try_unindexed_reference_cashflow_analytic_rate_fixings(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: _FXVolOption_ = NoInput(0),
    ) -> Result[DataFrame]:
        if isinstance(rate_curve, NoInput):
            return Err(ValueError(err.VE_NEEDS_RATE_CURVE))

        if self.rate_params.fixing_method == FloatFixingMethod.IBOR:
            return _UnindexedReferenceCashflowFixingsSensitivity._ibor(
                self=self, rate_curve=rate_curve
            )
        else:  # is RFR
            if isinstance(rate_curve, dict):
                rate_curve_: _BaseCurve = _get_rfr_curve_from_dict(rate_curve)
            else:
                rate_curve_ = rate_curve
            return _UnindexedReferenceCashflowFixingsSensitivity._rfr(
                self=self, rate_curve=rate_curve_
            )

    # def try_unindexed_reference_fixings_exposure(
    #     self,
    #     rate_curve: CurveOption_ = NoInput(0),
    #     disc_curve: _BaseCurve_ = NoInput(0),
    #     right: datetime_ = NoInput(0),
    # ) -> Result[DataFrame]:
    #     if self.rate_params.fixing_method == FloatFixingMethod.IBOR:
    #         return _FixingsExposureCalculator.ibor(
    #             p=self,
    #             rate_curve=rate_curve,
    #             disc_curve=disc_curve,
    #             right=right,
    #         )
    #     else:
    #         if isinstance(rate_curve, dict):
    #             rate_curve_: _BaseCurve_ = _get_rfr_curve_from_dict(rate_curve)
    #         else:
    #             rate_curve_ = rate_curve
    #         return _FixingsExposureCalculator.rfr(
    #             p=self,
    #             rate_curve=_validate_obj_not_no_input(rate_curve_, "rate_curve"),
    #             disc_curve=disc_curve,
    #             right=right,
    #         )

    def try_rate(self, rate_curve: CurveOption_) -> Result[DualTypes]:
        """
        Calculate the period rate, with lazy e

        Parameters
        ----------
        rate_curve: XXX
            The curve used to forecast rates, if the period has no fixing.

        Returns
        -------
        float, Dual, Dual2, Variable
        """
        rate_fixing = self.rate_params.rate_fixing.value
        if isinstance(rate_fixing, NoInput):
            return try_rate_value(
                start=self.rate_params.accrual_start,
                end=self.rate_params.accrual_end,
                rate_curve=NoInput(0) if rate_curve is None else rate_curve,
                rate_fixings=self.rate_params.rate_fixing.identifier,
                fixing_method=self.rate_params.fixing_method,
                method_param=self.rate_params.method_param,
                spread_compound_method=self.rate_params.spread_compound_method,
                float_spread=self.rate_params.float_spread,
                stub=self.period_params.stub,
                frequency=self.rate_params.fixing_frequency,
                rate_series=self.rate_params.fixing_series,
            )
        else:
            # the fixing value is a scalar so a Curve should not be required for this calculation
            return try_rate_value(
                start=self.rate_params.accrual_start,
                end=self.rate_params.accrual_end,
                rate_curve=NoInput(0),
                rate_fixings=rate_fixing,
                fixing_method=self.rate_params.fixing_method,
                method_param=self.rate_params.method_param,
                spread_compound_method=self.rate_params.spread_compound_method,
                float_spread=self.rate_params.float_spread,
                stub=self.period_params.stub,
                frequency=self.rate_params.fixing_frequency,
                rate_series=self.rate_params.fixing_series,
            )

    def rate(self, rate_curve: CurveOption_) -> DualTypes:
        """
        Calculate the period rate.

        Parameters
        ----------
        rate_curve: XXX
            The curve used to forecast rates, if the period has no fixing.

        Returns
        -------
        float, Dual, Dual2, Variable
        """
        return self.try_rate(rate_curve).unwrap()


class ZeroFloatPeriod(_BasePeriodStatic):
    r"""
    A *Period* defined by compounded floating rate *Periods*.

    The expected unindexed reference cashflow under the risk neutral distribution is defined as,

    .. math::

      \mathbb{E^Q}[\bar{C}_t] = - N \left ( \prod_{i=1}^n \left ( 1 + r_i(\mathbf{C}, R_j, z) d_i \right ) - 1 \right )

    For *analytic delta* purposes the :math:`\xi=-z`.

    .. rubric:: Examples

    .. ipython:: python
      :suppress:

      from rateslib.periods import ZeroFloatPeriod
      from rateslib.legs import CustomLeg
      from rateslib.scheduling import Schedule
      from datetime import datetime as dt

    .. ipython:: python

       fixings.add("MY_RATE_INDEX_6M", Series(
           index=[dt(2000, 1, 1), dt(2000, 7, 1), dt(2001, 1, 1), dt(2001, 7, 1)],
           data=[1.0, 2.0, 3.0, 4.0]
       ))
       period = ZeroFloatPeriod(
           schedule=Schedule(dt(2000, 1, 1), "2Y", "S"),
           fixing_method="IBOR",
           rate_fixings="MY_RATE_INDEX",
           convention="Act360",
           method_param=0,
       )
       period.cashflows()

    For more details of the individual compounded periods one can compose a
    :class:`~rateslib.legs.CustomLeg` and view the pseudo-cashflows.

    .. ipython:: python

       CustomLeg(period.float_periods).cashflows()

    .. ipython:: python
       :suppress:

       fixings.pop("MY_RATE_INDEX_6M")

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

       .. note::

          The following parameters are scheduling **period** parameters

    schedule: Schedule, :red:`required`
       The :class:`~rateslib.scheduling.Schedule` defining the individual *Periods*, including
       the *payment* and *ex-dividend* dates.

       .. note::

          The following define **floating rate** parameters.

    fixing_method: FloatFixingMethod, str, :green:`optional (set by 'defaults')`
        The :class:`~rateslib.enums.parameters.FloatFixingMethod` describing the determination
        of the floating rate for the period.
    method_param: int, :green:`optional (set by 'defaults')`
        A specific parameter that is used by the specific ``fixing_method``.
    fixing_frequency: Frequency, str, :green:`optional (set by 'frequency' or '1B')`
        The :class:`~rateslib.scheduling.Frequency` as a component of the
        :class:`~rateslib.data.fixings.FloatRateIndex`. If not given is assumed to match the
        frequency of the period for an IBOR type ``fixing_method`` or '1B' if RFR type.
    fixing_series: FloatRateSeries, str, :green:`optional (implied by other parameters)`
        The :class:`~rateslib.data.fixings.FloatRateSeries` as a component of the
        :class:`~rateslib.data.fixings.FloatRateIndex`. If not given inherits attributes given
        such as the ``calendar``, ``convention``, ``method_param`` etc.
    float_spread: float, Dual, Dual2, Variable, :green:`optional (set as 0.0)`
        The amount (in bps) added to the rate in the period rate determination. If not given is
        set to zero.
    spread_compound_method: SpreadCompoundMethod, str, :green:`optional (set by 'defaults')`
        The :class:`~rateslib.enums.parameters.SpreadCompoundMethod` used in the calculation
        of the period rate when combining a ``float_spread``. Used **only** with RFR type
        ``fixing_method``.
    rate_fixings: float, Dual, Dual2, Variable, Series, str, :green:`optional`
        The value of the rate fixing. If a scalar, is used directly. If a string identifier, links
        to the central ``fixings`` object and data loader.

        .. note::

          The following parameters define **non-deliverability**. If the *Period* is directly
          deliverable do not supply these parameters.

    pair: str, :green:`optional`
       The currency pair of the :class:`~rateslib.data.fixings.FXFixing` that determines
       settlement. The *reference currency* is implied from ``pair``. Must include ``currency``.
    fx_fixings: float, Dual, Dual2, Variable, Series, str, :green:`optional`
       The value of the :class:`~rateslib.data.fixings.FXFixing`. If a scalar is used directly.
       If a string identifier will link to the central ``fixings`` object and data loader.
    delivery: datetime, :green:`optional (set as 'payment')`
       The settlement delivery date of the :class:`~rateslib.data.fixings.FXFixing`.

       .. note::

          The following parameters define **indexation**. The *Period* will be considered
          indexed if any of ``index_method``, ``index_lag``, ``index_base``, ``index_fixings``
          are given.

    index_method : IndexMethod, str, :green:`optional (set by 'defaults')`
       The interpolation method, or otherwise, to determine index values from reference dates.
    index_lag: int, :green:`optional (set by 'defaults')`
       The indexation lag, in months, applied to the determination of index values.
    index_base: float, Dual, Dual2, Variable, :green:`optional`
       The specific value set of the base index value.
       If not given and ``index_fixings`` is a str fixings identifier that will be
       used to determine the base index value.
    index_fixings: float, Dual, Dual2, Variable, Series, str, :green:`optional`
       The index value for the reference date.
       If a scalar value this is used directly. If a string identifier will link to the
       central ``fixings`` object and data loader.
    index_only: bool, :green:`optional (set as False)`
       A flag which determines non-payment of notional on supported *Periods*.

    """  # noqa: E501

    @property
    def rate_params(self) -> _FloatRateParams:
        """The :class:`~rateslib.periods.parameters._FixedRateParams` of the *Period*."""
        return self.float_periods[0].rate_params

    @property
    def period_params(self) -> _PeriodParams:
        """The :class:`~rateslib.periods.parameters._PeriodParams` of the *Period*."""
        return self._period_params

    @property
    def schedule(self) -> Schedule:
        """The :class:`~rateslib.scheduling.Schedule` object for this *Period*."""
        return self._schedule

    @cached_property
    def dcf(self) -> float:
        """An overload for the calculation of the DCF, replacing `period_params.dcf`."""
        return sum(
            dcf(
                start=self.schedule.aschedule[i],
                end=self.schedule.aschedule[i + 1],
                convention=self.period_params.convention,
                termination=self.schedule.aschedule[-1],
                frequency=self.schedule.frequency_obj,
                stub=self.schedule._stubs[i],
                roll=NoInput(0),  # taken from Frequency obj
                calendar=self.schedule.calendar,
                adjuster=self.schedule.modifier,
            )
            for i in range(self.schedule.n_periods)
        )

    @property
    def float_spread(self) -> DualTypes:
        """The float spread parameter of each :class:`~rateslib.periods.FloatPeriod`."""
        return self._float_periods[0].rate_params.float_spread

    @float_spread.setter
    def float_spread(self, value: DualTypes) -> None:
        for period in self._float_periods:
            period.rate_params.float_spread = value

    @property
    def float_periods(self) -> list[FloatPeriod]:
        """
        The individual :class:`~rateslib.periods.FloatPeriod` that are
        compounded.
        """
        return self._float_periods

    def __init__(
        self,
        *,
        float_spread: DualTypes_ = NoInput(0),
        rate_fixings: DualTypes | Series[DualTypes] | str_ = NoInput(0),  # type: ignore[type-var]
        fixing_method: FloatFixingMethod | str_ = NoInput(0),
        method_param: int_ = NoInput(0),
        spread_compound_method: SpreadCompoundMethod | str_ = NoInput(0),
        fixing_frequency: Frequency | str_ = NoInput(0),
        fixing_series: FloatRateSeries | str_ = NoInput(0),
        schedule: Schedule,
        # currency args:
        notional: DualTypes_ = NoInput(0),
        currency: str_ = NoInput(0),
        # period params
        convention: str_ = NoInput(0),
        # non-deliverable args:
        pair: FXIndex | str_ = NoInput(0),
        fx_fixings: DualTypes | Series[DualTypes] | str_ = NoInput(0),  # type: ignore[type-var]
        delivery: datetime_ = NoInput(0),
        # index-args:
        index_base: DualTypes_ = NoInput(0),
        index_lag: int_ = NoInput(0),
        index_method: IndexMethod | str_ = NoInput(0),
        index_fixings: DualTypes | Series[DualTypes] | str_ = NoInput(0),  # type: ignore[type-var]
        index_only: bool_ = NoInput(0),
    ) -> None:
        self._schedule = schedule
        self._settlement_params = _init_SettlementParams_with_fx_pair(
            _currency=_drb(defaults.base_currency, currency).lower(),
            _payment=self.schedule.pschedule[-1],
            _notional=_drb(defaults.notional, notional),
            _ex_dividend=self.schedule.pschedule3[-1],
            _fx_pair=_maybe_get_fx_index(pair),
        )
        self._non_deliverable_params = _init_or_none_NonDeliverableParams(
            _currency=self.settlement_params.currency,
            _fx_index=pair,
            _delivery=_drb(self.settlement_params.payment, delivery),
            _fx_fixings=fx_fixings,
        )
        self._period_params = _PeriodParams(
            _start=self.schedule.aschedule[0],
            _end=self.schedule.aschedule[-1],
            _frequency=self.schedule.frequency_obj,
            _calendar=self.schedule.calendar,
            _adjuster=self.schedule.modifier,
            _stub=True,
            _convention=_get_convention(_drb(defaults.convention, convention)),
            _termination=self.schedule.aschedule[-1],
        )
        self._index_params = _init_or_none_IndexParams(
            _index_base=index_base,
            _index_lag=index_lag,
            _index_method=index_method,
            _index_fixings=index_fixings,
            _index_only=index_only,
            _index_base_date=self.schedule.aschedule[0],
            _index_reference_date=self.schedule.aschedule[-1],
        )
        rate_fixings_ = _leg_fixings_to_list(rate_fixings, self.schedule.n_periods)
        self._float_periods: list[FloatPeriod] = [
            FloatPeriod(
                float_spread=float_spread,
                rate_fixings=rate_fixings_[i],
                fixing_method=fixing_method,
                method_param=method_param,
                spread_compound_method=spread_compound_method,
                fixing_frequency=fixing_frequency,
                fixing_series=fixing_series,
                # currency args:
                payment=self.schedule.pschedule[i + 1],
                notional=notional,
                currency=currency,
                ex_dividend=self.schedule.pschedule3[i + 1],
                # period params
                start=self.schedule.aschedule[i],
                end=self.schedule.aschedule[i + 1],
                frequency=self.schedule.frequency_obj,
                convention=convention,
                termination=self.schedule.aschedule[-1],
                stub=self.schedule._stubs[i],
                roll=NoInput(0),  # inferred from frequency obj
                calendar=self.schedule.calendar,
                adjuster=self.schedule.modifier,
                # Each individual period is not genuine Period, only psuedo periods to derive the
                # cashflow calculation so no 'non-deliverable' or 'index' params are required.
            )
            for i in range(self.schedule.n_periods)
        ]

    def try_rate(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        **kwargs: Any,
    ) -> Result[DualTypes]:
        try:
            r_i = [period.rate(rate_curve=rate_curve) for period in self.float_periods]
            d_i = [period.period_params.dcf for period in self.float_periods]
        except Exception as e:
            return Err(e)

        f = self.schedule.periods_per_annum
        r = np.prod(1.0 + np.array(r_i) * np.array(d_i) / 100.0)
        r = r ** (1.0 / (self.dcf * f))
        r = (r - 1) * f * 100.0
        return Ok(r)

    def rate(self, *, rate_curve: CurveOption_ = NoInput(0)) -> DualTypes:
        r"""Calculate a single *rate* representation for the *Period's* cashflow.

        The *rate* is determined from the compounded *Period* rates according to:

        .. math::

           \left ( 1 + \frac{r}{f} \right )^{df} = \prod_{i=1}^n \left ( 1 + r_i(\mathbf{C}, R_j, z) d_i \right )

        Parameters
        ----------
        rate_curve: _BaseCurve or dict of such indexed by string tenor, optional
            Used to forecast floating period rates, if necessary.

        Returns
        -------
        float, Dual, Dual2 or Variable
        """  # noqa: E501

        return self.try_rate(rate_curve=rate_curve).unwrap()

    def unindexed_reference_cashflow(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        **kwargs: Any,
    ) -> DualTypes:
        # determine each rate from individual Periods
        r_i = [period.rate(rate_curve=rate_curve) for period in self.float_periods]
        d_i = [period.period_params.dcf for period in self.float_periods]
        r: DualTypes = np.prod(1.0 + np.array(r_i) * np.array(d_i) / 100.0) - 1.0
        return -self.settlement_params.notional * r

    def try_unindexed_reference_cashflow_analytic_delta(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
    ) -> Result[DualTypes]:
        try:
            r_i = [period.rate(rate_curve=rate_curve) for period in self._float_periods]
            d_i = [period.period_params.dcf for period in self._float_periods]
            a_i = [
                period.try_unindexed_reference_cashflow_analytic_delta(
                    rate_curve=rate_curve, disc_curve=disc_curve
                ).unwrap()
                for period in self._float_periods
            ]
        except Exception as e:
            return Err(e)

        lhs = np.prod(1.0 + np.array(r_i) * np.array(d_i) / 100.0)
        rhs = np.sum([a / (1 + r * d / 100.0) for (a, d, r) in zip(a_i, d_i, r_i, strict=False)])
        return Ok(lhs * rhs)

    def try_unindexed_reference_cashflow_analytic_rate_fixings(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: _FXVolOption_ = NoInput(0),
    ) -> Result[DataFrame]:
        try:
            r_i = [period.rate(rate_curve=rate_curve) for period in self.float_periods]
            d_i = [period.period_params.dcf for period in self.float_periods]
            dfs_i = [
                period.try_unindexed_reference_cashflow_analytic_rate_fixings(
                    rate_curve=rate_curve,
                    disc_curve=disc_curve,
                    fx=fx,
                    fx_vol=fx_vol,
                    index_curve=index_curve,
                ).unwrap()
                for period in self.float_periods
            ]
        except Exception as e:
            return Err(e)

        scalar = np.prod(1.0 + np.array(r_i) * np.array(d_i) / 100.0)
        dfs = [
            df * scalar / (1 + d * r / 100.0) for (df, d, r) in zip(dfs_i, d_i, r_i, strict=False)
        ]
        return Ok(concat(dfs))

    def cashflows(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: _FXVolOption_ = NoInput(0),
        base: str_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
    ) -> dict[str, Any]:
        d = super().cashflows(
            rate_curve=rate_curve,
            index_curve=index_curve,
            disc_curve=disc_curve,
            settlement=settlement,
            forward=forward,
            base=base,
        )
        d[defaults.headers["dcf"]] = self.dcf  # reinsert the overload
        return d


def _get_ibor_curve_from_dict(fixing_frequency: Frequency, d: dict[str, _BaseCurve]) -> _BaseCurve:
    remapped = {k.upper(): v for k, v in d.items()}
    try:
        freq_str = _get_tenor_from_frequency(fixing_frequency)
        return remapped[freq_str]
    except KeyError:
        raise ValueError(
            "If supplying `rate_curve` as dict must provide a tenor mapping key and curve for"
            f"the frequency of the given Period. The missing mapping is '{freq_str}'."
        )


def _get_ibor_curve_from_dict2(fixing_frequency: str, d: dict[str, _BaseCurve]) -> _BaseCurve:
    remapped = {k.upper(): v for k, v in d.items()}
    try:
        return remapped[fixing_frequency.upper()]
    except KeyError:
        raise ValueError(
            "If supplying `rate_curve` as dict must provide a tenor mapping key and curve for"
            f"the frequency of the given Period. The missing mapping is '{fixing_frequency}'."
        )


class _UnindexedReferenceCashflowFixingsSensitivity:
    @staticmethod
    def _ibor(
        self: FloatPeriod, rate_curve: _BaseCurve | dict[str, _BaseCurve]
    ) -> Result[DataFrame]:
        if self.period_params.stub:
            if isinstance(rate_curve, dict):
                rate_curve_: dict[str, _BaseCurve] = rate_curve
            else:
                rate_curve_ = {
                    _get_tenor_from_frequency(self.rate_params.fixing_frequency): rate_curve
                }
            return _UnindexedReferenceCashflowFixingsSensitivity._ibor_stub(
                self=self,
                rate_curve=rate_curve_,
                frequency_str=_get_tenor_from_frequency(self.rate_params.fixing_frequency),
            )
        else:
            if isinstance(rate_curve, dict):
                rate_curve__: _BaseCurve = _get_ibor_curve_from_dict(
                    self.rate_params.fixing_frequency, rate_curve
                )
            else:
                rate_curve__ = rate_curve
            return _UnindexedReferenceCashflowFixingsSensitivity._ibor_regular(
                self=self,
                rate_curve=rate_curve__,
                frequency_str=_get_tenor_from_frequency(self.rate_params.fixing_frequency),
            )

    @staticmethod
    def _ibor_regular(
        self: FloatPeriod,
        rate_curve: _BaseCurve,
        frequency_str: str,
    ) -> Result[DataFrame]:
        return Ok(
            DataFrame(
                index=Index(data=[self.rate_params.rate_fixing.date], name="obs_dates"),
                data=[
                    -self.settlement_params.notional * self.period_params.dcf * 0.0001
                    if isinstance(self.rate_params.rate_fixing.value, NoInput)
                    else 0.0
                ],
                columns=MultiIndex.from_tuples(
                    [
                        (
                            rate_curve.id,
                            self.settlement_params.currency,
                            self.settlement_params.notional_currency,
                            frequency_str,
                        )
                    ],
                    names=["identifier", "local_ccy", "display_ccy", "frequency"],
                ),
            )
        )

    @staticmethod
    def _ibor_stub(
        self: FloatPeriod,
        rate_curve: dict[str, _BaseCurve],
        frequency_str: str,
    ) -> Result[DataFrame]:
        # get consistent curves for the tenors of the stub fixings
        tenors, ends = _find_neighbouring_tenors(
            end=self.rate_params.rate_fixing.accrual_end,
            start=self.rate_params.rate_fixing.accrual_start,
            tenors=[_ for _ in rate_curve if _.upper() != "RFR"],
            rate_series=self.rate_params.rate_fixing.series,  # type: ignore[union-attr]
        )
        rate_curve_1: _BaseCurve = _get_ibor_curve_from_dict2(tenors[0], rate_curve)
        df1_res = _UnindexedReferenceCashflowFixingsSensitivity._ibor_regular(
            self=self,
            rate_curve=rate_curve_1,
            frequency_str=tenors[0],
        )
        if tenors[0] == tenors[1]:
            return df1_res  # then no multiple curves for the stub
        else:
            rate_curve_2: _BaseCurve = _get_ibor_curve_from_dict2(tenors[1], rate_curve)
            df2_res = _UnindexedReferenceCashflowFixingsSensitivity._ibor_regular(
                self=self,
                rate_curve=rate_curve_2,
                frequency_str=tenors[1],
            )
            alpha = (ends[1] - self.period_params.end) / (ends[1] - ends[0])
            return Ok(
                merge(
                    left=df1_res.unwrap() * alpha,
                    right=df2_res.unwrap() * (1 - alpha),
                    left_index=True,
                    right_index=True,
                )
            )

    @staticmethod
    def _rfr(
        self: FloatPeriod,
        rate_curve: _BaseCurve,
    ) -> Result[DataFrame]:
        rf: RFRFixing = self.rate_params.rate_fixing  # type: ignore[assignment]

        if isinstance(rf.value, NoInput):
            # then some sensitivity still exists
            drdr = _UnindexedReferenceCashflowFixingsSensitivity._rfr_drdr_approximation(
                self=self,
                rate_curve=rate_curve,
            )
        else:
            # all sensitivity is zero
            drdr = np.array([0.0 for _ in range(len(rf.dates_obs) - 1)])

        temp = Series(
            index=rf.dates_obs[:-1],
            data=-self.settlement_params.notional * self.period_params.dcf * 0.0001 * drdr,
        )
        temp[rf.populated.index] = 0.0

        df1 = DataFrame(
            index=Index(data=rf.dates_obs[:-1], name="obs_dates"),
            data=temp.to_list(),
            columns=MultiIndex.from_tuples(
                [
                    (
                        rate_curve.id,
                        self.settlement_params.currency,
                        self.settlement_params.notional_currency,
                        "1B",
                    )
                ],
                names=["identifier", "local_ccy", "display_ccy", "frequency"],
            ),
        )
        return Ok(df1)

    @staticmethod
    def _rfr_drdr_approximation(
        self: FloatPeriod,
        rate_curve: _BaseCurve,
    ) -> Arr1dObj:
        """
        Determine the value :math:`\frac{\\partial r(r_i, z)}{\\partial r_j}` for rate
        fixing sensitivity.

        For NoneSimple spread compounding this formula is exact, which covers most cases.

        For ISDAFlatCompounding this is approximated as the NoneSimple case so is an
        approximation.

        For ISDACompounding the geometric 1-day average rate is used as a component in the formula
        meaning the result is approximate.

        These values do **not** distinguish between published and unpublished fixings. This should
        be adjusted post.

        Returns
        -------
        ndarray
        """
        rf: RFRFixing = self.rate_params.rate_fixing  # type: ignore[assignment]
        d_hat_i = rf.dcfs_dcf
        z = self.rate_params.float_spread
        fixing_method = self.rate_params.fixing_method
        spread_compound_method = self.rate_params.spread_compound_method
        method_param = self.rate_params.method_param

        # approximate sensitivity to each fixing
        z = z / 100.0

        d = d_hat_i.sum()
        if fixing_method in [
            FloatFixingMethod.RFRLockoutAverage,
            FloatFixingMethod.RFRObservationShiftAverage,
            FloatFixingMethod.RFRLookbackAverage,
            FloatFixingMethod.RFRPaymentDelayAverage,
        ]:
            drdri: Arr1dObj = d_hat_i / d
        else:
            unpopulated = rf.unpopulated
            populated = rf.populated
            r_i = Series(index=rf.dates_obs[:-1], data=np.nan, dtype=object)
            r_i.update(populated)  #  type: ignore[arg-type]
            # determine the rate for the period, from the curve if necessary
            if unpopulated.index[0] < rate_curve.nodes.initial:
                raise ValueError(err.VE_BEFORE_INITIAL)
            _RFRRate._forecast_fixing_rates_from_curve(
                unpopulated=unpopulated,
                populated=populated,
                fixing_rates=r_i,  # type: ignore[arg-type]
                rate_curve=rate_curve,
                dates_obs=rf.dates_obs,
                dcfs_obs=rf.dcfs_obs,
            )
            r_star = _RFRRate._inefficient_calculation(
                fixing_rates=r_i,
                fixing_dcfs=d_hat_i,
                fixing_method=fixing_method,
                method_param=method_param,
                spread_compound_method=spread_compound_method,
                float_spread=self.rate_params.float_spread,
            ).unwrap()
            if spread_compound_method == SpreadCompoundMethod.ISDACompounding:
                # this makes a number of approximations including reversing a compounded spread
                # with a simple formula
                r_bar, d_bar, n = average_rate(
                    effective=self.period_params.start,
                    termination=self.period_params.end,
                    convention=self.rate_params.fixing_series.convention,
                    rate=r_star - z,
                    dcf=d,
                )
                drdri = d_hat_i / (1 + d_hat_i * (r_bar + z) / 100.0) * (1 + r_star / 100.0 * d) / d  # type: ignore[operator]
            # elif spread_compound_method == SpreadCompoundMethod.ISDAFlatCompounding:
            #     r_star = ((1 + d_bar * (r_bar + z) / 100.0) ** n - 1) * 100.0 / (n * d_bar)
            #     drdri1 = di / (1 + di * (r_bar + z) / 100.0) * ((r_star / 100.0 * d) + 1) / d
            #     drdri2 = di / (1 + di * r_bar / 100.0) * ((r_star0 / 100.0 * d) + 1) / d
            #     drdri = (drdri1 + drdri2) / 2.0
            else:  # spread_compound_method == SpreadCompoundMethod.NoneSimple:
                r_i_ = r_i.to_numpy()
                drdri = d_hat_i / (1 + d_hat_i * r_i_ / 100.0) * ((r_star - z) / 100.0 * d + 1) / d

        if fixing_method in [FloatFixingMethod.RFRLockoutAverage, FloatFixingMethod.RFRLockout]:
            for i in range(method_param):
                drdri[-(method_param + 1)] += drdri[-(i + 1)]
                drdri[-(i + 1)] = 0.0

        return drdri
