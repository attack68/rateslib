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

from pandas import DataFrame

import rateslib.errors as err
from rateslib import defaults
from rateslib.data.fixings import _maybe_get_fx_index
from rateslib.enums.generics import Err, NoInput, Ok, _drb
from rateslib.enums.parameters import IndexMethod
from rateslib.periods.parameters import (
    _FixedRateParams,
    _init_or_none_IndexParams,
    _init_or_none_NonDeliverableParams,
    _init_SettlementParams_with_fx_pair,
    _PeriodParams,
)
from rateslib.periods.protocols import _BasePeriodStatic
from rateslib.scheduling import Adjuster, Frequency, dcf, get_calendar
from rateslib.scheduling.adjuster import _get_adjuster
from rateslib.scheduling.convention import _get_convention
from rateslib.scheduling.frequency import _get_frequency

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        Any,
        CalInput,
        CurveOption_,
        DualTypes,
        DualTypes_,
        FXForwards_,
        FXIndex,
        Result,
        RollDay,
        Schedule,
        Series,
        _BaseCurve_,
        _FXVolOption_,
        bool_,
        datetime,
        datetime_,
        int_,
        str_,
    )


class FixedPeriod(_BasePeriodStatic):
    r"""
    A *Period* defined by a fixed interest rate.

    The expected unindexed reference cashflow under the risk neutral distribution is defined as,

    .. math::

       \mathbb{E^Q} [\bar{C}_t] = -N d R

    For *analytic delta* purposes the :math:`\xi=-R`.

    .. role:: red

    .. role:: green

    .. rubric:: Examples

    .. ipython:: python
       :suppress:

       from rateslib.periods import FixedPeriod
       from datetime import datetime as dt

    .. ipython:: python

       period = FixedPeriod(
           start=dt(2000, 1, 1),
           end=dt(2001, 1, 1),
           payment=dt(2001, 1, 1),
           fixed_rate=5.0,
           notional=1e6,
           convention="ActActICMA",
           frequency="A",
       )
       period.cashflows()


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
        fixed_rate: DualTypes_ = NoInput(0),
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
        self._rate_params = _FixedRateParams(fixed_rate)

    def unindexed_reference_cashflow(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        **kwargs: Any,
    ) -> DualTypes:
        if isinstance(self.rate_params.fixed_rate, NoInput):
            raise ValueError(err.VE_NEEDS_FIXEDRATE)
        else:
            return (
                -self.settlement_params.notional
                * self.rate_params.fixed_rate
                * 0.01
                * self.period_params.dcf
            )

    # def try_cashflow(
    #     self,
    #     *,
    #     rate_curve: CurveOption_ = NoInput(0),
    #     disc_curve: _BaseCurve_ = NoInput(0),
    #     index_curve: _BaseCurve_ = NoInput(0),
    #     fx: FXForwards_ = NoInput(0),
    #     fx_vol: _FXVolOption_ = NoInput(0),
    # ) -> Result[DualTypes]:
    #     if self.index_params is None:
    #         if self.non_deliverable_params is None:
    #             return self.try_unindexed_reference_cashflow(
    #                 rate_curve=rate_curve,
    #                 disc_curve=disc_curve,
    #                 index_curve=index_curve,
    #                 fx=fx,
    #                 fx_vol=fx_vol,
    #             )
    #         else:
    #             return self.try_unindexed_cashflow(
    #                 rate_curve=rate_curve,
    #                 disc_curve=disc_curve,
    #                 index_curve=index_curve,
    #                 fx=fx,
    #                 fx_vol=fx_vol,
    #             )
    #     else:
    #         if self.non_deliverable_params is None:
    #             return self.try_reference_cashflow(
    #                 rate_curve=rate_curve,
    #                 disc_curve=disc_curve,
    #                 index_curve=index_curve,
    #                 fx=fx,
    #                 fx_vol=fx_vol,
    #             )
    #         else:
    #             rc = self.try_reference_cashflow(
    #                 rate_curve=rate_curve,
    #                 index_curve=index_curve,
    #                 disc_curve=disc_curve,
    #                 fx=fx,
    #                 fx_vol=fx_vol,
    #             )
    #             return self.try_convert_deliverable(value=rc, fx=fx)

    def try_unindexed_reference_cashflow_analytic_delta(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
    ) -> Result[DualTypes]:
        return Ok(self.settlement_params.notional * 0.0001 * self.period_params.dcf)

    def try_unindexed_reference_cashflow_analytic_rate_fixings(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: _FXVolOption_ = NoInput(0),
    ) -> Result[DataFrame]:
        return Ok(DataFrame())


class ZeroFixedPeriod(_BasePeriodStatic):
    r"""
    A *Period* defined by a fixed interest rate, as a representation of multiple compounded *Periods*.

    The expected unindexed reference cashflow under the risk neutral distribution is defined as,

    .. math::

       \mathbb{E^Q}[\bar{C}_t] = - N \left ( \left ( 1 + \frac{R}{f} \right )^{df} - 1 \right ), \qquad d = \sum_{i=1}^n d_i

    For *analytic delta* purposes the :math:`\xi=-R`.

    .. rubric:: Examples

    .. ipython:: python
       :suppress:

       from rateslib.periods import ZeroFixedPeriod
       from rateslib.legs import CustomLeg
       from rateslib.scheduling import Schedule
       from datetime import datetime as dt

    .. ipython:: python

       period = ZeroFixedPeriod(
           schedule=Schedule(dt(2000, 1, 1), "5Y", "A"),
           fixed_rate=5.0,
           convention="1",
       )
       period.cashflows()

    For more details of the individual compounded periods one can compose a
    :class:`~rateslib.legs.CustomLeg` and view the pseudo-cashflows.

    .. ipython:: python

       CustomLeg(period.fixed_periods).cashflows()

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

           The following define **fixed rate** parameters.

    fixed_rate: float, Dual, Dual2, Variable, :green:`optional`
        The fixed rate to determine the *Period* cashflow.

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
    def rate_params(self) -> _FixedRateParams:
        """The :class:`~rateslib.periods.parameters._FixedRateParams` of the *Period*."""
        return self._rate_params

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
    def fixed_periods(self) -> list[FixedPeriod]:
        """
        The individual :class:`~rateslib.periods.FixedPeriod` that are
        compounded.
        """
        return self._fixed_periods

    def __init__(
        self,
        *,
        fixed_rate: DualTypes_ = NoInput(0),
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
        self._rate_params = _FixedRateParams(fixed_rate)
        self._fixed_periods: list[FixedPeriod] = [
            FixedPeriod(
                fixed_rate=fixed_rate,
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

    def unindexed_reference_cashflow(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        **kwargs: Any,
    ) -> DualTypes:
        if isinstance(self.rate_params.fixed_rate, NoInput):
            raise ValueError(err.VE_NEEDS_FIXEDRATE)
        else:
            f = self.schedule.periods_per_annum
            return -self.settlement_params.notional * (
                (1 + self.rate_params.fixed_rate / (f * 100)) ** (self.dcf * f) - 1
            )

    def try_unindexed_reference_cashflow_analytic_delta(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
    ) -> Result[DualTypes]:
        if isinstance(self.rate_params.fixed_rate, NoInput):
            return Err(ValueError(err.VE_NEEDS_FIXEDRATE))
        else:
            f = self.schedule.periods_per_annum
            return Ok(
                self.settlement_params.notional
                * 0.0001
                * self.dcf
                * ((1 + self.rate_params.fixed_rate / (f * 100)) ** (self.dcf * f - 1))
            )

    def try_unindexed_reference_cashflow_analytic_rate_fixings(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        index_curve: _BaseCurve_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        fx_vol: _FXVolOption_ = NoInput(0),
    ) -> Result[DataFrame]:
        return Ok(DataFrame())

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
        """
        Return aggregated cashflow data for the *Period*.

        .. warning::

           This method is a convenience method to provide a visual representation of all
           associated calculation data. Calling this method to extracting certain values
           should be avoided. It is more efficient to source relevant parameters or calculations
           from object attributes or other methods directly.

        Parameters
        ----------
        rate_curve: _BaseCurve or dict of such indexed by string tenor, optional
            Used to forecast floating period rates, if necessary.
        index_curve: _BaseCurve, optional
            Used to forecast index values for indexation, if necessary.
        disc_curve: _BaseCurve, optional
            Used to discount cashflows.
        fx: FXForwards, optional
            The :class:`~rateslib.fx.FXForwards` object used for forecasting the
            ``fx_fixing`` for deliverable cashflows, if necessary. Or, an
            :class:`~rateslib.fx.FXRates` object purely for immediate currency conversion.
        fx_vol: FXDeltaVolSmile, FXSabrSmile, FXDeltaVolSurface, FXSabrSurface, optional
            The FX volatility *Smile* or *Surface* object used for determining Black calendar
            day implied volatility values.
        base: str, optional
            The currency to convert the *local settlement* NPV to.
        settlement: datetime, optional
            The assumed settlement date of the *PV* determination. Used only to evaluate
            *ex-dividend* status.
        forward: datetime, optional
            The future date to project the *PV* to using the ``disc_curve``.

        Returns
        -------
        dict of values
        """
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
