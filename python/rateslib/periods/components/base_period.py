from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING

from rateslib import defaults
from rateslib.enums.generics import NoInput, _drb
from rateslib.enums.parameters import IndexMethod
from rateslib.periods.components.parameters import (
    _IndexParams,
    _init_or_none_IndexParams,
    _init_or_none_NonDeliverableParams,
    _init_SettlementParams_with_fx_pair,
    _NonDeliverableParams,
    _PeriodParams,
    _SettlementParams,
)
from rateslib.periods.components.protocols import (
    _WithAnalyticDeltaStatic,
    _WithNPVCashflowsStatic,
    _WithRateFixingsExposureStatic,
)
from rateslib.scheduling import Adjuster, Frequency, get_calendar
from rateslib.scheduling.adjuster import _get_adjuster
from rateslib.scheduling.convention import _get_convention
from rateslib.scheduling.frequency import _get_frequency

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        CalInput,
        DualTypes,
        DualTypes_,
        RollDay,
        Series,
        bool_,
        datetime,
        datetime_,
        int_,
        str_,
    )


class BasePeriod(
    _WithNPVCashflowsStatic,
    _WithAnalyticDeltaStatic,
    _WithRateFixingsExposureStatic,
    metaclass=ABCMeta,
):
    """
    An abstract base class for implementing structural *Periods*

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

    """

    @property
    def period_params(self) -> _PeriodParams:
        """The :class:`~rateslib.periods.components.parameters._PeriodParams` of the *Period*."""
        return self._period_params

    @property
    def settlement_params(self) -> _SettlementParams:
        """The :class:`~rateslib.periods.components.parameters._SettlementParams` of the
        *Period*."""
        return self._settlement_params

    @property
    def index_params(self) -> _IndexParams | None:
        """The :class:`~rateslib.periods.components.parameters._IndexParams` of the *Period*,
        if any."""
        return self._index_params

    @property
    def non_deliverable_params(self) -> _NonDeliverableParams | None:
        """The :class:`~rateslib.periods.components.parameters._NonDeliverableParams` of the
        *Period*., if any."""
        return self._non_deliverable_params

    @abstractmethod
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
        # non-deliverable args:
        pair: str_ = NoInput(0),
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
    ):
        self._settlement_params = _init_SettlementParams_with_fx_pair(
            _currency=_drb(defaults.base_currency, currency).lower(),
            _payment=payment,
            _notional=_drb(defaults.notional, notional),
            _ex_dividend=_drb(payment, ex_dividend),
            _fx_pair=pair,
        )
        self._non_deliverable_params = _init_or_none_NonDeliverableParams(
            _currency=self.settlement_params.currency,
            _pair=pair,
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
