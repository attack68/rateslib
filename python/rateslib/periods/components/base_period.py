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
    _init_SettlementParams_with_nd_pair,
    _PeriodParams,
    _SettlementParams,
)
from rateslib.periods.components.parameters.settlement import _NonDeliverableParams
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
    settlement_params: _SettlementParams
    period_params: _PeriodParams
    index_params: _IndexParams | None
    non_deliverable_params: _NonDeliverableParams | None

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
        self.settlement_params = _init_SettlementParams_with_nd_pair(
            _currency=_drb(defaults.base_currency, currency).lower(),
            _payment=payment,
            _notional=_drb(defaults.notional, notional),
            _ex_dividend=_drb(payment, ex_dividend),
            _non_deliverable_pair=pair,
        )
        self.non_deliverable_params = _init_or_none_NonDeliverableParams(
            _currency=self.settlement_params.currency,
            _pair=pair,
            _delivery=_drb(self.settlement_params.payment, delivery),
            _fx_fixings=fx_fixings,
        )
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
        self.index_params = _init_or_none_IndexParams(
            _index_base=index_base,
            _index_lag=index_lag,
            _index_method=index_method,
            _index_fixings=index_fixings,
            _index_only=index_only,
            _index_base_date=index_base_date,
            _index_reference_date=_drb(self.period_params.end, index_reference_date),
        )
