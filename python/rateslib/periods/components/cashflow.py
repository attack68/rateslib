from __future__ import annotations

from typing import TYPE_CHECKING

import rateslib.errors as err
from rateslib import defaults
from rateslib.enums.generics import NoInput, Ok, _drb
from rateslib.enums.parameters import IndexMethod
from rateslib.periods.components.parameters import (
    _CashflowRateParams,
    _IndexParams,
    _init_or_none_IndexParams,
    _PeriodParams,
    _SettlementParams,
)
from rateslib.periods.components.protocols import (
    _WithAnalyticDeltaStatic,
    _WithNPVCashflowsStatic,
    _WithRateFixingsExposureStatic,
)

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        Any,
        CurveOption_,
        DualTypes,
        DualTypes_,
        Result,
        Series,
        _BaseCurve_,
        bool_,
        datetime,
        datetime_,
        int_,
        str_,
    )


class Cashflow(_WithNPVCashflowsStatic, _WithAnalyticDeltaStatic, _WithRateFixingsExposureStatic):
    settlement_params: _SettlementParams
    period_params: _PeriodParams
    index_params: None | _IndexParams
    rate_params: _CashflowRateParams

    def __init__(
        self,
        *,
        # currency args:
        payment: datetime,
        notional: DualTypes,
        currency: str_ = NoInput(0),
        ex_dividend: datetime_ = NoInput(0),
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
        self.settlement_params = _SettlementParams(
            _notional=notional,
            _payment=payment,
            _currency=_drb(defaults.base_currency, currency).lower(),
            _pair=pair if isinstance(pair, NoInput) else pair.lower(),
            _fx_fixings=fx_fixings,
            _delivery=delivery,
            _ex_dividend=ex_dividend,
        )
        self.rate_params = _CashflowRateParams()
        self.period_params = _PeriodParams(  # data for `Cashflow` is placeholder
            _start=payment,
            _end=payment,
            _frequency=None,  # type: ignore[arg-type]
            _calendar=None,  # type: ignore[arg-type]
            _adjuster=None,  # type: ignore[arg-type]
            _stub=None,  # type: ignore[arg-type]
            _convention=None,  # type: ignore[arg-type]
            _termination=None,  # type: ignore[arg-type]
        )
        self.index_params = _init_or_none_IndexParams(
            _index_base=index_base,
            _index_lag=index_lag,
            _index_method=index_method,
            _index_fixings=index_fixings,
            _index_base_date=index_base_date,
            _index_reference_date=_drb(self.settlement_params.payment, index_reference_date),
            _index_only=index_only,
        )

    def try_unindexed_reference_cashflow(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
    ) -> Result[DualTypes]:
        return Ok(-self.settlement_params.notional)

    def try_unindexed_reference_analytic_delta(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
    ) -> Result[DualTypes]:
        return Ok(0.0)


class NonDeliverableCashflow(Cashflow):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not self.is_non_deliverable:
            raise ValueError(err.VE_NEEDS_ND_CURRENCY_PARAMS.format(type(self).__name__))
        if self.is_indexed:
            raise ValueError(err.VE_HAS_INDEX_PARAMS.format(type(self).__name__))


class IndexCashflow(Cashflow):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not self.is_indexed:
            raise ValueError(err.VE_NEEDS_INDEX_PARAMS.format(type(self).__name__))
        if self.is_non_deliverable:
            raise ValueError(err.VE_HAS_ND_CURRENCY_PARAMS.format(type(self).__name__))


class NonDeliverableIndexCashflow(Cashflow):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not self.is_indexed:
            raise ValueError(err.VE_NEEDS_INDEX_PARAMS.format(type(self).__name__))
        if not self.is_non_deliverable:
            raise ValueError(err.VE_NEEDS_ND_CURRENCY_PARAMS.format(type(self).__name__))
