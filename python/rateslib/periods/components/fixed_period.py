from __future__ import annotations

from typing import TYPE_CHECKING

import rateslib.errors as err
from rateslib.curves._parsers import _try_disc_required_maybe_from_curve
from rateslib.enums.generics import Err, NoInput, Ok
from rateslib.periods.components.base_period import BasePeriod
from rateslib.periods.components.parameters import (
    _FixedRateParams,
)

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        Any,
        CurveOption_,
        DualTypes,
        DualTypes_,
        Result,
        _BaseCurve_,
    )


class FixedPeriod(BasePeriod):
    rate_params: _FixedRateParams

    def __init__(self, *, fixed_rate: DualTypes_ = NoInput(0), **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.rate_params = _FixedRateParams(fixed_rate)

    def try_unindexed_reference_cashflow(
        self,
        rate_curve: CurveOption_ = NoInput(0),
    ) -> Result[DualTypes]:
        if isinstance(self.rate_params.fixed_rate, NoInput):
            return Err(ValueError(err.VE_NEEDS_FIXEDRATE))
        else:
            return Ok(
                -self.settlement_params.notional
                * self.rate_params.fixed_rate
                * 0.01
                * self.period_params.dcf
            )

    def try_unindexed_reference_analytic_delta(
        self,
        *,
        rate_curve: CurveOption_ = NoInput(0),
        disc_curve: _BaseCurve_ = NoInput(0),
    ) -> Result[DualTypes]:
        disc_curve_ = _try_disc_required_maybe_from_curve(curve=rate_curve, disc_curve=disc_curve)
        if isinstance(disc_curve_, Err):
            return disc_curve_
        return Ok(
            self.settlement_params.notional
            * 0.0001
            * self.period_params.dcf
            * disc_curve_.unwrap()[self.settlement_params.payment]
        )


class NonDeliverableFixedPeriod(FixedPeriod):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not self.is_non_deliverable:
            raise ValueError(err.VE_NEEDS_ND_CURRENCY_PARAMS.format(type(self).__name__))
        if self.is_indexed:
            raise ValueError(err.VE_HAS_INDEX_PARAMS.format(type(self).__name__))


class IndexFixedPeriod(FixedPeriod):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not self.is_indexed:
            raise ValueError(err.VE_NEEDS_INDEX_PARAMS.format(type(self).__name__))
        if self.is_non_deliverable:
            raise ValueError(err.VE_HAS_ND_CURRENCY_PARAMS.format(type(self).__name__))


class NonDeliverableIndexFixedPeriod(FixedPeriod):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not self.is_indexed:
            raise ValueError(err.VE_NEEDS_INDEX_PARAMS.format(type(self).__name__))
        if not self.is_non_deliverable:
            raise ValueError(err.VE_NEEDS_ND_CURRENCY_PARAMS.format(type(self).__name__))
