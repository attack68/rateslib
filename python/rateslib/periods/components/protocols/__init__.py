# ruff: noqa: I001

from abc import ABCMeta

from rateslib.periods.components.protocols.npv import (
    _WithIndexingStatic,
    _WithNonDeliverableStatic,
    _WithNPV,
    _WithNPVStatic,
)
from rateslib.periods.components.protocols.analytic_delta import (
    _WithAnalyticDelta,
    _WithAnalyticDeltaStatic,
)
from rateslib.periods.components.protocols.analytic_fixings import (
    _WithAnalyticRateFixings,
    _WithAnalyticRateFixingsStatic,
)
from rateslib.periods.components.protocols.analytic_greeks import _WithAnalyticFXOptionGreeks
from rateslib.periods.components.protocols.cashflows import (
    _WithCashflows,
    _WithCashflowsStatic,
)


class _BasePeriod(
    _WithCashflows,
    _WithAnalyticDelta,
    _WithAnalyticRateFixings,
    _WithAnalyticFXOptionGreeks,
    metaclass=ABCMeta,
):
    """Abstract base class for *Period* types."""

    pass


class _BasePeriodStatic(
    _WithCashflowsStatic,
    _WithAnalyticDeltaStatic,
    _WithAnalyticRateFixingsStatic,
    _BasePeriod,
    metaclass=ABCMeta,
):
    """Abstract base class for *Static Period* types."""

    pass


__all__ = [
    "_WithNPV",
    "_WithCashflows",
    "_WithAnalyticDelta",
    "_WithAnalyticRateFixings",
    "_WithAnalyticFXOptionGreeks",
    "_WithNPVStatic",
    "_WithCashflowsStatic",
    "_WithAnalyticDeltaStatic",
    "_WithAnalyticRateFixingsStatic",
    "_WithIndexingStatic",
    "_WithNonDeliverableStatic",
]
