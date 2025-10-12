from abc import ABCMeta

from rateslib.legs.components.protocols.analytic_delta import _WithAnalyticDelta
from rateslib.legs.components.protocols.cashflows import _WithCashflows
from rateslib.legs.components.protocols.fixings import _WithAnalyticRateFixingsSensitivity
from rateslib.legs.components.protocols.npv import _WithNPV


class _BaseLeg(
    _WithNPV,
    _WithCashflows,
    _WithAnalyticDelta,
    _WithAnalyticRateFixingsSensitivity,
    metaclass=ABCMeta,
):
    """Abstract base class used in the construction of *Legs*."""

    pass


__all__ = [
    "_WithNPV",
    "_WithCashflows",
    "_WithAnalyticDelta",
    "_WithAnalyticRateFixingsSensitivity",
    "_BaseLeg",
]
