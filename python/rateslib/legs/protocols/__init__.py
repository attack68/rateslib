from abc import ABCMeta

from rateslib.legs.protocols.analytic_delta import _WithAnalyticDelta
from rateslib.legs.protocols.analytic_fixings import _WithAnalyticRateFixings
from rateslib.legs.protocols.cashflows import _WithCashflows, _WithExDiv
from rateslib.legs.protocols.npv import _WithNPV


class _BaseLeg(
    _WithNPV,
    _WithCashflows,
    _WithAnalyticDelta,
    _WithAnalyticRateFixings,
    metaclass=ABCMeta,
):
    """Abstract base class used in the construction of *Legs*."""

    pass


__all__ = [
    "_WithNPV",
    "_WithCashflows",
    "_WithAnalyticDelta",
    "_WithAnalyticRateFixings",
    "_WithExDiv",
    "_BaseLeg",
]
