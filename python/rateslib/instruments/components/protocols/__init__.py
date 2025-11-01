from __future__ import annotations

from abc import ABCMeta
from typing import TYPE_CHECKING

from rateslib.instruments.components.protocols.analytic_delta import _WithAnalyticDelta
from rateslib.instruments.components.protocols.analytic_fixings import _WithAnalyticRateFixings
from rateslib.instruments.components.protocols.cashflows import _WithCashflows
from rateslib.instruments.components.protocols.npv import _WithNPV
from rateslib.instruments.components.protocols.rate import _WithRate
from rateslib.instruments.components.protocols.sensitivities import _WithSensitivities

if TYPE_CHECKING:
    pass
    # from rateslib.typing import ()


class _BaseInstrument(
    _WithSensitivities,
    _WithNPV,
    _WithRate,
    _WithCashflows,
    _WithAnalyticDelta,
    _WithAnalyticRateFixings,
    metaclass=ABCMeta,
):
    """Abstract base class used in the construction of *Instruments*."""


__all__ = [
    "_WithNPV",
    "_WithRate",
    "_WithCashflows",
    "_WithAnalyticDelta",
    "_WithAnalyticRateFixings",
    "_WithSensitivities",
    "_BaseInstrument",
]
