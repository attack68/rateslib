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

from abc import ABCMeta
from typing import TYPE_CHECKING

from rateslib.instruments.protocols.analytic_delta import _WithAnalyticDelta
from rateslib.instruments.protocols.analytic_fixings import _WithAnalyticRateFixings
from rateslib.instruments.protocols.cashflows import _WithCashflows
from rateslib.instruments.protocols.fixings import _WithFixings
from rateslib.instruments.protocols.kwargs import _KWArgs
from rateslib.instruments.protocols.npv import _WithNPV
from rateslib.instruments.protocols.rate import _WithRate
from rateslib.instruments.protocols.sensitivities import _WithSensitivities

if TYPE_CHECKING:
    pass
    # from rateslib.typing import ()


class _BaseInstrument(
    _WithSensitivities,
    _WithNPV,
    _WithRate,
    _WithCashflows,
    _WithFixings,
    _WithAnalyticDelta,
    _WithAnalyticRateFixings,
    metaclass=ABCMeta,
):
    """Abstract base class used in the construction of *Instruments*."""


__all__ = [
    "_KWArgs",
    "_WithNPV",
    "_WithRate",
    "_WithCashflows",
    "_WithFixings",
    "_WithAnalyticDelta",
    "_WithAnalyticRateFixings",
    "_WithSensitivities",
    "_BaseInstrument",
]
