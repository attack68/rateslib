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

# ruff: noqa: I001

from abc import ABCMeta

from rateslib.periods.protocols.npv import (
    _WithIndexingStatic,
    _WithNonDeliverableStatic,
    _WithNPV,
    _WithNPVStatic,
)
from rateslib.periods.protocols.analytic_delta import (
    _WithAnalyticDelta,
    _WithAnalyticDeltaStatic,
)
from rateslib.periods.protocols.analytic_fixings import (
    _WithAnalyticRateFixings,
    _WithAnalyticRateFixingsStatic,
)
from rateslib.periods.protocols.analytic_greeks import _WithAnalyticFXOptionGreeks
from rateslib.periods.protocols.cashflows import (
    _WithCashflows,
    _WithCashflowsStatic,
)
from rateslib.periods.protocols.fixings import (
    _WithFixings,
)


class _BasePeriod(
    _WithCashflows,
    _WithAnalyticDelta,
    _WithAnalyticRateFixings,
    _WithFixings,
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
    "_BasePeriod",
    "_BasePeriodStatic",
    "_WithNPV",
    "_WithCashflows",
    "_WithFixings",
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
