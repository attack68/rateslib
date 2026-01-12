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

from abc import ABCMeta

from rateslib.legs.protocols.analytic_delta import _WithAnalyticDelta
from rateslib.legs.protocols.analytic_fixings import _WithAnalyticRateFixings
from rateslib.legs.protocols.cashflows import _WithCashflows, _WithExDiv
from rateslib.legs.protocols.fixings import _WithFixings
from rateslib.legs.protocols.npv import _WithNPV


class _BaseLeg(
    _WithFixings,  # inherits _WIthNPV so first in MRO
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
    "_WithFixings",
    "_WithAnalyticDelta",
    "_WithAnalyticRateFixings",
    "_WithExDiv",
    "_BaseLeg",
]
