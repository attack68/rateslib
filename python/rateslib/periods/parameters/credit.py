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


class _CreditParams:
    """
    Parameters for *Period* cashflows associated with credit events.

    """

    _premium_accrued: bool

    def __init__(self, _premium_accrued: bool) -> None:
        self._premium_accrued = _premium_accrued

    @property
    def premium_accrued(self) -> bool:
        """Whether the premium is accrued within the period to default."""
        return self._premium_accrued
