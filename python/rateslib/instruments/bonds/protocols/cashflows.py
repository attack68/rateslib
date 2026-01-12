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

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        FixedLeg,
        FloatLeg,
        datetime,
    )


class _WithExDiv(Protocol):
    """
    Protocol to determine the *yield-to-maturity* of a bond type *Instrument*.
    """

    @property
    def leg1(self) -> FixedLeg | FloatLeg: ...

    def ex_div(self, settlement: datetime) -> bool:
        """
        Return a boolean whether the security is ex-div at the given settlement.

        Parameters
        ----------
        settlement : datetime
            The settlement date to test.

        Returns
        -------
        bool

        Notes
        -----
        Uses the UK DMO convention of returning *False* if ``settlement``
        **is on or before** the ex-div date for a regular coupon period.

        This is evaluated by analysing the attribute ``pschedule3`` of the associated
        :class:`~rateslib.scheduling.Schedule` object of the *Leg*.
        """
        return self.leg1.ex_div(settlement)
