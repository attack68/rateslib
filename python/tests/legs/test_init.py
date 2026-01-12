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

from datetime import datetime as dt

import pytest
from rateslib.legs.fixed import FixedLeg
from rateslib.scheduling import Schedule


class TestFixedLeg:
    def test_init(self):
        FixedLeg(
            schedule=Schedule(
                effective=dt(2000, 1, 1),
                termination=dt(2001, 1, 1),
                frequency="3M",
                payment_lag=2,
                payment_lag_exchange=0,
                extra_lag=-1,
            ),
            notional=1000000.0,
            amortization=1000.0,
            currency="USD",
            final_exchange=True,
            initial_exchange=True,
        )
