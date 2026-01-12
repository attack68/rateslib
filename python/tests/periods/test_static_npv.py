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
from rateslib.curves import Curve
from rateslib.periods import Cashflow


class TestStaticNPV:
    @pytest.mark.parametrize(
        ("settlement", "forward", "expected"),
        [
            (dt(2000, 1, 1), dt(2000, 1, 1), 80.0),
            (dt(2000, 1, 1), dt(2000, 1, 6), 100.0 * 0.8 / 0.75),
            (dt(2000, 1, 2), dt(2000, 1, 5), 100.0),
            (dt(2000, 1, 4), dt(2000, 1, 5), 0.0),
        ],
    )
    def test_settlement_forward(self, settlement, forward, expected):
        # test the example in the book
        curve = Curve(
            nodes={
                dt(2000, 1, 1): 1.0,
                dt(2000, 1, 2): 0.95,
                dt(2000, 1, 3): 0.90,
                dt(2000, 1, 4): 0.85,
                dt(2000, 1, 5): 0.80,
                dt(2000, 1, 6): 0.75,
            }
        )
        cf = Cashflow(
            currency="usd",
            notional=-100.0,
            payment=dt(2000, 1, 5),
            ex_dividend=dt(2000, 1, 3),
        )
        result = cf.npv(disc_curve=curve, settlement=settlement, forward=forward)
        assert abs(result - expected) < 1e-7
