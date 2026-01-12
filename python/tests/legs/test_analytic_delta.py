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
from rateslib.legs import FixedLeg
from rateslib.scheduling import Schedule


@pytest.fixture
def curve():
    nodes = {
        dt(2022, 1, 1): 1.00,
        dt(2022, 4, 1): 0.99,
        dt(2022, 7, 1): 0.98,
        dt(2022, 10, 1): 0.97,
    }
    return Curve(nodes=nodes, interpolation="log_linear")


def test_analytic_delta_protocol_local(curve):
    leg = FixedLeg(
        schedule=Schedule(
            effective=dt(2022, 1, 1),
            termination=dt(2022, 4, 1),
            frequency="M",
        ),
        fixed_rate=1.0,
    )
    result = leg.analytic_delta(disc_curve=curve, local=True)
    expected = {"usd": 24.827510962072353}
    assert result == expected


def test_forward_settlement(curve):
    # tset that the analytic delta reacts to the settlement/ex-div constraint
    leg = FixedLeg(
        schedule=Schedule(
            effective=dt(2021, 12, 2),
            termination=dt(2022, 4, 2),
            frequency="M",
            payment_lag=0,
        ),
        fixed_rate=1.0,
        notional=1e9,
    )
    result = leg.analytic_delta(disc_curve=curve, local=False)
    result2 = leg.analytic_delta(disc_curve=curve, local=False, settlement=dt(2022, 1, 3))
    assert result2 < (result - 5000)
