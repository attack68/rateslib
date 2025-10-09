from datetime import datetime as dt

import pytest
from rateslib.curves import Curve
from rateslib.legs.components import FixedLeg
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
