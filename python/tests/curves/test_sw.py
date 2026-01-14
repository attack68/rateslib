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
from datetime import timedelta

import numpy as np
import pytest
from rateslib import dual_log
from rateslib.curves.academic import SmithWilsonCurve
from rateslib.dual import Dual2
from rateslib.scheduling import Convention


def test_init():
    sw = SmithWilsonCurve(
        nodes={dt(2000, 1, 1): 0.10, dt(2001, 1, 1): -0.1, dt(2002, 1, 1): 0.5},
        ufr=4.2,
    )
    result = sw.rate(dt(2001, 1, 1), "1b")
    expected = 3.3906104222626796
    assert abs(result - expected) < 1e-5
    assert sw.meta.convention == Convention.Act365_25


def test_cache():
    sw = SmithWilsonCurve(
        nodes={dt(2000, 1, 1): 0.10, dt(2001, 1, 1): -0.1, dt(2002, 1, 1): 0.5},
        ufr=4.2,
    )
    sw.rate(dt(2001, 1, 1), "1b")
    assert dt(2001, 1, 1) in sw._cache

    old_state = sw._state
    sw._set_node_vector([1.0, 1.0], 0)
    assert sw._state != old_state
    assert dt(2001, 1, 1) not in sw._cache


def test_special_domain():
    sw = SmithWilsonCurve(
        nodes={dt(2000, 1, 1): 0.10, dt(2001, 1, 1): -0.1, dt(2002, 1, 1): 0.5},
        ufr=4.2,
    )
    assert sw[dt(2000, 1, 1)] == 1.0
    assert sw[dt(1999, 12, 31)] == 0.0


def test_getters():
    sw = SmithWilsonCurve(
        nodes={dt(2000, 1, 1): 0.10, dt(2001, 1, 1): -0.1, dt(2002, 1, 1): 0.5},
        ufr=4.2,
        id="v",
    )
    assert all(sw._get_node_vector() == np.array([-0.1, 0.5]))
    assert sw._get_node_vars() == ("v1", "v2")

    sw = SmithWilsonCurve(
        nodes={dt(2000, 1, 1): 0.10, dt(2001, 1, 1): -0.1, dt(2002, 1, 1): 0.5},
        ufr=4.2,
        solve_alpha=True,
        id="v",
    )
    assert all(sw._get_node_vector() == np.array([0.10, -0.1, 0.5]))
    assert sw._get_node_vars() == ("v0", "v1", "v2")


def test_set_ad_order():
    sw = SmithWilsonCurve(
        nodes={dt(2000, 1, 1): 0.10, dt(2001, 1, 1): -0.1, dt(2002, 1, 1): 0.5},
        ufr=4.2,
        id="v",
        ad=2,
    )
    assert isinstance(sw.nodes.values[0], Dual2)
    sw._set_ad_order(2)  # does nothing
    assert isinstance(sw.nodes.values[0], Dual2)

    with pytest.raises(ValueError):
        sw._set_ad_order(3)


def test_eiopa_example():
    from rateslib import FixedRateBond, Solver

    sw = SmithWilsonCurve(
        nodes={dt(2000, 1, 1): 0.12376, **{dt(2000 + i, 1, 1): 0.1 for i in range(1, 21)}},
        solve_alpha=False,
        ufr=4.2,
        id="academic_curve",
    )
    coupons = [
        0.2,
        0.225,
        0.3,
        0.425,
        0.55,
        0.7,
        0.85,
        1.0,
        1.15,
        1.275,
        1.4,
        1.475,
        1.575,
        1.65,
        1.7,
        1.75,
        1.8,
        1.825,
        1.85,
        1.875,
    ]
    bonds = [
        FixedRateBond(
            effective=dt(2000, 1, 1),
            termination=f"{i}Y",
            fixed_rate=coupons[i - 1],
            calendar="all",
            ex_div=1,
            convention="actacticma",
            frequency="A",
            curves="academic_curve",
            metric="dirty_price",
        )
        for i in range(1, 21)
    ]
    prices = [100.0] * 20
    Solver(curves=[sw], instruments=bonds, s=prices)

    assert abs(sw.k - 0.737944) < 5e-3

    eiopa_u = [
        0.00,
        0.25,
        0.50,
        0.75,
        1.00,
        1.25,
        1.50,
        1.75,
        2.00,
        2.25,
        2.50,
        2.75,
        3.00,
        3.25,
        3.50,
        3.75,
        4.00,
        4.25,
        4.50,
        4.75,
        5.00,
        5.25,
        5.50,
        5.75,
        6.00,
        6.25,
        6.50,
        6.75,
        7.00,
        7.25,
        7.50,
        7.75,
        8.00,
        8.25,
        8.50,
        8.75,
        9.00,
        9.25,
        9.50,
        9.75,
        10.00,
        10.25,
        10.50,
        10.75,
        11.00,
        11.25,
        11.50,
        11.75,
        12.00,
        12.25,
        12.50,
        12.75,
        13.00,
        13.25,
        13.50,
        13.75,
        14.00,
        14.25,
        14.50,
        14.75,
        15.00,
        15.25,
        15.50,
        15.75,
        16.00,
        16.25,
        16.50,
        16.75,
        17.00,
        17.25,
        17.50,
        17.75,
        18.00,
        18.25,
        18.50,
        18.75,
        19.00,
        19.25,
        19.50,
        19.75,
        20.00,
        40.00,
        60.0,
    ]
    eiopa_v = [
        1.0000,
        0.9996,
        0.9991,
        0.9986,
        0.9980,
        0.9975,
        0.9969,
        0.9962,
        0.9955,
        0.9947,
        0.9937,
        0.9925,
        0.9910,
        0.9894,
        0.9874,
        0.9854,
        0.9831,
        0.9808,
        0.9784,
        0.9757,
        0.9728,
        0.9696,
        0.9662,
        0.9625,
        0.9587,
        0.9547,
        0.9506,
        0.9463,
        0.9419,
        0.9373,
        0.9325,
        0.9275,
        0.9224,
        0.9170,
        0.9114,
        0.9059,
        0.9004,
        0.8950,
        0.8896,
        0.8841,
        0.8783,
        0.8723,
        0.8661,
        0.8601,
        0.8544,
        0.8493,
        0.8444,
        0.8395,
        0.8343,
        0.8287,
        0.8226,
        0.8164,
        0.8103,
        0.8045,
        0.7989,
        0.7935,
        0.7883,
        0.7833,
        0.7784,
        0.7736,
        0.7688,
        0.7640,
        0.7591,
        0.7540,
        0.7489,
        0.7437,
        0.7385,
        0.7334,
        0.7286,
        0.7242,
        0.7200,
        0.7159,
        0.7119,
        0.7077,
        0.7035,
        0.6993,
        0.6951,
        0.6909,
        0.6867,
        0.6825,
        0.6782,
        0.3330,
        0.1475,
    ]
    for i in range(80):
        date = dt(2000, 1, 1) + timedelta(days=round(eiopa_u[i] * 365.25, 0))
        rateslib_v = sw[date]
        assert abs(rateslib_v - eiopa_v[i]) < 2e-4


def test_2357_example():
    from rateslib import FixedRateBond, Solver

    sw = SmithWilsonCurve(
        nodes={
            dt(2000, 1, 1): 0.12376,
            **{dt(2000 + i, 1, 1): 0.1 for i in [2, 3, 5, 7]},
            # **{dt(2000+i, 1, 1): 0.1 for i in [1,2,3,4,5,6,7]}
        },
        solve_alpha=False,
        ufr=4.2,
        id="academic_curve",
    )
    sw2 = SmithWilsonCurve(
        nodes={
            dt(2000, 1, 1): 0.12376,
            # **{dt(2000+i, 1, 1): 0.1 for i in [2,3,5,7]}
            **{dt(2000 + i, 1, 1): 0.1 for i in [1, 2, 3, 4, 5, 6, 7]},
        },
        solve_alpha=False,
        ufr=4.2,
        id="academic_curve",
    )
    coupons = [1.5, 1.8, 2.2, 2.5]
    bonds = [
        FixedRateBond(
            effective=dt(2000, 1, 1),
            termination=f"{i}Y",
            fixed_rate=coupons[idx],
            frequency="A",
            convention="ActActICMA",
            calendar="all",
            modifier="F",
            curves="academic_curve",
            metric="dirty_price",
        )
        for (idx, i) in enumerate([2, 3, 5, 7])
    ]
    prices = [100.0] * 4
    Solver(curves=[sw], instruments=bonds, s=prices)
    Solver(curves=[sw2], instruments=bonds, s=prices)

    eiopa_u = [
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        1.00,
        1.10,
        1.20,
        1.30,
        1.40,
        1.50,
        1.60,
        1.70,
        1.80,
        1.90,
        2.00,
        2.10,
        2.20,
        2.30,
        2.40,
        2.50,
        2.60,
        2.70,
        2.80,
        2.90,
        3.00,
        3.10,
        3.20,
        3.30,
        3.40,
        3.50,
        3.60,
        3.70,
        3.80,
        3.90,
        4.00,
        4.10,
        4.20,
        4.30,
        4.40,
        4.50,
        4.60,
        4.70,
        4.80,
        4.90,
        5.00,
        5.10,
        5.20,
        5.30,
        5.40,
        5.50,
        5.60,
        5.70,
        5.80,
        5.90,
        6.00,
        6.10,
        6.20,
        6.30,
        6.40,
        6.50,
        6.60,
        6.70,
        6.80,
        6.90,
        7.00,
        7.10,
        7.20,
        7.30,
        7.40,
        7.50,
        7.60,
        7.70,
        7.80,
        7.90,
        8.00,
    ]
    eiopa_v = [
        0.9989,
        0.9977,
        0.9965,
        0.9953,
        0.9941,
        0.9929,
        0.9916,
        0.9903,
        0.9889,
        0.9875,
        0.9861,
        0.9846,
        0.9831,
        0.9815,
        0.9799,
        0.9781,
        0.9764,
        0.9745,
        0.9726,
        0.9706,
        0.9686,
        0.9664,
        0.9642,
        0.9620,
        0.9597,
        0.9573,
        0.9550,
        0.9526,
        0.9501,
        0.9477,
        0.9452,
        0.9428,
        0.9403,
        0.9378,
        0.9353,
        0.9328,
        0.9303,
        0.9277,
        0.9252,
        0.9226,
        0.9200,
        0.9174,
        0.9148,
        0.9122,
        0.9095,
        0.9069,
        0.9042,
        0.9015,
        0.8988,
        0.8961,
        0.8933,
        0.8906,
        0.8878,
        0.8850,
        0.8822,
        0.8794,
        0.8765,
        0.8737,
        0.8708,
        0.8680,
        0.8651,
        0.8623,
        0.8594,
        0.8565,
        0.8536,
        0.8507,
        0.8479,
        0.8450,
        0.8421,
        0.8392,
        0.8363,
        0.8335,
        0.8306,
        0.8277,
        0.8248,
        0.8220,
        0.8191,
        0.8163,
        0.8134,
        0.8106,
    ]

    # from matplotlib import pyplot as plt
    # fig, ax, lines = sw.plot("Z", comparators=[sw2])
    # ax.scatter(
    #     [dt(2000, 1, 1) + timedelta(days=round(u*365.25)) for u in eiopa_u],
    #     [100.0 * dual_log(v) / -t for v,t in zip(eiopa_v, eiopa_u)],
    # )
    # plt.show()

    for i in range(80):
        date = dt(2000, 1, 1) + timedelta(days=round(eiopa_u[i] * 365.25, 0))
        rateslib_v = sw[date]
        assert abs(rateslib_v - eiopa_v[i]) < 2e-4
