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

import numpy as np
import pytest
from rateslib.curves.academic import NelsonSiegelSvenssonCurve
from rateslib.dual import Dual2
from rateslib.scheduling import Convention


def test_init():
    ns = NelsonSiegelSvenssonCurve(
        dates=(dt(2000, 1, 1), dt(2030, 1, 1)),
        parameters=(0.01, 0.01, 0.05, 1.0, 0.05, 1.0),
    )
    result = ns.rate(dt(2001, 1, 1), "1b")
    expected = 5.046514607521035
    assert abs(result - expected) < 1e-5
    assert ns.meta.convention == Convention.ActActISDA


def test_cache():
    ns = NelsonSiegelSvenssonCurve(
        dates=(dt(2000, 1, 1), dt(2030, 1, 1)),
        parameters=(0.01, 0.01, 0.05, 1.0, 0.05, 1.0),
    )
    ns.rate(dt(2001, 1, 1), "1b")
    assert dt(2001, 1, 1) in ns._cache

    old_state = ns._state
    ns._set_node_vector([1.0, 1.0, 1.0, 1.0], 0)
    assert ns._state != old_state
    assert dt(2001, 1, 1) not in ns._cache


def test_special_domain():
    ns = NelsonSiegelSvenssonCurve(
        dates=(dt(2000, 1, 1), dt(2030, 1, 1)),
        parameters=(0.01, 0.01, 0.05, 1.0, 0.05, 1.0),
    )
    assert ns[dt(2000, 1, 1)] == 1.0
    assert ns[dt(1999, 12, 31)] == 0.0


def test_getters():
    ns = NelsonSiegelSvenssonCurve(
        dates=(dt(2000, 1, 1), dt(2030, 1, 1)),
        parameters=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        id="v",
    )
    assert all(ns._get_node_vector() == np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    assert ns._get_node_vars() == ("v0", "v1", "v2", "v3", "v4", "v5")


def test_set_ad_order():
    ns = NelsonSiegelSvenssonCurve(
        dates=(dt(2000, 1, 1), dt(2030, 1, 1)),
        parameters=(0.01, 0.01, 0.05, 1.0, 0.05, 1.0),
        id="v",
        ad=2,
    )
    assert isinstance(ns.params[0], Dual2)
    ns._set_ad_order(2)  # does nothing
    assert isinstance(ns.params[0], Dual2)

    with pytest.raises(ValueError):
        ns._set_ad_order(3)
