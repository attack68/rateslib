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
from itertools import combinations

import numpy as np
import pytest
from matplotlib import pyplot as plt
from pandas import DataFrame, Index, Series
from pandas.testing import assert_frame_equal, assert_series_equal
from rateslib import default_context
from rateslib.curves import CompositeCurve, Curve, LineCurve
from rateslib.data.fixings import IRSSeries
from rateslib.default import NoInput
from rateslib.dual import Dual, Dual2, Variable, gradient
from rateslib.volatility import (
    IRSabrCube,
    IRSabrSmile,
)
from rateslib.volatility.ir.utils import _bilinear_interp
from rateslib.volatility.utils import _SabrSmileNodes


@pytest.mark.parametrize(
    ("h", "v", "expected"),
    [
        ((1, 1), (1, 1), 10),
        ((0.5, 0.5), (0.5, 0.5), 5.0),
        ((0.0, 0.0), (0.0, 0.0), 0.0),
        ((0.0, 0.5), (0.0, 0.0), 0.0),
        ((0.0, 0.0), (0.8, 0.4), 4.80),
        ((0.1, 0.2), (0.4, 0.5), 4.0 * 0.1 * 0.5 + 6.0 * 0.8 * 0.4 + 10.0 * 0.2 * 0.5),
    ],
)
def test_bilinear_interp(h, v, expected):
    result = _bilinear_interp(0.0, 4.0, 6.0, 10.0, h, v)
    assert abs(result - expected) < 1e-10


def test_numpy_ravel_for_dates_posix():
    a = np.array([[1, 1, 2], [3, 4, 5]])
    b = np.reshape(list(a.ravel()), (2, 3))
    assert np.all(a == b)


@pytest.fixture
def curve():
    return Curve(
        nodes={
            dt(2022, 3, 1): 1.00,
            dt(2032, 3, 31): 0.50,
        },
        interpolation="log_linear",
        id="v",
        convention="Act360",
        ad=1,
    )


class TestIRSabrSmile:
    @pytest.mark.parametrize(
        ("strike", "vol"),
        [
            (1.2034, 19.49),
            (1.2050, 19.47),
            (1.3395, 18.31),  # f == k
            (1.3620, 18.25),
            (1.5410, 18.89),
            (1.5449, 18.93),
        ],
    )
    def test_sabr_vol(self, strike, vol):
        # repeat the same test developed for FXSabrSmile
        irss = IRSabrSmile(
            nodes={
                "alpha": 0.17431060,
                "beta": 1.0,
                "rho": -0.11268306,
                "nu": 0.81694072,
            },
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            irs_series="eur_irs6",
            tenor="2y",
            id="vol",
        )
        result = irss.get_from_strike(k=strike, f=1.3395).vol
        assert abs(result - vol) < 1e-2

    def test_sabr_vol_plot(self):
        # repeat the same test developed for FXSabrSmile
        irss = IRSabrSmile(
            nodes={
                "alpha": 0.17431060,
                "beta": 1.0,
                "rho": -0.11268306,
                "nu": 0.81694072,
            },
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            irs_series="eur_irs6",
            tenor="2y",
            id="vol",
        )
        result = irss.plot(f=1.0)
        _x = result[2][0]._x
        _y = result[2][0]._y
        assert (_x[0], _y[0]) == (0.7524348790033292, 23.108399874378378)
        assert (_x[-1], _y[-1]) == (1.3743407823531082, 21.950871667495214)

    def test_sabr_vol_plot_fail(self):
        # repeat the same test developed for FXSabrSmile
        irss = IRSabrSmile(
            nodes={
                "alpha": 0.17431060,
                "beta": 1.0,
                "rho": -0.11268306,
                "nu": 0.81694072,
            },
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            irs_series="eur_irs6",
            tenor="2y",
            id="vol",
        )
        with pytest.raises(ValueError, match=r"`f` \(ATM-forward FX rate\) is required by"):
            irss.plot()

    @pytest.mark.parametrize(("k", "f"), [(1.34, 1.34), (1.33, 1.35), (1.35, 1.33)])
    def test_sabr_vol_finite_diff_first_order(self, k, f):
        # Test all of the first order gradients using finite diff, for the case when f != k and
        # when f == k, which is a branched calculation to handle a undefined point.
        irss = IRSabrSmile(
            nodes={
                "alpha": 0.17431060,
                "beta": 1.0,
                "rho": -0.11268306,
                "nu": 0.81694072,
            },
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            irs_series="eur_irs6",
            tenor="2y",
            id="vol",
            ad=2,
        )
        # F_0,T is stated in section 3.5.4 as 1.3395
        base = irss.get_from_strike(k=Dual2(k, ["k"], [], []), f=Dual2(f, ["f"], [], [])).vol

        a = irss.nodes.alpha
        p = irss.nodes.rho
        v = irss.nodes.nu

        def inc_(key1, inc1):
            in_ = {"k": k, "f": f, "alpha": a, "rho": p, "nu": v}
            in_[key1] += inc1

            irss._nodes = _SabrSmileNodes(
                _alpha=in_["alpha"], _beta=1.0, _rho=in_["rho"], _nu=in_["nu"]
            )
            _ = (
                irss._d_sabr_d_k_or_f(
                    Dual2(in_["k"], ["k"], [], []),
                    Dual2(in_["f"], ["f"], [], []),
                    dt(2002, 1, 1),
                    False,
                    1,
                )[0]
                * 100.0
            )

            # reset
            irss._nodes = _SabrSmileNodes(_alpha=a, _beta=1.0, _rho=p, _nu=v)
            return _

        for key in ["k", "f", "alpha", "rho", "nu"]:
            map_ = {"k": "k", "f": "f", "alpha": "vol0", "rho": "vol1", "nu": "vol2"}
            up_ = inc_(key, 1e-5)
            dw_ = inc_(key, -1e-5)
            assert abs((up_ - dw_) / 2e-5 - gradient(base, [map_[key]])[0]) < 1e-5

    @pytest.mark.parametrize(
        ("k", "f"), [(1.34, 1.34), (1.33, 1.35), (1.35, 1.33), (1.3399, 1.34), (1.34, 1.3401)]
    )
    @pytest.mark.parametrize("pair", list(combinations(["k", "f", "alpha", "rho", "nu"], 2)))
    def test_sabr_vol_cross_finite_diff_second_order(self, k, f, pair):
        # Test all of the second order cross gradients using finite diff,
        # for the case when f != k and
        # when f == k, which is a branched calculation to handle a undefined point.
        irss = IRSabrSmile(
            nodes={
                "alpha": 0.17431060,
                "beta": 1.0,
                "rho": -0.11268306,
                "nu": 0.81694072,
            },
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            irs_series="eur_irs6",
            tenor="2y",
            id="v",
            ad=2,
        )

        a = irss.nodes.alpha
        p = irss.nodes.rho
        v = irss.nodes.nu

        # F_0,T is stated in section 3.5.4 as 1.3395
        base = irss.get_from_strike(k=Dual2(k, ["k"], [], []), f=Dual2(f, ["f"], [], [])).vol

        def inc_(key1, key2, inc1, inc2):
            in_ = {"k": k, "f": f, "alpha": a, "rho": p, "nu": v}
            in_[key1] += inc1
            in_[key2] += inc2

            irss._nodes = _SabrSmileNodes(
                _alpha=in_["alpha"], _beta=1.0, _rho=in_["rho"], _nu=in_["nu"]
            )
            _ = (
                irss._d_sabr_d_k_or_f(
                    Dual2(in_["k"], ["k"], [], []),
                    Dual2(in_["f"], ["f"], [], []),
                    dt(2002, 1, 1),
                    False,
                    1,
                )[0]
                * 100.0
            )

            # reset
            irss._nodes = _SabrSmileNodes(_alpha=a, _beta=1.0, _rho=p, _nu=v)
            return _

        v_map = {"k": "k", "f": "f", "alpha": "v0", "rho": "v1", "nu": "v2"}

        upup = inc_(pair[0], pair[1], 1e-3, 1e-3)
        updown = inc_(pair[0], pair[1], 1e-3, -1e-3)
        downup = inc_(pair[0], pair[1], -1e-3, 1e-3)
        downdown = inc_(pair[0], pair[1], -1e-3, -1e-3)
        expected = (upup + downdown - updown - downup) / 4e-6
        result = gradient(base, [v_map[pair[0]], v_map[pair[1]]], order=2)[0][1]
        assert abs(result - expected) < 1e-2

    @pytest.mark.parametrize(
        ("k", "f"), [(1.34, 1.34), (1.33, 1.35), (1.35, 1.33), (1.3399, 1.34), (1.34, 1.3401)]
    )
    @pytest.mark.parametrize("var", ["k", "f", "alpha", "rho", "nu"])
    def test_sabr_vol_same_finite_diff_second_order(self, k, f, var):
        # Test all of the second order cross gradients using finite diff,
        # for the case when f != k and
        # when f == k, which is a branched calculation to handle a undefined point.
        irss = IRSabrSmile(
            nodes={
                "alpha": 0.17431060,
                "beta": 1.0,
                "rho": -0.11268306,
                "nu": 0.81694072,
            },
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            irs_series="eur_irs6",
            tenor="2y",
            id="v",
            ad=2,
        )

        a = irss.nodes.alpha
        p = irss.nodes.rho
        v = irss.nodes.nu

        # F_0,T is stated in section 3.5.4 as 1.3395
        base = irss.get_from_strike(k=Dual2(k, ["k"], [], []), f=Dual2(f, ["f"], [], [])).vol

        def inc_(key1, inc1):
            in_ = {"k": k, "f": f, "alpha": a, "rho": p, "nu": v}
            in_[key1] += inc1

            irss._nodes = _SabrSmileNodes(
                _alpha=in_["alpha"], _beta=1.0, _rho=in_["rho"], _nu=in_["nu"]
            )
            _ = (
                irss._d_sabr_d_k_or_f(
                    Dual2(in_["k"], ["k"], [], []),
                    Dual2(in_["f"], ["f"], [], []),
                    dt(2002, 1, 1),
                    False,
                    1,
                )[0]
                * 100.0
            )

            # reset
            irss._nodes = _SabrSmileNodes(_alpha=a, _beta=1.0, _rho=p, _nu=v)
            return _

        v_map = {"k": "k", "f": "f", "alpha": "v0", "rho": "v1", "nu": "v2"}

        up = inc_(var, 1e-4)
        down = inc_(var, -1e-4)
        expected = (up + down - 2 * base) / 1e-8
        result = gradient(base, [v_map[var]], order=2)[0][0]
        assert abs(result - expected) < 5e-3

    def test_sabr_vol_root_multi_duals_neighbourhood(self):
        # test the SABR function when regular arithmetic operations produce an undefined 0/0
        # value so AD has to be hard coded into the solution. This occurs when f == k.
        # test by comparing derivatives with those captured at a nearby valid point
        irss = IRSabrSmile(
            nodes={
                "alpha": 0.17431060,
                "beta": 1.0,
                "rho": -0.11268306,
                "nu": 0.81694072,
            },
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            irs_series="eur_irs6",
            tenor="2y",
            id="v",
            ad=2,
        )
        # F_0,T is stated in section 3.5.4 as 1.3395
        base = irss.get_from_strike(k=Dual2(1.34, ["k"], [], []), f=Dual2(1.34, ["f"], [], [])).vol
        comparison1 = irss.get_from_strike(
            k=Dual2(1.341, ["k"], [], []), f=Dual2(1.34, ["f"], [], [])
        ).vol

        assert np.all(abs(base.dual - comparison1.dual) < 1e-1)
        diff = base.dual2 - comparison1.dual2
        dual2 = abs(diff) < 5e-1
        assert np.all(dual2)

    @pytest.mark.parametrize("param", ["alpha", "beta", "rho", "nu"])
    def test_missing_param_raises(self, param):
        nodes = {
            "alpha": 0.17431060,
            "beta": 1.0,
            "rho": -0.11268306,
            "nu": 0.81694072,
        }
        nodes.pop(param)
        with pytest.raises(ValueError):
            IRSabrSmile(
                nodes=nodes,
                eval_date=dt(2001, 1, 1),
                expiry=dt(2002, 1, 1),
                irs_series="eur_irs6",
                tenor="2y",
                id="v",
                ad=2,
            )

    def test_non_iterable(self):
        irss = IRSabrSmile(
            nodes={
                "alpha": 0.17431060,
                "beta": 1.0,
                "rho": -0.11268306,
                "nu": 0.81694072,
            },
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            irs_series="eur_irs6",
            tenor="2y",
            id="v",
            ad=2,
        )
        with pytest.raises(TypeError):
            list(irss)

    def test_update_node_raises(self):
        irss = IRSabrSmile(
            nodes={
                "alpha": 0.17431060,
                "beta": 1.0,
                "rho": -0.11268306,
                "nu": 0.81694072,
            },
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            irs_series="eur_irs6",
            tenor="2y",
            id="v",
            ad=2,
        )
        with pytest.raises(KeyError, match="'bananas' is not in `nodes`."):
            irss.update_node("bananas", 12.0)

    def test_set_ad_order_raises(self):
        irss = IRSabrSmile(
            nodes={
                "alpha": 0.17431060,
                "beta": 1.0,
                "rho": -0.11268306,
                "nu": 0.81694072,
            },
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            irs_series="eur_irs6",
            tenor="2y",
            id="v",
            ad=2,
        )
        with pytest.raises(ValueError, match="`order` can only be in {0, 1, 2} "):
            irss._set_ad_order(12)

    def test_get_node_vars_and_vector(self):
        irss = IRSabrSmile(
            nodes={
                "alpha": 0.20,
                "beta": 1.0,
                "rho": -0.10,
                "nu": 0.80,
            },
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            irs_series="eur_irs6",
            tenor="2y",
            id="myid",
        )
        result = irss._get_node_vars()
        expected = ("myid0", "myid1", "myid2")
        assert result == expected

        result = irss._get_node_vector()
        expected = np.array([0.20, -0.1, 0.80])
        assert np.all(result == expected)

    def test_get_from_strike_expiry_raises(self):
        irss = IRSabrSmile(
            nodes={
                "alpha": 0.20,
                "beta": 1.0,
                "rho": -0.10,
                "nu": 0.80,
            },
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            irs_series="eur_irs6",
            tenor="2y",
            id="myid",
        )
        with pytest.raises(ValueError, match="`expiry` of VolSmile and OptionPeriod do not match"):
            irss.get_from_strike(k=1.0, f=1.0, expiry=dt(1999, 1, 1))

    @pytest.mark.parametrize("k", [1.2034, 1.2050, 1.3620, 1.5410, 1.5449])
    def test_get_from_strike_ad_2(self, k) -> None:
        # Use finite diff to validate the 2nd order AD of the SABR function in alpha and rho.
        irss = IRSabrSmile(
            nodes={
                "alpha": 0.20,
                "beta": 1.0,
                "rho": -0.10,
                "nu": 0.80,
            },
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            irs_series="eur_irs6",
            tenor="2y",
            id="vol",
            ad=2,
        )

        kwargs = dict(
            k=k,
            f=1.350,
        )
        pv00 = irss.get_from_strike(**kwargs)

        irss.update_node("alpha", 0.20 + 0.00001)
        irss.update_node("rho", -0.10 + 0.00001)
        pv11 = irss.get_from_strike(**kwargs)

        irss.update_node("alpha", 0.20 + 0.00001)
        irss.update_node("rho", -0.10 - 0.00001)
        pv1_1 = irss.get_from_strike(**kwargs)

        irss.update_node("alpha", 0.20 - 0.00001)
        irss.update_node("rho", -0.10 - 0.00001)
        pv_1_1 = irss.get_from_strike(**kwargs)

        irss.update_node("alpha", 0.20 - 0.00001)
        irss.update_node("rho", -0.10 + 0.00001)
        pv_11 = irss.get_from_strike(**kwargs)

        finite_diff = (pv11.vol + pv_1_1.vol - pv1_1.vol - pv_11.vol) * 1e10 / 4.0
        ad_grad = gradient(pv00.vol, ["vol0", "vol1"], 2)[0, 1]

        assert abs(finite_diff - ad_grad) < 1e-4

    @pytest.mark.parametrize(("k", "f"), [(1.34, 1.34), (1.33, 1.35), (1.35, 1.33)])
    def test_sabr_derivative_finite_diff_first_order(self, k, f):
        # Test all of the first order gradients using finite diff, for the case when f != k and
        # when f == k, which is a branched calculation to handle a undefined point.
        irss = IRSabrSmile(
            nodes={
                "alpha": 0.20,
                "beta": 1.0,
                "rho": -0.10,
                "nu": 0.80,
            },
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            irs_series="eur_irs6",
            tenor="2y",
            id="vol",
            ad=2,
        )
        t = dt(2002, 1, 1)
        base = irss._d_sabr_d_k_or_f(
            Dual2(k, ["k"], [1.0], []), Dual2(f, ["f"], [1.0], []), t, False, 1
        )[1]

        a = irss.nodes.alpha
        p = irss.nodes.rho
        v = irss.nodes.nu

        def inc_(key1, inc1):
            in_ = {"k": k, "f": f, "alpha": a, "rho": p, "nu": v}
            in_[key1] += inc1

            irss._nodes = _SabrSmileNodes(
                _alpha=in_["alpha"], _beta=1.0, _rho=in_["rho"], _nu=in_["nu"]
            )
            _ = irss._d_sabr_d_k_or_f(
                Dual2(in_["k"], ["k"], [], []),
                Dual2(in_["f"], ["f"], [], []),
                dt(2002, 1, 1),
                False,
                1,
            )[1]

            # reset
            irss._nodes = _SabrSmileNodes(_alpha=a, _beta=1.0, _rho=p, _nu=v)
            return _

        for key in ["k", "f", "alpha", "rho", "nu"]:
            map_ = {"k": "k", "f": "f", "alpha": "vol0", "rho": "vol1", "nu": "vol2"}
            up_ = inc_(key, 1e-5)
            dw_ = inc_(key, -1e-5)
            expected = (up_ - dw_) / 2e-5
            result = gradient(base, [map_[key]])[0]
            assert abs(expected - result) < 7e-3

    @pytest.mark.parametrize(
        ("k", "f"), [(1.34, 1.34), (1.33, 1.35), (1.35, 1.33), (1.3395, 1.34), (1.34, 1.3405)]
    )
    @pytest.mark.parametrize("pair", list(combinations(["k", "f", "alpha", "rho", "nu"], 2)))
    def test_sabr_derivative_cross_finite_diff_second_order(self, k, f, pair):
        # Test all of the second order cross gradients using finite diff,
        # for the case when f != k and
        # when f == k, which is a branched calculation to handle a undefined point.
        irss = IRSabrSmile(
            nodes={
                "alpha": 0.20,
                "beta": 1.0,
                "rho": -0.10,
                "nu": 0.80,
            },
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            irs_series="eur_irs6",
            tenor="2y",
            id="v",
            ad=2,
        )

        a = irss.nodes.alpha
        p = irss.nodes.rho
        v = irss.nodes.nu

        # F_0,T is stated in section 3.5.4 as 1.3395
        base = irss._d_sabr_d_k_or_f(
            Dual2(k, ["k"], [], []), Dual2(f, ["f"], [], []), dt(2002, 1, 1), False, 1
        )[1]

        def inc_(key1, key2, inc1, inc2):
            in_ = {"k": k, "f": f, "alpha": a, "rho": p, "nu": v}
            in_[key1] += inc1
            in_[key2] += inc2

            irss._nodes = _SabrSmileNodes(
                _alpha=in_["alpha"], _beta=1.0, _rho=in_["rho"], _nu=in_["nu"]
            )
            _ = irss._d_sabr_d_k_or_f(
                Dual2(in_["k"], ["k"], [], []),
                Dual2(in_["f"], ["f"], [], []),
                dt(2002, 1, 1),
                False,
                1,
            )[1]

            # reset
            irss._nodes = _SabrSmileNodes(_alpha=a, _beta=1.0, _rho=p, _nu=v)
            return _

        v_map = {"k": "k", "f": "f", "alpha": "v0", "rho": "v1", "nu": "v2"}

        upup = inc_(pair[0], pair[1], 1e-3, 1e-3)
        updown = inc_(pair[0], pair[1], 1e-3, -1e-3)
        downup = inc_(pair[0], pair[1], -1e-3, 1e-3)
        downdown = inc_(pair[0], pair[1], -1e-3, -1e-3)
        expected = (upup + downdown - updown - downup) / 4e-6
        result = gradient(base, [v_map[pair[0]], v_map[pair[1]]], order=2)[0][1]
        assert abs(result - expected) < 5e-3

    @pytest.mark.parametrize(
        ("k", "f"),
        [(1.34, 1.34), (1.33, 1.35), (1.35, 1.33), (1.3395, 1.34), (1.34, 1.3405)],
    )
    @pytest.mark.parametrize("var", ["k", "f", "alpha", "rho", "nu"])
    def test_sabr_derivative_same_finite_diff_second_order(self, k, f, var):
        # Test all of the second order cross gradients using finite diff,
        # for the case when f != k and
        # when f == k, which is a branched calculation to handle a undefined point.
        irss = IRSabrSmile(
            nodes={
                "alpha": 0.20,
                "beta": 1.0,
                "rho": -0.10,
                "nu": 0.80,
            },
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            irs_series="eur_irs6",
            tenor="2y",
            id="v",
            ad=2,
        )

        a = irss.nodes.alpha
        p = irss.nodes.rho
        v = irss.nodes.nu

        # F_0,T is stated in section 3.5.4 as 1.3395
        base = irss._d_sabr_d_k_or_f(
            Dual2(k, ["k"], [], []), Dual2(f, ["f"], [], []), dt(2002, 1, 1), False, 1
        )[1]

        def inc_(key1, inc1):
            k_ = k
            f_ = f
            if key1 == "k":
                k_ = k + inc1
            elif key1 == "f":
                f_ = f + inc1
            else:
                irss.update_node(key1, getattr(irss.nodes, key1) + inc1)
                # irss.nodes[key1] = irss.nodes[key1] + inc1

            _ = irss._d_sabr_d_k_or_f(
                Dual2(k_, ["k"], [], []), Dual2(f_, ["f"], [], []), dt(2002, 1, 1), False, 1
            )[1]

            irss._nodes = _SabrSmileNodes(_alpha=a, _beta=1.0, _rho=p, _nu=v)
            return _

        v_map = {"k": "k", "f": "f", "alpha": "v0", "rho": "v1", "nu": "v2"}

        up = inc_(var, 1e-3)
        down = inc_(var, -1e-3)
        expected = (up + down - 2 * base) / 1e-6
        result = gradient(base, [v_map[var]], order=2)[0][0]
        assert abs(result - expected) < 3e-3

    def test_sabr_derivative_root_multi_duals_neighbourhood(self):
        # test the SABR function when regular arithmetic operations produce an undefined 0/0
        # value so AD has to be hard coded into the solution. This occurs when f == k.
        # test by comparing derivatives with those captured at a nearby valid point
        irss = IRSabrSmile(
            nodes={
                "alpha": 0.20,
                "beta": 1.0,
                "rho": -0.10,
                "nu": 0.80,
            },
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            irs_series="eur_irs6",
            tenor="2y",
            id="v",
            ad=2,
        )
        # F_0,T is stated in section 3.5.4 as 1.3395
        base = irss._d_sabr_d_k_or_f(
            Dual2(1.34, ["k"], [], []), Dual2(1.34, ["f"], [], []), dt(2002, 1, 1), False, 1
        )[1]
        comparison1 = irss._d_sabr_d_k_or_f(
            Dual2(1.341, ["k"], [], []), Dual2(1.34, ["f"], [], []), dt(2002, 1, 1), False, 1
        )[1]

        assert np.all(abs(base.dual - comparison1.dual) < 5e-3)
        diff = base.dual2 - comparison1.dual2
        dual2 = abs(diff) < 3e-2
        assert np.all(dual2)

    #
    # def test_plot_domain(self):
    #     ss = FXSabrSmile(
    #         eval_date=dt(2024, 5, 28),
    #         expiry=dt(2054, 5, 28),
    #         nodes={"alpha": 0.02, "beta": 1.0, "rho": 0.01, "nu": 0.05},
    #     )
    #     ax, fig, lines = ss.plot(f=1.60)
    #     assert abs(lines[0]._x[0] - 1.3427) < 1e-4
    #     assert abs(lines[0]._x[-1] - 1.9299) < 1e-4
    #     assert abs(lines[0]._y[0] - 2.0698) < 1e-4
    #     assert abs(lines[0]._y[-1] - 2.0865) < 1e-4
    #

    #
    # def test_solver_variable_numbers(self):
    #     from rateslib import IRS, FXBrokerFly, FXCall, FXRiskReversal, FXStraddle, FXSwap, Solver
    #
    #     usdusd = Curve({dt(2024, 5, 7): 1.0, dt(2024, 5, 30): 1.0}, calendar="nyc", id="usdusd")
    #     eureur = Curve({dt(2024, 5, 7): 1.0, dt(2024, 5, 30): 1.0}, calendar="tgt", id="eureur")
    #     eurusd = Curve({dt(2024, 5, 7): 1.0, dt(2024, 5, 30): 1.0}, id="eurusd")
    #
    #     # Create an FX Forward market with spot FX rate data
    #     fxr = FXRates({"eurusd": 1.0760}, settlement=dt(2024, 5, 9))
    #     fxf = FXForwards(
    #         fx_rates=fxr,
    #         fx_curves={"eureur": eureur, "usdusd": usdusd, "eurusd": eurusd},
    #     )
    #
    #     pre_solver = Solver(
    #         curves=[eureur, eurusd, usdusd],
    #         instruments=[
    #             IRS(dt(2024, 5, 9), "3W", spec="eur_irs", curves="eureur"),
    #             IRS(dt(2024, 5, 9), "3W", spec="usd_irs", curves="usdusd"),
    #             FXSwap(
    #                 dt(2024, 5, 9), "3W", pair="eurusd", curves=[None, "eurusd", None, "usdusd"]
    #             ),
    #         ],
    #         s=[3.90, 5.32, 8.85],
    #         fx=fxf,
    #         id="rates_sv",
    #     )
    #
    #     dv_smile = FXSabrSmile(
    #         nodes={"alpha": 0.05, "beta": 1.0, "rho": 0.01, "nu": 0.03},
    #         eval_date=dt(2024, 5, 7),
    #         expiry=dt(2024, 5, 28),
    #         id="eurusd_3w_smile",
    #         pair="eurusd",
    #     )
    #     option_args = dict(
    #         pair="eurusd",
    #         expiry=dt(2024, 5, 28),
    #         calendar="tgt|fed",
    #         delta_type="spot",
    #         curves=["eurusd", "usdusd"],
    #         vol="eurusd_3w_smile",
    #     )
    #
    #     dv_solver = Solver(
    #         pre_solvers=[pre_solver],
    #         curves=[dv_smile],
    #         instruments=[
    #             FXStraddle(strike="atm_delta", **option_args),
    #             FXRiskReversal(strike=("-25d", "25d"), **option_args),
    #             FXRiskReversal(strike=("-10d", "10d"), **option_args),
    #             FXBrokerFly(strike=(("-25d", "25d"), "atm_delta"), **option_args),
    #             FXBrokerFly(strike=(("-10d", "10d"), "atm_delta"), **option_args),
    #         ],
    #         s=[5.493, -0.157, -0.289, 0.071, 0.238],
    #         fx=fxf,
    #         id="dv_solver",
    #     )
    #
    #     fc = FXCall(
    #         expiry=dt(2024, 5, 28),
    #         pair="eurusd",
    #         strike=1.07,
    #         notional=100e6,
    #         curves=["eurusd", "usdusd"],
    #         vol="eurusd_3w_smile",
    #         premium=98.216647 * 1e8 / 1e4,
    #         premium_ccy="usd",
    #         delta_type="spot",
    #     )
    #     fc.delta(solver=dv_solver)
    #
    @pytest.mark.parametrize("a", [0.02, 0.06])
    @pytest.mark.parametrize("b", [0.0, 0.4, 0.65, 1.0])
    @pytest.mark.parametrize("p", [-0.1, 0.1])
    @pytest.mark.parametrize("v", [0.05, 0.15])
    @pytest.mark.parametrize("k", [1.05, 1.25, 1.6])
    def test_sabr_function_values(self, a, b, p, v, k):
        irss = IRSabrSmile(
            nodes={
                "alpha": a,
                "beta": b,
                "rho": p,
                "nu": v,
            },
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            irs_series="eur_irs6",
            tenor="2y",
            id="v",
            ad=2,
        )

        # this code is taken from PySabr, another library implementing SABR.
        # it is used as a benchmark
        def _x(rho, z):
            """Return function x used in Hagan's 2002 SABR lognormal vol expansion."""
            a = (1 - 2 * rho * z + z**2) ** 0.5 + z - rho
            b = 1 - rho
            return np.log(a / b)

        def lognormal_vol(k, f, t, alpha, beta, rho, volvol):
            """
            Hagan's 2002 SABR lognormal vol expansion.

            The strike k can be a scalar or an array, the function will return an array
            of lognormal vols.
            """
            # Negative strikes or forwards
            if k <= 0 or f <= 0:
                return 0.0
            eps = 1e-07
            logfk = np.log(f / k)
            fkbeta = (f * k) ** (1 - beta)
            a = (1 - beta) ** 2 * alpha**2 / (24 * fkbeta)
            b = 0.25 * rho * beta * volvol * alpha / fkbeta**0.5
            c = (2 - 3 * rho**2) * volvol**2 / 24
            d = fkbeta**0.5
            v = (1 - beta) ** 2 * logfk**2 / 24
            w = (1 - beta) ** 4 * logfk**4 / 1920
            z = volvol * fkbeta**0.5 * logfk / alpha
            # if |z| > eps
            if abs(z) > eps:
                vz = alpha * z * (1 + (a + b + c) * t) / (d * (1 + v + w) * _x(rho, z))
                return vz
            # if |z| <= eps
            else:
                v0 = alpha * (1 + (a + b + c) * t) / (d * (1 + v + w))
                return v0

        expected = lognormal_vol(k, 1.25, 1.0, a, b, p, v)
        result = irss.get_from_strike(k=k, f=1.25).vol / 100.0

        assert abs(result - expected) < 1e-4

    def test_init_raises_key(self):
        with pytest.raises(
            ValueError, match=r"'nu' is a required SABR parameter that must be inclu"
        ):
            IRSabrSmile(
                nodes={
                    "alpha": 0.05,
                    "beta": -0.03,
                    "rho": 0.1,
                    "bad": 0.1,
                },
                eval_date=dt(2001, 1, 1),
                expiry=dt(2002, 1, 1),
                irs_series="eur_irs6",
                tenor="2y",
                id="v",
                ad=2,
            )

    def test_attributes(self):
        irss = IRSabrSmile(
            nodes={
                "alpha": 0.05,
                "beta": -0.03,
                "rho": 0.1,
                "nu": 0.1,
            },
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            irs_series="eur_irs6",
            tenor="2y",
            id="v",
            ad=2,
        )
        assert irss._n == 4

    def test_get_from_strike_with_curves(self):
        curve = Curve({dt(2001, 1, 1): 1.0, dt(2003, 1, 1): 0.94})
        irss = IRSabrSmile(
            nodes={
                "alpha": 0.05,
                "beta": -0.03,
                "rho": 0.1,
                "nu": 0.1,
            },
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            irs_series="eur_irs6",
            tenor="2y",
            id="v",
        )
        result = irss.get_from_strike(k=3.0, curves=[curve])
        assert abs(result.f - 3.142139380) < 1e-6
        assert abs(result.vol - 1.575277) < 1e-4

    def test_set_node_vector(self):
        irss = IRSabrSmile(
            nodes={
                "alpha": 0.05,
                "beta": -0.03,
                "rho": 0.1,
                "nu": 0.1,
            },
            eval_date=dt(2001, 1, 1),
            expiry=dt(2002, 1, 1),
            irs_series="eur_irs6",
            tenor="2y",
            ad=2,
            id="v",
        )
        irss._set_node_vector(np.array([1.0, 2.0, 3.0]), ad=1)
        assert irss.nodes.alpha == Dual(1.0, ["v0"], [])
        assert irss.nodes.rho == Dual(2.0, ["v1"], [])
        assert irss.nodes.nu == Dual(3.0, ["v2"], [])


class TestIRSabrCube:
    def test_init(self):
        IRSabrCube(
            eval_date=dt(2026, 2, 16),
            expiries=["1m", "3m"],
            tenors=["1Y", "2y", "3y"],
            irs_series="usd_irs",
            id="usd_ir_vol",
            beta=0.5,
            alphas=np.array([[0.1, 0.2, 0.3], [0.11, 0.12, 0.13]]),
            rhos=np.array([[0.1, 0.2, 0.3], [0.11, 0.12, 0.13]]),
            nus=np.array([[0.1, 0.2, 0.3], [0.11, 0.12, 0.13]]),
        )
        pass

    @pytest.mark.parametrize(("ad", "klass"), [(1, Dual), (2, Dual2)])
    def test_constructed_sabr_smile_vars(self, ad, klass):
        irsc = IRSabrCube(
            eval_date=dt(2026, 2, 20),
            expiries=["1m", "3m"],
            tenors=["2y", "5y"],
            irs_series="usd_irs",
            beta=0.5,
            alphas=0.05,
            rhos=-0.01,
            nus=0.01,
            ad=ad,
            id="my-c",
        )
        _ = irsc.get_from_strike(k=1.0, f=1.02, expiry=dt(2026, 3, 30), tenor=dt(2028, 8, 12))
        smile = irsc._cache[(dt(2026, 3, 30), dt(2028, 8, 12))]
        assert smile.nodes.alpha.vars == ["my-c_a_0", "my-c_a_1", "my-c_a_2", "my-c_a_3"]
        assert smile.nodes.rho.vars == ["my-c_p_0", "my-c_p_1", "my-c_p_2", "my-c_p_3"]
        assert smile.nodes.nu.vars == ["my-c_v_0", "my-c_v_1", "my-c_v_2", "my-c_v_3"]
        assert isinstance(smile.nodes.alpha, klass)

    @pytest.mark.parametrize(
        ("expiry", "tenor", "expected"),
        [
            # tests on a node directly
            (dt(2001, 1, 1), dt(2002, 1, 1), (0.1, 1.0, 10.0)),
            (dt(2002, 1, 1), dt(2003, 1, 1), (0.3, 3.0, 30.0)),
            (dt(2001, 1, 1), dt(2003, 1, 1), (0.2, 2.0, 20.0)),
            (dt(2002, 1, 1), dt(2004, 1, 1), (0.4, 4.0, 40.0)),
            # test within bounds
            (
                dt(2001, 4, 1),
                dt(2002, 7, 1),
                (0.17424657534246576, 1.7424657534246577, 17.424657534246577),
            ),
            (
                dt(2001, 4, 1),
                dt(2003, 1, 1),
                (0.22465753424657536, 2.2465753424657535, 22.46575342465753),
            ),
            (
                dt(2001, 10, 1),
                dt(2003, 1, 1),
                (0.27479452054794523, 2.747945205479452, 27.47945205479452),
            ),
            (
                dt(2001, 10, 1),
                dt(2003, 7, 1),
                (0.32438356164383564, 3.243835616438356, 32.43835616438356),
            ),
            # test out of bounds
            (dt(2000, 7, 1), dt(2001, 1, 1), (0.1, 1.0, 10.0)),  # 6m6m
            (
                dt(2000, 7, 1),
                dt(2002, 1, 1),
                (0.1504109589041096, 1.504109589041096, 15.04109589041096),
            ),  # 6m18m
            (dt(2000, 7, 1), dt(2003, 7, 1), (0.2, 2.0, 20.0)),  # 6m3y
            (
                dt(2001, 7, 1),
                dt(2002, 1, 1),
                (0.1991780821917808, 1.9917808219178081, 19.91780821917808),
            ),  # 18m6m
            (
                dt(2001, 7, 1),
                dt(2004, 7, 1),
                (0.2991780821917808, 2.991780821917808, 29.91780821917808),
            ),  # 18m3y
            (dt(2003, 1, 1), dt(2003, 7, 1), (0.30, 3.0, 30.0)),  # 3y6m
            (
                dt(2003, 1, 1),
                dt(2004, 7, 1),
                (0.34986301369863015, 3.4986301369863018, 34.986301369863014),
            ),  # 3y18m
            (dt(2003, 1, 1), dt(2006, 1, 1), (0.4, 4.0, 40.0)),  # 3y3y
        ],
    )
    def test_interpolation_boundaries(self, expiry, tenor, expected):
        # test that the SabrCube will interpolate the parameters if the expiry and tenors are
        # - exactly falling on node dates
        # - some elements within the node-mesh
        # - some elements outside the node-mesh which are mapped to nearest components.
        irsc = IRSabrCube(
            eval_date=dt(2000, 1, 1),
            expiries=["1y", "2y"],
            tenors=["1y", "2y"],
            irs_series=IRSSeries(
                currency="usd",
                settle=0,
                frequency="A",
                convention="Act360",
                calendar="all",
                leg2_fixing_method="ibor(2)",
            ),
            beta=0.5,
            alphas=np.array([[0.1, 0.2], [0.3, 0.4]]),
            rhos=np.array([[1.0, 2.0], [3.0, 4.0]]),
            nus=np.array([[10.0, 20.0], [30.0, 40.0]]),
            id="my-c",
        )
        result = irsc._bilinear_interpolation(expiry=expiry, tenor=tenor)
        assert result == expected

    @pytest.mark.parametrize(
        ("expiry", "tenor", "expected"),
        [
            (dt(2000, 7, 1), dt(2001, 1, 1), (0.1, 1.0, 10.0)),
            (dt(2000, 7, 1), dt(2001, 7, 1), (0.1, 1.0, 10.0)),
            (
                dt(2000, 7, 1),
                dt(2002, 1, 1),
                (0.1504109589041096, 1.504109589041096, 15.04109589041096),
            ),
            (dt(2000, 7, 1), dt(2003, 7, 1), (0.2, 2.0, 20.0)),
            (dt(2001, 1, 1), dt(2001, 7, 1), (0.1, 1.0, 10.0)),
            (dt(2001, 1, 1), dt(2002, 1, 1), (0.1, 1.0, 10.0)),
            (
                dt(2001, 1, 1),
                dt(2002, 7, 1),
                (0.1495890410958904, 1.495890410958904, 14.95890410958904),
            ),
            (dt(2001, 1, 1), dt(2003, 7, 1), (0.2, 2.0, 20.0)),
            (dt(2002, 1, 1), dt(2002, 7, 1), (0.1, 1.0, 10.0)),
            (dt(2002, 1, 1), dt(2003, 1, 1), (0.1, 1.0, 10.0)),
            (
                dt(2002, 1, 1),
                dt(2003, 7, 1),
                (0.1495890410958904, 1.495890410958904, 14.95890410958904),
            ),
            (dt(2002, 1, 1), dt(2004, 7, 1), (0.2, 2.0, 20.0)),
        ],
    )
    def test_interpolation_single_expiry(self, expiry, tenor, expected):
        # test that the SabrCube will interpolate the parameters if the expiry and tenors are
        # - exactly falling on node dates
        # - some elements within the node-mesh
        # - some elements outside the node-mesh which are mapped to nearest components.
        irsc = IRSabrCube(
            eval_date=dt(2000, 1, 1),
            expiries=["1y"],
            tenors=["1y", "2y"],
            irs_series=IRSSeries(
                currency="usd",
                settle=0,
                frequency="A",
                convention="Act360",
                calendar="all",
                leg2_fixing_method="ibor(2)",
            ),
            beta=0.5,
            alphas=np.array([[0.1, 0.2]]),
            rhos=np.array([[1.0, 2.0]]),
            nus=np.array([[10.0, 20.0]]),
            id="my-c",
        )
        result = irsc._bilinear_interpolation(expiry=expiry, tenor=tenor)
        assert result == expected

    @pytest.mark.parametrize(
        ("expiry", "tenor", "expected"),
        [
            (dt(2000, 7, 1), dt(2001, 1, 1), (0.1, 1.0, 10.0)),
            (dt(2000, 7, 1), dt(2001, 7, 1), (0.1, 1.0, 10.0)),
            (dt(2000, 7, 1), dt(2002, 1, 1), (0.1, 1.0, 10.0)),
            (dt(2001, 1, 1), dt(2001, 7, 1), (0.1, 1.0, 10.0)),
            (dt(2001, 1, 1), dt(2002, 1, 1), (0.1, 1.0, 10.0)),
            (dt(2001, 1, 1), dt(2002, 7, 1), (0.1, 1.0, 10.0)),
            (
                dt(2001, 7, 1),
                dt(2002, 1, 1),
                (0.1495890410958904, 1.495890410958904, 14.95890410958904),
            ),
            (
                dt(2001, 7, 1),
                dt(2002, 7, 1),
                (0.1495890410958904, 1.495890410958904, 14.95890410958904),
            ),
            (
                dt(2001, 7, 1),
                dt(2003, 1, 1),
                (0.1495890410958904, 1.495890410958904, 14.95890410958904),
            ),
            (dt(2002, 7, 1), dt(2003, 1, 1), (0.2, 2.0, 20.0)),
            (dt(2002, 7, 1), dt(2003, 7, 1), (0.2, 2.0, 20.0)),
            (dt(2002, 7, 1), dt(2004, 7, 1), (0.2, 2.0, 20.0)),
        ],
    )
    def test_interpolation_single_tenor(self, expiry, tenor, expected):
        # test that the SabrCube will interpolate the parameters if the expiry and tenors are
        # - exactly falling on node dates
        # - some elements within the node-mesh
        # - some elements outside the node-mesh which are mapped to nearest components.
        irsc = IRSabrCube(
            eval_date=dt(2000, 1, 1),
            expiries=["1y", "2y"],
            tenors=["1y"],
            irs_series=IRSSeries(
                currency="usd",
                settle=0,
                frequency="A",
                convention="Act360",
                calendar="all",
                leg2_fixing_method="ibor(2)",
            ),
            beta=0.5,
            alphas=np.array([[0.1], [0.2]]),
            rhos=np.array([[1.0], [2.0]]),
            nus=np.array([[10.0], [20.0]]),
            id="my-c",
        )
        result = irsc._bilinear_interpolation(expiry=expiry, tenor=tenor)
        assert result == expected

    def test_alphas(self):
        irsc = IRSabrCube(
            eval_date=dt(2026, 2, 16),
            expiries=["1m", "3m"],
            tenors=["1Y", "2Y"],
            irs_series="usd_irs",
            id="usd_ir_vol",
            beta=0.5,
            alphas=np.array([[0.1, 0.2], [0.11, 0.12]]),
            rhos=np.array([[0.1, 0.3], [0.11, 0.12]]),
            nus=np.array([[0.1, 0.4], [0.11, 0.12]]),
        )
        expected = DataFrame(
            index=Index(["1m", "3m"], name="expiry"),
            columns=Index(["1Y", "2Y"], name="tenor"),
            data=[[0.1, 0.2], [0.11, 0.12]],
            dtype=object,
        )
        assert_frame_equal(expected, irsc.alpha)
        expected = DataFrame(
            index=Index(["1m", "3m"], name="expiry"),
            columns=Index(["1Y", "2Y"], name="tenor"),
            data=[[0.1, 0.3], [0.11, 0.12]],
            dtype=object,
        )
        assert_frame_equal(expected, irsc.rho)
        expected = DataFrame(
            index=Index(["1m", "3m"], name="expiry"),
            columns=Index(["1Y", "2Y"], name="tenor"),
            data=[[0.1, 0.4], [0.11, 0.12]],
            dtype=object,
        )
        assert_frame_equal(expected, irsc.nu)
        assert irsc._n == 12

    def test_cache(self):
        irsc = IRSabrCube(
            eval_date=dt(2026, 2, 16),
            expiries=["1m", "3m"],
            tenors=["1Y", "2Y"],
            irs_series="usd_irs",
            id="usd_ir_vol",
            beta=0.5,
            alphas=np.array([[0.1, 0.2], [0.11, 0.12]]),
            rhos=np.array([[0.1, 0.3], [0.11, 0.12]]),
            nus=np.array([[0.1, 0.4], [0.11, 0.12]]),
        )
        irsc.get_from_strike(k=1.02, f=1.04, expiry=dt(2026, 3, 30), tenor=dt(2027, 8, 12))
        assert (dt(2026, 3, 30), dt(2027, 8, 12)) in irsc._cache


class TestStateAndCache:
    @pytest.mark.parametrize(
        "obj",
        [
            IRSabrSmile(
                nodes={
                    "alpha": 0.1,
                    "beta": 0.5,
                    "rho": -0.05,
                    "nu": 0.1,
                },
                eval_date=dt(2001, 1, 1),
                expiry=dt(2002, 1, 1),
                irs_series="eur_irs6",
                tenor="2y",
                id="v",
                ad=2,
            ),
            IRSabrCube(
                eval_date=dt(2026, 2, 16),
                expiries=["1m", "3m"],
                tenors=["1Y", "2y", "3y"],
                irs_series="usd_irs",
                id="usd_ir_vol",
                beta=0.5,
                alphas=np.array([[0.1, 0.2, 0.3], [0.11, 0.12, 0.13]]),
                rhos=np.array([[0.1, 0.2, 0.3], [0.11, 0.12, 0.13]]),
                nus=np.array([[0.1, 0.2, 0.3], [0.11, 0.12, 0.13]]),
            ),
        ],
    )
    @pytest.mark.parametrize(("method", "args"), [("_set_ad_order", (1,))])
    def test_method_does_not_change_state(self, obj, method, args):
        before = obj._state
        getattr(obj, method)(*args)
        after = obj._state
        assert before == after

    @pytest.mark.parametrize(
        "obj",
        [
            IRSabrSmile(
                nodes={
                    "alpha": 0.1,
                    "beta": 0.5,
                    "rho": -0.05,
                    "nu": 0.1,
                },
                eval_date=dt(2001, 1, 1),
                expiry=dt(2002, 1, 1),
                irs_series="eur_irs6",
                tenor="2y",
                id="v",
                ad=2,
            ),
        ],
    )
    @pytest.mark.parametrize(
        ("method", "args"),
        [
            ("_set_node_vector", ([0.99, 0.98, 0.99], 1)),
            ("update_node", ("alpha", 0.98)),
        ],
    )
    def test_method_changes_state(self, obj, method, args):
        before = obj._state
        getattr(obj, method)(*args)
        after = obj._state
        assert before != after

    @pytest.mark.parametrize(
        "curve",
        [
            IRSabrSmile(
                nodes={
                    "alpha": 0.1,
                    "beta": 0.5,
                    "rho": -0.05,
                    "nu": 0.1,
                },
                eval_date=dt(2001, 1, 1),
                expiry=dt(2002, 1, 1),
                irs_series="eur_irs6",
                tenor="2y",
                id="v",
                ad=2,
            ),
            IRSabrCube(
                eval_date=dt(2026, 2, 16),
                expiries=["1m"],
                tenors=["1Y"],
                irs_series="usd_irs",
                id="usd_ir_vol",
                beta=0.5,
                alphas=np.array([[0.1]]),
                rhos=np.array([[0.2]]),
                nus=np.array([[0.3]]),
            ),
        ],
    )
    @pytest.mark.parametrize(
        ("method", "args"),
        [
            ("_set_node_vector", ([0.99, 0.98, 0.99], 1)),
            ("update_node", ("alpha", 0.98)),
        ],
    )
    def test_method_changes_state_sabr(self, curve, method, args):
        before = curve._state
        getattr(curve, method)(*args)
        after = curve._state
        assert before != after

    #
    #     def test_populate_cache(self):
    #         # objects have yet to implement cache handling
    #         pass
    #
    #     def test_method_clears_cache(self):
    #         # objects have yet to implement cache handling
    #         pass
    #
    @pytest.mark.parametrize(
        ("method", "args"),
        [
            ("_set_node_vector", ([0.99, 0.98, 1.0], 1)),
            ("_set_ad_order", (2,)),
        ],
    )
    def test_surface_clear_cache(self, method, args):
        surf = IRSabrCube(
            eval_date=dt(2026, 2, 16),
            expiries=["1m"],
            tenors=["1Y"],
            irs_series="usd_irs",
            id="usd_ir_vol",
            beta=0.5,
            alphas=np.array([[0.1]]),
            rhos=np.array([[0.2]]),
            nus=np.array([[0.3]]),
        )
        surf.get_from_strike(f=1.0, k=1.01, expiry=dt(2026, 3, 1), tenor=dt(2027, 3, 1))
        assert (dt(2026, 3, 1), dt(2027, 3, 1)) in surf._cache

        getattr(surf, method)(*args)
        assert len(surf._cache) == 0


#
#     @pytest.mark.parametrize(
#         ("method", "args"),
#         [
#             ("get_from_strike", (1.0, 1.0, dt(2000, 5, 3), NoInput(0))),
#             ("_get_index", (0.9, dt(2000, 5, 3))),
#             ("get_smile", (dt(2000, 5, 3),)),
#         ],
#     )
#     def test_surface_populate_cache(self, method, args):
#         surf = FXDeltaVolSurface(
#             expiries=[dt(2000, 1, 1), dt(2001, 1, 1)],
#             delta_indexes=[0.5],
#             node_values=[[10.0], [9.0]],
#             eval_date=dt(1999, 1, 1),
#             delta_type="forward",
#         )
#         before = surf._cache_len
#         getattr(surf, method)(*args)
#         assert surf._cache_len == before + 1
#
#     @pytest.mark.parametrize(
#         ("method", "args"),
#         [
#             ("_set_node_vector", ([0.99, 0.98, 0.99, 0.99, 0.98, 0.99], 1)),
#         ],
#     )
#     @pytest.mark.parametrize(
#         "surface",
#         [
#             FXDeltaVolSurface(
#                 expiries=[dt(2000, 1, 1), dt(2001, 1, 1)],
#                 delta_indexes=[0.25, 0.5, 0.75],
#                 node_values=[[10.0, 9.0, 8.0], [9.0, 8.0, 7.0]],
#                 eval_date=dt(1999, 1, 1),
#                 delta_type="forward",
#             ),
#             FXSabrSurface(
#                 expiries=[dt(2000, 1, 1), dt(2001, 1, 1)],
#                 node_values=[[10.0, 1.0, 8.0, 9.0], [9.0, 1.0, 8.0, 7.0]],
#                 eval_date=dt(1999, 1, 1),
#             ),
#         ],
#     )
#     def test_surface_change_state(self, method, args, surface):
#         pre_state = surface._state
#         getattr(surface, method)(*args)
#         assert surface._state != pre_state
#
#     @pytest.mark.parametrize(
#         ("method", "args"),
#         [
#             ("_set_ad_order", (2,)),
#         ],
#     )
#     @pytest.mark.parametrize(
#         "surface",
#         [
#             FXDeltaVolSurface(
#                 expiries=[dt(2000, 1, 1), dt(2001, 1, 1)],
#                 delta_indexes=[0.25, 0.5, 0.75],
#                 node_values=[[10.0, 9.0, 8.0], [9.0, 8.0, 7.0]],
#                 eval_date=dt(1999, 1, 1),
#                 delta_type="forward",
#             ),
#             FXSabrSurface(
#                 expiries=[dt(2000, 1, 1), dt(2001, 1, 1)],
#                 node_values=[[10.0, 1.0, 8.0, 9.0], [9.0, 1.0, 8.0, 7.0]],
#                 eval_date=dt(1999, 1, 1),
#             ),
#         ],
#     )
#     def test_surface_maintain_state(self, method, args, surface):
#         pre_state = surface._state
#         getattr(surface, method)(*args)
#         assert surface._state == pre_state
#
#     def test_surface_validate_states(self):
#         # test the get_smile method validates the states after a mutation
#         surf = FXDeltaVolSurface(
#             expiries=[dt(2000, 1, 1), dt(2001, 1, 1)],
#             delta_indexes=[0.5],
#             node_values=[[10.0], [9.0]],
#             eval_date=dt(1999, 1, 1),
#             delta_type="forward",
#         )
#         pre_state = surf._state
#         surf.smiles[0].update_node(0.5, 11.0)
#         surf.get_smile(dt(2000, 1, 9))
#         post_state = surf._state
#         assert pre_state != post_state  # validate states has been run and updated the state.
#
#     @pytest.mark.parametrize(
#         "smile",
#         [
#             FXDeltaVolSmile(
#                 nodes={0.25: 10.0, 0.5: 10.0, 0.75: 11.0},
#                 delta_type="forward",
#                 eval_date=dt(2023, 3, 16),
#                 expiry=dt(2023, 6, 16),
#                 id="vol",
#             ),
#             FXSabrSmile(
#                 nodes={
#                     "alpha": 0.17431060,
#                     "beta": 1.0,
#                     "rho": -0.11268306,
#                     "nu": 0.81694072,
#                 },
#                 eval_date=dt(2023, 3, 16),
#                 expiry=dt(2023, 6, 16),
#                 id="vol",
#             ),
#         ],
#     )
#     def test_initialisation_state_smile(self, smile):
#         assert smile._state != 0
#
#     def test_initialisation_state_surface(self):
#         surf = FXDeltaVolSurface(
#             expiries=[dt(2000, 1, 1), dt(2001, 1, 1)],
#             delta_indexes=[0.5],
#             node_values=[[10.0], [9.0]],
#             eval_date=dt(1999, 1, 1),
#             delta_type="forward",
#         )
#         assert surf._state != 0
#
#
# def test_validate_delta_type() -> None:
#     with pytest.raises(ValueError, match="`delta_type` as string: 'BAD_TYPE' i"):
#         _get_fx_delta_type("BAD_TYPE")
