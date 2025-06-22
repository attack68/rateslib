import pytest
from datetime import datetime as dt
from pandas import DataFrame, date_range, Series
from pandas.testing import assert_frame_equal
import numpy as np

import context
from rateslib import defaults
from rateslib.default import NoInput
from rateslib.instruments import (
    IRS,
    FixedRateBond,
    Bill,
    FloatRateNote,
    BondFuture,
    IndexFixedRateBond,
)
from rateslib.dual import Dual, Dual2
from rateslib.calendars import dcf, get_calendar
from rateslib.curves import Curve, IndexCurve, LineCurve
from rateslib.fx import FXRates
from rateslib.solver import Solver


@pytest.fixture()
def curve():
    nodes = {
        dt(2022, 1, 1): 1.00,
        dt(2022, 4, 1): 0.99,
        dt(2022, 7, 1): 0.98,
        dt(2022, 10, 1): 0.97,
    }
    # convention = "Act360"
    return Curve(nodes=nodes, interpolation="log_linear")


@pytest.fixture()
def curve2():
    nodes = {
        dt(2022, 1, 1): 1.00,
        dt(2022, 4, 1): 0.98,
        dt(2022, 7, 1): 0.97,
        dt(2022, 10, 1): 0.95,
    }
    return Curve(nodes=nodes, interpolation="log_linear")


class TestFixedRateBond:

    def test_accrued_in_text(self):
        bond = FixedRateBond(
            effective=dt(2022, 1, 1),
            termination=dt(2023, 1, 1),
            fixed_rate=5.0,
            spec="cadgb",
        )
        assert abs(bond.accrued(dt(2022, 4, 15)) - 1.42465753) < 1e-8

        bond = FixedRateBond(
            effective=dt(2022, 1, 1),
            termination=dt(2023, 1, 1),
            fixed_rate=5.0,
            spec="gilt",
        )
        assert abs(bond.accrued(dt(2022, 4, 15)) - 1.43646409) < 1e-8

    # UK Gilts Tests: Data from public DMO website.

    @pytest.mark.parametrize("settlement, exp", [
        (dt(1999, 5, 24), False),
        (dt(1999, 5, 26), False),
        (dt(1999, 5, 27), True),
        (dt(1999, 6, 7), True)  # on payment date the
    ])
    def test_ex_div(self, settlement, exp):
        ukg = FixedRateBond(
            effective=dt(1998, 1, 1),
            termination=dt(2015, 12, 7),
            frequency="S",
            fixed_rate=8.0,
            convention="ActActICMA",
            calendar="ldn",
            ex_div=7,
            modifier="NONE",
        )
        assert ukg.ex_div(settlement) is exp

    def test_fixed_rate_bond_price_ukg(self):
        # test pricing functions against Gilt Example prices from UK DMO
        bond = FixedRateBond(
            dt(1995, 1, 1),
            dt(2015, 12, 7),
            "S",
            convention="ActActICMA",
            fixed_rate=8,
            ex_div=7,
            calendar="ldn",
            modifier="NONE",
        )
        assert abs(bond.price(4.445, dt(1999, 5, 24), True) - 145.012268) < 1e-6
        assert abs(bond.price(4.445, dt(1999, 5, 26), True) - 145.047301) < 1e-6
        assert abs(bond.price(4.445, dt(1999, 5, 27), True) - 141.070132) < 1e-6
        assert abs(bond.price(4.445, dt(1999, 6, 7), True) - 141.257676) < 1e-6

        bond = FixedRateBond(
            dt(1997, 1, 1),
            dt(2004, 11, 26),
            "S",
            convention="ActActICMA",
            fixed_rate=6.75,
            ex_div=7,
            calendar="ldn",
            modifier="F"
        )
        assert abs(bond.price(4.634, dt(1999, 5, 10), True) - 113.315543) < 1e-6
        assert abs(bond.price(4.634, dt(1999, 5, 17), True) - 113.415969) < 1e-6
        assert abs(bond.price(4.634, dt(1999, 5, 18), True) - 110.058738) < 1e-6
        assert abs(bond.price(4.634, dt(1999, 5, 26), True) - 110.170218) < 1e-6

    def test_fixed_rate_bond_price_ukg_back_stub(self):
        bond = FixedRateBond(
            dt(1995, 12, 7),
            dt(2015, 1, 23),
            "S",
            stub="SHORTBACK",
            roll=7,
            convention="ActActICMA",
            fixed_rate=8,
            ex_div=7,
            calendar="ldn",
            modifier="NONE",
            calc_mode="ukg",
        )
        result = bond.price(ytm=8.00, settlement=dt(1995, 12, 7))
        expected = 100.00334028292  # compounded back stub does not yield par
        assert abs(result - expected) < 1e-9

    def test_fixed_rate_bond_yield_ukg(self):
        # test pricing functions against Gilt Example prices from UK DMO
        bond = FixedRateBond(
            dt(1995, 1, 1),
            dt(2015, 12, 7),
            "S",
            convention="ActActICMA",
            fixed_rate=8,
            ex_div=7,
            calendar="ldn",
            modifier="NONE",
        )
        assert bond.ytm(135.0, dt(1999, 5, 24), True) - 5.1620635 < 1e-6
        assert bond.ytm(135.0, dt(1999, 5, 26), True) - 5.1649111 < 1e-6
        assert bond.ytm(135.0, dt(1999, 5, 27), True) - 4.871425 < 1e-6
        assert bond.ytm(135.0, dt(1999, 6, 7), True) - 4.8856785 < 1e-6

        bond = FixedRateBond(
            dt(1997, 1, 1),
            dt(2004, 11, 26),
            "S",
            convention="ActActICMA",
            fixed_rate=6.75,
            ex_div=7,
            calendar="ldn",
            modifier="F",
        )
        assert bond.ytm(108.0, dt(1999, 5, 10), True) - 5.7009527 < 1e-6
        assert bond.ytm(108.0, dt(1999, 5, 17), True) - 5.7253361 < 1e-6
        assert bond.ytm(108.0, dt(1999, 5, 18), True) - 5.0413308 < 1e-6
        assert bond.ytm(108.0, dt(1999, 5, 26), True) - 5.0652248 < 1e-6

    def test_fixed_rate_bond_accrual(self):
        # test pricing functions against Gilt Example prices from UK DMO, with stub
        bond = FixedRateBond(
            dt(1999, 5, 7),
            dt(2002, 12, 7),
            "S",
            convention="ActActICMA",
            front_stub=dt(1999, 12, 7),
            fixed_rate=6,
            ex_div=7,
            calendar="ldn",
            modifier="NONE",
        )
        bond.accrued(dt(1999, 5, 8)) == 0.016484
        bond.accrued(dt(1999, 6, 8)) == 0.527382
        bond.accrued(dt(1999, 7, 8)) == 1.019186
        bond.accrued(dt(1999, 11, 8)) == 3.035579
        bond.accrued(dt(1999, 11, 26)) == 3.330661
        bond.accrued(dt(1999, 11, 27)) == -0.16393
        bond.accrued(dt(1999, 12, 6)) == -0.01639
        bond.accrued(dt(1999, 12, 7)) == 0.0

    def test_fixed_rate_bond_stub_ytm(self):
        # if a regular bond is set to stub similar output should be gotten
        bond = FixedRateBond(
            dt(1999, 6, 7),
            dt(2002, 12, 7),
            "S",
            convention="ActActICMA",
            fixed_rate=6,
            ex_div=7,
            calendar="ldn",
            modifier="NONE",
        )
        regular_ytm = bond.ytm(101, dt(1999, 11, 8), dirty=True)
        bond.leg1.periods[0].stub = True
        stubbed_ytm = bond.ytm(101, dt(1999, 11, 8), dirty=True)
        assert regular_ytm == stubbed_ytm

    # US Treasury Tests. Examples from Rulebook.

    @pytest.mark.parametrize("e, t, s, fr, ec, ed, y, se", [
        (dt(1990, 5, 15), dt(2020, 5, 15), NoInput(0), 8.75, 99.057893, 99.057893, 8.84, dt(1990, 5, 15)),  # A
        (dt(1990, 4, 2), dt(1992, 3, 31), NoInput(0), 8.5, 99.838183, 99.838183, 8.59, dt(1990, 4, 2)),  # B
        (dt(1990, 3, 1), dt(1995, 5, 15), dt(1990, 11, 15), 8.5, 99.805118, 99.805118, 8.53, dt(1990, 3, 1)),  # C
        (dt(1985, 11, 15), dt(1995, 11, 15), NoInput(0), 9.5, 99.730918, 100.098321, 9.54, dt(1985, 11, 29)),  # D
        (dt(1985, 7, 2), dt(2005, 8, 15), dt(1986, 2, 15), 10.75, 102.214586, 105.887384, 10.47, dt(1985, 11, 4)),  # E
        (dt(1983, 5, 16), dt(1991, 5, 15), dt(1983, 11, 15), 10.5, 99.777074, 102.373541, 10.53, dt(1983, 8, 15)),  # F
        (dt(1988, 10, 15), dt(1994, 12, 15), dt(1989, 6, 15), 9.75, 99.738045, 100.563865, 9.79, dt(1988, 11, 15)),  # G
    ])
    def test_fixed_rate_bond_price_ust(self, e, t, s, fr, ec, ed, y, se):
        # The UST tests are from:
        # https://www.ecfr.gov/current/title-31/subtitle-B/chapter-II/subchapter-A/part-356/appendix-Appendix%20B%20to%20Part%20356
        ust = FixedRateBond(
            effective=e,
            termination=t,
            front_stub=s,
            fixed_rate=fr,
            frequency="S",
            calendar="nyc",
            convention="ActActICMA",
            calc_mode="ust_31Bii",
            ex_div=1,
            modifier="NONE",
        )
        res1 = ust.price(ytm=y, settlement=se, dirty=False)
        res2 = ust.price(ytm=y, settlement=se, dirty=True)
        assert abs(res1-ec) < 1e-6
        assert abs(res2-ed) < 1e-6

    @pytest.mark.parametrize("s, exp, acc", [
        (dt(2025, 2, 14), 99.106414, 1.926970),
        (dt(2025, 2, 18), 99.107179, 0.032113),
        (dt(2025, 8, 15), 99.151393, 0.0)
    ])
    def test_ust_price_street(self, s, exp, acc):
        bond = FixedRateBond(
            effective=dt(2023, 8, 15),
            termination=dt(2033, 8, 15),
            fixed_rate=3.875,
            spec="ust"
        )
        result = bond.price(ytm=4, settlement=s)
        accrued = bond.accrued(settlement=s)
        assert abs(accrued - acc) < 1e-6
        assert abs(result - exp) < 1e-5

    # Swedish Government Bond Tests. Data from alternative systems.

    @pytest.mark.parametrize("settlement, exp_accrued, exp_price", [
        (dt(2024, 5, 3), 0.73125, 88.134),
        # (dt(2024, 5, 5), 0.735417, 88.150), # ambiguous Sunday
        (dt(2024, 5, 6), -0.0125, 88.158),
        (dt(2024, 5, 7), -0.0104, 88.165),
        (dt(2024, 5, 8), -0.008333, 88.173),
        (dt(2024, 5, 12), 0.0, 88.203),
        (dt(2024, 5, 13), 0.002083, 88.210),
    ])
    def test_sgb_1060s_price_and_accrued(self, settlement, exp_accrued, exp_price):
        sgb = FixedRateBond(
            effective=dt(2023, 5, 12),
            termination=dt(2028, 5, 12),
            frequency="A",
            convention="ActActICMA",
            calendar="stk",
            ex_div=5,
            modifier="NONE",
            fixed_rate=0.75,
            calc_mode="sgb",
        )
        accrued = sgb.accrued(settlement)
        assert abs(accrued-exp_accrued) < 1e-4
        price = sgb.price(ytm=4.0, settlement=settlement, dirty=False)
        assert abs(price-exp_price) < 1e-3

    def test_fixed_rate_bond_price_sgb_back_stub(self):
        bond = FixedRateBond(
            dt(1995, 12, 7),
            dt(2015, 1, 23),
            "A",
            stub="SHORTBACK",
            roll=7,
            convention="ActActICMA",
            fixed_rate=8,
            ex_div=7,
            calendar="ldn",
            modifier="NONE",
            calc_mode="sgb",
        )
        result = bond.price(ytm=8.00, settlement=dt(1995, 12, 7))
        expected = 100.0018153890108  # simple period back stub yields close to par
        assert abs(result - expected) < 1e-9

    # Canadian Government Bond Tests. Data from alternative systems
    # and from https://iiac-accvm.ca/wp-content/uploads/Canadian-Conventions-in-FI-Markets-Release-1.3.pdf

    @pytest.mark.parametrize("settlement, exp", [
        (dt(2005, 12, 1), 1.671232),
        (dt(2006, 1, 31), 2.486301),
    ])
    def test_settlement_accrued(self, settlement, exp):
        bond = FixedRateBond(
            effective=dt(2004, 8, 1),
            termination=dt(2008, 2, 1),
            fixed_rate=5.0,
            modifier="NONE",
            frequency="S",
            convention="ActActICMA_stub365f",
            calc_mode="cadgb",
            ex_div=1,
        )
        result = bond.accrued(settlement=settlement)
        assert abs(result-exp) < 1e-6

    @pytest.mark.skip(reason="<1Y CAD bonds NotImplemented")
    @pytest.mark.parametrize("s, exp, acc", [
        (dt(2024, 8, 1), 99.839907, 0.0),
        (dt(2024, 7, 17), 99.866051, 1.715753),
        (dt(2024, 8, 7), 99.842641, 0.061644),
    ])
    def test_cadgb_price(self, s, exp, acc):
        bond = FixedRateBond(
            effective=dt(2022, 11, 2),
            termination=dt(2025, 2, 1),
            fixed_rate=3.75,
            modifier="NONE",
            convention="ActActICMA_STUB365f",
            frequency="S",
            calc_mode="cadgb",
            roll=1,
            stub="FRONT",
            ex_div=1,
        )
        result = bond.price(ytm=4.0, settlement=s)
        accrued = bond.accrued(settlement=s)
        assert abs(accrued-acc) < 1e-6
        # Price fails becuase bond is <1Y from maturity needs a branched formula.
        assert abs(result-exp) < 1e-6

    @pytest.mark.parametrize("s, exp, acc", [
        (dt(2024, 11, 26), 91.055145, 1.341096),
        (dt(2024, 12, 2), 91.069934, 0.007534),
        (dt(2024, 6, 3), 90.634570, 0.015068),
    ])
    def test_cadgb_price2(self, s, exp, acc):
        bond = FixedRateBond(
            effective=dt(2023, 2, 2),
            termination=dt(2033, 6, 1),
            fixed_rate=2.75,
            modifier="NONE",
            convention="ActActICMA_STUB365f",
            frequency="S",
            calc_mode="cadgb",
            roll=1,
            stub="FRONT",
            ex_div=1,
        )
        result = bond.price(ytm=4.0, settlement=s)
        accrued = bond.accrued(settlement=s)
        assert abs(accrued - acc) < 1e-6
        assert abs(result - exp) < 1e-6

    def test_cadgb_price3(self):
        bond = FixedRateBond(
            effective=dt(2018, 7, 27),
            termination=dt(2029, 6, 1),
            fixed_rate=2.25,
            modifier="NONE",
            convention="ActActICMA_STUB365f",
            frequency="S",
            calc_mode="cadgb",
            roll=1,
            stub="FRONT",
            ex_div=1,
        )
        result = bond.price(ytm=2.249977, settlement=dt(2018, 10, 16))
        accrued = bond.accrued(settlement=dt(2018, 10, 16))
        stub_cash = bond.leg1.periods[0].cashflow
        assert abs(accrued - 0.499315) < 1e-6
        assert abs(result - 100.00) < 1e-5
        assert abs(stub_cash + 7828.77) < 1e-2

    # General Method Coverage

    def test_fixed_rate_bond_yield_domains(self):
        bond = FixedRateBond(
            dt(1995, 1, 1),
            dt(2015, 12, 7),
            "S",
            convention="ActActICMA",
            fixed_rate=8,
            ex_div=7,
            calendar="ldn",
        )
        assert bond.ytm(500.0, dt(1999, 5, 24), True) + 5.86484231333 < 1e-8
        assert bond.ytm(200, dt(1999, 5, 24), True) - 1.4366895440550 < 1e-8
        assert bond.ytm(100, dt(1999, 5, 24), True) - 8.416909601459 < 1e-8
        assert bond.ytm(50, dt(1999, 5, 24), True) - 18.486840866431 < 1e-6
        assert bond.ytm(1, dt(1999, 5, 24), True) - 13421775210.82037 < 1e-3

    def test_fixed_rate_bond_ytm_duals(self):
        bond = FixedRateBond(
            dt(1995, 1, 1),
            dt(2015, 12, 7),
            "S",
            convention="ActActICMA",
            fixed_rate=8,
            ex_div=7,
            calendar="ldn",
        )

        dPdy = bond.duration(4, dt(1995, 1, 1))
        P = bond.price(4, dt(1995, 1, 1))
        result = bond.ytm(Dual(P, ["a", "b"], [1, -0.5]), dt(1995, 1, 1))
        expected = Dual(4.00, ["a", "b"], [-1 / dPdy, 0.5 / dPdy])
        assert abs(result - expected) < 1e-13
        assert all(np.isclose(expected.dual, result.dual))

        d2ydP2 = -bond.convexity(4, dt(1995, 1, 1)) * -(dPdy**-3)
        result = bond.ytm(Dual2(P, ["a", "b"], [1, -0.5], []), dt(1995, 1, 1))
        expected = Dual2(
            4.00,
            ["a", "b"],
            [-1 / dPdy, 0.5 / dPdy],
            [d2ydP2*0.5, d2ydP2 * -0.25, d2ydP2 * -0.25, d2ydP2 * 0.125],
        )
        assert abs(result-expected) < 1e-13
        assert all(np.isclose(result.dual, expected.dual))
        assert all(np.isclose(result.dual2, expected.dual2).flat)

    @pytest.mark.skip(reason="Bills have Z frequency, this no longer raises")
    def test_fixed_rate_bond_zero_frequency_raises(self):
        with pytest.raises(ValueError, match="FixedRateBond `frequency`"):
            FixedRateBond(dt(1999, 5, 7), dt(2002, 12, 7), "Z", convention="ActActICMA")

    @pytest.mark.parametrize("metric", ["risk", "duration", "modified"])
    def test_fixed_rate_bond_duration(self, metric):
        gilt = FixedRateBond(
            effective=dt(1998, 12, 7),
            termination=dt(2015, 12, 7),
            frequency="S",
            calendar="ldn",
            currency="gbp",
            convention="ActActICMA",
            ex_div=7,
            fixed_rate=8.0,
        )
        price0 = gilt.price(4.445, dt(1999, 5, 27))
        price1 = gilt.price(4.446, dt(1999, 5, 27))
        if metric == "risk":
            numeric = price0 - price1
        elif metric == "modified":
            numeric = (price0 - price1) / price0 * 100
        elif metric == "duration":
            numeric = (price0 - price1) / price0 * (1 + 4.445 / (100 * 2)) * 100

        result = gilt.duration(4.445, dt(1999, 5, 27), metric=metric)
        assert (result - numeric * 1000) < 1e-1

    def test_fixed_rate_bond_convexity(self):
        gilt = FixedRateBond(
            effective=dt(1998, 12, 7),
            termination=dt(2015, 12, 7),
            frequency="S",
            calendar="ldn",
            currency="gbp",
            convention="ActActICMA",
            ex_div=7,
            fixed_rate=8.0,
        )
        numeric = gilt.duration(4.445, dt(1999, 5, 27)) - gilt.duration(4.446, dt(1999, 5, 27))
        result = gilt.convexity(4.445, dt(1999, 5, 27))
        assert (result - numeric * 1000) < 1e-3

    def test_fixed_rate_bond_rate(self):
        gilt = FixedRateBond(
            effective=dt(1998, 12, 7),
            termination=dt(2015, 12, 7),
            frequency="S",
            calendar="ldn",
            currency="gbp",
            convention="ActActICMA",
            ex_div=7,
            fixed_rate=8.0,
            settle=0,
        )
        curve = Curve({dt(1998, 12, 9): 1.0, dt(2015, 12, 7): 0.50})
        clean_price = gilt.rate(curve, metric="clean_price")
        result = gilt.rate(curve, metric="clean_price", forward_settlement=dt(1998, 12, 9))
        assert abs(result - clean_price) < 1e-8

        result = gilt.rate(curve, metric="dirty_price")
        expected = clean_price + gilt.accrued(dt(1998, 12, 9))
        assert result == expected
        result = gilt.rate(curve, metric="dirty_price", forward_settlement=dt(1998, 12, 9))
        assert abs(result - clean_price - gilt.accrued(dt(1998, 12, 9))) < 1e-8

        result = gilt.rate(curve, metric="ytm")
        expected = gilt.ytm(clean_price, dt(1998, 12, 9), False)
        assert abs(result - expected) < 1e-8

    def test_fixed_rate_bond_npv(self):
        gilt = FixedRateBond(
            effective=dt(1998, 12, 7),
            termination=dt(2015, 12, 7),
            frequency="S",
            calendar="ldn",
            currency="gbp",
            convention="ActActICMA",
            ex_div=7,
            fixed_rate=8.0,
            notional=-100,
            settle=0,
        )
        curve = Curve({dt(2010, 11, 25): 1.0, dt(2015, 12, 7): 0.75})
        result = gilt.npv(curve)
        expected = 113.22198344812742
        assert abs(result - expected) < 1e-6

        gilt.kwargs["settle"] = 2
        result = gilt.npv(curve)  # bond is ex div on settlement 27th Nov 2010
        expected = 109.229489312983  # bond has dropped a coupon payment of 4.
        assert abs(result - expected) < 1e-6

        result = gilt.npv(curve, local=True)
        assert abs(result["gbp"] - expected) < 1e-6

    def test_fixed_rate_bond_npv_private(self):
        # this test shadows 'fixed_rate_bond_npv' but extends it for projection on 27th Nov ex div.
        curve = Curve({dt(2004, 11, 25): 1.0, dt(2010, 11, 25): 1.0, dt(2015, 12, 7): 0.75})
        gilt = FixedRateBond(
            effective=dt(1998, 12, 7),
            termination=dt(2015, 12, 7),
            frequency="S",
            calendar="ldn",
            currency="gbp",
            convention="ActActICMA",
            ex_div=7,
            fixed_rate=8.0,
            notional=-100,
            settle=0,
        )
        result = gilt._npv_local(NoInput(0), curve, NoInput(0), NoInput(0), dt(2010, 11, 27), dt(2010, 11, 25))
        expected = 109.229489312983  # npv should match associated test
        assert abs(result - expected) < 1e-6

    def test_fixed_rate_bond_analytic_delta(self):
        gilt = FixedRateBond(
            effective=dt(1998, 12, 7),
            termination=dt(2015, 12, 7),
            frequency="S",
            calendar="ldn",
            currency="gbp",
            convention="ActActICMA",
            ex_div=7,
            fixed_rate=8.0,
            notional=-1000000,
            settle=0,
        )
        curve = Curve({dt(2010, 11, 25): 1.0, dt(2015, 12, 7): 1.0})
        result = gilt.analytic_delta(curve)
        expected = -550.0
        assert abs(result - expected) < 1e-6

        gilt.kwargs["settle"] = 2
        result = gilt.analytic_delta(curve)  # bond is ex div on settle 27th Nov 2010
        expected = -500.0  # bond has dropped a 6m coupon payment
        assert abs(result - expected) < 1e-6

    def test_fixed_rate_bond_cashflows(self):
        gilt = FixedRateBond(
            effective=dt(1998, 12, 7),
            termination=dt(2015, 12, 7),
            frequency="S",
            calendar="ldn",
            currency="gbp",
            convention="ActActICMA",
            ex_div=7,
            fixed_rate=8.0,
            notional=-100,
            settle=1,
        )
        curve = Curve({dt(2010, 11, 25): 1.0, dt(2015, 12, 7): 0.75})

        flows = gilt.cashflows(curve)  # bond is ex div on 26th nov 2010
        result = flows[defaults.headers["npv"]].sum()
        expected = gilt.npv(curve)
        assert abs(result - expected) < 1e-6

        gilt.settle = 0
        flows = gilt.cashflows(curve)  # settlement from curve initial node
        result = flows[defaults.headers["npv"]].sum()
        expected = gilt.npv(curve)
        assert abs(result - expected) < 1e-6

    def test_fixed_rate_bond_rate_raises(self):
        gilt = FixedRateBond(
            effective=dt(1998, 12, 7),
            termination=dt(2015, 12, 7),
            frequency="S",
            calendar="ldn",
            currency="gbp",
            convention="ActActICMA",
            ex_div=7,
            fixed_rate=8.0,
            notional=-100,
        )
        curve = Curve({dt(1998, 12, 7): 1.0, dt(2015, 12, 7): 0.50})
        with pytest.raises(ValueError, match="`metric` must be in"):
            gilt.rate(curve, metric="bad_metric")

    def test_fixed_rate_bond_no_amortization(self):
        with pytest.raises(NotImplementedError, match="`amortization` for"):
            FixedRateBond(
                effective=dt(1998, 12, 7),
                termination=dt(2015, 12, 7),
                frequency="S",
                calendar="ldn",
                currency="gbp",
                convention="ActActICMA",
                ex_div=7,
                fixed_rate=8.0,
                notional=-100,
                amortization=100,
            )

    @pytest.mark.parametrize(
        "f_s, exp",
        [
            (dt(2001, 12, 31), 99.997513754),  # compounding of mid year coupon
            (dt(2002, 1, 1), 99.9975001688),  # this is now ex div on last coupon
        ],
    )
    def test_fixed_rate_bond_forward_price_analogue(self, f_s, exp):
        gilt = FixedRateBond(
            effective=dt(2001, 1, 1),
            termination=dt(2002, 1, 1),
            frequency="S",
            calendar=NoInput(0),
            currency="gbp",
            convention="Act365f",
            ex_div=0,
            fixed_rate=1.0,
            notional=-100,
            settle=0,
        )
        result = gilt.fwd_from_repo(100.0, dt(2001, 1, 1), f_s, 1.0, "act365f")
        assert abs(result - exp) < 1e-6

    @pytest.mark.parametrize(
        "f_s, exp",
        [
            (dt(2001, 12, 31), 100.49888361793),  # compounding of mid year coupon
            (dt(2002, 1, 1), 99.9975001688),  # this is now ex div on last coupon
        ],
    )
    def test_fixed_rate_bond_forward_price_analogue_dirty(self, f_s, exp):
        gilt = FixedRateBond(
            effective=dt(2001, 1, 1),
            termination=dt(2002, 1, 1),
            frequency="S",
            calendar=NoInput(0),
            currency="gbp",
            convention="Act365f",
            ex_div=1,
            fixed_rate=1.0,
            notional=-100,
            settle=0,
        )
        result = gilt.fwd_from_repo(100.0, dt(2001, 1, 1), f_s, 1.0, "act365f", dirty=True)
        assert abs(result - exp) < 1e-6

    @pytest.mark.parametrize(
        "s, f_s, exp",
        [
            (dt(2010, 11, 25), dt(2011, 11, 25), 99.9975000187), # div div
            (dt(2010, 11, 28), dt(2011, 11, 29), 99.997471945), # ex-div ex-div
            (dt(2010, 11, 28), dt(2011, 11, 25), 99.997419419), # ex-div div
            (dt(2010, 11, 25), dt(2011, 11, 29), 99.9975516607),  # div ex-div
        ],
    )
    def test_fixed_rate_bond_forward_price_analogue_ex_div(self, s, f_s, exp):
        gilt = FixedRateBond(
            effective=dt(1998, 12, 7),
            termination=dt(2015, 12, 7),
            frequency="S",
            calendar="ldn",
            currency="gbp",
            convention="act365f",
            ex_div=7,
            fixed_rate=1.0,
            notional=-100,
            settle=0,
        )
        result = gilt.fwd_from_repo(100.0, s, f_s, 1.0, "act365f")
        assert abs(result - exp) < 1e-6

    @pytest.mark.parametrize(
        "f_s, f_p",
        [
            (dt(2001, 12, 31), 99.997513754),  # compounding of mid year coupon
            (dt(2002, 1, 1), 99.9975001688),  # this is now ex div on last coupon
        ],
    )
    def test_fixed_rate_bond_implied_repo(self, f_s, f_p):
        gilt = FixedRateBond(
            effective=dt(2001, 1, 1),
            termination=dt(2002, 1, 1),
            frequency="S",
            calendar=NoInput(0),
            currency="gbp",
            convention="Act365f",
            ex_div=0,
            fixed_rate=1.0,
            notional=-100,
            settle=0,
        )
        result = gilt.repo_from_fwd(100.0, dt(2001, 1, 1), f_s, f_p, "act365f")
        assert abs(result - 1.00) < 1e-8

    @pytest.mark.parametrize(
        "f_s, f_p",
        [
            (dt(2001, 12, 31), 100.49888361793),  # compounding of mid year coupon
            (dt(2002, 1, 1), 99.9975001688),  # this is now ex div on last coupon
        ],
    )
    def test_fixed_rate_bond_implied_repo_analogue_dirty(self, f_s, f_p):
        gilt = FixedRateBond(
            effective=dt(2001, 1, 1),
            termination=dt(2002, 1, 1),
            frequency="S",
            calendar=NoInput(0),
            currency="gbp",
            convention="Act365f",
            ex_div=1,
            fixed_rate=1.0,
            notional=-100,
            settle=0,
        )
        result = gilt.repo_from_fwd(100.0, dt(2001, 1, 1), f_s, f_p, "act365f", dirty=True)
        assert abs(result - 1.0) < 1e-8

    @pytest.mark.parametrize("price, tol", [
        (112.0, 1e-10),
        (104.0, 1e-10),
        (96.0, 1e-9),
        (91.0, 1e-7)
    ])
    def test_oaspread(self, price, tol):
        gilt = FixedRateBond(
            effective=dt(1998, 12, 7),
            termination=dt(2015, 12, 7),
            frequency="S",
            calendar="ldn",
            currency="gbp",
            convention="ActActICMA",
            ex_div=7,
            fixed_rate=8.0,
            notional=-100,
            settle=0,
        )
        curve = Curve({dt(2010, 11, 25): 1.0, dt(2015, 12, 7): 0.75})
        # result = gilt.npv(curve) = 113.22198344812742
        result = gilt.oaspread(curve, price=price)
        curve_z = curve.shift(result, composite=False)
        result = gilt.rate(curve_z, metric="clean_price")
        assert abs(result - price) < tol

    @pytest.mark.parametrize("price, tol", [
        (85, 1e-8),
        (75, 1e-6),
        (65, 1e-4),
        (55, 1e-3),
        (45, 1e-1),
        (35, 0.20),
    ])
    def test_oaspread_low_price(self, price, tol):
        gilt = FixedRateBond(
            effective=dt(1998, 12, 7),
            termination=dt(2015, 12, 7),
            frequency="S",
            calendar="ldn",
            currency="gbp",
            convention="ActActICMA",
            ex_div=7,
            fixed_rate=1.0,
            notional=-100,
            settle=0,
        )
        curve = Curve({dt(1999, 11, 25): 1.0, dt(2015, 12, 7): 0.85})
        # result = gilt.npv(curve) = 113.22198344812742
        result = gilt.oaspread(curve, price=price)
        curve_z = curve.shift(result, composite=False)
        result = gilt.rate(curve_z, metric="clean_price")
        assert abs(result - price) < tol

    def test_cashflows_no_curve(self):
        gilt = FixedRateBond(
            effective=dt(2001, 1, 1),
            termination="1Y",
            spec="ukt",
            fixed_rate=5.0
        )
        result = gilt.cashflows()  # no curve argument is passed to cashflows
        assert isinstance(result, DataFrame)


class TestIndexFixedRateBond:
    def test_fixed_rate_bond_price(self):
        # test pricing functions against Nominal Gilt Example prices from UK DMO
        # these prices should be equivalent for the REAL component of Index Bonds
        bond = IndexFixedRateBond(
            dt(1995, 1, 1),
            dt(2015, 12, 7),
            "S",
            convention="ActActICMA",
            fixed_rate=8,
            ex_div=7,
            calendar="ldn",
            index_base=100.0,
        )
        assert abs(bond.price(4.445, dt(1999, 5, 24), True) - 145.012268) < 1e-6
        assert abs(bond.price(4.445, dt(1999, 5, 26), True) - 145.047301) < 1e-6
        assert abs(bond.price(4.445, dt(1999, 5, 27), True) - 141.070132) < 1e-6
        assert abs(bond.price(4.445, dt(1999, 6, 7), True) - 141.257676) < 1e-6

        bond = IndexFixedRateBond(
            dt(1997, 1, 1),
            dt(2004, 11, 26),
            "S",
            convention="ActActICMA",
            fixed_rate=6.75,
            ex_div=7,
            calendar="ldn",
            index_base=100.0,
        )
        assert abs(bond.price(4.634, dt(1999, 5, 10), True) - 113.315543) < 1e-6
        assert abs(bond.price(4.634, dt(1999, 5, 17), True) - 113.415969) < 1e-6
        assert abs(bond.price(4.634, dt(1999, 5, 18), True) - 110.058738) < 1e-6
        assert abs(bond.price(4.634, dt(1999, 5, 26), True) - 110.170218) < 1e-6

    @pytest.mark.skip(reason="Frequency of zero calculates but is wrong. Docs do not allow.")
    def test_fixed_rate_bond_zero_frequency_raises(self):
        with pytest.raises(ValueError, match="`frequency` must be provided"):
            IndexFixedRateBond(dt(1999, 5, 7), dt(2002, 12, 7), "Z", convention="ActActICMA", fixed_rate=1.0)

    def test_fixed_rate_bond_no_amortization(self):
        with pytest.raises(NotImplementedError, match="`amortization` for"):
            IndexFixedRateBond(
                effective=dt(1998, 12, 7),
                termination=dt(2015, 12, 7),
                frequency="S",
                calendar="ldn",
                currency="gbp",
                convention="ActActICMA",
                ex_div=7,
                fixed_rate=8.0,
                notional=-100,
                amortization=100,
                index_base=100.0,
            )

    def test_fixed_rate_bond_rate_raises(self):
        gilt = IndexFixedRateBond(
            effective=dt(1998, 12, 7),
            termination=dt(2015, 12, 7),
            frequency="S",
            calendar="ldn",
            currency="gbp",
            convention="ActActICMA",
            ex_div=7,
            fixed_rate=8.0,
            notional=-100,
            index_base=100.0,
        )
        curve = Curve({dt(1998, 12, 7): 1.0, dt(2015, 12, 7): 0.50})
        with pytest.raises(ValueError, match="`metric` must be in"):
            gilt.rate(curve, metric="bad_metric")

    @pytest.mark.parametrize(
        "i_fixings, expected",
        [(NoInput(0), 1.161227269), (Series([90, 290], index=[dt(2022, 4, 1), dt(2022, 4, 29)]), 2.00)],
    )
    def test_index_ratio(self, i_fixings, expected):
        i_curve = IndexCurve(
            {dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99},
            index_lag=3,
            index_base=110.0,
        )
        bond = IndexFixedRateBond(
            dt(2022, 1, 1),
            "9m",
            "Q",
            convention="ActActICMA",
            fixed_rate=4,
            ex_div=0,
            calendar="ldn",
            index_base=95.0,
            index_fixings=i_fixings,
            index_method="daily",
        )
        result = bond.index_ratio(settlement=dt(2022, 4, 15), curve=i_curve)
        assert abs(result - expected) < 1e-5

    def test_index_ratio_raises_float_index_fixings(self):
        i_curve = IndexCurve(
            {dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99},
            index_lag=3,
            index_base=110.0,
        )
        bond = IndexFixedRateBond(
            dt(2022, 1, 1),
            "9m",
            "Q",
            convention="ActActICMA",
            fixed_rate=4,
            ex_div=0,
            calendar="ldn",
            index_base=95.0,
            index_fixings=[100.0, 200.0],
            index_method="daily",
        )
        with pytest.raises(ValueError, match="Must provide `index_fixings` as a Seri"):
            bond.index_ratio(settlement=dt(2022, 4, 15), curve=i_curve)

    def test_fixed_rate_bond_npv_private(self):
        # this test shadows 'fixed_rate_bond_npv' but extends it for projection
        curve = Curve({dt(2004, 11, 25): 1.0, dt(2010, 11, 25): 1.0, dt(2015, 12, 7): 0.75})
        index_curve = IndexCurve({dt(2004, 11, 25): 1.0, dt(2034, 1, 1): 1.0}, index_base=100.0)
        gilt = IndexFixedRateBond(
            effective=dt(1998, 12, 7),
            termination=dt(2015, 12, 7),
            frequency="S",
            calendar="ldn",
            currency="gbp",
            convention="ActActICMA",
            ex_div=7,
            fixed_rate=8.0,
            notional=-100,
            settle=0,
            index_base=50.0,
            index_lag=3,
            index_method="daily",
        )
        result = gilt._npv_local(index_curve, curve, NoInput(0), NoInput(0), dt(2010, 11, 27), dt(2010, 11, 25))
        expected = 109.229489312983 * 2.0  # npv should match associated test
        assert abs(result - expected) < 1e-6

    def test_index_base_forecast(self, curve):
        i_curve = IndexCurve(
            {dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.99},
            index_lag=3,
            index_base=95.0,
        )
        bond = IndexFixedRateBond(
            dt(2022, 1, 1),
            "9m",
            "Q",
            convention="ActActICMA",
            fixed_rate=4,
            ex_div=0,
            calendar=NoInput(0),
            index_method="daily",
            settle=0,
        )
        cashflows = bond.cashflows([i_curve, curve])
        for i in range(4):
            assert cashflows.iloc[i]["Index Base"] == 95.0

        result = bond.npv([i_curve, curve])
        expected = -1006875.3812
        assert abs(result - expected) < 1e-4

        result = bond.rate([i_curve, curve], metric="index_dirty_price")
        assert abs(result * -1e4 - expected) < 1e-4

    def test_fixed_rate_bond_fwd_rate(self):
        gilt = IndexFixedRateBond(
            effective=dt(1998, 12, 7),
            termination=dt(2015, 12, 7),
            frequency="S",
            calendar="ldn",
            currency="gbp",
            convention="ActActICMA",
            ex_div=7,
            fixed_rate=8.0,
            settle=0,
            index_base=50.0,
        )
        curve = Curve({dt(1998, 12, 9): 1.0, dt(2015, 12, 7): 0.50})
        i_curve = IndexCurve({dt(1998, 12, 9): 1.0, dt(2015, 12, 7): 1.0}, index_base=100.0)
        clean_price = gilt.rate([i_curve, curve], metric="clean_price")
        index_clean_price = gilt.rate([i_curve, curve], metric="index_clean_price")
        assert abs(index_clean_price * 0.5 - clean_price) < 1e-3

        result = gilt.rate(
            [i_curve, curve], metric="clean_price", forward_settlement=dt(1998, 12, 9)
        )
        assert abs(result - clean_price) < 1e-8
        result = gilt.rate(
            [i_curve, curve], metric="index_clean_price", forward_settlement=dt(1998, 12, 9)
        )
        assert abs(result * 0.5 - clean_price) < 1e-8

        result = gilt.rate([i_curve, curve], metric="dirty_price")
        expected = clean_price + gilt.accrued(dt(1998, 12, 9))
        assert result == expected
        result = gilt.rate(
            [i_curve, curve], metric="dirty_price", forward_settlement=dt(1998, 12, 9)
        )
        assert abs(result - clean_price - gilt.accrued(dt(1998, 12, 9))) < 1e-8
        result = gilt.rate(
            [i_curve, curve], metric="index_dirty_price", forward_settlement=dt(1998, 12, 9)
        )
        assert abs(result * 0.5 - clean_price - gilt.accrued(dt(1998, 12, 9))) < 1e-8

        result = gilt.rate([i_curve, curve], metric="ytm")
        expected = gilt.ytm(clean_price, dt(1998, 12, 9), False)
        assert abs(result - expected) < 1e-8

    # TODO: implement these tests
    # def test_fwd_from_repo(self):
    #     assert False
    #
    # def test_repo_from_fwd(self):
    #     assert False
    #
    # def test_duration(self):
    #     assert False
    #
    # def test_convexity(self):
    #     assert False

    def test_latest_fixing(self):
        # this is German government inflation bond with fixings given for a specific settlement
        # calculation

        ibnd = IndexFixedRateBond(
            effective=dt(2021, 2, 11),
            front_stub=dt(2022, 4, 15),
            termination=dt(2033, 4, 15),
            convention="ActActICMA",
            calendar="tgt",
            frequency="A",
            index_lag=3,
            index_base=124.17000 / 1.18851,  # implying from 1st Jan 2024 on webpage
            index_method="daily",
            payment_lag=0,
            currency="eur",
            fixed_rate=0.1,
            ex_div=1,
            settle=1,
            index_fixings=Series(
                data=[124.17, 123.46],
                index=[dt(2024, 1, 1), dt(2024, 2, 1)]
            ),
        )
        result = ibnd.ytm(price=100.32, settlement=dt(2024, 1, 5))
        expected = 0.065
        assert (result - expected) < 1e-2


class TestBill:
    def test_bill_discount_rate(self):
        # test pricing functions against Treasury Bill Example from US Treasury
        bill = Bill(
            effective=dt(2004, 1, 22),
            termination=dt(2004, 2, 19),
            calendar="nyc",
            currency="usd",
            convention="Act360",
            calc_mode="ustb",
        )

        assert bill.discount_rate(99.93777, dt(2004, 1, 22)) == 0.8000999999999543
        assert bill.price(0.800, dt(2004, 1, 22)) == 99.93777777777778

    def test_bill_ytm(self):
        bill = Bill(
            effective=dt(2004, 1, 22),
            termination=dt(2004, 2, 19),
            calendar="nyc",
            currency="usd",
            convention="Act360",
            calc_mode="ustb",
        )
        # this YTM is equivalent to the FixedRateBond ytm with coupon of 0.0

        result = bill.ytm(99.937778, dt(2004, 1, 22))

        # TODO this does not match US treasury example because the method is different
        assert abs(result - 0.814) < 1e-2

    def test_bill_ytm2(self):
        # this is a longer than 6m period
        bill = Bill(
            effective=dt(1990, 6, 7),
            termination=dt(1991, 6, 6),
            convention="act360",
            calc_mode="ustb",
        )
        price = bill.price(7.65, settlement=dt(1990, 6, 7))
        result = bill.ytm(price, settlement=dt(1990, 6, 7))
        assert abs(result - 8.237) < 1e-3

    def test_bill_simple_rate(self):
        bill = Bill(
            effective=dt(2004, 1, 22),
            termination=dt(2004, 2, 19),
            calendar="nyc",
            currency="usd",
            convention="Act360",
            calc_mode="ustb",
        )
        d = dcf(dt(2004, 1, 22), dt(2004, 2, 19), "Act360")
        expected = 100 * (1 / (1 - 0.0080009999999 * d) - 1) / d  # floating point truncation
        expected = 100 * (100 / 99.93777777777778 - 1) / d
        result = bill.simple_rate(99.93777777777778, dt(2004, 1, 22))
        assert abs(result - expected) < 1e-6

    def test_bill_rate(self):
        curve = Curve({dt(2004, 1, 22): 1.00, dt(2005, 1, 22): 0.992})

        bill = Bill(
            effective=dt(2004, 1, 22),
            termination=dt(2004, 2, 19),
            calendar="nyc",
            currency="usd",
            convention="Act360",
            settle=0,
            calc_mode="ustb"
        )

        result = bill.rate(curve, metric="price")
        expected = 99.9385705675
        assert abs(result - expected) < 1e-6

        result = bill.rate(curve, metric="discount_rate")
        expected = bill.discount_rate(99.9385705675, dt(2004, 1, 22))
        assert abs(result - expected) < 1e-6

        result = bill.rate(curve, metric="simple_rate")
        expected = bill.simple_rate(99.9385705675, dt(2004, 1, 22))
        assert abs(result - expected) < 1e-6

        result = bill.rate(curve, metric="ytm")
        expected = bill.ytm(99.9385705675, dt(2004, 1, 22))
        assert abs(result - expected) < 1e-6

        bill.kwargs['settle'] = 2  # set the bill to T+2 settlement and re-run the calculations

        result = bill.rate(curve, metric="price")
        expected = 99.94734388985547
        assert abs(result - expected) < 1e-6

        result = bill.rate(curve, metric="discount_rate")
        expected = bill.discount_rate(99.94734388985547, dt(2004, 1, 26))
        assert abs(result - expected) < 1e-6

        result = bill.rate(curve, metric="simple_rate")
        expected = bill.simple_rate(99.94734388985547, dt(2004, 1, 26))
        assert abs(result - expected) < 1e-6

        result = bill.rate(curve, metric="ytm")
        expected = bill.ytm(99.94734388985547, dt(2004, 1, 26))
        assert abs(result - expected) < 1e-6

    def test_bill_rate_raises(self):
        curve = Curve({dt(2004, 1, 22): 1.00, dt(2005, 1, 22): 0.992})

        bill = Bill(
            effective=dt(2004, 1, 22),
            termination=dt(2004, 2, 19),
            calendar="nyc",
            currency="usd",
            convention="Act360",
        )

        with pytest.raises(ValueError, match="`metric` must be in"):
            bill.rate(curve, metric="bad vibes")

    def test_sgbb(self):
        bill = Bill(
            effective=dt(2023, 3, 15),
            termination=dt(2024, 3, 20),
            spec="sgbb",
        )
        result = bill.price(3.498, settlement=dt(2023, 3, 15))
        expected = 96.520547
        assert abs(result - expected) < 1e-6

        ytm = bill.ytm(price=96.520547, settlement=dt(2023, 3, 15))
        assert abs(ytm - 3.5546338) < 1e-5

    def test_text_example(self):
        bill = Bill(effective=dt(2023, 5, 17), termination=dt(2023, 9, 26), spec="ustb")
        result = bill.ytm(99.75, settlement=dt(2023, 9, 7))
        bond = FixedRateBond(
            effective=dt(2023, 3, 26),
            termination=dt(2023, 9, 26),
            fixed_rate=0.0,
            spec="ust"
        )
        expected = bond.ytm(99.75, settlement=dt(2023, 9, 7))
        assert abs(result - expected) < 1e-14
        assert abs(result - 4.90740754) < 1e-7

    @pytest.mark.parametrize("price, tol", [
        (96.0, 1e-6),
        (95.0, 1e-6),
        (93.0, 1e-5),
        (80.0, 1e-2)
    ])
    def test_oaspread(self, price, tol):
        bill = Bill(
            effective=dt(1998, 12, 7),
            termination=dt(1999, 10, 7),
            spec="ustb",
        )
        curve = Curve({dt(1998, 12, 7): 1.0, dt(2015, 12, 7): 0.75})
        # result = bill.rate(curve, metric="price") # = 98.605
        result = bill.oaspread(curve, price=price)
        curve_z = curve.shift(result, composite=False)
        result = bill.rate(curve_z, metric="clean_price")
        assert abs(result - price) < tol


class TestFloatRateNote:
    @pytest.mark.parametrize(
        "curve_spd, method, float_spd, expected",
        [
            (10, NoInput(0), 0, 10.055032859883),
            (500, NoInput(0), 0, 508.93107035125325),
            (-200, NoInput(0), 0, -200.053341848676),
            (10, "isda_compounding", 0, 10.00000120),
            (500, "isda_compounding", 0, 500.050371345),
            (-200, "isda_compounding", 0, -200.003309580533),
            (10, NoInput(0), 25, 10.055032859883),
            (500, NoInput(0), 250, 508.93107035125325),
            (10, "isda_compounding", 25, 10.00000120),
            (500, "isda_compounding", 250, 500.00635330533544),
            (10, NoInput(0), -25, 10.055032859883),
            (500, NoInput(0), -250, 508.93107035125325),
            (10, "isda_compounding", -25, 10.00000120),
            (500, "isda_compounding", -250, 500.16850637415),
        ],
    )
    def test_float_rate_bond_rate_spread(self, curve_spd, method, float_spd, expected):
        """
        When a DF curve is shifted it bumps daily rates.
        But under the "none_simple" compounding method this does not compound daily
        therefore the `float_spread` should be slightly higher than the bumped curve.
        When the method is "isda_compounding" this closely matches the bumping method
        of the curve.
        """

        bond = FloatRateNote(
            effective=dt(2007, 1, 1),
            termination=dt(2017, 1, 1),
            frequency="S",
            convention="Act365f",
            ex_div=0,
            settle=0,
            float_spread=float_spd,
            spread_compound_method=method,
        )
        curve = Curve({dt(2007, 1, 1): 1.0, dt(2017, 1, 1): 0.9}, convention="Act365f")
        disc_curve = curve.shift(curve_spd)
        result = bond.rate([curve, disc_curve], metric="spread")
        assert abs(result - expected) < 1e-4

        bond.float_spread = result
        validate = bond.npv([curve, disc_curve])
        assert abs(validate + bond.leg1.notional) < 0.30 * abs(curve_spd)

    @pytest.mark.parametrize(
        "curve_spd, method, float_spd, expected",
        [
            (10, "isda_compounding", 0, 10.00000120),
        ],
    )
    def test_float_rate_bond_rate_spread_fx(self, curve_spd, method, float_spd, expected):
        bond = FloatRateNote(
            effective=dt(2007, 1, 1),
            termination=dt(2017, 1, 1),
            frequency="S",
            convention="Act365f",
            ex_div=0,
            settle=0,
            float_spread=float_spd,
            spread_compound_method=method,
        )
        curve = Curve({dt(2007, 1, 1): 1.0, dt(2017, 1, 1): 0.9}, convention="Act365f")
        disc_curve = curve.shift(curve_spd)
        fxr = FXRates({"usdnok": 10.0}, settlement=dt(2007, 1, 1))
        result = bond.rate(
            [curve, disc_curve],
            metric="spread",
            fx=fxr,
        )
        assert abs(result - expected) < 1e-4

        bond.float_spread = result
        validate = bond.npv([curve, disc_curve], fx=fxr)
        assert abs(validate + bond.leg1.notional) < 0.30 * abs(curve_spd)

    def test_float_rate_bond_accrued(self):
        fixings = Series(2.0, index=date_range(dt(2009, 12, 1), dt(2010, 3, 1)))
        bond = FloatRateNote(
            effective=dt(2007, 1, 1),
            termination=dt(2017, 1, 1),
            frequency="S",
            convention="Act365f",
            ex_div=3,
            float_spread=100,
            fixing_method="rfr_observation_shift",
            fixings=fixings,
            method_param=5,
            spread_compound_method="none_simple",
        )
        result = bond.accrued(dt(2010, 3, 3))
        expected = 0.5019199020076  # 3% * 2 / 12
        assert abs(result - expected) < 1e-8

    @pytest.mark.parametrize(
        "metric, spd, exp",
        [
            ("clean_price", 0.0, 100.0),
            ("dirty_price", 0.0, 100.0),
            ("clean_price", 10.0, 99.99982764447981),  # compounding diff between shift
            ("dirty_price", 10.0, 100.0165399732469),
        ],
    )
    def test_float_rate_bond_rate_metric(self, metric, spd, exp):
        fixings = Series(0.0, index=date_range(dt(2009, 12, 1), dt(2010, 3, 1)))
        bond = FloatRateNote(
            effective=dt(2007, 1, 1),
            termination=dt(2017, 1, 1),
            frequency="S",
            convention="Act365f",
            ex_div=3,
            float_spread=spd,
            fixing_method="rfr_observation_shift",
            fixings=fixings,
            method_param=5,
            spread_compound_method="none_simple",
            settle=2,
        )
        curve = Curve({dt(2010, 3, 1): 1.0, dt(2017, 1, 1): 1.0}, convention="act365f")
        disc_curve = curve.shift(spd)

        result = bond.rate(curves=[curve, disc_curve], metric=metric)
        assert abs(result - exp) < 1e-8

    @pytest.mark.parametrize(
        "settlement, expected",
        [(dt(2010, 3, 3), 0.501369863013698), (dt(2010, 12, 30), -0.005479452054)],
    )
    def test_float_rate_bond_accrued_ibor(self, settlement, expected):
        fixings = Series(2.0, index=date_range(dt(2009, 12, 1), dt(2010, 3, 1)))
        bond = FloatRateNote(
            effective=dt(2007, 1, 1),
            termination=dt(2017, 1, 1),
            frequency="S",
            convention="Act365f",
            ex_div=3,
            float_spread=100,
            fixing_method="ibor",
            fixings=fixings,
            method_param=2,
            spread_compound_method="none_simple",
        )
        result = bond.accrued(settlement)
        assert abs(result - expected) < 1e-8

    def test_float_rate_bond_raise_frequency(self):
        with pytest.raises(ValueError, match="FloatRateNote `frequency`"):
            FloatRateNote(
                effective=dt(2007, 1, 1),
                termination=dt(2017, 1, 1),
                frequency="Z",
                convention="Act365f",
                ex_div=3,
                float_spread=100,
                fixing_method="rfr_observation_shift",
                fixings=NoInput(0),
                method_param=5,
                spread_compound_method="none_simple",
            )

    @pytest.mark.parametrize(
        "fixings",
        [
            Series(2.0, index=date_range(dt(2009, 12, 1), dt(2010, 3, 8))),
            [2.0, [2.0, 2.0]],
        ],
    )
    def test_negative_accrued_needs_forecasting(self, fixings):
        bond = FloatRateNote(
            effective=dt(2009, 9, 16),
            termination=dt(2017, 3, 16),
            frequency="Q",
            convention="Act365f",
            ex_div=6,
            float_spread=0,
            fixing_method="rfr_observation_shift",
            fixings=fixings,
            method_param=5,
            spread_compound_method="none_simple",
            calendar=NoInput(0),
        )
        with pytest.warns(UserWarning):
            result = bond.accrued(dt(2010, 3, 11))

        # approximate calculation 5 days of negative accrued at 2% = -0.027397
        assert abs(result + 2 * 5 / 365) < 1e-3

    @pytest.mark.parametrize(
        "fixings",
        [
            NoInput(0),
        ],
    )
    def test_negative_accrued_raises(self, fixings):
        bond = FloatRateNote(
            effective=dt(2009, 9, 16),
            termination=dt(2017, 3, 16),
            frequency="Q",
            convention="Act365f",
            ex_div=5,
            float_spread=0,
            fixing_method="rfr_observation_shift",
            fixings=fixings,
            method_param=5,
            spread_compound_method="none_simple",
            calendar=NoInput(0),
        )
        with pytest.raises(TypeError, match="`fixings` or `curve` are not available for RFR"):
            bond.accrued(dt(2010, 3, 11))

        with pytest.raises(ValueError, match="For RFR FRNs `ex_div` must be less than"):
            bond = FloatRateNote(
                effective=dt(2009, 9, 16),
                termination=dt(2017, 3, 16),
                frequency="Q",
                ex_div=5,
                fixing_method="rfr_observation_shift",
                method_param=3,
            )

    def test_accrued_no_fixings_in_period(self):
        bond = FloatRateNote(
            effective=dt(2010, 3, 16),
            termination=dt(2017, 3, 16),
            frequency="Q",
            convention="Act365f",
            ex_div=0,
            float_spread=0,
            fixing_method="rfr_observation_shift",
            fixings=NoInput(0),
            method_param=0,
            spread_compound_method="none_simple",
            calendar=NoInput(0),
        )
        result = bond.accrued(dt(2010, 3, 16))
        assert result == 0.0

    def test_float_rate_bond_analytic_delta(self):
        frn = FloatRateNote(
            effective=dt(2010, 6, 7),
            termination=dt(2015, 12, 7),
            frequency="S",
            calendar="ldn",
            currency="gbp",
            convention="ActActICMA",
            ex_div=7,
            float_spread=100,
            notional=-1000000,
            settle=0,
            fixing_method="ibor",
            fixings=2.0,
        )
        curve = Curve({dt(2010, 11, 25): 1.0, dt(2015, 12, 7): 1.0})
        result = frn.analytic_delta(curve)
        expected = -550.0
        assert abs(result - expected) < 1e-6

        frn.kwargs['settle'] = 2
        result = frn.analytic_delta(curve)  # bond is ex div on settle 27th Nov 2010
        expected = -500.0  # bond has dropped a 6m coupon payment
        assert abs(result - expected) < 1e-6

    @pytest.mark.parametrize(
        "metric, spd, exp",
        [
            ("clean_price", 0.0, 100),
            ("dirty_price", 0.0, 100),
            ("clean_price", 50.0, 99.99601798513253),
            ("dirty_price", 50.0, 100.03848373855718),
        ],
    )
    def test_float_rate_bond_forward_prices(self, metric, spd, exp):
        fixings = Series(
            data=2.0,
            index=date_range(start=dt(2007, 1, 1), end=dt(2010, 2, 28), freq=get_calendar("bus"))
        )
        bond = FloatRateNote(
            effective=dt(2007, 1, 1),
            termination=dt(2017, 1, 1),
            frequency="S",
            convention="Act365f",
            ex_div=3,
            float_spread=spd,
            fixing_method="rfr_observation_shift",
            calendar="bus",
            fixings=fixings,
            method_param=5,
            spread_compound_method="none_simple",
            settle=2,
        )
        curve = Curve(
            {dt(2010, 3, 1): 1.0, dt(2017, 1, 1): 1.0},
            convention="act365f",
            calendar="bus"
        )
        disc_curve = curve.shift(spd)

        result = bond.rate(
            curves=[curve, disc_curve], metric=metric, forward_settlement=dt(2010, 8, 1)
        )
        assert abs(result - exp) < 1e-8

    def test_float_rate_bond_forward_accrued(self):
        bond = FloatRateNote(
            effective=dt(2007, 1, 1),
            termination=dt(2017, 1, 1),
            frequency="S",
            convention="Act365f",
            ex_div=3,
            float_spread=0,
            fixing_method="rfr_observation_shift",
            method_param=5,
            spread_compound_method="none_simple",
            settle=2,
        )
        curve = Curve({dt(2010, 3, 1): 1.0, dt(2017, 1, 1): 0.9}, convention="act365f")
        # disc_curve = curve.shift(0)
        result = bond.accrued(dt(2010, 8, 1), curve=curve)
        expected = 0.13083715795372267
        assert abs(result - expected) < 1e-8

    def test_rate_raises(self):
        bond = FloatRateNote(
            effective=dt(2007, 1, 1),
            termination=dt(2017, 1, 1),
            frequency="S",
            convention="Act365f",
            ex_div=3,
            float_spread=0.0,
            fixing_method="rfr_observation_shift",
            method_param=5,
            spread_compound_method="none_simple",
            settle=2,
        )

        with pytest.raises(ValueError, match="`metric` must be in"):
            bond.rate(NoInput(0), metric="BAD")

    def test_forecast_ibor(self, curve):
        f_curve = LineCurve({dt(2022, 1, 1): 3.0, dt(2022, 2, 1): 4.0})
        frn = FloatRateNote(
            effective=dt(2022, 2, 1),
            termination="3m",
            frequency="Q",
            fixing_method="ibor",
            method_param=0,
        )
        result = frn.accrued(dt(2022, 2, 5), curve=f_curve)
        expected = 0.044444444
        assert abs(result - expected) < 1e-4

    @pytest.mark.parametrize("price, tol", [
        (98.0, 1e-7),
        (95.0, 1e-5),
        (90.0, 1e-3),
        (80.0, 1e-2)
    ])
    def test_oaspread(self, price, tol):
        bond = FloatRateNote(
            effective=dt(1998, 12, 7),
            termination=dt(2008, 12, 7),
            frequency="q",
            fixing_method="rfr_payment_delay",
            fixings=[[4.0]],
        )
        curve = Curve({dt(1998, 12, 7): 1.0, dt(2015, 12, 7): 0.75})
        # result = bond.rate(curve, metric="clean_price") = 99.999999999999953
        result = bond.oaspread(curve, price=price)
        curve_z = curve.shift(result, composite=False)
        result = bond.rate([curve, curve_z], metric="clean_price")
        assert abs(result - price) < tol

    def test_curve_is_used_for_future_accrued_forecasting_when_needed(self):
        frn = FloatRateNote(
            effective=dt(2022, 1, 1),
            termination="1Y",
            frequency="Q",
            settle=3,
        )
        curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.95})
        # frn settles in 3 days, has three days of accrued. Fixings required are forecast

    def test_settle_method_param_combinations(self):
        # for RFR when method_param is less than settle curve based pricing methods will
        # require forecasting from RFR curve to correctly calculate the accrued.
        fixings = Series(
            [2.0, 3.0, 4.0, 5.0, 6.0],
            index=[dt(2022, 1, 2), dt(2022, 1, 3), dt(2022, 1, 4), dt(2022, 1, 5), dt(2022, 1, 6)]
        )
        frn = FloatRateNote(
            effective=dt(2022, 1, 5),
            termination="1Y",
            frequency="Q",
            settle=3,
            method_param=2,
            fixing_method="rfr_observation_shift",
            fixings=fixings,
            convention="Act365F",
            ex_div=1,
        )
        curve = Curve({dt(2022, 1, 7): 1.0, dt(2023, 1, 7): 0.95})

        # Case1: All fixings are known and are published
        # in this case a Curve is not required and is not given
        result = frn.accrued(settlement=dt(2022, 1, 9))
        assert abs(result - 0.04932400) < 1e-6

        # Case2: Some fixings are unknown and must be forecast by a curve.
        # None are supplied so a UserWarning is generated and they are forward filled.
        with pytest.warns(UserWarning):
            result = frn.accrued(settlement=dt(2022, 1, 10))
            assert abs(result - 0.065770465) < 1e-6

        # Case3: Some fixings are unknown and must be forecast by a curve.
        # A curve is given so this is used to forecast the values.
        result = frn.accrued(settlement=dt(2022, 1, 10), curve=curve)
        assert abs(result - 0.06319248) < 1e-6

        # Case4: The bond settles on Issue date and there is no accrued if curve supplied or not
        result1 = frn.accrued(settlement=dt(2022, 1, 5))
        result2 = frn.accrued(settlement=dt(2022, 1, 5), curve=curve)
        assert abs(result1) < 1e-6 and abs(result2) < 1e-6

        # Case5: The bond settles on a coupon date and there is no accrued if curve supplied or not
        result1 = frn.accrued(settlement=dt(2022, 4, 5))
        result2 = frn.accrued(settlement=dt(2022, 4, 5), curve=curve)
        assert abs(result1) < 1e-6 and abs(result2) < 1e-6

        # Case6: Bond settles on issue date and there is no accrued. No fixings are input
        frn_no_fixings = FloatRateNote(
            effective=dt(2022, 1, 5),
            termination="1Y",
            frequency="Q",
            settle=3,
            method_param=2,
            fixing_method="rfr_observation_shift",
            convention="Act365F",
            ex_div=1,
        )
        result1 = frn_no_fixings.accrued(settlement=dt(2022, 1, 5))
        result2 = frn_no_fixings.accrued(settlement=dt(2022, 1, 5), curve=curve)
        assert abs(result1) < 1e-6 and abs(result2) < 1e-6

        # Case7: Bond settles a few days forward(settle) no previous fixings are given, all
        # can be forecast from curve
        frn_no_fixings = FloatRateNote(
            effective=dt(2022, 1, 7),
            termination="1Y",
            frequency="Q",
            settle=3,
            method_param=0,
            fixing_method="rfr_observation_shift",
            convention="Act365F",
            ex_div=1,
        )
        result = frn_no_fixings.accrued(settlement=dt(2022, 1, 10), curve=curve)
        assert abs(result - 0.04159011) < 1e-6

        # Case8: bond settles a few days forward, no fixings are given and no curve. Must error.
        with pytest.raises(TypeError, match="`fixings` or `curve` are not available for"):
            frn_no_fixings.accrued(settlement=dt(2022, 1, 10))


class TestBondFuture:
    @pytest.mark.parametrize(
        "delivery, mat, coupon, exp",
        [
            (dt(2023, 6, 12), dt(2032, 2, 15), 0.0, 0.603058),
            (dt(2023, 6, 12), dt(2032, 8, 15), 1.7, 0.703125),
            (dt(2023, 6, 12), dt(2033, 2, 15), 2.3, 0.733943),
            (dt(2023, 9, 11), dt(2032, 8, 15), 1.7, 0.709321),
            (dt(2023, 9, 11), dt(2033, 2, 15), 2.3, 0.739087),
            (dt(2023, 12, 11), dt(2032, 8, 15), 1.7, 0.715464),
            (dt(2023, 12, 11), dt(2033, 2, 15), 2.3, 0.744390),
        ],
    )
    def test_conversion_factors_eurex_bund(self, delivery, mat, coupon, exp):
        # The expected results are downloaded from the EUREX website
        # regarding precalculated conversion factors.
        # this test allows for an error in the cf < 1e-4.
        kwargs = dict(
            effective=dt(2020, 1, 1),
            stub="ShortFront",
            frequency="A",
            calendar="tgt",
            currency="eur",
            convention="ActActICMA",
        )
        bond1 = FixedRateBond(termination=mat, fixed_rate=coupon, **kwargs)

        fut = BondFuture(delivery=delivery, coupon=6.0, basket=[bond1])
        result = fut.cfs
        assert abs(result[0] - exp) < 1e-4

    @pytest.mark.parametrize(
        "mat, coupon, exp",
        [
            (dt(2032, 6, 7), 4.25, 1.0187757),
            (dt(2033, 7, 31), 0.875, 0.7410593),
            (dt(2034, 9, 7), 4.5, 1.0449380),
            (dt(2035, 7, 31), 0.625, 0.6773884),
            (dt(2036, 3, 7), 4.25, 1.0247516),
        ],
    )
    def test_conversion_factors_ice_gilt(self, mat, coupon, exp):
        # The expected results are downloaded from the ICE LIFFE website
        # regarding precalculated conversion factors.
        # this test allows for an error in the cf < 1e-6.
        kwargs = dict(
            effective=dt(2020, 1, 1),
            stub="ShortFront",
            frequency="S",
            calendar="ldn",
            currency="gbp",
            convention="ActActICMA",
            ex_div=7,
        )
        bond1 = FixedRateBond(termination=mat, fixed_rate=coupon, **kwargs)

        fut = BondFuture(delivery=(dt(2023, 6, 1), dt(2023, 6, 30)), coupon=4.0, basket=[bond1])
        result = fut.cfs
        assert abs(result[0] - exp) < 1e-6

    @pytest.mark.parametrize(
        "mat, coupon, calc_mode, exp",
        [
            (dt(2010, 10, 31), 1.5, "ust_short", 0.9229),
            (dt(2013, 10, 31), 2.75, "ust_short", 0.8653),
            (dt(2018, 11, 15), 3.75, "ust_long", 0.8357),
            (dt(2038, 5, 15), 4.5, "ust_long", 0.7943),
        ],
    )
    def test_conversion_factors_cme_treasury(self, mat, coupon, calc_mode, exp):
        # The expected results are downloaded from the CME website
        # regarding precalculated conversion factors.
        # this test allows for an error in the cf < 1e-6.
        kwargs = dict(
            effective=dt(2005, 1, 1),
            spec="ust",
        )
        bond1 = FixedRateBond(termination=mat, fixed_rate=coupon, **kwargs)

        fut = BondFuture(
            delivery=(dt(2008, 12, 1), dt(2008, 12, 29)),
            coupon=6.0,
            basket=[bond1],
            calc_mode=calc_mode,
        )
        result = fut.cfs
        assert abs(result[0] - exp) < 1e-6

    def test_dlv_screen_print(self):
        kws = dict(ex_div=7, frequency="S", convention="ActActICMA", calendar=NoInput(0))
        bonds = [
            FixedRateBond(dt(1999, 1, 1), dt(2009, 12, 7), fixed_rate=5.75, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2011, 7, 12), fixed_rate=9.00, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2010, 11, 25), fixed_rate=6.25, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2012, 8, 6), fixed_rate=9.00, **kws),
        ]
        future = BondFuture(delivery=(dt(2000, 6, 1), dt(2000, 6, 30)), coupon=7.0, basket=bonds)
        result = future.dlv(
            future_price=112.98,
            prices=[102.732, 131.461, 107.877, 134.455],
            repo_rate=6.24,
            settlement=dt(2000, 3, 16),
            convention="Act365f",
        )
        expected = DataFrame(
            {
                "Bond": [
                    "5.750% 07-12-2009",
                    "9.000% 12-07-2011",
                    "6.250% 25-11-2010",
                    "9.000% 06-08-2012",
                ],
                "Price": [102.732, 131.461, 107.877, 134.455],
                "YTM": [5.384243, 5.273217, 5.275481, 5.193851],
                "C.Factor": [0.914225, 1.152571, 0.944931, 1.161956],
                "Gross Basis": [-0.557192, 1.243582, 1.118677, 3.177230],
                "Implied Repo": [7.381345, 3.564685, 2.199755, -1.414670],
                "Actual Repo": [6.24, 6.24, 6.24, 6.24],
                "Net Basis": [-0.343654, 1.033668, 1.275866, 3.010371],
            }
        )
        assert_frame_equal(result, expected)

        result2 = future.dlv(
            future_price=112.98,
            prices=[102.732, 131.461, 107.877, 134.455],
            repo_rate=[6.24, 6.24, 6.24, 6.24],  # test individual repo input
            settlement=dt(2000, 3, 16),
            convention="Act365f",
        )
        assert_frame_equal(result2, expected)

    def test_notional(self):
        future = BondFuture(
            coupon=0, delivery=dt(2000, 6, 1), basket=[], nominal=100000, contracts=10
        )
        assert future.notional == -1e6

    def test_dirty_in_methods(self):
        kws = dict(ex_div=7, frequency="S", convention="ActActICMA", calendar=NoInput(0))
        bonds = [
            FixedRateBond(dt(1999, 1, 1), dt(2009, 12, 7), fixed_rate=5.75, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2011, 7, 12), fixed_rate=9.00, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2010, 11, 25), fixed_rate=6.25, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2012, 8, 6), fixed_rate=9.00, **kws),
        ]
        future = BondFuture(delivery=(dt(2000, 6, 1), dt(2000, 6, 30)), coupon=7.0, basket=bonds)
        prices = [102.732, 131.461, 107.877, 134.455]
        dirty_prices = [
            price + future.basket[i].accrued(dt(2000, 3, 16)) for i, price in enumerate(prices)
        ]
        result = future.gross_basis(112.98, dirty_prices, dt(2000, 3, 16), True)
        expected = future.gross_basis(112.98, prices, dt(2000, 3, 16), False)
        assert result == expected

    def test_delivery_in_methods(self):
        kws = dict(ex_div=7, frequency="S", convention="ActActICMA", calendar=NoInput(0))
        bonds = [
            FixedRateBond(dt(1999, 1, 1), dt(2009, 12, 7), fixed_rate=5.75, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2011, 7, 12), fixed_rate=9.00, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2010, 11, 25), fixed_rate=6.25, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2012, 8, 6), fixed_rate=9.00, **kws),
        ]
        future = BondFuture(delivery=(dt(2000, 6, 1), dt(2000, 6, 30)), coupon=7.0, basket=bonds)
        prices = [102.732, 131.461, 107.877, 134.455]
        expected = future.net_basis(112.98, prices, 6.24, dt(2000, 3, 16))
        result = future.net_basis(112.98, prices, 6.24, dt(2000, 3, 16), delivery=dt(2000, 6, 30))
        assert result == expected

        expected = future.implied_repo(112.98, prices, dt(2000, 3, 16))
        result = future.implied_repo(112.98, prices, dt(2000, 3, 16), delivery=dt(2000, 6, 30))
        assert result == expected

        expected = future.ytm(112.98)
        result = future.ytm(112.98, delivery=dt(2000, 6, 30))
        assert result == expected

        expected = future.duration(112.98)
        result = future.duration(112.98, delivery=dt(2000, 6, 30))
        assert result == expected

    def test_ctd_index(self):
        kws = dict(ex_div=7, frequency="S", convention="ActActICMA", calendar=NoInput(0))
        bonds = [
            FixedRateBond(dt(1999, 1, 1), dt(2009, 12, 7), fixed_rate=5.75, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2011, 7, 12), fixed_rate=9.00, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2010, 11, 25), fixed_rate=6.25, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2012, 8, 6), fixed_rate=9.00, **kws),
        ]
        future = BondFuture(delivery=(dt(2000, 6, 1), dt(2000, 6, 30)), coupon=7.0, basket=bonds)
        prices = [102.732, 131.461, 107.877, 134.455]
        assert future.ctd_index(112.98, prices, dt(2000, 3, 16)) == 0

    @pytest.mark.parametrize("metric, expected", [("future_price", 112.98), ("ytm", 5.301975)])
    @pytest.mark.parametrize("delivery", [NoInput(0), dt(2000, 6, 30)])
    def test_futures_rates(self, metric, expected, delivery):
        curve = Curve(
            nodes={
                dt(2000, 3, 15): 1.0,
                dt(2000, 6, 30): 1.0,
                dt(2009, 12, 7): 1.0,
                dt(2010, 11, 25): 1.0,
                dt(2011, 7, 12): 1.0,
                dt(2012, 8, 6): 1.0,
            },
            id="gilt_curve",
            convention="act365f",
        )
        kws = dict(
            ex_div=7,
            frequency="S",
            convention="ActActICMA",
            calendar=NoInput(0),
            settle=1,
            curves="gilt_curve",
        )
        bonds = [
            FixedRateBond(dt(1999, 1, 1), dt(2009, 12, 7), fixed_rate=5.75, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2011, 7, 12), fixed_rate=9.00, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2010, 11, 25), fixed_rate=6.25, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2012, 8, 6), fixed_rate=9.00, **kws),
        ]
        solver = Solver(
            curves=[curve],
            instruments=[
                IRS(
                    dt(2000, 3, 15),
                    dt(2000, 6, 30),
                    "A",
                    convention="act365f",
                    curves="gilt_curve",
                )
            ]
            + bonds,
            s=[7.381345, 102.732, 131.461, 107.877, 134.455],
        )  # note the repo rate as defined by 'gilt_curve' is set to analogue implied
        future = BondFuture(
            coupon=7.0,
            delivery=(dt(2000, 6, 1), dt(2000, 6, 30)),
            basket=bonds,
        )
        result = future.rate(NoInput(0), solver, metric=metric, delivery=delivery)
        assert abs(result - expected) < 1e-3

    def test_future_rate_raises(self):
        kws = dict(
            ex_div=7,
            frequency="S",
            convention="ActActICMA",
            calendar=NoInput(0),
            settle=1,
            curves="gilt_curve",
        )
        bonds = [
            FixedRateBond(dt(1999, 1, 1), dt(2009, 12, 7), fixed_rate=5.75, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2011, 7, 12), fixed_rate=9.00, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2010, 11, 25), fixed_rate=6.25, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2012, 8, 6), fixed_rate=9.00, **kws),
        ]
        future = BondFuture(
            coupon=7.0,
            delivery=(dt(2000, 6, 1), dt(2000, 6, 30)),
            basket=bonds,
        )
        with pytest.raises(ValueError, match="`metric`"):
            future.rate(metric="badstr")

    def test_futures_npv(self):
        curve = Curve(
            nodes={
                dt(2000, 3, 15): 1.0,
                dt(2000, 6, 30): 1.0,
                dt(2009, 12, 7): 1.0,
                dt(2010, 11, 25): 1.0,
                dt(2011, 7, 12): 1.0,
                dt(2012, 8, 6): 1.0,
            },
            id="gilt_curve",
            convention="act365f",
        )
        kws = dict(
            ex_div=7,
            frequency="S",
            convention="ActActICMA",
            calendar=NoInput(0),
            settle=1,
            curves="gilt_curve",
            currency="gbp",
        )
        bonds = [
            FixedRateBond(dt(1999, 1, 1), dt(2009, 12, 7), fixed_rate=5.75, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2011, 7, 12), fixed_rate=9.00, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2010, 11, 25), fixed_rate=6.25, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2012, 8, 6), fixed_rate=9.00, **kws),
        ]
        solver = Solver(
            curves=[curve],
            instruments=[
                IRS(
                    dt(2000, 3, 15),
                    dt(2000, 6, 30),
                    "A",
                    convention="act365f",
                    curves="gilt_curve",
                )
            ]
            + bonds,
            s=[7.381345, 102.732, 131.461, 107.877, 134.455],
            algorithm="gauss_newton",
        )  # note the repo rate as defined by 'gilt_curve' is set to analogue implied
        future = BondFuture(
            coupon=7.0,
            delivery=(dt(2000, 6, 1), dt(2000, 6, 30)),
            basket=bonds,
            nominal=100000,
            contracts=10,
            currency="gbp",
        )
        result = future.npv(NoInput(0), solver, local=False)
        expected = 1129798.770872
        assert abs(result - expected) < 1e-5

        result2 = future.npv(NoInput(0), solver, local=True)
        assert abs(result2["gbp"] - expected) < 1e-5

    @pytest.mark.parametrize("delivery", [NoInput(0), dt(2000, 6, 30)])
    def test_futures_duration_and_convexity(self, delivery):
        kws = dict(
            ex_div=7,
            frequency="S",
            convention="ActActICMA",
            calendar=NoInput(0),
            settle=1,
            curves="gilt_curve",
        )
        bonds = [
            FixedRateBond(dt(1999, 1, 1), dt(2009, 12, 7), fixed_rate=5.75, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2011, 7, 12), fixed_rate=9.00, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2010, 11, 25), fixed_rate=6.25, **kws),
            FixedRateBond(dt(1999, 1, 1), dt(2012, 8, 6), fixed_rate=9.00, **kws),
        ]
        future = BondFuture(
            coupon=7.0,
            delivery=(dt(2000, 6, 1), dt(2000, 6, 30)),
            basket=bonds,
        )
        result = future.duration(112.98, delivery=delivery)[0]
        expected = 8.20178546111
        assert abs(result - expected) < 1e-3

        expected = (
            future.duration(112.98, delivery=delivery)[0]
            - future.duration(112.98 - result / 100, delivery=delivery)[0]
        )
        result2 = future.convexity(112.98, delivery=delivery)[0]
        assert abs(result2 - expected * 100) < 1e-3

        # Bond future duration which is not risk is not adjusted by CFs
        result = future.duration(112.98, delivery=delivery, metric="modified")[0]
        expected = 7.23419455163
        assert abs(result - expected) < 1e-3