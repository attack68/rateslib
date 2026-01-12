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
from pandas import Index
from rateslib import fixings
from rateslib.rs import Adjuster, Cal, Modifier, NamedCal, RollDay, UnionCal
from rateslib.scheduling import get_calendar
from rateslib.serialization import from_json


class TestRollDay:
    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        [
            (RollDay.IMM(), RollDay.IMM(), True),
            (RollDay.Day(20), RollDay.Day(20), True),
            (RollDay.Day(20), RollDay.Day(30), False),
            (RollDay.Day(31), RollDay.IMM(), False),
        ],
    )
    def test_equality(self, left, right, expected):
        result = left == right
        assert result is expected


@pytest.mark.parametrize(
    "modifier",
    [
        Modifier.Act,
        Modifier.F,
        Modifier.ModF,
        Modifier.P,
        Modifier.ModP,
    ],
)
def test_modifier_pickle(modifier) -> None:
    import pickle

    assert modifier == pickle.loads(pickle.dumps(modifier))


@pytest.fixture
def simple_cal():
    return Cal([dt(2015, 9, 5), dt(2015, 9, 7)], [5, 6])  # Saturday and Monday


@pytest.fixture
def simple_union(simple_cal):
    return UnionCal([simple_cal], None)


@pytest.fixture
def multi_union(simple_cal):
    add_cal = Cal([dt(2015, 9, 3), dt(2015, 9, 8)], [5, 6])
    return UnionCal([simple_cal, add_cal], None)


class TestCal:
    def test_cal_construct(self) -> None:
        cal = Cal([dt(2015, 9, 5), dt(2015, 9, 7)], [5, 6])
        UnionCal([cal], None)

    def test_cal_from_name(self):
        cal1 = Cal.from_name("ldn")
        cal2 = NamedCal("ldn")
        assert cal1 == cal2
        assert type(cal1) is not type(cal2)

    def test_is_business_day(self, simple_cal, simple_union) -> None:
        assert not simple_cal.is_bus_day(dt(2015, 9, 7))  # Monday Holiday
        assert simple_cal.is_bus_day(dt(2015, 9, 8))  # Tuesday
        assert not simple_cal.is_bus_day(dt(2015, 9, 12))  # Saturday

        assert not simple_union.is_bus_day(dt(2015, 9, 7))
        assert simple_union.is_bus_day(dt(2015, 9, 8))

    @pytest.mark.parametrize("cal", ["basic", "union"])
    def test_add_cal_days(self, simple_cal, simple_union, cal) -> None:
        cal = simple_cal if cal == "basic" else simple_union
        expected = dt(2015, 9, 8)
        result = cal.add_cal_days(dt(2015, 9, 4), 2, Adjuster.FollowingSettle())
        assert result == expected

        expected = dt(2015, 9, 6)
        result = cal.add_cal_days(dt(2015, 9, 5), 1, Adjuster.Actual())
        assert result == expected

    @pytest.mark.parametrize("cal", ["basic", "union"])
    @pytest.mark.parametrize(
        ("start", "days", "expected"),
        [
            (dt(2015, 9, 4), 0, dt(2015, 9, 4)),
            (dt(2015, 9, 4), 1, dt(2015, 9, 8)),
            (dt(2015, 9, 8), -1, dt(2015, 9, 4)),
            (dt(2015, 9, 4), -1, dt(2015, 9, 3)),
            (dt(2015, 9, 8), 1, dt(2015, 9, 9)),
        ],
    )
    def test_add_bus_days(self, simple_cal, simple_union, cal, start, days, expected) -> None:
        cal = simple_cal if cal == "basic" else simple_union

        result = cal.add_bus_days(start, days, True)
        assert result == expected

    def test_add_bus_days_raises(self, simple_cal, simple_union) -> None:
        with pytest.raises(ValueError, match="Cannot add business days"):
            simple_cal.add_bus_days(dt(2015, 9, 5), 1, True)

    @pytest.mark.parametrize("cal", ["basic", "union"])
    @pytest.mark.parametrize(
        ("start", "months", "expected"),
        [
            (dt(2015, 9, 4), 2, dt(2015, 11, 4)),
            (dt(2015, 9, 4), 36, dt(2018, 9, 4)),
        ],
    )
    def test_add_months(self, cal, simple_cal, simple_union, start, months, expected) -> None:
        cal = simple_cal if cal == "basic" else simple_union
        result = cal.add_months(start, months, Adjuster.FollowingSettle(), None)
        assert result == expected

    def test_pickle_cal(self, simple_cal) -> None:
        import pickle

        pickled_cal = pickle.dumps(simple_cal)
        pickle.loads(pickled_cal)

    def test_pickle_union(self, simple_union) -> None:
        import pickle

        pickled_cal = pickle.dumps(simple_union)
        pickle.loads(pickled_cal)

    @pytest.mark.parametrize(
        ("cal", "exp"),
        [
            ("basic", [dt(2015, 9, 5), dt(2015, 9, 7)]),
            ("union", [dt(2015, 9, 3), dt(2015, 9, 5), dt(2015, 9, 7), dt(2015, 9, 8)]),
        ],
    )
    def test_holidays(self, cal, exp, simple_cal, multi_union) -> None:
        cal = simple_cal if cal == "basic" else multi_union
        assert cal.holidays == exp

    # def test_rules(self):
    #     rules = get_calendar("tyo").rules
    #     assert rules[:10] == "Jan 1 (New"

    def test_tyo_cal(self) -> None:
        tokyo = get_calendar("tyo")
        assert tokyo.holidays[0] == dt(1970, 1, 1)

    def test_fed_cal(self) -> None:
        cal = get_calendar("fed")
        assert cal.holidays[0] == dt(1970, 1, 1)

    def test_wlg_cal(self):
        cal = get_calendar("wlg")
        assert cal.holidays[0] == dt(1970, 1, 1)

    def test_mum_cal(self):
        cal = get_calendar("mum")
        assert cal.holidays[0] == dt(1970, 1, 26)

    def test_json_round_trip(self, simple_cal) -> None:
        json = simple_cal.to_json()
        from_cal = from_json(json)
        assert simple_cal == from_cal

    def test_json_round_trip_union(self, multi_union) -> None:
        json = multi_union.to_json()
        from_cal = from_json(json)
        assert multi_union == from_cal

    def test_json_raises(self) -> None:
        with pytest.raises(ValueError, match="Could not create Class or Struct from given JSON"):
            from_json('{"Cal":{"holidays":[]}}')

        with pytest.raises(ValueError, match="Could not create Class or Struct from given JSON"):
            from_json('{"UnionCal":{"settlement_calendars":[]}}')

    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        [
            (Cal([], [5, 6]), Cal([], [5, 6]), True),
            (Cal([dt(2006, 1, 2)], [5, 6]), Cal([dt(2006, 1, 2)], [5, 6]), True),
            (Cal([dt(2006, 1, 2)], [5, 6]), Cal([dt(2007, 1, 2)], [5, 6]), False),
            (Cal([], [4, 6]), Cal([], [5, 6]), False),
            (UnionCal([Cal([], [5, 6])]), Cal([], [5, 6]), True),
            (UnionCal([Cal([dt(2006, 1, 2)], [5, 6])]), Cal([], [5, 6]), False),
            (
                UnionCal([Cal([dt(2006, 1, 2)], [5, 6])]),
                Cal([dt(2006, 1, 2)], [5, 6]),
                True,
            ),
            (
                UnionCal([Cal([dt(2006, 1, 2)], [5, 6]), Cal([dt(2006, 1, 3)], [5, 6])]),
                Cal([dt(2006, 1, 2), dt(2006, 1, 3)], [5, 6]),
                True,
            ),
            (
                UnionCal([Cal([dt(2006, 1, 2)], [5, 6]), Cal([dt(2006, 1, 3)], [5, 6])]),
                UnionCal([Cal([dt(2006, 1, 2), dt(2006, 1, 3)], [5, 6])]),
                True,
            ),
        ],
    )
    def test_equality(self, left, right, expected) -> None:
        assert (left == right) is expected
        assert (right == left) is expected

    def test_attributes(self) -> None:
        ncal = get_calendar("tgt,LDN|Fed")
        assert ncal.name == "tgt,ldn|fed"
        assert isinstance(ncal.union_cal, UnionCal)
        assert len(ncal.union_cal.calendars) == 2
        assert len(ncal.union_cal.settlement_calendars) == 1

        ncal = get_calendar("tgt")
        assert ncal.union_cal.settlement_calendars is None

    def test_adjusts(self, simple_cal):
        dates = [dt(2015, 9, 4), dt(2015, 9, 5), dt(2015, 9, 6), dt(2015, 9, 7)]
        result = simple_cal.adjusts(dates, Adjuster.Following())
        expected = [dt(2015, 9, 4), dt(2015, 9, 8), dt(2015, 9, 8), dt(2015, 9, 8)]
        assert result == expected

    def test_roll(self, simple_cal):
        result = simple_cal.roll(dt(2015, 9, 5), "F", False)
        assert result == dt(2015, 9, 8)


class TestUnionCal:
    def test_week_mask(self, multi_union) -> None:
        result = multi_union.week_mask
        assert result == {5, 6}

    def test_adjusts(self, simple_union):
        dates = [dt(2015, 9, 4), dt(2015, 9, 5), dt(2015, 9, 6), dt(2015, 9, 7)]
        result = simple_union.adjusts(dates, Adjuster.Following())
        expected = [dt(2015, 9, 4), dt(2015, 9, 8), dt(2015, 9, 8), dt(2015, 9, 8)]
        assert result == expected

    def test_roll(self, simple_union):
        result = simple_union.roll(dt(2015, 9, 5), "F", False)
        assert result == dt(2015, 9, 8)


class TestNamedCal:
    def test_equality_named_cal(self) -> None:
        cal = get_calendar("fed", named=False)
        ncal = NamedCal("fed")
        assert cal == ncal
        assert ncal == cal

        ucal = get_calendar("ldn,tgt|fed", named=False)
        ncal = NamedCal("ldn,tgt|fed")
        assert ucal == ncal
        assert ncal == ucal

    def test_adjusts(self):
        ncal = NamedCal("fed")
        dates = [dt(2015, 9, 4), dt(2015, 9, 5), dt(2015, 9, 6), dt(2015, 9, 7)]
        result = ncal.adjusts(dates, Adjuster.Following())
        expected = [dt(2015, 9, 4), dt(2015, 9, 8), dt(2015, 9, 8), dt(2015, 9, 8)]
        assert result == expected

    def test_roll(self):
        ncal = NamedCal("fed")
        result = ncal.roll(dt(2015, 9, 5), "F", False)
        assert result == dt(2015, 9, 8)


@pytest.mark.parametrize(
    ("datafile", "calendar", "known_exceptions"),
    [
        ("usd_rfr", "nyc", []),
        ("gbp_rfr", "ldn", []),
        ("cad_rfr", "tro", []),
        ("eur_rfr", "tgt", []),
        ("jpy_rfr", "tyo", []),
        ("sek_rfr", "stk", []),
        ("nok_rfr", "osl", []),
        ("aud_rfr", "syd", []),
        ("inr_rfr", "mum", []),
    ],
)
def test_calendar_against_historical_fixings(datafile, calendar, known_exceptions):
    fixings_ = fixings[datafile][1]
    calendar_ = get_calendar(calendar)
    bus_days = Index(calendar_.bus_date_range(fixings_.index[0], fixings_.index[-1]))
    diff = fixings_.index.symmetric_difference(bus_days)

    errors = 0
    if len(diff) != 0:
        print(f"{calendar} for {datafile}")
        for i, date in enumerate(diff):
            if date in known_exceptions:
                continue
            elif date in fixings_.index:
                print(f"{date} exists in fixings: does calendar wrongly classify as a holiday?")
            else:
                # print(f'Holiday("adhoc{i}", year={date.year}, month={date.month}, day={date.day}),')  # noqa: E501
                print(f"{date} exists in calendar: should this date be classified as a holiday?")
            errors += 1

    assert errors == 0


class TestAdjuster:
    def test_adjusts(self, simple_cal):
        dates = [dt(2015, 9, 4), dt(2015, 9, 5), dt(2015, 9, 6), dt(2015, 9, 7)]
        result = Adjuster.Following().adjusts(dates, simple_cal)
        expected = [dt(2015, 9, 4), dt(2015, 9, 8), dt(2015, 9, 8), dt(2015, 9, 8)]
        assert result == expected
