import pytest
import context
from rateslib.calendars import _get_modifier, get_calendar
from rateslib.rs import Cal, UnionCal, Modifier, RollDay
from datetime import datetime as dt


@pytest.fixture()
def simple_cal():
    return Cal([dt(2015, 9, 5), dt(2015, 9, 7)], [5, 6])  # Saturday and Monday


@pytest.fixture()
def simple_union(simple_cal):
    return UnionCal([simple_cal], None)


@pytest.fixture()
def multi_union(simple_cal):
    add_cal = Cal([dt(2015, 9, 3), dt(2015, 9, 8)], [5, 6])
    return UnionCal([simple_cal, add_cal], None)


class TestCal:

    def test_cal_construct(self):
        cal = Cal([dt(2015, 9, 5), dt(2015, 9, 7)], [5, 6])
        UnionCal([cal], None)

    def test_is_business_day(self, simple_cal, simple_union):
        assert not simple_cal.is_bus_day(dt(2015, 9, 7))
        assert simple_cal.is_bus_day(dt(2015, 9, 8))
        assert not simple_union.is_bus_day(dt(2015, 9, 7))
        assert simple_union.is_bus_day(dt(2015, 9, 8))

    @pytest.mark.parametrize("cal", ["basic", "union"])
    def test_add_days(self, simple_cal, simple_union, cal):
        cal = simple_cal if cal == "basic" else simple_union
        expected = dt(2015, 9, 8)
        result = cal.add_days(dt(2015, 9, 4), 2, Modifier.F, True)
        assert result == expected

        expected = dt(2015, 9, 6)
        result = cal.add_days(dt(2015, 9, 5), 1, Modifier.Act, True)
        assert result == expected

    def test_get_modifier_raises(self, simple_cal, simple_union):
        with pytest.raises(ValueError, match="`modifier` must be in {'F'"):
            _get_modifier("bad")

    @pytest.mark.parametrize("cal", ["basic", "union"])
    @pytest.mark.parametrize("start, days, expected", [
        (dt(2015, 9, 4), 0, dt(2015, 9, 4)),
        (dt(2015, 9, 4), 0, dt(2015, 9, 4)),
        (dt(2015, 9, 4), 1, dt(2015, 9, 8)),
        (dt(2015, 9, 8), -1, dt(2015, 9, 4)),
        (dt(2015, 9, 4), -1, dt(2015, 9, 3)),
        (dt(2015, 9, 8), 1, dt(2015, 9, 9)),
    ])
    def test_add_bus_days(self, simple_cal, simple_union, cal, start, days, expected):
        cal = simple_cal if cal == "basic" else simple_union

        result = cal.add_bus_days(start, days, True)
        assert result == expected

    def test_add_bus_days_raises(self, simple_cal, simple_union):
        with pytest.raises(ValueError, match="Cannot add business days"):
            simple_cal.add_bus_days(dt(2015, 9, 5), 1, True)

    @pytest.mark.parametrize("cal", ["basic", "union"])
    @pytest.mark.parametrize("start, months, expected", [
        (dt(2015, 9, 4), 2, dt(2015, 11, 4)),
        (dt(2015, 9, 4), 36, dt(2018, 9, 4)),
    ])
    def test_add_months(self, cal, simple_cal, simple_union, start, months, expected):
        cal = simple_cal if cal == "basic" else simple_union
        result = cal.add_months(start, months, Modifier.F, RollDay.Unspecified(), True)
        assert result == expected

    def test_pickle_cal(self, simple_cal):
        import pickle
        pickled_cal = pickle.dumps(simple_cal)
        pickle.loads(pickled_cal)

    def test_pickle_union(self, simple_union):
        import pickle
        pickled_cal = pickle.dumps(simple_union)
        pickle.loads(pickled_cal)

    @pytest.mark.parametrize("cal, exp", [
        ("basic", [dt(2015, 9, 5), dt(2015, 9, 7)]),
        ("union", [dt(2015, 9, 3), dt(2015, 9, 5), dt(2015, 9, 7), dt(2015, 9, 8)]),
    ])
    def test_holidays(self, cal, exp, simple_cal, multi_union):
        cal = simple_cal if cal == "basic" else multi_union
        assert cal.holidays == exp

    # def test_rules(self):
    #     rules = get_calendar("tyo").rules
    #     assert rules[:10] == "Jan 1 (New"

    def test_tyo_cal(self):
        tokyo = get_calendar("tyo")
        assert tokyo.holidays[0] == dt(1970,1,1)
