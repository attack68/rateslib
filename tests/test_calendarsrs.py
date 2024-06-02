import pytest
import context
from rateslib.rateslibrs import Cal, UnionCal
from datetime import datetime as dt

@pytest.fixture()
def simple_cal():
    return Cal([dt(2015, 9, 5), dt(2015, 9, 7)], [5, 6])  # Saturday and Monday

@pytest.fixture()
def simple_union(simple_cal):
    return UnionCal([simple_cal], None)

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
        result = cal.add_days(dt(2015, 9, 4), 2, "F", True)
        assert result == expected

        expected = dt(2015, 9, 6)
        result = cal.add_days(dt(2015, 9, 5), 1, "NONE", True)
        assert result == expected

    def test_add_days_raises(self, simple_cal, simple_union):
        with pytest.raises(ValueError, match="`modifier` must be in {'F'"):
            simple_cal.add_days(dt(2015, 9, 5), 1, "bad", True)

        with pytest.raises(ValueError, match="`modifier` must be in {'F'"):
            simple_union.add_days(dt(2015, 9, 5), 1, "bad", True)

    @pytest.mark.parametrize("cal", ["basic", "union"])
    @pytest.mark.parametrize("start, days, mod, expected", [
        (dt(2015, 9, 4), 0, "F", dt(2015, 9, 4)),
        (dt(2015, 9, 4), 0, "P", dt(2015, 9, 4)),
        (dt(2015, 9, 4), 1, "F", dt(2015, 9, 8)),
        (dt(2015, 9, 8), -1, "P", dt(2015, 9, 4)),
        (dt(2015, 9, 5), 0, "P", dt(2015, 9, 4)),
        (dt(2015, 9, 5), 0, "F", dt(2015, 9, 8)),
        (dt(2015, 9, 4), -1, "P", dt(2015, 9, 3)),
        (dt(2015, 9, 8), 1, "F", dt(2015, 9, 9)),
        (dt(2015, 9, 5), -1, "P", dt(2015, 9, 3)),
        (dt(2015, 9, 5), 1, "F", dt(2015, 9, 9)),
    ])
    def test_add_bus_days(self, simple_cal, simple_union, cal, start, days, mod, expected):
        cal = simple_cal if cal == "basic" else simple_union

        result = cal.add_bus_days(start, days, mod, True)
        assert result == expected

    def test_add_bus_days_raises(self, simple_cal, simple_union):
        with pytest.raises(ValueError, match="`modifier` must be in {'F'"):
            simple_cal.add_bus_days(dt(2015, 9, 5), 1, "bad", True)

        with pytest.raises(ValueError, match="`modifier` must be in {'F'"):
            simple_union.add_bus_days(dt(2015, 9, 5), 1, "bad", True)