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
