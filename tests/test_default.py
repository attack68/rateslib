import pytest
from pandas import Series

import context
from rateslib import defaults, default_context


@pytest.mark.parametrize("name", ["estr", "sonia", "sofr", "swestr", "nowa"])
def test_fixings(name):
    result = getattr(defaults.fixings, name, None)
    assert isinstance(result, Series)
