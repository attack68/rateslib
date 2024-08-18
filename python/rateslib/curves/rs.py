from datetime import datetime as dt

from rateslib.default import NoInput, _drb
from rateslib.dual import _get_adorder
from rateslib.rs import Curve as CurveObj  # noqa: F401
from rateslib.rs import (
    FlatBackwardInterpolator,
    FlatForwardInterpolator,
    LinearInterpolator,
    LinearZeroRateInterpolator,
    LogLinearInterpolator,
)


class CurveRs:
    def __init__(self, nodes, interpolation, id, ad, index_base=NoInput(0)):
        interpolation = _get_interpolator(interpolation)
        self.obj = CurveObj(nodes, interpolation, _get_adorder(ad), id, _drb(None, index_base))

    def to_json(self):
        return '{"Py":' + self.obj.to_json() + "}"

    @classmethod
    def __init_from_obj__(cls, obj):
        new = cls({dt(2000, 1, 1): 1.0}, "linear", "_", 0)
        new.obj = obj
        return new

    def __eq__(self, other):
        if not isinstance(other, CurveRs):
            return False
        return self.obj.__eq__(other.obj)


def _get_interpolator(name: str):
    name_ = name.lower()
    if name_ == "log_linear":
        return LogLinearInterpolator()
    elif name_ == "linear":
        return LinearInterpolator()
    elif name_ == "linear_zero_rate":
        return LinearZeroRateInterpolator()
    elif name_ == "flat_forward":
        return FlatForwardInterpolator()
    elif name_ == "flat_backward":
        return FlatBackwardInterpolator()
    else:
        raise ValueError("Interpolator `name` is invalid.")
