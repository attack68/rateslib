from rateslib.rs import Curve as CurveObj  # noqa: F401
from rateslib.rs import (
    LinearInterpolator,
    LinearZeroRateInterpolator,
    LogLinearInterpolator,
)
from rateslib.default import _make_py_json
from rateslib.dual import _get_adorder
from datetime import datetime as dt


class CurveRs:

    def __init__(self, nodes, interpolation, id, ad):
        interpolation = _get_interpolator(interpolation)
        self.obj = CurveObj(nodes, interpolation, _get_adorder(ad), id)

    def to_json(self):
        return _make_py_json(self.obj.to_json(), "Curve")

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
    else:
        raise ValueError("Interpolator `name` is invalid.")
