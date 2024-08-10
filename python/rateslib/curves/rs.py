from rateslib.rs import Curve as CurveObj  # noqa: F401
from rateslib.rs import (
    LinearInterpolator,
    LinearZeroRateInterpolator,
    LogLinearInterpolator,
)
from rateslib.default import _make_py_json
from rateslib.dual import _get_adorder


class CurveRs:

    def __init__(self, nodes, interpolation, id, ad):
        interpolation = _get_interpolator(interpolation)
        self.obj = CurveObj(nodes, interpolation, _get_adorder(ad), id)

    def to_json(self):
        return _make_py_json(self.obj.to_json(), "Curve")


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
