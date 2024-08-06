from rateslib.rs import (
    LinearInterpolator,
    LinearZeroRateInterpolator,
    LogLinearInterpolator,
)
from rateslib.rs import Curve as CurveRs

def _get_interpolation_enum(name: str):
    name_ = name.upper()
    if name_ == "log_linear":
        return LogLinearInterpolation()
    elif name_ == "linear":
        return LinearInterpolation()
    elif name_ == "linear_zero_rate":
        return LinearZeroRateInterpolation()
    else:
        raise ValueError("Interp `name` is invalid.")