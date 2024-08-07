from rateslib.rs import (
    Curve as CurveObj,
    LinearInterpolator,
    LinearZeroRateInterpolator,
    LogLinearInterpolator,
)


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
