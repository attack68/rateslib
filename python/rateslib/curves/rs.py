from rateslib.rs import (
    LinearInterpolation,
    LinearZeroRateInterpolation,
    LogLinearInterpolation,
)


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