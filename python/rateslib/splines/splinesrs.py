from rateslib.rateslibrs import (
    PPSplineF64,
    PPSplineDual,
    PPSplineDual2,
    bsplev_single,
    bspldnev_single,
)

PPSplineF64.__doc__ = "Piecewise polynomial spline composed of float values on the x and y axes."
PPSplineDual.__doc__ = "Piecewise polynomial spline composed of float values on the x-axis and Dual values on the y-axis."
PPSplineDual2.__doc__ = "Piecewise polynomial spline composed of float values on the x-axis and Dual2 values on the y-axis."
