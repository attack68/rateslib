SPLINE_CORE_PY = False

if SPLINE_CORE_PY:
    from rateslib.splines.splines import (
        bsplev_single, bspldnev_single, PPSpline
    )
    PPSplineF64 = PPSpline
    PPSplineDual = PPSpline
    PPSplineDual2 = PPSpline
else:
    from rateslibrs import (
        PPSplineF64,
        PPSplineDual,
        PPSplineDual2,
        bsplev_single,
        bspldnev_single,
    )

    PPSplineF64 = PPSpline
    PPSplineDual = PPSpline
    PPSplineDual2 = PPSpline
