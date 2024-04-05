from rateslib.dual import DUAL_CORE_PY

if DUAL_CORE_PY:
    from rateslib.splines.splines import (
        bsplev_single, bspldnev_single, PPSpline
    )
    PPSplineF64 = PPSpline
    PPSplineDual = PPSpline
    PPSplineDual2 = PPSpline
else:
    from rateslib.splines.splinesrs import (
        PPSplineF64,
        PPSplineDual,
        PPSplineDual2,
        bsplev_single,
        bspldnev_single,
    )
    # for legacy reasons allow a PPSpline class which allows only f64 datatypes.
    from rateslib.splines.splinesrs import PPSplineF64 as PPSpline
