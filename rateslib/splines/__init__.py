from rateslib.dual import DUAL_CORE_PY, Dual, Dual2
from typing import Union

if DUAL_CORE_PY:
    from rateslib.splines.splines import bsplev_single, bspldnev_single, PPSpline

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


def _interpolate(
    spline: Union[PPSplineF64, PPSplineDual, PPSplineDual2],
    x: Union[float, Dual, Dual2],
    m: int
):
    if isinstance(x, Dual):
        return spline.ppdnev_single_dual(x, m)
    elif isinstance(x, Dual2):
        return spline.ppdnev_single_dual2(x, m)
    else:
        return spline.ppdnev_single(x, m)

