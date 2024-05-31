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


def evaluate(
    spline: Union[PPSplineF64, PPSplineDual, PPSplineDual2],
    x: Union[float, Dual, Dual2],
    m: int = 0,
) -> Union[float, Dual, Dual2]:
    """
    Evaluate a single x-axis data point, or a derivative value, on a *Spline*.

    This method automatically calls :meth:`~rateslib.splines.PPSplineF64.ppdnev_single`,
    :meth:`~rateslib.splines.PPSplineF64.ppdnev_single_dual` or
    :meth:`~rateslib.splines.PPSplineF64.ppdnev_single_dual2`  based on the input form of ``x``.

    This method is AD safe.

    Parameters
    ----------
    spline: PPSplineF64, PPSplineDual, PPSplineDual2
        The *Spline* on which to evaluate the data point.
    x: float, Dual, Dual2
        The x-axis data point to evaluate.
    m: int, optional
        The order of derivative to evaluate. If seeking value only use *m=0*.

    Returns
    -------
    float, Dual, Dual2
    """
    if isinstance(x, Dual):
        return spline.ppdnev_single_dual(x, m)
    elif isinstance(x, Dual2):
        return spline.ppdnev_single_dual2(x, m)
    else:
        return spline.ppdnev_single(x, m)
