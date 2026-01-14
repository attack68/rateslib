# SPDX-License-Identifier: LicenseRef-Rateslib-Dual
#
# Copyright (c) 2026 Siffrorna Technology Limited
#
# Dual-licensed: Free Educational Licence or Paid Commercial Licence (commercial/professional use)
# Source-available, not open source.
#
# See LICENSE and https://rateslib.com/py/en/latest/i_licence.html for details,
# and/or contact info (at) rateslib (dot) com
####################################################################################################

from __future__ import annotations

from typing import TYPE_CHECKING

from rateslib import defaults
from rateslib.dual import Dual, Dual2, Variable
from rateslib.rs import PPSplineDual, PPSplineDual2, PPSplineF64

if TYPE_CHECKING:
    from rateslib.typing import DualTypes, Number


def evaluate(
    spline: PPSplineF64 | PPSplineDual | PPSplineDual2,
    x: DualTypes,
    m: int = 0,
) -> Number:
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
    if isinstance(x, Variable):
        if isinstance(spline, PPSplineDual):
            x_: float | Dual | Dual2 = x._to_dual_type(order=1)
        elif isinstance(spline, PPSplineDual2):
            x_ = x._to_dual_type(order=2)
        else:
            x_ = x._to_dual_type(order=defaults._global_ad_order)
    else:
        x_ = x

    if isinstance(x_, Dual):
        return spline.ppdnev_single_dual(x_, m)
    elif isinstance(x_, Dual2):
        return spline.ppdnev_single_dual2(x_, m)
    else:
        return spline.ppdnev_single(x_, m)
