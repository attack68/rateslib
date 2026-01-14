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

from rateslib.rs import PPSplineDual, PPSplineDual2, PPSplineF64, bspldnev_single, bsplev_single
from rateslib.splines.evaluate import evaluate

PPSplineF64.__doc__ = """
Piecewise polynomial spline composed of float-64 values on the x-axis and
float-64 values on the y-axis.

Parameters
----------
k: int
    The order of the spline.
t: sequence of float
    The knot sequence of the spline.
c: sequence of float, optional
    The coefficients of the spline.

See Also
--------

.. seealso::
   :class:`~rateslib.splines.PPSplineDual`: Spline where the y-axis contains :class:`~rateslib.dual.Dual`  data types.

   :class:`~rateslib.splines.PPSplineDual2`: Spline where the y-axis contains :class:`~rateslib.dual.Dual2` data types.
"""  # noqa: E501

PPSplineDual.__doc__ = """
Piecewise polynomial spline composed of float-64 values on the x-axis and
:class:`~rateslib.dual.Dual` values on the y-axis.

Parameters
----------
k: int
    The order of the spline.
t: sequence of float
    The knot sequence of the spline.
c: sequence of Dual, optional
    The coefficients of the spline.

See Also
--------

.. seealso::
   :class:`~rateslib.splines.PPSplineF64`: Spline where the y-axis contains float-64 data types.

   :class:`~rateslib.splines.PPSplineDual2`: Spline where the y-axis contains :class:`~rateslib.dual.Dual2` data types.
"""  # noqa: E501

PPSplineDual2.__doc__ = """
Piecewise polynomial spline composed of float-64 values on the x-axis and
:class:`~rateslib.dual.Dual2` values on the y-axis.

Parameters
----------
k: int
    The order of the spline.
t: sequence of float
    The knot sequence of the spline.
c: sequence of Dual2, optional
    The coefficients of the spline.

.. seealso::
   :class:`~rateslib.splines.PPSplineF64`: Spline where the y-axis contains float-64 data types.

   :class:`~rateslib.splines.PPSplineDual`: Spline where the y-axis contains :class:`~rateslib.dual.Dual` data types.
"""  # noqa: E501

__all__ = (
    "PPSplineDual",
    "PPSplineDual2",
    "PPSplineF64",
    "bspldnev_single",
    "bsplev_single",
    "evaluate",
)
