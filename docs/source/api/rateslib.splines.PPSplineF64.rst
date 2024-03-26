PPSplineF64
============

.. currentmodule:: rateslib.splines

.. py:class:: PPSplineF64(k, t, c)

   Piecewise polynomial spline composed of float-64 values on the x and y axes.

   :param k: The order of the spline.
   :type k: int

   :param t: The knot sequence of the spline.
   :type t: sequence of float

   :param c: The coefficients of the spline, optional.
   :type c: sequence of float or None

   .. rubric:: Attributes

   :ivar c: sequence of float
   :ivar k: int
   :ivar n: int
   :ivar t: sequence of float

   .. seealso::
      :class:`~rateslib.splines.PPSplineDual`: Spline where the y-axis contains :class:`~rateslib.dual.Dual` data types.

      :class:`~rateslib.splines.PPSplineDual2`: Spline where the y-axis contains :class:`~rateslib.dual.Dual2` data types.

   .. rubric:: Examples

   See LINK.

   .. rubric:: Methods Summary

   .. include:: rateslib.splines.PPSplineF64.csolve.rst
