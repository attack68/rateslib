PPSplineDual
============

.. currentmodule:: rateslib.splines

.. py:class:: PPSplineDual(k, t, c)

   Piecewise polynomial spline composed of float-64 values on the x-axis and :class:`~rateslib.dual.Dual` values
   on the y-axis.

   :param k: The order of the spline.
   :type k: int

   :param t: The knot sequence of the spline.
   :type t: sequence of float

   :param c: The coefficients of the spline, optional.
   :type c: sequence of Dual or None

   .. rubric:: Attributes

   :ivar c: sequence of Dual
   :ivar k: int
   :ivar n: int
   :ivar t: sequence of float

   .. seealso::
      :class:`~rateslib.splines.PPSplineF64`: Spline where the y-axis contains float-64 data types.

      :class:`~rateslib.splines.PPSplineDual2`: Spline where the y-axis contains :class:`~rateslib.dual.Dual2` data types.

   .. rubric:: Notes

   For all associated methods review :class:`~rateslib.splines.PPSplineF64`.

