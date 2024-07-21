PPSplineDual2
=============

.. currentmodule:: rateslib.splines

.. py:class:: PPSplineDual2(k, t, c)

   Piecewise polynomial spline composed of float-64 values on the x-axis and :class:`~rateslib.dual.Dual2` values
   on the y-axis.

   :param k: The order of the spline.
   :type k: int

   :param t: The knot sequence of the spline.
   :type t: sequence of float

   :param c: The coefficients of the spline, optional.
   :type c: sequence of Dual or None

   .. rubric:: Attributes

   :ivar c: sequence of Dual2
   :ivar k: int
   :ivar n: int
   :ivar t: sequence of float

   .. seealso::
      :class:`~rateslib.splines.PPSplineF64`: Spline where the y-axis contains float-64 data types.

      :class:`~rateslib.splines.PPSplineDual`: Spline where the y-axis contains :class:`~rateslib.dual.Dual` data types.

   .. rubric:: Methods Summary

   .. autosummary::

      ~PPSplineDual2.bsplev
      ~PPSplineDual2.bspldnev
      ~PPSplineDual2.bsplmatrix
      ~PPSplineDual2.csolve
      ~PPSplineDual2.ppev
      ~PPSplineDual2.ppev_single
      ~PPSplineDual2.ppev_single_dual2
      ~PPSplineDual2.ppdnev
      ~PPSplineDual2.ppdnev_single
      ~PPSplineDual2.ppdnev_single_dual2

   .. rubric:: Methods Documentation

   .. automethod:: rateslib.splines.PPSplineDual2.bsplev
   .. automethod:: rateslib.splines.PPSplineDual2.bspldnev
   .. automethod:: rateslib.splines.PPSplineDual2.bsplmatrix
   .. automethod:: rateslib.splines.PPSplineDual2.csolve
   .. automethod:: rateslib.splines.PPSplineDual2.ppev
   .. automethod:: rateslib.splines.PPSplineDual2.ppev_single
   .. automethod:: rateslib.splines.PPSplineDual2.ppev_single_dual2
   .. automethod:: rateslib.splines.PPSplineDual2.ppdnev
   .. automethod:: rateslib.splines.PPSplineDual2.ppdnev_single
   .. automethod:: rateslib.splines.PPSplineDual2.ppdnev_single_dual2
