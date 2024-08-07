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

   .. rubric:: Notes

   This class implements a piecewise polynomial spline curve defined by:

   .. math::

      $_{k, \mathbf{t}}(x) = \sum_{i=1}^n c_i B_{i,k,\mathbf{t}}(x)

   where :math:`B_{i,k,\mathbf{t}}(x)` is one of the *n* b-splines, of order *k* over
   knot sequence :math:`\mathbf{t}`, evaluated at *x*,
   and :math:`c_i` is the coefficient of the *i*'th b-spline for this specific
   piecewise polynomial.

   .. rubric:: Examples

   See :ref:`splines in user guide <splines-doc>`.

   .. rubric:: Methods Summary

   .. autosummary::

      ~PPSplineF64.bsplev
      ~PPSplineF64.bspldnev
      ~PPSplineF64.bsplmatrix
      ~PPSplineF64.csolve
      ~PPSplineF64.ppev
      ~PPSplineF64.ppev_single
      ~PPSplineF64.ppev_single_dual
      ~PPSplineF64.ppev_single_dual2
      ~PPSplineF64.ppdnev_single
      ~PPSplineF64.ppdnev_single_dual
      ~PPSplineF64.ppdnev_single_dual2

   .. rubric:: Methods Documentation

   .. automethod:: rateslib.splines.PPSplineF64.bsplev
   .. automethod:: rateslib.splines.PPSplineF64.bspldnev
   .. automethod:: rateslib.splines.PPSplineF64.bsplmatrix
   .. automethod:: rateslib.splines.PPSplineF64.csolve
   .. automethod:: rateslib.splines.PPSplineF64.ppev
   .. automethod:: rateslib.splines.PPSplineF64.ppev_single
   .. automethod:: rateslib.splines.PPSplineF64.ppev_single_dual
   .. automethod:: rateslib.splines.PPSplineF64.ppev_single_dual2
   .. automethod:: rateslib.splines.PPSplineF64.ppdnev
   .. automethod:: rateslib.splines.PPSplineF64.ppdnev_single
   .. automethod:: rateslib.splines.PPSplineF64.ppdnev_single_dual
   .. automethod:: rateslib.splines.PPSplineF64.ppdnev_single_dual2
