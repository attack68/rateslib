.. vars_from
   ==========

.. .. currentmodule:: rateslib.dual

.. py:method:: PPSplineF64.ppev(x)

   Evaluate an array of *x* coordinates derivatives on the pp spline.

   Repeatedly applies :meth:`~rateslib.splines.PPSplineF64.ppev_single`.

   :param x: x-axis coordinates.
   :type x: ndarray of float

   :rtype: ndarray
