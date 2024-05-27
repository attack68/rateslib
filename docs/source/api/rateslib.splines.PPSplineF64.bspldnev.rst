.. vars_from
   ==========

.. .. currentmodule:: rateslib.dual

.. py:method:: PPSplineF64.bspldnev(x, i, m)

   Evaluate *m* order derivative on the *i* th b-spline at *x* coordinates.

   Repeatedly applies :meth:`~rateslib.splines.bspldnev_single`.

   .. warning::

      The *x* coordinates supplied to this function are treated as *float*, or are
      **converted** to *float*. Therefore it does not guarantee the preservation of AD
      sensitivities.

   :param x: x-axis coordinates.
   :type x: ndarray of float

   :param i: The index of the b-spline to evaluate.
   :type i: int

   :param m: The order of derivative to calculate value for.
   :type m: int

   :rtype: ndarray
