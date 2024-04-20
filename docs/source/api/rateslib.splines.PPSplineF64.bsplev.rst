.. vars_from
   ==========

.. .. currentmodule:: rateslib.dual

.. py:method:: PPSplineF64.bsplev(x, i)

   Evaluate value of the *i* th b-spline at x coordinates.

   Repeatedly applies :meth:`~rateslib.splines.bsplev_single`.

   .. warning::

      The *x* coordinates supplied to this function are treated as *float*, or are
      **converted** to *float*. Therefore it does not guarantee the preservation of AD
      sensitivities.

   :param x: x-axis coordinates.
   :type x: ndarray of float

   :param i: The index of the b-spline to evaluate.
   :type i: int

   :rtype: ndarray
