.. vars_from
   ==========

.. .. currentmodule:: rateslib.dual

.. py:method:: PPSplineF64.ppdnev(x, m)

   Evaluate an array of x coordinates derivatives on the pp spline.

   Repeatedly applies :meth:`~rateslib.splines.PPSplineF64.ppdnev_single`.

   .. warning::

      The *x* coordinates supplied to this function are treated as *float*, or are
      **converted** to *float*. Therefore it does not guarantee the preservation of AD
      sensitivities.

   :param x: x-axis coordinates.
   :type x: ndarray of float

   :param m: The order of derivative to calculate value for.
   :type m: int

   :rtype: ndarray
