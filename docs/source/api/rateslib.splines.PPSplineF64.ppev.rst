.. vars_from
   ==========

.. .. currentmodule:: rateslib.dual

.. py:method:: PPSplineF64.ppev(x)

   Evaluate an array of *x* coordinates derivatives on the pp spline.

   Repeatedly applies :meth:`~rateslib.splines.PPSplineF64.ppev_single`, and is typically
   used for minor performance gains in chart plotting.

   .. warning::

      The *x* coordinates supplied to this function are treated as *float*, or are
      **converted** to *float*. Therefore it does not guarantee the preservation of AD
      sensitivities. If you need to index by *x* values which are :class:`~rateslib.dual.Dual`
      or :class:`~rateslib.dual.Dual2`, then you should choose to iteratively map the
      provided methods :meth:`~rateslib.splines.PPSplineF64.ppev_single_dual` or
      :meth:`~rateslib.splines.PPSplineF64.ppev_single_dual2` respectively.

   :param x: x-axis coordinates.
   :type x: ndarray of float

   :rtype: ndarray
