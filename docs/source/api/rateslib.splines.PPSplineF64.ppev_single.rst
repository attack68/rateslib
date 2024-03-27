.. vars_from
   ==========

.. .. currentmodule:: rateslib.dual

.. py:method:: PPSplineF64.ppev_single(x)

   Evaluate a single *x* coordinate value on the pp spline.

   :param x: The x-axis value at which to evaluate value.
   :type x: float

   :rtype: float (or Dual or Dual2 depending upon spline type)

   .. rubric:: Notes

   The value of the spline at *x* is the sum of the value of each b-spline
   evaluated at *x* multiplied by the spline coefficients, *c*.

   .. math::

      \$(x) = \sum_{i=1}^n c_i B_{(i,k,\mathbf{t})}(x)
