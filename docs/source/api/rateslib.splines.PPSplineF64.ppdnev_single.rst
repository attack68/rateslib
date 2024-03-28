.. vars_from
   ==========

.. .. currentmodule:: rateslib.dual

.. py:method:: PPSplineF64.ppdnev_single(x, m)

   Evaluate a single *x* coordinate derivative from the right on the pp spline.

   :param x: The x-axis value at which to evaluate value.
   :type x: float

   :param m: The order of derivative to calculate value for.
   :type m: int

   :rtype: float (or Dual or Dual2 depending upon spline type)

   .. rubric:: Notes

   The value of derivatives of the spline at *x* is the sum of the value of each
   b-spline derivatives evaluated at *x* multiplied by the spline
   coefficients, *c*.

   Due to the definition of the splines this derivative will return the value
   from the right where derivatives are discontinuous.

   .. math::

      \frac{d^m\$(x)}{d x^m} = \sum_{i=1}^n c_i \frac{d^m B_{(i,k,\mathbf{t})}(x)}{d x^m}
