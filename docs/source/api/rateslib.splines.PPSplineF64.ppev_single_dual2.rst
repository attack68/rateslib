.. vars_from
   ==========

.. .. currentmodule:: rateslib.dual

.. py:method:: PPSplineF64.ppev_single_dual2(x)

   Evaluate a single *x* coordinate value on the pp spline.

   :param x: The x-axis value at which to evaluate value.
   :type x: Dual2

   :rtype: Dual2

   .. rubric:: Notes

   The value of the spline at *x* is the sum of the value of each b-spline
   evaluated at *x* multiplied by the spline coefficients, *c*.

   .. math::

      \$(x) = \sum_{i=1}^n c_i B_{(i,k,\mathbf{t})}(x)

   This function guarantees preservation of accurate AD :class:`~rateslib.dual.Dual2`
   sensitivities.
