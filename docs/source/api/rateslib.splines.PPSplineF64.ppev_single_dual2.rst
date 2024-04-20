.. vars_from
   ==========

.. .. currentmodule:: rateslib.dual

.. py:method:: PPSplineF64.ppev_single_dual2(x)

   Evaluate a single *x* coordinate value on the pp spline.

   :param x: The x-axis value at which to evaluate value.
   :type x: Dual2

   :rtype: Dual2

   .. rubric:: Notes

   This function guarantees preservation of accurate AD :class:`~rateslib.dual.Dual2`
   sensitivities.
