.. vars_from
   ==========

.. .. currentmodule:: rateslib.dual

.. py:method:: PPSplineF64.ppev_single_dual(x)

   Evaluate a single *x* coordinate value on the pp spline.

   :param x: The x-axis value at which to evaluate value.
   :type x: Dual

   :rtype: Dual

   .. rubric:: Notes

   This function guarantees preservation of accurate AD :class:`~rateslib.dual.Dual`
   sensitivities.
