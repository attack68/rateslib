.. vars_from
   ==========

.. .. currentmodule:: rateslib.dual

.. py:method:: PPSplineF64.ppdnev_single_dual2(x, m)

   Evaluate a single *x* coordinate derivative from the right on the pp spline.

   :param x: The x-axis value at which to evaluate value.
   :type x: Dual2

   :param m: The order of derivative to calculate value for.
   :type m: int

   :rtype: Dual2

   .. rubric:: Notes

   This function guarantees preservation of accurate AD :class:`~rateslib.dual.Dual2`
   sensitivities.
