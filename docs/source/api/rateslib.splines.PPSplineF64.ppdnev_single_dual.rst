.. vars_from
   ==========

.. .. currentmodule:: rateslib.dual

.. py:method:: PPSplineF64.ppdnev_single_dual(x, m)

   Evaluate a single *x* coordinate derivative from the right on the pp spline.

   :param x: The x-axis value at which to evaluate value.
   :type x: Dual

   :param m: The order of derivative to calculate value for.
   :type m: int

   :rtype: Dual

   .. rubric:: Notes

   This function guarantees preservation of accurate AD :class:`~rateslib.dual.Dual`
   sensitivities.
