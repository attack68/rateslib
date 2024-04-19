.. vars_from
   ==========

.. .. currentmodule:: rateslib.dual

.. py:method:: PPSplineF64.csolve(tau, y, left_n, right_n, allow_lsq)

   Solve the coefficients of the spline given the data sites and the endpoint constraints.

   :param tau: The x-axis data sites.
   :type tau: sequence of float

   :param y: The y-axis data site values, of a type associated with the spline
   :type y: sequence of float (or Dual or Dual2)

   :param left_n: The derivative order of the left side endpoint constraint.
   :type left_n: int

   :param right_n: The derivative order of the right side endpoint constraint.
   :type right_n: int

   :param allow_lsq: Whether to permit least squares solving.
   :type allow_lsq: bool

   :rtype: None
