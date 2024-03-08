vars_from
==========

.. currentmodule:: rateslib.dual

.. py:method:: Dual.vars_from(other, real, vars, dual)

   Create a :class:`~rateslib.dual.Dual` object with ``vars`` linked with another.

   :param other: The other Dual from which to link vars.
   :type other: Dual

   :param real: The real coefficient of the dual number
   :type real: float, int

   :param vars: The labels of the variables for which to record derivatives. If empty,
       the dual number represents a constant, equivalent to a float.
   :type vars: tuple of str

   :param dual: First derivative information contained as coefficient of linear manifold.
       Defaults to an array of ones the length of ``vars`` if empty.
   :type dual: list of float

   :rtype: Dual
