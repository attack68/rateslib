Dual
==========

.. currentmodule:: rateslib.dual

.. py:class:: Dual(real, vars, dual)

   Dual number data type to perform first derivative automatic differentiation.

   :param real: The real coefficient of the dual number
   :type real: float, int

   :param vars: The labels of the variables for which to record derivatives. If empty,
       the dual number represents a constant, equivalent to a float.
   :type vars: tuple of str

   :param dual: First derivative information contained as coefficient of linear manifold.
       Defaults to an array of ones the length of ``vars`` if empty.
   :type dual: list of float

   .. rubric:: Attributes

   :ivar real: float
   :ivar vars: sequence of str
   :ivar dual: 1d ndarray

   .. seealso::
      :class:`~rateslib.dual.Dual2`: Dual number data type to perform second derivative automatic differentiation.

   .. rubric:: Examples

   .. ipython:: python

      from rateslib.dual import Dual, gradient
      def func(x, y):
          return 5 * x**2 + 10 * y**3

      x = Dual(1.0, ["x"], [])
      y = Dual(1.0, ["y"], [])
      gradient(func(x,y), ["x", "y"])

   .. rubric:: Methods Summary

   .. include:: rateslib.dual.Dual.vars_from.rst
