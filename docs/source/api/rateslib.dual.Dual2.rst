Dual2
==========

.. currentmodule:: rateslib.dual

.. py:class:: Dual2(real, vars, dual, dual2)

   Dual number data type to perform first derivative automatic differentiation.

   :param real: The real coefficient of the dual number
   :type real: float, int

   :param vars: The labels of the variables for which to record derivatives. If empty,
       the dual number represents a constant, equivalent to a float.
   :type vars: tuple of str

   :param dual: First derivative information contained as coefficient of linear manifold.
       Defaults to an array of ones the length of ``vars`` if empty.
   :type dual: list of float

   :param dual2: Second derivative information contained as coefficients of quadratic manifold.
       Defaults to a 2d array of zeros the size of ``vars`` if empty.
       These values represent a 2d array but must be given as a 1d list of values in row-major order.
   :type dual2: list of float

   .. rubric:: Attributes

   :ivar real: float
   :ivar vars: sequence of str
   :ivar dual: 1d ndarray
   :ivar dual2: 2d ndarray

   .. seealso::
      :class:`~rateslib.dual.Dual`: Dual number data type to perform first derivative automatic differentiation.

   .. rubric:: Examples

   .. ipython:: python

      from rateslib.dual import Dual2, gradient
      def func(x, y):
          return 5 * x**2 + 10 * y**3

      x = Dual2(1.0, ["x"], [], [])
      y = Dual2(1.0, ["y"], [], [])
      gradient(func(x,y), ["x", "y"], order=2)

   .. rubric:: Methods Summary

   .. include:: rateslib.dual.Dual2.vars_from.rst
