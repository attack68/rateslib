.. vars_from
   ==========

.. .. currentmodule:: rateslib.dual

.. py:method:: Dual2.vars_from(other, real, vars, dual, dual2)

   Create a :class:`~rateslib.dual.Dual2` object with ``vars`` linked with another.

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

   :param dual2: Second derivative information contained as coefficients of quadratic manifold.
       Defaults to a 2d array of zeros the size of ``vars`` if empty.
       These values represent a 2d array but must be given as a 1d list of values in row-major order.
   :type dual2: list of float

   :rtype: Dual2

   .. rubric:: Notes

   Variables are constantly checked when operations are performed between dual numbers. In Rust the variables
   are stored within an ARC pointer. It is much faster to check the equivalence of two ARC pointers than if the elements
   within a variables Set, say, are the same *and* in the same order. This method exists to create dual data types
   with shared ARC pointers directly.

   .. ipython:: python

      from rateslib import Dual2

      x1 = Dual2(1.0, ["x"], [], [])
      x2 = Dual2(2.0, ["x"], [], [])
      # x1 and x2 have the same variables (["x"]) but it is a different object
      x1.ptr_eq(x2)

      x3 = Dual2.vars_from(x1, 3.0, ["x"], [], [])
      # x3 contains shared object variables with x1
      x1.ptr_eq(x3)
