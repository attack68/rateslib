.. vars_from
   ==========

.. .. currentmodule:: rateslib.dual

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

   .. rubric:: Notes

   Variables are constantly checked when operations are performed between dual numbers. In Rust the variables
   are stored within an ARC pointer. It is much faster to check the equivalence of two ARC pointers than if the elements
   within a variables Set, say, are the same *and* in the same order. This method exists to create dual data types
   with shared ARC pointers directly.

   .. ipython:: python

      from rateslib import Dual

      x1 = Dual(1.0, ["x"], [])
      x2 = Dual(2.0, ["x"], [])
      # x1 and x2 have the same variables (["x"]) but it is a different object
      x1.ptr_eq(x2)

      x3 = Dual.vars_from(x1, 3.0, ["x"], [])
      # x3 contains shared object variables with x1
      x1.ptr_eq(x3)
