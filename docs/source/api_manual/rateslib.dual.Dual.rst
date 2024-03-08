Dual
==========

.. currentmodule:: rateslib.dual

.. py:class:: Dual
   Dual number data type to perform first derivative automatic differentiation.

   Parameters
   ----------
   real : float, int
       The real coefficient of the dual number
   vars : tuple of str, optional
       The labels of the variables for which to record derivatives. If not given
       the dual number represents a constant, equivalent to an int or float.
   dual : 1d ndarray, optional
       First derivative information contained as coefficient of linear manifold.
       Defaults to an array of ones the length of ``vars`` if not given.


   Attributes
   ----------
   real : float, int
   vars : str, tuple of str
   dual : 1d ndarray

   See Also
   --------
   Dual2 : Dual number data type to perform second derivative automatic differentiation.
