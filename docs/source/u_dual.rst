.. _dual-doc:

.. ipython:: python
   :suppress:

   import numpy as np

************
Dual Numbers
************

The ``rateslib.dual`` module creates dual number datatypes that provide this library
with the ability to perform automatic
differentiation (AD). The implementation style here is known as *forward mode*, as
opposed to *reverse mode* (otherwise called *automatic adjoint differentiation* or
*back propagation*).


Summary
*******

Classes
-------
.. autosummary::
   rateslib.dual.Dual
   rateslib.dual.Dual2

Methods
-------
.. autosummary::
   rateslib.dual.gradient
   rateslib.dual.dual_exp
   rateslib.dual.dual_log
   rateslib.dual.dual_solve

Example
*******

First Derivatives
-----------------

.. math::

   f(x, y, z) = x^6 + e^{\frac{x}{y}} + \ln {z} \\
   f(2, 1, 2) = 72.0822...

.. ipython:: python

   from rateslib.dual import *
   def func(x, y, z):
       return x**6 + dual_exp(x/y) + dual_log(z)

   func(2, 1, 2)

For extracting only first derivatives it is more efficient
to use :class:`~rateslib.dual.Dual`:

.. math::

   \frac{\partial f}{\partial x} &= \left . 6 x^5 + \frac{1}{y} e^{\frac{x}{y}} \right |_{(2,1,2)} = 199.3890... \\
   \frac{\partial f}{\partial y} &= \left . -\frac{x}{y^2} e^{\frac{x}{y}} \right |_{(2,1,2)} = -14.7781... \\
   \frac{\partial f}{\partial z} &= \left . \frac{1}{z} \right |_{(2,1,2)} = 0.50 \\

.. ipython:: python

   x, y, z = Dual(2, ["x"], []), Dual(1, ["y"], []), Dual(2, ["z"], [])
   func(x, y, z)
   gradient(func(x, y, z), ["x", "y", "z"])

Second Derivatives
------------------

For extracting second derivatives we must use :class:`~rateslib.dual.Dual2`:

.. ipython:: python

    x, y, z = Dual2(2, ["x"], [], []), Dual2(1, ["y"], [], []), Dual2(2, ["z"], [], [])
    func(x, y, z)
    gradient(func(x, y, z), ["x", "y", "z"])
    gradient(func(x, y, z), ["x", "y"], order=2)

The ``keep_manifold`` argument is also exclusively available
for :class:`~rateslib.dual.Dual2`. When
extracting a first order gradient from a :class:`~rateslib.dual.Dual2` this is
will use information about
second order and transfer it to first order thus representing a linear manifold
of the gradient. This is useful for allowing composited automatic calculation of
second order gradients. For example
consider the following functions, :math:`g(x)=x^2` and :math:`h(y)=y^2`, evaluated at
the points :math:`x=2` and :math:`y=4`. This creates the quadratic manifolds centered
at those points expressed in the following :class:`~rateslib.dual.Dual2` numbers:

.. ipython:: python

    g = Dual2(4, ["x"], [4], [1])  # g(x=2)
    h = Dual2(16, ["y"], [8], [1])  # h(y=4)

If we wish to multiply these two functions and evaluate the second order derivatives
at (2, 4) we can simply do,

.. ipython:: python

    gradient(g*h, order=2)

And observe that, say, :math:`\frac{\partial (gh)}{\partial x \partial y} = 4xy|_{(2, 4)} = 32`,
as shown in the above array.

But, we can also use the product rule of differentiation to assert that,

.. math::

   d_{x\zeta}^2(gh) = d_x \left ( d_\zeta(g)h + gd_\zeta(h) \right ) \\\\
   d_{y\zeta}^2(gh) = d_y \left ( d_\zeta(g)h + gd_\zeta(h) \right ) \\\\

which we express in our dual language as,

.. ipython:: python

    gradient(g, ["x", "y"], keep_manifold=True) * h + g * gradient(h, ["x", "y"], keep_manifold=True)

If the manifold is not maintained the product rule fails because information that is
required to ultimately determine that desired second derivative is discarded.

.. ipython:: python

    gradient(g, ["x", "y"]) * h + g * gradient(h, ["x", "y"])

More specifically,

.. ipython:: python

    gradient(g, ["x", "y"], keep_manifold=True)

while,

.. ipython:: python

    gradient(g, ["x", "y"])


Implementation
***************

Forward mode AD is implemented using operating overloading
and custom compatible functions. The operations implemented are;

  - addition (+),
  - subtraction and negation (-),
  - multiplication (*),
  - division and inversion (/) (\*\*-1),
  - n'th power where n is an integer or a float (\*\*n),
  - exponential and logarithms (which require the specific methods below),
  - equality of dual numbers with integers and floats and with each other.

.. warning::
    :class:`~rateslib.dual.Dual` and :class:`~rateslib.dual.Dual2` are
    not designed to operate with each other. The purpose
    for this is to avoid miscalculation of second
    derivatives. :class:`~rateslib.dual.Dual` should always
    be replaced by :class:`~rateslib.dual.Dual2` in this instance.
    ``TypeErrors`` will be raised otherwise.


Compatability with NumPy
************************

To enable this library to perform its calculations in a vectorised way we need to
leverage NumPy's array calculations. NumPy arrays containing dual numbers are
forced to have an ``object`` dtype configuration. This is imposed by NumPy and means
that certain functions may not be compatible, for example ``np.einsum`` (although,
support for ``object`` dtypes was added to ``np.einsum`` as of version 1.25.0).
However, many functions are compatible.

Broadcasting
------------

Operations of :class:`~rateslib.dual.Dual` and :class:`~rateslib.dual.Dual2`
with ``int`` and ``float`` dtypes permit the NumPy versions; np.int8, np.int16,
np.int32, np.int64, np.float16, np.float32, np.float64, and np.float128.
Broadcasting of arrays has been implemented so that the following
operations work as expected.

.. ipython:: python

    np_arr = np.array([1, 2])
    Dual(3, ["x"], []) * np_arr
    np_arr / Dual(4, ["y"], [])
    Dual(4, ["x"], []) ** np_arr

Elementwise Operations
----------------------

Simple operations on tensors also work as expected.

.. ipython:: python

    x = np.array([Dual(1, ["x"], []), Dual(2, ["y"], [])])
    y = np.array([Dual(3, ["x"], []), Dual(4, ["y"], [])])
    x + y
    x * y
    x / y

Linear Algebra
--------------

Common linear algebraic operations are also available, such as:

  - ``np.matmul``
  - ``np.inner``
  - ``np.dot``
  - ``np.tensordot``

.. ipython:: python

   np.dot(x, y)
   np.inner(x, y)
   np.matmul(x[:, np.newaxis], y[np.newaxis, :])
   np.tensordot(x[np.newaxis, :, np.newaxis], y[np.newaxis, :], (1, 1))

Solving the linear system, :math:`Ax=b`, is not not directly possible from NumPy,
thus a custom solver, :meth:`~rateslib.dual.dual_solve`, has been implemented
using the Doolittle algorithm with partial pivoting.

.. ipython:: python

   A = np.array([
       [1, 0],
       [Dual(2, ["z"], []), 1]
   ], dtype="object")
   b = np.array([Dual(2, ["y"], []), Dual(5, ["x", "y"], [])])[:, np.newaxis]
   x = dual_solve(A, b)
   x
   np.matmul(A, x)
