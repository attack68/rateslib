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

.. container:: twocol

   .. container:: leftside40

      .. image:: _static/thumb_coding_2_1.png
         :alt: Coding Interest Rates: FX, Swaps and Bonds
         :target: https://www.amazon.com/dp/0995455562
         :width: 145
         :align: center

   .. container:: rightside60

      The mathematics and theory used to implement *rateslib's* AD is documented thoroughly in
      the companion book *Coding Interest Rates: FX, Swaps and Bonds*.

.. raw:: html

   <div class="clear"></div>

Summary
*******

Classes
-------
.. autosummary::
   rateslib.dual.Dual
   rateslib.dual.Dual2
   rateslib.dual.Variable

Methods
-------
.. autosummary::
   rateslib.dual.gradient
   rateslib.dual.dual_exp
   rateslib.dual.dual_log
   rateslib.dual.dual_norm_pdf
   rateslib.dual.dual_norm_cdf
   rateslib.dual.dual_inv_norm_cdf
   rateslib.dual.dual_solve
   rateslib.dual.newton_1dim
   rateslib.dual.newton_ndim
   rateslib.dual.ift_1dim
   rateslib.dual.quadratic_eqn


Example
*******

Below, a standard Python function is created and is called with standard *floats*.

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

For extracting first derivatives, or gradients, we can depend on the *rateslib*
:class:`~rateslib.dual.Dual` datatypes and its operator overloading:

.. math::

   \frac{\partial f}{\partial x} &= \left . 6 x^5 + \frac{1}{y} e^{\frac{x}{y}} \right |_{(2,1,2)} = 199.3890... \\
   \frac{\partial f}{\partial y} &= \left . -\frac{x}{y^2} e^{\frac{x}{y}} \right |_{(2,1,2)} = -14.7781... \\
   \frac{\partial f}{\partial z} &= \left . \frac{1}{z} \right |_{(2,1,2)} = 0.50 \\

.. ipython:: python

   x = Dual(2, ["x"], [])
   y = Dual(1, ["y"], [])
   z = Dual(2, ["z"], [])

   value = func(x, y, z)
   value.real
   gradient(value, ["x", "y", "z"])

Second Derivatives
------------------

For extracting second derivatives we must use the :class:`~rateslib.dual.Dual2` datatype:

.. ipython:: python

    x = Dual2(2, ["x"], [], [])
    y = Dual2(1, ["y"], [], [])
    z = Dual2(2, ["z"], [], [])

    value = func(x, y, z)
    value.real
    value.dual
    gradient(value, ["x", "y"], order=2)

Implementation
***************

.. ipython:: python

   x = Dual(2.0, ["x"], [])
   y = Dual(3.0, ["y"], [])

Forward mode AD is implemented using operating overloading
and custom compatible functions. The operations implemented are;

  - addition (+)

    .. ipython:: python

       x + y + 1.5

  - subtraction and negation (-),

    .. ipython:: python

       x - y - 1.5

  - multiplication (*),

    .. ipython:: python

       x * y * 1.5

  - division and inversion (/) (\*\*-1),

    .. ipython:: python

       x / y / 1.5

  - dual and float type powers (\*\*),

    .. ipython:: python

       x ** y ** 1.5

  - exponential and logarithms (which require the specific methods below),

    .. ipython:: python

       dual_exp(x)
       dual_log(y)

  - equality of dual numbers with integers and floats and with each other.

    .. ipython:: python

       x == y

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

Exogenous Variables
*********************

The :class:`~rateslib.dual.Variable` class allows users to inject sensitivity into calculations
without knowing which AD order is required for calculations - calculations will **not**
raise *TypeErrors*. Upon first instance of a binary operation
with an object of either order it will return an object of associated type.

.. ipython:: python

   x = Variable(2.5, ["x"])
   x * Dual(1.5, ["y"], [2.2])
   x * Dual2(1.5, ["y"], [2.2], [])

In order for other internal processes to function dynamically, *rateslib* maintains a **global
AD order**. When a *Variable* performs a self operation from which it cannot infer the AD order, it
will refer to this global state.

.. ipython:: python

   defaults._global_ad_order = 1
   dual_exp(x)

   defaults._global_ad_order = 2
   dual_exp(x)

   defaults._global_ad_order = 1  # Reset

Product Rule and ``keep_manifold``
***********************************

The ``keep_manifold`` argument is also exclusively available for :class:`~rateslib.dual.Dual2`.
When extracting a first order gradient from a :class:`~rateslib.dual.Dual2` there is
information available to also express the gradient of this gradient.

Consider the function :math:`g(x)=x^2` at :math:`x=2`. This is the object:

.. ipython:: python

   g = Dual2(4.0, ["x"], [4.0], [1.0])

The default, first order, ``gradient`` extraction will simply yield floating point values:

.. ipython:: python

   gradient(g, order=1)

However, by directly requesting to ``keep_manifold`` then the output will yield a
:class:`~rateslib.dual.Dual2` datatype of this information with the appropriate sensitivity of
this gradient derived from second order.

.. ipython:: python

   gradient(g, order=1, keep_manifold=True)

This is not used frequently within *rateslib* but its purpose is to preserve the chain rule,
and allow two chained operations of the :meth:`~rateslib.dual.gradient` method.

Consider the additional function :math:`h(y)=y^2`, evaluated at
the point :math:`y=4`. This is the object:

.. ipython:: python

   h = Dual2(16.0, ["y"], [8.0], [1.0])

It is natural to derive second order gradients of the product of these functions at the
supposed points using:

.. ipython:: python

   gradient(g * h, order=2)

Below demonstrates the dual application of the method to derive the same values.

.. ipython:: python

   gradient(gradient(g * h, keep_manifold=True)[0])
   gradient(gradient(g * h, keep_manifold=True)[1])
