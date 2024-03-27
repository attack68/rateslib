bspldnev_single
================

.. currentmodule:: rateslib.splines

.. py:function:: bspldnev_single(x, i, k, t, m, org_k)

   Calculate the *m* th order derivative from the right of an indexed b-spline at *x*.

   :param x: The *x* value at which to evaluate the b-spline.
   :type x: float

   :param i: The index of the b-spline to evaluate.
   :type i: int

   :param k: The order of the b-spline (note that k=4 is a cubic spline).
   :type k: int

   :param t: The knot sequence of the pp spline.
   :type t: sequence of float

   :param m: The order of the derivative of the b-spline to evaluate.
   :type m: int

   :param org_k: The original k input. Used only internally when recursively calculating successive b-splines.
       Users will not typically use this parameters.
   :type org_k: int, optional

   .. rubric:: Notes

   B-splines derivatives can be recursively defined as:

   .. math::

      \frac{d}{dx}B_{i,k,\mathbf{t}}(x) = (k-1) \left ( \frac{B_{i,k-1,\mathbf{t}}(x)}{t_{i+k-1}-t_i} - \frac{B_{i+1,k-1,\mathbf{t}}(x)}{t_{i+k}-t_{i+1}} \right )

   and such that the basic, stepwise, b-spline derivative is:

   .. math::

      \frac{d}{dx}B_{i,1,\mathbf{t}}(x) = 0

   During this recursion the original order of the spline is registered so that under
   the given knot sequence, :math:`\mathbf{t}`, lower order b-splines which are not
   the rightmost will register a unit value. For example, the 4'th order knot sequence
   [1,1,1,1,2,2,2,3,4,4,4,4] defines 8 b-splines. The rightmost is measured
   across the knots [3,4,4,4,4]. When the knot sequence remains constant and the
   order is lowered to 3 the rightmost, 9'th, b-spline is measured across [4,4,4,4],
   which is effectively redundant since its domain has zero width. The 8'th b-spline
   which is measured across the knots [3,4,4,4] is that which will impact calculations
   and is therefore given the value 1 at the right boundary. This is controlled by
   the information provided by ``org_k``.

   .. rubric:: Examples

   The derivative of the 4th b-spline of the following knot sequence
   is discontinuous at `x` = 2.0.

   .. ipython:: python

      t = [1,1,1,1,2,2,2,3,4,4,4,4]
      bspldnev_single(x=2.0, i=3, k=4, t=t, m=1)
      bspldnev_single(x=1.99999999, i=3, k=4, t=t, m=1)

   .. plot::

      from rateslib.splines import *
      import matplotlib.pyplot as plt
      from datetime import datetime as dt
      import numpy as np
      t = [1,1,1,1,2,2,2,3,4,4,4,4]
      spline = PPSpline(k=4, t=t)
      x = np.linspace(1, 4, 76)
      fix, ax = plt.subplots(1,1)
      ax.plot(x, spline.bspldnev(x, 3, 0))
      plt.show()

