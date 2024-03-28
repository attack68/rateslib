bsplev_single
================

.. currentmodule:: rateslib.splines

.. py:function:: bsplev_single(x, i, k, t, org_k)

   Calculate the *m* th order derivative from the right of an indexed b-spline at *x*.

   :param x: The *x* value at which to evaluate the b-spline.
   :type x: float

   :param i: The index of the b-spline to evaluate.
   :type i: int

   :param k: The order of the b-spline (note that k=4 is a cubic spline).
   :type k: int

   :param t: The knot sequence of the pp spline.
   :type t: sequence of float

   :param org_k: The original k input. Used only internally when recursively calculating successive b-splines.
       Users will not typically use this parameters.
   :type org_k: int, optional

   .. rubric:: Notes

   B-splines can be recursively defined as:

   .. math::

      B_{i,k,\mathbf{t}}(x) = \frac{x-t_i}{t_{i+k-1}-t_i}B_{i,k-1,\mathbf{t}}(x) + \frac{t_{i+k}-x}{t_{i+k}-t_{i+1}}B_{i+1,k-1,\mathbf{t}}(x)

   and such that the basic, stepwise, b-spline or order 1 are:

   .. math::

      B_{i,1,\mathbf{t}}(x) = \left \{ \begin{matrix} 1, & t_i \leq x < t_{i+1} \\ 0, & \text{otherwise} \end{matrix} \right .

   For continuity on the right boundary the rightmost basic b-spline is also set equal
   to 1 there: :math:`B_{n,1,\mathbf{t}}(t_{n+k})=1`.
