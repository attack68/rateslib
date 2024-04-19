.. vars_from
   ==========

.. .. currentmodule:: rateslib.dual

.. py:method:: PPSplineF64.bsplmatrix(tau, left_n, right_n)

   Evaluate the 2d spline collocation matrix at each data site.

   :param tau: The data sites along the `x` axis which will instruct the pp spline.
   :type tau: ndarray of float

   :param left_n: The order of derivative to use for the left most data site and top row
       of the spline collocation matrix.
   :type left_n: int

   :param right_n: The order of derivative to use for the right most data site and bottom row
       of the spline collocation matrix.
   :type right_n: int

   :rtype: ndarray

   .. rubric:: Notes

   The spline collocation matrix is defined as,

   .. math::

      [\mathbf{B}_{k, \mathbf{t}}(\mathbf{\tau})]_{j,i} = B_{i,k,\mathbf{t}}(\tau_j)

   where each row is a call to :meth:`~rateslib.splines.PPSplineF64.bsplev`, except the top and bottom rows
   which can be specifically adjusted to account for
   ``left_n`` and ``right_n`` such that, for example, the first row might be,

   .. math::

      [\mathbf{B}_{k, \mathbf{t}}(\mathbf{\tau})]_{1,i} = \frac{d^n}{dx}B_{i,k,\mathbf{t}}(\tau_1)