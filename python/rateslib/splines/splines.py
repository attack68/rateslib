"""
.. ipython:: python
   :suppress:

   from rateslib.splines import *
   from datetime import datetime as dt
"""

import numpy as np
from rateslib.dual import dual_solve
from datetime import timedelta


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


def bsplev_single(x, i, k, t, org_k=None):
    """
    Calculate the value of an indexed b-spline of the pp at the coordinate `x`.

    Parameters
    ----------
    x : Any
        The single coordinate at which to evaluate the b-spline. Must be of appropriate
        type, e.g. float or datetime.
    i : int in [0, n-1]
        The index of the b-spline to evaluate.
    k : int
        The order of the b-spline (note that k=4 is a cubic spline).
    t : sequence
        The knot sequence of the pp where each element is of the same type as ``x``.
    org_k : int, optional
        The original ``k`` input. Used only internally when recursively calculating
        successive b-splines. Users will not typically use this parameters.

    Returns
    -------
    value

    Notes
    -----
    B-splines can be recursively defined as:

    .. math::

       B_{i,k,\\mathbf{t}}(x) = \\frac{x-t_i}{t_{i+k-1}-t_i}B_{i,k-1,\\mathbf{t}}(x) + \\frac{t_{i+k}-x}{t_{i+k}-t_{i+1}}B_{i+1,k-1,\\mathbf{t}}(x)

    and such that the basic, stepwise, b-spline or order 1 are:

    .. math::

       B_{i,1,\\mathbf{t}}(x) = \\left \{ \\begin{matrix} 1, & t_i \leq x < t_{i+1} \\\\ 0, & \\text{otherwise} \end{matrix} \\right .

    For continuity on the right boundary the rightmost basic b-spline is also set equal
    to 1 there: :math:`B_{n,1,\\mathbf{t}}(t_{n+k})=1`.
    """
    # Short circuit (positivity and support property)
    if x < t[i] or x > t[i + k]:
        return 0.0

    org_k = org_k or k  # original_k adds support for derivative recursion
    # Right side endpoint support
    if x == t[-1] and i >= (len(t) - org_k - 1):
        return 1.0

    # Recursion
    if k == 1:
        if t[i] <= x < t[i + 1]:
            return 1.0
        return 0.0
    else:
        left, right = 0.0, 0.0
        if t[i] != t[i + k - 1]:
            left = (x - t[i]) / (t[i + k - 1] - t[i]) * bsplev_single(x, i, k - 1, t)
        if t[i + 1] != t[i + k]:
            right = (t[i + k] - x) / (t[i + k] - t[i + 1]) * bsplev_single(x, i + 1, k - 1, t)
        return left + right


def bspldnev_single(x, i, k, t, m, org_k=None):
    """
    Calculate the `m` th order derivative from the right of an indexed b-spline at `x`.

    Parameters
    ----------
    x : Any
        The single coordinate at which to evaluate the b-spline. Must be of appropriate
        type, e.g. float or datetime.
    i : int in [0, n-1]
        The index of the b-spline to evaluate.
    k : int
        The order of the b-spline (note that k=4 is a cubic spline).
    t : sequence
        The knot sequence of the pp where each element is of the same type as ``x``.
    m : int
        The order of the derivative of the b-spline to evaluate
    org_k : int, optional
        The original ``k`` input. Used only internally when recursively calculating
        successive b-splines. Users will not typically use this parameters.

    Returns
    -------
    value

    Notes
    -----
    B-splines derivatives can be recursively defined as:

    .. math::

       \\frac{d}{dx}B_{i,k,\\mathbf{t}}(x) = (k-1) \\left ( \\frac{B_{i,k-1,\\mathbf{t}}(x)}{t_{i+k-1}-t_i} - \\frac{B_{i+1,k-1,\\mathbf{t}}(x)}{t_{i+k}-t_{i+1}} \\right )

    and such that the basic, stepwise, b-spline derivative is:

    .. math::

       \\frac{d}{dx}B_{i,1,\\mathbf{t}}(x) = 0

    During this recursion the original order of the spline is registered so that under
    the given knot sequence, :math:`\\mathbf{t}`, lower order b-splines which are not
    the rightmost will register a unit value. For example, the 4'th order knot sequence
    [1,1,1,1,2,2,2,3,4,4,4,4] defines 8 b-splines. The rightmost is measured
    across the knots [3,4,4,4,4]. When the knot sequence remains constant and the
    order is lowered to 3 the rightmost, 9'th, b-spline is measured across [4,4,4,4],
    which is effectively redundant since its domain has zero width. The 8'th b-spline
    which is measured across the knots [3,4,4,4] is that which will impact calculations
    and is therefore given the value 1 at the right boundary. This is controlled by
    the information provided by ``org_k``.

    Examples
    --------
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
    """
    if m == 0:
        return bsplev_single(x, i, k, t)
    elif k == 1 or m >= k:
        return 0

    org_k = org_k or k
    r, div1, div2 = 0, t[i + k - 1] - t[i], t[i + k] - t[i + 1]
    if isinstance(div1, timedelta):
        div1 = div1 / timedelta(days=1)
    if isinstance(div2, timedelta):
        div2 = div2 / timedelta(days=1)

    if m == 1:
        if div1 != 0:
            r += bsplev_single(x, i, k - 1, t, org_k) / div1
        if div2 != 0:
            r -= bsplev_single(x, i + 1, k - 1, t, org_k) / div2
        r *= k - 1
    else:
        if div1 != 0:
            r += bspldnev_single(x, i, k - 1, t, m - 1, org_k) / div1
        if div2 != 0:
            r -= bspldnev_single(x, i + 1, k - 1, t, m - 1, org_k) / div2
        r *= k - 1
    return r


class PPSpline:
    """
    Implements a data site instructed piecewise polynomial (pp) spline using b-splines.

    Parameters
    ----------
    k : int
        The order of the spline (note cubic splines are of order 4).
    t : sequence
        The knot sequence of the pp.
    c : sequence, optional
        The b-spline coefficients. If not given must use :meth:`csolve` to determine.

    Attributes
    ----------
    k : int
    t : sequence
    n : int
    c : sequence

    Notes
    -----
    This class implements a piecewise polynomial spline curve defined by:

    .. math::

       $_{k, \\mathbf{t}}(x) = \\sum_{i=1}^n c_i B_{i,k,\\mathbf{t}}(x)

    where :math:`B_{i,k,\\mathbf{t}}(x)` is one of the *n* b-splines, of order *k* over
    knot sequence :math:`\\mathbf{t}`, evaluated at *x*,
    and :math:`c_i` is the coefficient of the *i*'th b-spline for this specific
    piecewise polynomial.
    """

    # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
    # Commercial use of this code, and/or copying and redistribution is prohibited.
    # Contact rateslib at gmail.com if this code is observed outside its intended sphere.

    def __init__(self, k, t, c=None):
        self.t = t
        self.k = k
        self.n = len(t) - k
        self.c = c

    def __copy__(self):
        ret = PPSpline(self.k, self.t, getattr(self, "c", None))
        return ret

    def __eq__(self, other):
        if not isinstance(other, PPSpline):
            return False
        if len(self.t) != len(other.t):
            return False
        for i in range(len(self.t)):
            if self.t[i] != other.t[i]:
                return False
        for attr in ["k", "n"]:
            if getattr(self, attr, None) != getattr(other, attr, None):
                return False
        if self.c is None or other.c is None:
            return False
        else:
            for i in range(len(self.c)):
                if self.c[i] != other.c[i]:
                    return False
        return True

    def bsplev(self, x, i, otypes=["float64"]):
        """
        Evaluate `x` coordinates on the `i` th B-Spline.

        Is the vectorised version of :meth:`bsplev_single`.

        Parameters
        ----------
        x : ndarray
            Array of `x` coordinates. Must be of appropriate
            type e.g. float or datetime.
        i : int in [0, n-1]
            The index of the b-spline to evaluate.

        Returns
        -------
        ndarray
        """
        func = np.vectorize(bsplev_single, excluded=["k", "t"], otypes=otypes)
        return func(x, i=i, k=self.k, t=self.t)

    def bspldnev(self, x, i, m, otypes=["float64"]):
        """
        Evaluate `m` order derivative on the `i` th b-spline at `x` coordinates.

        Is the vectorised version of :meth:`bspldnev_single`.

        Parameters
        ----------
        x : ndarray
            Array of `x` coordinates. Must be of appropriate
            type e.g. float or datetime.
        i : int in [0, n-1]
            The index of the b-spline to evaluate.
        m : int
            The order of derivative to calculate the value for.

        Returns
        -------
        ndarray
        """
        func = np.vectorize(bspldnev_single, excluded=["k", "t"], otypes=otypes)
        return func(x, i=i, k=self.k, t=self.t, m=m)

    def bsplmatrix(self, tau, left_n=0, right_n=0):
        """
        Evaluate the 2d spline collocation matrix at each data site.

        Parameters
        ----------
        tau : sequence
            The data sites along the `x` axis which will instruct the pp spline.
        left_n : int
            The order of derivative to use for the left most data site and top row
            of the spline collocation matrix.
        right_n : int
            The order of derivative to use for the right most data site and bottom row
            of the spline collocation matrix.

        Returns
        -------
        2d matrix : ndarray

        Notes
        -----
        The spline collocation matrix is defined as,

        .. math::

           [\\mathbf{B}_{k, \\mathbf{t}}(\\mathbf{\\tau})]_{j,i} = B_{i,k,\\mathbf{t}}(\\tau_j)

        where each row is a call to :meth:`bsplev`, except the top and bottom rows
        which can be specifically adjusted to account for
        ``left_n`` and ``right_n`` such that, for example, the first row might be,

        .. math::

           [\\mathbf{B}_{k, \\mathbf{t}}(\\mathbf{\\tau})]_{1,i} = \\frac{d^n}{dx}B_{i,k,\\mathbf{t}}(\\tau_1)
        """
        B_ji = np.zeros(shape=(len(tau), self.n))
        for i in range(self.n):
            B_ji[0, i] = bspldnev_single(tau[0], i, self.k, self.t, left_n)
            B_ji[1:-1, i] = self.bsplev(tau[1:-1], i=i)
            B_ji[-1, i] = bspldnev_single(tau[-1], i, self.k, self.t, right_n)
        return B_ji

    def csolve(self, tau, y, left_n, right_n, allow_lsq=False, **kwargs):
        """
        Evaluates and sets the b-spline coefficients, `c`, that parametrise the pp.

        Parameters
        ----------
        tau : sequence
            The data sites along the `x` axis which will instruct the pp spline.
        y : sequence
            The function values obtained at the given data sites. If using
            ``left_n`` and ``right_n`` the first and last values in ``y`` should
            correspond to either the function value or its derivatives depending upon
            the relevant input.
        left_n : int
            The order of function derivative which is supplied for the
            first value of ``y``. Use `0` if function value is given.
        right_n : int
            The order of function derivative which is supplied for
            the last  value of ``y``. Use `0` if function value is given.
        allow_lsq : bool
            If the number of data sites is greater than the dimension of the pp spline
            this setting allows the coefficients to be solved as a least squares
            problem rather than raising dimensionality exceptions.
        kwargs : dict
            Additional keyword args passed to the `dual_solve` linear algebra function.

        Returns
        -------
        Sets the class attribute ``c`` : None

        Notes
        -----
        B-spline coefficients are solved via the linear system,

        .. math::

           \\mathbf{B}_{k, \\mathbf{t}}(\\mathbf{\\tau}) \\mathbf{c} = g(\\mathbf{\\tau}), \quad \\text{where} \quad [\\mathbf{B}_{k, \\mathbf{t}}(\\mathbf{\\tau})]_{j,i} = B_{i,k,\\mathbf{t}}(\\tau_j)

        Where the top and bottom rows of the spline collocation matrix are adjusted
        to account for the specified derivatives.
        """
        if len(tau) != self.n:
            if allow_lsq and len(tau) > self.n:
                pass
            else:
                raise ValueError(
                    f"If not `allow_lsq` then `tau` must have length equal to pp "
                    f"dimension, otherwise `tau` must be larger than `n`: "
                    f"`tau`: {len(tau)}, `n`: {self.n}"
                )
        if len(tau) != len(y):
            raise ValueError(
                f"`tau` and `y` must have the same length, " f"`tau`: {len(tau)}, `y`: {len(y)}"
            )
        y = np.asarray(y)
        B_ji = self.bsplmatrix(tau, left_n, right_n)
        c = dual_solve(B_ji, y[:, np.newaxis], allow_lsq=allow_lsq, **kwargs)
        self.c = c[:, 0]
        return None

    # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
    # Commercial use of this code, and/or copying and redistribution is prohibited.
    # Contact rateslib at gmail.com if this code is observed outside its intended sphere.

    def ppev_single(self, x):
        """
        Evaluate a single `x` coordinate on the piecewise polynomial spline.

        Parameters
        ----------
        x : Any
            The point at which to evaluate the spline. Should be of consistent datatype
            with the x-axis values of the spline, e.g. float or datetime.

        Returns
        -------
        Any

        Notes
        -----
        The value of the spline at *x* is the sum of the value of each b-spline
        evaluated at *x* multiplied by the spline coefficients, *c*.

        .. math::

           \\$(x) = \\sum_{i=1}^n c_i B_{(i,k,\\mathbf{t})}(x)
        """
        _ = np.array([bsplev_single(x, i, self.k, self.t) for i in range(self.n)])
        return np.dot(_, self.c)

    def ppev(self, x):
        """
        Evaluate an array of `x` coordinates on the piecewise polynomial spline.

        Is a vectorisation of the :meth:`ppev_single` method.

        Parameters
        ----------
        x : ndarray
            The points at which to evaluate the spline. Should be of consistent datatype
            with the x-axis values of the spline, e.g. float or datetime.

        Returns
        -------
        ndarray
        """
        func = np.vectorize(self.ppev_single)
        return func(x)

    def ppdnev_single(self, x, m):
        """
        Evaluate a single `x` coordinate derivative from the right on the pp spline.

        Parameters
        ----------
        x : Any
            The point at which to evaluate the spline. Should be of consistent datatype
            with the x-axis values of the spline, e.g. float or datetime.
        m : int
            The order of derivative to calculate the value for.

        Returns
        -------
        Any

        Notes
        -----
        The value of derivatives of the spline at *x* is the sum of the value of each
        b-spline derivatives evaluated at *x* multiplied by the spline
        coefficients, *c*.

        Due to the definition of the splines this derivative will return the value
        from the right where derivatives are discontinuous.

        .. math::

           \\frac{d^m\\$(x)}{d x^m} = \\sum_{i=1}^n c_i \\frac{d^m B_{(i,k,\\mathbf{t})}(x)}{d x^m}
        """
        sum = 0
        for i, c_ in enumerate(self.c):
            sum += c_ * bspldnev_single(x, i, self.k, self.t, m)
        return sum

    def ppdnev(self, x, m):
        """
        Evaluate an array of `x` coordinates derivatives on the pp spline.

        Is the vectorised version of :meth:`ppdnev_single`.

        Parameters
        ----------
        x : ndarray
            The points at which to evaluate the spline. Should be of consistent datatype
            with the x-axis values of the spline, e.g. float or datetime.
        m : int
            The order of derivative to calculate the value for.

        Returns
        -------
        Any
        """
        func = np.vectorize(self.ppdnev_single)
        return func(x, m)


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.
