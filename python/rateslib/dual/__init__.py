# SPDX-License-Identifier: LicenseRef-Rateslib-Dual
#
# Copyright (c) 2026 Siffrorna Technology Limited
#
# Dual-licensed: Free Educational Licence or Paid Commercial Licence (commercial/professional use)
# Source-available, not open source.
#
# See LICENSE and https://rateslib.com/py/en/latest/i_licence.html for details,
# and/or contact info (at) rateslib (dot) com
####################################################################################################


from __future__ import annotations

from rateslib.dual.ift import ift_1dim
from rateslib.dual.newton import newton_1dim, newton_ndim
from rateslib.dual.quadratic import quadratic_eqn
from rateslib.dual.utils import (
    dual_exp,
    dual_inv_norm_cdf,
    dual_log,
    dual_norm_cdf,
    dual_norm_pdf,
    dual_solve,
    gradient,
    set_order,
    set_order_convert,
)
from rateslib.dual.variable import Variable
from rateslib.rs import ADOrder, Dual, Dual2

Dual.__doc__ = """
Dual number data type to perform first derivative automatic differentiation.

Parameters
----------
real : float
   The real coefficient of the dual number: its value.
vars : tuple/list of str
   The labels of the variables for which to record derivatives. If empty,
   the dual number represents a constant, equivalent to a float.
dual : list of float
   First derivative information contained as coefficient of linear manifold.
   Defaults to an array of ones the length of ``vars`` if empty.

See Also
---------

.. seealso::
   :class:`~rateslib.dual.Dual2`: Dual number data type to perform second derivative automatic differentiation.

Examples
---------
.. ipython:: python
   :suppress:

   from rateslib.dual import Dual, gradient

.. ipython:: python

   def func(x, y):
       return 5 * x**2 + 10 * y**3

   x = Dual(1.0, ["x"], [])
   y = Dual(1.0, ["y"], [])
   gradient(func(x,y), ["x", "y"])

"""  # noqa: E501

Dual2.__doc__ = """
Dual number data type to perform second derivative automatic differentiation.

Parameters
-----------
real : float
   The real coefficient of the dual number: its value.
vars : tuple/list of str
   The labels of the variables for which to record derivatives. If empty,
   the dual number represents a constant, equivalent to a float.
dual : list of float
   First derivative information contained as coefficient of linear manifold.
   Defaults to an array of ones the length of ``vars`` if empty.
dual2 : list of float
   Second derivative information contained as coefficients of quadratic manifold.
   Defaults to a 2d array of zeros the size of ``vars`` if empty.
   These values represent a 2d array but must be given as a 1d list of values in row-major order,
   which is reshaped.

See Also
--------

.. seealso::
   :class:`~rateslib.dual.Dual`: Dual number data type to perform first derivative automatic differentiation.

Examples
---------

.. ipython:: python

   from rateslib.dual import Dual2, gradient
   def func(x, y):
       return 5 * x**2 + 10 * y**3

   x = Dual2(1.0, ["x"], [], [])
   y = Dual2(1.0, ["y"], [], [])
   gradient(func(x,y), ["x", "y"], order=2)

"""  # noqa: E501

# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.

__all__ = [
    "ADOrder",
    "Dual",
    "Dual2",
    "Variable",
    "dual_log",
    "dual_exp",
    "dual_solve",
    "dual_norm_pdf",
    "dual_norm_cdf",
    "dual_inv_norm_cdf",
    "gradient",
    "set_order_convert",
    "set_order",
    "newton_ndim",
    "newton_1dim",
    "ift_1dim",
    "quadratic_eqn",
]
