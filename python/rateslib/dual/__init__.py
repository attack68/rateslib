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
from rateslib.rs import Dual, Dual2

Dual.__doc__ = "Dual number data type to perform first derivative automatic differentiation."
Dual2.__doc__ = "Dual number data type to perform second derivative automatic differentiation."

# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.

__all__ = [
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
