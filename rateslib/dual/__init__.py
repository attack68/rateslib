
# These imports will be used to transition in case of using Rust pyo3 bindings.

DUAL_CORE_PY = True

if DUAL_CORE_PY:
    from rateslib.dual.dual import (
        DualTypes,
        DualBase,
        Dual,
        Dual2,
        dual_log,
        dual_exp,
        dual_solve,
        set_order_convert,
        set_order,
        gradient,
        _plu_decomp,
        _pivot_matrix,
    )