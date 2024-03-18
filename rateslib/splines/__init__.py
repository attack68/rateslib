
# These imports will be used to transition in case of using Rust pyo3 bindings.

SPLINES_CORE_PY = True

if SPLINES_CORE_PY:
    from rateslib.splines.splines import (
        PPSpline,
        bsplev_single,
        bspldnev_single,
    )
