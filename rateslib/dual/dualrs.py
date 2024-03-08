from rateslibrs import Dual, Dual2, dsolve
import numpy as np

Dual.__doc__ = "Dual number data type to perform first derivative automatic differentiation."
# Dual2.__doc__ = "My String2"

def dual_solve(A, b, allow_lsq=False):
    # TODO: map types of A and b to ensure Rust type compliance
    a = [item for sublist in A.tolist() for item in sublist]  # 1D array of
    b = b.tolist()
    _ = np.array(dsolve(a, b, allow_lsq))
    return _
