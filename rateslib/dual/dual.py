from math import isclose
from abc import abstractmethod, ABCMeta
from typing import Optional
from statistics import NormalDist
import math
import numpy as np
from rateslib.default import NoInput

PRECISION = 1e-14
FLOATS = (float, np.float16, np.float32, np.float64, np.longdouble)
INTS = (int, np.int8, np.int16, np.int32, np.int32, np.int64)

# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


class DualBase(metaclass=ABCMeta):
    """
    Base class for dual number implementation.
    """

    dual: np.ndarray = np.zeros(0)
    dual2: np.ndarray = np.zeros(0)

    def __init__(self, real: float, vars: tuple[str, ...] = tuple()) -> None:  # pragma: no cover
        # each dual overloads init
        self.real: float = real
        self.vars: tuple[str, ...] = vars

    def __float__(self):
        return float(self.real)

    def __abs__(self):
        return abs(self.real)

    def __lt__(self, argument):
        """Compare an argument by evaluating the size of the real."""
        if not isinstance(argument, type(self)):
            if not isinstance(argument, (*FLOATS, *INTS)):
                raise TypeError(f"Cannot compare {type(self)} with incompatible type.")
            argument = type(self)(float(argument))
        if float(self) < float(argument):
            return True
        return False

    def __gt__(self, argument):
        """Compare an argument by evaluating the size of the real."""
        if not isinstance(argument, type(self)):
            if not isinstance(argument, (*FLOATS, *INTS)):
                raise TypeError(f"Cannot compare {type(self)} with incompatible type.")
            argument = type(self)(float(argument))
        if float(self) > float(argument):
            return True
        return False

    def __eq__(self, argument):
        """Compare an argument with a Dual number for equality."""
        if not isinstance(argument, type(self)):
            if isinstance(argument, NoInput):
                return False
            elif not isinstance(argument, (*FLOATS, *INTS)):
                raise TypeError(f"Cannot compare {type(self)} with incompatible type.")
            argument = type(self)(float(argument))
        if self.vars == argument.vars:
            return self.__eq_coeffs__(argument, PRECISION)
        else:
            self_, argument = self.__upcast_combined__(argument)
            return self_.__eq__(argument)

    def __eq_coeffs__(self, argument, precision):
        """Compare the coefficients of two Dual numbers for equality."""
        if not isclose(self.real, argument.real, abs_tol=precision):
            return False
        elif not np.all(np.isclose(self.dual, argument.dual, atol=precision)):
            return False
        if type(self) is Dual2 and type(argument) is Dual2:
            if not np.all(np.isclose(self.dual2, argument.dual2, atol=precision)):
                return False
        elif type(self) is Dual2 or type(argument) is Dual2:
            # this line should not be hit TypeError should raise earlier
            # cannot compare Dual with Dual2
            return False  # pragma: no cover
        return True

    def __upcast_combined__(self, arg):
        """Combines, and inserts, the vars of two Dual numbers to match each other."""
        new_vars = sorted(list(set(self.vars).union(set(arg.vars))))
        new_self = self if new_vars == self.vars else self.__upcast_vars__(new_vars)
        new_arg = arg if new_vars == arg.vars else arg.__upcast_vars__(new_vars)
        return new_self, new_arg

    @abstractmethod
    def __upcast_vars__(self, new_vars: list[str]):
        pass  # pragma: no cover

    def grad(self, vars=None, order=1, keep_manifold=False):
        """
        Return derivatives of a dual number.

        Parameters
        ----------
        vars : str, tuple, list optional
            Name of the variables which to return gradients for. If not given
            defaults to all vars attributed to the instance.
        order : {1, 2}
            Whether to return the first or second derivative of the dual number.
            Second order will raise if applied to a ``Dual`` and not ``Dual2`` instance.
        keep_manifold : bool
            If ``order`` is 1 and the type is ``Dual2`` one can return a ``Dual2``
            where the ``dual2`` values are converted to ``dual`` values to represent
            a first order manifold of the first derivative (and the ``dual2`` values
            set to zero). Useful for propagation in iterations.

        Returns
        -------
        float, ndarray, Dual2
        """
        if vars is None:
            vars = self.vars
            _ = self
        else:
            _, __ = self.__upcast_combined__(type(self)(0, vars))

        if isinstance(vars, str):
            ix_ = [_.vars.index(vars)]
        else:
            ix_ = list(map(lambda x: _.vars.index(x), vars))

        if order == 2:
            return 2 * _.dual2[np.ix_(ix_, ix_)]
        elif not keep_manifold or isinstance(self, Dual):
            return _.dual[ix_]
        else:
            ret = np.array([Dual2(v, vars) for v in _.dual[ix_]])
            for ix, du in zip(ix_, ret):
                du.dual = 2 * _.dual2[ix, ix_]
            return ret

    def grad1(self, vars=None):
        return self.grad(vars, order=1, keep_manifold=False)

    def grad1_manifold(self, vars=None):
        return self.grad(vars, order=1, keep_manifold=True)

    def grad2(self, vars=None, keep_manifold=False):
        return self.grad(vars, order=2, keep_manifold=keep_manifold)


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


class Dual2(DualBase):
    """
    Dual number data type to perform second derivative automatic differentiation.

    Parameters
    ----------
    real : float, int
        The real coefficient of the dual number
    vars : str, or list or tuple of str
        The labels of the variables for which to record derivatives.
    dual : 1d ndarray
        First derivative information contained as coefficient of linear manifold.
        Defaults to an array of ones the length of vars if not given.
    dual2 : 2d ndarray
        Second derivative information contained as coefficient of quadratic manifold.
        Defaults to an array of zeros of square size the shape of vars in not given.


    Attributes
    ----------
    real : float, int
    vars : tuple of str, optional
    dual : 1d ndarray, optional
    dual2 : 2d ndarray, optional

    See Also
    --------
    Dual : Dual number data type to perform first derivative automatic differentiation.
    """

    def __init__(self, real, vars=(), dual=None, dual2=None):
        if isinstance(vars, str):
            self.vars = (vars,)
        else:
            self.vars = tuple(vars)
        n = len(self.vars)
        self.real = real
        if dual is None or len(dual) == 0:
            self.dual: np.ndarray = np.ones(n)
        else:
            self.dual = np.asarray(dual.copy())

        if isinstance(dual2, list):
            if len(dual2) == 0:
                self.dual2 = np.zeros((n, n))
            elif len(dual2) == n**2:
                self.dual2 = np.asarray(dual2).reshape((n, n))
        elif dual2 is not None:
            self.dual2 = np.asarray(dual2.copy())
        else:
            self.dual2 = np.zeros((n, n))

    def __repr__(self):
        name, final = "Dual2", ", [[...]]"
        vars = ", ".join(self.vars[:3])
        dual = ", ".join([f"{_:.1f}" for _ in self.dual[:3]])
        if len(self.vars) > 3:
            vars += ",..."
            dual += ",..."
        return f"<{name}: {self.real:,.6f}, ({vars}), [{dual}]{final}>"

    def __str__(self):
        output = f" val = {self.real:.8f}\n"
        for i, tag in enumerate(self.vars):
            output += f"  d{tag} = {self.dual[i]:.6f}\n"
        for i, ltag in enumerate(self.vars):
            for j, rtag in enumerate(self.vars):
                if j >= i:
                    output += f"d{ltag}d{rtag} = {2 * self.dual2[i,j]:.6f}\n"
        return output

    def __neg__(self):
        return Dual2(-self.real, self.vars, -self.dual, -self.dual2)

    def __add__(self, argument):
        if not isinstance(argument, Dual2):
            if isinstance(argument, np.ndarray):
                return argument + self
            elif isinstance(argument, (*FLOATS, *INTS)):
                return Dual2(self.real + argument, self.vars, self.dual, self.dual2)
            raise TypeError("Dual2 operations defined between float, int or Dual2.")

        if self.vars == argument.vars:
            return Dual2(
                self.real + argument.real,
                self.vars,
                self.dual + argument.dual,
                self.dual2 + argument.dual2,
            )
        else:
            self_, argument = self.__upcast_combined__(argument)
            return self_ + argument

    # def __str__(self):
    #     output = f"       f = {self.real:.8f}\n"
    #     for i, tag in enumerate(self.vars):
    #         output += f"   df/d{tag} = {self.dual[i]:.6f}\n"
    #     for i, ltag in enumerate(self.vars):
    #         for j, rtag in enumerate(self.vars):
    #             if j >= i:
    #                 output += f"d2f/d{ltag}d{rtag} = {2 * self.dual2[i,j]:.6f}\n"
    #     return output

    __radd__ = __add__

    def __sub__(self, argument):
        return self + (-argument)

    def __rsub__(self, argument):
        return -(self - argument)

    def __mul__(self, argument):
        if not isinstance(argument, Dual2):
            if isinstance(argument, np.ndarray):
                return argument * self
            elif isinstance(argument, (*FLOATS, *INTS)):
                return Dual2(
                    self.real * argument,
                    self.vars,
                    self.dual * argument,
                    self.dual2 * argument,
                )
            raise TypeError("Dual2 operations defined between float, int or Dual2.")

        if self.vars == argument.vars:
            dual2 = self.dual2 * argument.real + argument.dual2 * self.real
            _ = np.einsum("i,j", self.dual, argument.dual)
            dual2 += (_ + _.T) / 2
            return Dual2(
                self.real * argument.real,
                self.vars,
                self.dual * argument.real + argument.dual * self.real,
                dual2,
            )
        else:
            self_, argument = self.__upcast_combined__(argument)
            return self_ * argument

    __rmul__ = __mul__

    def __truediv__(self, argument):
        if not isinstance(argument, Dual2):
            if isinstance(argument, np.ndarray):
                return argument.__rtruediv__(self)
            elif isinstance(argument, (*FLOATS, *INTS)):
                return Dual2(
                    self.real / argument,
                    self.vars,
                    self.dual / argument,
                    self.dual2 / argument,
                )
            raise TypeError("Dual2 operations defined between float, int or Dual2.")

        if self.vars == argument.vars:
            return self * argument**-1
        else:
            self_, argument = self.__upcast_combined__(argument)
            return self_ * argument**-1

    def __rtruediv__(self, argument):
        """x / z = (x * ^z) / (z * ^z)"""
        if not isinstance(argument, (*FLOATS, *INTS)):
            raise TypeError("Dual2 operations defined between float, int or Dual2.")
        numerator = Dual2(argument)
        return numerator / self

    def __pow__(self, power):
        if isinstance(power, (*FLOATS, *INTS)):
            coeff = power * self.real ** (power - 1)
            coeff2 = power * (power - 1) * self.real ** (power - 2) * 0.5
            return Dual2(
                self.real**power,
                self.vars,
                self.dual * coeff,
                self.dual2 * coeff + np.einsum("i,j", self.dual, self.dual) * coeff2,
            )
        elif isinstance(power, np.ndarray):
            return power.__rpow__(self)
        raise TypeError("Dual2 power defined only with float, int.")

    def __exp__(self):
        const = math.exp(self.real)
        return Dual2(
            const,
            self.vars,
            self.dual * const,
            const * (self.dual2 + np.einsum("i,j", self.dual, self.dual) * 0.5),
        )

    def __log__(self):
        return Dual2(
            math.log(self.real),
            self.vars,
            self.dual / self.real,
            self.dual2 / self.real - np.einsum("i,j", self.dual, self.dual) * 0.5 / self.real**2,
        )

    def __norm_cdf__(self):
        base = NormalDist().cdf(self.real)
        scalar = 1 / math.sqrt(2 * math.pi) * math.exp(-0.5 * self.real**2)
        scalar2 = scalar * -self.real
        return Dual2(
            base,
            self.vars,
            scalar * self.dual,
            scalar * self.dual2 + 0.5 * scalar2 * np.einsum("i,j", self.dual, self.dual),
        )

    def __norm_inv_cdf__(self):
        base = NormalDist().inv_cdf(self.real)
        scalar = math.sqrt(2 * math.pi) * math.exp(0.5 * base**2)
        scalar2 = base * scalar**2
        return Dual2(
            base,
            self.vars,
            scalar * self.dual,
            scalar * self.dual2 + 0.5 * scalar2 * np.einsum("i,j", self.dual, self.dual),
        )

    def __upcast_vars__(self, new_vars):
        n = len(new_vars)
        dual, dual2 = np.zeros(n), np.zeros((n, n))
        ix_ = list(map(lambda x: new_vars.index(x), self.vars))
        dual[ix_] = (self.dual,)
        dual2[np.ix_(ix_, ix_)] = self.dual2
        return Dual2(self.real, new_vars, dual, dual2)

    def __downcast_vars__(self):
        """removes variables where first and second order sensitivity is zero"""
        ix_ = np.where(~np.isclose(self.dual, 0, atol=PRECISION))[0]
        ix2_ = np.where(~np.isclose(self.dual2.sum(axis=0), 0, atol=PRECISION))[0]
        ixu = np.union1d(ix_, ix2_)
        new_vars = [self.vars[i] for i in ixu]
        return Dual2(self.real, new_vars, self.dual[ixu], self.dual2[np.ix_(ixu, ixu)])

    def _set_order(self, order):
        if order == 1:
            return Dual(self.real, self.vars, self.dual)
        if order == 2:
            return self
        if order == 0:
            return float(self)

    @staticmethod
    def vars_from(other, real, vars=(), dual=None, dual2=None):
        if other.vars == vars:
            return Dual2(real, vars, dual, dual2)
        else:
            return Dual2(real, vars, dual, dual2).__upcast_vars__(other.vars)


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


class Dual(DualBase):
    """
    Dual number data type to perform first derivative automatic differentiation.

    Parameters
    ----------
    real : float, int
        The real coefficient of the dual number
    vars : tuple of str, optional
        The labels of the variables for which to record derivatives. If not given
        the dual number represents a constant, equivalent to an int or float.
    dual : 1d ndarray, optional
        First derivative information contained as coefficient of linear manifold.
        Defaults to an array of ones the length of ``vars`` if not given.


    Attributes
    ----------
    real : float, int
    vars : str, tuple of str
    dual : 1d ndarray

    See Also
    --------
    Dual2 : Dual number data type to perform second derivative automatic differentiation.
    """

    def __init__(
        self,
        real: float,
        vars: tuple[str, ...] = (),
        dual: Optional[np.ndarray] = None,
    ):
        self.real: float = real
        if isinstance(vars, str):
            self.vars: tuple[str, ...] = (vars,)
        else:
            self.vars = tuple(vars)
        n = len(self.vars)
        if dual is None or len(dual) == 0:
            self.dual: np.ndarray = np.ones(n)
        else:
            self.dual = np.asarray(dual.copy())

    @property
    def dual2(self):
        raise ValueError("`Dual` variable cannot possess `dual2` attribute")

    def __repr__(self):
        name, final = "Dual", ""
        vars = ", ".join(self.vars[:3])
        dual = ", ".join([f"{_:.1f}" for _ in self.dual[:3]])
        if len(self.vars) > 3:
            vars += ",..."
            dual += ",..."
        return f"<{name}: {self.real:,.6f}, ({vars}), [{dual}]{final}>"

    def __str__(self):
        output = f" val = {self.real:.8f}\n"
        for i, tag in enumerate(self.vars):
            output += f"  d{tag} = {self.dual[i]:.6f}\n"
        return output

    def __neg__(self):
        return Dual(-self.real, self.vars, -self.dual)

    def __add__(self, argument):
        if not isinstance(argument, Dual):
            if isinstance(argument, np.ndarray):
                return argument + self
            elif isinstance(argument, (*FLOATS, *INTS)):
                return Dual(self.real + float(argument), self.vars, self.dual)
            raise TypeError("Dual operations defined between float, int or Dual.")

        if self.vars == argument.vars:
            return Dual(self.real + argument.real, self.vars, self.dual + argument.dual)
        else:
            self_, argument = self.__upcast_combined__(argument)
            return self_ + argument

    __radd__ = __add__

    def __sub__(self, argument):
        return self + (-argument)

    def __rsub__(self, argument):
        return -(self - argument)

    def __mul__(self, argument):
        if not isinstance(argument, Dual):
            if isinstance(argument, np.ndarray):
                return argument * self
            elif isinstance(argument, (*FLOATS, *INTS)):
                return Dual(self.real * float(argument), self.vars, self.dual * float(argument))
            raise TypeError(
                f"Dual operations defined between float, int or Dual, got: {type(argument)}"
            )

        if self.vars == argument.vars:
            return Dual(
                self.real * argument.real,
                self.vars,
                self.dual * argument.real + argument.dual * self.real,
            )
        else:
            self_, argument = self.__upcast_combined__(argument)
            return self_ * argument

    __rmul__ = __mul__

    def __truediv__(self, argument):
        if not isinstance(argument, Dual):
            if isinstance(argument, np.ndarray):
                return argument.__rtruediv__(self)
            if not isinstance(argument, (*FLOATS, *INTS)):
                raise TypeError("Dual operations defined between float, int or Dual.")
            return Dual(self.real / float(argument), self.vars, self.dual / float(argument))
        if self.vars == argument.vars:
            return self * argument**-1
        else:
            self_, argument = self.__upcast_combined__(argument)
            return self_ * argument**-1

    def __rtruediv__(self, argument):
        if not isinstance(argument, (*FLOATS, *INTS)):
            raise TypeError("Dual operations defined between float, int or Dual.")
        numerator = Dual(float(argument))
        return numerator / self

    def __pow__(self, power):
        if isinstance(power, (*FLOATS, *INTS)):
            pow: float = float(power)
            return Dual(
                self.real**pow,
                self.vars,
                self.dual * pow * self.real ** (pow - 1),
            )
        elif isinstance(power, np.ndarray):
            return power.__rpow__(self)
        raise TypeError("Dual power defined only with float, int.")

    def __exp__(self):
        const = math.exp(self.real)
        return Dual(const, self.vars, self.dual * const)

    def __log__(self):
        return Dual(math.log(self.real), self.vars, self.dual / self.real)

    def __norm_cdf__(self):
        base = NormalDist().cdf(self.real)
        scalar = 1 / math.sqrt(2 * math.pi) * math.exp(-0.5 * self.real**2)
        return Dual(base, self.vars, scalar * self.dual)

    def __norm_inv_cdf__(self):
        base = NormalDist().inv_cdf(self.real)
        scalar = math.sqrt(2 * math.pi) * math.exp(0.5 * base**2)
        return Dual(base, self.vars, scalar * self.dual)

    def __upcast_vars__(self, new_vars):
        n = len(new_vars)
        dual = np.zeros(n)
        ix_ = list(map(lambda x: new_vars.index(x), self.vars))
        dual[ix_] = self.dual
        return Dual(self.real, new_vars, dual)

    def __downcast_vars__(self):  # pragma: no cover
        """removes variables where first order sensitivity is zero"""
        # this function is not used within the library but left for backwards compat
        ix_ = np.where(~np.isclose(self.dual, 0, atol=PRECISION))[0]
        new_vars = tuple(self.vars[i] for i in ix_)
        return Dual(self.real, new_vars, self.dual[ix_])

    def _set_order(self, order):
        if order == 1:
            return self
        if order == 2:
            return Dual2(self.real, self.vars, self.dual, [])
        if order == 0:
            return float(self)

    @staticmethod
    def vars_from(other, real, vars=(), dual=None):
        if other.vars == vars:
            return Dual(real, vars, dual)
        else:
            return Dual(real, vars, dual).__upcast_vars__(other.vars)

    # def __str__(self):
    #     output = f"    f = {self.real:.8f}\n"
    #     for i, tag in enumerate(self.vars):
    #         output += f"df/d{tag} = {self.dual[i]:.6f}\n"
    #     return output


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


def _pivot_matrix(A, method=1):
    """
    Returns the pivoting matrix for P, used in Doolittle's method.

    Notes
    -----
    Partial pivoting can fail. If the solution detects that it has failed
    it will switch to method 2 and try a slightly different pivoting
    technique, which occasionally results in a different permutation matrix and valid
    solution. See :meth:`test_pivoting`.
    """
    n = A.shape[0]
    P = np.eye(n, dtype="object")
    PA = A.copy()
    _ = A.copy()
    # Pivot P such that the largest element of each column of A is on diagonal
    for j in range(n):
        # row = np.argmax(np.abs(_[j:, j]))  <- alternative but seems slower
        row = max(range(j, n), key=lambda i: abs(_[i][j]))
        if j != row:
            P[[j, row]] = P[[row, j]]  # Define a row swap in P
            PA[[j, row]] = PA[[row, j]]
            if method == 1:
                _[[j, row]] = _[[row, j]]  # alters the pivoting by updating underlying
    return P, PA


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


def _plu_decomp(A, method=1):
    """Performs an LU Decomposition of A (which must be square)
    into PA = LU. The function returns P, L and U. Uses Doolittle algorithm.

    `method` is passed to the pivoting technique.
    """
    if method == 3:
        raise ArithmeticError("Partial pivoting has failed on matrix and cannot solve.")
    n = A.shape[0]
    # Create zero matrices for L and U
    L, U = np.zeros((n, n), dtype="object"), np.zeros((n, n), dtype="object")

    # Create the pivot matrix P and the multipled matrix PA
    P, PA = _pivot_matrix(A, method=method)

    # Perform the LU Decomposition
    for j in range(n):
        # All diagonal entries of L are set to unity
        L[j, j] = 1.0

        # LaTeX: u_{ij} = a_{ij} - \sum_{k=1}^{i-1} u_{kj} l_{ik}
        for i in range(j + 1):
            sx = np.matmul(L[i, :i], U[:i, j])
            # s1 = sum(U[k][j] * L[i][k] for k in range(i))
            U[i, j] = PA[i, j] - sx

        # LaTeX: l_{ij} = \frac{1}{u_{jj}} (a_{ij} - \sum_{k=1}^{j-1} u_{kj} l_{ik})
        for i in range(j, n):
            sy = np.matmul(L[i, :j], U[:j, j])
            # s2 = sum(U[k][j] * L[i][k] for k in range(j))
            if abs(U[j, j]) < 1e-16:
                return _plu_decomp(A, method + 1)  # retry with altered pivoting technique
            L[i, j] = (PA[i, j] - sy) / U[j, j]

    return P, L, U


def _solve_lower_triangular_1d(L, b):
    """dual_solve the equation Lx = b, for L lower diagonal matrix, b is 1 dimension"""
    n = L.shape[0]
    x = np.zeros(shape=(n, 1), dtype="object")
    for i in range(n):
        val = b[i] - np.sum(np.matmul(L[i, :i], x[:i, 0]))
        x[i, 0] = val / L[i, i]
    return x[:, 0]


def _solve_lower_triangular(L, b):
    """dual_solve the equation Lx = b, for L lower diagonal matrix"""
    n, m = L.shape[0], b.shape[1]
    x = np.zeros(shape=(n, m), dtype="object")
    for j in range(m):
        x[:, j] = _solve_lower_triangular_1d(L, b[:, j])
    return x


def _solve_upper_triangular(U, b):
    """dual_solve the equation Ux = b, for U upper diagonal matrix"""
    return _solve_lower_triangular(U[::-1, ::-1], b[::-1, ::-1])[::-1, ::-1]


def _dsolve(A, b, allow_lsq=False):
    """
    Solve the linear system Ax=b.

    Parameters
    ----------
    A : ndarray of object dtype
        Array which can contain dual number data types.
    b : ndarray of object dtype
        Array which can contain dual number data types.
    allow_lsq : bool
        Allow the solution to be least squares if rows in ``A`` exceed the columns.

    Returns
    -------
    ndarray of object dtype

    Notes
    -----
    Solves via the permutation,

    .. math::

        PAx=LUx=Pb, \\quad Ly=Pb, \\quad Ux=y

    """
    if allow_lsq and A.shape[0] > A.shape[1]:
        b, A = np.matmul(A.T, b), np.matmul(A.T, A)

    P, L, U = _plu_decomp(A)
    y = _solve_lower_triangular(L, np.matmul(P, b))
    x = _solve_upper_triangular(U, y)
    return x


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.
