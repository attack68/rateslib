from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from rateslib import defaults
from rateslib.default import NoInput
from rateslib.rs import Dual, Dual2

if TYPE_CHECKING:
    from rateslib.typing import Arr1dF64

PRECISION = 1e-14
FLOATS = float | np.float16 | np.float32 | np.float64 | np.longdouble
INTS = int | np.int8 | np.int16 | np.int32 | np.int32 | np.int64


class Variable:
    """
    A user defined, exogenous variable that automatically converts to a
    :class:`~rateslib.dual.Dual` or
    :class:`~rateslib.dual.Dual2` type dependent upon the overall AD calculation order.

    See :ref:`what is an exogenous variable? <cook-exogenous-doc>`

    Parameters
    ----------
    real : float
        The real coefficient of the underlying dual number.
    vars : tuple of str, optional
        The labels of the variables for which to record derivatives. If not given
        the *Variable* represents a constant - it would be better to define just a float.
    dual : 1d ndarray, optional
        First derivative information contained as coefficient of linear manifold.
        Defaults to an array of ones the length of ``vars`` if not given.

    Attributes
    ----------
    real : float
    vars : str, tuple of str
    dual : 1d ndarray
    """

    def __init__(
        self,
        real: float,
        vars: Sequence[str] = (),  # noqa: A002
        dual: list[float] | Arr1dF64 | NoInput = NoInput(0),
    ):
        self.real: float = float(real)
        self.vars: tuple[str, ...] = tuple(vars)
        n = len(self.vars)
        if isinstance(dual, NoInput) or len(dual) == 0:
            self.dual: Arr1dF64 = np.ones(n, dtype=np.float64)
        else:
            self.dual = np.asarray(dual.copy())  # type: ignore[assignment]

    def _to_dual_type(self, order: int) -> Dual | Dual2:
        if order == 1:
            _: Dual | Dual2 = self.to_dual()
            return _
        elif order == 2:
            _ = self.to_dual2()
            return _
        else:
            raise TypeError(
                f"`Variable` can only be converted with `order` in [1, 2], got order: {order}."
            )

    def to_json(self) -> str:
        """
        Serialize this object to JSON format.

        The object can be deserialized using the :meth:`~rateslib.serialization.from_json` method.

        Returns
        -------
        str
        """
        obj = dict(
            PyNative=dict(
                Variable=dict(
                    real=self.real,
                    vars=self.vars,
                    dual=list(self.dual),
                )
            )
        )
        return json.dumps(obj)

    @classmethod
    def _from_json(cls, loaded_json: dict[str, Any]) -> Variable:
        return Variable(
            real=loaded_json["real"],
            vars=loaded_json["vars"],
            dual=loaded_json["dual"],
        )

    def to_dual(self) -> Dual:
        return Dual(self.real, vars=self.vars, dual=self.dual)

    def to_dual2(self) -> Dual2:
        return Dual2(self.real, vars=self.vars, dual=self.dual, dual2=[])

    def __eq__(self, argument: Any) -> bool:
        """
        Compare an argument with a Variable for equality.
        This does not account for variable ordering.
        """
        if not isinstance(argument, type(self)):
            return False
        if self.vars == argument.vars:
            return self.__eq_coeffs__(argument, PRECISION)
        return False

    def __lt__(self, other: Any) -> bool:
        return self.real.__lt__(other)

    def __le__(self, other: Any) -> bool:
        return self.real.__le__(other)

    def __gt__(self, other: Any) -> bool:
        return self.real.__gt__(other)

    def __ge__(self, other: Any) -> bool:
        return self.real.__ge__(other)

    def __eq_coeffs__(self, argument: Dual | Dual2 | Variable, precision: float) -> bool:
        """Compare the coefficients of two dual array numbers for equality."""
        return not (
            not math.isclose(self.real, argument.real, abs_tol=precision)
            or not np.all(np.isclose(self.dual, argument.dual, atol=precision))
        )

    # def __float__(self):
    #  This does not work well with rust.
    #  See: https://github.com/PyO3/pyo3/issues/3672
    #  and https://github.com/PyO3/pyo3/discussions/3911
    #     return self.real

    def __abs__(self) -> float:
        return abs(self.real)

    def __neg__(self) -> Variable:
        return Variable(-self.real, vars=self.vars, dual=-self.dual)

    def __add__(self, other: Dual | Dual2 | float | Variable) -> Dual | Dual2 | Variable:
        if isinstance(other, Variable):
            _1 = self._to_dual_type(defaults._global_ad_order)
            _2 = other._to_dual_type(defaults._global_ad_order)
            return _1.__add__(_2)
        elif isinstance(other, FLOATS | INTS):
            return Variable(self.real + float(other), vars=self.vars, dual=self.dual)
        elif isinstance(other, Dual):
            return Dual(self.real, vars=self.vars, dual=self.dual).__add__(other)
        elif isinstance(other, Dual2):
            return Dual2(self.real, vars=self.vars, dual=self.dual, dual2=[]).__add__(other)
        else:
            raise TypeError(f"No operation defined between `Variable` and type: `{type(other)}`")

    def __radd__(self, other: Dual | Dual2 | float | Variable) -> Dual | Dual2 | Variable:
        return self.__add__(other)

    def __rsub__(self, other: Dual | Dual2 | float | Variable) -> Dual | Dual2 | Variable:
        return (self.__neg__()).__add__(other)

    def __sub__(self, other: Dual | Dual2 | float | Variable) -> Dual | Dual2 | Variable:
        return self.__add__(other.__neg__())

    def __mul__(self, other: Dual | Dual2 | float | Variable) -> Dual | Dual2 | Variable:
        if isinstance(other, Variable):
            _1 = self._to_dual_type(defaults._global_ad_order)
            _2 = other._to_dual_type(defaults._global_ad_order)
            return _1.__mul__(_2)
        elif isinstance(other, FLOATS | INTS):
            return Variable(self.real * float(other), vars=self.vars, dual=self.dual * float(other))  # type: ignore[arg-type]
        elif isinstance(other, Dual):
            return Dual(self.real, vars=self.vars, dual=self.dual).__mul__(other)
        elif isinstance(other, Dual2):
            return Dual2(self.real, vars=self.vars, dual=self.dual, dual2=[]).__mul__(other)
        else:
            raise TypeError(f"No operation defined between `Variable` and type: `{type(other)}`")

    def __rmul__(self, other: Dual | Dual2 | float | Variable) -> Dual | Dual2 | Variable:
        return self.__mul__(other)

    def __truediv__(self, other: Dual | Dual2 | float | Variable) -> Dual | Dual2 | Variable:
        if isinstance(other, Variable):
            _1 = self._to_dual_type(defaults._global_ad_order)
            _2 = other._to_dual_type(defaults._global_ad_order)
            return _1.__truediv__(_2)
        elif isinstance(other, FLOATS | INTS):
            return Variable(self.real / float(other), vars=self.vars, dual=self.dual / float(other))  # type: ignore[arg-type]
        elif isinstance(other, Dual):
            return Dual(self.real, vars=self.vars, dual=self.dual).__truediv__(other)
        elif isinstance(other, Dual2):
            return Dual2(self.real, vars=self.vars, dual=self.dual, dual2=[]).__truediv__(other)
        else:
            raise TypeError(f"No operation defined between `Variable` and type: `{type(other)}`")

    def __rtruediv__(self, other: Dual | Dual2 | float | Variable) -> Dual | Dual2 | Variable:
        if isinstance(other, Variable):
            # cannot reach this line
            raise TypeError("Impossible line execution - please report issue.")  # pragma: no cover
        elif isinstance(other, FLOATS | INTS):
            _1 = Variable(other, ())
            return _1 / self
        elif isinstance(other, Dual):
            _ = Dual(self.real, vars=self.vars, dual=self.dual)
            return other.__truediv__(_)
        elif isinstance(other, Dual2):
            _ = Dual2(self.real, vars=self.vars, dual=self.dual, dual2=[])
            return other.__truediv__(_)
        else:
            raise TypeError(f"No operation defined between `Variable` and type: `{type(other)}`")

    def __exp__(self) -> Dual | Dual2:
        _1 = self._to_dual_type(defaults._global_ad_order)
        return _1.__exp__()

    def __log__(self) -> Dual | Dual2:
        _1 = self._to_dual_type(defaults._global_ad_order)
        return _1.__log__()

    def __norm_cdf__(self) -> Dual | Dual2:
        _1 = self._to_dual_type(defaults._global_ad_order)
        return _1.__norm_cdf__()

    def __norm_inv_cdf__(self) -> Dual | Dual2:
        _1 = self._to_dual_type(defaults._global_ad_order)
        return _1.__norm_inv_cdf__()

    def __pow__(self, exponent: float | Dual | Dual2, modulo: int | None = None) -> Dual | Dual2:
        _1 = self._to_dual_type(defaults._global_ad_order)
        return _1.__pow__(exponent, modulo)

    def __repr__(self) -> str:
        a = ", ".join(self.vars[:3])
        b = ", ".join([str(_) for _ in self.dual[:3]])
        if len(self.vars) > 3:
            a += ", ..."
            b += ", ..."
        return f"<Variable: {self.real:.6}, ({a}), [{b}]>"
