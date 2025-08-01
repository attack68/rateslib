from __future__ import annotations

import warnings
from itertools import combinations
from math import log
from time import time
from typing import TYPE_CHECKING, ParamSpec
from uuid import uuid4

import numpy as np
from pandas import DataFrame, MultiIndex, Series, concat
from pandas.errors import PerformanceWarning

from rateslib import defaults
from rateslib.curves import CompositeCurve, Curve, MultiCsaCurve, ProxyCurve, _BaseCurve
from rateslib.default import NoInput, _drb
from rateslib.dual import Dual, Dual2, dual_solve, gradient
from rateslib.dual.newton import _solver_result
from rateslib.dual.utils import _dual_float
from rateslib.fx import FXForwards, FXRates

# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.
from rateslib.fx_volatility import FXVols
from rateslib.mutability import (
    _new_state_post,
    _no_interior_validation,
    _validate_states,
    _WithState,
)

P = ParamSpec("P")

if TYPE_CHECKING:
    from numpy import float64 as Nf64  # noqa: N812
    from numpy import object_ as Nobject  # noqa: N812
    from numpy.typing import NDArray

    from rateslib.typing import (
        FX_,
        Any,
        Callable,
        DualTypes,
        FXDeltaVolSmile,
        FXDeltaVolSurface,
        FXSabrSmile,
        FXSabrSurface,
        FXVolObj,
        Sequence,
        SupportsRate,
        Variable,
        str_,
    )


class Gradients:
    """
    A catalogue of all the gradients used in optimisation routines and risk
    sensitivties.
    """

    _grad_s_vT_method: str = "_grad_s_vT_final_iteration_analytical"
    _grad_s_vT_final_iteration_algo: str = "gauss_newton_final"

    _J: NDArray[Nf64] | None
    _J_pre: NDArray[Nf64] | None
    _J2: NDArray[Nf64] | None
    _J2_pre: NDArray[Nf64] | None
    _grad_s_vT: NDArray[Nf64] | None
    _grad_s_vT_pre: NDArray[Nf64] | None
    _grad_s_s_vT: NDArray[Nf64] | None
    _grad_s_s_vT_pre: NDArray[Nf64] | None

    _reset_properties_: Callable[..., None]
    _update_step_: Callable[[str], NDArray[Nobject]]
    _set_ad_order: Callable[[int], None]
    iterate: Callable[..., None]

    func_tol: float
    conv_tol: float
    pre_solvers: tuple[Solver, ...]
    r: NDArray[Nobject]  # instrument rates at iterate
    r_pre: NDArray[Nobject]  # instrument rates at iterate including pre_
    s: NDArray[Nobject]  # target instrument rates
    m: int  # number of instruments
    pre_m: int  # number of instruments including pre_
    n: int  # number of parameters/variables
    pre_n: int  # number of parameters/variables in all solvers including pre_
    g: Dual | Dual2  # solver objective function value
    variables: tuple[str, ...]  # string tags for AD coordination
    pre_variables: tuple[str, ...]  # string tags for AD coordination
    pre_rate_scalars: list[float]  # scalars for the rate attribute of instruments
    _ad: int  # ad order
    instruments: tuple[tuple[SupportsRate, tuple[Any, ...], dict[str, Any]], ...]  # calibrators

    @property
    def J(self) -> NDArray[Nf64]:
        """
        2d Jacobian array of calibrating instrument rates with respect to curve
        variables, of size (n, m);

        .. math::

           [J]_{i,j} = [\\nabla_\\mathbf{v} \\mathbf{r^T}]_{i,j} = \\frac{\\partial r_j}{\\partial v_i}

        Depends on ``self.r``.
        """  # noqa: E501
        if self._J is None:
            self._J = np.array([gradient(rate, self.variables) for rate in self.r]).T
        return self._J

    @property
    def grad_v_rT(self) -> NDArray[Nf64]:
        """
        Alias of ``J``.
        """
        return self.J

    @property
    def J2(self) -> NDArray[Nf64]:
        """
        3d array of second derivatives of calibrating instrument rates with
        respect to curve variables, of size (n, n, m);

        .. math::

           [J2]_{i,j,k} = [\\nabla_\\mathbf{v} \\nabla_\\mathbf{v} \\mathbf{r^T}]_{i,j,k} = \\frac{\\partial^2 r_k}{\\partial v_i \\partial v_j}

        Depends on ``self.r``.
        """  # noqa: E501
        if self._J2 is None:
            if self._ad != 2:
                raise ValueError(
                    f"Cannot perform second derivative calculations when ad mode is {self._ad}.",
                )

            rates = np.array([_[0].rate(*_[1], **_[2]) for _ in self.instruments])
            # solver is passed in order to extract curves as string
            _ = np.array([gradient(rate, self.variables, order=2) for rate in rates])
            self._J2 = np.transpose(_, (1, 2, 0))
        return self._J2

    @property
    def grad_v_v_rT(self) -> NDArray[Nf64]:
        """
        Alias of ``J2``.
        """
        return self.J2  # pragma: no cover

    @property
    def grad_s_vT(self) -> NDArray[Nf64]:
        """
        2d Jacobian array of curve variables with respect to calibrating instruments,
        of size (m, n);

        .. math::

           [\\nabla_\\mathbf{s}\\mathbf{v^T}]_{i,j} = \\frac{\\partial v_j}{\\partial s_i} = \\mathbf{J^+}
        """  # noqa: E501
        if self._grad_s_vT is None:
            self._grad_s_vT = getattr(self, self._grad_s_vT_method)()
        return self._grad_s_vT

    def _grad_s_vT_final_iteration_dual(self, algorithm: str | None = None) -> NDArray[Nf64]:
        """
        This is not the ideal method since it requires reset_properties to reassess.
        """
        algorithm = algorithm or self._grad_s_vT_final_iteration_algo
        _s = self.s
        self.s = np.array([Dual(v, [f"s{i}"], []) for i, v in enumerate(self.s)])
        self._reset_properties_()
        v_1 = self._update_step_(algorithm)
        s_vars = [f"s{i}" for i in range(self.m)]
        grad_s_vT = np.array([gradient(v, s_vars) for v in v_1]).T
        self.s = _s
        return grad_s_vT

    def _grad_s_vT_final_iteration_analytical(self) -> NDArray[Nf64]:
        """Uses a pseudoinverse algorithm on floats"""
        grad_s_vT: NDArray[Nf64] = np.linalg.pinv(self.J)  # type: ignore[assignment]
        return grad_s_vT

    def _grad_s_vT_fixed_point_iteration(self) -> NDArray[Nf64]:
        """
        This is not the ideal method because it requires second order and reset props.
        """
        self._set_ad_order(2)
        self._reset_properties_()
        _s = self.s
        self.s = np.array([Dual2(v, [f"s{i}"], [], []) for i, v in enumerate(self.s)])
        s_vars = tuple(f"s{i}" for i in range(self.m))
        grad2 = gradient(self.g, self.variables + s_vars, order=2)
        grad_v_vT_f = grad2[: self.n, : self.n]
        grad_s_vT_f = grad2[self.n :, : self.n]
        grad_s_vT: NDArray[Nf64] = np.linalg.solve(grad_v_vT_f, -grad_s_vT_f.T).T  # type: ignore[assignment]

        # The following are alternative representations. Actually faster to calculate and
        # do not require sensitivity against S variables to be measured.
        # See 'coding interest rates' equation 12.38
        # _1 = np.einsum("iy, yz, jz", self.J, self.W, self.J)
        # _2 = np.einsum("z, zy, ijy", self.x.astype(float), self.W, self.J2)
        # _3 = 2* (_1 + _2)
        # _11 = -2 * np.einsum("iz,zj->ji", self.J, self.W)

        self.s = _s
        self._set_ad_order(1)
        self._reset_properties_()
        return grad_s_vT

    @property
    def grad_s_s_vT(self) -> NDArray[Nf64]:
        """
        3d array of second derivatives of curve variables with respect to
        calibrating instruments, of size (m, m, n);

        .. math::

           [\\nabla_\\mathbf{s} \\nabla_\\mathbf{s} \\mathbf{v^T}]_{i,j,k} = \\frac{\\partial^2 v_k}{\\partial s_i \\partial s_j}
        """  # noqa: E501
        if self._grad_s_s_vT is None:
            self._grad_s_s_vT = self._grad_s_s_vT_final_iteration_analytical()
        return self._grad_s_s_vT

    def _grad_s_s_vT_fwd_difference_method(self) -> NDArray[Nf64]:
        """Use a numerical method, iterating through changes in s to calculate."""
        ds = 10 ** (int(log(self.func_tol, 10) / 2))
        grad_s_vT_0 = np.copy(self.grad_s_vT)
        grad_s_s_vT = np.zeros(shape=(self.m, self.m, self.n))

        for i in range(self.m):
            self.s[i] += ds
            self.iterate()
            grad_s_s_vT[:, i, :] = (self.grad_s_vT - grad_s_vT_0) / ds
            self.s[i] -= ds

        # ensure exact symmetry (maybe redundant)
        grad_s_s_vT = (grad_s_s_vT + np.swapaxes(grad_s_s_vT, 0, 1)) / 2
        self.iterate()
        return grad_s_s_vT

    def _grad_s_s_vT_final_iteration_analytical(self, use_pre: bool = False) -> NDArray[Nf64]:
        """
        Use an analytical formula and second order AD to calculate.

        Not: must have 2nd order AD set to function, and valid properties set to
        function
        """
        if use_pre:
            J2, grad_s_vT = self.J2_pre, self.grad_s_vT_pre
        else:
            J2, grad_s_vT = self.J2, self.grad_s_vT

        # dv/dr_l * d2r_l / dvdv
        _: NDArray[Nf64] = np.tensordot(J2, grad_s_vT, (2, 0))  # type: ignore[assignment]
        # dv_z /ds * d2v / dv_zdv
        _ = np.tensordot(grad_s_vT, _, (1, 0))  # type: ignore[assignment]
        # dv_h /ds * d2v /dvdv_h
        _ = -np.tensordot(grad_s_vT, _, (1, 1))  # type: ignore[assignment]
        grad_s_s_vT = _
        return grad_s_s_vT
        # _ = np.matmul(grad_s_vT, np.matmul(J2, grad_s_vT))
        # grad_s_s_vT = -np.tensordot(grad_s_vT, _, (1, 0))
        # return grad_s_s_vT

    # _pre versions incorporate all variables of solver and pre_solvers

    def grad_f_rT_pre(self, fx_vars: Sequence[str]) -> NDArray[Nf64]:
        """
        2d Jacobian array of calibrating instrument rates with respect to FX rate
        variables, of size (len(fx_vars), pre_m);

        .. math::

           [\\nabla_\\mathbf{f}\\mathbf{r^T}]_{i,j} = \\frac{\\partial r_j}{\\partial f_i}

        Parameters
        ----------
        fx_vars : list or tuple of str
            The variable name tags for the FX rate sensitivities.
        """
        grad_f_rT = np.array([gradient(rate, fx_vars) for rate in self.r_pre]).T
        return grad_f_rT

    @property
    def J2_pre(self) -> NDArray[Nf64]:
        """
        3d array of second derivatives of calibrating instrument rates with
        respect to curve variables for all ``Solvers`` including ``pre_solvers``,
        of size (pre_n, pre_n, pre_m);

        .. math::

           [J2]_{i,j,k} = [\\nabla_\\mathbf{v} \\nabla_\\mathbf{v} \\mathbf{r^T}]_{i,j,k} = \\frac{\\partial^2 r_k}{\\partial v_i \\partial v_j}

        Depends on ``self.r`` and ``pre_solvers.J2``.
        """  # noqa: E501
        if len(self.pre_solvers) == 0:
            return self.J2

        if self._J2_pre is None:
            if self._ad != 2:
                raise ValueError(
                    f"Cannot perform second derivative calculations when ad mode is {self._ad}.",
                )

            J2 = np.zeros(shape=(self.pre_n, self.pre_n, self.pre_m))
            i, j = 0, 0
            for pre_slvr in self.pre_solvers:
                J2[
                    i : i + pre_slvr.pre_n,
                    i : i + pre_slvr.pre_n,
                    j : j + pre_slvr.pre_m,
                ] = pre_slvr.J2_pre
                i, j = i + pre_slvr.pre_n, j + pre_slvr.pre_m

            rates = np.array([_[0].rate(*_[1], **_[2]) for _ in self.instruments])
            # solver is passed in order to extract curves as string
            _ = np.array([gradient(r, self.pre_variables, order=2) for r in rates])
            J2[:, :, -self.m :] = np.transpose(_, (1, 2, 0))
            self._J2_pre = J2
        return self._J2_pre

    def grad_f_v_rT_pre(self, fx_vars: Sequence[str]) -> NDArray[Nf64]:
        """
        3d array of second derivatives of calibrating instrument rates with respect to
        FX rates and curve variables, of size (len(fx_vars), pre_n, pre_m);

        .. math::

           [\\nabla_\\mathbf{f} \\nabla_\\mathbf{v} \\mathbf{r^T}]_{i,j,k} = \\frac{\\partial^2 r_k}{\\partial f_i \\partial v_j}

        Parameters
        ----------
        fx_vars : list or tuple of str
            The variable name tags for the FX rate sensitivities.
        """  # noqa: E501
        # FX sensitivity requires reverting through all pre-solvers rates.
        all_gradients = np.array(
            [gradient(rate, self.pre_variables + tuple(fx_vars), order=2) for rate in self.r_pre],
        ).swapaxes(0, 2)

        grad_f_v_rT = all_gradients[self.pre_n :, : self.pre_n, :]
        return grad_f_v_rT

    def grad_f_f_rT_pre(self, fx_vars: Sequence[str]) -> NDArray[Nf64]:
        """
        3d array of second derivatives of calibrating instrument rates with respect to
        FX rates, of size (len(fx_vars), len(fx_vars), pre_m);

        .. math::

           [\\nabla_\\mathbf{f} \\nabla_\\mathbf{f} \\mathbf{r^T}]_{i,j,k} = \\frac{\\partial^2 r_k}{\\partial f_i \\partial f_j}

        Parameters
        ----------
        fx_vars : list or tuple of str
            The variable name tags for the FX rate sensitivities.
        """  # noqa: E501
        # FX sensitivity requires reverting through all pre-solvers rates.
        grad_f_f_rT = np.array([gradient(rate, fx_vars, order=2) for rate in self.r_pre]).swapaxes(
            0,
            2,
        )
        return grad_f_f_rT

    @property
    def grad_s_s_vT_pre(self) -> NDArray[Nf64]:
        """
        3d array of second derivatives of curve variables with respect to
        calibrating instruments, of size (pre_m, pre_m, pre_n);

        .. math::

           [\\nabla_\\mathbf{s} \\nabla_\\mathbf{s} \\mathbf{v^T}]_{i,j,k} = \\frac{\\partial^2 v_k}{\\partial s_i \\partial s_j}
        """  # noqa: E501
        if len(self.pre_solvers) == 0:
            return self.grad_s_s_vT

        if self._grad_s_s_vT_pre is None:
            self._grad_s_s_vT_pre = self._grad_s_s_vT_final_iteration_analytical(use_pre=True)
        return self._grad_s_s_vT_pre

    @property
    def grad_v_v_rT_pre(self) -> NDArray[Nf64]:
        """
        Alias of ``J2_pre``.
        """
        return self.J2_pre  # pragma: no cover

    def grad_f_s_vT_pre(self, fx_vars: Sequence[str]) -> NDArray[Nf64]:
        """
        3d array of second derivatives of curve variables with respect to
        FX rates and calibrating instrument rates, of size (len(fx_vars), pre_m, pre_n);

        .. math::

           [\\nabla_\\mathbf{f} \\nabla_\\mathbf{s} \\mathbf{v^T}]_{i,j,k} = \\frac{\\partial^2 v_k}{\\partial f_i \\partial s_j}

        Parameters
        ----------
        fx_vars : list or tuple of str
            The variable name tags for the FX rate sensitivities.
        """  # noqa: E501
        # FX sensitivity requires reverting through all pre-solvers rates.
        _ = -np.tensordot(self.grad_f_v_rT_pre(fx_vars), self.grad_s_vT_pre, (1, 1)).swapaxes(1, 2)
        _ = np.tensordot(_, self.grad_s_vT_pre, (2, 0))
        grad_f_s_vT: NDArray[Nf64] = _
        return grad_f_s_vT

    def grad_f_f_vT_pre(self, fx_vars: Sequence[str]) -> NDArray[Nf64]:
        """
        3d array of second derivatives of curve variables with respect to
        FX rates, of size (len(fx_vars), len(fx_vars), pre_n);

        .. math::

           [\\nabla_\\mathbf{f} \\nabla_\\mathbf{f} \\mathbf{v^T}]_{i,j,k} = \\frac{\\partial^2 v_k}{\\partial f_i \\partial f_j}

        Parameters
        ----------
        fx_vars : list or tuple of str
            The variable name tags for the FX rate sensitivities.
        """  # noqa: E501
        # FX sensitivity requires reverting through all pre-solvers rates.
        _ = -np.tensordot(self.grad_f_f_rT_pre(fx_vars), self.grad_s_vT_pre, (2, 0))
        _ -= np.tensordot(self.grad_f_rT_pre(fx_vars), self.grad_f_s_vT_pre(fx_vars), (1, 1))
        grad_f_f_vT: NDArray[Nf64] = _  # type: ignore[assignment]
        return grad_f_f_vT

    def grad_f_vT_pre(self, fx_vars: Sequence[str]) -> NDArray[Nf64]:
        """
        2d array of the derivatives of curve variables with respect to FX rates, of
        size (len(fx_vars), pre_n).

        .. math::

           [\\nabla_\\mathbf{f}\\mathbf{v^T}]_{i,j} = \\frac{\\partial v_j}{\\partial f_i} = -\\frac{\\partial r_z}{\\partial f_i} \\frac{\\partial v_j}{\\partial s_z}

        Parameters
        ----------
        fx_vars : list or tuple of str
            The variable name tags for the FX rate sensitivities
        """  # noqa: E501
        # FX sensitivity requires reverting through all pre-solvers rates.
        grad_f_rT = np.array([gradient(rate, fx_vars) for rate in self.r_pre]).T
        _: NDArray[Nf64] = -np.matmul(grad_f_rT, self.grad_s_vT_pre)
        return _

    def grad_f_f(self, f: Dual | Dual2 | Variable, fx_vars: Sequence[str]) -> NDArray[Nf64]:
        """
        1d array of total derivatives of FX conversion rate with respect to
        FX rate variables, of size (len(fx_vars));

        .. math::

           [\\nabla_\\mathbf{f} f_{loc:bas}]_{i} = \\frac{d f}{d f_i}

        Parameters
        ----------
        f : Dual or Dual2
            The value of the local to base FX conversion rate.
        fx_vars : list or tuple of str
            The variable name tags for the FX rate sensitivities
        """
        grad_f_f = gradient(f, fx_vars)
        grad_f_f += np.matmul(self.grad_f_vT_pre(fx_vars), gradient(f, self.pre_variables))
        ret: NDArray[Nf64] = grad_f_f
        return ret

    @property
    def grad_s_vT_pre(self) -> NDArray[Nf64]:
        """
        2d Jacobian array of curve variables with respect to calibrating instruments
        including all pre solvers attached to the Solver, of size (pre_m, pre_n).

        .. math::

           [\\nabla_\\mathbf{s}\\mathbf{v^T}]_{i,j} = \\frac{\\partial v_j}{\\partial s_i} = \\mathbf{J^+}
        """  # noqa: E501
        if len(self.pre_solvers) == 0:
            return self.grad_s_vT

        if self._grad_s_vT_pre is None:
            grad_s_vT = np.zeros(shape=(self.pre_m, self.pre_n))

            i, j = 0, 0
            for pre_solver in self.pre_solvers:
                # create the left side block matrix
                m, n = pre_solver.pre_m, pre_solver.pre_n
                grad_s_vT[i : i + m, j : j + n] = pre_solver.grad_s_vT_pre

                # create the right column dependencies
                grad_v_r = np.array([gradient(r, pre_solver.pre_variables) for r in self.r]).T
                block = np.matmul(grad_v_r, self.grad_s_vT)
                block = -1 * np.matmul(pre_solver.grad_s_vT_pre, block)
                grad_s_vT[i : i + m, -self.n :] = block

                i, j = i + m, j + n

            # create bottom right block
            grad_s_vT[-self.m :, -self.n :] = self.grad_s_vT
            self._grad_s_vT_pre = grad_s_vT
        return self._grad_s_vT_pre

    def grad_s_f_pre(self, f: Dual | Dual2 | Variable) -> NDArray[Nf64]:
        """
        1d array of FX conversion rate with respect to calibrating instruments,
        of size (pre_m);

        .. math::

           [\\nabla_\\mathbf{s} f_{loc:bas}]_{i} = \\frac{\\partial f}{\\partial s_i}

        Parameters
        ----------
        f : Dual or Dual2
            The value of the local to base FX conversion rate.
        """
        grad_s_f: NDArray[Nf64] = np.tensordot(
            self.grad_s_vT_pre, gradient(f, self.pre_variables), (1, 0)
        )  # type: ignore[assignment]
        return grad_s_f

    def grad_s_sT_f_pre(self, f: Dual | Dual2 | Variable) -> NDArray[Nf64]:
        """
        2d array of derivatives of FX conversion rate with respect to
        calibrating instruments, of size (pre_m, pre_m);

        .. math::

           [\\nabla_\\mathbf{s} \\nabla_\\mathbf{s}^\\mathbf{T} f_{loc:bas}]_{i,j} = \\frac{\\partial^2 f}{\\partial s_i \\partial s_j}

        Parameters
        ----------
        f : Dual or Dual2
            The value of the local to base FX conversion rate.
        """  # noqa: E501
        grad_s_vT = self.grad_s_vT_pre
        grad_v_vT_f = gradient(f, self.pre_variables, order=2)

        _: NDArray[Nf64] = np.tensordot(grad_s_vT, grad_v_vT_f, (1, 0))  # type: ignore[assignment]
        _ = np.tensordot(_, grad_s_vT, (1, 1))  # type: ignore[assignment]

        grad_s_sT_f = _
        return grad_s_sT_f

    def grad_f_sT_f_pre(self, f: Dual | Dual2 | Variable, fx_vars: Sequence[str]) -> NDArray[Nf64]:
        """
        2d array of derivatives of FX conversion rate with respect to
        calibrating instruments, of size (pre_m, pre_m);

        .. math::

           [\\nabla_\\mathbf{f} \\nabla_\\mathbf{s}^\\mathbf{T} f_{loc:bas}(\\mathbf{v(s, f), f)})]_{i,j} = \\frac{d^2 f}{d f_i \\partial s_j}

        Parameters
        ----------
        f : Dual or Dual2
            The value of the local to base FX conversion rate.
        fx_vars : list or tuple of str
            The variable name tags for the FX rate sensitivities
        """  # noqa: E501
        grad_s_vT = self.grad_s_vT_pre
        grad_v_f = gradient(f, self.pre_variables)
        grad_f_sT_v = self.grad_f_s_vT_pre(fx_vars)
        _ = gradient(f, self.pre_variables + tuple(fx_vars), order=2)
        grad_v_vT_f = _[: self.pre_n, : self.pre_n]
        grad_f_vT_f = _[self.pre_n :, : self.pre_n]
        # grad_f_fT_f = _[self.pre_n :, self.pre_n :]
        grad_f_vT = self.grad_f_vT_pre(fx_vars)

        _ = np.tensordot(grad_f_sT_v, grad_v_f, (2, 0))
        _ += np.tensordot(grad_f_vT_f, grad_s_vT, (1, 1))

        __ = np.tensordot(grad_f_vT, grad_v_vT_f, (1, 0))
        __ = np.tensordot(__, grad_s_vT, (1, 1))

        grad_f_sT_f: NDArray[Nf64] = _ + __
        return grad_f_sT_f

    def grad_f_fT_f_pre(self, f: Dual | Dual2 | Variable, fx_vars: Sequence[str]) -> NDArray[Nf64]:
        """
        2d array of derivatives of FX conversion rate with respect to
        calibrating instruments, of size (pre_m, pre_m);

        .. math::

           [\\nabla_\\mathbf{f} \\nabla_\\mathbf{f}^\\mathbf{T} f_{loc:bas}(\\mathbf{v(s, f), f)})]_{i,j} = \\frac{d^2 f}{d f_i d f_j}

        Parameters
        ----------
        f : Dual or Dual2
            The value of the local to base FX conversion rate.
        fx_vars : list or tuple of str
            The variable name tags for the FX rate sensitivities
        """  # noqa: E501
        # grad_s_vT = self.grad_s_vT_pre
        grad_v_f = gradient(f, self.pre_variables)
        # grad_f_sT_v = self.grad_f_s_vT_pre(fx_vars)
        _ = gradient(f, self.pre_variables + tuple(fx_vars), order=2)
        grad_v_vT_f = _[: self.pre_n, : self.pre_n]
        grad_f_vT_f = _[self.pre_n :, : self.pre_n]
        grad_f_fT_f = _[self.pre_n :, self.pre_n :]
        grad_f_vT = self.grad_f_vT_pre(fx_vars)
        grad_f_fT_v = self.grad_f_f_vT_pre(fx_vars)

        _ = grad_f_fT_f
        _ += 2.0 * np.tensordot(grad_f_vT_f, grad_f_vT, (1, 1))
        _ += np.tensordot(grad_f_fT_v, grad_v_f, (2, 0))

        __ = np.tensordot(grad_f_vT, grad_v_vT_f, (1, 0))
        __ = np.tensordot(__, grad_f_vT, (1, 1))

        grad_f_fT_f = _ + __
        return grad_f_fT_f

    # grad_v_v_f: calculated within grad_s_vT_fixed_point_iteration

    # delta and gamma calculations require all solver and pre_solver variables

    def grad_s_Ploc(self, npv: Dual | Dual2 | Variable) -> NDArray[Nf64]:
        """
        1d array of derivatives of local currency PV with respect to calibrating
        instruments, of size (pre_m).

        .. math::

           \\nabla_\\mathbf{s} P^{loc} = \\frac{\\partial P^{loc}}{\\partial s_i}

        Parameters:
            npv : Dual or Dual2
                A local currency NPV of a period of a leg.
        """
        grad_s_P: NDArray[Nf64] = np.matmul(self.grad_s_vT_pre, gradient(npv, self.pre_variables))
        return grad_s_P

    def grad_f_Ploc(self, npv: Dual | Dual2 | Variable, fx_vars: Sequence[str]) -> NDArray[Nf64]:
        r"""
        1d array of derivatives of local currency PV with respect to FX rate variable,
        of size (len(fx_vars)).

        .. math::

           \\nabla_\\mathbf{f} P^{loc}(\\mathbf{v(s, f), f}) = \\frac{\\partial P^{loc}}{\\partial f_i}+  \\frac{\partial v_z}{\\partial f_i} \\frac{\\partial P^{loc}}{\\partial v_z}

        Parameters:
            npv : Dual or Dual2
                A local currency NPV of a period of a leg.
            fx_vars : list or tuple of str
                The variable tags for automatic differentiation of FX rate sensitivity
        """  # noqa: E501
        grad_f_P = gradient(npv, fx_vars)
        grad_f_P += np.matmul(self.grad_f_vT_pre(fx_vars), gradient(npv, self.pre_variables))
        ret: NDArray[Nf64] = grad_f_P
        return ret

    def grad_s_Pbase(
        self, npv: Dual | Dual2 | Variable, grad_s_P: NDArray[Nf64], f: Dual | Dual2 | Variable
    ) -> NDArray[Nf64]:
        """
        1d array of derivatives of base currency PV with respect to calibrating
        instruments, of size (pre_m).

        .. math::

           \\nabla_\\mathbf{s} P^{bas}(\\mathbf{v(s, f)}) = \\nabla_\\mathbf{s} P^{loc}(\\mathbf{v(s, f)})  f_{loc:bas} + P^{loc} \\nabla_\\mathbf{s} f_{loc:bas}

        Parameters:
            npv : Dual or Dual2
                A local currency NPV of a period of a leg.
            grad_s_P : ndarray
                The local currency delta risks w.r.t. calibrating instruments.
            f : Dual or Dual2
                The local:base FX rate.
        """  # noqa: E501
        grad_s_Pbas: NDArray[Nf64] = _dual_float(npv) * np.matmul(
            self.grad_s_vT_pre, gradient(f, self.pre_variables)
        )
        grad_s_Pbas += grad_s_P * _dual_float(f)  # <- use float to cast float array not Dual
        return grad_s_Pbas

    def grad_f_Pbase(
        self,
        npv: Dual | Dual2 | Variable,
        grad_f_P: NDArray[Nf64],
        f: Dual | Dual2 | Variable,
        fx_vars: Sequence[str],
    ) -> NDArray[Nf64]:
        """
        1d array of derivatives of base currency PV with respect to FX rate variables,
        of size (len(fx_vars)).

        .. math::

           \\nabla_\\mathbf{s} P^{bas}(\\mathbf{v(s, f)}) = \\nabla_\\mathbf{s} P^{loc}(\\mathbf{v(s, f)})  f_{loc:bas} + P^{loc} \\nabla_\\mathbf{s} f_{loc:bas}

        Parameters:
            npv : Dual or Dual2
                A local currency NPV of a period of a leg.
            grad_f_P : ndarray
                The local currency delta risks w.r.t. FX pair variables.
            f : Dual or Dual2
                The local:base FX rate.
            fx_vars : list or tuple of str
                The variable tags for automatic differentiation of FX rate sensitivity
        """  # noqa: E501
        # use float here to cast float array not Dual
        ret: NDArray[Nf64] = grad_f_P * _dual_float(f)
        ret += _dual_float(npv) * self.grad_f_f(f, fx_vars)
        return ret

    def grad_s_sT_Ploc(self, npv: Dual2 | Variable) -> NDArray[Nf64]:
        """
        2d array of derivatives of local currency PV with respect to calibrating
        instruments, of size (pre_m, pre_m).

        .. math::

           \\nabla_\\mathbf{s} \\nabla_\\mathbf{s}^\\mathbf{T} P^{loc}(\\mathbf{v, f}) = \\frac{ \\partial^2 P^{loc}(\\mathbf{v(s, f)}) }{\\partial s_i \\partial s_j}

        Parameters:
            npv : Dual2
                A local currency NPV of a period of a leg.
        """  # noqa: E501
        # instrument-instrument cross gamma:
        _ = np.tensordot(gradient(npv, self.pre_variables, order=2), self.grad_s_vT_pre, (1, 1))
        _ = np.tensordot(self.grad_s_vT_pre, _, (1, 0))

        _ += np.tensordot(self.grad_s_s_vT_pre, gradient(npv, self.pre_variables), (2, 0))
        grad_s_sT_P: NDArray[Nf64] = _
        return grad_s_sT_P
        # grad_s_sT_P = np.matmul(
        #     self.grad_s_vT_pre,
        #     np.matmul(
        #         npv.gradient(self.pre_variables, order=2), self.grad_s_vT_pre.T
        #     ),
        # )
        # grad_s_sT_P += np.matmul(
        #     self.grad_s_s_vT_pre, npv.gradient(self.pre_variables)[:, None]
        # )[:, :, 0]

    def gradp_f_vT_Ploc(
        self, npv: Dual | Dual2 | Variable, fx_vars: Sequence[str]
    ) -> NDArray[Nf64]:
        """
        2d array of (partial) derivatives of local currency PV with respect to
        FX rate variables and curve variables, of size (len(fx_vars), pre_n).

        .. math::

           \\nabla_\\mathbf{f} \\nabla_\\mathbf{v}^\\mathbf{T} P^{loc}(\\mathbf{v, f}) = \\frac{ \\partial ^2 P^{loc}(\\mathbf{v, f)}) }{\\partial f_i \\partial v_j}

        Parameters:
            npv : Dual2
                A local currency NPV of a period of a leg.
            fx_vars : list or tuple of str
                The variable tags for automatic differentiation of FX rate sensitivity
        """  # noqa: E501
        grad_x_xT_Ploc = gradient(npv, self.pre_variables + tuple(fx_vars), order=2)
        grad_f_vT_Ploc = grad_x_xT_Ploc[self.pre_n :, : self.pre_n]
        return grad_f_vT_Ploc

    def grad_f_sT_Ploc(self, npv: Dual | Dual2 | Variable, fx_vars: Sequence[str]) -> NDArray[Nf64]:
        """
        2d array of derivatives of local currency PV with respect to calibrating
        instruments, of size (pre_m, pre_m).

        .. math::

           \\nabla_\\mathbf{f} \\nabla_\\mathbf{s}^\\mathbf{T} P^{loc}(\\mathbf{v(s, f), f}) = \\frac{ d^2 P^{loc}(\\mathbf{v(s, f), f)}) }{d f_i \\partial s_j}

        Parameters:
            npv : Dual2
                A local currency NPV of a period of a leg.
            fx_vars : list or tuple of str
                The variable tags for automatic differentiation of FX rate sensitivity
        """  # noqa: E501
        # fx_rate-instrument cross gamma:
        _ = np.tensordot(
            self.grad_f_vT_pre(fx_vars),
            gradient(npv, self.pre_variables, order=2),
            (1, 0),
        )
        _ += self.gradp_f_vT_Ploc(npv, fx_vars)
        _ = np.tensordot(_, self.grad_s_vT_pre, (1, 1))
        _ += np.tensordot(self.grad_f_s_vT_pre(fx_vars), gradient(npv, self.pre_variables), (2, 0))
        grad_f_sT_Ploc: NDArray[Nf64] = _
        return grad_f_sT_Ploc

    def grad_f_fT_Ploc(self, npv: Dual | Dual2 | Variable, fx_vars: Sequence[str]) -> NDArray[Nf64]:
        """
        2d array of derivatives of local currency PV with respect to FX rate variables,
        of size (len(fx_vars), len(fx_vars)).

        .. math::

           \\nabla_\\mathbf{f} \\nabla_\\mathbf{s}^\\mathbf{T} P^{loc}(\\mathbf{v(s, f), f}) = \\frac{ d^2 P^{loc}(\\mathbf{v(s, f), f)}) }{d f_i d f_j}

        Parameters:
            npv : Dual2
                A local currency NPV of a period of a leg.
            fx_vars : list or tuple of str
                The variable tags for automatic differentiation of FX rate sensitivity
        """  # noqa: E501
        # fx_rate-instrument cross gamma:
        gradp_f_vT_Ploc = self.gradp_f_vT_Ploc(npv, fx_vars)
        grad_f_vT_pre = self.grad_f_vT_pre(fx_vars)
        grad_v_Ploc = gradient(npv, self.pre_variables)
        grad_v_vT_Ploc = gradient(npv, self.pre_variables, order=2)

        _ = gradient(npv, fx_vars, order=2)
        _ += np.tensordot(self.grad_f_f_vT_pre(fx_vars), grad_v_Ploc, (2, 0))
        _ += np.tensordot(grad_f_vT_pre, gradp_f_vT_Ploc, (1, 1))
        _ += np.tensordot(gradp_f_vT_Ploc, grad_f_vT_pre, (1, 1))

        __ = np.tensordot(grad_f_vT_pre, grad_v_vT_Ploc, (1, 0))
        __ = np.tensordot(__, grad_f_vT_pre, (1, 1))

        grad_f_f_Ploc: NDArray[Nf64] = _ + __
        return grad_f_f_Ploc

    def grad_s_sT_Pbase(
        self,
        npv: Dual | Dual2 | Variable,
        grad_s_sT_P: NDArray[Nf64],
        f: Dual | Dual2 | Variable,
    ) -> NDArray[Nf64]:
        """
        2d array of derivatives of base currency PV with respect to calibrating
        instrument rate variables, of size (pre_m, pre_m).

        .. math::

           \\nabla_\\mathbf{s} \\nabla_\\mathbf{s}^\\mathbf{T} P^{bas}(\\mathbf{v(s, f), f})

        Parameters:
            npv : Dual or Dual2
                A local currency NPV of a period of a leg.
            grad_s_sT_P : ndarray
                The local currency gamma risks w.r.t. calibrating instrument variables.
            f : Dual or Dual2
                The local:base FX rate.
        """
        grad_s_f = self.grad_s_f_pre(f)
        grad_s_sT_f = self.grad_s_sT_f_pre(f)
        grad_s_P = self.grad_s_Ploc(npv)

        _ = _dual_float(f) * grad_s_sT_P
        _ += np.tensordot(grad_s_f[:, None], grad_s_P[None, :], (1, 0))
        _ += np.tensordot(grad_s_P[:, None], grad_s_f[None, :], (1, 0))
        _ += _dual_float(npv) * grad_s_sT_f  # <- use float to cast float array not Dual

        grad_s_sT_Pbas: NDArray[Nf64] = _
        return grad_s_sT_Pbas

    def grad_f_sT_Pbase(
        self,
        npv: Dual | Dual2 | Variable,
        grad_f_sT_P: NDArray[Nf64],
        f: Dual | Dual2 | Variable,
        fx_vars: Sequence[str],
    ) -> NDArray[Nf64]:
        """
        2d array of derivatives of base currency PV with respect to FX variables and
        calibrating instrument rate variables, of size (len(fx_vars), pre_m).

        .. math::

           \\nabla_\\mathbf{f} \\nabla_\\mathbf{s}^\\mathbf{T} P^{bas}(\\mathbf{v(s, f), f})

        Parameters:
            npv : Dual or Dual2
                A local currency NPV of a period of a leg.
            grad_f_sT_P : ndarray
                The local currency gamma risks w.r.t. FX rate variables and
                calibrating instrument variables.
            f : Dual or Dual2
                The local:base FX rate.
            fx_vars : list or tuple of str
                The variable tags for automatic differentiation of FX rate sensitivity
        """
        grad_s_f = self.grad_s_f_pre(f)
        grad_f_f = self.grad_f_f(f, fx_vars)
        grad_s_P = self.grad_s_Ploc(npv)
        grad_f_P = self.grad_f_Ploc(npv, fx_vars)
        grad_f_sT_f = self.grad_f_sT_f_pre(f, fx_vars)

        _ = _dual_float(f) * grad_f_sT_P
        _ += np.tensordot(grad_f_f[:, None], grad_s_P[None, :], (1, 0))
        _ += np.tensordot(grad_f_P[:, None], grad_s_f[None, :], (1, 0))
        _ += _dual_float(npv) * grad_f_sT_f  # <- use float to cast float array not Dual

        grad_s_sT_Pbas: NDArray[Nf64] = _
        return grad_s_sT_Pbas

    def grad_f_fT_Pbase(
        self,
        npv: Dual | Dual2 | Variable,
        grad_f_fT_P: NDArray[Nf64],
        f: Dual | Dual2 | Variable,
        fx_vars: Sequence[str],
    ) -> NDArray[Nf64]:
        """
        2d array of derivatives of base currency PV with respect to calibrating
        instrument rate variables, of size (pre_m, pre_m).

        .. math::

           \\nabla_\\mathbf{s} \\nabla_\\mathbf{s}^\\mathbf{T} P^{bas}(\\mathbf{v(s, f), f})

        Parameters:
            npv : Dual or Dual2
                A local currency NPV of a period of a leg.
            grad_f_fT_P : ndarray
                The local currency gamma risks w.r.t. FX rate variables.
            f : Dual or Dual2
                The local:base FX rate.
            fx_vars : list or tuple of str
                The variable tags for automatic differentiation of FX rate sensitivity
        """
        # grad_s_f = self.grad_s_f_pre(f)
        grad_f_f = self.grad_f_f(f, fx_vars)
        # grad_s_P = self.grad_s_Ploc(npv)
        grad_f_P = self.grad_f_Ploc(npv, fx_vars)
        grad_f_fT_f = self.grad_f_fT_f_pre(f, fx_vars)

        _ = _dual_float(f) * grad_f_fT_P
        _ += np.tensordot(grad_f_f[:, None], grad_f_P[None, :], (1, 0))
        _ += np.tensordot(grad_f_P[:, None], grad_f_f[None, :], (1, 0))
        _ += _dual_float(npv) * grad_f_fT_f  # <- use float to cast float array not Dual

        grad_s_sT_Pbas: NDArray[Nf64] = _
        return grad_s_sT_Pbas


class Solver(Gradients, _WithState):
    """
    A numerical solver to determine node values on multiple pricing objects simultaneously.

    Parameters
    ----------
    curves : sequence
        Sequence of :class:`Curve` or :class:`Smile` objects where each one
        has been individually configured for its node dates and interpolation structures,
        and has a unique ``id``. Each object will be dynamically updated/mutated by the Solver.
    surfaces : sequence
        Sequence of :class:`Surface` objects where each *surface* has been configured
        with a unique ``id``. Each *surface* will be dynamically updated/mutated.
        Internally, *Surfaces* are appended to ``curves`` and provide nothing more than
        organisational distinction.
    instruments : sequence
        Sequence of calibrating instrument specifications that will be used by
        the solver to determine the solved curves. See notes.
    s : sequence
        Sequence of objective rates that each solved calibrating instrument will solve
        to. Must have the same length and order as ``instruments``.
    weights : sequence, optional
        The weights that should be used within the objective function when determining
        the loss function associated with each calibrating instrument. Should be of
        same length as ``instruments``. If not given defaults to all ones.
    algorithm : str in {"levenberg_marquardt", "gauss_newton", "gradient_descent"}
        The optimisation algorithm to use when solving curves via :meth:`iterate`.
    fx : FXForwards, FXRates, optional
        The fx object used in FX rate calculations for ``instruments`` rates or sensitivities.
    instrument_labels : list of str, optional
        The names of the calibrating instruments which will be used in delta risk
        outputs.
    id : str, optional
        The identifier used to denote the instance and attribute risk factors.
    pre_solvers : list,
        A collection of ``Solver`` s that have already determined curves to which this
        instance has a dependency. Used for aggregation of risk sensitivities.
    max_iter : int
        The maximum number of iterations to perform.
    func_tol : float
        The tolerance to determine convergence if the objective function is lower
        than a specific value. Defaults to 1e-12.
    conv_tol : float
        The tolerance to determine convergence if successive objective function
        values are similar. Defaults to 1e-17.
    ini_lambda : 3-tuple of float, optional
        Parameters to control the Levenberg-Marquardt algorithm, defined as the
        initial lambda value, the scaling factor for a successful iteration and the
        scaling factor for an unsuccessful iteration. Defaults to (1000, 0.25, 2).
    callback : callable, optional
        Is called after each iteration. Used for debugging or optimisation.

    Notes
    -----
    Once initialized, the ``Solver`` will numerically determine and set, via mutation, all the
    relevant node values on each *Curve*, *Smile*, or *Surface* simultaneously by
    calling :meth:`iterate`. This mutation of those pricing objects will override any local AD
    variables pre-configured by a user and use the *Solver's* own variable tags, for proper
    *delta* and *gamma* management.

    Each *Instrument* provided to ``instruments`` can have its pricing objects (i.e. ``curves``
    and ``vol``) and ``metric`` preset at initialization, so that the
    :meth:`~rateslib.instruments.Metrics.rate` method for each *Instrument* in scope is
    well defined. As an example,

    .. code-block:: python

       instruments=[
           ...
           FXCall([args], curves=[None, eur, None, usd], vol=smile, metric="vol"),
           ...
       ]

    The ``fx`` argument used in the :meth:`~rateslib.instruments.Metrics.rate` call will
    be passed directly to each *Instrument* from the *Solver's* ``fx`` argument, being
    representative of a consistent *FXForwards* object for all *Instruments*.

    If the pricing objects and/or *metric* are not preset then the *Solver* ``instruments`` can be
    given as a tuple where the second and third items are a tuple and dict representing positional
    and keyword arguments passed directly to the :meth:`~rateslib.instruments.Metrics.rate`
    method. Usually using the keyword arguments, and using an empty positional arguments tuple,
    is more explicit. An example is:

    .. code-block:: python

       instruments=[
           ...
           (FixedRateBond([args]), (), {"curves": bond_curve, "metric": "ytm"}),
           ...
       ]

    Examples
    --------

    See the documentation user guide :ref:`here <c-solver-doc>`.

    """

    def __init__(
        self,
        curves: Sequence[Curve | FXDeltaVolSmile | FXSabrSmile] = (),
        surfaces: Sequence[FXDeltaVolSurface | FXSabrSurface] = (),
        instruments: Sequence[SupportsRate] = (),
        s: Sequence[DualTypes] = (),
        weights: Sequence[float] | NoInput = NoInput(0),
        algorithm: str_ = NoInput(0),
        fx: FXForwards | FXRates | NoInput = NoInput(0),
        instrument_labels: Sequence[str] | NoInput = NoInput(0),
        id: str_ = NoInput(0),  # noqa: A002
        pre_solvers: Sequence[Solver] = (),
        max_iter: int = 100,
        func_tol: float = 1e-11,
        conv_tol: float = 1e-14,
        ini_lambda: tuple[float, float, float] | NoInput = NoInput(0),
        callback: Callable[[Solver, int, NDArray[Nobject]], None] | NoInput = NoInput(0),
    ) -> None:
        self._do_not_validate_ = False
        self.callback = callback
        self.algorithm = _drb(defaults.algorithm, algorithm).lower()
        self.ini_lambda = _drb(defaults.ini_lambda, ini_lambda)
        self.id: str = _drb(uuid4().hex[:5] + "_", id)  # 1 in a million clash
        self.m = len(instruments)
        self.func_tol, self.conv_tol, self.max_iter = func_tol, conv_tol, max_iter
        self.pre_solvers = tuple(pre_solvers)

        # validate `id`s so that DataFrame indexing does not share duplicated keys.
        if len(set([self.id] + [p.id for p in self.pre_solvers])) < 1 + len(self.pre_solvers):
            raise ValueError(
                "Solver `id`s must be unique when supplying `pre_solvers`, "
                f"got ids: {[self.id] + [p.id for p in self.pre_solvers]}",
            )

        # validate `s` and `instruments` with a naive length comparison
        if len(s) != len(instruments):
            raise ValueError(
                f"`s: {len(s)}` (rates)  must be same length as `instruments: {len(instruments)}`."
            )
        self.s = np.asarray(s)

        # validate `instrument_labels` if given is same length as `m`
        if not isinstance(instrument_labels, NoInput):
            if self.m != len(instrument_labels):
                raise ValueError(
                    f"`instrument_labels: {len(instrument_labels)}` must be same length as "
                    f"`instruments: {len(instruments)}`."
                )
            else:
                self.instrument_labels = tuple(instrument_labels)
        else:
            self.instrument_labels = tuple(f"{self.id}{i}" for i in range(self.m))

        if isinstance(weights, NoInput):
            self.weights: NDArray[Nf64] = np.ones(len(instruments), dtype=np.float64)
        else:
            if len(weights) != self.m:
                raise ValueError(
                    f"`weights: {len(weights)}` must be same length as "
                    f"`instruments: {len(instruments)}`."
                )
            self.weights = np.asarray(weights)
        self.W = np.diag(self.weights)

        # `surfaces` are treated identically to `curves`. Introduced in PR
        self.curves = {
            curve.id: curve
            for curve in list(curves) + list(surfaces)
            if type(curve) not in [ProxyCurve, CompositeCurve, MultiCsaCurve]
            # Proxy and Composite curves have no parameters of their own
        }
        self.variables = ()
        for curve in self.curves.values():
            curve._set_ad_order(1)  # solver uses gradients in optimisation
            self.variables += curve._get_node_vars()
        self.n = len(self.variables)

        # aggregate and organise variables and labels including pre_solvers
        self.pre_curves: dict[str, Curve | FXVolObj] = {}
        self.pre_variables: tuple[str, ...] = ()
        self.pre_instrument_labels: tuple[tuple[str, str], ...] = ()
        self.pre_instruments: tuple[tuple[SupportsRate, tuple[Any, ...], dict[str, Any]], ...] = ()
        self.pre_rate_scalars = []
        self.pre_m, self.pre_n = self.m, self.n
        curve_collection: list[Curve | FXVolObj] = []
        for pre_solver in self.pre_solvers:
            self.pre_variables += pre_solver.pre_variables
            self.pre_instrument_labels += pre_solver.pre_instrument_labels
            self.pre_instruments += pre_solver.pre_instruments
            self.pre_rate_scalars.extend(pre_solver.pre_rate_scalars)
            self.pre_m += pre_solver.pre_m
            self.pre_n += pre_solver.pre_n
            self.pre_curves.update(pre_solver.pre_curves)
            curve_collection.extend(pre_solver.pre_curves.values())
        self.pre_curves.update(self.curves)
        self.pre_curves.update(
            {
                curve.id: curve
                for curve in curves
                if type(curve) in [ProxyCurve, CompositeCurve, MultiCsaCurve]
                # Proxy and Composite curves added to the collection without variables
            },
        )
        curve_collection.extend(curves)
        for curve1, curve2 in combinations(curve_collection, 2):
            if curve1.id == curve2.id:
                raise ValueError(
                    "`curves` must each have their own unique `id`. If using "
                    "pre-solvers as part of a dependency chain a curve can only be "
                    "specified as a variable in one solver.",
                )
        self.pre_variables += self.variables
        self.pre_instrument_labels += tuple((self.id, lbl) for lbl in self.instrument_labels)

        # Final elements
        self._ad = 1
        self.fx: FXRates | FXForwards | NoInput = fx
        if isinstance(self.fx, FXRates | FXForwards):
            self.fx._set_ad_order(1)
        elif not isinstance(self.fx, NoInput):
            raise ValueError(
                "`fx` argument to Solver must be either FXRates, FXForwards or NoInput(0)."
            )
        self.instruments: tuple[tuple[SupportsRate, tuple[Any, ...], dict[str, Any]], ...] = tuple(
            self._parse_instrument(inst) for inst in instruments
        )
        self.pre_instruments += self.instruments
        self.rate_scalars = tuple(inst[0]._rate_scalar for inst in self.instruments)
        self.pre_rate_scalars += self.rate_scalars

        # TODO need to check curves associated with fx object and set order.
        # self._reset_properties_()  performed in iterate
        self._result = {
            "status": "INITIALISED",
            "state": 0,
            "g": None,
            "iterations": 0,
            "time": None,
        }
        self.iterate()

    def __repr__(self) -> str:
        return f"<rl.Solver:{self.id} at {hex(id(self))}>"

    def _set_new_state(self) -> None:
        self._states = self._associated_states()
        self._state = hash(sum(v for v in self._states.values()))

    @property
    def _do_not_validate(self) -> bool:
        return self._do_not_validate_

    @_do_not_validate.setter
    def _do_not_validate(self, value: bool) -> None:
        self._do_not_validate_ = value
        for solver in self.pre_solvers:
            solver._do_not_validate = value

    def _validate_state(self) -> None:
        if self._do_not_validate:
            return None  # do not perform state validation during iterations
        if self._state != self._get_composited_state():
            # then something has been mutated
            states_ = self._associated_states()
            fx_state_ = states_.pop("fx")

            for k, v in states_.items():
                if self._states[k] != v:
                    raise ValueError(
                        "The `curves` associated with `solver` have been updated without the "
                        "`solver` performing additional iterations.\n"
                        f"In particular the object with id: '{k}' contained in solver with id: "
                        f"'{self.id}' is detected to have been mutated.\n"
                        "Calculations are prevented in this "
                        "state because they will likely be erroneous or a consequence of a bad "
                        "design pattern."
                    )

            if not isinstance(self.fx, NoInput) and fx_state_ != self._states["fx"]:
                warnings.warn(
                    f"The `fx` object associated with `solver` having id '{self.id}' "
                    "has been updated without "
                    "the `solver` performing additional iterations.\nCalculations can still be "
                    "performed but, dependent upon those updates, errors may be negligible "
                    "or significant.",
                    UserWarning,
                )

    @staticmethod
    def _validate_and_get_state(obj: Any) -> int:
        obj._validate_state()
        return obj._state  # type: ignore[no-any-return]

    def _associated_states(self) -> dict[str, int]:
        states_: dict[str, int] = {
            k: self._validate_and_get_state(v) for k, v in self.pre_curves.items()
        }
        if not isinstance(self.fx, NoInput):
            states_["fx"] = self._validate_and_get_state(self.fx)
        else:
            states_["fx"] = 0
        return states_

    def _get_composited_state(self) -> int:
        _: int = hash(sum(v for v in self._associated_states().values()))
        return _

    def _parse_instrument(
        self, value: SupportsRate | tuple[SupportsRate, tuple[Any, ...], dict[str, Any]]
    ) -> tuple[SupportsRate, tuple[Any, ...], dict[str, Any]]:
        """
        Parses different input formats for an instrument given to the ``Solver``.

        Parameters
        ----------
        value : Instrument or 3-tuple.
            If a 3-tuple then it must have the following items:

            - The ``Instrument``.
            - Positional args supplied to the ``rate`` method as a tuple, or None.
            - Keyword args supplied to the ``rate`` method as a dict, or None.

        Returns
        -------
        tuple :
            A 3-tuple attaching the self solver and self fx object as pricing params.

        Examples
        --------
        ``value=Instrument()``

        ``value=(Instrument(), (curve, None, fx), {"other_arg": 10.0})``

        ``value=(Instrument(), None, {"other_arg": 10.0})``

        ``value=(Instrument(), (curve, None, fx), None)``

        ``value=(Instrument(), (curve,), {})``
        """
        if not isinstance(value, tuple):
            # is a direct Instrument so convert to tuple with pricing params
            _: tuple[SupportsRate, tuple[Any, ...], dict[str, Any]] = (
                value,
                tuple(),
                {"solver": self, "fx": self.fx},
            )
            return _
        else:
            # object is tuple
            if len(value) != 3:
                raise ValueError(
                    "`Instrument` supplied to `Solver` as tuple must be a 3-tuple of "
                    "signature: (Instrument, positional args[tuple], keyword "
                    "args[dict]).",
                )
            ret0 = value[0]
            ret1: tuple[Any, ...] = tuple()
            ret2: dict[str, Any] = {"solver": self, "fx": self.fx}
            if not (value[1] is None or value[1] == ()):
                ret1 = value[1]
            if not (value[2] is None or value[2] == {}):
                ret2 = {**ret2, **value[2]}
            return ret0, ret1, ret2

    def _reset_properties_(self, dual2_only: bool = False) -> None:
        """
        Set all calculated attributes to `None` requiring re-evaluation.

        Parameters
        ----------
        dual2_only : bool
            Choose whether to reset properties only for the calculation of the
            properties whose derivation **requires** Dual2 datatypes. Since the
            ``Solver`` iterates ``Curve`` s by default it necessarily uses Dual
            datatypes and first order derivatives. For the calculation of:

              - ``J2`` and ``J2_pre``:
                :math:`\frac{\\partial^2 r_i}{\\partial v_j \\partial v_k}`
              - ``grad_s_s_vT`` and ``grad_s_s_vT_pre``:
                :math:`\frac{\\partial^2 v_i}{\\partial s_j \\partial s_k}`

        Returns
        -------
        None
        """
        if not dual2_only:
            self._v: NDArray[Nobject] | None = None  # depends on self.curves
            self._r: NDArray[Nobject] | None = (
                None  # depends on self.pre_curves and self.instruments
            )
            self._r_pre: NDArray[Nobject] | None = None  # depends on pre_solvers and self.r
            self._x: NDArray[Nobject] | None = None  # depends on self.r, self.s
            self._g: Dual | Dual2 | None = None  # depends on self.x, self.weights
            self._J: NDArray[Nf64] | None = None  # depends on self.r
            self._grad_s_vT: NDArray[Nf64] | None = (
                None  # final_iter_dual: depends on self.s and iteration
            )
            # fixed_point_iter: depends on self.f
            # final_iter_anal: depends on self.J
            self._grad_s_vT_pre: NDArray[Nf64] | None = (
                None  # depends on self.grad_s_vT and pre_solvers.
            )

        self._J2 = None  # defines its own self.r under dual2
        self._J2_pre = None  # depends on self.r and pre_solvers
        self._grad_s_s_vT = None  # final_iter: depends on self.J2 and self.grad_s_vT
        # finite_diff: TODO update comment
        self._grad_s_s_vT_pre = None  # final_iter: depends on pre versions of above
        # finite_diff: TODO update comment

        # self._grad_v_v_f = None
        # self._Jkm = None  # keep manifold originally used for exploring J2 calc method

    @_validate_states
    def _get_pre_curve(self, obj: str) -> Curve:
        ret: Curve | FXVols = self.pre_curves[obj]
        if isinstance(ret, _BaseCurve):
            return ret
        else:
            raise ValueError(
                f"A type of `Curve` object was sought with id:'{obj}' from Solver but another "
                f"type object was returned:'{type(ret)}'."
            )

    @_validate_states
    def _get_pre_fxvol(self, obj: str) -> FXVols:
        _: Curve | FXVols = self.pre_curves[obj]
        if isinstance(_, FXVols):
            return _
        else:
            raise ValueError(
                f"A type of `FXVol` object was sought with id:'{obj}' from Solver but another "
                f"type object was returned:'{type(_)}'."
            )

    @_validate_states
    def _get_fx(self) -> FXRates | FXForwards | NoInput:
        return self.fx

    @property
    def result(self) -> dict[str, Any]:
        """
        Show statistics relevant to the last *Solver* iteration.

        Valid *Solver* states are:

        - 1: Success within tolerance of objective function close to zero.
        - 2: Success within tolerance of successive iteration values.
        - -1: Failed to satisfy tolerance after maximal allowed iteration.
        """
        return self._result

    @property
    def v(self) -> NDArray[Nobject]:
        """
        1d array of curve node variables for each ordered curve, size (n,).

        Depends on ``self.curves``.
        """
        if self._v is None:
            self._v = np.block([_._get_node_vector() for _ in self.curves.values()])
        return self._v

    @property
    def r(self) -> NDArray[Nobject]:  # type: ignore[override]
        """
        1d array of mid-market rates of each calibrating instrument with given curves,
        size (m,).

        Depends on ``self.pre_curves`` and ``self.instruments``.
        """
        if self._r is None:
            self._r = np.array([_[0].rate(*_[1], **_[2]) for _ in self.instruments])
            # solver and fx are passed by default via parse_args to get string curves
        return self._r

    @property
    def r_pre(self) -> NDArray[Nobject]:  # type: ignore[override]
        if len(self.pre_solvers) == 0:
            return self.r

        if self._r_pre is None:
            r_pre = np.empty(self.pre_m, dtype="object")

            i = 0
            for pre_solver in self.pre_solvers:
                m = pre_solver.pre_m
                r_pre[i : i + m] = pre_solver.r_pre
                i = i + m

            # create bottom right block
            r_pre[-self.m :] = self.r
            self._r_pre = r_pre
        return self._r_pre

    @property
    def x(self) -> NDArray[Nobject]:
        """
        1d array of error in each calibrating instrument rate, of size (m,).

        .. math::

           \\mathbf{x} = \\mathbf{r-S}

        Depends on ``self.r`` and ``self.s``.
        """
        if self._x is None:
            self._x = self.r - self.s
        return self._x

    @property
    def error(self) -> Series[float]:
        """
        Return the error in calibrating instruments, including ``pre_solvers``, scaled
        to the risk representation factor.

        Returns
        -------
        Series
        """
        pre_s: Series[float] | None = None
        for pre_solver in self.pre_solvers:
            if pre_s is None:
                pre_s = pre_solver.error
            else:
                pre_s = concat([pre_solver.error, pre_s])

        _: Series[float] = Series(
            self.x.astype(float) * 100 / self.rate_scalars,
            index=MultiIndex.from_tuples([(self.id, inst) for inst in self.instrument_labels]),
        )
        if pre_s is None:
            s: Series[float] = _
        else:
            s = concat([pre_s, _])
        return s

    @property
    def g(self) -> Dual | Dual2:  # type: ignore[override]
        """
        Objective function scalar value of the solver;

        .. math::

           g = \\mathbf{(r-S)^{T}W(r-S)}

        Depends on ``self.x`` and ``self.weights``.
        """
        if self._g is None:
            self._g = np.dot(self.x, self.weights * self.x)
        return self._g

    # def Jkm(self, extra_vars=[]):
    #     """
    #     2d Jacobian array of rates with respect to discount factors, of size (n, m); :math:`[J]_{i,j} = \\frac{\\partial r_j}{\\partial v_i}`.  # noqa: E501
    #     """
    #     _Jkm = np.array([rate.gradient(self.variables + extra_vars, keep_manifold=True) for rate in self.r]).T  # noqa: E501
    #     return _Jkm

    def _update_step_(self, algorithm: str) -> NDArray[Nobject]:
        if algorithm == "gradient_descent":
            grad_v_g = gradient(self.g, self.variables)
            y = np.matmul(self.J.transpose(), grad_v_g[:, np.newaxis])[:, 0]
            alpha = np.dot(y, self.weights * self.x) / np.dot(y, self.weights * y)
            v_1: NDArray[Nobject] = self.v - grad_v_g * alpha.real
        elif algorithm == "gauss_newton":
            if self.J.shape[0] == self.J.shape[1]:  # square system
                A = self.J.transpose()
                b = -np.array([x.real for x in self.x])[:, np.newaxis]
            else:
                A = np.matmul(self.J, np.matmul(self.W, self.J.transpose()))
                b = -0.5 * gradient(self.g, self.variables)[:, np.newaxis]
            delta: NDArray[Nobject] = np.linalg.solve(A, b)[:, 0]
            v_1 = self.v + delta
        elif algorithm == "levenberg_marquardt":
            if self.g_list[-2] < self.g.real:
                # reject previous iteration and rescale lambda:
                self.lambd *= self.ini_lambda[2]
                # self._update_curves_with_parameters(self.v_prev)
            else:
                self.lambd *= self.ini_lambda[1]
            # self.lambd *= self.ini_lambda[2] if self.g_prev < self.g.real else self.ini_lambda[1]
            A = np.matmul(self.J, np.matmul(self.W, self.J.transpose()))
            A += self.lambd * np.eye(self.n)
            b = -0.5 * gradient(self.g, self.variables)[:, np.newaxis]
            delta = np.linalg.solve(A, b)[:, 0]
            v_1 = self.v + delta
        # elif algorithm == "gradient_descent_final":
        #     _ = np.matmul(self.Jkm, np.matmul(self.W, self.x[:, np.newaxis]))
        #     y = 2 * np.matmul(self.Jkm.transpose(), _)[:, 0]
        #     alpha = np.dot(y, self.weights * self.x) / np.dot(y, self.weights * y)
        #     v_1 = self.v - 2 * alpha * _[:, 0]
        elif algorithm == "gauss_newton_final":
            if self.J.shape[0] == self.J.shape[1]:  # square system
                A = self.J.transpose()
                b = -self.x[:, np.newaxis]
            else:
                A = np.matmul(self.J, np.matmul(self.W, self.J.transpose()))
                b = -np.matmul(np.matmul(self.J, self.W), self.x[:, np.newaxis])

            delta = dual_solve(A, b)[:, 0]  # type: ignore[assignment]
            v_1 = self.v + delta
        else:
            raise NotImplementedError(f"`algorithm`: {algorithm} (spelled correctly?)")
        return v_1

    @_new_state_post
    def _update_fx(self) -> None:
        if not isinstance(self.fx, NoInput):
            self.fx.update()  # note: with no variables this only updates states
        for solver in self.pre_solvers:
            solver._update_fx()

    @_no_interior_validation
    def iterate(self) -> None:
        r"""
        Solve the DF node values and update all the ``curves``.

        This method uses a gradient based optimisation routine, to solve for all
        the curve variables, :math:`\mathbf{v}`, as follows,

        .. math::

           \mathbf{v} = \underset{\mathbf{v}}{\mathrm{argmin}} \;\; f(\mathbf{v}) = \underset{\mathbf{v}}{\mathrm{argmin}} \;\; (\mathbf{r(v)} - \mathbf{S})\mathbf{W}(\mathbf{r(v)} - \mathbf{S})^\mathbf{T}

        where :math:`\mathbf{r}` are the mid-market rates of the calibrating
        instruments, :math:`\mathbf{S}` are the observed and target rates, and
        :math:`\mathbf{W}` is the diagonal array of weights.

        Returns
        -------
        None
        """  # noqa: E501

        # Initialise data and clear and caches
        self.g_list: list[float] = [1e10]
        self.lambd: float = self.ini_lambda[0]
        self._reset_properties_()
        # self._update_fx()
        t0 = time()

        # Begin iteration
        for i in range(self.max_iter):
            self.g_list.append(self.g.real)
            if self.g.real < self.g_list[i] and (self.g_list[i] - self.g.real) < self.conv_tol:
                # condition is set to less than to avoid the case where a null update
                # results in the same solution and this is erroneously categorised
                # as a converged solution.
                return self._solver_result(1, i, time() - t0)
            elif self.g.real < self.func_tol:
                return self._solver_result(2, i, time() - t0)

            # v_0 = self.v.copy()
            v_1 = self._update_step_(self.algorithm)
            # self.v_prev = v_0
            self._update_curves_with_parameters(v_1)

            if not isinstance(self.callback, NoInput):
                self.callback(self, i, v_1)

        return self._solver_result(-1, self.max_iter, time() - t0)

    def _solver_result(self, state: int, i: int, time: float) -> None:
        self._result = _solver_result(state, i, self.g.real, time, True, self.algorithm)
        self._set_new_state()

    @_new_state_post
    def _update_curves_with_parameters(self, v_new: NDArray[Nobject]) -> None:
        """Populate the variable curves with the new values"""
        var_counter = 0
        for curve in self.curves.values():
            # this was amended in PR126 as performance improvement to keep consistent `vars`
            # and was restructured in PR## to decouple methods to accomodate vol surfaces
            n_vars = curve._n - curve._ini_solve
            curve._set_node_vector(v_new[var_counter : var_counter + n_vars], self._ad)  # type: ignore[arg-type]
            var_counter += n_vars

        self._update_fx()
        self._reset_properties_()

    def _set_ad_order(self, order: int) -> None:
        """Defines the node DF in terms of float, Dual or Dual2 for AD order calcs."""
        for pre_solver in self.pre_solvers:
            pre_solver._set_ad_order(order=order)
        self._ad = order
        for _, curve in self.pre_curves.items():
            curve._set_ad_order(order)
        if not isinstance(self.fx, NoInput):
            self.fx._set_ad_order(order)
        self._reset_properties_()

    # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
    # Commercial use of this code, and/or copying and redistribution is prohibited.
    # Contact rateslib at gmail.com if this code is observed outside its intended sphere.

    @_validate_states
    @_no_interior_validation
    def delta(
        self, npv: dict[str, Dual], base: str_ = NoInput(0), fx: FX_ = NoInput(0)
    ) -> DataFrame:
        """
        Calculate the delta risk sensitivity of an instrument's NPV to the
        calibrating instruments of the :class:`~rateslib.solver.Solver`, and to
        FX rates.

        Parameters
        ----------
        npv : dict,
            The NPV (Dual) of the instrument or portfolio of instruments to risk.
            Must be indexed by 3-digit currency
            to discriminate between values expressed in different currencies.
        base : str, optional
            The currency (3-digit code) to report risk metrics in. If not given will
            default to the local currency of the cashflows.
        fx : FXRates, FXForwards, optional
            The FX object to use to convert risk metrics. If needed but not given
            will default to the ``fx`` object associated with the
            :class:`~rateslib.solver.Solver`. It is not recommended to use this
            argument with multi-currency instruments, see notes.

        Returns
        -------
        DataFrame

        Notes
        -----

        **Output Structure**

        .. note::

           *Instrument* values are scaled to 1bp (1/10000th of a unit) when they are
           rate based. *FX* values are scaled to pips (1/10000th of an FX rate unit).

        The output ``DataFrame`` has the following structure:

        - A 3-level index by *'type'*, *'solver'*, and *'label'*;

          - **type** is either *'instruments'* or *'fx'*, and fx exposures are only
            calculated and displayed in some cases where genuine FX exposure arises.
          - **solver** lists the different solver ``id`` s to identify between
            different instruments in dependency chains from ``pre_solvers``.
          - **label** lists the given instrument names in each solver using the
            ``instrument_labels``.

        - A 2-level column header index by *'local_ccy'* and *'display_ccy'*;

          - **local_ccy** displays the currency for which cashflows are payable, and
            therefore the local currency risk sensitivity amount.
          - **display_ccy** displays the currency which the local currency risk
            sensitivity has been converted to via an FX transformation.

        Converting a delta from a local currency to another ``base`` currency also
        introduces FX risk to the NPV of the instrument, which is included in the
        output.

        **Best Practice**

        The ``fx`` option is provided to allow tactical and fast conversion of
        delta risks to ``Instruments``. When constructing and pricing multi-currency
        instruments it is likely that the :class:`~rateslib.solver.Solver` used is
        associated with an :class:`~rateslib.fx.FXForwards` object to consistently
        produce FX forward rates within an aribitrage free framework. In that case
        it is more consistent to re-use those FX associations. If such an
        association exists and a direct ``fx`` object is supplied a warning may be
        emitted if they are not the same object.
        """
        self._do_not_validate = True  # state is validated prior to the call
        base, fx = self._get_base_and_fx(base, fx)
        if isinstance(fx, FXRates | FXForwards):
            fx_vars: tuple[str, ...] = fx.variables
        else:
            fx_vars = tuple()

        inst_scalar = np.array(self.pre_rate_scalars) / 100  # instruments scalar
        fx_scalar = 0.0001
        container = {}
        for ccy in npv:
            container[("instruments", ccy, ccy)] = self.grad_s_Ploc(npv[ccy]) * inst_scalar
            container[("fx", ccy, ccy)] = self.grad_f_Ploc(npv[ccy], fx_vars) * fx_scalar

            if not isinstance(base, NoInput) and base != ccy:
                # is validated by  `_get_base_and _fx`
                assert isinstance(fx, FXForwards | FXRates)  # noqa: S101
                # extend the derivatives
                f: Dual | Dual2 = fx.rate(f"{ccy}{base}")  # type: ignore[assignment]
                container[("instruments", ccy, base)] = (
                    self.grad_s_Pbase(
                        npv[ccy],
                        container[("instruments", ccy, ccy)] / inst_scalar,
                        f,
                    )
                    * inst_scalar
                )
                container[("fx", ccy, base)] = (
                    self.grad_f_Pbase(npv[ccy], container[("fx", ccy, ccy)] / fx_scalar, f, fx_vars)
                    * fx_scalar
                )

        # construct the DataFrame from container with hierarchical indexes
        inst_idx = MultiIndex.from_tuples(
            [("instruments",) + label for label in self.pre_instrument_labels],
            names=["type", "solver", "label"],
        )
        fx_idx = MultiIndex.from_tuples(
            [("fx", "fx", f[3:]) for f in fx_vars],
            names=["type", "solver", "label"],
        )
        indexes = {"instruments": inst_idx, "fx": fx_idx}
        r_idx = inst_idx.append(fx_idx)  # type: ignore[no-untyped-call]
        c_idx = MultiIndex.from_tuples([], names=["local_ccy", "display_ccy"])
        df = DataFrame(None, index=r_idx, columns=c_idx)
        for key, array in container.items():
            df.loc[indexes[key[0]], (key[1], key[2])] = array

        if not isinstance(base, NoInput):
            df.loc[r_idx, ("all", base)] = df.loc[r_idx, (slice(None), base)].sum(axis=1)  # type: ignore[index]

        sorted_cols = df.columns.sort_values()
        ret: DataFrame = df.loc[:, sorted_cols].astype("float64")
        self._do_not_validate = False
        return ret

    def _get_base_and_fx(self, base: str_, fx: FX_) -> tuple[str_, FX_]:
        # method is used by delta, gamma, and exo_delta. prohibit fx as scalar because it cannot
        # convert from arbitrary currencies.
        if not isinstance(fx, NoInput | FXRates | FXForwards):
            raise ValueError(
                "`fx` used in sensitivity calculations cannot be a scalar. An FXRates or "
                "FXForwards object is required, or the input left as NoInput(0), in which case "
                "the `fx` object associated with a Solver is used in place."
            )

        if not isinstance(base, NoInput):
            base = base.lower()
            # then a valid fx object that can convert is required.
            if not isinstance(fx, FXRates | FXForwards) and isinstance(self.fx, NoInput):
                raise ValueError(
                    "`base` is given but `fx` is not given as either FXRates or FXForwards, "
                    "and Solver does not contain its own `fx` attributed which can be substituted."
                )

        if isinstance(fx, NoInput):
            fx = self.fx
        elif not isinstance(self.fx, NoInput) and id(fx) != id(self.fx):
            warnings.warn(
                "Solver contains an `fx` object but an `fx` argument has been "
                "supplied as object which is not the same. This can lead to risk sensitivity "
                "inconsistencies, mathematically.",
                UserWarning,
            )

        return base, fx

    @_validate_states
    @_no_interior_validation
    def gamma(
        self, npv: dict[str, Dual2], base: str_ = NoInput(0), fx: FX_ = NoInput(0)
    ) -> DataFrame:
        """
        Calculate the cross-gamma risk sensitivity of an instrument's NPV to the
        calibrating instruments of the :class:`~rateslib.solver.Solver`.

        Parameters
        ----------
        npv : Dual2,
            The NPV of the instrument or composition of instruments to risk.
        base : str, optional
            The currency (3-digit code) to report risk metrics in. If not given will
            default to the local currency of the cashflows.
        fx : FXRates, FXForwards, optional
            The FX object to use to convert risk metrics. If needed but not given
            will default to the ``fx`` object associated with the
            :class:`~rateslib.solver.Solver`. It is not recommended to use this
            argument with multi-currency instruments, see
            :meth:`Solver.delta <rateslib.solver.Solver.delta>`.

        Returns
        -------
        DataFrame

        Notes
        -----
        .. note::

           *Instrument* values are scaled to 1bp (1/10000th of a unit) when they are
           rate based.

           *FX* values are scaled to pips (1/10000th of an FX unit).

        The output ``DataFrame`` has the following structure:

        - A 5-level index by *'local_ccy'*, *'display_ccy'*, *'type'*, *'solver'*,
          and *'label'*;

          - **local_ccy** displays the currency for which cashflows are payable, and
            therefore the local currency risk sensitivity amount.
          - **display_ccy** displays the currency which the local currency risk
            sensitivity has been converted to via an FX transformation.
          - **type** is either *'instruments'* or *'fx'*, and fx exposures are only
            calculated and displayed in some cases where genuine FX exposure arises.
          - **solver** lists the different solver ``id`` s to identify between
            different instruments in dependency chains from ``pre_solvers``.
          - **label** lists the given instrument names in each solver using the
            ``instrument_labels``.

        - A 3-level column header index using the last three levels of the above;

        Converting a gamma/delta from a local currency to another ``base`` currency also
        introduces FX risk to the NPV of the instrument, which is included in the
        output.

        Examples
        --------
        This example replicates the analytical calculations demonstrated in
        *Pricing and Trading Interest Rate Derivatives (2022)*, derived from
        first principles.
        The results are stated in the cross-gamma grid in figure 22.1.

        .. ipython:: python

           curve_r = Curve(
               nodes={
                   dt(2022, 1, 1): 1.0,
                   dt(2023, 1, 1): 0.99,
                   dt(2024, 1, 1): 0.98,
                   dt(2025, 1, 1): 0.97,
                   dt(2026, 1, 1): 0.96,
                   dt(2027, 1, 1): 0.95,
               },
               id="r"
           )
           curve_z = Curve(
               nodes={
                   dt(2022, 1, 1): 1.0,
                   dt(2023, 1, 1): 0.99,
                   dt(2024, 1, 1): 0.98,
                   dt(2025, 1, 1): 0.97,
                   dt(2026, 1, 1): 0.96,
                   dt(2027, 1, 1): 0.95,
               },
               id="z"
           )
           curve_s = Curve(
               nodes={
                   dt(2022, 1, 1): 1.0,
                   dt(2023, 1, 1): 0.99,
                   dt(2024, 1, 1): 0.98,
                   dt(2025, 1, 1): 0.97,
                   dt(2026, 1, 1): 0.96,
                   dt(2027, 1, 1): 0.95,
               },
               id="s"
           )
           args = dict(termination="1Y", frequency="A", fixing_method="ibor", leg2_fixing_method="ibor")
           instruments = [
               SBS(dt(2022, 1, 1), curves=["r", "s", "s", "s"], **args),
               SBS(dt(2023, 1, 1), curves=["r", "s", "s", "s"], **args),
               SBS(dt(2024, 1, 1), curves=["r", "s", "s", "s"], **args),
               SBS(dt(2025, 1, 1), curves=["r", "s", "s", "s"], **args),
               SBS(dt(2026, 1, 1), curves=["r", "s", "s", "s"], **args),
               SBS(dt(2022, 1, 1), curves=["r", "s", "z", "s"], **args),
               SBS(dt(2023, 1, 1), curves=["r", "s", "z", "s"], **args),
               SBS(dt(2024, 1, 1), curves=["r", "s", "z", "s"], **args),
               SBS(dt(2025, 1, 1), curves=["r", "s", "z", "s"], **args),
               SBS(dt(2026, 1, 1), curves=["r", "s", "z", "s"], **args),
               IRS(dt(2022, 1, 1), "1Y", "A", curves=["r", "s"], leg2_fixing_method="ibor"),
               IRS(dt(2023, 1, 1), "1Y", "A", curves=["r", "s"], leg2_fixing_method="ibor"),
               IRS(dt(2024, 1, 1), "1Y", "A", curves=["r", "s"], leg2_fixing_method="ibor"),
               IRS(dt(2025, 1, 1), "1Y", "A", curves=["r", "s"], leg2_fixing_method="ibor"),
               IRS(dt(2026, 1, 1), "1Y", "A", curves=["r", "s"], leg2_fixing_method="ibor"),
           ]
           solver = Solver(
               curves=[curve_r, curve_s, curve_z],
               instruments=instruments,
               s=[0.]*5 + [0.]*5 + [1.5]*5,
               id="sonia",
               instrument_labels=[
                   "s1", "s2", "s3", "s4", "s5",
                   "z1", "z2", "z3", "z4", "z5",
                   "r1", "r2", "r3", "r4", "r5",
               ],
           )
           irs = IRS(dt(2022, 1, 1), "5Y", "A", notional=-8.3e8, curves=["z", "s"], leg2_fixing_method="ibor", fixed_rate=25.0)
           irs.delta(solver=solver)
           irs.gamma(solver=solver)
        """  # noqa: E501
        if self._ad != 2:
            raise ValueError("`Solver` must be in ad order 2 to use `gamma` method.")

        # new
        base, fx = self._get_base_and_fx(base, fx)
        if isinstance(fx, FXRates | FXForwards):
            fx_vars: tuple[str, ...] = fx.variables
        else:
            fx_vars = tuple()

        inst_scalar = np.array(self.pre_rate_scalars) / 100  # instruments scalar
        fx_scalar = np.ones(len(fx_vars)) * 0.0001
        container: dict[tuple[str, str], dict[tuple[str, ...], Any]] = {}
        for ccy in npv:
            container[(ccy, ccy)] = {}
            container[(ccy, ccy)]["instruments", "instruments"] = self.grad_s_sT_Ploc(
                npv[ccy],
            ) * np.matmul(inst_scalar[:, None], inst_scalar[None, :])
            container[(ccy, ccy)]["fx", "instruments"] = self.grad_f_sT_Ploc(
                npv[ccy],
                fx_vars,
            ) * np.matmul(fx_scalar[:, None], inst_scalar[None, :])
            container[(ccy, ccy)]["instruments", "fx"] = container[(ccy, ccy)][
                ("fx", "instruments")
            ].T
            container[(ccy, ccy)]["fx", "fx"] = self.grad_f_fT_Ploc(npv[ccy], fx_vars) * np.matmul(
                fx_scalar[:, None],
                fx_scalar[None, :],
            )

            if not isinstance(base, NoInput) and base != ccy:
                # validated by `_get_base_and_fx`
                assert isinstance(fx, FXRates | FXForwards)  # noqa: S101
                # extend the derivatives
                f: Dual | Dual2 = fx.rate(f"{ccy}{base}")  # type: ignore[assignment]
                container[(ccy, base)] = {}
                container[(ccy, base)]["instruments", "instruments"] = self.grad_s_sT_Pbase(
                    npv[ccy],
                    container[(ccy, ccy)]["instruments", "instruments"]
                    / np.matmul(inst_scalar[:, None], inst_scalar[None, :]),
                    f,
                ) * np.matmul(inst_scalar[:, None], inst_scalar[None, :])
                container[(ccy, base)]["fx", "instruments"] = self.grad_f_sT_Pbase(
                    npv[ccy],
                    container[(ccy, ccy)]["fx", "instruments"]
                    / np.matmul(fx_scalar[:, None], inst_scalar[None, :]),
                    f,
                    fx_vars,
                ) * np.matmul(fx_scalar[:, None], inst_scalar[None, :])
                container[(ccy, base)]["instruments", "fx"] = container[(ccy, base)][
                    ("fx", "instruments")
                ].T
                container[(ccy, base)]["fx", "fx"] = self.grad_f_fT_Pbase(
                    npv[ccy],
                    container[(ccy, ccy)]["fx", "fx"]
                    / np.matmul(fx_scalar[:, None], fx_scalar[None, :]),
                    f,
                    fx_vars,
                ) * np.matmul(fx_scalar[:, None], fx_scalar[None, :])

        # construct the DataFrame from container with hierarchical indexes
        currencies = list(npv.keys())
        local_keys = [(ccy, ccy) for ccy in currencies]
        base_keys = [] if base is NoInput.blank else [(ccy, base) for ccy in currencies]
        all_keys = sorted(set(local_keys + base_keys))
        inst_keys = [("instruments",) + label for label in self.pre_instrument_labels]
        fx_keys = [("fx", "fx", f[3:]) for f in fx_vars]
        idx_tuples = [c + _ for c in all_keys for _ in inst_keys + fx_keys]
        ridx = MultiIndex.from_tuples(
            list(idx_tuples),
            names=["local_ccy", "display_ccy", "type", "solver", "label"],
        )
        if base is not NoInput.blank:
            ridx = ridx.append(  # type: ignore[no-untyped-call]
                MultiIndex.from_tuples(
                    [("all", base) + _ for _ in inst_keys + fx_keys],
                    names=["local_ccy", "display_ccy", "type", "solver", "label"],
                ),
            )
        cidx = MultiIndex.from_tuples(list(inst_keys + fx_keys), names=["type", "solver", "label"])
        df = DataFrame(None, index=ridx, columns=cidx)
        for key, d in container.items():
            array = np.block(
                [
                    [d[("instruments", "instruments")], d[("instruments", "fx")]],
                    [d[("fx", "instruments")], d[("fx", "fx")]],
                ],
            )
            locator = key + (slice(None), slice(None), slice(None))

            with warnings.catch_warnings():
                # TODO: pandas 3.0.0 can optionally turn off these PerformanceWarnings
                warnings.simplefilter(action="ignore", category=PerformanceWarning)
                df.loc[locator, :] = array

        if not isinstance(base, NoInput):
            # sum over all the base rows to aggregate
            gdf = (
                df.loc[(currencies, base, slice(None), slice(None), slice(None)), :]
                .groupby(level=[2, 3, 4])
                .sum()
            )
            gdf.index = MultiIndex.from_tuples([("all", base) + _ for _ in gdf.index])
            df.loc[("all", base, slice(None), slice(None), slice(None))] = gdf

        return df.astype("float64")

    def _pnl_explain(
        self,
        npv: Dual | Dual2,
        ds: Sequence[float],
        dfx: Sequence[float] | None = None,
        base: str_ = NoInput(0),
        fx: FX_ = NoInput(0),
        order: int = 1,
    ) -> DataFrame:
        """
        Calculate PnL from market movements over delta and, optionally, gamma.

        Parameters
        ----------
        npv : Dual or Dual2,
            The initial NPV of the instrument or composition of instruments to value.
        ds : sequence of float
            The projected market movements of calibrating instruments of the solver,
            scaled to the appropriate value amount matching the delta representation.
        dfx : sequence of float
            The projected market movements of FX rates,
            scaled to the appropriate value amount matching the delta representation.
        base : str, optional
            The currency (3-digit code) to report risk metrics in. If not given will
            default to the local currency of the cashflows.
        fx : FXRates, FXForwards, optional
            The FX object to use to convert risk metrics. If needed but not given
            will default to the ``fx`` object associated with the
            :class:`~rateslib.solver.Solver`.
        order : int in {1, 2}
            Whether to return a first order delta PnL explain or a second order one
            including gamma contribution.

        Returns
        -------
        DataFrame
        """
        raise NotImplementedError()

    @_validate_states
    @_no_interior_validation
    def market_movements(self, solver: Solver) -> DataFrame:
        """
        Determine market movements between the *Solver's* instrument rates and those rates priced
        from a second *Solver*.

        Parameters
        ----------
        solver: Solver
            The other *Solver* whose *Curves* are to be used for measuring the final instrument
            rates of the existing *Solver's* instruments.

        Returns
        -------
        DataFrame

        Notes
        -----
        .. warning::
           Market movement calculations are only possible between *Solvers* whose ``instruments``
           are associated with *Curves* with string ID mappings (which is best practice and
           demonstrated in :ref:`Mechanisms <mechanisms-curves-doc>`). This allows two different
           *Solvers* to contain their own *Curves* (which may or may not be equivalent models),
           and for the instrument rates of one *Solver* to be evaluated by the *Curves* present
           in another *Solver*.
        """
        r_0 = self.r_pre
        r_1 = np.array(
            [
                _[0].rate(*_[1], **{**_[2], "solver": solver, "fx": solver.fx})
                for _ in self.pre_instruments
            ],
        )
        return DataFrame(
            (r_1 - r_0) * 100 / np.array(self.pre_rate_scalars),
            index=self.pre_instrument_labels,
        )

    @_validate_states
    @_no_interior_validation
    def jacobian(self, solver: Solver) -> DataFrame:
        """
        Calculate the Jacobian with respect to another *Solver's* instruments.

        Parameters
        ----------
        solver : Solver
            The other ``Solver`` for which the Jacobian is to be determined.

        Returns
        -------
        DataFrame

        Notes
        -----
        This Jacobian converts risk sensitivities expressed in the underlying *Solver's*
        instruments to the instruments in the other ``solver``.

        .. warning::
           A Jacobian transformation is only possible between *Solvers* whose ``instruments``
           are associated with *Curves* with string ID mappings (which is best practice and
           demonstrated in :ref:`Mechanisms <mechanisms-curves-doc>`). This allows two different
           *Solvers* to contain their own *Curves* (which may or may not be equivalent models),
           and for the instrument rates of one *Solver* to be evaluated by the *Curves* present
           in another *Solver*

        Examples
        --------
        This example creates a Jacobian transformation between par tenor IRS and forward tenor
        IRS. These models are completely consistent and lossless.

        .. ipython:: python

           par_curve = Curve(
               nodes={
                   dt(2022, 1, 1): 1.0,
                   dt(2023, 1, 1): 1.0,
                   dt(2024, 1, 1): 1.0,
                   dt(2025, 1, 1): 1.0,
               },
               id="curve",
           )
           par_instruments = [
               IRS(dt(2022, 1, 1), "1Y", "A", curves="curve"),
               IRS(dt(2022, 1, 1), "2Y", "A", curves="curve"),
               IRS(dt(2022, 1, 1), "3Y", "A", curves="curve"),
           ]
           par_solver = Solver(
               curves=[par_curve],
               instruments=par_instruments,
               s=[1.21, 1.635, 1.99],
               id="par_solver",
               instrument_labels=["1Y", "2Y", "3Y"],
           )

           fwd_curve = Curve(
               nodes={
                   dt(2022, 1, 1): 1.0,
                   dt(2023, 1, 1): 1.0,
                   dt(2024, 1, 1): 1.0,
                   dt(2025, 1, 1): 1.0,
               },
               id="curve"
           )
           fwd_instruments = [
               IRS(dt(2022, 1, 1), "1Y", "A", curves="curve"),
               IRS(dt(2023, 1, 1), "1Y", "A", curves="curve"),
               IRS(dt(2024, 1, 1), "1Y", "A", curves="curve"),
           ]
           s_fwd = [float(_.rate(solver=par_solver)) for _ in fwd_instruments]
           fwd_solver = Solver(
               curves=[fwd_curve],
               instruments=fwd_instruments,
               s=s_fwd,
               id="fwd_solver",
               instrument_labels=["1Y", "1Y1Y", "2Y1Y"],
           )

           par_solver.jacobian(fwd_solver)

        """
        # Get the instrument rates for self solver evaluated using the curves and links of other
        r = np.array(
            [
                _[0].rate(*_[1], **{**_[2], "solver": solver, "fx": solver.fx})
                for _ in self.pre_instruments
            ],
        )
        # Get the gradient of these rates with respect to the variable in other
        grad_v_rT = np.array([gradient(_, solver.pre_variables) for _ in r]).T
        return DataFrame(
            np.matmul(solver.grad_s_vT_pre, grad_v_rT),
            columns=self.pre_instrument_labels,
            index=solver.pre_instrument_labels,
        )

    @_validate_states
    @_no_interior_validation
    def exo_delta(
        self,
        npv: dict[str, Dual | Dual2],
        vars: Sequence[str],  # noqa: A002
        vars_scalar: Sequence[float] | NoInput = NoInput(0),
        vars_labels: Sequence[str] | NoInput = NoInput(0),
        base: str_ = NoInput(0),
        fx: FX_ = NoInput(0),
    ) -> DataFrame:
        """
        Calculate risk sensitivity to user defined, exogenous variables in the
        *Solver Instruments* and the ``npv``.

        See :ref:`What are exogenous variables? <cook-exogenous-doc>` in the cookbook.

        Parameters
        -----------
        npv : dict,
            The NPV (Dual) of the instrument or portfolio of instruments to risk.
            Must be indexed by 3-digit currency
            to discriminate between values expressed in different currencies.
        vars : list[str]
            The variable tags which to determine sensitivities for.
        vars_scalar : list[float], optional
            Scaling factors for each variable, for example converting rates to basis point etc.
            Defaults to ones.
        vars_labels : list[str], optional
            Alternative names to relabel variables in DataFrames.
        base : str, optional
            The currency (3-digit code) to report risk metrics in. If not given will
            default to the local currency of the cashflows.
        fx : FXRates, FXForwards, optional
            The FX object to use to convert risk metrics. If needed but not given
            will default to the ``fx`` object associated with the
            :class:`~rateslib.solver.Solver`. It is not recommended to use this
            argument with multi-currency instruments, see notes.

        Returns
        -------
        DataFrame
        """
        base, fx = self._get_base_and_fx(base, fx)

        if isinstance(vars_scalar, NoInput):
            vars_scalar = [1.0] * len(vars)
        if isinstance(vars_labels, NoInput):
            vars_labels = vars

        container = {}
        for ccy in npv:
            container[("exogenous", ccy, ccy)] = self.grad_f_Ploc(npv[ccy], vars) * vars_scalar

            if not isinstance(base, NoInput) and base != ccy:
                assert isinstance(fx, FXRates | FXForwards)  # noqa S101
                # extend the derivatives
                f: Dual | Dual2 = fx.rate(f"{ccy}{base}")  # type: ignore[assignment]
                container[("exogenous", ccy, base)] = (
                    self.grad_f_Pbase(
                        npv[ccy],
                        container[("exogenous", ccy, ccy)] / vars_scalar,
                        f,
                        vars,
                    )
                    * vars_scalar
                )

        # construct the DataFrame from container with hierarchical indexes
        exo_idx = MultiIndex.from_tuples(
            [("exogenous",) + (self.id, label) for label in vars_labels],
            names=["type", "solver", "label"],
        )

        indexes = {"exogenous": exo_idx}
        r_idx = exo_idx
        c_idx = MultiIndex.from_tuples([], names=["local_ccy", "display_ccy"])
        df = DataFrame(None, index=r_idx, columns=c_idx)
        for key, array in container.items():
            df.loc[indexes[key[0]], (key[1], key[2])] = array

        if not isinstance(base, NoInput):
            df.loc[r_idx, ("all", base)] = df.loc[r_idx, (slice(None), base)].sum(axis=1)  # type: ignore[index]

        sorted_cols = df.columns.sort_values()
        _: DataFrame = df.loc[:, sorted_cols].astype("float64")
        return _


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.
