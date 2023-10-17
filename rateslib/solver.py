from __future__ import annotations

from typing import Optional, Union
from itertools import combinations
from uuid import uuid4
from time import time
import numpy as np
import warnings
from pandas import DataFrame, MultiIndex, concat, Series

from rateslib import defaults
from rateslib.default import NoInput
from rateslib.dual import Dual, Dual2, dual_log, dual_solve
from rateslib.curves import CompositeCurve, ProxyCurve, MultiCsaCurve
from rateslib.fx import FXRates, FXForwards


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


class Gradients:
    """
    A catalogue of all the gradients used in optimisation routines and risk
    sensitivties.
    """

    @property
    def J(self):
        """
        2d Jacobian array of calibrating instrument rates with respect to curve
        variables, of size (n, m);

        .. math::

           [J]_{i,j} = [\\nabla_\\mathbf{v} \\mathbf{r^T}]_{i,j} = \\frac{\\partial r_j}{\\partial v_i}

        Depends on ``self.r``.
        """
        if self._J is None:
            self._J = np.array([rate.gradient(self.variables) for rate in self.r]).T
        return self._J

    @property
    def grad_v_rT(self):
        """
        Alias of ``J``.
        """
        return self.J

    @property
    def J2(self):
        """
        3d array of second derivatives of calibrating instrument rates with
        respect to curve variables, of size (n, n, m);

        .. math::

           [J2]_{i,j,k} = [\\nabla_\\mathbf{v} \\nabla_\\mathbf{v} \\mathbf{r^T}]_{i,j,k} = \\frac{\\partial^2 r_k}{\\partial v_i \\partial v_j}

        Depends on ``self.r``.
        """
        if self._J2 is None:
            if self._ad != 2:
                raise ValueError(
                    "Cannot perform second derivative calculations when ad mode is " f"{self._ad}."
                )

            rates = np.array([_[0].rate(*_[1], **_[2]) for _ in self.instruments])
            # solver is passed in order to extract curves as string
            _ = np.array([rate.gradient(self.variables, order=2) for rate in rates])
            self._J2 = np.transpose(_, (1, 2, 0))
        return self._J2

    @property
    def grad_v_v_rT(self):
        """
        Alias of ``J2``.
        """
        return self.J2  # pragma: no cover

    @property
    def grad_s_vT(self):
        """
        2d Jacobian array of curve variables with respect to calibrating instruments,
        of size (m, n);

        .. math::

           [\\nabla_\\mathbf{s}\\mathbf{v^T}]_{i,j} = \\frac{\\partial v_j}{\\partial s_i} = \\mathbf{J^+}
        """
        if self._grad_s_vT is None:
            self._grad_s_vT = getattr(self, self._grad_s_vT_method)()
        return self._grad_s_vT

    def _grad_s_vT_final_iteration_dual(self, algorithm: Optional[str] = None):
        """
        This is not the ideal method since it requires reset_properties to reassess.
        """
        algorithm = algorithm or self._grad_s_vT_final_iteration_algo
        _s = self.s
        self.s = np.array([Dual(v, f"s{i}") for i, v in enumerate(self.s)])
        self._reset_properties_()
        v_1 = self._update_step_(algorithm)
        s_vars = [f"s{i}" for i in range(self.m)]
        grad_s_vT = np.array([v.gradient(s_vars) for v in v_1]).T
        self.s = _s
        return grad_s_vT

    def _grad_s_vT_final_iteration_analytical(self):
        """Uses a pseudoinverse algorithm on floats"""
        grad_s_vT = np.linalg.pinv(self.J)
        return grad_s_vT

    def _grad_s_vT_fixed_point_iteration(self):
        """
        This is not the ideal method becuase it requires second order and reset props.
        """
        self._set_ad_order(2)
        self._reset_properties_()
        _s = self.s
        self.s = np.array([Dual2(v, f"s{i}") for i, v in enumerate(self.s)])
        s_vars = tuple(f"s{i}" for i in range(self.m))
        grad2 = self.g.gradient(self.variables + s_vars, order=2)
        grad_v_vT_f = grad2[: self.n, : self.n]
        grad_s_vT_f = grad2[self.n :, : self.n]
        grad_s_vT = np.linalg.solve(grad_v_vT_f, -grad_s_vT_f.T).T

        self.s = _s
        self._set_ad_order(1)
        self._reset_properties_()
        return grad_s_vT

    @property
    def grad_s_s_vT(self):
        """
        3d array of second derivatives of curve variables with respect to
        calibrating instruments, of size (m, m, n);

        .. math::

           [\\nabla_\\mathbf{s} \\nabla_\\mathbf{s} \\mathbf{v^T}]_{i,j,k} = \\frac{\\partial^2 v_k}{\\partial s_i \\partial s_j}
        """
        if self._grad_s_s_vT is None:
            self._grad_s_s_vT = self._grad_s_s_vT_final_iteration_analytical()
        return self._grad_s_s_vT

    def _grad_s_s_vT_fwd_difference_method(self):
        """Use a numerical method, iterating through changes in s to calculate."""
        ds = 10 ** (int(dual_log(self.func_tol, 10) / 2))
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
        # self._grad_s_vT_fixed_point_iteration()  # TODO: returns nothing: what is purpose
        return grad_s_s_vT

    def _grad_s_s_vT_final_iteration_analytical(self, use_pre=False):
        """
        Use an analytical formula and second order AD to calculate.

        Not: must have 2nd order AD set to function, and valid properties set to
        function
        """
        if use_pre:
            J2, grad_s_vT = self.J2_pre, self.grad_s_vT_pre
        else:
            J2, grad_s_vT = self.J2, self.grad_s_vT

        _ = np.tensordot(J2, grad_s_vT, (2, 0))  # dv/dr_l * d2r_l / dvdv
        _ = np.tensordot(grad_s_vT, _, (1, 0))  # dv_z /ds * d2v / dv_zdv
        _ = -np.tensordot(grad_s_vT, _, (1, 1))  # dv_h /ds * d2v /dvdv_h
        grad_s_s_vT = _
        return grad_s_s_vT
        # _ = np.matmul(grad_s_vT, np.matmul(J2, grad_s_vT))
        # grad_s_s_vT = -np.tensordot(grad_s_vT, _, (1, 0))
        # return grad_s_s_vT

    # _pre versions incorporate all variables of solver and pre_solvers

    def grad_f_rT_pre(self, fx_vars):
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
        grad_f_rT = np.array([rate.gradient(fx_vars) for rate in self.r_pre]).T
        return grad_f_rT

    @property
    def J2_pre(self):
        """
        3d array of second derivatives of calibrating instrument rates with
        respect to curve variables for all ``Solvers`` including ``pre_solvers``,
        of size (pre_n, pre_n, pre_m);

        .. math::

           [J2]_{i,j,k} = [\\nabla_\\mathbf{v} \\nabla_\\mathbf{v} \\mathbf{r^T}]_{i,j,k} = \\frac{\\partial^2 r_k}{\\partial v_i \\partial v_j}

        Depends on ``self.r`` and ``pre_solvers.J2``.
        """
        if len(self.pre_solvers) == 0:
            return self.J2

        if self._J2_pre is None:
            if self._ad != 2:
                raise ValueError(
                    "Cannot perform second derivative calculations when ad mode is " f"{self._ad}."
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
            _ = np.array([r.gradient(self.pre_variables, order=2) for r in rates])
            J2[:, :, -self.m :] = np.transpose(_, (1, 2, 0))
            self._J2_pre = J2
        return self._J2_pre

    def grad_f_v_rT_pre(self, fx_vars):
        """
        3d array of second derivatives of calibrating instrument rates with respect to
        FX rates and curve variables, of size (len(fx_vars), pre_n, pre_m);

        .. math::

           [\\nabla_\\mathbf{f} \\nabla_\\mathbf{v} \\mathbf{r^T}]_{i,j,k} = \\frac{\\partial^2 r_k}{\\partial f_i \\partial v_j}

        Parameters
        ----------
        fx_vars : list or tuple of str
            The variable name tags for the FX rate sensitivities.
        """
        # FX sensitivity requires reverting through all pre-solvers rates.
        all_gradients = np.array(
            [rate.gradient(self.pre_variables + tuple(fx_vars), order=2) for rate in self.r_pre]
        ).swapaxes(0, 2)

        grad_f_v_rT = all_gradients[self.pre_n :, : self.pre_n, :]
        return grad_f_v_rT

    def grad_f_f_rT_pre(self, fx_vars):
        """
        3d array of second derivatives of calibrating instrument rates with respect to
        FX rates, of size (len(fx_vars), len(fx_vars), pre_m);

        .. math::

           [\\nabla_\\mathbf{f} \\nabla_\\mathbf{f} \\mathbf{r^T}]_{i,j,k} = \\frac{\\partial^2 r_k}{\\partial f_i \\partial f_j}

        Parameters
        ----------
        fx_vars : list or tuple of str
            The variable name tags for the FX rate sensitivities.
        """
        # FX sensitivity requires reverting through all pre-solvers rates.
        grad_f_f_rT = np.array([rate.gradient(fx_vars, order=2) for rate in self.r_pre]).swapaxes(
            0, 2
        )
        return grad_f_f_rT

    @property
    def grad_s_s_vT_pre(self):
        """
        3d array of second derivatives of curve variables with respect to
        calibrating instruments, of size (pre_m, pre_m, pre_n);

        .. math::

           [\\nabla_\\mathbf{s} \\nabla_\\mathbf{s} \\mathbf{v^T}]_{i,j,k} = \\frac{\\partial^2 v_k}{\\partial s_i \\partial s_j}
        """
        if len(self.pre_solvers) == 0:
            return self.grad_s_s_vT

        if self._grad_s_s_vT_pre is None:
            self._grad_s_s_vT_pre = self._grad_s_s_vT_final_iteration_analytical(use_pre=True)
        return self._grad_s_s_vT_pre

    @property
    def grad_v_v_rT_pre(self):
        """
        Alias of ``J2_pre``.
        """
        return self.J2_pre  # pragma: no cover

    def grad_f_s_vT_pre(self, fx_vars):
        """
        3d array of second derivatives of curve variables with respect to
        FX rates and calibrating instrument rates, of size (len(fx_vars), pre_m, pre_n);

        .. math::

           [\\nabla_\\mathbf{f} \\nabla_\\mathbf{s} \\mathbf{v^T}]_{i,j,k} = \\frac{\\partial^2 v_k}{\\partial f_i \\partial s_j}

        Parameters
        ----------
        fx_vars : list or tuple of str
            The variable name tags for the FX rate sensitivities.
        """
        # FX sensitivity requires reverting through all pre-solvers rates.
        _ = -np.tensordot(self.grad_f_v_rT_pre(fx_vars), self.grad_s_vT_pre, (1, 1)).swapaxes(1, 2)
        _ = np.tensordot(_, self.grad_s_vT_pre, (2, 0))
        grad_f_s_vT = _
        return grad_f_s_vT

    def grad_f_f_vT_pre(self, fx_vars):
        """
        3d array of second derivatives of curve variables with respect to
        FX rates, of size (len(fx_vars), len(fx_vars), pre_n);

        .. math::

           [\\nabla_\\mathbf{f} \\nabla_\\mathbf{f} \\mathbf{v^T}]_{i,j,k} = \\frac{\\partial^2 v_k}{\\partial f_i \\partial f_j}

        Parameters
        ----------
        fx_vars : list or tuple of str
            The variable name tags for the FX rate sensitivities.
        """
        # FX sensitivity requires reverting through all pre-solvers rates.
        _ = -np.tensordot(self.grad_f_f_rT_pre(fx_vars), self.grad_s_vT_pre, (2, 0))
        _ -= np.tensordot(self.grad_f_rT_pre(fx_vars), self.grad_f_s_vT_pre(fx_vars), (1, 1))
        grad_f_f_vT = _
        return grad_f_f_vT

    def grad_f_vT_pre(self, fx_vars):
        """
        2d array of the derivatives of curve variables with respect to FX rates, of
        size (len(fx_vars), pre_n).

        .. math::

           [\\nabla_\\mathbf{f}\\mathbf{v^T}]_{i,j} = \\frac{\\partial v_j}{\\partial f_i} = -\\frac{\\partial r_z}{\\partial f_i} \\frac{\\partial v_j}{\\partial s_z}

        Parameters
        ----------
        fx_vars : list or tuple of str
            The variable name tags for the FX rate sensitivities
        """
        # FX sensitivity requires reverting through all pre-solvers rates.
        grad_f_rT = np.array([rate.gradient(fx_vars) for rate in self.r_pre]).T
        return -np.matmul(grad_f_rT, self.grad_s_vT_pre)

    def grad_f_f(self, f, fx_vars):
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
        grad_f_f = f.gradient(fx_vars)
        grad_f_f += np.matmul(self.grad_f_vT_pre(fx_vars), f.gradient(self.pre_variables))
        return grad_f_f

    @property
    def grad_s_vT_pre(self):
        """
        2d Jacobian array of curve variables with respect to calibrating instruments
        including all pre solvers attached to the Solver, of size (pre_m, pre_n).

        .. math::

           [\\nabla_\\mathbf{s}\\mathbf{v^T}]_{i,j} = \\frac{\\partial v_j}{\\partial s_i} = \\mathbf{J^+}
        """
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
                grad_v_r = np.array([r.gradient(pre_solver.pre_variables) for r in self.r]).T
                block = np.matmul(grad_v_r, self.grad_s_vT)
                block = -1 * np.matmul(pre_solver.grad_s_vT_pre, block)
                grad_s_vT[i : i + m, -self.m :] = block

                i, j = i + m, j + n

            # create bottom right block
            grad_s_vT[-self.m :, -self.m :] = self.grad_s_vT
            self._grad_s_vT_pre = grad_s_vT
        return self._grad_s_vT_pre

    def grad_s_f_pre(self, f):
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
        _ = np.tensordot(self.grad_s_vT_pre, f.gradient(self.pre_variables), (1, 0))
        grad_s_f = _
        return grad_s_f

    def grad_s_sT_f_pre(self, f):
        """
        2d array of derivatives of FX conversion rate with respect to
        calibrating instruments, of size (pre_m, pre_m);

        .. math::

           [\\nabla_\\mathbf{s} \\nabla_\\mathbf{s}^\\mathbf{T} f_{loc:bas}]_{i,j} = \\frac{\\partial^2 f}{\\partial s_i \\partial s_j}

        Parameters
        ----------
        f : Dual or Dual2
            The value of the local to base FX conversion rate.
        """
        grad_s_vT = self.grad_s_vT_pre
        grad_v_vT_f = f.gradient(self.pre_variables, order=2)

        _ = np.tensordot(grad_s_vT, grad_v_vT_f, (1, 0))
        _ = np.tensordot(_, grad_s_vT, (1, 1))

        grad_s_sT_f = _
        return grad_s_sT_f

    def grad_f_sT_f_pre(self, f, fx_vars):
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
        """
        grad_s_vT = self.grad_s_vT_pre
        grad_v_f = f.gradient(self.pre_variables)
        grad_f_sT_v = self.grad_f_s_vT_pre(fx_vars)
        _ = f.gradient(self.pre_variables + tuple(fx_vars), order=2)
        grad_v_vT_f = _[: self.pre_n, : self.pre_n]
        grad_f_vT_f = _[self.pre_n :, : self.pre_n]
        # grad_f_fT_f = _[self.pre_n :, self.pre_n :]
        grad_f_vT = self.grad_f_vT_pre(fx_vars)

        _ = np.tensordot(grad_f_sT_v, grad_v_f, (2, 0))
        _ += np.tensordot(grad_f_vT_f, grad_s_vT, (1, 1))

        __ = np.tensordot(grad_f_vT, grad_v_vT_f, (1, 0))
        __ = np.tensordot(__, grad_s_vT, (1, 1))

        grad_f_sT_f = _ + __
        return grad_f_sT_f

    def grad_f_fT_f_pre(self, f, fx_vars):
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
        """
        # grad_s_vT = self.grad_s_vT_pre
        grad_v_f = f.gradient(self.pre_variables)
        # grad_f_sT_v = self.grad_f_s_vT_pre(fx_vars)
        _ = f.gradient(self.pre_variables + tuple(fx_vars), order=2)
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

    def grad_s_Ploc(self, npv):
        """
        1d array of derivatives of local currency PV with respect to calibrating
        instruments, of size (pre_m).

        .. math::

           \\nabla_\\mathbf{s} P^{loc} = \\frac{\\partial P^{loc}}{\\partial s_i}

        Parameters:
            npv : Dual or Dual2
                A local currency NPV of a period of a leg.
        """
        grad_s_P = np.matmul(self.grad_s_vT_pre, npv.gradient(self.pre_variables))
        return grad_s_P

    def grad_f_Ploc(self, npv, fx_vars):
        """
        1d array of derivatives of local currency PV with respect to FX rate variable,
        of size (len(fx_vars)).

        .. math::

           \\nabla_\\mathbf{f} P^{loc}(\\mathbf{v(s, f), f}) = \\frac{\\partial P^{loc}}{\\partial f_i}+  \\frac{\partial v_z}{\\partial f_i} \\frac{\\partial P^{loc}}{\\partial v_z}

        Parameters:
            npv : Dual or Dual2
                A local currency NPV of a period of a leg.
            fx_vars : list or tuple of str
                The variable tags for automatic differentiation of FX rate sensitivity
        """
        grad_f_P = npv.gradient(fx_vars)
        grad_f_P += np.matmul(self.grad_f_vT_pre(fx_vars), npv.gradient(self.pre_variables))
        return grad_f_P

    def grad_s_Pbase(self, npv, grad_s_P, f):
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
        """
        grad_s_Pbas = float(npv) * np.matmul(self.grad_s_vT_pre, f.gradient(self.pre_variables))
        grad_s_Pbas += grad_s_P * float(f)  # <- use float to cast float array not Dual
        return grad_s_Pbas

    def grad_f_Pbase(self, npv, grad_f_P, f, fx_vars):
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
        """
        ret = grad_f_P * float(f)  # <- use float here to cast float array not Dual
        ret += float(npv) * self.grad_f_f(f, fx_vars)
        return ret

    def grad_s_sT_Ploc(self, npv):
        """
        2d array of derivatives of local currency PV with respect to calibrating
        instruments, of size (pre_m, pre_m).

        .. math::

           \\nabla_\\mathbf{s} \\nabla_\\mathbf{s}^\\mathbf{T} P^{loc}(\\mathbf{v, f}) = \\frac{ \\partial^2 P^{loc}(\\mathbf{v(s, f)}) }{\\partial s_i \\partial s_j}

        Parameters:
            npv : Dual2
                A local currency NPV of a period of a leg.
        """
        # instrument-instrument cross gamma:
        _ = np.tensordot(npv.gradient(self.pre_variables, order=2), self.grad_s_vT_pre, (1, 1))
        _ = np.tensordot(self.grad_s_vT_pre, _, (1, 0))

        _ += np.tensordot(self.grad_s_s_vT_pre, npv.gradient(self.pre_variables), (2, 0))
        grad_s_sT_P = _
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

    def gradp_f_vT_Ploc(self, npv, fx_vars):
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
        """
        grad_x_xT_Ploc = npv.gradient(self.pre_variables + tuple(fx_vars), order=2)
        grad_f_vT_Ploc = grad_x_xT_Ploc[self.pre_n :, : self.pre_n]
        return grad_f_vT_Ploc

    def grad_f_sT_Ploc(self, npv, fx_vars):
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
        """
        # fx_rate-instrument cross gamma:
        _ = np.tensordot(
            self.grad_f_vT_pre(fx_vars),
            npv.gradient(self.pre_variables, order=2),
            (1, 0),
        )
        _ += self.gradp_f_vT_Ploc(npv, fx_vars)
        _ = np.tensordot(_, self.grad_s_vT_pre, (1, 1))
        _ += np.tensordot(self.grad_f_s_vT_pre(fx_vars), npv.gradient(self.pre_variables), (2, 0))
        grad_f_sT_Ploc = _
        return grad_f_sT_Ploc

    def grad_f_fT_Ploc(self, npv, fx_vars):
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
        """
        # fx_rate-instrument cross gamma:
        gradp_f_vT_Ploc = self.gradp_f_vT_Ploc(npv, fx_vars)
        grad_f_vT_pre = self.grad_f_vT_pre(fx_vars)
        grad_v_Ploc = npv.gradient(self.pre_variables)
        grad_v_vT_Ploc = npv.gradient(self.pre_variables, order=2)

        _ = npv.gradient(fx_vars, order=2)
        _ += np.tensordot(self.grad_f_f_vT_pre(fx_vars), grad_v_Ploc, (2, 0))
        _ += np.tensordot(grad_f_vT_pre, gradp_f_vT_Ploc, (1, 1))
        _ += np.tensordot(gradp_f_vT_Ploc, grad_f_vT_pre, (1, 1))

        __ = np.tensordot(grad_f_vT_pre, grad_v_vT_Ploc, (1, 0))
        __ = np.tensordot(__, grad_f_vT_pre, (1, 1))

        grad_f_f_Ploc = _ + __
        return grad_f_f_Ploc

    def grad_s_sT_Pbase(self, npv, grad_s_sT_P, f):
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

        _ = float(f) * grad_s_sT_P
        _ += np.tensordot(grad_s_f[:, None], grad_s_P[None, :], (1, 0))
        _ += np.tensordot(grad_s_P[:, None], grad_s_f[None, :], (1, 0))
        _ += float(npv) * grad_s_sT_f  # <- use float to cast float array not Dual

        grad_s_sT_Pbas = _
        return grad_s_sT_Pbas

    def grad_f_sT_Pbase(self, npv, grad_f_sT_P, f, fx_vars):
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

        _ = float(f) * grad_f_sT_P
        _ += np.tensordot(grad_f_f[:, None], grad_s_P[None, :], (1, 0))
        _ += np.tensordot(grad_f_P[:, None], grad_s_f[None, :], (1, 0))
        _ += float(npv) * grad_f_sT_f  # <- use float to cast float array not Dual

        grad_s_sT_Pbas = _
        return grad_s_sT_Pbas

    def grad_f_fT_Pbase(self, npv, grad_f_fT_P, f, fx_vars):
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

        _ = float(f) * grad_f_fT_P
        _ += np.tensordot(grad_f_f[:, None], grad_f_P[None, :], (1, 0))
        _ += np.tensordot(grad_f_P[:, None], grad_f_f[None, :], (1, 0))
        _ += float(npv) * grad_f_fT_f  # <- use float to cast float array not Dual

        grad_s_sT_Pbas = _
        return grad_s_sT_Pbas


class Solver(Gradients):
    """
    A numerical solver to determine node values on multiple curves simultaneously.

    Parameters
    ----------
    curves : sequence
        Sequence of :class:`Curve` objects where each curve has been individually
        configured for its node dates and interpolation structures, and has a unique
        ``id``. Each :class:`Curve` will be dynamically updated by the Solver.
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
    fx : FXForwards, optional
        The ``FXForwards`` object used in FX rate calculations for ``instruments``.
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

    Notes
    -----
    Once initialised the ``Solver`` will numerically determine and set all of the
    relevant DF node values on each curve simultaneously by calling :meth:`iterate`.

    Each instrument provided to ``instruments`` must be a tuple with the following
    items:

      - An instrument which is a class endowed with a :meth:`rate` method which
        will return a mid-market rate as a :class:`Dual` number so derivatives can
        be automatically determined.
      - Positional arguments to be supplied to the :meth:`rate` method when determining
        the mid-market rate.
      - Keyword arguments to be supplied to the :meth:`rate` method when determining
        the mid-market rate.

    An example is `(IRS, (curve, disc_curve), {})`.

    Attributes
    ----------
    curves : dict
    instruments : sequence
    weights : sequence
    s : sequence
    algorithm : str
    fx : FXForwards
    id : str
    tol : float
    max_iter : int
    n : int
        The total number of curve variables to solve for.
    m : int
        The total number of calibrating instruments provided to the Solver.
    W : 2d array
        A diagonal array constructed from ``weights``.
    variables : list[str]
        List of variable name tags used in extracting derivatives automatically.
    instrument_labels : list[str]
        List of calibrating instrument names for delta risk visualization.
    pre_solvers : list
    pre_variables : list[str]
        List of variable name tags used in extracting derivatives automatically.
    pre_m : int
        The total number of calibrating instruments provided to the Solver including
        those in pre-solvers
    pre_n : int
        The total number of curve variables solved for, including those in pre-solvers.
    """

    _grad_s_vT_method = "_grad_s_vT_final_iteration_analytical"
    _grad_s_vT_final_iteration_algo = "gauss_newton_final"

    def __init__(
        self,
        curves: Union[list, tuple] = (),
        instruments: Union[tuple[tuple], list[tuple]] = (),
        s: list[float] = [],
        weights: Optional[list] = NoInput(0),
        algorithm: Optional[str] = NoInput(0),
        fx: Union[FXForwards, FXRates, NoInput] = NoInput(0),
        instrument_labels: Optional[tuple[str], list[str]] = NoInput(0),
        id: Optional[str] = NoInput(0),
        pre_solvers: Union[tuple[Solver], list[Solver]] = (),
        max_iter: int = 100,
        func_tol: float = 1e-11,
        conv_tol: float = 1e-14,
        ini_lambda: Union[tuple[float, float, float], NoInput] = NoInput(0)
    ) -> None:
        self.algorithm = defaults.algorithm if algorithm is NoInput.blank else algorithm
        if ini_lambda is NoInput.blank:
            self.ini_lambda = defaults.ini_lambda
        else:
            self.ini_lambda = ini_lambda
        self.m = len(instruments)
        self.func_tol, self.conv_tol, self.max_iter = func_tol, conv_tol, max_iter
        self.id = uuid4().hex[:5] + "_" if id is NoInput.blank else id # 1 in a million clash
        self.pre_solvers = tuple(pre_solvers)

        # validate `id`s so that DataFrame indexing does not share duplicated keys.
        if len(set([self.id] + [p.id for p in self.pre_solvers])) < 1 + len(self.pre_solvers):
            raise ValueError(
                "Solver `id`s must be unique when supplying `pre_solvers`, "
                f"got ids: {[self.id] + [p.id for p in self.pre_solvers]}"
            )

        # validate `s` and `instruments` with a naive length comparison
        if len(s) != len(instruments):
            raise ValueError("`instrument_rates` must be same length as `instruments`.")
        self.s = np.asarray(s)

        # validate `instrument_labels` if given is same length as `m`
        if instrument_labels is not NoInput.blank:
            if self.m != len(instrument_labels):
                raise ValueError("`instrument_labels` must have length `instruments`.")
            else:
                self.instrument_labels = tuple(instrument_labels)
        else:
            self.instrument_labels = tuple(f"{self.id}{i}" for i in range(self.m))

        if weights is NoInput.blank:
            self.weights = np.ones(len(instruments))
        else:
            if len(weights) != self.m:
                raise ValueError("`weights` must be same length as `instruments`.")
            self.weights = np.asarray(weights)
        self.W = np.diag(self.weights)

        self.curves = {
            curve.id: curve
            for curve in curves
            if not type(curve) in [ProxyCurve, CompositeCurve, MultiCsaCurve]
            # Proxy and Composite curves have no parameters of their own
        }
        self.variables = ()
        for curve in self.curves.values():
            curve._set_ad_order(1)  # solver uses gradients in optimisation
            curve_vars = tuple((f"{curve.id}{i}" for i in range(curve._ini_solve, curve.n)))
            self.variables += curve_vars
        self.n = len(self.variables)

        # aggregate and organise variables and labels including pre_solvers
        self.pre_curves = {}
        self.pre_variables = ()
        self.pre_instrument_labels = ()
        self.pre_instruments = ()
        self.pre_rate_scalars = []
        self.pre_m, self.pre_n = self.m, self.n
        curve_collection = []
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
            }
        )
        curve_collection.extend(curves)
        for curve1, curve2 in combinations(curve_collection, 2):
            if curve1.id == curve2.id:
                raise ValueError(
                    "`curves` must each have their own unique `id`. If using "
                    "pre-solvers as part of a dependency chain a curve can only be "
                    "specified as a variable in one solver."
                )
        self.pre_variables += self.variables
        self.pre_instrument_labels += tuple((self.id, lbl) for lbl in self.instrument_labels)

        # Final elements
        self._ad = 1
        self.fx = fx
        self.instruments = tuple((self._parse_instrument(inst) for inst in instruments))
        self.pre_instruments += self.instruments
        self.rate_scalars = tuple((inst[0]._rate_scalar for inst in self.instruments))
        self.pre_rate_scalars += self.rate_scalars

        # TODO need to check curves associated with fx object and set order.
        # self._reset_properties_()  performed in iterate
        self.iterate()

    def _parse_instrument(self, value):
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
            return (value, tuple(), {"solver": self, "fx": self.fx})
        else:
            # object is tuple
            if len(value) != 3:
                raise ValueError(
                    "`Instrument` supplied to `Solver` as tuple must be a 3-tuple of "
                    "signature: (Instrument, positional args[tuple], keyword "
                    "args[dict])."
                )
            ret0, ret1, ret2 = value[0], tuple(), {"solver": self, "fx": self.fx}
            if not (value[1] is None or value[1] == ()):
                ret1 = value[1]
            if not (value[2] is None or value[2] == {}):
                ret2 = {**ret2, **value[2]}
            return (ret0, ret1, ret2)

    def _reset_properties_(self, dual2_only=False):
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
            self._v = None  # depends on self.curves
            self._r = None  # depends on self.pre_curves and self.instruments
            self._r_pre = None  # depends on pre_solvers and self.r
            self._x = None  # depends on self.r, self.s
            self._g = None  # depends on self.x, self.weights
            self._J = None  # depends on self.r
            self._grad_s_vT = None  # final_iter_dual: depends on self.s and iteration
            # fixed_point_iter: depends on self.f
            # final_iter_anal: depends on self.J
            self._grad_s_vT_pre = None  # depends on self.grad_s_vT and pre_solvers.

        self._J2 = None  # defines its own self.r under dual2
        self._J2_pre = None  # depends on self.r and pre_solvers
        self._grad_s_s_vT = None  # final_iter: depends on self.J2 and self.grad_s_vT
        # finite_diff: TODO update comment
        self._grad_s_s_vT_pre = None  # final_iter: depends on pre versions of above
        # finite_diff: TODO update comment

        # self._grad_v_v_f = None
        # self._Jkm = None  # keep manifold originally used for exploring J2 calc method

    @property
    def v(self):
        """
        1d array of curve node variables for each ordered curve, size (n,).

        Depends on ``self.curves``.
        """
        if self._v is None:
            _ = []
            for id, curve in self.curves.items():
                _.extend([v for v in list(curve.nodes.values())[curve._ini_solve :]])
            self._v = np.array(_)
        return self._v

    @property
    def r(self):
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
    def r_pre(self):
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
    def x(self):
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
    def error(self):
        """
        Return the error in calibrating instruments, including ``pre_solvers``, scaled
        to the risk representation factor.

        Returns
        -------
        Series
        """
        s = None
        for pre_solver in self.pre_solvers:
            if s is None:
                s = pre_solver.error
            else:
                s = concat([pre_solver.error, s])

        _ = Series(
            self.x.astype(float) * 100 / self.rate_scalars,
            index=MultiIndex.from_tuples([(self.id, inst) for inst in self.instrument_labels]),
        )
        if s is None:
            s = _
        else:
            s = concat([s, _])
        return s

    @property
    def g(self):
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
    #     2d Jacobian array of rates with respect to discount factors, of size (n, m); :math:`[J]_{i,j} = \\frac{\\partial r_j}{\\partial v_i}`.
    #     """
    #     _Jkm = np.array([rate.gradient(self.variables + extra_vars, keep_manifold=True) for rate in self.r]).T
    #     return _Jkm

    def _update_step_(self, algorithm):
        if algorithm == "gradient_descent":
            grad_v_g = self.g.gradient(self.variables)
            y = np.matmul(self.J.transpose(), grad_v_g[:, np.newaxis])[:, 0]
            alpha = np.dot(y, self.weights * self.x) / np.dot(y, self.weights * y)
            v_1 = self.v - grad_v_g * alpha.real
        elif algorithm == "gauss_newton":
            if self.J.shape[0] == self.J.shape[1]:  # square system
                A = self.J.transpose()
                b = -np.array([x.real for x in self.x])[:, np.newaxis]
            else:
                A = np.matmul(self.J, np.matmul(self.W, self.J.transpose()))
                b = -0.5 * self.g.gradient(self.variables)[:, np.newaxis]
            delta = np.linalg.solve(A, b)[:, 0]
            v_1 = self.v + delta
        elif algorithm == "levenberg_marquardt":
            self.lambd *= self.ini_lambda[2] if self.g_prev < self.g.real else self.ini_lambda[1]
            A = np.matmul(self.J, np.matmul(self.W, self.J.transpose()))
            A += self.lambd * np.eye(self.n)
            b = -0.5 * self.g.gradient(self.variables)[:, np.newaxis]
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

            delta = dual_solve(A, b)[:, 0]
            v_1 = self.v + delta
        else:
            raise NotImplementedError(f"`algorithm`: {algorithm} (spelled correctly?)")
        return v_1

    def _update_fx(self):
        if self.fx is not NoInput.blank:
            self.fx.update()  # note: with no variables this does nothing.

    def iterate(self):
        """
        Solve the DF node values and update all the ``curves``.

        This method uses a gradient based optimisation routine, to solve for all
        the curve variables, :math:`\\mathbf{v}`, as follows,

        .. math::

           \\mathbf{v} = \\underset{\\mathbf{v}}{\\mathrm{argmin}} \;\; f(\\mathbf{v}) = \\underset{\\mathbf{v}}{\\mathrm{argmin}} \;\; (\\mathbf{r(v)} - \\mathbf{S})\\mathbf{W}(\\mathbf{r(v)} - \\mathbf{S})^\\mathbf{T}

        where :math:`\\mathbf{r}` are the mid-market rates of the calibrating
        instruments, :math:`\\mathbf{S}` are the observed and target rates, and
        :math:`\\mathbf{W}` is the diagonal array of weights.

        Returns
        -------
        None
        """
        DualType = Dual if self._ad == 1 else Dual2
        self.g_prev, self.g_list, self.lambd = 1e10, [], self.ini_lambda[0]
        self._reset_properties_()
        self._update_fx()
        t0 = time()
        for i in range(self.max_iter):
            g_val = self.g.real
            self.g_list.append(g_val)
            # condition is set to less than to avoid the case where a null update
            # results in the same solution and this is erroneously categorised
            # as a converged solution.
            if self.g.real < self.g_prev and (self.g_prev - self.g.real) < self.conv_tol:
                print(
                    f"SUCCESS: `conv_tol` reached after {i} iterations "
                    f"({self.algorithm}), `f_val`: {self.g.real}, "
                    f"`time`: {time() - t0:.4f}s"
                )
                return None
            elif self.g.real < self.func_tol:
                print(
                    f"SUCCESS: `func_tol` reached after {i} iterations "
                    f"({self.algorithm}) , `f_val`: {self.g.real}, "
                    f"`time`: {time() - t0:.4f}s"
                )
                return None
            self.g_prev = self.g.real
            v_1 = self._update_step_(self.algorithm)
            _ = 0
            for id, curve in self.curves.items():
                for k in curve.node_dates[curve._ini_solve :]:
                    curve.nodes[k] = DualType(v_1[_].real, curve.nodes[k].vars)
                    _ += 1
                curve.csolve()
            self._reset_properties_()
            self._update_fx()
        print(
            f"FAILURE: `max_iter` of {self.max_iter} iterations breached, "
            f"`f_val`: {self.g.real}, `time`: {time() - t0:.4f}s"
        )
        return None
        # raise ValueError(f"Max iterations reached, func: {self.f.real}")

    def _set_ad_order(self, order):
        """Defines the node DF in terms of float, Dual or Dual2 for AD order calcs."""
        for pre_solver in self.pre_solvers:
            pre_solver._set_ad_order(order=order)
        self._ad = order
        self._reset_properties_()
        for _, curve in self.curves.items():
            curve._set_ad_order(order)
        if self.fx is not NoInput.blank:
            self.fx._set_ad_order(order)

    # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
    # Commercial use of this code, and/or copying and redistribution is prohibited.
    # Contact rateslib at gmail.com if this code is observed outside its intended sphere.

    def delta(self, npv, base: Union[str, NoInput] = NoInput(0), fx=None):
        """
        Calculate the delta risk sensitivity of an instrument's NPV to the
        calibrating instruments of the :class:`~rateslib.solver.Solver`, and to
        FX rates.

        Parameters
        ----------
        npv : Dual,
            The NPV of the instrument or composition of instruments to risk.
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
        base, fx = self._get_base_and_fx(base, fx)
        fx_vars = tuple() if fx is NoInput.blank else fx.variables

        inst_scalar = np.array(self.pre_rate_scalars) / 100  # instruments scalar
        fx_scalar = 0.0001
        container = {}
        for ccy in npv:
            container[("instruments", ccy, ccy)] = self.grad_s_Ploc(npv[ccy]) * inst_scalar
            container[("fx", ccy, ccy)] = self.grad_f_Ploc(npv[ccy], fx_vars) * fx_scalar

            if base is not NoInput.blank and base != ccy:
                # extend the derivatives
                f = fx.rate(f"{ccy}{base}")
                container[("instruments", ccy, base)] = (
                    self.grad_s_Pbase(
                        npv[ccy], container[("instruments", ccy, ccy)] / inst_scalar, f
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
            [("fx", "fx", f[3:]) for f in fx_vars], names=["type", "solver", "label"]
        )
        indexes = {"instruments": inst_idx, "fx": fx_idx}
        r_idx = inst_idx.append(fx_idx)
        c_idx = MultiIndex.from_tuples([], names=["local_ccy", "display_ccy"])
        df = DataFrame(None, index=r_idx, columns=c_idx)
        for key, array in container.items():
            df.loc[indexes[key[0]], (key[1], key[2])] = array

        if base is not NoInput.blank:
            df.loc[r_idx, ("all", base)] = df.loc[r_idx, (slice(None), base)].sum(axis=1)

        sorted_cols = df.columns.sort_values()
        return df.loc[:, sorted_cols].astype("float64")

    def _get_base_and_fx(
        self, base: Union[str, NoInput], fx: Union[FXForwards, FXRates, float, NoInput]
    ):
        if base is not NoInput.blank and self.fx is NoInput.blank and fx is NoInput.blank:
            raise ValueError(
                "`base` is given but `fx` is not and Solver does not "
                "contain an attached FXForwards object."
            )
        elif fx is NoInput.blank:
            fx = self.fx
        elif fx is not NoInput.blank and self.fx is not NoInput.blank:
            if id(fx) != id(self.fx):
                warnings.warn(
                    "Solver contains an `fx` attribute but an `fx` argument has been "
                    "supplied which is not the same. This can lead to risk sensitivity "
                    "inconsistencies, mathematically.",
                    UserWarning,
                )
        if base is not NoInput.blank:
            base = base.lower()
        return base, fx

    def gamma(self, npv, base=None, fx=None):
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
            :class:`~rateslib.solver.Solver`.

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
        """
        if self._ad != 2:
            raise ValueError("`Solver` must be in ad order 2 to use `gamma` method.")

        # new
        base, fx = self._get_base_and_fx(base, fx)
        fx_vars = tuple() if fx is NoInput.blank else fx.variables

        inst_scalar = np.array(self.pre_rate_scalars) / 100  # instruments scalar
        fx_scalar = np.ones(len(fx_vars)) * 0.0001
        container = {}
        for ccy in npv:
            container[(ccy, ccy)] = {}
            container[(ccy, ccy)]["instruments", "instruments"] = self.grad_s_sT_Ploc(
                npv[ccy]
            ) * np.matmul(inst_scalar[:, None], inst_scalar[None, :])
            container[(ccy, ccy)]["fx", "instruments"] = self.grad_f_sT_Ploc(
                npv[ccy], fx_vars
            ) * np.matmul(fx_scalar[:, None], inst_scalar[None, :])
            container[(ccy, ccy)]["instruments", "fx"] = container[(ccy, ccy)][
                ("fx", "instruments")
            ].T
            container[(ccy, ccy)]["fx", "fx"] = self.grad_f_fT_Ploc(npv[ccy], fx_vars) * np.matmul(
                fx_scalar[:, None], fx_scalar[None, :]
            )

            if base is not NoInput.blank and base != ccy:
                # extend the derivatives
                f = fx.rate(f"{ccy}{base}")
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
        all_keys = sorted(list(set(local_keys + base_keys)))
        inst_keys = [("instruments",) + label for label in self.pre_instrument_labels]
        fx_keys = [("fx", "fx", f[3:]) for f in fx_vars]
        idx_tuples = [c + _ for c in all_keys for _ in inst_keys + fx_keys]
        ridx = MultiIndex.from_tuples(
            [key for key in idx_tuples],
            names=["local_ccy", "display_ccy", "type", "solver", "label"],
        )
        if base is not NoInput.blank:
            ridx = ridx.append(
                MultiIndex.from_tuples(
                    [("all", base) + _ for _ in inst_keys + fx_keys],
                    names=["local_ccy", "display_ccy", "type", "solver", "label"],
                )
            )
        cidx = MultiIndex.from_tuples(
            [_ for _ in inst_keys + fx_keys], names=["type", "solver", "label"]
        )
        df = DataFrame(None, index=ridx, columns=cidx)
        for key, d in container.items():
            array = np.block(
                [
                    [d[("instruments", "instruments")], d[("instruments", "fx")]],
                    [d[("fx", "instruments")], d[("fx", "fx")]],
                ]
            )
            locator = key + (slice(None), slice(None), slice(None))
            df.loc[locator, :] = array

        if base is not NoInput.blank:
            # sum over all the base rows to aggregate
            gdf = (
                df.loc[(currencies, base, slice(None), slice(None), slice(None)), :]
                .groupby(level=[2, 3, 4])
                .sum()
            )
            gdf.index = MultiIndex.from_tuples([("all", base) + _ for _ in gdf.index])
            df.loc[("all", base, slice(None), slice(None), slice(None))] = gdf

        return df.astype("float64")

    def _pnl_explain(self, npv, ds, dfx=None, base=None, fx=None, order=1):
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

    def market_movements(self, solver: Solver):
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
           demonstrated HERE XXX). This allows two different
           *Solvers* to contain their own *Curves* (which may or may not be equivalent models),
           and for the instrument rates of one *Solver* to be evaluated by the *Curves* present
           in another *Solver*.
        """
        r_0 = self.r_pre
        r_1 = np.array(
            [_[0].rate(*_[1], **{**_[2], **{"solver": solver, "fx": solver.fx}}) for _ in
             self.pre_instruments]
        )
        return DataFrame(
            (r_1 - r_0) * 100 / np.array(self.pre_rate_scalars),
            index=self.pre_instrument_labels
        )

    def jacobian(self, solver: Solver):
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
           demonstrated HERE XXX). This allows two different
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
                _[0].rate(*_[1], **{**_[2], **{"solver": solver, "fx": solver.fx}})
                for _ in self.pre_instruments
            ]
        )
        # Get the gradient of these rates with respect to the variable in other
        grad_v_rT = np.array([_.gradient(solver.pre_variables) for _ in r]).T
        return DataFrame(
            np.matmul(solver.grad_s_vT_pre, grad_v_rT),
            columns=self.pre_instrument_labels,
            index=solver.pre_instrument_labels,
        )

# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.
