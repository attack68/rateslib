from __future__ import annotations

from typing import Optional, Union
from itertools import combinations
from uuid import uuid4
import numpy as np
from pandas import DataFrame, MultiIndex

from rateslib.dual import Dual, Dual2, dual_log, dual_solve
from rateslib.fx import FXForwards

# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


# TODO: validate solver_id are unique in pre_solver_chain
class Solver:
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
    algorithm : str in {"gradient_descent", "gauss_newton", "levenberg_marquardt"}
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
        curves: list = [],
        instruments: list[tuple] = [],
        s: list[float] = [],
        weights: Optional[list] = None,
        algorithm: str = "gauss_newton",
        fx: Optional[FXForwards] = None,
        instrument_labels: Optional[list[str]] = None,
        id: Optional[str] = None,
        pre_solvers: list[Solver] = [],
        max_iter: int = 100,
        func_tol: float = 1e-10,
        conv_tol: float = 1e-13,
    ) -> None:
        self.algorithm, self.m = algorithm, len(instruments)
        self.func_tol, self.conv_tol, self.max_iter = func_tol, conv_tol, max_iter
        self.id = id or uuid4().hex[:5] + "_"  # 1 in a million clash
        self.pre_solvers = pre_solvers
        if len(s) != len(instruments):
            raise ValueError("`instrument_rates` must be same length as `instruments`.")
        self.s = np.asarray(s)
        if instrument_labels is not None:
            if self.m != len(instrument_labels):
                raise ValueError("`instrument_labels` must have length `instruments`.")
            else:
                self.instrument_labels = instrument_labels
        else:
            self.instrument_labels = [f"{self.id}{i}" for i in range(self.m)]

        if weights is None:
            self.weights = np.ones(len(instruments))
        else:
            if len(weights) != self.m:
                raise ValueError("`weights` must be same length as `instruments`.")
            self.weights = np.asarray(weights)
        self.W = np.diag(self.weights)

        self.curves = {curve.id: curve for curve in curves}
        self.variables = []
        for curve in self.curves.values():
            curve._set_ad_order(1)  # solver uses gradients in optimisation
            curve_vars = [f"{curve.id}{i}" for i in range(curve._ini_solve, curve.n)]
            self.variables.extend(curve_vars)
        self.n = len(self.variables)

        # aggregate and organise variables and labels including pre_solvers
        self.pre_curves = {}
        self.pre_variables = []
        self.pre_instrument_labels = []
        self.pre_m, self.pre_n = self.m, self.n
        curve_collection = []
        for pre_solver in self.pre_solvers:
            self.pre_variables.extend(pre_solver.pre_variables)
            self.pre_instrument_labels.extend(pre_solver.pre_instrument_labels)
            self.pre_m += pre_solver.pre_m
            self.pre_n += pre_solver.pre_n
            self.pre_curves.update(pre_solver.pre_curves)
            curve_collection.extend(pre_solver.pre_curves.values())
        self.pre_curves.update(self.curves)
        curve_collection.extend(curves)
        for curve1, curve2 in combinations(curve_collection, 2):
            if curve1.id == curve2.id:
                raise ValueError(
                    "`curves` must each have their own unique `id`. If using "
                    "pre-solvers as part of a dependency chain a curve can only be "
                    "specified as a variable in one solver."
                )
        self.pre_variables.extend(self.variables)
        self.pre_instrument_labels.extend([(self.id, lbl) for lbl in self.instrument_labels])

        # Final elements
        self._ad = 1
        self.fx = fx
        self.instruments = [self._parse_instrument(inst) for inst in instruments]
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
            if not(value[1] is None or value[1] == ()):
                ret1 = value[1]
            if not(value[2] is None or value[2] == {}):
                ret2 = {**ret2, **value[2]}
            return (ret0, ret1, ret2)


    # convert to tuple


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
                :math:`\frac{\partial^2 r_i}{\partial v_j \partial v_k}`
              - ``grad_s_s_vT`` and ``grad_s_s_vT_pre``:
                :math:`\frac{\partial^2 v_i}{\partial s_j \partial s_k}`

        Returns
        -------
        None
        """
        if not dual2_only:
            self._v = None  # depends on self.curves
            self._r = None  # depends on self.pre_curves and self.instruments
            self._x = None  # depends on self.r, self.s
            self._f = None  # depends on self.x, self.weights
            self._J = None  # depends on self.r
            self._grad_s_vT = None  # final_iter_dual: depends on self.s and iteration
                                    # fixed_point_iter: depends on self.f
                                    # final_iter_anal: depends on self.J
            self._grad_s_vT_pre = None  # depends on self.grad_s_vT and pre_solvers.

        self._J2 = None      # defines its own self.r under dual2
        self._J2_pre = None  # depends on self.r and pre_solvers
        self._grad_s_s_vT = None  # final_iter: depends on self.J2 and self.grad_s_vT
                                  # finite_diff: TODO update comment
        self._grad_s_s_vT_pre = None  # final_iter: depends pre versions of above
                                      # finite_diff: TODO update comment

        # self._grad_v_v_f = None
        # self._Jkm = None  # keep manifold originally used for exploring J2 calc method

    @property
    def v(self):
        """
        1d array of DF solver variables for each ordered curve, size (n,).

        Depends on ``self.curves``.
        """
        if self._v is None:
            _ = []
            for id, curve in self.curves.items():
                _.extend([v for v in list(curve.nodes.values())[curve._ini_solve:]])
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
            self._r = np.array([
                _[0].rate(*_[1], **_[2]) for _ in self.instruments
                # solver is passed in order to extract curves as string
            ])
        return self._r

    @property
    def x(self):
        """
        1d array of error in each calibrating instrument rate, of size (m,).

        Depends on ``self.r`` and ``self.s``.
        """
        if self._x is None:
            self._x = self.r - self.s
        return self._x

    @property
    def f(self):
        """
        Objective function scalar value of the solver;

        .. math::

           f = \\mathbf{(r-S)^{T}W(r-S)}

        Depends on ``self.x`` and ``self.weights``.
        """
        if self._f is None:
            self._f = np.dot(self.x, self.weights * self.x)
        return self._f

    @property
    def J(self):
        """
        2d Jacobian array of rates with respect to discount factors, of size (n, m);

        .. math::

           [J]_{i,j} = [\\nabla_\mathbf{v} \mathbf{r^T}]_{i,j} = \\frac{\\partial r_j}{\\partial v_i}

        Depends on ``self.r``.
        """
        if self._J is None:
            self._J = np.array([rate.gradient(self.variables) for rate in self.r]).T
        return self._J

    @property
    def J2(self):
        """
        3d array of second differentials of rates with respect to discount factors,
        of size (n, n, m);

        .. math::

           [J2]_{i,j,k} = [\\nabla_\mathbf{v} \\nabla_\mathbf{v} \mathbf{r^T}]_{i,j,k} = \\frac{\\partial^2 r_k}{\\partial v_i \\partial v_j}

        Depends on ``self.r``.
        """
        if self._J2 is None:
            if self._ad != 2:
                raise ValueError(
                    "Cannot perform second derivative calculations when ad mode is "
                    "{self._ad}."
                )

            rates = np.array([_[0].rate(*_[1], **_[2]) for _ in self.instruments])
            # solver is passed in order to extract curves as string
            _ = np.array([rate.gradient(self.variables, order=2) for rate in rates])
            self._J2 = np.transpose(_, (1, 2, 0))
        return self._J2

    @property
    def J2_pre(self):
        """
        3d array of second differentials of rates with respect to discount factors,
        of size (n, n, m);

        .. math::

           [J2]_{i,j,k} = [\\nabla_\mathbf{v} \\nabla_\mathbf{v} \mathbf{r^T}]_{i,j,k} = \\frac{\\partial^2 r_k}{\\partial v_i \\partial v_j}

        Depends on ``self.r`` and ``pre_solvers.J2``.
        """
        if len(self.pre_solvers) == 0:
            return self.J2

        if self._J2_pre is None:
            if self._ad != 2:
                raise ValueError(
                    "Cannot perform second derivative calculations when ad mode is "
                    "{self._ad}."
                )

            J2 = np.zeros(shape=(self.pre_n, self.pre_n, self.pre_m))
            i, j = 0, 0
            for pre_slvr in self.pre_solvers:
                J2[i:pre_slvr.pre_n, i:pre_slvr.pre_n, j:pre_slvr.pre_m] = pre_slvr.J2_pre
                i, j = i + pre_slvr.pre_n, j + pre_slvr.pre_m

            rates = np.array([_[0].rate(*_[1], **_[2]) for _ in self.instruments])
            # solver is passed in order to extract curves as string
            _ = np.array([r.gradient(self.pre_variables, order=2) for r in rates])
            J2[:, :, -self.m:] = np.transpose(_, (1, 2, 0))
            self._J2_pre = J2
        return self._J2_pre

    # def Jkm(self, extra_vars=[]):
    #     """
    #     2d Jacobian array of rates with respect to discount factors, of size (n, m); :math:`[J]_{i,j} = \\frac{\\partial r_j}{\\partial v_i}`.
    #     """
    #     _Jkm = np.array([rate.gradient(self.variables + extra_vars, keep_manifold=True) for rate in self.r]).T
    #     return _Jkm

    def _update_step_(self, algorithm):
        if algorithm == "gradient_descent":
            grad_v_f = self.f.gradient(self.variables)
            y = np.matmul(self.J.transpose(), grad_v_f[:, np.newaxis])[:, 0]
            alpha = np.dot(y, self.weights * self.x) / np.dot(y, self.weights * y)
            v_1 = self.v - grad_v_f * alpha.real
        elif algorithm == "gauss_newton":
            if self.J.shape[0] == self.J.shape[1]:  # square system
                A = self.J.transpose()
                b = -np.array([x.real for x in self.x])[:, np.newaxis]
            else:
                A = np.matmul(self.J, np.matmul(self.W, self.J.transpose()))
                b = -0.5 * self.f.gradient(self.variables)[:, np.newaxis]
            delta = np.linalg.solve(A, b)[:, 0]
            v_1 = self.v + delta
        elif algorithm == "levenberg_marquardt":
            self.lambd *= 2 if self.f_prev < self.f.real else 0.25
            A = np.matmul(self.J, np.matmul(self.W, self.J.transpose()))
            A += self.lambd * np.eye(self.n)
            b = -0.5 * self.f.gradient(self.variables)[:, np.newaxis]
            delta = np.linalg.solve(A, b)[:, 0]
            v_1 = self.v + delta
        # elif algorithm == "gradient_descent_final":
        #     _ = np.matmul(self.Jkm, np.matmul(self.W, self.x[:, np.newaxis]))
        #     y = 2 * np.matmul(self.Jkm.transpose(), _)[:, 0]
        #     alpha = np.dot(y, self.weights * self.x) / np.dot(y, self.weights * y)
        #     v_1 = self.v - 2 * alpha * _[:, 0]
        elif algorithm == "gauss_newton_final":
            # TODO deal with square and not square matrices here.
            A = self.J.transpose()
            b = -self.x[:, np.newaxis]
            delta = dual_solve(A, b)[:, 0]
            v_1 = self.v + delta
        else:
            raise NotImplementedError(f"`algorithm`: {algorithm} (spelled correctly?)")
        return v_1

    def _update_fx(self):
        if self.fx is not None:
            self.fx.update() # note: with no variables this does nothing.

    def iterate(self):
        """
        Solve the DF node values and update all the ``curves``.

        This method uses a gradient based optimisation routine, to solve for all
        the curve variables, :math:`\mathbf{v}`, as follows,

        .. math::

           \mathbf{v} = \\underset{\mathbf{v}}{\mathrm{argmin}} \;\; f(\mathbf{v}) = \\underset{\mathbf{v}}{\mathrm{argmin}} \;\; (\mathbf{r(v)} - \mathbf{S})\mathbf{W}(\mathbf{r(v)} - \mathbf{S})^\mathbf{T}

        where :math:`\mathbf{r}` are the mid-market rates of the calibrating
        instruments, :math:`\mathbf{S}` are the observed and target rates, and
        :math:`\mathbf{W}` is the diagonal array of weights.

        Returns
        -------
        None
        """
        DualType = Dual if self._ad == 1 else Dual2
        self.f_prev, self.f_list, self.lambd = 1e10, [], 1000
        self._reset_properties_()
        self._update_fx()
        for i in range(self.max_iter):
            f_val = self.f.real
            self.f_list.append(f_val)
            # TODO check whether less than or equal to is correct in below condition
            if self.f.real < self.f_prev and (self.f_prev - self.f.real) < self.conv_tol:
                print(
                    f"SUCCESS: `conv_tol` reached after {i} iterations "
                    f"({self.algorithm}), `f_val`: {self.f.real}"
                )
                return None
            elif self.f.real < self.func_tol:
                print(
                    f"SUCCESS: `func_tol` reached after {i} iterations "
                    f"({self.algorithm}) , `f_val`: {self.f.real}"
                )
                return None
            self.f_prev = self.f.real
            v_1 = self._update_step_(self.algorithm)
            _ = 0
            for id, curve in self.curves.items():
                for k in curve.node_dates[curve._ini_solve:]:
                    curve.nodes[k] = DualType(v_1[_].real, curve.nodes[k].vars)
                    _ += 1
                curve.csolve()
            self._reset_properties_()
            self._update_fx()
        print(
            f"FAILURE: `max_iter` of {self.max_iter} iterations breached, "
            f"`f_val`: {self.f.real}"
        )
        return None
        # raise ValueError(f"Max iterations reached, func: {self.f.real}")

    def _set_ad_order(self, order):
        """Defines the node DF in terms of float, Dual or Dual2 for AD order calcs."""
        for pre_solver in self.pre_solvers:
            pre_solver._set_ad_order(order=order)
        self._ad = order
        for _, curve in self.curves.items():
            curve._set_ad_order(order)
        if self.fx is not None:
            self.fx._set_ad_order(order)

# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.

    @property
    def grad_s_vT(self):
        """
        2d Jacobian array of DFs with respect to calibrating instruments,
        of size (m, n);

        .. math::

           [\\nabla_\mathbf{s}\mathbf{v^T}]_{i,j} = \\frac{\\partial v_j}{\\partial s_i}
        """
        if self._grad_s_vT is None:
            self._grad_s_vT = getattr(self, self._grad_s_vT_method)()
        return self._grad_s_vT

    @property
    def grad_s_vT_pre(self):
        """
        2d Jacobian array of DFs with respect to calibrating instruments including all
        pre solvers attached to the Solver.
        """
        if len(self.pre_solvers) == 0:
            return self.grad_s_vT

        if self._grad_s_vT_pre is None:
            grad_s_vT = np.zeros(shape=(self.pre_m, self.pre_n))

            i, j = 0, 0
            for pre_solver in self.pre_solvers:
                # create the left side block matrix
                m, n = pre_solver.pre_m, pre_solver.pre_n
                grad_s_vT[i:m, j:n] = pre_solver.grad_s_vT_pre

                # create the right column dependencies
                grad_v_r = np.array(
                   [r.gradient(pre_solver.pre_variables) for r in self.r]
                ).T
                block = np.matmul(grad_v_r, self.grad_s_vT)
                block = -1 * np.matmul(pre_solver.grad_s_vT_pre, block)
                grad_s_vT[i:m, -self.m:] = block

                i, j = i + m, j + n

            # create bottom right block
            grad_s_vT[-self.m:, -self.m:] = self.grad_s_vT
            self._grad_s_vT_pre = grad_s_vT
        return self._grad_s_vT_pre

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
        s_vars = [f"s{i}" for i in range(self.m)]
        grad2 = self.f.gradient(self.variables + s_vars, order=2)
        grad_v_vT_f = grad2[:self.n, :self.n]
        grad_s_vT_f = grad2[self.n:, :self.n]
        grad_s_vT = np.linalg.solve(grad_v_vT_f, -grad_s_vT_f.T).T

        self.s = _s
        self._set_ad_order(1)
        self._reset_properties_()
        return grad_s_vT

    @property
    def grad_s_s_vT(self):
        """
        3d array of second differentials of DFs with respect to calibrating instruments,
        of size (m, m, n);

        .. math::

           [\\nabla_\mathbf{s} \\nabla_\mathbf{s} \mathbf{v^T}]_{i,j,k} = \\frac{\\partial^2 v_k}{\\partial s_i \\partial s_j}
        """
        if self._grad_s_s_vT is None:
            self._grad_s_s_vT = self._grad_s_s_vT_final_iteration_analytical()
        return self._grad_s_s_vT

    @property
    def grad_s_s_vT_pre(self):
        """
        3d array of second differentials of DFs with respect to calibrating instruments,
        of size (m, m, n);

        .. math::

           [\\nabla_\mathbf{s} \\nabla_\mathbf{s} \mathbf{v^T}]_{i,j,k} = \\frac{\\partial^2 v_k}{\\partial s_i \\partial s_j}
        """
        if len(self.pre_solvers) == 0:
            return self.grad_s_s_vT

        if self._grad_s_s_vT_pre is None:
            self._grad_s_s_vT_pre = self._grad_s_s_vT_final_iteration_analytical(
                use_pre=True
            )
        return self._grad_s_s_vT_pre

    def _grad_s_s_vT_fwd_difference_method(self):
        """Use a numerical method, iterating through changes in s to calculate."""
        ds = 10 ** (int(dual_log(self.tol, 10) / 2))
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

        _ = np.matmul(grad_s_vT, np.matmul(J2, grad_s_vT))
        grad_s_s_vT = -np.tensordot(grad_s_vT, _, (1, 0))
        return grad_s_s_vT

    # grad_v_v_f: calculated within grad_s_vT_fixed_point_iteration

    def delta(self, npv):
        """
        Calculate the delta risk sensitivity of an instrument's NPV.

        Parameters
        ----------
        npv : Dual,
            The NPV of the instrument or composition of instruments to risk.

        Returns
        -------
        DataFrame
        """
        # if no pre_solvers this reduces to solving without the 'pre'
        grad_s_P = np.matmul(self.grad_s_vT_pre, npv.gradient(self.pre_variables))

        if len(self.pre_solvers) == 0:
            idx = self.instrument_labels
        else:
            idx = MultiIndex.from_tuples(self.pre_instrument_labels)

        return DataFrame({"Delta": grad_s_P / 100}, index=idx)

    def gamma(self, npv):
        if self._ad != 2:
            raise ValueError( "`Solver` must be in ad order 2 to use `gamma` method.")

        grad_s_sT_P = np.matmul(
            self.grad_s_vT_pre, np.matmul(
                npv.gradient(self.pre_variables, order=2), self.grad_s_vT_pre.T
            )
        )
        grad_s_sT_P += np.matmul(
            self.grad_s_s_vT_pre, npv.gradient(self.pre_variables)[:, None]
        )[:, :, 0]

        if len(self.pre_solvers) == 0:
            idx = self.instrument_labels
        else:
            idx = MultiIndex.from_tuples(self.pre_instrument_labels)

        return DataFrame(grad_s_sT_P / 10000, columns=idx, index=idx)

    def jacobian(self, solver: Solver):
        """
        Calculate the Jacobian with respect to another ``Solver`` instruments.

        Parameters
        ----------
        solver : Solver
            The other ``Solver`` for which the Jacobian is to be determined.

        Returns
        -------
        DataFrame
        """
        self.s = np.array([
            _[0].rate(*_[1], **{**_[2], "solver": solver}) for _ in self.instruments
        ])
        self._reset_properties_()
        self.iterate()
        rates = np.array([
            _[0].rate(*_[1], **{"solver": self, **_[2]}) for _ in solver.instruments
        ])
        grad_v_rT = np.array([
            _.gradient(self.variables) for _ in rates
        ]).T
        return DataFrame(
            np.matmul(self.grad_s_vT, grad_v_rT),
            index=self.instrument_labels,
            columns=solver.instrument_labels,
        )

# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.
