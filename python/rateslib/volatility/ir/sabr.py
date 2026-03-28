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


from __future__ import annotations  # type hinting

from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np
from pandas import DataFrame, Index

from rateslib.data.fixings import IRSSeries, _get_irs_series
from rateslib.dual import Dual, Dual2, Variable, set_order_convert
from rateslib.dual.utils import _dual_float, _to_number, dual_exp, dual_inv_norm_cdf
from rateslib.enums.generics import NoInput, _drb
from rateslib.enums.parameters import OptionPricingModel
from rateslib.mutability import (
    _new_state_post,
)
from rateslib.volatility.ir.base import _BaseIRCube, _BaseIRSmile, _WithMutability
from rateslib.volatility.ir.utils import (
    _IRCubeMeta,
    _IRSmileMeta,
    _IRVolPricingParams,
)
from rateslib.volatility.utils import _SabrModel, _SabrSmileNodes

UTC = timezone.utc

if TYPE_CHECKING:
    from rateslib.local_types import (  # pragma: no cover
        Arr1dObj,
        Arr2dObj,
        Arr3dObj,
        DualTypes,
        DualTypes_,
        Iterable,
        Number,
        Series,
        float_,
    )


class IRSabrSmile(_BaseIRSmile, _WithMutability):
    r"""
    Create an *IR Volatility Smile* at a given expiry indexed for a specific IRS tenor
    using SABR parameters.

    .. warning::

       *Swaptions* and *IR Volatility* are in Beta status introduced in v2.7.0

    .. role:: green

    .. role:: red

    Parameters
    ----------
    nodes: dict[str, float], :red:`required`
        The parameters for the SABR model. Keys must be *'alpha', 'rho', 'nu'*. See below.
    beta: float, Variable, :red:`required`
        The SABR beta parameter assumed by this *Smile*.
    eval_date: datetime, :red:`required`
        Acts like the initial node of a *Curve*. Should be assigned today's immediate date.
    expiry: datetime, :red:`required`
        The expiry date of the options associated with this *Smile*.
    irs_series: IRSSeries, :red:`required`
        The :class:`~rateslib.data.fixings.IRSSeries` that contains the parameters for the
        underlying :class:`~rateslib.instruments.IRS` that the swaptions are settled against.
    tenor: datetime, str, :red:`required`
        The tenor parameter for the underlying :class:`~rateslib.instruments.IRS` that the
        swaptions are settled against.
    shift: float, Variable, :green:`optional (set as zero)`
        The number of basis points to apply to the strike and forward under a 'Black Shifted
        Volatility' model.
    id: str, optional, :green:`optional (set as random)`
        The unique identifier to distinguish between *Smiles* in a multicurrency framework
        and/or *Surface*.
    ad: int, :green:`optional (set by default)`
        Sets the automatic differentiation order. Defines whether to convert node
        values to float, :class:`~rateslib.dual.Dual` or
        :class:`~rateslib.dual.Dual2`. It is advised against
        using this setting directly. It is mainly used internally.

    Notes
    -----
    The keys for ``nodes`` are described as the following:

    - ``alpha``: The initial volatility parameter (e.g. 0.10 for 10%) of the SABR model,
      in (0, inf).
    - ``rho``: The correlation between spot and volatility of the SABR model,
      e.g. -0.10, in [-1.0, 1.0)
    - ``nu``: The volatility of volatility parameter of the SABR model, e.g. 0.80.

    The parameters :math:`\alpha, \rho, \nu` will be calibrated/mutated by
    a :class:`~rateslib.solver.Solver` object. These should be entered as *float* and the argument
    ``ad`` can be used to automatically tag these as variables.

    The parameter :math:`\beta` will **not** be calibrated/mutated by a
    :class:`~rateslib.solver.Solver`. This value can be entered either as a *float*, or a
    :class:`~rateslib.dual.Variable` to capture exogenous sensitivities.

    Examples
    --------
    See :ref:`Constructing a Smile <c-fx-smile-constructing-doc>`.

    """

    @_new_state_post
    def __init__(
        self,
        nodes: dict[str, DualTypes],
        beta: float | Variable,
        eval_date: datetime,
        expiry: datetime | str,
        irs_series: IRSSeries | str,
        tenor: datetime | str,
        *,
        shift: DualTypes_ = NoInput(0),
        id: str | NoInput = NoInput(0),  # noqa: A002
        ad: int | None = 0,
    ):
        self._id: str = (
            uuid4().hex[:5] + "_" if isinstance(id, NoInput) else id
        )  # 1 in a million clash
        self._meta: _IRSmileMeta = _IRSmileMeta(
            _tenor_input=tenor,
            _irs_series=_get_irs_series(irs_series),
            _eval_date=eval_date,
            _expiry_input=expiry,
            _plot_x_axis="strike",
            _plot_y_axis="black_vol",
            _shift=_drb(0.0, shift),
            _pricing_model=OptionPricingModel.Black76,
        )

        try:
            self._nodes: _SabrSmileNodes = _SabrSmileNodes(
                _alpha=_to_number(nodes["alpha"]),
                _beta=beta,
                _rho=_to_number(nodes["rho"]),
                _nu=_to_number(nodes["nu"]),
            )
        except KeyError as e:
            for _ in ["alpha", "rho", "nu"]:
                if _ not in nodes:
                    raise ValueError(
                        f"'{_}' is a required SABR parameter that must be included in ``nodes``"
                    )
            raise e  # pragma: no cover

        self._set_ad_order(ad)

    ### Object unique elements

    @property
    def _n(self) -> int:
        return self.nodes.n

    @property
    def _ini_solve(self) -> int:
        return 1

    @property
    def id(self) -> str:
        """A str identifier to name the *Smile* used in
        :class:`~rateslib.solver.Solver` mappings."""
        return self._id

    @property
    def nodes(self) -> _SabrSmileNodes:
        """An instance of :class:`~rateslib.volatility.utils._SabrSmileNodes`."""
        return self._nodes

    def _d_sabr_d_k_or_f(
        self,
        k: DualTypes,
        f: DualTypes,
        expiry: datetime,
        as_float: bool,
        derivative: int,
    ) -> tuple[DualTypes, DualTypes | None]:
        """Get the derivative of sabr vol with respect to strike

        as_float: bool
            Allow expedited calculation by avoiding dual numbers. Useful during the root solving
            phase of Newton iterations.
        derivative: int
            For with respect to `k` use 1, or `f` use 2.
        """
        t_e = (expiry - self._meta.eval_date).days / 365.0
        K = k + self.meta.rate_shift
        F = f + self.meta.rate_shift
        del k, f

        if as_float:
            k_: Number = _dual_float(K)
            f_: Number = _dual_float(F)
            a_: Number = _dual_float(self.nodes.alpha)
            b_: float | Variable = _dual_float(self.nodes.beta)
            p_: Number = _dual_float(self.nodes.rho)
            v_: Number = _dual_float(self.nodes.nu)
        else:
            k_ = _to_number(K)
            f_ = _to_number(F)
            a_ = self.nodes.alpha  #
            b_ = self.nodes.beta
            p_ = self.nodes.rho
            v_ = self.nodes.nu

        return _SabrModel._d_sabr_d_k_or_f(k_, f_, t_e, a_, b_, p_, v_, derivative)

    ### _WithMutability ABCs:

    def _get_node_vector(self) -> np.ndarray[tuple[int, ...], np.dtype[np.object_]]:
        """Get a 1d array of variables associated with nodes of this object updated by Solver"""
        return np.array([self.nodes.alpha, self.nodes.rho, self.nodes.nu])

    def _get_node_vars(self) -> tuple[str, ...]:
        """Get the variable names of elements updated by a Solver"""
        return tuple(f"{self.id}{i}" for i in range(3))

    def _set_node_vector_direct(
        self, vector: np.ndarray[tuple[int, ...], np.dtype[np.object_]], ad: int
    ) -> None:
        """
        Update the node values in a Solver. ``ad`` in {1, 2}.
        Only the real values in vector are used, dual components are dropped and restructured.
        """
        DualType: type[Dual] | type[Dual2] = Dual if ad == 1 else Dual2
        DualArgs: tuple[list[float]] | tuple[list[float], list[float]] = (
            ([],) if ad == 1 else ([], [])
        )
        base_obj = DualType(0.0, [f"{self.id}{i}" for i in range(3)], *DualArgs)
        ident = np.eye(3)

        self._nodes = _SabrSmileNodes(
            _beta=self.nodes.beta,
            _alpha=DualType.vars_from(
                base_obj,  # type: ignore[arg-type]
                vector[0].real,
                base_obj.vars,
                ident[0, :].tolist(),
                *DualArgs[1:],
            ),
            _rho=DualType.vars_from(
                base_obj,  # type: ignore[arg-type]
                vector[1].real,
                base_obj.vars,
                ident[1, :].tolist(),
                *DualArgs[1:],
            ),
            _nu=DualType.vars_from(
                base_obj,  # type: ignore[arg-type]
                vector[2].real,
                base_obj.vars,
                ident[2, :].tolist(),
                *DualArgs[1:],
            ),
        )

    def _set_ad_order_direct(self, order: int | None) -> None:
        """This does not alter the beta node, since that is not varied by a Solver.
        beta values that are AD sensitive should be given as a Variable and not Dual/Dual2.

        Using `None` allows this Smile to be constructed without overwriting any variable names.
        """
        # -1, -2 force updates to new variables
        if order is None or order == getattr(self, "ad", None):
            return None
        elif abs(order) not in [0, 1, 2]:
            raise ValueError("`order` can only be in {0, 1, 2} for auto diff calcs.")

        self._ad = abs(order)
        self._nodes = _SabrSmileNodes(
            _beta=self.nodes.beta,
            _alpha=set_order_convert(self.nodes.alpha, order, [f"{self.id}0"]),
            _rho=set_order_convert(self.nodes.rho, order, [f"{self.id}1"]),
            _nu=set_order_convert(self.nodes.nu, order, [f"{self.id}2"]),
        )

    def _set_single_node(self, key: str, value: DualTypes) -> None:
        params = ["alpha", "rho", "nu", "beta"]
        if key not in params:
            raise KeyError(f"'{key}' is not in `nodes`.")
        kwargs = {f"_{_}": getattr(self.nodes, _) for _ in params if _ != key}
        kwargs.update({f"_{key}": value})
        self._nodes = _SabrSmileNodes(**kwargs)
        self._set_ad_order(self.ad)

    # _BaseIRSmile ABCS:

    def _plot(
        self,
        x_axis: str,
        f: float,
        y_axis: str,
        tgt_shift: float_,
    ) -> tuple[Iterable[float], Iterable[float]]:
        shf = _dual_float(self.meta.shift) / 100.0
        v_ = _dual_float(self.get_from_strike(k=f, f=f).vol) / 100.0
        sq_t = self._meta.t_expiry_sqrt
        x_low = _dual_float(
            dual_exp(0.5 * v_**2 * sq_t**2 - dual_inv_norm_cdf(0.95) * v_ * sq_t) * (f + shf) - shf
        )
        x_top = _dual_float(
            dual_exp(0.5 * v_**2 * sq_t**2 - dual_inv_norm_cdf(0.05) * v_ * sq_t) * (f + shf) - shf
        )

        x = np.linspace(x_low, x_top, 301, dtype=np.float64)
        y: Iterable[float] = [_dual_float(self.get_from_strike(k=_, f=f).vol) for _ in x]

        return self._plot_conversion(
            y_axis=y_axis, x_axis=x_axis, f=f, shift=shf, tgt_shift=_drb(shf, tgt_shift), x=x, y=y
        )

    @property
    def ad(self) -> int:
        return self._ad

    @property
    def pricing_params(self) -> tuple[float | Dual | Dual2 | Variable, ...]:
        return self.nodes.alpha, self.nodes.rho, self.nodes.nu

    @property
    def meta(self) -> _IRSmileMeta:
        return self._meta

    def _get_from_strike(self, k: DualTypes, f: DualTypes) -> _IRVolPricingParams:
        """
        Given an option strike return the volatility.

        Parameters
        -----------
        k: float, Dual, Dual2
            The strike of the option.
        f: float, Dual, Dual2
            The forward rate at delivery of the option.
        expiry: datetime, optional
            The expiry of the option. Required for temporal interpolation.
        tenor: datetime, optional
            The termination date of the underlying *IRS*, required for parameter interpolation.
        curves: _Curves,
            Pricing objects. See **Pricing** on :class:`~rateslib.instruments.IRSCall`
            for details of allowed inputs.

        Returns
        -------
        _IRVolPricingParams
        """
        vol_ = _SabrModel._d_sabr_d_k_or_f(
            _to_number(k + self.meta.rate_shift),
            _to_number(f + self.meta.rate_shift),
            self._meta.t_expiry,
            self.nodes.alpha,
            self.nodes.beta,
            self.nodes.rho,
            self.nodes.nu,
            derivative=0,
        )[0]
        return _IRVolPricingParams(
            vol=vol_ * 100.0,
            k=k,
            f=f,
            shift=self.meta.shift,
            pricing_model=OptionPricingModel.Black76,
            t_e=self._meta.t_expiry,
        )


class IRSabrCube(_BaseIRCube[str], _WithMutability):
    r"""
    Create an *IR Volatility Cube* parametrized by :class:`~rateslib.volatility.IRSabrSmile` at
    different expiries and *IRS* tenors.

    .. warning::

       *Swaptions* and *IR Volatility* are in Beta status introduced in v2.7.0

    See also the :ref:`IR Vol Smiles & Cubes <c-ir-smile-doc>` section in the user guide.

    .. role:: green

    .. role:: red

    Parameters
    ----------
    eval_date: datetime, :red:`required`
        Acts as the initial node of a *Curve*. Should be assigned today's immediate date.
        If expiry is given as string used to derive the specific date.
    expiries: list[datetime | str], :red:`required`
        Datetimes representing the expiries of each parametrized *Smile*, in ascending order.
    tenors: list[str], :red:`required`
        The tenors of each underlying *IRS* from each expiry for the parameterised *Smiles*.
    alpha: float, Variable, or 2D-ndarray of such, :red:`required`
        The alpha, :math:`\alpha_{expiry, tenor}`, parameters of each (expiry, tenor) node.
    rho: float, Variable, or 2D-ndarray of such, :red:`required`
        The rho, :math:`\rho_{expiry, tenor}`, parameters of each (expiry, tenor) node.
    nu: float, Variable, or 2D-ndarray of such, :red:`required`
        The nu, :math:`\nu_{expiry, tenor}`, parameters of each (expiry, tenor) node.
    irs_series: str, IRSSeries, :red:`required`
        The :class:`~rateslib.data.fixings.IRSSeries` that contains the parameters for the
        underlying :class:`~rateslib.instruments.IRS` that the swaptions are settled against.
    beta: float, Variable, :red:`required`
        The beta, :math:`\beta`, parameter of the SABR model.
    weights: Series, optional
       Weights used for temporal volatility interpolation. See notes.
    id: str, optional
       The unique identifier to label the *Surface* and its variables.
    ad: int, optional
       Sets the automatic differentiation order. Defines whether to convert node
       values to float, :class:`~rateslib.dual.Dual` or
       :class:`~rateslib.dual.Dual2`. It is advised against
       using this setting directly. It is mainly used internally.

    Notes
    -----
    TBD
    """

    _ini_solve = 0
    _SmileType = IRSabrSmile
    _meta: _IRCubeMeta
    _id: str

    def __init__(
        self,
        eval_date: datetime,
        expiries: list[datetime | str],
        tenors: list[str],
        alpha: DualTypes | Arr2dObj,
        rho: DualTypes | Arr2dObj,
        nu: DualTypes | Arr2dObj,
        irs_series: str | IRSSeries,
        beta: DualTypes,
        shift: DualTypes_ = NoInput(0),
        weights: Series[float] | NoInput = NoInput(0),
        id: str | NoInput = NoInput(0),  # noqa: A002
        ad: int = 0,
    ):
        self._id: str = (
            uuid4().hex[:5] + "_" if isinstance(id, NoInput) else id
        )  # 1 in a million clash

        self._meta = _IRCubeMeta(
            _eval_date=eval_date,
            _tenors=tenors,
            _weights=weights,
            _expiries=expiries,
            _irs_series=_get_irs_series(irs_series),
            _shift=_drb(0.0, shift),
            _indexes=["alpha", "rho", "nu"],
            _smile_params=dict(beta=beta),
        )

        _shape = (self.meta._n_expiries, self.meta._n_tenors)
        self._node_values_: Arr3dObj = np.empty(shape=_shape + (3,), dtype=object)
        for i, kw in enumerate([alpha, rho, nu]):
            if isinstance(kw, float | Dual | Dual2 | Variable):
                self._node_values_[:, :, i] = np.full(fill_value=kw, shape=_shape)
            else:
                self._node_values_[:, :, i] = np.asarray(kw)

        self._set_ad_order(ad)  # includes csolve on each smile
        self._set_new_state()

    @property
    def beta(self) -> DualTypes:
        """The *beta*  value of each :class:`~rateslib.volatility.IRSabrSmile` associated with
        this *Cube*."""
        return self.meta.smile_params["beta"]  # type: ignore[no-any-return]

    @property
    def alpha(self) -> DataFrame:
        """The *alpha*  value of each :class:`~rateslib.volatility.IRSabrSmile` associated with
        this *Cube*."""
        return DataFrame(
            index=Index(data=self.meta.expiries, name="expiry"),
            columns=Index(data=self.meta.tenors, name="tenor"),
            data=self._node_values_[:, :, 0],
        )

    @property
    def alpha_float(self) -> DataFrame:
        """The *alpha*  value of each :class:`~rateslib.volatility.IRSabrSmile` associated with
        this *Cube* in float format."""
        return self.alpha.map(lambda x: _dual_float(x))

    @property
    def rho(self) -> DataFrame:
        """The *rho*  value of each :class:`~rateslib.volatility.IRSabrSmile` associated with
        this *Cube*."""
        return DataFrame(
            index=Index(data=self.meta.expiries, name="expiry"),
            columns=Index(data=self.meta.tenors, name="tenor"),
            data=self._node_values_[:, :, 1],
        )

    @property
    def rho_float(self) -> DataFrame:
        """The *rho*  value of each :class:`~rateslib.volatility.IRSabrSmile` associated with
        this *Cube* in float format."""
        return self.rho.map(lambda x: _dual_float(x))

    @property
    def nu(self) -> DataFrame:
        """The *nu*  value of each :class:`~rateslib.volatility.IRSabrSmile` associated with
        this *Cube*."""
        return DataFrame(
            index=Index(data=self.meta.expiries, name="expiry"),
            columns=Index(data=self.meta.tenors, name="tenor"),
            data=self._node_values_[:, :, 2],
        )

    @property
    def nu_float(self) -> DataFrame:
        """The *nu*  value of each :class:`~rateslib.volatility.IRSabrSmile` associated with
        this *Cube* in float format."""
        return self.nu.map(lambda x: _dual_float(x))

    @property
    def _n(self) -> int:
        """Number of pricing parameters of the *Cube*."""
        en = self._node_values_.shape[0]
        tn = self._node_values_.shape[1]
        return en * tn * 3  # alpha, beta, rho

    @property
    def id(self) -> str:
        """A str identifier to name the *Surface* used in
        :class:`~rateslib.solver.Solver` mappings."""
        return self._id

    @property
    def meta(self) -> _IRCubeMeta:
        """An instance of :class:`~rateslib.volatility._IRCubeMeta`."""
        return self._meta

    @property
    def pricing_params(self) -> Arr3dObj:
        """The pricing parameters of the *Cube* as 3-d array by (expiry, tenor, strike)."""
        return self._node_values_

    @property
    def ad(self) -> int:
        """Int in {0,1,2} describing the AD order associated with the *Surface*."""
        return self._ad

    def _set_ad_order_direct(self, order: int | None) -> None:
        if order == getattr(self, "ad", None):
            return None
        elif order not in [0, 1, 2]:
            raise ValueError("`order` can only be in {0, 1, 2} for auto diff calcs.")

        self._ad = order
        vec = self._get_node_vector()
        vars_ = self._get_node_vars()
        new_vec = [set_order_convert(v, order, [t]) for v, t in zip(vec, vars_, strict=False)]
        en = self._node_values_.shape[0]
        tn = self._node_values_.shape[1]
        n = en * tn
        self._node_values_[:, :, 0] = np.reshape(list(new_vec[:n]), (en, tn))
        self._node_values_[:, :, 1] = np.reshape(list(new_vec[n : 2 * n]), (en, tn))
        self._node_values_[:, :, 2] = np.reshape(list(new_vec[2 * n :]), (en, tn))
        return None

    def _set_node_vector_direct(
        self, vector: np.ndarray[tuple[int, ...], np.dtype[np.object_]], ad: int
    ) -> None:
        en = self._node_values_.shape[0]
        tn = self._node_values_.shape[1]
        n = en * tn
        if ad == 0:
            self._node_values_[:, :, 0] = np.reshape([_dual_float(_) for _ in vector[:n]], (en, tn))
            self._node_values_[:, :, 1] = np.reshape(
                [_dual_float(_) for _ in vector[n : 2 * n]], (en, tn)
            )
            self._node_values_[:, :, 2] = np.reshape(
                [_dual_float(_) for _ in vector[2 * n :]], (en, tn)
            )
        else:
            DualType: type[Dual] | type[Dual2] = Dual if ad == 1 else Dual2
            DualArgs: tuple[list[float]] | tuple[list[float], list[float]] = (
                ([],) if ad == 1 else ([], [])
            )
            vars_ = self._get_node_vars()
            base_obj = DualType(0.0, vars_, *DualArgs)
            ident = np.eye(len(vars_))
            for i in range(3):
                self._node_values_[:, :, i] = np.reshape(
                    [
                        DualType.vars_from(
                            base_obj,  #  type: ignore[arg-type]
                            _dual_float(vector[n * i + j]),
                            base_obj.vars,
                            ident[n * i + j, :].tolist(),
                            *DualArgs[1:],
                        )
                        for j in range(n)
                    ],
                    (en, tn),
                )

    def _get_node_vector(self) -> Arr1dObj:
        """Get a 1d array of variables associated with nodes of this object updated by Solver"""
        return np.block(
            [
                self._node_values_[:, :, 0].ravel(),  # alphas
                self._node_values_[:, :, 1].ravel(),  # rhos
                self._node_values_[:, :, 2].ravel(),  # nus
            ]
        )

    def _get_node_vars(self) -> tuple[str, ...]:
        """Get the variable names of elements updated by a Solver"""
        vars_: tuple[str, ...] = ()
        for tag in ["_a_", "_p_", "_v_"]:
            vars_ += tuple(
                f"{self.id}{tag}{i}_{j}"
                for i in range(self._node_values_.shape[0])
                for j in range(self._node_values_.shape[1])
            )
        return vars_

    def _set_single_node_direct(
        self, key: tuple[datetime, datetime, str], value: DualTypes
    ) -> None:
        """
        Update some generic parameters on the *SabrCube*.

        Parameters
        ----------
        key: tuple of (datetime, datetime, str in {"alpha", "rho", "nu"})
            The node value to update, indexed by (expiry, tenor, SABR param).
        value: Array, float, Dual, Dual2, Variable
            Value to update on the *Cube*.

        Returns
        -------
        None

        Notes
        -----
        This function may update all of the AD variable names to be a consistent pricing object
        familiar to a :class:`~rateslib.solver.Solver`.

        .. warning::

           *Rateslib* is an object-oriented library that uses complex associations. Although
           Python may not object to directly mutating attributes of a *Curve* instance, this
           should be avoided in *rateslib*. Only use official ``update`` methods to mutate the
           values of an existing *Curve* instance.
           This class is labelled as a **mutable on update** object.

        """
        params = ["alpha", "rho", "nu"]
        if key[2] not in params:
            raise KeyError(f"'{key[2]}' is not in `nodes`.")

        tenor_row = self.meta.expiry_dates.index(key[0])
        self._node_values_[
            self.meta.expiry_dates.index(key[0]),
            self.meta.tenor_dates[tenor_row].tolist().index(key[1]),
            self.meta.indexes.index(key[2]),
        ] = value

        self._set_ad_order(self.ad)
        return None
