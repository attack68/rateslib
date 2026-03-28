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

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Generic, NoReturn, TypeAlias, TypeVar

import numpy as np

from rateslib.curves.interpolation import index_left
from rateslib.default import PlotOutput, plot
from rateslib.dual import Dual, Dual2, Variable
from rateslib.dual.utils import _dual_float
from rateslib.enums.generics import NoInput, _drb
from rateslib.enums.parameters import OptionPricingModel
from rateslib.mutability import _clear_cache_post, _new_state_post, _WithCache, _WithState
from rateslib.volatility.ir.utils import (
    _bilinear_interp,
    _get_ir_expiry,
    _get_ir_tenor,
    _IRCubeMeta,
    _IRSmileMeta,
)

UTC = timezone.utc
T = TypeVar("T")

if TYPE_CHECKING:
    from rateslib.local_types import (  # pragma: no cover
        Any,
        Arr1dObj,
        Arr3dObj,
        CurvesT_,
        DualTypes_,
        Iterable,
        Sequence,
        _IRVolPricingParams,
        datetime_,
        float_,
    )

DualTypes: TypeAlias = "float | Dual | Dual2 | Variable"  # if not defined causes _WithCache failure


class _WithMutability(ABC):
    """Abstract base class containing the necessary methods to interoperate with a
    :class:`~rateslib.solver.Solver`."""

    # Get methods allow the Solver to extract and order the parameters of the pricing object.

    @property
    @abstractmethod
    def _n(self) -> int:
        """The number of parameters associated with the pricing object."""
        pass

    @property
    @abstractmethod
    def _ini_solve(self) -> int:
        """The number of parameters that are initially ignored by
        :class:`~rateslib.solver.Solver` and not mutated during iterations."""
        pass

    @abstractmethod
    def _get_node_vector(self) -> np.ndarray[tuple[int, ...], np.dtype[np.object_]]:
        """Get a 1d array of variables associated with nodes of this object updated by Solver"""
        pass

    @abstractmethod
    def _get_node_vars(self) -> tuple[str, ...]:
        """Get the variable names of elements updated by a Solver"""
        pass

    # Set methods allow the Solver to make mutable updates to the pricing object
    # Direct methods implement the underlying operations, wrapped methods (which are
    # automatically provided) control additionals such as cache clearing and state management.

    @abstractmethod
    def _set_node_vector_direct(
        self, vector: np.ndarray[tuple[int, ...], np.dtype[np.object_]], ad: int
    ) -> None:
        """
        Allow Solver to update parameter values of the pricing object.
        ``ad`` in {1, 2}.
        Only the real values in vector are used, dual components are dropped and restructured.
        """
        pass

    @abstractmethod
    def _set_ad_order_direct(self, order: int | None) -> None:
        """
        Update the parameter values of the pricing object.

        None: Do nothing regardless of the AD order of the parameters as stated.
        0: Convert all values to float.
        1: Convert to Dual with vars ordered by `_get_node_vars`
        2: Convert to Dual2 with vars ordered by `_get_node_vars`
        """
        pass

    @abstractmethod
    def _set_single_node(self, key: Any, value: DualTypes) -> None:
        """
        Update a single named node on the pricing object.
        """
        pass

    @_new_state_post
    @_clear_cache_post
    def _set_node_vector(
        self, vector: np.ndarray[tuple[int, ...], np.dtype[np.object_]], ad: int
    ) -> None:
        """
        Update the node values in a Solver. ``ad`` in {1, 2}.
        Only the real values in vector are used, dual components are dropped and restructured.
        """
        return self._set_node_vector_direct(vector, ad)

    @_clear_cache_post
    def _set_ad_order(self, order: int | None) -> None:
        """
        When pricing objects are mutated by a Solver this method should convert pricing
        parameters to DualTypes with `vars` as defined by the solver, i.e. overwriting
        any user specific DualTypes.

        If `order` is *None*, this method will do nothing.

        If `order` is in [0, 1, 2] and that matches the existing AD order of the object then
        nothing is also done.

        If `order` is in [0, 1, 2] and that represents a new AD order then values are converted
        using `vars` configured and expected by a Solver.

        If `order` is in [-1, -2] this forces a conversion to the appropriate order, even if the
        object matches the requested AD order. I.e. user variables will be overridden regardless.
        """
        return self._set_ad_order_direct(order)

    @_new_state_post
    @_clear_cache_post
    def update_node(self, key: str, value: DualTypes) -> None:
        """
        Update a single node value on the *Smile*.

        Parameters
        ----------
        key: str in {"alpha", "beta", "rho", "nu"}
            The node value to update.
        value: float, Dual, Dual2, Variable
            Value to update on the *Smile*.

        Returns
        -------
        None

        Notes
        -----

        .. warning::

           *Rateslib* is an object-oriented library that uses complex associations. Although
           Python may not object to directly mutating attributes of a *Curve* instance, this
           should be avoided in *rateslib*. Only use official ``update`` methods to mutate the
           values of an existing *Curve* instance.
           This class is labelled as a **mutable on update** object.

        """
        return self._set_single_node(key, value)


class _BaseIRSmile(_WithState, _WithCache[float, DualTypes], ABC):
    """
    Abstract base class for implementing *IR Smiles*.

    Any :class:`~rateslib.volatility._BaseIRSmile` is required to implement the following
    **properties**:

    - **ad** (int)
    - **meta** (:class:`~rateslib.volatility._IRSmileMeta`)
    - **pricing_params** (Iterable[float | Dual | Dual2 | Variable])

    Any :class:`~rateslib.volatility._BaseIRSmile` is required to implement the following
    **methods**:

    - **_plot(x_axis, f, y_axis, curves)**
    - **_get_from_strike(k, f, curves)**

    The directly provided methods with these implementations are:

    - :meth:`~rateslib.volatility._BaseIRSmile.plot`.
    - :meth:`~rateslib.volatility._BaseIRSmile.get_from_strike`.

    """

    _default_plot_x_axis: str

    @property
    @abstractmethod
    def id(self) -> str:
        """
        A str identifier to name the *Smile* used in :class:`~rateslib.solver.Solver` mappings.
        """
        pass

    @property
    @abstractmethod
    def ad(self) -> int:
        """Int in {0,1,2} describing the AD order associated with the
        :class:`~rateslib.volatility._BaseIRSmile`."""
        pass

    @property
    @abstractmethod
    def meta(self) -> _IRSmileMeta:
        """An instance of :class:`~rateslib.volatility.ir.utils._IRSmileMeta`."""
        pass

    @property
    @abstractmethod
    def pricing_params(self) -> Iterable[float | Dual | Dual2 | Variable]:
        """An ordered set of pricing parameters associated with the
        :class:`~rateslib.volatility._BaseIRSmile`."""
        pass

    @abstractmethod
    def _get_from_strike(
        self,
        k: DualTypes,
        f: DualTypes,
    ) -> _IRVolPricingParams:
        """
        Given an option strike and forward rate return the volatility.

        Note this function does not validate the expiry and tenor of the intended option.

        Parameters
        -----------
        k: float, Dual, Dual2
            The strike of the option.
        f: float, Dual, Dual2
            The forward rate at delivery of the option.

        Returns
        -------
        _IRVolPricingParams
        """
        pass

    @abstractmethod
    def _plot(
        self,
        x_axis: str,
        f: float,
        y_axis: str,
        tgt_shift: float_,
    ) -> tuple[Iterable[float], Iterable[float]]:
        """Perform the necessary calculation to derive (x,y) coordinates for a chart."""
        pass

    def _plot_conversion(
        self,
        y_axis: str,
        x_axis: str,
        f: float,
        shift: float,
        tgt_shift: float,
        x: Iterable[float],
        y: Iterable[float],
    ) -> tuple[Iterable[float], Iterable[float]]:
        # def _hagan_convert(k: DualTypes, sigma_b: DualTypes) -> DualTypes:
        #     if abs(f - k) < 1e-13:
        #         center = f + shf
        #     else:
        #         center = (f - k) / dual_log((f + shf) / (k + shf))
        #     return sigma_b * center * (1 - sigma_b ** 2 * sq_t / 24)

        match (self.meta.pricing_model, y_axis.lower()):
            case (OptionPricingModel.Black76, "black_vol"):
                if shift == tgt_shift:
                    y_ = y
                else:
                    y_ = [
                        _
                        * (((f + shift) * (k + shift)) / ((f + tgt_shift) * (k + tgt_shift))) ** 0.5
                        for _, k in zip(y, x, strict=True)
                    ]
            case (OptionPricingModel.Bachelier, "normal_vol"):
                y_ = y
            case (OptionPricingModel.Black76, "normal_vol"):
                y_ = [
                    sigma_b * ((f + shift) * (k + shift)) ** 0.5
                    for (k, sigma_b) in zip(x, y, strict=True)
                ]
            case (OptionPricingModel.Bachelier, "black_vol"):
                y_ = [
                    sigma_n * ((f + tgt_shift) * (k + tgt_shift)) ** -0.5
                    for (k, sigma_n) in zip(x, y, strict=True)
                ]
            case _:
                raise ValueError("`y_axis` must be in {'normal_vol', 'black_vol'}.")

        if x_axis == "moneyness":
            u: Iterable[float] = x / f  # type: ignore[operator, assignment]
            return u, y_
        else:  # x_axis = "strike"
            return x, y_

    def plot(
        self,
        comparators: list[_BaseIRSmile] | NoInput = NoInput(0),
        labels: list[str] | NoInput = NoInput(0),
        x_axis: str | NoInput = NoInput(0),
        y_axis: str | NoInput = NoInput(0),
        f: DualTypes | NoInput = NoInput(0),
        curves: CurvesT_ = NoInput(0),
        shift: float_ = NoInput(0),
    ) -> PlotOutput:
        r"""
        Plot volatilities associated with the *Smile*.

        .. role:: green

        .. role:: red

        Parameters
        ----------
        comparators: list[Smile], :green:`optional`
            A list of Smiles which to include on the same plot as comparators.
        labels : list[str], :green:`optional`
            A list of strings associated with the plot and comparators. Must be same
            length as number of plots.
        x_axis : str in {"strike", "moneyness"}, :green:`optional (set by object)`
            *'strike'* is the natural option for this *SabrSmile*.
            If *'moneyness'* the strikes are converted using ``f``.
        y_axis : str in {"black_vol", "normal_vol"}, :green:`optional (set by object)`
            Convert the y-axis to a different representation using an approximation.
        f: DualTypes, :green:`optional`
            The mid-market IRS rate. If ``curves`` are not given then ``f`` is required.
        curves: Curves, :green:`optional`
            The *Curves* in the required form for an :class:`~rateslib.instruments.IRS`. If ``f``
            is not given then ``curves`` are required.
        shift: float, :green:`optional`
            If plotting a *'black_vol'* this will use an approximation to convert any native
            shift into another that is specified here. If not given uses the native shift meta
            attribute of the *Smile*.

        Returns
        -------
        (fig, ax, line) : Matplotlib.Figure, Matplotplib.Axes, Matplotlib.Lines2D

        Notes
        -----
        Any approximations converting between *normal* and *black* vol are done so with the
        first order approximation generally attributable to Fei Zhou. These approximations are only
        used for charting. Actual instrument pricing metrics are determined more accurately
        with root solvers.

        .. math::

           \sigma_{LN+h} \approx \frac{\sigma_{N}}{\sqrt{(F+h)(K+h)}}

        and,

        .. math::

           \sigma_{LN+h} \approx \sigma_{LN+h2} \sqrt{ \frac{(F+h_2)(K+h_2)}{(F+h)(K+h)}}

        for *h* and :math:`h_2` potentially different shifts.

        """  # noqa: E501
        if isinstance(f, NoInput) and isinstance(curves, NoInput):
            raise ValueError("`f` (ATM-forward interest rate) is required by `_BaseIRSmile.plot`.")
        elif isinstance(f, float | Dual | Dual2 | Variable):
            f_: float = _dual_float(f)
        elif not isinstance(curves, NoInput):
            f_ = _dual_float(self.meta.irs_fixing.irs.rate(curves=curves))
        del f

        # reversed for intuitive strike direction
        comparators = _drb([], comparators)
        labels = _drb([], labels)

        x_axis_: str = _drb(self.meta.plot_x_axis, x_axis)
        y_axis_: str = _drb(self.meta.plot_y_axis, y_axis)
        del x_axis, y_axis

        x_, y_ = self._plot(x_axis_, f_, y_axis_, shift)

        x: list[list[float]] = [list(x_)]
        y: list[list[float]] = [list(y_)]
        if not isinstance(comparators, NoInput):
            for smile in comparators:
                if not isinstance(smile, _BaseIRSmile):
                    raise ValueError("A `comparator` must be a valid IR Smile type.")
                x_, y_ = smile._plot(x_axis_, f_, y_axis_, shift)
                x.append(list(x_))
                y.append(list(y_))

        return plot(x, y, labels)

    def get_from_strike(
        self,
        k: DualTypes,
        expiry: datetime_ = NoInput(0),
        tenor: datetime_ = NoInput(0),
        f: DualTypes_ = NoInput(0),
        curves: CurvesT_ = NoInput(0),
    ) -> _IRVolPricingParams:
        """
        Given an option strike return the volatility.

        Note if the ``expiry`` and ``tenor`` are given these will be validated against the
        *_BaseIRSmile* *meta* parameters.

        .. role:: red

        .. role:: green

        Parameters
        -----------
        k: float, Dual, Dual2, Variable, :red:`required`
            The strike of the option.
        expiry: datetime, :green:`optional`
            The expiry of the option. Required for temporal interpolation.
        tenor: datetime, :green:`optional`
            The termination date of the underlying *IRS*, required for parameter interpolation.
        f: float, Dual, Dual2, Variable, :green:`optional`
            The forward rate at delivery of the option.
        curves: _Curves, :green:`optional`
            Pricing objects. See **Pricing** on :class:`~rateslib.instruments.IRSCall`
            for details of allowed inputs. Required if ``f`` is not given.

        Returns
        -------
        _IRVolPricingParams
        """
        if not isinstance(expiry, NoInput) and self.meta.expiry != expiry:
            raise ValueError(
                f"`expiry` of _BaseIRSmile and intended price do not match. Got: {expiry} "
                f"and {self.meta.expiry}.\nCalculation aborted due to potential pricing errors.",
            )
        if not isinstance(tenor, NoInput) and self.meta.irs_fixing.termination != tenor:
            raise ValueError(
                f"`tenor` of _BaseIRSmile and intended price do not match. Got: {tenor} "
                f"and {self.meta.irs_fixing.termination}.\nCalculation aborted due to potential "
                f"pricing errors.",
            )

        if isinstance(f, NoInput):
            f_: DualTypes = self.meta.irs_fixing.irs.rate(curves=curves)
        else:
            f_ = f
        del f

        return self._get_from_strike(f=f_, k=k)

    def __iter__(self) -> NoReturn:
        raise TypeError("`_BaseIRSmile` types are not iterable.")


class _BaseIRCube(Generic[T], _WithState, _WithCache[tuple[datetime, datetime], _BaseIRSmile], ABC):
    """
    Abstract base class for implementing *IR Cubes*.

    Any :class:`~rateslib.volatility._BaseIRCube` is required to implement the following
    **properties**:

    - **ad** (int)
    - **meta** (:class:`~rateslib.volatility._IRCubeMeta`)
    - **pricing_params** (3D ndarray)

    Any :class:`~rateslib.volatility._BaseIRCube` is required to implement the following
    **methods**:

    - **_construct_smile(expiry, tenor, params)**
    - **_get_from_strike(k, f, curves)**

    The directly provided methods with these implementations are:

    - :meth:`~rateslib.volatility._BaseIRCube.plot`.
    - :meth:`~rateslib.volatility._BaseIRCube.get_from_strike`.

    """

    _SmileType: type[_BaseIRSmile]
    _node_values_: Arr3dObj

    @property
    @abstractmethod
    def id(self) -> str:
        """
        A str identifier to name the *Cube* used in :class:`~rateslib.solver.Solver` mappings.
        """
        pass

    @property
    @abstractmethod
    def ad(self) -> int:
        """Int in {0,1,2} describing the AD order associated with the
        :class:`~rateslib.volatility._BaseIRCube`."""
        pass

    @property
    @abstractmethod
    def meta(self) -> _IRCubeMeta:
        """An instance of :class:`~rateslib.volatility.ir.utils._IRCubeMeta`."""
        pass

    @property
    @abstractmethod
    def pricing_params(self) -> Arr3dObj:
        """A 3-d array of pricing parameters with axes (expiry, tenor, strike)."""
        pass

    def _bilinear_interpolation(
        self,
        expiry: datetime,
        tenor: datetime,
    ) -> Arr1dObj:
        """
        Linearly interpolate the expiries / tenors array and return interpolated values
        for the alpha, rho and nu parameters.

        Returns
        -------
        (alpha, rho, nu)
        """
        # For out of bounds expiry values convert to boundary expiries with tenor time adjustment
        if expiry < self.meta.expiry_dates[0]:
            return self._bilinear_interpolation(
                expiry=self.meta.expiry_dates[0],
                tenor=tenor + (self.meta.expiry_dates[0] - expiry),
            )
        elif expiry > self.meta.expiry_dates[-1]:
            return self._bilinear_interpolation(
                expiry=self.meta.expiry_dates[-1],
                tenor=tenor - (expiry - self.meta.expiry_dates[-1]),
            )

        e_posix = expiry.replace(tzinfo=UTC).timestamp()
        t_posix = tenor.replace(tzinfo=UTC).timestamp()

        match (self.meta._n_expiries, self.meta._n_tenors):
            case (1, 1):
                # nothing to interpolate: return the only parameters of the surface
                return self.pricing_params[0, 0, :]

            case (1, _):
                # interpolate only over tenor
                e_l = 0
                e_l_p = 0
                t_posix_1 = t_posix - (e_posix - self.meta.expiries_posix[0])
                t_l_1 = index_left(
                    list_input=self.meta.tenor_dates_posix[0, :],  # type: ignore[arg-type]
                    list_length=self.meta._n_tenors,
                    value=t_posix_1,
                )
                t_l_1_p = t_l_1 + 1
                v_ = (0.0, 0.0)  # only one expiry so no interpolation over that dimension
                t_l_2, t_l_2_p = t_l_1, t_l_1_p
                h_: tuple[float, float] = (
                    (t_posix_1 - self.meta.tenor_dates_posix[e_l, t_l_1])
                    / (
                        self.meta.tenor_dates_posix[e_l, t_l_1_p]
                        - self.meta.tenor_dates_posix[e_l, t_l_1]
                    ),
                ) * 2

            case (_, 1):
                # interpolate only over expiry
                e_l = index_left(
                    list_input=self.meta.expiries_posix,
                    list_length=self.meta._n_expiries,
                    value=e_posix,
                )
                e_l_p = e_l + 1
                t_l_1, t_l_2 = 0, 0
                t_l_1_p, t_l_2_p = 0, 0
                h_ = (0, 0)
                v_ = (
                    (e_posix - self.meta.expiries_posix[e_l])
                    / (self.meta.expiries_posix[e_l_p] - self.meta.expiries_posix[e_l]),
                ) * 2

            case _:
                # perform true bilinear interpolation
                e_l = index_left(
                    list_input=self.meta.expiries_posix,
                    list_length=self.meta._n_expiries,
                    value=e_posix,
                )
                e_l_p = e_l + 1
                v_ = (
                    (e_posix - self.meta.expiries_posix[e_l])
                    / (self.meta.expiries_posix[e_l_p] - self.meta.expiries_posix[e_l]),
                ) * 2

                # these are the relative tenors as measured per each benchmark expiry
                t_posix_1 = t_posix - (e_posix - self.meta.expiries_posix[e_l])
                t_posix_2 = t_posix - (e_posix - self.meta.expiries_posix[e_l_p])

                t_l_1 = index_left(
                    list_input=self.meta.tenor_dates_posix[e_l, :],  # type: ignore[arg-type]
                    list_length=self.meta._n_tenors,
                    value=t_posix_1,
                )
                t_l_1_p = t_l_1 + 1
                t_l_2 = index_left(
                    list_input=self.meta.tenor_dates_posix[e_l_p, :],  # type: ignore[arg-type]
                    list_length=self.meta._n_tenors,
                    value=t_posix_2,
                )
                t_l_2_p = t_l_2 + 1

                h_ = (
                    (t_posix_1 - self.meta.tenor_dates_posix[e_l, t_l_1])
                    / (
                        self.meta.tenor_dates_posix[e_l, t_l_1 + 1]
                        - self.meta.tenor_dates_posix[e_l, t_l_1]
                    ),
                    (t_posix_2 - self.meta.tenor_dates_posix[e_l_p, t_l_2])
                    / (
                        self.meta.tenor_dates_posix[e_l_p, t_l_2 + 1]
                        - self.meta.tenor_dates_posix[e_l_p, t_l_2]
                    ),
                )

        h_ = (min(max(h_[0], 0), 1), min(max(h_[1], 0), 1))

        return np.array(
            [
                _bilinear_interp(
                    tl=param[e_l, t_l_1],
                    tr=param[e_l, t_l_1_p],
                    bl=param[e_l_p, t_l_2],
                    br=param[e_l_p, t_l_2_p],
                    h=h_,
                    v=v_,
                )
                for param in [
                    self.pricing_params[:, :, i] for i in range(self.pricing_params.shape[2])
                ]
            ]
        )

    def _construct_smile(
        self,
        expiry: datetime,
        tenor: datetime,
        params: Sequence[DualTypes] | Arr1dObj,
    ) -> _BaseIRSmile:
        return self._SmileType(  # type: ignore[call-arg]
            nodes=dict(zip(self.meta.indexes, params, strict=True)),
            eval_date=self.meta.eval_date,
            expiry=expiry,
            irs_series=self.meta.irs_series,
            tenor=tenor,
            shift=self.meta.shift,
            ad=None,  # inherit the AD variables from the params
            **self.meta.smile_params,
        )

    def get_from_strike(
        self,
        k: DualTypes,
        expiry: datetime,
        tenor: datetime,
        f: DualTypes_ = NoInput(0),
        curves: CurvesT_ = NoInput(0),
    ) -> _IRVolPricingParams:
        """
        Given an option strike, expiry and tenor, return the volatility.

        .. role:: red

        .. role:: green

        Parameters
        -----------
        k: float, Dual, Dual2, Variable, :red:`required`
            The strike of the option.
        expiry: datetime, :red:`required`
            The expiry of the option. Required for temporal interpolation.
        tenor: datetime, :red:`required`
            The termination date of the underlying *IRS*, required for parameter interpolation.
        f: float, Dual, Dual2, :green:`optional`
            The forward rate at delivery of the option.
        curves: _Curves, :green:`optional`
            Pricing objects. See **Pricing** notes of an :class:`~rateslib.instruments.IRSCall`
            for details of allowed inputs.

        Returns
        -------
        _IRVolPricingParams
        """
        smile = self.get_smile(expiry, tenor)
        return smile.get_from_strike(k=k, f=f, curves=curves)

    def get_smile(self, expiry: datetime | str, tenor: datetime | str) -> _BaseIRSmile:
        """
        Return a constructed :class:`~rateslib.volatility._BaseIRSmile` for a given
        expiry and tenor.

        .. role:: red

        .. role:: green

        Parameters
        -----------
        expiry: datetime, str, :red:`required`
            The expiry of the option. Required for temporal interpolation.
        tenor: datetime, str, :red:`required`
            The termination date of the underlying *IRS*, required for parameter interpolation.

        Returns
        -------
        _BaseIRSmile
        """
        expiry_ = _get_ir_expiry(
            eval_date=self.meta.eval_date, irs_series=self.meta.irs_series, expiry=expiry
        )
        tenor_ = _get_ir_tenor(expiry=expiry_, irs_series=self.meta.irs_series, tenor=tenor)
        del expiry, tenor

        if (expiry_, tenor_) in self._cache:
            smile = self._cache[expiry_, tenor_]
        else:
            params = self._bilinear_interpolation(expiry=expiry_, tenor=tenor_)
            smile = self._cached_value(
                key=(expiry_, tenor_), val=self._construct_smile(expiry_, tenor_, params)
            )
        return smile

    def _get_node_vector(self) -> Arr1dObj:
        """Get a 1d array of variables associated with nodes of this object updated by Solver"""
        return self.pricing_params.ravel()

    def _get_node_vars(self) -> tuple[str, ...]:
        """Get the variable names of elements updated by a Solver"""
        vars_: tuple[str, ...] = tuple(
            f"{self.id}{i}"
            for i in range(self.meta._n_expiries * self.meta._n_tenors * len(self.meta.indexes))
        )
        return vars_

    def _set_single_node(
        self, key: tuple[datetime | str, datetime | str, T], value: DualTypes
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
        expiry_ = _get_ir_expiry(
            eval_date=self.meta.eval_date, irs_series=self.meta.irs_series, expiry=key[0]
        )
        tenor_ = _get_ir_tenor(expiry=expiry_, irs_series=self.meta.irs_series, tenor=key[1])

        if expiry_ not in self.meta.expiry_dates:
            raise KeyError(f"'{expiry_}' is not in `meta.expiry_dates`.")

        tenor_row = self.meta.expiry_dates.index(expiry_)
        if tenor_ not in self.meta.tenor_dates[tenor_row]:
            raise KeyError(f"'{tenor_}' is not in `meta.tenor_dates`.")

        return self._set_single_node_direct((expiry_, tenor_, key[2]), value)

    @abstractmethod
    def _set_single_node_direct(self, key: tuple[datetime, datetime, T], value: DualTypes) -> None:
        pass
