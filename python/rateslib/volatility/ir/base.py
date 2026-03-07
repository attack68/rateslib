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
from typing import TYPE_CHECKING, NoReturn, TypeAlias

import numpy as np

from rateslib.default import PlotOutput, plot
from rateslib.dual import Dual, Dual2, Variable
from rateslib.dual.utils import _dual_float
from rateslib.enums.generics import NoInput, _drb
from rateslib.enums.parameters import OptionPricingModel
from rateslib.mutability import _clear_cache_post, _new_state_post, _WithCache, _WithState
from rateslib.volatility.ir.utils import _IRSmileMeta

if TYPE_CHECKING:
    from rateslib.local_types import (  # pragma: no cover
        Any,
        CurvesT_,
        DualTypes_,
        Iterable,
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
        """This does not alter the beta node, since that is not varied by a Solver.
        beta values that are AD sensitive should be given as a Variable and not Dual/Dual2.

        Using `None` allows this Smile to be constructed without overwriting any variable names.
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
    - **params** (tuple[float | Dual | Dual2 | Variable])

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
            Pricing objects. See **Pricing** on :class:`~rateslib.instruments.IRCall`
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
