from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ParamSpec

from pandas import DataFrame

from rateslib.default import NoInput
from rateslib.fx import FXForwards, FXRates
from rateslib.instruments.utils import (
    _get_curves_fx_and_base_maybe_from_solver,
)
from rateslib.solver import Solver

if TYPE_CHECKING:
    from rateslib.typing import FX_, NPV, Curves_, Dual2, DualTypes
P = ParamSpec("P")


class Sensitivities:
    """
    Base class to add risk sensitivity calculations to an object with an ``npv()``
    method.
    """

    npv: Callable[..., NPV]
    cashflows: Callable[..., DataFrame]

    def delta(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver | NoInput = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str | NoInput = NoInput(0),
        local: bool = False,
        **kwargs: Any,
    ) -> DataFrame:
        """
        Calculate delta risk of an *Instrument* against the calibrating instruments in a
        :class:`~rateslib.curves.Solver`.

        Parameters
        ----------
        curves : Curve, str or list of such, optional
            A single :class:`~rateslib.curves.Curve` or id or a list of such.
            A list defines the following curves in the order:

            - Forecasting :class:`~rateslib.curves.Curve` for ``leg1``.
            - Discounting :class:`~rateslib.curves.Curve` for ``leg1``.
            - Forecasting :class:`~rateslib.curves.Curve` for ``leg2``.
            - Discounting :class:`~rateslib.curves.Curve` for ``leg2``.
        solver : Solver, optional
            The :class:`~rateslib.solver.Solver` that calibrates
            *Curves* from given *Instruments*.
        fx : float, FXRates, FXForwards, optional
            The immediate settlement FX rate that will be used to convert values
            into another currency. A given `float` is used directly. If giving a
            :class:`~rateslib.fx.FXRates` or :class:`~rateslib.fx.FXForwards` object,
            converts from local currency into ``base``.
        base : str, optional
            The base currency to convert cashflows into (3-digit code), set by default.
            Only used if ``fx_rate`` is an :class:`~rateslib.fx.FXRates` or
            :class:`~rateslib.fx.FXForwards` object.
        local : bool, optional
            If `True` will ignore ``base`` - this is equivalent to setting ``base`` to *None*.
            Included only for argument signature consistent with *npv*.

        Returns
        -------
        DataFrame
        """
        if isinstance(solver, NoInput):
            raise ValueError("`solver` is required for delta/gamma methods.")
        npv: dict[str, DualTypes] = self.npv(curves, solver, fx, base, local=True, **kwargs)  # type: ignore[assignment]
        _, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            NoInput(0),
            solver,
            NoInput(0),
            fx,
            base,
            NoInput(0),
        )
        if local:
            base_ = NoInput(0)
        return solver.delta(npv, base_, fx_)  # type: ignore[arg-type]

    def exo_delta(
        self,
        vars: list[str],  # noqa: A002
        curves: Curves_ = NoInput(0),
        solver: Solver | NoInput = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str | NoInput = NoInput(0),
        local: bool = False,
        vars_scalar: list[float] | NoInput = NoInput(0),
        vars_labels: list[str] | NoInput = NoInput(0),
        **kwargs: Any,
    ) -> DataFrame:
        """
        Calculate delta risk of an *Instrument* against some exogenous user created *Variables*.

        See :ref:`What are exogenous variables? <cook-exogenous-doc>` in the cookbook.

        Parameters
        ----------
        vars : list[str]
            The variable tags which to determine sensitivities for.
        curves : Curve, str or list of such, optional
            A single :class:`~rateslib.curves.Curve` or id or a list of such.
            A list defines the following curves in the order:

            - Forecasting :class:`~rateslib.curves.Curve` for ``leg1``.
            - Discounting :class:`~rateslib.curves.Curve` for ``leg1``.
            - Forecasting :class:`~rateslib.curves.Curve` for ``leg2``.
            - Discounting :class:`~rateslib.curves.Curve` for ``leg2``.

        solver : Solver, optional
            The :class:`~rateslib.solver.Solver` that calibrates
            *Curves* from given *Instruments*.
        fx : float, FXRates, FXForwards, optional
            The immediate settlement FX rate that will be used to convert values
            into another currency. A given `float` is used directly. If giving a
            :class:`~rateslib.fx.FXRates` or :class:`~rateslib.fx.FXForwards` object,
            converts from local currency into ``base``.
        base : str, optional
            The base currency to convert cashflows into (3-digit code), set by default.
            Only used if ``fx_rate`` is an :class:`~rateslib.fx.FXRates` or
            :class:`~rateslib.fx.FXForwards` object.
        local : bool, optional
            If `True` will ignore ``base`` - this is equivalent to setting ``base`` to *None*.
            Included only for argument signature consistent with *npv*.
        vars_scalar : list[float], optional
            Scaling factors for each variable, for example converting rates to basis point etc.
            Defaults to ones.
        vars_labels : list[str], optional
            Alternative names to relabel variables in DataFrames.

        Returns
        -------
        DataFrame
        """
        if isinstance(solver, NoInput):
            raise ValueError("`solver` is required for delta/gamma methods.")
        npv = self.npv(curves, solver, fx, base, local=True, **kwargs)
        _, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            NoInput(0),
            solver,
            NoInput(0),
            fx,
            base,
            NoInput(0),
        )
        if local:
            base_ = NoInput(0)
        return solver.exo_delta(
            npv=npv,  # type: ignore[arg-type]
            vars=vars,
            base=base_,
            fx=fx_,
            vars_scalar=vars_scalar,
            vars_labels=vars_labels,
        )

    def gamma(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver | NoInput = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str | NoInput = NoInput(0),
        local: bool = False,
        **kwargs: Any,
    ) -> DataFrame:
        """
        Calculate cross-gamma risk of an *Instrument* against the calibrating instruments of a
        :class:`~rateslib.curves.Solver`.

        Parameters
        ----------
        curves : Curve, str or list of such, optional
            A single :class:`~rateslib.curves.Curve` or id or a list of such.
            A list defines the following curves in the order:

            - Forecasting :class:`~rateslib.curves.Curve` for ``leg1``.
            - Discounting :class:`~rateslib.curves.Curve` for ``leg1``.
            - Forecasting :class:`~rateslib.curves.Curve` for ``leg2``.
            - Discounting :class:`~rateslib.curves.Curve` for ``leg2``.
        solver : Solver, optional
            The :class:`~rateslib.solver.Solver` that calibrates
            *Curves* from given *Instruments*.
        fx : float, FXRates, FXForwards, optional
            The immediate settlement FX rate that will be used to convert values
            into another currency. A given `float` is used directly. If giving a
            :class:`~rateslib.fx.FXRates` or :class:`~rateslib.fx.FXForwards` object,
            converts from local currency into ``base``.
        base : str, optional
            The base currency to convert cashflows into (3-digit code), set by default.
            Only used if ``fx_rate`` is an :class:`~rateslib.fx.FXRates` or
            :class:`~rateslib.fx.FXForwards` object.
        local : bool, optional
            If `True` will ignore ``base``. This is equivalent to setting ``base`` to *None*.
            Included only for argument signature consistent with *npv*.

        Returns
        -------
        DataFrame
        """
        if isinstance(solver, NoInput):
            raise ValueError("`solver` is required for delta/gamma methods.")
        _, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            NoInput(0),
            solver,
            NoInput(0),
            fx,
            base,
            NoInput(0),
        )
        if local:
            base_ = NoInput(0)

        # store original order
        if id(solver.fx) != id(fx_) and isinstance(fx_, FXRates | FXForwards):
            # then the fx_ object is available on solver but that is not being used.
            _ad2 = fx_._ad
            fx_._set_ad_order(2)

        _ad1 = solver._ad
        solver._set_ad_order(2)

        npv: dict[str, Dual2] = self.npv(curves, solver, fx_, base_, local=True, **kwargs)  # type: ignore[assignment]
        grad_s_sT_P: DataFrame = solver.gamma(npv, base_, fx_)

        # reset original order
        if id(solver.fx) != id(fx_) and isinstance(fx_, FXRates | FXForwards):
            fx_._set_ad_order(_ad2)
        solver._set_ad_order(_ad1)

        return grad_s_sT_P
