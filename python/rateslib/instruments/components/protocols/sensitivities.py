from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from rateslib.enums.generics import NoInput
from rateslib.fx import FXForwards, FXRates
from rateslib.instruments.components.protocols.npv import _WithNPV
from rateslib.instruments.components.protocols.pricing import (
    _get_fx_forwards_maybe_from_solver,
)

if TYPE_CHECKING:
    from rateslib.typing import (
        CurvesT_,
        DataFrame,
        Dual,
        Dual2,
        FXForwards_,
        NoInput,
        Solver_,
        VolT_,
        datetime_,
        str_,
    )


class _WithSensitivities(_WithNPV, Protocol):
    """
    Protocol to establish **delta** and **gamma** calculations using a
    :class:`~rateslib.solver.Solver` of any *Instrument* type.
    """

    def delta(
        self,
        *,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: VolT_ = NoInput(0),
        base: str_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
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

        Returns
        -------
        DataFrame
        """
        if isinstance(solver, NoInput):
            raise ValueError("`solver` is required for delta/gamma methods.")
        npv: dict[str, Dual] = self.npv(  # type: ignore[assignment]
            curves=curves,
            solver=solver,
            fx=fx,
            vol=vol,
            base=base,
            forward=forward,
            settlement=settlement,
            local=True,
        )
        return solver.delta(
            npv=npv, base=base, fx=_get_fx_forwards_maybe_from_solver(fx=fx, solver=solver)
        )

    def exo_delta(
        self,
        *,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: VolT_ = NoInput(0),
        base: str_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
        vars: list[str],  # noqa: A002
        vars_scalar: list[float] | NoInput = NoInput(0),
        vars_labels: list[str] | NoInput = NoInput(0),
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
        npv: dict[str, Dual | Dual2] = self.npv(  # type: ignore[assignment]
            curves=curves,
            solver=solver,
            fx=fx,
            vol=vol,
            base=base,
            forward=forward,
            settlement=settlement,
            local=True,
        )
        return solver.exo_delta(
            npv=npv,
            vars=vars,
            base=base,
            fx=_get_fx_forwards_maybe_from_solver(fx=fx, solver=solver),
            vars_scalar=vars_scalar,
            vars_labels=vars_labels,
        )

    def gamma(
        self,
        *,
        curves: CurvesT_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FXForwards_ = NoInput(0),
        vol: VolT_ = NoInput(0),
        base: str_ = NoInput(0),
        settlement: datetime_ = NoInput(0),
        forward: datetime_ = NoInput(0),
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

        Returns
        -------
        DataFrame
        """
        if isinstance(solver, NoInput):
            raise ValueError("`solver` is required for delta/gamma methods.")

        fx_ = _get_fx_forwards_maybe_from_solver(fx=fx, solver=solver)
        # store original order
        if id(solver.fx) != id(fx_) and isinstance(fx_, FXRates | FXForwards):
            # then the fx_ object is available on solver but that is not being used.
            _ad_fx = fx_._ad
            fx_._set_ad_order(2)

        _ad_svr = solver._ad
        solver._set_ad_order(2)

        npv: dict[str, Dual2] = self.npv(  # type: ignore[assignment]
            curves=curves,
            solver=solver,
            fx=fx_,
            vol=vol,
            base=NoInput(0),  # local override
            settlement=settlement,
            forward=forward,
            local=True,
        )
        grad_s_sT_P: DataFrame = solver.gamma(npv, base, fx_)

        # reset original order
        if id(solver.fx) != id(fx_) and isinstance(fx_, FXRates | FXForwards):
            fx_._set_ad_order(_ad_fx)
        solver._set_ad_order(_ad_svr)

        return grad_s_sT_P
