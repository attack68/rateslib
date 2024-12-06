from __future__ import annotations

import abc
import warnings

from pandas import DataFrame, concat, isna

from rateslib import defaults
from rateslib.curves import Curve
from rateslib.default import NoInput
from rateslib.dual import Dual, Dual2, DualTypes
from rateslib.fx import FXForwards, FXRates
from rateslib.fx_volatility import FXVols
from rateslib.solver import Solver


def _get_curve_from_solver(curve, solver):
    if isinstance(curve, dict):
        # When supplying a curve as a dictionary of curves (for IBOR stubs) use recursion
        return {k: _get_curve_from_solver(v, solver) for k, v in curve.items()}
    elif getattr(curve, "_is_proxy", False):
        # TODO: (mid) consider also adding CompositeCurves as exceptions under the same rule
        # proxy curves exist outside of solvers but still have Dual variables associated
        # with curves inside the solver, so can still generate risks to calibrating
        # instruments
        return curve
    elif isinstance(curve, str):
        return solver.pre_curves[curve]
    elif curve is NoInput.blank or curve is None:
        # pass through a None curve. This will either raise errors later or not be needed
        return NoInput(0)
    else:
        try:
            # it is a safeguard to load curves from solvers when a solver is
            # provided and multiple curves might have the same id
            _ = solver.pre_curves[curve.id]
            if id(_) != id(curve):  # Python id() is a memory id, not a string label id.
                raise ValueError(
                    "A curve has been supplied, as part of ``curves``, which has the same "
                    f"`id` ('{curve.id}'),\nas one of the curves available as part of the "
                    "Solver's collection but is not the same object.\n"
                    "This is ambiguous and cannot price.\n"
                    "Either refactor the arguments as follows:\n"
                    "1) remove the conflicting curve: [curves=[..], solver=<Solver>] -> "
                    "[curves=None, solver=<Solver>]\n"
                    "2) change the `id` of the supplied curve and ensure the rateslib.defaults "
                    "option 'curve_not_in_solver' is set to 'ignore'.\n"
                    "   This will remove the ability to accurately price risk metrics.",
                )
            return _
        except AttributeError:
            raise AttributeError(
                "`curve` has no attribute `id`, likely it not a valid object, got: "
                f"{curve}.\nSince a solver is provided have you missed labelling the `curves` "
                f"of the instrument or supplying `curves` directly?",
            )
        except KeyError:
            if defaults.curve_not_in_solver == "ignore":
                return curve
            elif defaults.curve_not_in_solver == "warn":
                warnings.warn("`curve` not found in `solver`.", UserWarning)
                return curve
            else:
                raise ValueError("`curve` must be in `solver`.")


def _get_base_maybe_from_fx(
    fx: float | FXRates | FXForwards | NoInput,
    base: str | NoInput,
    local_ccy: str | NoInput,
) -> str | NoInput:
    if fx is NoInput.blank and base is NoInput.blank:
        # base will not be inherited from a 2nd level inherited object, i.e.
        # from solver.fx, to preserve single currency instruments being defaulted
        # to their local currency.
        base_ = local_ccy
    elif isinstance(fx, (FXRates, FXForwards)) and base is NoInput.blank:
        base_ = fx.base
    else:
        base_ = base
    return base_


def _get_fx_maybe_from_solver(
    solver: Solver | NoInput,
    fx: float | FXRates | FXForwards | NoInput,
) -> float | FXRates | FXForwards | NoInput:
    if fx is NoInput.blank:
        if solver is NoInput.blank:
            fx_ = NoInput(0)
            # fx_ = 1.0
        elif solver is not NoInput.blank:
            if solver.fx is NoInput.blank:
                fx_ = NoInput(0)
                # fx_ = 1.0
            else:
                fx_ = solver.fx
    else:
        fx_ = fx
        if (
            solver is not NoInput.blank
            and solver.fx is not NoInput.blank
            and id(fx) != id(solver.fx)
        ):
            warnings.warn(
                "Solver contains an `fx` attribute but an `fx` argument has been "
                "supplied which will be used but is not the same. This can lead "
                "to calculation inconsistencies, mathematically.",
                UserWarning,
            )

    return fx_


def _get_curves_maybe_from_solver(
    curves_attr: Curve | str | list | NoInput,
    solver: Solver | NoInput,
    curves: Curve | str | list | NoInput,
) -> tuple:
    """
    Attempt to resolve curves as a variety of input types to a 4-tuple consisting of:
    (leg1 forecasting, leg1 discounting, leg2 forecasting, leg2 discounting)
    """
    if curves is NoInput.blank and curves_attr is NoInput.blank:
        # no data is available so consistently return a 4-tuple of no data
        return (NoInput(0), NoInput(0), NoInput(0), NoInput(0))
    elif curves is NoInput.blank:
        # set the `curves` input as that which is set as attribute at instrument init.
        curves = curves_attr

    if not isinstance(curves, (list, tuple)):
        # convert isolated value input to list
        curves = [curves]

    if solver is NoInput.blank:

        def check_curve(curve):
            if isinstance(curve, str):
                raise ValueError("`curves` must contain Curve, not str, if `solver` not given.")
            elif curve is None or curve is NoInput(0):
                return NoInput(0)
            elif isinstance(curve, dict):
                return {k: check_curve(v) for k, v in curve.items()}
            return curve

        curves_ = tuple(check_curve(curve) for curve in curves)
    else:
        try:
            curves_ = tuple(_get_curve_from_solver(curve, solver) for curve in curves)
        except KeyError as e:
            raise ValueError(
                "`curves` must contain str curve `id` s existing in `solver` "
                "(or its associated `pre_solvers`).\n"
                f"The sought id was: '{e.args[0]}'.\n"
                f"The available ids are {list(solver.pre_curves.keys())}.",
            )

    if len(curves_) == 1:
        curves_ *= 4
    elif len(curves_) == 2:
        curves_ *= 2
    elif len(curves_) == 3:
        curves_ += (curves_[1],)
    elif len(curves_) > 4:
        raise ValueError("Can only supply a maximum of 4 `curves`.")

    return curves_


def _get_curves_fx_and_base_maybe_from_solver(
    curves_attr: Curve | str | list | None,
    solver: Solver | None,
    curves: Curve | str | list | None,
    fx: float | FXRates | FXForwards | None,
    base: str | None,
    local_ccy: str | None,
) -> tuple:
    """
    Parses the ``solver``, ``curves``, ``fx`` and ``base`` arguments in combination.

    Parameters
    ----------
    curves_attr
        The curves attribute attached to the class.
    solver
        The solver argument passed in the outer method.
    curves
        The curves argument passed in the outer method.
    fx
        The fx argument agrument passed in the outer method.

    Returns
    -------
    tuple : (leg1 forecasting, leg1 discounting, leg2 forecasting, leg2 discounting), fx, base

    Notes
    -----
    If only one curve is given this is used as all four curves.

    If two curves are given the forecasting curve is used as the forecasting
    curve on both legs and the discounting curve is used as the discounting
    curve for both legs.

    If three curves are given the single discounting curve is used as the
    discounting curve for both legs.
    """
    # First process `base`.
    base_ = _get_base_maybe_from_fx(fx, base, local_ccy)
    # Second process `fx`
    fx_ = _get_fx_maybe_from_solver(solver, fx)
    # Third process `curves`
    curves_ = _get_curves_maybe_from_solver(curves_attr, solver, curves)
    return curves_, fx_, base_


def _get_vol_maybe_from_solver(
    vol_attr: DualTypes | str | FXVols | NoInput,
    vol: DualTypes | str | FXVols | NoInput,
    solver: Solver | NoInput,
):
    """
    Try to retrieve a general vol input from a solver or the default vol object associated with
    instrument.

    Parameters
    ----------
    vol_attr: DualTypes, str or FXDeltaVolSmile
        The vol attribute associated with the object at initialisation.
    vol: DualTypes, str of FXDeltaVolSMile
        The specific vol argument supplied at price time. Will take precendence.
    solver: Solver, optional
        A solver object

    Returns
    -------
    DualTypes, FXDeltaVolSmile or NoInput.blank
    """
    if vol is None:  # capture blank user input and reset
        vol = NoInput(0)

    if vol is NoInput.blank and vol_attr is NoInput.blank:
        return NoInput(0)
    elif vol is NoInput.blank:
        vol = vol_attr

    if solver is NoInput.blank:
        if isinstance(vol, str):
            raise ValueError(
                "String `vol` ids require a `solver` to be mapped. No `solver` provided.",
            )
        return vol
    elif isinstance(vol, (float, Dual, Dual2)):
        return vol
    elif isinstance(vol, str):
        return solver.pre_curves[vol]
    else:  # vol is a Smile or Surface - check that it is in the Solver
        try:
            # it is a safeguard to load curves from solvers when a solver is
            # provided and multiple curves might have the same id
            _ = solver.pre_curves[vol.id]
            if id(_) != id(vol):  # Python id() is a memory id, not a string label id.
                raise ValueError(
                    "A ``vol`` object has been supplied which has the same "
                    f"`id` ('{vol.id}'),\nas one of those available as part of the "
                    "Solver's collection but is not the same object.\n"
                    "This is ambiguous and may lead to erroneous prices.\n",
                )
            return _
        except AttributeError:
            raise AttributeError(
                "`vol` has no attribute `id`, likely it not a valid object, got: "
                f"{vol}.\nSince a solver is provided have you missed labelling the `vol` "
                f"of the instrument or supplying `vol` directly?",
            )
        except KeyError:
            if defaults.curve_not_in_solver == "ignore":
                return vol
            elif defaults.curve_not_in_solver == "warn":
                warnings.warn("`vol` not found in `solver`.", UserWarning)
                return vol
            else:
                raise ValueError("`vol` must be in `solver`.")


class Sensitivities:
    """
    Base class to add risk sensitivity calculations to an object with an ``npv()``
    method.
    """

    def delta(
        self,
        curves: Curve | str | list | NoInput = NoInput(0),
        solver: Solver | NoInput = NoInput(0),
        fx: FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
        local: bool = False,
        **kwargs,
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
        if solver is NoInput.blank:
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
        return solver.delta(npv, base_, fx_)

    def exo_delta(
        self,
        vars: list[str],
        curves: Curve | str | list | NoInput = NoInput(0),
        solver: Solver | NoInput = NoInput(0),
        fx: FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
        local: bool = False,
        vars_scalar: list[float] | NoInput = NoInput(0),
        vars_labels: list[str] | NoInput = NoInput(0),
        **kwargs,
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

        if solver is NoInput.blank:
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
            npv=npv, vars=vars, base=base_, fx=fx_, vars_scalar=vars_scalar, vars_labels=vars_labels
        )

    def gamma(
        self,
        curves: Curve | str | list | NoInput = NoInput(0),
        solver: Solver | NoInput = NoInput(0),
        fx: FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
        local: bool = False,
        **kwargs,
    ):
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
        if solver is NoInput.blank:
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
        if fx_ is not NoInput.blank:
            _ad2 = fx_._ad
            fx_._set_ad_order(2)

        _ad1 = solver._ad
        solver._set_ad_order(2)

        npv = self.npv(curves, solver, fx_, base_, local=True, **kwargs)
        grad_s_sT_P = solver.gamma(npv, base_, fx_)

        # reset original order
        if fx_ is not NoInput.blank:
            fx_._set_ad_order(_ad2)
        solver._set_ad_order(_ad1)

        return grad_s_sT_P

    def cashflows_table(
        self,
        curves: Curve | str | list | NoInput = NoInput(0),
        solver: Solver | NoInput = NoInput(0),
        fx: float | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
        **kwargs,
    ):
        """
        Aggregate the values derived from a :meth:`~rateslib.instruments.BaseMixin.cashflows`
        method on an *Instrument*.

        Parameters
        ----------
        curves : CurveType, str or list of such, optional
            Argument input to the underlying ``cashflows`` method of the *Instrument*.
        solver : Solver, optional
            Argument input to the underlying ``cashflows`` method of the *Instrument*.
        fx : float, FXRates, FXForwards, optional
            Argument input to the underlying ``cashflows`` method of the *Instrument*.
        base : str, optional
            Argument input to the underlying ``cashflows`` method of the *Instrument*.
        kwargs : dict
            Additional arguments input the underlying ``cashflows`` method of the *Instrument*.

        Returns
        -------
        DataFrame
        """
        cashflows = self.cashflows(curves, solver, fx, base, **kwargs)
        cashflows = cashflows[
            [
                defaults.headers["currency"],
                defaults.headers["collateral"],
                defaults.headers["payment"],
                defaults.headers["cashflow"],
            ]
        ]
        _ = cashflows.groupby(
            [
                defaults.headers["currency"],
                defaults.headers["collateral"],
                defaults.headers["payment"],
            ],
            dropna=False,
        )
        _ = _.sum().unstack([0, 1]).droplevel(0, axis=1)
        _.columns.names = ["local_ccy", "collateral_ccy"]
        _.index.names = ["payment"]
        _ = _.sort_index(ascending=True, axis=0).fillna(0.0)
        return _


class BaseMixin:
    _fixed_rate_mixin = False
    _float_spread_mixin = False
    _leg2_fixed_rate_mixin = False
    _leg2_float_spread_mixin = False
    _index_base_mixin = False
    _leg2_index_base_mixin = False
    _rate_scalar = 1.0

    @property
    def fixed_rate(self):
        """
        float or None : If set will also set the ``fixed_rate`` of the contained
        leg1.

        .. note::
           ``fixed_rate``, ``float_spread``, ``leg2_fixed_rate`` and
           ``leg2_float_spread`` are attributes only applicable to certain
           ``Instruments``. *AttributeErrors* are raised if calling or setting these
           is invalid.

        """
        return self._fixed_rate

    @fixed_rate.setter
    def fixed_rate(self, value):
        if not self._fixed_rate_mixin:
            raise AttributeError("Cannot set `fixed_rate` for this Instrument.")
        self._fixed_rate = value
        self.leg1.fixed_rate = value

    @property
    def leg2_fixed_rate(self):
        """
        float or None : If set will also set the ``fixed_rate`` of the contained
        leg2.
        """
        return self._leg2_fixed_rate

    @leg2_fixed_rate.setter
    def leg2_fixed_rate(self, value):
        if not self._leg2_fixed_rate_mixin:
            raise AttributeError("Cannot set `leg2_fixed_rate` for this Instrument.")
        self._leg2_fixed_rate = value
        self.leg2.fixed_rate = value

    @property
    def float_spread(self):
        """
        float or None : If set will also set the ``float_spread`` of contained
        leg1.
        """
        return self._float_spread

    @float_spread.setter
    def float_spread(self, value):
        if not self._float_spread_mixin:
            raise AttributeError("Cannot set `float_spread` for this Instrument.")
        self._float_spread = value
        self.leg1.float_spread = value
        # if getattr(self, "_float_mixin_leg", None) is NoInput.blank:
        #     self.leg1.float_spread = value
        # else:
        #     # allows fixed_rate and float_rate to exist simultaneously for diff legs.
        #     leg = getattr(self, "_float_mixin_leg", None)
        #     getattr(self, f"leg{leg}").float_spread = value

    @property
    def leg2_float_spread(self):
        """
        float or None : If set will also set the ``float_spread`` of contained
        leg2.
        """
        return self._leg2_float_spread

    @leg2_float_spread.setter
    def leg2_float_spread(self, value):
        if not self._leg2_float_spread_mixin:
            raise AttributeError("Cannot set `leg2_float_spread` for this Instrument.")
        self._leg2_float_spread = value
        self.leg2.float_spread = value

    @property
    def index_base(self):
        """
        float or None : If set will also set the ``index_base`` of the contained
        leg1.

        .. note::
           ``index_base`` and ``leg2_index_base`` are attributes only applicable to certain
           ``Instruments``. *AttributeErrors* are raised if calling or setting these
           is invalid.

        """
        return self._index_base

    @index_base.setter
    def index_base(self, value):
        if not self._index_base_mixin:
            raise AttributeError("Cannot set `index_base` for this Instrument.")
        self._index_base = value
        self.leg1.index_base = value

    @property
    def leg2_index_base(self):
        """
        float or None : If set will also set the ``index_base`` of the contained
        leg1.

        .. note::
           ``index_base`` and ``leg2_index_base`` are attributes only applicable to certain
           ``Instruments``. *AttributeErrors* are raised if calling or setting these
           is invalid.

        """
        return self._leg2_index_base

    @leg2_index_base.setter
    def leg2_index_base(self, value):
        if not self._leg2_index_base_mixin:
            raise AttributeError("Cannot set `leg2_index_base` for this Instrument.")
        self._leg2_index_base = value
        self.leg2.index_base = value

    @abc.abstractmethod
    def analytic_delta(self, *args, leg=1, **kwargs):
        """
        Return the analytic delta of a leg of the derivative object.

        Parameters
        ----------
        args :
            Required positional arguments supplied to
            :meth:`BaseLeg.analytic_delta<rateslib.legs.BaseLeg.analytic_delta>`.
        leg : int in [1, 2]
            The leg identifier of which to take the analytic delta.
        kwargs :
            Required Keyword arguments supplied to
            :meth:`BaseLeg.analytic_delta()<rateslib.legs.BaseLeg.analytic_delta>`.

        Returns
        -------
        float, Dual, Dual2

        Examples
        --------
        .. ipython:: python
           :suppress:

           from rateslib import Curve, FXRates, IRS, dt

        .. ipython:: python

           curve = Curve({dt(2021,1,1): 1.00, dt(2025,1,1): 0.83}, id="SONIA")
           fxr = FXRates({"gbpusd": 1.25}, base="usd")

        .. ipython:: python

           irs = IRS(
               effective=dt(2022, 1, 1),
               termination="6M",
               frequency="Q",
               currency="gbp",
               notional=1e9,
               fixed_rate=5.0,
           )
           irs.analytic_delta(curve, curve)
           irs.analytic_delta(curve, curve, fxr)
           irs.analytic_delta(curve, curve, fxr, "gbp")
        """
        return getattr(self, f"leg{leg}").analytic_delta(*args, **kwargs)

    @abc.abstractmethod
    def cashflows(
        self,
        curves: Curve | str | list | NoInput = NoInput(0),
        solver: Solver | NoInput = NoInput(0),
        fx: float | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
    ):
        """
        Return the properties of all legs used in calculating cashflows.

        Parameters
        ----------
        curves : CurveType, str or list of such, optional
            A single :class:`~rateslib.curves.Curve`,
            :class:`~rateslib.curves.LineCurve` or id or a
            list of such. A list defines the following curves in the order:

            - Forecasting :class:`~rateslib.curves.Curve` or
              :class:`~rateslib.curves.LineCurve` for ``leg1``.
            - Discounting :class:`~rateslib.curves.Curve` for ``leg1``.
            - Forecasting :class:`~rateslib.curves.Curve` or
              :class:`~rateslib.curves.LineCurve` for ``leg2``.
            - Discounting :class:`~rateslib.curves.Curve` for ``leg2``.
        solver : Solver, optional
            The numerical :class:`~rateslib.solver.Solver` that constructs
            ``Curves`` from calibrating instruments.
        fx : float, FXRates, FXForwards, optional
            The immediate settlement FX rate that will be used to convert values
            into another currency. A given `float` is used directly. If giving a
            :class:`~rateslib.fx.FXRates` or :class:`~rateslib.fx.FXForwards` object,
            converts from local currency into ``base``.
        base : str, optional
            The base currency to convert cashflows into (3-digit code).
            Only used if ``fx`` is an :class:`~rateslib.fx.FXRates` or
            :class:`~rateslib.fx.FXForwards` object. If not given defaults
            to ``fx.base``.

        Returns
        -------
        DataFrame

        Notes
        -----
        If **only one curve** is given this is used as all four curves.

        If **two curves** are given the forecasting curve is used as the forecasting
        curve on both legs and the discounting curve is used as the discounting
        curve for both legs.

        If **three curves** are given the single discounting curve is used as the
        discounting curve for both legs.

        Examples
        --------
        .. ipython:: python

           irs.cashflows([curve], fx=fxr)
        """
        curves, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            fx,
            base,
            self.leg1.currency,
        )

        df1 = self.leg1.cashflows(curves[0], curves[1], fx_, base_)
        df2 = self.leg2.cashflows(curves[2], curves[3], fx_, base_)
        # filter empty or all NaN
        dfs_filtered = [_ for _ in [df1, df2] if not (_.empty or isna(_).all(axis=None))]

        with warnings.catch_warnings():
            # TODO: pandas 2.1.0 has a FutureWarning for concatenating DataFrames with Null entries
            warnings.filterwarnings("ignore", category=FutureWarning)
            _ = concat(dfs_filtered, keys=["leg1", "leg2"])
        return _

    @abc.abstractmethod
    def npv(
        self,
        curves: Curve | str | list | NoInput = NoInput(0),
        solver: Solver | NoInput = NoInput(0),
        fx: float | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
        local: bool = False,
    ):
        """
        Return the NPV of the derivative object by summing legs.

        Parameters
        ----------
        curves : Curve, LineCurve, str or list of such
            A single :class:`~rateslib.curves.Curve`,
            :class:`~rateslib.curves.LineCurve` or id or a
            list of such. A list defines the following curves in the order:

            - Forecasting :class:`~rateslib.curves.Curve` or
              :class:`~rateslib.curves.LineCurve` for ``leg1``.
            - Discounting :class:`~rateslib.curves.Curve` for ``leg1``.
            - Forecasting :class:`~rateslib.curves.Curve` or
              :class:`~rateslib.curves.LineCurve` for ``leg2``.
            - Discounting :class:`~rateslib.curves.Curve` for ``leg2``.
        solver : Solver, optional
            The numerical :class:`~rateslib.solver.Solver` that constructs
            ``Curves`` from calibrating instruments.
        fx : float, FXRates, FXForwards, optional
            The immediate settlement FX rate that will be used to convert values
            into another currency. A given `float` is used directly. If giving a
            :class:`~rateslib.fx.FXRates` or :class:`~rateslib.fx.FXForwards` object,
            converts from local currency into ``base``.
        base : str, optional
            The base currency to convert cashflows into (3-digit code).
            Only used if ``fx`` is an :class:`~rateslib.fx.FXRates` or
            :class:`~rateslib.fx.FXForwards` object. If not given defaults
            to ``fx.base``.
        local : bool, optional
            If `True` will return a dict identifying NPV by local currencies on each
            leg. Useful for multi-currency derivatives and for ensuring risk
            sensitivities are allocated to local currencies without conversion.

        Returns
        -------
        float, Dual or Dual2, or dict of such.

        Notes
        -----
        If **only one curve** is given this is used as all four curves.

        If **two curves** are given the forecasting curve is used as the forecasting
        curve on both legs and the discounting curve is used as the discounting
        curve for both legs.

        If **three curves** are given the single discounting curve is used as the
        discounting curve for both legs.

        Examples
        --------
        .. ipython:: python

           irs.npv(curve)
           irs.npv([curve], fx=fxr)
           irs.npv([curve], fx=fxr, base="gbp")
        """
        curves, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            fx,
            base,
            self.leg1.currency,
        )
        leg1_npv = self.leg1.npv(curves[0], curves[1], fx_, base_, local)
        leg2_npv = self.leg2.npv(curves[2], curves[3], fx_, base_, local)
        if local:
            return {
                k: leg1_npv.get(k, 0) + leg2_npv.get(k, 0) for k in set(leg1_npv) | set(leg2_npv)
            }
        else:
            return leg1_npv + leg2_npv

    @abc.abstractmethod
    def rate(self, *args, **kwargs):
        """
        Return the `rate` or typical `price` for a derivative instrument.

        Returns
        -------
        Dual

        Notes
        -----
        This method must be implemented for instruments to function effectively in
        :class:`Solver` iterations.
        """
        pass  # pragma: no cover

    def __repr__(self):
        return f"<rl.{type(self).__name__} at {hex(id(self))}>"


def _get(kwargs: dict, leg: int = 1, filter=()):
    """
    A parser to return kwarg dicts for relevant legs.
    Internal structuring only.
    Will return kwargs relevant to leg1 OR leg2.
    Does not return keys that are specified in the filter.
    """
    if leg == 1:
        _ = {k: v for k, v in kwargs.items() if "leg2" not in k and k not in filter}
    else:
        _ = {k[5:]: v for k, v in kwargs.items() if "leg2_" in k and k not in filter}
    return _


def _push(spec: str | None, kwargs: dict):
    """
    Push user specified kwargs to a default specification.
    Values from the `spec` dict will not overwrite specific user values already in `kwargs`.
    """
    if spec is NoInput.blank:
        return kwargs
    else:
        try:
            spec_kwargs = defaults.spec[spec.lower()]
        except KeyError:
            raise ValueError(f"Given `spec`, '{spec}', cannot be found in defaults.")

        user = {k: v for k, v in kwargs.items() if not isinstance(v, NoInput)}
        return {**kwargs, **spec_kwargs, **user}


def _update_not_noinput(base_kwargs, new_kwargs):
    """
    Update the `base_kwargs` with `new_kwargs` (user values) unless those new values are NoInput.
    """
    updaters = {
        k: v for k, v in new_kwargs.items() if k not in base_kwargs or not isinstance(v, NoInput)
    }
    return {**base_kwargs, **updaters}


def _update_with_defaults(base_kwargs, default_kwargs):
    """
    Update the `base_kwargs` with `default_kwargs` if the values are NoInput.blank.
    """
    updaters = {
        k: v
        for k, v in default_kwargs.items()
        if k in base_kwargs and base_kwargs[k] is NoInput.blank
    }
    return {**base_kwargs, **updaters}


def _inherit_or_negate(kwargs: dict, ignore_blank=False):
    """Amend the values of leg2 kwargs if they are defaulted to inherit or negate from leg1."""

    def _replace(k, v):
        # either inherit or negate the value in leg2 from that in leg1
        if "leg2_" in k:
            if not isinstance(v, NoInput):
                return v  # do nothing if the attribute is an input

            try:
                leg1_v = kwargs[k[5:]]
            except KeyError:
                return v

            if leg1_v is NoInput.blank:
                if ignore_blank:
                    return v  # this allows an inheritor or negator to be called a second time
                else:
                    return NoInput(0)

            if v is NoInput(-1):
                return leg1_v * -1.0
            elif v is NoInput(1):
                return leg1_v
        return v  # do nothing to leg1 attributes

    return {k: _replace(k, v) for k, v in kwargs.items()}


def _lower(val: str | NoInput):
    if isinstance(val, str):
        return val.lower()
    return val


def _upper(val: str | NoInput):
    if isinstance(val, str):
        return val.upper()
    return val


def _composit_fixings_table(df_result, df):
    """
    Add a DataFrame to an existing fixings table by extending or adding to relevant columns.

    Parameters
    ----------
    df_result: The main DataFrame that will be updated
    df: The incoming DataFrame with new data to merge

    Returns
    -------
    DataFrame
    """
    # reindex the result DataFrame
    if df_result.empty:
        return df
    else:
        df_result = df_result.reindex(index=df_result.index.union(df.index))

    # update existing columns with missing data from the new available data
    for c in [c for c in df.columns if c in df_result.columns and c[1] in ["dcf", "rates"]]:
        df_result[c] = df_result[c].combine_first(df[c])

    # merge by addition existing values with missing filled to zero
    m = [c for c in df.columns if c in df_result.columns and c[1] in ["notional", "risk"]]
    if len(m) > 0:
        df_result[m] = df_result[m].add(df[m], fill_value=0.0)

    # append new columns without additional calculation
    a = [c for c in df.columns if c not in df_result.columns]
    if len(a) > 0:
        df_result[a] = df[a]

    # df_result.columns = MultiIndex.from_tuples(df_result.columns)
    return df_result
