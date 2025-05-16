from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from pandas import DataFrame

from rateslib import defaults
from rateslib.curves import Curve
from rateslib.curves._parsers import _disc_required_maybe_from_curve
from rateslib.default import NoInput, _drb
from rateslib.dual.utils import _dual_float
from rateslib.instruments.base import BaseDerivative
from rateslib.instruments.utils import (
    _composit_fixings_table,
    _get,
    _get_curves_fx_and_base_maybe_from_solver,
    _update_not_noinput,
)
from rateslib.legs import (
    FixedLeg,
    FloatLeg,
    ZeroFixedLeg,
    ZeroFloatLeg,
)
from rateslib.periods.utils import (
    _get_fx_and_base,
    _maybe_local,
    _trim_df_by_index,
)

if TYPE_CHECKING:
    from typing import Any, NoReturn

    from rateslib.typing import (
        FX_,
        NPV,
        CurveOption_,
        Curves_,
        DualTypes,
        FixedPeriod,
        FixingsRates_,
        FloatPeriod,
        Solver_,
    )

# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# This code cannot be installed or executed on a corporate computer without a paid licence extension
# Contact info at rateslib.com if this code is observed outside its intended sphere of use.


class IRS(BaseDerivative):
    """
    Create an interest rate swap composing a :class:`~rateslib.legs.FixedLeg`
    and a :class:`~rateslib.legs.FloatLeg`.

    Parameters
    ----------
    args : dict
        Required positional args to :class:`BaseDerivative`.
    fixed_rate : float or None
        The fixed rate applied to the :class:`~rateslib.legs.FixedLeg`. If `None`
        will be set to mid-market when curves are provided.
    leg2_float_spread : float, optional
        The spread applied to the :class:`~rateslib.legs.FloatLeg`. Can be set to
        `None` and designated
        later, perhaps after a mid-market spread for all periods has been calculated.
    leg2_spread_compound_method : str, optional
        The method to use for adding a floating spread to compounded rates. Available
        options are `{"none_simple", "isda_compounding", "isda_flat_compounding"}`.
    leg2_fixings : float, list, or Series optional
        If a float scalar, will be applied as the determined fixing for the first
        period. If a list of *n* fixings will be used as the fixings for the first *n*
        periods. If any sublist of length *m* is given, is used as the first *m* RFR
        fixings for that :class:`~rateslib.periods.FloatPeriod`. If a datetime
        indexed ``Series`` will use the fixings that are available in that object,
        and derive the rest from the ``curve``.
    leg2_fixing_method : str, optional
        The method by which floating rates are determined, set by default. See notes.
    leg2_method_param : int, optional
        A parameter that is used for the various ``fixing_method`` s. See notes.
    kwargs : dict
        Required keyword arguments to :class:`BaseDerivative`.

    Notes
    ------
    The various different ``leg2_fixing_methods``, which describe how an
    individual *FloatPeriod* calculates its *rate*, are
    fully documented in the notes for the :class:`~rateslib.periods.FloatPeriod`.
    These configurations provide the mechanics to differentiate between IBOR swaps, and
    OISs with different mechanisms such as *payment delay*, *observation shift*,
    *lockout*, and/or *averaging*.
    Similarly some information is provided in that same link regarding
    ``leg2_fixings``, but a cookbook article is also produced for
    :ref:`working with fixings <cook-fixings-doc>`.

    Examples
    --------
    Construct a curve to price the example.

    .. ipython:: python

       usd = Curve(
           nodes={
               dt(2022, 1, 1): 1.0,
               dt(2023, 1, 1): 0.965,
               dt(2024, 1, 1): 0.94
           },
           id="usd"
       )

    Create the IRS, and demonstrate the :meth:`~rateslib.instruments.IRS.rate`,
    :meth:`~rateslib.instruments.IRS.npv`,
    :meth:`~rateslib.instruments.IRS.analytic_delta`, and
    :meth:`~rateslib.instruments.IRS.spread`.

    .. ipython:: python

       irs = IRS(
           effective=dt(2022, 1, 1),
           termination="18M",
           frequency="A",
           calendar="nyc",
           currency="usd",
           fixed_rate=3.269,
           convention="Act360",
           notional=100e6,
           curves=["usd"],
       )
       irs.rate(curves=usd)
       irs.npv(curves=usd)
       irs.analytic_delta(curve=usd)
       irs.spread(curves=usd)

    A DataFrame of :meth:`~rateslib.instruments.IRS.cashflows`.

    .. ipython:: python

       irs.cashflows(curves=usd)

    For accurate sensitivity calculations; :meth:`~rateslib.instruments.IRS.delta`
    and :meth:`~rateslib.instruments.IRS.gamma`, construct a curve model.

    .. ipython:: python

       sofr_kws = dict(
           effective=dt(2022, 1, 1),
           frequency="A",
           convention="Act360",
           calendar="nyc",
           currency="usd",
           curves=["usd"]
       )
       instruments = [
           IRS(termination="1Y", **sofr_kws),
           IRS(termination="2Y", **sofr_kws),
       ]
       solver = Solver(
           curves=[usd],
           instruments=instruments,
           s=[3.65, 3.20],
           instrument_labels=["1Y", "2Y"],
           id="sofr",
       )
       irs.delta(solver=solver)
       irs.gamma(solver=solver)
    """

    _fixed_rate_mixin = True
    _leg2_float_spread_mixin = True

    leg1: FixedLeg
    leg2: FloatLeg

    def __init__(
        self,
        *args: Any,
        fixed_rate: DualTypes | NoInput = NoInput(0),
        leg2_float_spread: DualTypes | NoInput = NoInput(0),
        leg2_spread_compound_method: str | NoInput = NoInput(0),
        leg2_fixings: FixingsRates_ = NoInput(0),  # type: ignore[type-var]
        leg2_fixing_method: str | NoInput = NoInput(0),
        leg2_method_param: int | NoInput = NoInput(0),
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        user_kwargs = dict(
            fixed_rate=fixed_rate,
            leg2_float_spread=leg2_float_spread,
            leg2_spread_compound_method=leg2_spread_compound_method,
            leg2_fixings=leg2_fixings,
            leg2_fixing_method=leg2_fixing_method,
            leg2_method_param=leg2_method_param,
        )
        self.kwargs: dict[str, Any] = _update_not_noinput(self.kwargs, user_kwargs)

        self._fixed_rate = fixed_rate
        self._leg2_float_spread = leg2_float_spread
        self.leg1 = FixedLeg(**_get(self.kwargs, leg=1))
        self.leg2 = FloatLeg(**_get(self.kwargs, leg=2))

    def _set_pricing_mid(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
    ) -> None:
        # the test for an unpriced IRS is that its fixed rate is not set.
        if isinstance(self.fixed_rate, NoInput):
            # set a fixed rate for the purpose of generic methods NPV will be zero.
            mid_market_rate = self.rate(curves, solver)
            self.leg1.fixed_rate = _dual_float(mid_market_rate)

    def analytic_delta(self, *args: Any, **kwargs: Any) -> DualTypes:
        """
        Return the analytic delta of a leg of the derivative object.

        See :meth:`BaseDerivative.analytic_delta`.
        """
        return super().analytic_delta(*args, **kwargs)

    def npv(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str | NoInput = NoInput(0),
        local: bool = False,
    ) -> NPV:
        """
        Return the NPV of the derivative by summing legs.

        See :meth:`BaseDerivative.npv`.
        """
        self._set_pricing_mid(curves, solver)
        return super().npv(curves, solver, fx, base, local)

    def rate(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str | NoInput = NoInput(0),
    ) -> DualTypes:
        """
        Return the mid-market rate of the IRS.

        Parameters
        ----------
        curves : Curve, str or list of such
            A single :class:`~rateslib.curves.Curve` or id or a list of such.
            A list defines the following curves in the order:

            - Forecasting :class:`~rateslib.curves.Curve` for floating leg.
            - Discounting :class:`~rateslib.curves.Curve` for both legs.
        solver : Solver, optional
            The numerical :class:`~rateslib.solver.Solver` that
            constructs :class:`~rateslib.curves.Curve` from calibrating instruments.

            .. note::

               The arguments ``fx`` and ``base`` are unused by single currency
               derivatives rates calculations.

        Returns
        -------
        float, Dual or Dual2

        Notes
        -----
        The arguments ``fx`` and ``base`` are unused by single currency derivatives
        rates calculations.
        """
        curves_, _, _ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            fx,
            base,
            self.leg1.currency,
        )
        leg2_npv: DualTypes = self.leg2.npv(curves_[2], curves_[3], local=False)  # type: ignore[assignment]
        return self.leg1._spread(-leg2_npv, curves_[0], curves_[1]) / 100
        # leg1_analytic_delta = self.leg1.analytic_delta(curves[0], curves[1])
        # return leg2_npv / (leg1_analytic_delta * 100)

    def cashflows(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str | NoInput = NoInput(0),
    ) -> DataFrame:
        """
        Return the properties of all legs used in calculating cashflows.

        See :meth:`BaseDerivative.cashflows`.
        """
        self._set_pricing_mid(curves, solver)
        return super().cashflows(curves, solver, fx, base)

    # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
    # Commercial use of this code, and/or copying and redistribution is prohibited.
    # Contact rateslib at gmail.com if this code is observed outside its intended sphere.

    def spread(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str | NoInput = NoInput(0),
    ) -> DualTypes:
        """
        Return the mid-market float spread (bps) required to equate to the fixed rate.

        Parameters
        ----------
        curves : Curve, str or list of such
            A single :class:`~rateslib.curves.Curve` or id or a list of such.
            A list defines the following curves in the order:

            - Forecasting :class:`~rateslib.curves.Curve` for floating leg.
            - Discounting :class:`~rateslib.curves.Curve` for both legs.
        solver : Solver, optional
            The numerical :class:`~rateslib.solver.Solver` that constructs
            :class:`~rateslib.curves.Curve` from calibrating instruments.

            .. note::

               The arguments ``fx`` and ``base`` are unused by single currency
               derivatives rates calculations.

        Returns
        -------
        float, Dual or Dual2

        Notes
        -----
        If the :class:`IRS` is specified without a ``fixed_rate`` this should always
        return the current ``leg2_float_spread`` value or zero since the fixed rate used
        for calculation is the implied rate including the current ``leg2_float_spread``
        parameter.

        Examples
        --------
        For the most common parameters this method will be exact.

        .. ipython:: python

           irs.spread(curves=usd)
           irs.leg2_float_spread = -6.948753
           irs.npv(curves=usd)

        When a non-linear spread compound method is used for float RFR legs this is
        an approximation, via second order Taylor expansion.

        .. ipython:: python

           irs = IRS(
               effective=dt(2022, 2, 15),
               termination=dt(2022, 8, 15),
               frequency="Q",
               convention="30e360",
               leg2_convention="Act360",
               leg2_fixing_method="rfr_payment_delay",
               leg2_spread_compound_method="isda_compounding",
               payment_lag=2,
               fixed_rate=2.50,
               leg2_float_spread=0,
               notional=50000000,
               currency="usd",
           )
           irs.spread(curves=usd)
           irs.leg2_float_spread = -111.060143
           irs.npv(curves=usd)
           irs.spread(curves=usd)

        The ``leg2_float_spread`` is determined through NPV differences. If the difference
        is small since the defined spread is already quite close to the solution the
        approximation is much more accurate. This is shown above where the second call
        to ``irs.spread`` is different to the previous call, albeit the difference
        is 1/10000th of a basis point.
        """
        irs_npv: DualTypes = self.npv(curves, solver, local=False)  # type: ignore[assignment]
        specified_spd: DualTypes = _drb(0.0, self.leg2.float_spread)
        curves_, _, _ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            fx,
            base,
            self.leg1.currency,
        )
        return self.leg2._spread(-irs_npv, curves_[2], curves_[3]) + specified_spd
        # leg2_analytic_delta = self.leg2.analytic_delta(curves[2], curves[3])
        # return irs_npv / leg2_analytic_delta + specified_spd

    def fixings_table(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str | NoInput = NoInput(0),
        approximate: bool = False,
        right: datetime | NoInput = NoInput(0),
    ) -> DataFrame:
        """
        Return a DataFrame of fixing exposures on the :class:`~rateslib.legs.FloatLeg`.

        Parameters
        ----------
        curves : Curve, str or list of such
            A single :class:`~rateslib.curves.Curve` or id or a list of such.
            A list defines the following curves in the order:

            - Forecasting :class:`~rateslib.curves.Curve` for floating leg.
            - Discounting :class:`~rateslib.curves.Curve` for both legs.

        solver : Solver, optional
            The numerical :class:`~rateslib.solver.Solver` that constructs
            :class:`~rateslib.curves.Curve` from calibrating instruments.

            .. note::

               The arguments ``fx`` and ``base`` are unused by single currency
               derivatives rates calculations.

        approximate : bool, optional
            Perform a calculation that is broadly 10x faster but potentially loses
            precision upto 0.1%.
        right : datetime, optional
            Only calculate fixing exposures upto and including this date.

        Returns
        -------
        DataFrame
        """
        curves_, _, _ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            NoInput(0),
            NoInput(0),
            self.leg1.currency,
        )
        return self.leg2.fixings_table(
            curve=curves_[2], approximate=approximate, disc_curve=curves_[3], right=right
        )


class STIRFuture(IRS):
    """
    Create a short term interest rate (STIR) future.

    Parameters
    ----------
    args : dict
        Required positional args to :class:`BaseDerivative`.
    price : float
        The traded price of the future. Defined as 100 minus the fixed rate.
    contracts : int
        The number of traded contracts.
    bp_value : float.
        The value of 1bp on the contract as specified by the exchange, e.g. SOFR 3M futures are
        $25 per bp. This is not the same as tick value where the tick size can be different across
        different futures.
    nominal : float
        The nominal value of the contract. E.g. SOFR 3M futures are $1mm. If not given will use the
        default notional.
    fixed_rate : float or None
        The fixed rate applied to the :class:`~rateslib.legs.FixedLeg`. If `None`
        will be set to mid-market when curves are provided.
    leg2_float_spread : float, optional
        The spread applied to the :class:`~rateslib.legs.FloatLeg`. Can be set to
        `None` and designated
        later, perhaps after a mid-market spread for all periods has been calculated.
    leg2_spread_compound_method : str, optional
        The method to use for adding a floating spread to compounded rates. Available
        options are `{"none_simple", "isda_compounding", "isda_flat_compounding"}`.
    leg2_fixings : float, list, or Series optional
        If a float scalar, will be applied as the determined fixing for the first
        period. If a list of *n* fixings will be used as the fixings for the first *n*
        periods. If any sublist of length *m* is given, is used as the first *m* RFR
        fixings for that :class:`~rateslib.periods.FloatPeriod`. If a datetime
        indexed ``Series`` will use the fixings that are available in that object,
        and derive the rest from the ``curve``.
    leg2_fixing_method : str, optional
        The method by which floating rates are determined, set by default. See notes.
    leg2_method_param : int, optional
        A parameter that is used for the various ``fixing_method`` s. See notes.
    kwargs : dict
        Required keyword arguments to :class:`BaseDerivative`.

    Examples
    --------
    Construct a curve to price the example.

    .. ipython:: python

       usd = Curve(
           nodes={
               dt(2022, 1, 1): 1.0,
               dt(2023, 1, 1): 0.965,
               dt(2024, 1, 1): 0.94
           },
           id="usd_stir"
       )

    Create the *STIRFuture*, and demonstrate the :meth:`~rateslib.instruments.STIRFuture.rate`,
    :meth:`~rateslib.instruments.STIRFuture.npv`,

    .. ipython:: python

       stir = STIRFuture(
            effective=dt(2022, 3, 16),
            termination=dt(2022, 6, 15),
            spec="usd_stir",
            curves=usd,
            price=99.50,
            contracts=10,
        )
       stir.rate(metric="price")
       stir.npv()

    """

    _fixed_rate_mixin = True
    _leg2_float_spread_mixin = True

    leg1: FixedLeg
    leg2: FloatLeg

    def __init__(
        self,
        *args: Any,
        price: float | NoInput = NoInput(0),
        contracts: int = 1,
        bp_value: float | NoInput = NoInput(0),
        nominal: float | NoInput = NoInput(0),
        leg2_float_spread: float | NoInput = NoInput(0),
        leg2_spread_compound_method: str | NoInput = NoInput(0),
        leg2_fixings: FixingsRates_ = NoInput(0),
        leg2_fixing_method: str | NoInput = NoInput(0),
        leg2_method_param: int | NoInput = NoInput(0),
        **kwargs: Any,
    ) -> None:
        nominal_: float = _drb(defaults.notional, nominal)
        # TODO this overwrite breaks positional arguments
        kwargs["notional"] = nominal_ * contracts * -1.0
        super(IRS, self).__init__(*args, **kwargs)  # call BaseDerivative.__init__()
        user_kwargs = dict(
            price=price,
            fixed_rate=NoInput(0) if isinstance(price, NoInput) else (100 - price),
            leg2_float_spread=leg2_float_spread,
            leg2_spread_compound_method=leg2_spread_compound_method,
            leg2_fixings=leg2_fixings,
            leg2_fixing_method=leg2_fixing_method,
            leg2_method_param=leg2_method_param,
            nominal=nominal_,
            bp_value=bp_value,
            contracts=contracts,
        )
        self.kwargs = _update_not_noinput(self.kwargs, user_kwargs)

        self._fixed_rate = self.kwargs["fixed_rate"]
        self._leg2_float_spread = leg2_float_spread
        self.leg1 = FixedLeg(
            **_get(self.kwargs, leg=1, filter=("price", "nominal", "bp_value", "contracts")),
        )
        self.leg2 = FloatLeg(**_get(self.kwargs, leg=2))

    def npv(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str | NoInput = NoInput(0),
        local: bool = False,
    ) -> NPV:
        """
        Return the NPV of the derivative by summing legs.

        See :meth:`BaseDerivative.npv`.
        """
        # the test for an unpriced IRS is that its fixed rate is not set.
        mid_price = self.rate(curves, solver, fx, base, metric="price")
        if isinstance(self.fixed_rate, NoInput):
            # set a fixed rate for the purpose of generic methods NPV will be zero.
            mid_rate = _dual_float(100 - mid_price)
            self.leg1.fixed_rate = mid_rate
            traded_price: DualTypes = 100.0 - mid_rate
        else:
            traded_price = 100 - self.fixed_rate
        _ = (mid_price - traded_price) * 100 * self.kwargs["contracts"] * self.kwargs["bp_value"]
        return _maybe_local(_, local, self.leg1.currency, fx, base)

    def rate(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str | NoInput = NoInput(0),
        metric: str = "rate",
    ) -> DualTypes:
        """
        Return the mid-market rate of the IRS.

        Parameters
        ----------
        curves : Curve, str or list of such
            A single :class:`~rateslib.curves.Curve` or id or a list of such.
            A list defines the following curves in the order:

            - Forecasting :class:`~rateslib.curves.Curve` for floating leg.
            - Discounting :class:`~rateslib.curves.Curve` for both legs.
        solver : Solver, optional
            The numerical :class:`~rateslib.solver.Solver` that
            constructs :class:`~rateslib.curves.Curve` from calibrating instruments.

            .. note::

               The arguments ``fx`` and ``base`` are unused by single currency
               derivatives rates calculations.
        metric : str in {"rate", "price"}
            The calculation metric that will be returned.

        Returns
        -------
        float, Dual or Dual2

        Notes
        -----
        The arguments ``fx`` and ``base`` are unused by single currency derivatives
        rates calculations.
        """
        curves_, _, _ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            fx,
            base,
            self.leg1.currency,
        )
        leg2_npv: DualTypes = self.leg2.npv(curves_[2], curves_[3], local=False)  # type: ignore[assignment]

        ret: DualTypes = self.leg1._spread(-leg2_npv, curves_[0], curves_[1]) / 100
        if metric.lower() == "rate":
            return ret
        elif metric.lower() == "price":
            return 100 - ret
        else:
            raise ValueError("`metric` must be in {'price', 'rate'}.")

    def analytic_delta(  # type: ignore[override]
        self,
        curve: CurveOption_ = NoInput(0),
        disc_curve: CurveOption_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str | NoInput = NoInput(0),
    ) -> DualTypes:
        """
        Return the analytic delta of the *STIRFuture*.

        See :meth:`BasePeriod.analytic_delta()<rateslib.periods.BasePeriod.analytic_delta>`.
        For *STIRFuture* this method requires no arguments.
        """
        fx_, base_ = _get_fx_and_base(self.kwargs["currency"], fx, base)
        _: DualTypes = fx_ * (-1.0 * self.kwargs["contracts"] * self.kwargs["bp_value"])
        return _

    def cashflows(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str | NoInput = NoInput(0),
    ) -> DataFrame:
        return DataFrame.from_records(
            [
                {
                    defaults.headers["type"]: type(self).__name__,
                    defaults.headers["stub_type"]: "Regular",
                    defaults.headers["currency"]: self.leg1.currency.upper(),
                    defaults.headers["a_acc_start"]: self.leg1.schedule.effective,
                    defaults.headers["a_acc_end"]: self.leg1.schedule.termination,
                    defaults.headers["payment"]: None,
                    defaults.headers["convention"]: "Exchange",
                    defaults.headers["dcf"]: _dual_float(self.leg1.notional)
                    / self.kwargs["nominal"]
                    * self.kwargs["bp_value"]
                    / 100.0,
                    defaults.headers["notional"]: _dual_float(self.leg1.notional),
                    defaults.headers["df"]: 1.0,
                    defaults.headers["collateral"]: self.leg1.currency.lower(),
                },
            ],
        )

    def spread(self) -> NoReturn:  # type: ignore[override]
        """
        Not implemented for *STIRFuture*.
        """
        raise NotImplementedError("`spread` method is not implemented on STIRFuture.")

    def fixings_table(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str | NoInput = NoInput(0),
        approximate: bool = False,
        right: datetime | NoInput = NoInput(0),
    ) -> DataFrame:
        """
        Return a DataFrame of fixing exposures on the :class:`~rateslib.legs.FloatLeg`.

        Parameters
        ----------
        curves : Curve, str or list of such
            A single :class:`~rateslib.curves.Curve` or id or a list of such.
            A list defines the following curves in the order:

            - Forecasting :class:`~rateslib.curves.Curve` for floating leg.
            - Discounting :class:`~rateslib.curves.Curve` for both legs.

        solver : Solver, optional
            The numerical :class:`~rateslib.solver.Solver` that constructs
            :class:`~rateslib.curves.Curve` from calibrating instruments.

            .. note::

               The arguments ``fx`` and ``base`` are unused by single currency
               derivatives rates calculations.

        approximate : bool, optional
            Perform a calculation that is broadly 10x faster but potentially loses
            precision upto 0.1%.
        right : datetime, optional
            Only calculate fixing exposures upto and including this date.

        Returns
        -------
        DataFrame
        """
        curves_, _, _ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            NoInput(0),
            NoInput(0),
            self.leg2.currency,
        )
        risk: float = -1.0 * self.kwargs["contracts"] * self.kwargs["bp_value"]
        df = self.leg2.fixings_table(
            curve=curves_[2], approximate=approximate, disc_curve=curves_[3]
        )

        # TODO: handle curves as dict. "id" is not available this is typing mismatch
        total_risk = df[(curves_[2].id, "risk")].sum()  # type: ignore[union-attr]
        df[[(curves_[2].id, "notional"), (curves_[2].id, "risk")]] *= risk / total_risk  # type: ignore[union-attr]
        return _trim_df_by_index(df, NoInput(0), right)


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# This code cannot be installed or executed on a corporate computer without a paid licence extension
# Contact info at rateslib.com if this code is observed outside its intended sphere of use.


class ZCS(BaseDerivative):
    """
    Create a zero coupon swap (ZCS) composing a :class:`~rateslib.legs.ZeroFixedLeg`
    and a :class:`~rateslib.legs.ZeroFloatLeg`.

    Parameters
    ----------
    args : dict
        Required positional args to :class:`BaseDerivative`.
    fixed_rate : float, optional
        The fixed rate applied to the :class:`~rateslib.legs.ZeroFixedLeg`. If not given
        will be set to mid-market when curves are provided.
    leg2_float_spread : float, optional
        The spread applied to the :class:`~rateslib.legs.FloatLeg`. Can be set to
        `None` and designated
        later, perhaps after a mid-market spread for all periods has been calculated.
    leg2_spread_compound_method : str, optional
        The method to use for adding a floating spread to compounded rates. Available
        options are `{"none_simple", "isda_compounding", "isda_flat_compounding"}`.
    leg2_fixings : float, list, or Series optional
        If a float scalar, will be applied as the determined fixing for the first
        period. If a list of *n* fixings will be used as the fixings for the first *n*
        periods. If any sublist of length *m* is given, is used as the first *m* RFR
        fixings for that :class:`~rateslib.periods.FloatPeriod`. If a datetime
        indexed ``Series`` will use the fixings that are available in that object,
        and derive the rest from the ``curve``.
    leg2_fixing_method : str, optional
        The method by which floating rates are determined, set by default. See notes.
    leg2_method_param : int, optional
        A parameter that is used for the various ``fixing_method`` s. See notes.
    kwargs : dict
        Required keyword arguments to :class:`BaseDerivative`.

    Notes
    -----
    For individual pricing formulae see the separate *Leg* type class definitions.

    Examples
    --------
    Construct a curve to price the example.

    .. ipython:: python

       usd = Curve(
           nodes={
               dt(2022, 1, 1): 1.0,
               dt(2027, 1, 1): 0.85,
               dt(2032, 1, 1): 0.70,
           },
           id="usd"
       )

    Create the ZCS, and demonstrate the :meth:`~rateslib.instruments.ZCS.rate`,
    :meth:`~rateslib.instruments.ZCS.npv`,
    :meth:`~rateslib.instruments.ZCS.analytic_delta`, and

    .. ipython:: python

       zcs = ZCS(
           effective=dt(2022, 1, 1),
           termination="10Y",
           frequency="Q",
           calendar="nyc",
           currency="usd",
           fixed_rate=4.0,
           convention="Act360",
           notional=100e6,
           curves=["usd"],
       )
       zcs.rate(curves=usd)
       zcs.npv(curves=usd)
       zcs.analytic_delta(curve=usd)

    A DataFrame of :meth:`~rateslib.instruments.ZCS.cashflows`.

    .. ipython:: python

       zcs.cashflows(curves=usd)

    For accurate sensitivity calculations; :meth:`~rateslib.instruments.ZCS.delta`
    and :meth:`~rateslib.instruments.ZCS.gamma`, construct a curve model.

    .. ipython:: python

       sofr_kws = dict(
           effective=dt(2022, 1, 1),
           frequency="A",
           convention="Act360",
           calendar="nyc",
           currency="usd",
           curves=["usd"]
       )
       instruments = [
           IRS(termination="5Y", **sofr_kws),
           IRS(termination="10Y", **sofr_kws),
       ]
       solver = Solver(
           curves=[usd],
           instruments=instruments,
           s=[3.40, 3.60],
           instrument_labels=["5Y", "10Y"],
           id="sofr",
       )
       zcs.delta(solver=solver)
       zcs.gamma(solver=solver)
    """

    _fixed_rate_mixin = True
    _leg2_float_spread_mixin = True

    leg1: ZeroFixedLeg
    leg2: ZeroFloatLeg

    def __init__(
        self,
        *args: Any,
        fixed_rate: float | NoInput = NoInput(0),
        leg2_float_spread: float | NoInput = NoInput(0),
        leg2_spread_compound_method: str | NoInput = NoInput(0),
        leg2_fixings: FixingsRates_ = NoInput(0),
        leg2_fixing_method: str | NoInput = NoInput(0),
        leg2_method_param: int | NoInput = NoInput(0),
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        user_kwargs = dict(
            fixed_rate=fixed_rate,
            leg2_float_spread=leg2_float_spread,
            leg2_spread_compound_method=leg2_spread_compound_method,
            leg2_fixings=leg2_fixings,
            leg2_fixing_method=leg2_fixing_method,
            leg2_method_param=leg2_method_param,
        )
        self.kwargs: dict[str, Any] = _update_not_noinput(self.kwargs, user_kwargs)
        self._fixed_rate = fixed_rate
        self._leg2_float_spread = leg2_float_spread
        self.leg1 = ZeroFixedLeg(**_get(self.kwargs, leg=1))
        self.leg2 = ZeroFloatLeg(**_get(self.kwargs, leg=2))

    def analytic_delta(self, *args: Any, **kwargs: Any) -> DualTypes:
        """
        Return the analytic delta of a leg of the derivative object.

        See
        :meth:`BaseDerivative.analytic_delta<rateslib.instruments.BaseDerivative.analytic_delta>`.
        """
        return super().analytic_delta(*args, **kwargs)

    def _set_pricing_mid(self, curves: Curves_, solver: Solver_) -> None:
        if isinstance(self.fixed_rate, NoInput):
            # set a fixed rate for the purpose of pricing NPV, which should be zero.
            mid_market_rate = self.rate(curves, solver)
            self.leg1.fixed_rate = _dual_float(mid_market_rate)

    def npv(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str | NoInput = NoInput(0),
        local: bool = False,
    ) -> NPV:
        """
        Return the NPV of the derivative by summing legs.

        See :meth:`BaseDerivative.npv`.
        """
        self._set_pricing_mid(curves, solver)
        return super().npv(curves, solver, fx, base, local)

    def rate(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str | NoInput = NoInput(0),
    ) -> DualTypes:
        """
        Return the mid-market rate of the ZCS.

        Parameters
        ----------
        curves : Curve, str or list of such
            A single :class:`~rateslib.curves.Curve` or id or a list of such.
            A list defines the following curves in the order:

            - Forecasting :class:`~rateslib.curves.Curve` for floating leg.
            - Discounting :class:`~rateslib.curves.Curve` for both legs.
        solver : Solver, optional
            The numerical :class:`~rateslib.solver.Solver` that
            constructs :class:`~rateslib.curves.Curve` from calibrating instruments.

            .. note::

               The arguments ``fx`` and ``base`` are unused by single currency
               derivatives rates calculations.

        Returns
        -------
        float, Dual or Dual2

        Notes
        -----
        The arguments ``fx`` and ``base`` are unused by single currency derivatives
        rates calculations.

        The *'irr'* ``fixed_rate`` defines a cashflow by:

        .. math::

           -notional * ((1 + irr / f)^{f \\times dcf} - 1)

        where :math:`f` is associated with the compounding frequency.
        """
        curves_, _, _ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            fx,
            base,
            self.leg1.currency,
        )
        leg2_npv: DualTypes = self.leg2.npv(curves_[2], curves_[3], local=False)  # type: ignore[assignment]
        ret: DualTypes = self.leg1._spread(-leg2_npv, curves_[0], curves_[1]) / 100
        return ret

    def cashflows(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str | NoInput = NoInput(0),
    ) -> DataFrame:
        """
        Return the properties of all legs used in calculating cashflows.

        See :meth:`BaseDerivative.cashflows`.
        """
        self._set_pricing_mid(curves, solver)
        return super().cashflows(curves, solver, fx, base)

    def fixings_table(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str | NoInput = NoInput(0),
        approximate: bool = False,
        right: datetime | NoInput = NoInput(0),
    ) -> DataFrame:
        """
        Return a DataFrame of fixing exposures on the :class:`~rateslib.legs.ZeroFloatLeg`.

        Parameters
        ----------
        curves : Curve, str or list of such
            A single :class:`~rateslib.curves.Curve` or id or a list of such.
            A list defines the following curves in the order:

            - Forecasting :class:`~rateslib.curves.Curve` for floating leg.
            - Discounting :class:`~rateslib.curves.Curve` for both legs.

        solver : Solver, optional
            The numerical :class:`~rateslib.solver.Solver` that constructs
            :class:`~rateslib.curves.Curve` from calibrating instruments.

            .. note::

               The arguments ``fx`` and ``base`` are unused by single currency
               derivatives rates calculations.

        approximate : bool, optional
            Perform a calculation that is broadly 10x faster but potentially loses
            precision upto 0.1%.
        right : datetime, optional
            Only calculate fixing exposures upto and including this date.

        Returns
        -------
        DataFrame
        """
        curves_, _, _ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            NoInput(0),
            NoInput(0),
            self.leg1.currency,
        )
        return self.leg2.fixings_table(
            curve=curves_[2], approximate=approximate, disc_curve=curves_[3], right=right
        )


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# This code cannot be installed or executed on a corporate computer without a paid licence extension
# Contact info at rateslib.com if this code is observed outside its intended sphere of use.


class SBS(BaseDerivative):
    """
    Create a single currency basis swap composing two
    :class:`~rateslib.legs.FloatLeg` s.

    Parameters
    ----------
    args : tuple
        Required positional args to :class:`BaseDerivative`.
    float_spread : float, optional
        The spread applied to the :class:`~rateslib.legs.FloatLeg`. Can be set to
        `None` and designated
        later, perhaps after a mid-market spread for all periods has been calculated.
    spread_compound_method : str, optional
        The method to use for adding a floating spread to compounded rates. Available
        options are `{"none_simple", "isda_compounding", "isda_flat_compounding"}`.
    fixings : float, list, or Series optional
        If a float scalar, will be applied as the determined fixing for the first
        period. If a list of *n* fixings will be used as the fixings for the first *n*
        periods. If any sublist of length *m* is given, is used as the first *m* RFR
        fixings for that :class:`~rateslib.periods.FloatPeriod`. If a datetime
        indexed ``Series`` will use the fixings that are available in that object,
        and derive the rest from the ``curve``.
    fixing_method : str, optional
        The method by which floating rates are determined, set by default. See notes.
    method_param : int, optional
        A parameter that is used for the various ``fixing_method`` s. See notes.
    leg2_float_spread : float or None
        The floating spread applied in a simple way (after daily compounding) to the
        second :class:`~rateslib.legs.FloatLeg`. If `None` will be set to zero.
        float_spread : float, optional
        The spread applied to the :class:`~rateslib.legs.FloatLeg`. Can be set to
        `None` and designated
        later, perhaps after a mid-market spread for all periods has been calculated.
    leg2_spread_compound_method : str, optional
        The method to use for adding a floating spread to compounded rates. Available
        options are `{"none_simple", "isda_compounding", "isda_flat_compounding"}`.
    leg2_fixings : float, list, or Series optional
        If a float scalar, will be applied as the determined fixing for the first
        period. If a list of *n* fixings will be used as the fixings for the first *n*
        periods. If any sublist of length *m* is given, is used as the first *m* RFR
        fixings for that :class:`~rateslib.periods.FloatPeriod`. If a datetime
        indexed ``Series`` will use the fixings that are available in that object,
        and derive the rest from the ``curve``.
    leg2_fixing_method : str, optional
        The method by which floating rates are determined, set by default. See notes.
    leg2_method_param : int, optional
        A parameter that is used for the various ``fixing_method`` s. See notes.
    kwargs : dict
        Required keyword arguments to :class:`BaseDerivative`.

    Examples
    --------
    Construct curves to price the example.

    .. ipython:: python

       eur3m = Curve(
           nodes={
               dt(2022, 1, 1): 1.0,
               dt(2023, 1, 1): 0.965,
               dt(2024, 1, 1): 0.94
           },
           id="eur3m",
       )
       eur6m = Curve(
           nodes={
               dt(2022, 1, 1): 1.0,
               dt(2023, 1, 1): 0.962,
               dt(2024, 1, 1): 0.936
           },
           id="eur6m",
       )

    Create the SBS, and demonstrate the :meth:`~rateslib.instruments.SBS.rate`,
    :meth:`~rateslib.instruments.SBS.npv`,
    :meth:`~rateslib.instruments.SBS.analytic_delta`, and
    :meth:`~rateslib.instruments.SBS.spread`.

    .. ipython:: python

       sbs = SBS(
           effective=dt(2022, 1, 1),
           termination="18M",
           frequency="Q",
           leg2_frequency="S",
           calendar="tgt",
           currency="eur",
           fixing_method="ibor",
           method_param=2,
           convention="Act360",
           leg2_float_spread=-22.9,
           notional=100e6,
           curves=["eur3m", "eur3m", "eur6m", "eur3m"],
       )
       sbs.rate(curves=[eur3m, eur3m, eur6m, eur3m])
       sbs.npv(curves=[eur3m, eur3m, eur6m, eur3m])
       sbs.analytic_delta(curve=eur6m, disc_curve=eur3m, leg=2)
       sbs.spread(curves=[eur3m, eur3m, eur6m, eur3m], leg=2)

    A DataFrame of :meth:`~rateslib.instruments.SBS.cashflows`.

    .. ipython:: python

       sbs.cashflows(curves=[eur3m, eur3m, eur6m, eur3m])

    For accurate sensitivity calculations; :meth:`~rateslib.instruments.SBS.delta`
    and :meth:`~rateslib.instruments.SBS.gamma`, construct a curve model.

    .. ipython:: python

       irs_kws = dict(
           effective=dt(2022, 1, 1),
           frequency="A",
           leg2_frequency="Q",
           convention="30E360",
           leg2_convention="Act360",
           leg2_fixing_method="ibor",
           leg2_method_param=2,
           calendar="tgt",
           currency="eur",
           curves=["eur3m", "eur3m"],
       )
       sbs_kws = dict(
           effective=dt(2022, 1, 1),
           frequency="Q",
           leg2_frequency="S",
           convention="Act360",
           fixing_method="ibor",
           method_param=2,
           leg2_convention="Act360",
           calendar="tgt",
           currency="eur",
           curves=["eur3m", "eur3m", "eur6m", "eur3m"]
       )
       instruments = [
           IRS(termination="1Y", **irs_kws),
           IRS(termination="2Y", **irs_kws),
           SBS(termination="1Y", **sbs_kws),
           SBS(termination="2Y", **sbs_kws),
       ]
       solver = Solver(
           curves=[eur3m, eur6m],
           instruments=instruments,
           s=[1.55, 1.6, 5.5, 6.5],
           instrument_labels=["1Y", "2Y", "1Y 3s6s", "2Y 3s6s"],
           id="eur",
       )
       sbs.delta(solver=solver)
       sbs.gamma(solver=solver)

    """

    _float_spread_mixin = True
    _leg2_float_spread_mixin = True
    _rate_scalar = 100.0

    leg1: FloatLeg
    leg2: FloatLeg

    def __init__(
        self,
        *args: Any,
        float_spread: float | NoInput = NoInput(0),
        spread_compound_method: str | NoInput = NoInput(0),
        fixings: FixingsRates_ = NoInput(0),
        fixing_method: str | NoInput = NoInput(0),
        method_param: int | NoInput = NoInput(0),
        leg2_float_spread: float | NoInput = NoInput(0),
        leg2_spread_compound_method: str | NoInput = NoInput(0),
        leg2_fixings: FixingsRates_ = NoInput(0),
        leg2_fixing_method: str | NoInput = NoInput(0),
        leg2_method_param: int | NoInput = NoInput(0),
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        user_kwargs = dict(
            float_spread=float_spread,
            spread_compound_method=spread_compound_method,
            fixings=fixings,
            fixing_method=fixing_method,
            method_param=method_param,
            leg2_float_spread=leg2_float_spread,
            leg2_spread_compound_method=leg2_spread_compound_method,
            leg2_fixings=leg2_fixings,
            leg2_fixing_method=leg2_fixing_method,
            leg2_method_param=leg2_method_param,
        )
        self.kwargs: dict[str, Any] = _update_not_noinput(self.kwargs, user_kwargs)
        self._float_spread = float_spread
        self._leg2_float_spread = leg2_float_spread
        self.leg1 = FloatLeg(**_get(self.kwargs, leg=1))
        self.leg2 = FloatLeg(**_get(self.kwargs, leg=2))

    def _set_pricing_mid(self, curves: Curves_, solver: Solver_) -> None:
        if isinstance(self.float_spread, NoInput) and isinstance(self.leg2_float_spread, NoInput):
            # set a pricing parameter for the purpose of pricing NPV at zero.
            rate = self.rate(curves, solver)
            self.leg1.float_spread = _dual_float(rate)

    def analytic_delta(self, *args: Any, **kwargs: Any) -> DualTypes:
        """
        Return the analytic delta of a leg of the derivative object.

        See :meth:`BaseDerivative.analytic_delta`.
        """
        return super().analytic_delta(*args, **kwargs)

    def cashflows(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str | NoInput = NoInput(0),
    ) -> DataFrame:
        """
        Return the properties of all legs used in calculating cashflows.

        See :meth:`BaseDerivative.cashflows`.
        """
        self._set_pricing_mid(curves, solver)
        return super().cashflows(curves, solver, fx, base)

    def npv(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str | NoInput = NoInput(0),
        local: bool = False,
    ) -> NPV:
        """
        Return the NPV of the derivative object by summing legs.

        See :meth:`BaseDerivative.npv`.
        """
        self._set_pricing_mid(curves, solver)
        return super().npv(curves, solver, fx, base, local)

    def rate(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str | NoInput = NoInput(0),
        leg: int = 1,
    ) -> DualTypes:
        """
        Return the mid-market float spread on the specified leg of the SBS.

        Parameters
        ----------
        curves : Curve, str or list of such
            A list defines the following curves in the order:

            - Forecasting :class:`~rateslib.curves.Curve` for floating leg1.
            - Discounting :class:`~rateslib.curves.Curve` for both legs.
            - Forecasting :class:`~rateslib.curves.Curve` for floating leg2.
        solver : Solver, optional
            The numerical :class:`~rateslib.solver.Solver` that constructs
            :class:`~rateslib.curves.Curve` from calibrating
            instruments.
        leg: int in [1, 2]
            Specify which leg the spread calculation is applied to.

        Returns
        -------
        float, Dual or Dual2
        """
        core_npv: DualTypes = super().npv(curves, solver, local=False)  # type: ignore[assignment]
        curves_, _, _ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            fx,
            base,
            self.leg1.currency,
        )
        if leg == 1:
            leg_obj, args = self.leg1, (curves_[0], curves_[1])
        else:
            leg_obj, args = self.leg2, (curves_[2], curves_[3])

        specified_spd = _drb(0.0, leg_obj.float_spread)
        ret: DualTypes = leg_obj._spread(-core_npv, *args) + specified_spd
        return ret

        # irs_npv = self.npv(curves, solver)
        # curves, _ = self._get_curves_and_fx_maybe_from_solver(solver, curves, None)
        # if leg == 1:
        #     args = (curves[0], curves[1])
        # else:
        #     args = (curves[2], curves[3])
        # leg_analytic_delta = getattr(self, f"leg{leg}").analytic_delta(*args)
        # adjust = getattr(self, f"leg{leg}").float_spread
        # adjust = 0 if adjust is NoInput.blank else adjust
        # _ = irs_npv / leg_analytic_delta + adjust
        # return _

    def spread(self, *args: Any, **kwargs: Any) -> DualTypes:
        """
        Return the mid-market float spread on the specified leg of the SBS.

        Alias for :meth:`~rateslib.instruments.SBS.rate`.
        """
        return self.rate(*args, **kwargs)

    def fixings_table(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str | NoInput = NoInput(0),
        approximate: bool = False,
        right: datetime | NoInput = NoInput(0),
    ) -> DataFrame:
        """
        Return a DataFrame of fixing exposures on the :class:`~rateslib.legs.FloatLeg`.

        Parameters
        ----------
        curves : Curve, str or list of such
            A single :class:`~rateslib.curves.Curve` or id or a list of such.
            A list defines the following curves in the order:

            - Forecasting :class:`~rateslib.curves.Curve` for floating leg.
            - Discounting :class:`~rateslib.curves.Curve` for both legs.

        solver : Solver, optional
            The numerical :class:`~rateslib.solver.Solver` that constructs
            :class:`~rateslib.curves.Curve` from calibrating instruments.

            .. note::

               The arguments ``fx`` and ``base`` are unused by single currency
               derivatives rates calculations.

        approximate : bool, optional
            Perform a calculation that is broadly 10x faster but potentially loses
            precision upto 0.1%.
        right : datetime, optional
            Only calculate fixing exposures upto and including this date.

        Returns
        -------
        DataFrame
        """
        curves_, _, _ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            NoInput(0),
            NoInput(0),
            self.leg1.currency,
        )
        df1 = self.leg1.fixings_table(
            curve=curves_[0], approximate=approximate, disc_curve=curves_[1], right=right
        )
        df2 = self.leg2.fixings_table(
            curve=curves_[2], approximate=approximate, disc_curve=curves_[3], right=right
        )
        return _composit_fixings_table(df1, df2)


class FRA(BaseDerivative):
    """
    Create a forward rate agreement composing single period :class:`~rateslib.legs.FixedLeg`
    and :class:`~rateslib.legs.FloatLeg` valued in a customised manner.

    Parameters
    ----------
    args : dict
        Required positional args to :class:`BaseDerivative`.
    fixed_rate : float or None
        The fixed rate applied to the :class:`~rateslib.legs.FixedLeg`. If `None`
        will be set to mid-market when curves are provided.
    method_param : int, optional
        A parameter that is used for the various ``fixing_method`` s. See notes.
    fixings : float or list, optional
        If a float scalar, will be applied as the determined fixing for the first
        period. If a list of *n* fixings will be used as the fixings for the first *n*
        periods. If any sublist of length *m* is given as the first *m* RFR fixings
        within individual curve and composed into the overall rate.
    kwargs : dict
        Required keyword arguments to :class:`BaseDerivative`.

    Notes
    -----
    A FRA is a derivative whose *FloatLeg* ``fixing_method`` is set to *"ibor"*.
    Additionally, there is no concept of ``float_spread`` for the IBOR fixing rate on an
    *FRA*, and it is therefore set to 0.0.

    Examples
    --------
    Construct curves to price the example.

    .. ipython:: python

       eur3m = Curve(
           nodes={
               dt(2022, 1, 1): 1.0,
               dt(2023, 1, 1): 0.965,
               dt(2024, 1, 1): 0.94
           },
           id="eur3m",
       )

    Create the FRA, and demonstrate the :meth:`~rateslib.instruments.FRA.rate`,
    :meth:`~rateslib.instruments.FRA.npv`,
    :meth:`~rateslib.instruments.FRA.analytic_delta`.

    .. ipython:: python

       fra = FRA(
           effective=dt(2023, 2, 15),
           termination="3M",
           frequency="Q",
           calendar="tgt",
           currency="eur",
           method_param=2,
           convention="Act360",
           notional=100e6,
           fixed_rate=2.617,
           curves=["eur3m"],
       )
       fra.rate(curves=eur3m)
       fra.npv(curves=eur3m)
       fra.analytic_delta(curve=eur3m)

    A DataFrame of :meth:`~rateslib.instruments.FRA.cashflows`.

    .. ipython:: python

       fra.cashflows(curves=eur3m)

    For accurate sensitivity calculations; :meth:`~rateslib.instruments.FRA.delta`
    and :meth:`~rateslib.instruments.FRA.gamma`, construct a curve model.

    .. ipython:: python

       irs_kws = dict(
           effective=dt(2022, 1, 1),
           frequency="A",
           leg2_frequency="Q",
           convention="30E360",
           leg2_convention="Act360",
           leg2_fixing_method="ibor",
           leg2_method_param=2,
           calendar="tgt",
           currency="eur",
           curves=["eur3m", "eur3m"],
       )
       instruments = [
           IRS(termination="1Y", **irs_kws),
           IRS(termination="2Y", **irs_kws),
       ]
       solver = Solver(
           curves=[eur3m],
           instruments=instruments,
           s=[1.55, 1.6],
           instrument_labels=["1Y", "2Y"],
           id="eur",
       )
       fra.delta(solver=solver)
       fra.gamma(solver=solver)

    """

    _fixed_rate_mixin = True

    leg1: FixedLeg
    leg2: FloatLeg

    def __init__(
        self,
        *args: Any,
        fixed_rate: DualTypes | NoInput = NoInput(0),
        method_param: int | NoInput = NoInput(0),
        fixings: FixingsRates_ = NoInput(0),
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        user_kwargs = {
            "fixed_rate": fixed_rate,
            "leg2_method_param": method_param,
            "leg2_fixings": fixings,
            # overload BaseDerivative
            "leg2_fixing_method": "ibor",
            "leg2_float_spread": 0.0,
        }
        self.kwargs: dict[str, Any] = _update_not_noinput(self.kwargs, user_kwargs)

        # Build
        self._fixed_rate = self.kwargs["fixed_rate"]
        self.leg1 = FixedLeg(**_get(self.kwargs, leg=1))
        self.leg2 = FloatLeg(**_get(self.kwargs, leg=2))

        # Instrument Validation
        if self.leg1.schedule.n_periods != 1 or self.leg2.schedule.n_periods != 1:
            raise ValueError("FRA scheduling inputs did not define a single period.")
        if self.leg1.convention != self.leg2.convention:
            raise ValueError("FRA cannot have different `convention` on either Leg.")
        if self.leg1.schedule.frequency != self.leg2.schedule.frequency:
            raise ValueError("FRA cannot have different `frequency` on either Leg.")
        if self.leg1.schedule.modifier != self.leg2.schedule.modifier:
            raise ValueError("FRA cannot have different `modifier` on either Leg.")

    def _set_pricing_mid(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
    ) -> None:
        if isinstance(self.fixed_rate, NoInput):
            mid_market_rate = self.rate(curves, solver)
            self.leg1.fixed_rate = mid_market_rate.real

    def analytic_delta(  # type: ignore[override]
        self,
        curve: CurveOption_,
        disc_curve: CurveOption_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str | NoInput = NoInput(0),
    ) -> DualTypes:
        """
        Return the analytic delta of the FRA.

        For arguments see :meth:`~rateslib.periods.BasePeriod.analytic_delta`.
        """
        disc_curve_: Curve = _disc_required_maybe_from_curve(curve, disc_curve)
        fx, base = _get_fx_and_base(self.leg1.currency, fx, base)
        rate = self.rate([curve])
        dcf = self._fixed_period.dcf
        _: DualTypes = self.leg1.notional * dcf * disc_curve_[self._payment_date] / 10000
        return fx * _ / (1 + dcf * rate / 100)

    def npv(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str | NoInput = NoInput(0),
        local: bool = False,
    ) -> NPV:
        """
        Return the NPV of the derivative.

        See :meth:`BaseDerivative.npv`.
        """

        self._set_pricing_mid(curves, solver)
        curves_, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            fx,
            base,
            self.leg1.currency,
        )
        fx__, _ = _get_fx_and_base(self.leg1.currency, fx_, base_)
        disc_curve_ = _disc_required_maybe_from_curve(curves_[0], curves_[1])
        value = self._cashflow_or_raise(curves_[0]) * disc_curve_[self._payment_date]
        if local:
            return {self.leg1.currency: value}
        else:
            return fx__ * value

    def rate(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str | NoInput = NoInput(0),
    ) -> DualTypes:
        """
        Return the mid-market rate of the FRA.

        Only the forecasting curve is required to price an FRA.

        Parameters
        ----------
        curves : Curve, str or list of such
            A single :class:`~rateslib.curves.Curve` or id or a list of such.
            A list defines the following curves in the order:

            - Forecasting :class:`~rateslib.curves.Curve` for floating leg.
            - Discounting :class:`~rateslib.curves.Curve` for floating leg.
        solver : Solver, optional
            The numerical :class:`~rateslib.solver.Solver` that
            constructs :class:`~rateslib.curves.Curve` from calibrating instruments.
        fx : unused
        base : unused

        Returns
        -------
        float, Dual or Dual2
        """
        curves_, _, _ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            fx,
            base,
            self.leg1.currency,
        )
        return self._float_period.rate(curves_[0])

    def cashflow(self, curve: CurveOption_) -> DualTypes | None:
        """
        Calculate the local currency cashflow on the FRA from current floating rate
        and fixed rate.

        Parameters
        ----------
        curve : Curve or LineCurve,
            The forecasting curve for determining the floating rate.

        Returns
        -------
        float, Dual or Dual2
        """
        cf1 = self._fixed_period.cashflow
        rate = self._float_period.rate(curve)
        cf2 = self.kwargs["notional"] * self._float_period.dcf * rate / 100
        if not isinstance(cf1, NoInput) and not isinstance(cf2, NoInput):
            cf: DualTypes = cf1 + cf2
        else:
            return None

        # FRA specification discounts cashflows by the IBOR rate.
        cf /= 1 + self._float_period.dcf * rate / 100
        return cf

    def _cashflow_or_raise(self, curve: CurveOption_) -> DualTypes:
        cf_ = self.cashflow(curve)
        if cf_ is None:
            raise ValueError("Must supply a `curve` to determine cashflow for FRA.")
        return cf_

    def cashflows(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str | NoInput = NoInput(0),
    ) -> DataFrame:
        """
        Return the properties of the leg used in calculating cashflows.

        Parameters
        ----------
        args :
            Positional arguments supplied to :meth:`~rateslib.periods.BasePeriod.cashflows`.
        kwargs :
            Keyword arguments supplied to :meth:`~rateslib.periods.BasePeriod.cashflows`.

        Returns
        -------
        DataFrame
        """
        self._set_pricing_mid(curves, solver)
        curves_, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            fx,
            base,
            self.leg1.currency,
        )
        fx__, base_ = _get_fx_and_base(self.leg1.currency, fx_, base_)

        if isinstance(self.fixed_rate, NoInput):
            _fix = None
            _cf = None
        else:
            _fix = -_dual_float(self.fixed_rate)
            _cf = _dual_float(self.cashflow(curves_[0]))  # type: ignore[arg-type]

        if isinstance(curves_[1], NoInput):
            _df = None
        else:
            _df = _dual_float(curves_[1][self._payment_date])

        _spd = self.rate(curves_[0])
        if _spd is not None:
            _spd = -_dual_float(_spd) * 100.0

        if _cf is not None and _df is not None:
            _npv_local = _cf * _df
            _npv_fx = _npv_local * _dual_float(fx__)
        else:
            _npv_local = None
            _npv_fx = None

        cfs = self._fixed_period.cashflows(curves_[0], curves_[1], fx__, base_)
        cfs[defaults.headers["type"]] = "FRA"
        cfs[defaults.headers["payment"]] = self._payment_date
        cfs[defaults.headers["cashflow"]] = _cf
        cfs[defaults.headers["rate"]] = _fix
        cfs[defaults.headers["spread"]] = _spd
        cfs[defaults.headers["npv"]] = _npv_local
        cfs[defaults.headers["df"]] = _df
        cfs[defaults.headers["fx"]] = _dual_float(fx__)
        cfs[defaults.headers["npv_fx"]] = _npv_fx

        _: DataFrame = DataFrame.from_records([cfs])
        return _

    def fixings_table(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str | NoInput = NoInput(0),
        approximate: bool = False,
        right: datetime | NoInput = NoInput(0),
    ) -> DataFrame:
        """
        Return a DataFrame of fixing exposures on the :class:`~rateslib.legs.FloatLeg`.

        Parameters
        ----------
        curves : Curve, str or list of such
            A single :class:`~rateslib.curves.Curve` or id or a list of such.
            A list defines the following curves in the order:

            - Forecasting :class:`~rateslib.curves.Curve` for floating leg.
            - Discounting :class:`~rateslib.curves.Curve` for both legs.

        solver : Solver, optional
            The numerical :class:`~rateslib.solver.Solver` that constructs
            :class:`~rateslib.curves.Curve` from calibrating instruments.

            .. note::

               The arguments ``fx`` and ``base`` are unused by single currency
               derivatives rates calculations.

        approximate : bool, optional
            Perform a calculation that is broadly 10x faster but potentially loses
            precision upto 0.1%.
        right : datetime, optional
            Only calculate fixing exposures upto and including this date.

        Returns
        -------
        DataFrame
        """
        curves_, _, _ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            NoInput(0),
            NoInput(0),
            self.leg2.currency,
        )
        if isinstance(curves_[2], NoInput) or isinstance(curves_[3], NoInput):
            raise ValueError("`curves` are not supplied correctly.")

        df = self.leg2.fixings_table(
            curve=curves_[2], approximate=approximate, disc_curve=curves_[3]
        )
        rate = self._float_period.rate(curve=curves_[2])
        scalar: DualTypes = curves_[3][self._payment_date] / curves_[3][self._float_period.payment]
        scalar *= 1.0 / (1.0 + self._float_period.dcf * rate / 100.0)
        df[(curves_[2].id, "risk")] *= scalar  # type: ignore[operator, union-attr]
        df[(curves_[2].id, "notional")] *= scalar  # type: ignore[operator, union-attr]
        return _trim_df_by_index(df, NoInput(0), right)

    def delta(self, *args: Any, **kwargs: Any) -> DataFrame:
        """
        Calculate the delta of the *Instrument*.

        For arguments see :meth:`Sensitivities.delta()<rateslib.instruments.Sensitivities.delta>`.
        """
        return super().delta(*args, **kwargs)

    def gamma(self, *args: Any, **kwargs: Any) -> DataFrame:
        """
        Calculate the gamma of the *Instrument*.

        For arguments see :meth:`Sensitivities.gamma()<rateslib.instruments.Sensitivities.gamma>`.
        """
        return super().gamma(*args, **kwargs)

    @property
    def _payment_date(self) -> datetime:
        """
        Get the adjusted payment date for the FRA under regular FRA specifications.

        This date is calculated as a lagged amount of business days after the Accrual Start
        Date, under the calendar applicable to the Instrument.
        """
        return self.leg1.schedule.pschedule[0]

    @property
    def _fixed_period(self) -> FixedPeriod:
        return self.leg1.periods[0]  # type: ignore[return-value]

    @property
    def _float_period(self) -> FloatPeriod:
        return self.leg2.periods[0]  # type: ignore[return-value]
