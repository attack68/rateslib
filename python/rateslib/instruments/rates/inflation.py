from __future__ import annotations

from typing import TYPE_CHECKING

from rateslib.curves._parsers import _validate_curve_is_not_dict, _validate_curve_not_no_input
from rateslib.default import NoInput, _drb
from rateslib.dual.utils import _dual_float
from rateslib.instruments.base import BaseDerivative
from rateslib.instruments.utils import (
    _get,
    _get_curves_fx_and_base_maybe_from_solver,
    _update_not_noinput,
)
from rateslib.legs import FloatLeg, IndexFixedLeg, ZeroFixedLeg, ZeroIndexLeg

if TYPE_CHECKING:
    from rateslib.typing import (
        FX_,
        NPV,
        Any,
        Curves_,
        DataFrame,
        DualTypes,
        DualTypes_,
        FixingsRates_,
        Series,
        Solver_,
        datetime_,
        int_,
        str_,
    )


class ZCIS(BaseDerivative):
    """
    Create a zero coupon index swap (ZCIS) composing an
    :class:`~rateslib.legs.ZeroFixedLeg`
    and a :class:`~rateslib.legs.ZeroIndexLeg`.

    For more information see the :ref:`Cookbook Article:<cookbook-doc>` *"Using Curves with an
    Index and Inflation Instruments"*.

    Parameters
    ----------
    args : dict
        Required positional args to :class:`BaseDerivative`.
    fixed_rate : float or None
        The fixed rate applied to the :class:`~rateslib.legs.ZeroFixedLeg`. If `None`
        will be set to mid-market when curves are provided.
    leg2_index_base : float or None, optional
        The base index applied to all periods.
    leg2_index_fixings : float, or Series, optional
        If a float scalar, will be applied as the index fixing for the first
        period.
        If a list of *n* fixings will be used as the index fixings for the first *n*
        periods.
        If a datetime indexed ``Series`` will use the fixings that are available in
        that object, and derive the rest from the ``curve``.
    leg2_index_method : str
        Whether the indexing uses a daily measure for settlement or the most recently
        monthly data taken from the first day of month.
    leg2_index_lag : int, optional
        The number of months by which the index value is lagged. Used to ensure
        consistency between curves and forecast values. Defined by default.
    kwargs : dict
        Required keyword arguments to :class:`BaseDerivative`.

    Examples
    --------
    Construct a curve to price the example.

    .. ipython:: python

       usd = Curve(
           nodes={
               dt(2022, 1, 1): 1.0,
               dt(2027, 1, 1): 0.85,
               dt(2032, 1, 1): 0.65,
           },
           id="usd",
       )
       us_cpi = Curve(
           nodes={
               dt(2022, 1, 1): 1.0,
               dt(2027, 1, 1): 0.85,
               dt(2032, 1, 1): 0.70,
           },
           id="us_cpi",
           index_base=100,
           index_lag=3,
       )

    Create the ZCIS, and demonstrate the :meth:`~rateslib.instruments.ZCIS.rate`,
    :meth:`~rateslib.instruments.ZCIS.npv`,
    :meth:`~rateslib.instruments.ZCIS.analytic_delta`, and

    .. ipython:: python

       zcis = ZCIS(
           effective=dt(2022, 1, 1),
           termination="10Y",
           spec="usd_zcis",
           fixed_rate=2.05,
           notional=100e6,
           leg2_index_base=100.0,
           curves=["usd", "usd", "us_cpi", "usd"],
       )
       zcis.rate(curves=[us_cpi, usd])
       zcis.npv(curves=[us_cpi, usd])
       zcis.analytic_delta(usd, usd)

    A DataFrame of :meth:`~rateslib.instruments.ZCIS.cashflows`.

    .. ipython:: python

       zcis.cashflows(curves=[us_cpi, usd])

    For accurate sensitivity calculations; :meth:`~rateslib.instruments.ZCIS.delta`
    and :meth:`~rateslib.instruments.ZCIS.gamma`, construct a curve model.

    .. ipython:: python

       instruments = [
           IRS(dt(2022, 1, 1), "5Y", spec="usd_irs", curves="usd"),
           IRS(dt(2022, 1, 1), "10Y", spec="usd_irs", curves="usd"),
           ZCIS(dt(2022, 1, 1), "5Y", spec="usd_zcis", curves=["us_cpi", "usd"]),
           ZCIS(dt(2022, 1, 1), "10Y", spec="usd_zcis", curves=["us_cpi", "usd"]),
       ]
       solver = Solver(
           curves=[usd, us_cpi],
           instruments=instruments,
           s=[3.40, 3.60, 2.2, 2.05],
           instrument_labels=["5Y", "10Y", "5Yi", "10Yi"],
           id="us",
       )
       zcis.delta(solver=solver)
       zcis.gamma(solver=solver)
    """

    _fixed_rate_mixin = True
    _leg2_index_base_mixin = True

    leg1: ZeroFixedLeg
    leg2: ZeroIndexLeg

    def __init__(
        self,
        *args: Any,
        fixed_rate: DualTypes_ = NoInput(0),
        leg2_index_base: DualTypes_ | Series[DualTypes] = NoInput(0),  # type: ignore[type-var]
        leg2_index_fixings: DualTypes_ | Series[DualTypes] = NoInput(0),  # type: ignore[type-var]
        leg2_index_method: str_ = NoInput(0),
        leg2_index_lag: int_ = NoInput(0),
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        user_kwargs: dict[str, Any] = dict(
            fixed_rate=fixed_rate,
            leg2_index_base=leg2_index_base,
            leg2_index_fixings=leg2_index_fixings,
            leg2_index_lag=leg2_index_lag,
            leg2_index_method=leg2_index_method,
        )
        self.kwargs: dict[str, Any] = _update_not_noinput(self.kwargs, user_kwargs)
        self._fixed_rate = fixed_rate
        # TODO: correct - issue 412
        self._leg2_index_base = leg2_index_base  # type: ignore[assignment]
        self.leg1 = ZeroFixedLeg(**_get(self.kwargs, leg=1))
        self.leg2 = ZeroIndexLeg(**_get(self.kwargs, leg=2))

    def _set_pricing_mid(self, curves: Curves_, solver: Solver_) -> None:
        if isinstance(self.fixed_rate, NoInput):
            # set a fixed rate for the purpose of pricing NPV, which should be zero.
            mid_market_rate = self.rate(curves, solver)
            self.leg1.fixed_rate = _dual_float(mid_market_rate)

    def cashflows(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
    ) -> DataFrame:
        self._set_pricing_mid(curves, solver)
        return super().cashflows(curves, solver, fx, base)

    def npv(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
        local: bool = False,
    ) -> NPV:
        self._set_pricing_mid(curves, solver)
        return super().npv(curves, solver, fx, base, local)

    def rate(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
    ) -> DualTypes:
        """
        Return the mid-market IRR rate of the ZCIS.

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
        if isinstance(self.leg2_index_base, NoInput):
            # must forecast for the leg
            i_curve = _validate_curve_not_no_input(_validate_curve_is_not_dict(curves_[2]))
            forecast_value = i_curve.index_value(
                self.leg2.schedule.effective,
                self.leg2.index_lag,
                self.leg2.index_method,
            )
            if abs(forecast_value) < 1e-13:
                raise ValueError(
                    "Forecasting the `index_base` for the ZCIS yielded 0.0, which is infeasible.\n"
                    "This might occur if the ZCIS starts in the past, or has a 'monthly' "
                    "`index_method` which uses the 1st day of the effective month, which is in the "
                    "past.\nA known `index_base` value should be input with the ZCIS "
                    "specification.",
                )
            self.leg2.index_base = forecast_value
        leg2_npv: DualTypes = self.leg2.npv(curves_[2], curves_[3], local=False)  # type: ignore[assignment]

        return self.leg1._spread(-leg2_npv, curves_[0], curves_[1]) / 100


class IIRS(BaseDerivative):
    """
    Create an indexed interest rate swap (IIRS) composing an
    :class:`~rateslib.legs.IndexFixedLeg` and a :class:`~rateslib.legs.FloatLeg`.

    For more information see the :ref:`Cookbook Article:<cookbook-doc>` *"Using Curves with an
    Index and Inflation Instruments"*.

    Parameters
    ----------
    args : dict
       Required positional args to :class:`BaseDerivative`.
    fixed_rate : float or None
       The fixed rate applied to the :class:`~rateslib.legs.ZeroFixedLeg`. If `None`
       will be set to mid-market when curves are provided.
    index_base : float or None, optional
       The base index applied to all periods.
    index_fixings : float, or Series, optional
       If a float scalar, will be applied as the index fixing for the first
       period.
       If a list of *n* fixings will be used as the index fixings for the first *n*
       periods.
       If a datetime indexed ``Series`` will use the fixings that are available in
       that object, and derive the rest from the ``curve``.
    index_method : str
       Whether the indexing uses a daily measure for settlement or the most recently
       monthly data taken from the first day of month.
    index_lag : int, optional
       The number of months by which the index value is lagged. Used to ensure
       consistency between curves and forecast values. Defined by default.
    notional_exchange : bool, optional
       Whether the legs include final notional exchanges and interim
       amortization notional exchanges.
    kwargs : dict
       Required keyword arguments to :class:`BaseDerivative`.

    Examples
    --------
    Construct a curve to price the example.

    .. ipython:: python

      usd = Curve(
          nodes={
              dt(2022, 1, 1): 1.0,
              dt(2027, 1, 1): 0.85,
              dt(2032, 1, 1): 0.65,
          },
          id="usd",
      )
      us_cpi = Curve(
          nodes={
              dt(2022, 1, 1): 1.0,
              dt(2027, 1, 1): 0.85,
              dt(2032, 1, 1): 0.70,
          },
          id="us_cpi",
          index_base=100,
          index_lag=3,
      )

    Create the IIRS, and demonstrate the :meth:`~rateslib.instruments.IIRS.rate`, and
    :meth:`~rateslib.instruments.IIRS.npv`.

    .. ipython:: python

      iirs = IIRS(
          effective=dt(2022, 1, 1),
          termination="4Y",
          frequency="A",
          calendar="nyc",
          currency="usd",
          fixed_rate=2.05,
          convention="1+",
          notional=100e6,
          index_base=100.0,
          index_method="monthly",
          index_lag=3,
          notional_exchange=True,
          leg2_convention="Act360",
          curves=["us_cpi", "usd", "usd", "usd"],
      )
      iirs.rate(curves=[us_cpi, usd, usd, usd])
      iirs.npv(curves=[us_cpi, usd, usd, usd])

    A DataFrame of :meth:`~rateslib.instruments.IIRS.cashflows`.

    .. ipython:: python

      iirs.cashflows(curves=[us_cpi, usd, usd, usd])

    For accurate sensitivity calculations; :meth:`~rateslib.instruments.IIRS.delta`
    and :meth:`~rateslib.instruments.IIRS.gamma`, construct a curve model.

    .. ipython:: python

      sofr_kws = dict(
          effective=dt(2022, 1, 1),
          frequency="A",
          convention="Act360",
          calendar="nyc",
          currency="usd",
          curves=["usd"]
      )
      cpi_kws = dict(
          effective=dt(2022, 1, 1),
          frequency="A",
          convention="1+",
          calendar="nyc",
          leg2_index_method="monthly",
          currency="usd",
          curves=["usd", "usd", "us_cpi", "usd"]
      )
      instruments = [
          IRS(termination="5Y", **sofr_kws),
          IRS(termination="10Y", **sofr_kws),
          ZCIS(termination="5Y", **cpi_kws),
          ZCIS(termination="10Y", **cpi_kws),
      ]
      solver = Solver(
          curves=[usd, us_cpi],
          instruments=instruments,
          s=[3.40, 3.60, 2.2, 2.05],
          instrument_labels=["5Y", "10Y", "5Yi", "10Yi"],
          id="us",
      )
      iirs.delta(solver=solver)
      iirs.gamma(solver=solver)
    """

    _fixed_rate_mixin = True
    _index_base_mixin = True
    _leg2_float_spread_mixin = True

    leg1: IndexFixedLeg
    leg2: FloatLeg

    def __init__(
        self,
        *args: Any,
        fixed_rate: DualTypes_ = NoInput(0),
        index_base: DualTypes_ | Series[DualTypes] = NoInput(0),  # type: ignore[type-var]
        index_fixings: DualTypes_ | Series[DualTypes] = NoInput(0),  # type: ignore[type-var]
        index_method: str_ = NoInput(0),
        index_lag: int_ = NoInput(0),
        notional_exchange: bool = False,
        payment_lag_exchange: int_ = NoInput(0),
        leg2_float_spread: DualTypes_ = NoInput(0),
        leg2_fixings: FixingsRates_ = NoInput(0),  # type: ignore[type-var]
        leg2_fixing_method: str_ = NoInput(0),
        leg2_method_param: int_ = NoInput(0),
        leg2_spread_compound_method: str_ = NoInput(0),
        leg2_payment_lag_exchange: int_ = NoInput(1),
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if leg2_payment_lag_exchange is NoInput.inherit:
            leg2_payment_lag_exchange = payment_lag_exchange
        user_kwargs = dict(
            fixed_rate=fixed_rate,
            index_base=index_base,
            index_fixings=index_fixings,
            index_method=index_method,
            index_lag=index_lag,
            initial_exchange=False,
            final_exchange=notional_exchange,
            payment_lag_exchange=payment_lag_exchange,
            leg2_float_spread=leg2_float_spread,
            leg2_spread_compound_method=leg2_spread_compound_method,
            leg2_fixings=leg2_fixings,
            leg2_fixing_method=leg2_fixing_method,
            leg2_method_param=leg2_method_param,
            leg2_payment_lag_exchange=leg2_payment_lag_exchange,
            leg2_initial_exchange=False,
            leg2_final_exchange=notional_exchange,
        )
        self.kwargs: dict[str, Any] = _update_not_noinput(self.kwargs, user_kwargs)

        self._index_base = self.kwargs["index_base"]
        self._fixed_rate = self.kwargs["fixed_rate"]
        self.leg1 = IndexFixedLeg(**_get(self.kwargs, leg=1))
        self.leg2 = FloatLeg(**_get(self.kwargs, leg=2))

    def _set_pricing_mid(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
    ) -> None:
        mid_market_rate = self.rate(curves, solver)
        self.leg1.fixed_rate = _dual_float(mid_market_rate)

    def npv(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
        local: bool = False,
    ) -> NPV:
        curves_, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            fx,
            base,
            self.leg1.currency,
        )
        if isinstance(self.index_base, NoInput):
            # must forecast for the leg
            i_curve = _validate_curve_not_no_input(_validate_curve_is_not_dict(curves_[0]))
            self.leg1.index_base = i_curve.index_value(
                self.leg1.schedule.effective,
                self.leg1.index_lag,
                self.leg1.index_method,
            )
        if isinstance(self.fixed_rate, NoInput):
            # set a fixed rate for the purpose of pricing NPV, which should be zero.
            self._set_pricing_mid(curves_, solver)
        return super().npv(curves_, solver, fx_, base_, local)

    def cashflows(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
    ) -> DataFrame:
        curves_, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            fx,
            base,
            self.leg1.currency,
        )
        if isinstance(self.index_base, NoInput):
            # must forecast for the leg
            i_curve = _validate_curve_not_no_input(_validate_curve_is_not_dict(curves_[0]))
            self.leg1.index_base = i_curve.index_value(
                self.leg1.schedule.effective,
                self.leg1.index_lag,
                self.leg1.index_method,
            )
        if isinstance(self.fixed_rate, NoInput):
            # set a fixed rate for the purpose of pricing NPV, which should be zero.
            self._set_pricing_mid(curves_, solver)
        return super().cashflows(curves_, solver, fx_, base_)

    def rate(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
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
        if isinstance(self.index_base, NoInput):
            # must forecast for the leg
            i_curve = _validate_curve_not_no_input(_validate_curve_is_not_dict(curves_[0]))
            self.leg1.index_base = i_curve.index_value(
                self.leg1.schedule.effective,
                self.leg1.index_lag,
                self.leg1.index_method,
            )
        leg2_npv: DualTypes = self.leg2.npv(curves_[2], curves_[3], local=False)  # type: ignore[assignment]

        if isinstance(self.fixed_rate, NoInput):
            self.leg1.fixed_rate = 0.0
            _existing: DualTypes = 0.0
        else:
            _existing = self.fixed_rate

        leg1_npv: DualTypes = self.leg1.npv(curves_[0], curves_[1], local=False)  # type: ignore[assignment]

        ret: DualTypes = self.leg1._spread(-leg2_npv - leg1_npv, curves_[0], curves_[1]) / 100
        return ret + _existing

    def spread(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
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

    def fixings_table(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
        approximate: bool = False,
        right: datetime_ = NoInput(0),
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

        Returns
        -------
        DataFrame
        """
        curves, _, _ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            NoInput(0),
            NoInput(0),
            self.leg2.currency,
        )
        df = self.leg2.fixings_table(
            curve=curves[2], approximate=approximate, disc_curve=curves[3], right=right
        )
        return df
