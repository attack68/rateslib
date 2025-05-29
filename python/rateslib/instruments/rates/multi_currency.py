from __future__ import annotations

import warnings
from datetime import datetime
from typing import TYPE_CHECKING

from pandas import DataFrame, DatetimeIndex, MultiIndex

from rateslib import defaults
from rateslib.calendars import _get_fx_expiry_and_delivery, get_calendar
from rateslib.curves._parsers import _validate_curve_not_no_input
from rateslib.default import NoInput, _drb
from rateslib.dual import Dual, Dual2, Variable
from rateslib.dual.utils import _dual_float
from rateslib.fx import FXForwards, FXRates, forward_fx
from rateslib.instruments.base import BaseDerivative, Metrics
from rateslib.instruments.sensitivities import Sensitivities
from rateslib.instruments.utils import (
    _composit_fixings_table,
    _get,
    _get_curves_fx_and_base_maybe_from_solver,
    _push,
    _update_not_noinput,
    _update_with_defaults,
)
from rateslib.legs import (
    FixedLeg,
    FixedLegMtm,
    FloatLeg,
    FloatLegMtm,
)
from rateslib.periods import (
    Cashflow,
    NonDeliverableCashflow,
)
from rateslib.periods.utils import _get_fx_fixings_from_non_fx_forwards, _maybe_local

# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# This code cannot be installed or executed on a corporate computer without a paid licence extension
# Contact info at rateslib.com if this code is observed outside its intended sphere of use.

if TYPE_CHECKING:
    from rateslib.typing import (
        FX_,
        NPV,
        Any,
        CalInput,
        Curve_,
        Curves_,
        DualTypes,
        DualTypes_,
        FixingsFx_,
        FixingsRates_,
        Solver_,
        bool_,
        datetime_,
        int_,
        str_,
    )


class FXExchange(Sensitivities, Metrics):
    """
    Create a simple exchange of two currencies.

    Parameters
    ----------
    settlement : datetime
        The date of the currency exchange.
    pair: str
        The curreny pair of the exchange, e.g. "eurusd", using 3-digit iso codes.
    fx_rate : float, optional
        The FX rate used to derive the notional exchange on *Leg2*.
    notional : float
        The cashflow amount of the LHS currency.
    curves : Curve, LineCurve, str or list of such, optional
        For *FXExchange* only discounting curves are required in each currency and not rate
        forecasting curves.
        The signature should be: `[None, eur_curve, None, usd_curve]` for a "eurusd" pair.
    """

    leg1: Cashflow  # type: ignore[assignment]
    leg2: Cashflow  # type: ignore[assignment]

    def __init__(
        self,
        settlement: datetime,
        pair: str,
        fx_rate: DualTypes_ = NoInput(0),
        notional: DualTypes_ = NoInput(0),
        curves: Curves_ = NoInput(0),
    ):
        self.curves = curves
        self.settlement = settlement
        self.pair = pair.lower()
        self.leg1 = Cashflow(
            notional=-1.0 * _drb(defaults.notional, notional),
            currency=self.pair[0:3],
            payment=settlement,
            stub_type="Exchange",
            rate=NoInput(0),
        )
        self.leg2 = Cashflow(
            notional=1.0,  # will be determined by setting fx_rate
            currency=self.pair[3:6],
            payment=settlement,
            stub_type="Exchange",
            rate=fx_rate,
        )
        self.fx_rate = fx_rate

    @property
    def fx_rate(self) -> DualTypes_:
        return self._fx_rate

    @fx_rate.setter
    def fx_rate(self, value: DualTypes_) -> None:
        self._fx_rate = value
        self.leg2.notional = _drb(0.0, value) * -self.leg1.notional
        self.leg2._rate = value

    def _set_pricing_mid(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
    ) -> None:
        if isinstance(self.fx_rate, NoInput):
            mid_market_rate = self.rate(curves, solver, fx)
            self.fx_rate = _dual_float(mid_market_rate)
            self._fx_rate = NoInput(0)

    def npv(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
        local: bool = False,
    ) -> NPV:
        """
        Return the NPV of the *FXExchange* by summing legs.

        For arguments see :meth:`BaseMixin.npv<rateslib.instruments.BaseMixin.npv>`
        """
        self._set_pricing_mid(curves, solver, fx)

        curves_, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            fx,
            base,
            self.leg1.currency,
        )

        if isinstance(fx_, NoInput):
            raise ValueError(
                "Must have some FX information to price FXExchange, either `fx` or "
                "`solver` containing an FX object.",
            )
        elif not isinstance(fx_, FXRates | FXForwards):
            # force base_ leg1 currency to be converted consistent.
            leg1_npv: NPV = self.leg1.npv(curves_[0], curves_[1], fx_, base_, local)
            leg2_npv: NPV = self.leg2.npv(curves_[2], curves_[3], 1.0, base_, local)
            warnings.warn(
                "When valuing multi-currency derivatives it not best practice to "
                "supply `fx` as numeric.\nYour input:\n"
                f"`npv(solver={'None' if isinstance(solver, NoInput) else '<Solver>'}, "
                f"fx={fx}, base='{base if isinstance(base, NoInput) else 'None'}')\n"
                "has been implicitly converted into the following by this operation:\n"
                f"`npv(solver={'None' if isinstance(solver, NoInput) else '<Solver>'}, "
                f"fx=FXRates({{'{self.leg2.currency}{self.leg1.currency}: {fx}}}), "
                f"base='{self.leg2.currency}')\n.",
                UserWarning,
            )
        else:
            leg1_npv = self.leg1.npv(curves_[0], curves_[1], fx_, base_, local)
            leg2_npv = self.leg2.npv(curves_[2], curves_[3], fx_, base_, local)

        if local:
            return {
                k: leg1_npv.get(k, 0) + leg2_npv.get(k, 0)  # type: ignore[union-attr]
                for k in set(leg1_npv) | set(leg2_npv)  # type: ignore[arg-type]
            }
        else:
            return leg1_npv + leg2_npv  # type: ignore[operator]

    def cashflows(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
    ) -> DataFrame:
        """
        Return the cashflows of the *FXExchange* by aggregating legs.

        For arguments see :meth:`BaseMixin.npv<rateslib.instruments.BaseMixin.cashflows>`
        """
        self._set_pricing_mid(curves, solver, fx)
        curves_, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            fx,
            base,
            NoInput(0),
        )
        seq = [
            self.leg1.cashflows(curves_[0], curves_[1], fx_, base_),
            self.leg2.cashflows(curves_[2], curves_[3], fx_, base_),
        ]
        _: DataFrame = DataFrame.from_records(seq)
        _.index = MultiIndex.from_tuples([("leg1", 0), ("leg2", 0)])
        return _

    def rate(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
    ) -> DualTypes:
        """
        Return the mid-market rate of the instrument.

        For arguments see :meth:`BaseMixin.rate<rateslib.instruments.BaseMixin.rate>`
        """
        curves_, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            fx,
            base,
            self.leg1.currency,
        )
        curves_1 = _validate_curve_not_no_input(curves_[1])
        curves_3 = _validate_curve_not_no_input(curves_[3])

        if isinstance(fx_, FXRates | FXForwards):
            imm_fx: DualTypes = fx_.rate(self.pair)
        elif isinstance(fx_, NoInput):
            raise ValueError(
                "`fx` must be supplied to price FXExchange object.\n"
                "Note: it can be attached to and then gotten from a Solver.",
            )
        else:
            imm_fx = fx_

        _: DualTypes = forward_fx(self.settlement, curves_1, curves_3, imm_fx)
        return _

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

    def analytic_delta(self, *args: Any, **kwargs: Any) -> DualTypes:
        raise NotImplementedError("`analytic_delta` for FXExchange not defined.")


class NDF(Sensitivities, Metrics):
    """
    Create a non-deliverable forward (NDF).

    Parameters
    ----------
    settlement: datetime or str
        The date on which settlement will occur. String tenors are allowed, e.g. "3M".
    pair: str
        The FX pair against which settlement takes place (2 x 3-digit code).
    notional: float, Variable, optional
        The notional amount expressed in terms of buying units of the *reference currency*, and not
        *settlement currency*.
    currency: str, optional
        The *settlement currency* of the contract. If not given is assumed to be currency 2 of the
        ``pair``, e.g. USD in BRLUSD. Must be one of the currencies in ``pair``. The
        *reference currency* is inferred as the other currency in the ``pair``.
    fx_rate: float, Variable, optional
        The agreed price on the NDF contact. May be omitted for unpriced contracts.
    fx_fixing: float, Variable, optional
        The rate against which settlement takes place. Will be forecast if not given or not known.
    eval_date: datetime, optional
        Required only if ``settlement`` is given as string tenor. Should be entered as today
        (also called horizon), and **not** spot. Spot is derived from ``payment_lag`` and
        ``calendar``.
    calendar: str or Calendar, optional
        Determines settlement if given as string tenor and fixing date from settlement.
    modifier: str, optional
        Date modifier for determining string tenor.
    payment_lag: int, optional
        Number of business day until settlement delivery. Defaults to 2 (spot) if not given.
        Used to derive the ``fx_fixing`` date and *spot* if using an ``eval_date``.
    eom: bool, optional
        Whether to allow end of month rolls to ``settlement`` as tenor.
    curves : Curve, str or list of such, optional
        Only one curve is required for an *NDF*. This curve should discount cashflows in the
        given ``currency`` at a known collateral rate.
    spec : str, optional
        An identifier to pre-populate many fields with conventional values. See
        :ref:`here<defaults-doc>` for more info and available values.

    Examples
    --------
    This is a standard *non-deliverable forward* with the ``notional`` expressed in EUR
    settling in USD.

    .. ipython:: python
       :suppress:

       from rateslib.instruments import NDF

    .. ipython:: python

       ndf = NDF(
           settlement="3m",
           pair="eurusd",  # <- EUR is the reference currency
           currency="usd",  # <- USD is the settlement currency
           notional=1e6,  # <- this is long 1mm EUR
           eval_date=dt(2000, 7, 1),
           calendar="tgt|fed",
           modifier="mf",
           payment_lag=2,
           fx_rate=1.04166666,  # (=125/120) <- implies short 1.0416mm USD
        )

    This has the ``pair`` reversed but the notional is still expressed in EUR.

    .. ipython:: python

       ndf = NDF(
           settlement="3m",
           pair="usd",  # <- EUR is still reference currency
           currency="usd",  # <- USD is the settlement currency
           notional=1e6,  # <- this is long 1mm EUR
           eval_date=dt(2000, 7, 1),
           calendar="tgt|fed",
           modifier="mf",
           payment_lag=2,
           fx_rate=0.96,  # (=120/125) <- implies short 1.0416mm USD
        )
    """

    periods: tuple[NonDeliverableCashflow, Cashflow]

    def __init__(
        self,
        settlement: datetime | str,
        pair: str,
        notional: DualTypes_ = NoInput(0),
        fx_rate: DualTypes_ = NoInput(0),
        fx_fixing: DualTypes_ = NoInput(0),
        eval_date: datetime_ = NoInput(0),
        calendar: CalInput = NoInput(0),
        modifier: str_ = NoInput(0),
        currency: str_ = NoInput(0),
        payment_lag: int_ = NoInput(0),
        eom: bool_ = NoInput(0),
        curves: Curves_ = NoInput(0),
        spec: str_ = NoInput(0),
    ):
        self.kwargs: dict[str, Any] = dict(
            pair=pair.lower(),
            currency=_drb(pair[3:], currency).lower(),
            notional=notional,
            fx_rate=fx_rate,
            settlement=settlement,
            fx_fixing=fx_fixing,
            eval_date=eval_date,
            calendar=get_calendar(calendar),
            modifier=modifier,
            payment_lag=payment_lag,
            eom=eom,
        )
        self.kwargs = _push(spec, self.kwargs)

        # set some defaults if missing
        default_kws = {
            "modifier": defaults.modifier,
            "notional": defaults.notional,
            "payment_lag": defaults.payment_lag_specific[type(self).__name__],
            "eom": defaults.eom_fx,
        }
        self.kwargs = _update_with_defaults(self.kwargs, default_kws)

        self.kwargs["fixing_date"], self.kwargs["settlement"] = _get_fx_expiry_and_delivery(
            eval_date,
            self.kwargs["settlement"],
            self.kwargs["payment_lag"],
            self.kwargs["calendar"],
            self.kwargs["modifier"],
            self.kwargs["eom"],
        )

        if self.kwargs["currency"] not in self.kwargs["pair"]:
            raise ValueError("`currency` must be one of the currencies in `pair`.")
        reference_currency = (
            self.kwargs["pair"][0:3]
            if self.kwargs["pair"][0:3] != self.kwargs["currency"]
            else self.kwargs["pair"][3:]
        )

        self.periods = (
            NonDeliverableCashflow(
                notional=-self.kwargs["notional"],
                currency=reference_currency,
                settlement_currency=self.kwargs["currency"],
                payment=self.kwargs["settlement"],
                fixing_date=self.kwargs["calendar"].lag(
                    self.kwargs["settlement"], -self.kwargs["payment_lag"], False
                ),  # a fixing date can be on a non-settlable date
                fx_fixing=self.kwargs["fx_fixing"],
                reversed=self.kwargs["pair"][0:3] == self.kwargs["currency"],
            ),
            Cashflow(
                notional=0.0,  # will be set by set_cashflow_notional
                currency=self.kwargs["currency"],
                payment=self.kwargs["settlement"],
                stub_type=self.kwargs["pair"].upper(),
                rate=self.kwargs["fx_rate"],
            ),
        )
        self._set_cashflow_notional(NoInput(0), init=True)
        self.curves = curves
        self.spec = spec

    @property
    def _unpriced(self) -> bool:
        return isinstance(self.kwargs["fx_rate"], NoInput)

    def _set_cashflow_notional(self, fx: FX_, init: bool) -> None:
        """
        Sets the notionals on the *Cashflow* types of the NDF.

        Parameters
        ----------
        init: bool
            Flag to indicate if the instance method is being run at initialisation, i.e. first time.
        """
        # set the notional based on direction of ``pair`` relative to the ``currency``.
        if init:
            if self._unpriced:
                return None  # do nothing - wait for price time to set mid-market
            else:
                fx_rate: DualTypes = self.kwargs["fx_rate"]
        else:
            if self._unpriced:
                if isinstance(fx, FXForwards):
                    fx_rate = fx.rate(self.kwargs["pair"], self.kwargs["settlement"])
                else:
                    fx_rate = _get_fx_fixings_from_non_fx_forwards(0, 1)[0]
                fx_rate = _dual_float(fx_rate)  # priced insts set parameters to float for risk.
            else:
                return None  # do nothing - already set at init using priced fx_rate

        # set pricing notional
        if self.kwargs["currency"] == self.kwargs["pair"][3:]:
            self.periods[1].notional = self.kwargs["notional"] * fx_rate
        else:
            self.periods[1].notional = self.kwargs["notional"] / fx_rate

    def _set_pricing_mid(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
    ) -> None:
        if self._unpriced:
            self._set_cashflow_notional(fx, init=False)

    def rate(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
    ) -> DualTypes:
        """Return the mid-market pricing parameter of the *NDF*.

        Parameters
        ----------
        curves: str or Curve
            Not used by *NDF.rate*.
        solver: Solver, optional
            The numerical :class:`~rateslib.solver.Solver` which may contain a mapping of
            ``curves``, ``fx`` and/or ``base``.
        fx: FXForwards, optional
            An FXForwards market for forecasting.
        base: str, optional
            Not used by *NDF.rate*.

        Returns
        -------
        float, Dual, Dual2
        """
        curves_, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            fx,
            base,
            self.kwargs["currency"],
        )
        return self.periods[0].rate(fx_)

    def cashflows(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
    ) -> DataFrame:
        """
        Return the cashflows of the *NDF*.

        For arguments see :meth:`BaseMixin.cashflows<rateslib.instruments.BaseMixin.cashflows>`
        """
        self._set_pricing_mid(curves, solver, fx)
        curves_, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            fx,
            base,
            NoInput(0),
        )
        seq = [
            self.periods[0].cashflows(curves_[0], curves_[1], fx_, base_),
            self.periods[1].cashflows(curves_[0], curves_[1], fx_, base_),
        ]
        _: DataFrame = DataFrame.from_records(seq)
        _.index = MultiIndex.from_tuples([("leg1", 0), ("leg1", 1)])
        return _

    def npv(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
        local: bool = False,
    ) -> NPV:
        """Return the NPV of the *NDF*.

        Parameters
        ----------
        curves: str or Curve
            The discount curve used for the settlement currency.
        solver: Solver, optional
            The numerical :class:`~rateslib.solver.Solver` which may contain a mapping of
            ``curves``, ``fx`` and/or ``base``.
        fx: FXForwards, optional
            An FXForwards market for forecasting.
        base: str, optional
            The base currency to convert cashflows into (3-digit code).
            Only used if ``fx`` is an FXRates or FXForwards object. If not given defaults to
            ``fx.base``.
        local: bool, optional
            If True will return a dict identifying NPV by settlement currency.

        Returns
        -------
        float, Dual, Dual2 or dict of such.
        """
        curves_, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            fx,
            base,
            self.kwargs["currency"],
        )
        self._set_pricing_mid(NoInput(0), NoInput(0), fx_)
        _ = self.periods[0].npv(NoInput(0), curves_[1], fx_, self.kwargs["currency"], local=False)
        _ += self.periods[1].npv(NoInput(0), curves_[1], fx_, self.kwargs["currency"], local=False)  # type: ignore[operator]
        return _maybe_local(_, local, self.kwargs["currency"], fx_, base_)

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

    def analytic_delta(self, *args: Any, **kwargs: Any) -> DualTypes:
        """
        Return the analytic delta of the *NDF*.

        Defined as zero.

        Returns
        -------
        float
        """
        return 0.0


class XCS(BaseDerivative):
    """
    Create a cross-currency swap (XCS) composing relevant fixed or floating *Legs*.

    MTM-XCSs will introduce a MTM *Leg* as *Leg2*.

    .. warning::

       ``leg2_notional`` is unused by *XCS*. That notional is always dynamically determined by
       ``fx_fixings``, i.e. an initial FX fixing and/or forecast forward FX rates if ``leg2_mtm``
       is set to *True*. See also the parameter definition for ``fx_fixings``.

    Parameters
    ----------
    args : tuple
        Required positional arguments for :class:`~rateslib.instruments.BaseDerivative`.
    fixed : bool, optional
        Whether *leg1* is fixed or floating rate. Defaults to *False*.
    payment_lag_exchange : int
        The number of business days by which to delay notional exchanges, aligned with
        the accrual schedule.
    fixed_rate : float, optional
        If ``fixed``, the fixed rate of *leg1*.
    float_spread : float, optional
        If not ``fixed``, the spread applied to the :class:`~rateslib.legs.FloatLeg`. Can be set to
        `None` and designated
        later, perhaps after a mid-market spread for all periods has been calculated.
    spread_compound_method : str, optional
        If not ``fixed``, the method to use for adding a floating spread to compounded rates.
        Available options are `{"none_simple", "isda_compounding", "isda_flat_compounding"}`.
    fixings : float, list, or Series optional
        If not ``fixed``, then if a float scalar, will be applied as the determined fixing for
        the first period. If a list of *n* fixings will be used as the fixings for the first *n*
        periods. If any sublist of length *m* is given, is used as the first *m* RFR
        fixings for that :class:`~rateslib.periods.FloatPeriod`. If a datetime
        indexed ``Series`` will use the fixings that are available in that object,
        and derive the rest from the ``curve``.
    fixing_method : str, optional
        If not ``fixed``, the method by which floating rates are determined, set by default.
        See notes.
    method_param : int, optional
        If not ``fixed`` A parameter that is used for the various ``fixing_method`` s. See notes.
    leg2_fixed : bool, optional
        Whether *leg2* is fixed or floating rate. Defaults to *False*
    leg2_mtm : bool optional
        Whether *leg2* is a mark-to-market leg. Defaults to *True*
    leg2_payment_lag_exchange : int
        The number of business days by which to delay notional exchanges, aligned with
        the accrual schedule.
    leg2_fixed_rate : float, optional
        If ``leg2_fixed``, the fixed rate of *leg2*.
    leg2_float_spread : float, optional
        If not ``leg2_fixed``, the spread applied to the :class:`~rateslib.legs.FloatLeg`.
        Can be set to `None` and designated
        later, perhaps after a mid-market spread for all periods has been calculated.
    leg2_spread_compound_method : str, optional
        If not ``leg2_fixed``, the method to use for adding a floating spread to compounded rates.
        Available options are `{"none_simple", "isda_compounding", "isda_flat_compounding"}`.
    leg2_fixings : float, list, or Series optional
        If not ``leg2_fixed``, then if a float scalar, will be applied as the determined fixing for
        the first period. If a list of *n* fixings will be used as the fixings for the first *n*
        periods. If any sublist of length *m* is given, is used as the first *m* RFR
        fixings for that :class:`~rateslib.periods.FloatPeriod`. If a datetime
        indexed ``Series`` will use the fixings that are available in that object,
        and derive the rest from the ``curve``.
    leg2_fixing_method : str, optional
        If not ``leg2_fixed``, the method by which floating rates are determined, set by default.
        See notes.
    leg2_method_param : int, optional
        If not ``leg2_fixed`` A parameter that is used for the various ``fixing_method`` s.
        See notes.
    fx_fixings : float, Dual, Dual2, list of such, optional
        Specify a known initial FX fixing or a list of such for ``mtm`` legs, where leg 1 is
        considered the domestic currency. For example for an ESTR/SOFR XCS in 100mm EUR notional
        a value of 1.10 EURUSD for fx_fixings implies the notional on leg 2 is 110m USD. Fixings
        that are not specified will be forecast at pricing time with an
        :class:`~rateslib.fx.FXForwards` object.
    kwargs : dict
        Required keyword arguments for :class:`~rateslib.instruments.BaseDerivative`.
    """

    leg1: FixedLeg | FloatLeg
    leg2: FixedLeg | FloatLeg | FloatLegMtm | FixedLegMtm  # type: ignore[assignment]

    def __init__(
        self,
        *args: Any,
        fixed: bool_ = NoInput(0),
        payment_lag_exchange: int_ = NoInput(0),
        fixed_rate: DualTypes_ = NoInput(0),
        float_spread: DualTypes_ = NoInput(0),
        spread_compound_method: str_ = NoInput(0),
        fixings: FixingsRates_ = NoInput(0),  # type: ignore[type-var]
        fixing_method: str_ = NoInput(0),
        method_param: int_ = NoInput(0),
        leg2_fixed: bool_ = NoInput(0),
        leg2_mtm: bool_ = NoInput(0),
        leg2_payment_lag_exchange: int_ = NoInput(1),
        leg2_fixed_rate: DualTypes_ = NoInput(0),
        leg2_float_spread: DualTypes_ = NoInput(0),
        leg2_fixings: FixingsRates_ = NoInput(0),
        leg2_fixing_method: str_ = NoInput(0),
        leg2_method_param: int_ = NoInput(0),
        leg2_spread_compound_method: str_ = NoInput(0),
        fx_fixings: FixingsFx_ = NoInput(0),  # type: ignore[type-var]
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        # set defaults for missing values
        default_kwargs = dict(
            fixed=False if isinstance(fixed, NoInput) else fixed,
            leg2_fixed=False if isinstance(leg2_fixed, NoInput) else leg2_fixed,
            leg2_mtm=True if isinstance(leg2_mtm, NoInput) else leg2_mtm,
        )
        self.kwargs: dict[str, Any] = _update_not_noinput(self.kwargs, default_kwargs)

        if self.kwargs["fixed"]:
            self.kwargs.pop("spread_compound_method", None)
            self.kwargs.pop("fixing_method", None)
            self.kwargs.pop("method_param", None)
            self._fixed_rate_mixin = True
            self._fixed_rate = fixed_rate
            leg1_user_kwargs: dict[str, Any] = dict(fixed_rate=fixed_rate)
            Leg1: type[FixedLeg] | type[FloatLeg] = FixedLeg
        else:
            self._rate_scalar = 100.0
            self._float_spread_mixin = True
            self._float_spread = float_spread
            leg1_user_kwargs = dict(
                float_spread=float_spread,
                spread_compound_method=spread_compound_method,
                fixings=fixings,
                fixing_method=fixing_method,
                method_param=method_param,
            )
            Leg1 = FloatLeg
        leg1_user_kwargs.update(
            dict(
                payment_lag_exchange=payment_lag_exchange,
                initial_exchange=True,
                final_exchange=True,
            ),
        )

        if leg2_payment_lag_exchange is NoInput.inherit:
            leg2_payment_lag_exchange = payment_lag_exchange
        if self.kwargs["leg2_fixed"]:
            self.kwargs.pop("leg2_spread_compound_method", None)
            self.kwargs.pop("leg2_fixing_method", None)
            self.kwargs.pop("leg2_method_param", None)
            self._leg2_fixed_rate_mixin = True
            self._leg2_fixed_rate = leg2_fixed_rate
            leg2_user_kwargs: dict[str, Any] = dict(leg2_fixed_rate=leg2_fixed_rate)
            Leg2: type[FloatLeg] | type[FixedLeg] | type[FloatLegMtm] | type[FixedLegMtm] = (
                FixedLeg if not leg2_mtm else FixedLegMtm
            )
        else:
            self._leg2_float_spread_mixin = True
            self._leg2_float_spread = leg2_float_spread
            leg2_user_kwargs = dict(
                leg2_float_spread=leg2_float_spread,
                leg2_spread_compound_method=leg2_spread_compound_method,
                leg2_fixings=leg2_fixings,
                leg2_fixing_method=leg2_fixing_method,
                leg2_method_param=leg2_method_param,
            )
            Leg2 = FloatLeg if not leg2_mtm else FloatLegMtm
        leg2_user_kwargs.update(
            dict(
                leg2_payment_lag_exchange=leg2_payment_lag_exchange,
                leg2_initial_exchange=True,
                leg2_final_exchange=True,
            ),
        )

        if self.kwargs["leg2_mtm"]:
            leg2_user_kwargs.update(
                dict(
                    leg2_alt_currency=self.kwargs["currency"],
                    leg2_alt_notional=-self.kwargs["notional"],
                    leg2_fx_fixings=fx_fixings,
                ),
            )

        self.kwargs = _update_not_noinput(self.kwargs, {**leg1_user_kwargs, **leg2_user_kwargs})

        self.leg1 = Leg1(**_get(self.kwargs, leg=1, filter=("fixed",)))
        self.leg2 = Leg2(**_get(self.kwargs, leg=2, filter=("leg2_fixed", "leg2_mtm")))
        self._initialise_fx_fixings(fx_fixings)

    def _initialise_fx_fixings(self, fx_fixings: FixingsFx_) -> None:
        """
        Sets the `fx_fixing` for non-mtm XCS instruments, which require only a single
        value.
        """
        if not isinstance(self.leg2, FloatLegMtm | FixedLegMtm):
            # then we are not dealing with mark-to-market so only a single FX fixing is required
            self.pair = self.leg1.currency + self.leg2.currency

            # if self.fx_fixing is NoInput this indicates the swap is unfixed and will be set
            # later (i.e. at price time). In that case, for the sake of obtaining reasonable
            # delta risks, which assume a mid-market priced derivative, any forecast FX fixing
            # is converted to float and affixed to the Instrument specification.
            # If a fixing is given directly by a user including AD that AD is not downcast to float
            # (i.e. it is assumed the user knows what they are doing) and is maintained.
            # Users passing float will be, possibly ignorantly, unaffected.
            if isinstance(fx_fixings, FXForwards):
                self.fx_fixings = _dual_float(
                    fx_fixings.rate(self.pair, self.leg2._exchange_periods[0].payment)  # type: ignore[union-attr]
                )
            elif isinstance(fx_fixings, FXRates):
                self.fx_fixings = _dual_float(fx_fixings.rate(self.pair))
            elif isinstance(fx_fixings, float | Dual | Dual2 | Variable):
                # If a fixing is input directly
                self.fx_fixings = fx_fixings
            else:
                self._fx_fixings: FixingsFx_ = NoInput(0)
                return None  # cannot set leg2_notional yet (wait for price time)
            self._set_leg2_notional_nonmtm(self.fx_fixings)
        else:
            self._fx_fixings = fx_fixings

    @property
    def fx_fixings(self) -> FixingsFx_:
        return self._fx_fixings

    @fx_fixings.setter
    def fx_fixings(self, value: FixingsFx_) -> None:
        self._fx_fixings = value

    def _set_fx_fixings(self, fx: FX_) -> None:
        """
        Checks the `fx_fixings` and sets them according to given object if null.

        Used by ``rate`` and ``npv`` methods when ``fx_fixings`` are not
        initialised but required for pricing and can be inferred from an FX object.
        """
        if not isinstance(
            self.leg2, FloatLegMtm | FixedLegMtm
        ):  # then we manage the initial FX from the pricing object.
            if isinstance(self.fx_fixings, NoInput):
                if isinstance(fx, NoInput):
                    if defaults.no_fx_fixings_for_xcs.lower() == "raise":
                        raise ValueError(
                            "`fx` is required when `fx_fixings` is not pre-set and "
                            "if rateslib option `no_fx_fixings_for_xcs` is set to "
                            "'raise'.",
                        )
                    else:
                        fx_fixing: DualTypes = 1.0
                        if defaults.no_fx_fixings_for_xcs.lower() == "warn":
                            warnings.warn(
                                "Using 1.0 for FX, no `fx` or `fx_fixings` given and "
                                "rateslib option `no_fx_fixings_for_xcs` is set to "
                                "'warn'.",
                                UserWarning,
                            )
                else:
                    if isinstance(fx, FXForwards):
                        # this is the correct pricing path
                        fx_fixing = fx.rate(self.pair, self.leg2._exchange_periods[0].payment)  # type: ignore[union-attr]
                    elif isinstance(fx, FXRates):
                        # maybe used in debugging
                        fx_fixing = fx.rate(self.pair)
                    else:
                        # possible float used in debugging also
                        fx_fixing = fx
                self._set_leg2_notional_nonmtm(fx_fixing)
        else:
            self._set_leg2_notional_mtm(fx)

    def _set_leg2_notional_mtm(self, fx: FX_) -> None:
        """
        Update the notional on leg2 (foreign leg) if the fx fixings are unknown

        ----------
        fx : DualTypes, FXRates or FXForwards
            For MTM XCSs this input must be ``FXForwards``.
            The FX object from which to determine FX rates used as the initial
            notional fixing, and to determine MTM cashflow exchanges.
        """
        # only called if leg2 is MTM type
        self.leg2._set_periods_mtm(fx)  # type: ignore[union-attr]
        self.leg2_notional = self.leg2.notional

    def _set_leg2_notional_nonmtm(self, fx: DualTypes) -> None:
        """
        Update the notional on leg2 (foreign leg) based on a given fixing.

        Parameters
        ----------
        fx : DualTypes
            Multiplies the leg1 notional to derive a leg2 notional.
        """
        self.leg2_notional = self.leg1.notional * -fx
        self.leg2.notional = self.leg2_notional
        if not isinstance(self.kwargs["amortization"], NoInput):
            self.leg2_amortization = self.leg1.amortization * -fx
            self.leg2.amortization = self.leg2_amortization  # type: ignore[assignment]

    @property
    def _is_unpriced(self) -> bool:
        if getattr(self, "_unpriced", None) is True:
            return True
        if self._fixed_rate_mixin and self._leg2_fixed_rate_mixin:
            # Fixed/Fixed where one leg is unpriced.
            # Return True if at least one of the `fixed_rates` is NoInput
            return isinstance(self.fixed_rate, NoInput) or isinstance(self.leg2_fixed_rate, NoInput)
        elif self._fixed_rate_mixin and isinstance(self.fixed_rate, NoInput):
            # Fixed/Float where fixed leg is unpriced
            return True
        elif self._float_spread_mixin and isinstance(self.float_spread, NoInput):
            # Float leg1 where leg1 is
            pass  # goto 2)
        else:
            return False

        # 2) leg1 is Float
        # Return True if the pricing parameter on leg2 is NoInput, either if it is a floating
        # spread or a fixed rate
        return (self._leg2_fixed_rate_mixin and isinstance(self.leg2_fixed_rate, NoInput)) or (
            self._leg2_float_spread_mixin and isinstance(self.leg2_float_spread, NoInput)
        )

    def _set_pricing_mid(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
    ) -> None:
        leg: int = 1
        lookup = {
            1: ["_fixed_rate_mixin", "_float_spread_mixin"],
            2: ["_leg2_fixed_rate_mixin", "_leg2_float_spread_mixin"],
        }
        if self._leg2_fixed_rate_mixin and isinstance(self.leg2_fixed_rate, NoInput):
            # Fixed/Fixed or Float/Fixed
            leg = 2

        rate = self.rate(curves, solver, fx, leg=leg)
        if getattr(self, lookup[leg][0]):
            getattr(self, f"leg{leg}").fixed_rate = _dual_float(rate)
        elif getattr(self, lookup[leg][1]):
            getattr(self, f"leg{leg}").float_spread = _dual_float(rate)
        else:
            # this line should not be hit: internal code check
            raise AttributeError("BaseXCS leg1 must be defined fixed or float.")  # pragma: no cover

    def npv(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
        local: bool = False,
    ) -> NPV:
        """
        Return the NPV of the derivative by summing legs.

        .. warning::

           If ``fx_fixing`` has not been set for the instrument requires
           ``fx`` as an FXForwards object to dynamically determine this.

        See :meth:`BaseDerivative.npv`.
        """
        curves_, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            fx,
            base,
            self.leg1.currency,
        )

        if self._is_unpriced:
            self._set_pricing_mid(curves_, solver, fx_)

        self._set_fx_fixings(fx_)

        if isinstance(self.leg2, FloatLegMtm | FixedLegMtm):
            self.leg2._do_not_repeat_set_periods = True
            ret = super().npv(curves_, solver, fx_, base_, local)
            self.leg2._do_not_repeat_set_periods = False  # reset for next calculation
        else:
            ret = super().npv(curves_, solver, fx_, base_, local)

        return ret

    def rate(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        leg: int = 1,
    ) -> DualTypes:
        """
        Return the mid-market pricing parameter of the XCS.

        Parameters
        ----------
        curves : list of Curves
            A list defines the following curves in the order:

            - Forecasting :class:`~rateslib.curves.Curve` for leg1 (if floating).
            - Discounting :class:`~rateslib.curves.Curve` for leg1.
            - Forecasting :class:`~rateslib.curves.Curve` for leg2 (if floating).
            - Discounting :class:`~rateslib.curves.Curve` for leg2.
        solver : Solver, optional
            The numerical :class:`~rateslib.solver.Solver` that
            constructs :class:`~rateslib.curves.Curve` from calibrating instruments.
        fx : FXForwards, optional
            The FX forwards object that is used to determine the initial FX fixing for
            determining ``leg2_notional``, if not specified at initialisation, and for
            determining mark-to-market exchanges on mtm XCSs.
        leg : int in [1, 2]
            The leg whose pricing parameter is to be determined.

        Returns
        -------
        float, Dual or Dual2

        Notes
        -----
        Fixed legs have pricing parameter returned in percentage terms, and
        float legs have pricing parameter returned in basis point (bp) terms.

        If the ``XCS`` type is specified without a ``fixed_rate`` on any leg then an
        implied ``float_spread`` will return as its originaly value or zero since
        the fixed rate used
        for calculation is the implied mid-market rate including the
        current ``float_spread`` parameter.

        """
        curves_, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            fx,
            NoInput(0),
            self.leg1.currency,
        )

        if leg == 1:
            tgt_fore_curve, tgt_disc_curve = curves_[0], curves_[1]
            alt_fore_curve, alt_disc_curve = curves_[2], curves_[3]
        else:
            tgt_fore_curve, tgt_disc_curve = curves_[2], curves_[3]
            alt_fore_curve, alt_disc_curve = curves_[0], curves_[1]

        leg2 = 1 if leg == 2 else 2
        # tgt_str, alt_str = "" if leg == 1 else "leg2_", "" if leg2 == 1 else "leg2_"
        tgt_leg, alt_leg = getattr(self, f"leg{leg}"), getattr(self, f"leg{leg2}")
        base_ = tgt_leg.currency

        _is_float_tgt_leg = "Float" in type(tgt_leg).__name__
        _is_float_alt_leg = "Float" in type(alt_leg).__name__
        if not _is_float_alt_leg and isinstance(alt_leg.fixed_rate, NoInput):
            raise ValueError(
                "Cannot solve for a `fixed_rate` or `float_spread` where the "
                "`fixed_rate` on the non-solvable leg is NoInput.",
            )

        # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
        # Commercial use of this code, and/or copying and redistribution is prohibited.
        # Contact rateslib at gmail.com if this code is observed outside its intended sphere.

        if not _is_float_tgt_leg:
            tgt_leg_fixed_rate = tgt_leg.fixed_rate
            if isinstance(tgt_leg_fixed_rate, NoInput):
                # set the target fixed leg to a null fixed rate for calculation
                tgt_leg.fixed_rate = 0.0
            else:
                # set the fixed rate to a float for calculation and no Dual Type crossing PR: XXX
                tgt_leg.fixed_rate = _dual_float(tgt_leg_fixed_rate)

        self._set_fx_fixings(fx_)
        if isinstance(self.leg2, FloatLegMtm | FixedLegMtm):
            self.leg2._do_not_repeat_set_periods = True

        tgt_leg_npv = tgt_leg.npv(tgt_fore_curve, tgt_disc_curve, fx_, base_)
        alt_leg_npv = alt_leg.npv(alt_fore_curve, alt_disc_curve, fx_, base_)
        fx_a_delta = 1.0 if not isinstance(tgt_leg, FloatLegMtm | FixedLegMtm) else fx_
        _: DualTypes = tgt_leg._spread(
            -(tgt_leg_npv + alt_leg_npv),
            tgt_fore_curve,
            tgt_disc_curve,
            fx_a_delta,
        )

        specified_spd = 0.0
        if _is_float_tgt_leg and not isinstance(tgt_leg.float_spread, NoInput):
            specified_spd = tgt_leg.float_spread
        elif not _is_float_tgt_leg:
            specified_spd = tgt_leg.fixed_rate * 100

        _ += specified_spd

        if isinstance(self.leg2, FloatLegMtm | FixedLegMtm):
            self.leg2._do_not_repeat_set_periods = False  # reset the mtm calc

        return _ if _is_float_tgt_leg else _ * 0.01

    def spread(self, *args: Any, **kwargs: Any) -> DualTypes:
        """
        Alias for :meth:`~rateslib.instruments.BaseXCS.rate`
        """
        return self.rate(*args, **kwargs)

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

        if self._is_unpriced:
            self._set_pricing_mid(curves_, solver, fx_)

        self._set_fx_fixings(fx_)
        if isinstance(self.leg2, FloatLegMtm | FixedLegMtm):
            self.leg2._do_not_repeat_set_periods = True
            ret = super().cashflows(curves_, solver, fx_, base_)
            self.leg2._do_not_repeat_set_periods = False  # reset the mtm calc
        else:
            ret = super().cashflows(curves_, solver, fx_, base_)

        return ret

    def fixings_table(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
        approximate: bool = False,
        right: datetime | NoInput = NoInput(0),
    ) -> DataFrame:
        """
        Return a DataFrame of fixing exposures on any :class:`~rateslib.legs.FloatLeg` or
        :class:`~rateslib.legs.FloatLegMtm` associated with the *XCS*.

        Parameters
        ----------
        curves : Curve, str or list of such
            A list defines the following curves in the order:

            - Forecasting :class:`~rateslib.curves.Curve` for leg1 (if floating).
            - Discounting :class:`~rateslib.curves.Curve` for leg1.
            - Forecasting :class:`~rateslib.curves.Curve` for leg2 (if floating).
            - Discounting :class:`~rateslib.curves.Curve` for leg2.

        solver : Solver, optional
            The numerical :class:`~rateslib.solver.Solver` that constructs
            :class:`~rateslib.curves.Curve` from calibrating instruments.

            .. note::

               The arguments ``fx`` and ``base`` are unused and results are returned in
               local currency of each *Leg*.

        approximate : bool, optional
            Perform a calculation that is broadly 10x faster but potentially loses
            precision upto 0.1%.
        right : datetime, optional
            Only calculate fixing exposures upto and including this date.

        Returns
        -------
        DataFrame
        """
        curves_, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            fx,
            base,
            self.leg1.currency,
        )

        try:
            df1 = self.leg1.fixings_table(  # type: ignore[union-attr]
                curve=curves_[0],
                disc_curve=curves_[1],
                fx=fx_,
                base=base_,
                approximate=approximate,
                right=right,
            )
        except AttributeError:
            df1 = DataFrame(
                index=DatetimeIndex([], name="obs_dates"),
            )

        try:
            df2 = self.leg2.fixings_table(  # type: ignore[union-attr]
                curve=curves_[2],
                disc_curve=curves_[3],
                fx=fx_,
                base=base_,
                approximate=approximate,
                right=right,
            )
        except AttributeError:
            df2 = DataFrame(
                index=DatetimeIndex([], name="obs_dates"),
            )

        return _composit_fixings_table(df1, df2)


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# This code cannot be installed or executed on a corporate computer without a paid licence extension
# Contact info at rateslib.com if this code is observed outside its intended sphere of use.


class FXSwap(XCS):
    """
    Create an FX swap simulated via a *Fixed-Fixed* :class:`XCS`.

    Parameters
    ----------
    args : dict
        Required positional args to :class:`XCS`.
    pair : str, optional
        The FX pair, e.g. "eurusd" as 3-digit ISO codes. If not given, fallsback to the base
        implementation of *XCS* which defines separate inputs as ``currency`` and ``leg2_currency``.
        If overspecified, ``pair`` will dominate.
    fx_fixings : float, Variable, FXForwards, optional
        The initial FX fixing where leg 1 is considered the domestic currency. For
        example for a EURUSD FXSwap in 100mm EUR notional a value of 1.10
        implies the notional on leg 2 is 110m USD. If not given determines this
        dynamically.
    points : float, optional
        The pricing parameter for the FX Swap, which will determine the implicit
        fixed rate on leg2.
    split_notional : float, optional
        The accrued notional at termination of the domestic leg accounting for interest
        payable at domestic interest rates.
    kwargs : dict
        Required keyword arguments to :class:`XCS`.

    Notes
    -----
    .. warning::

       ``leg2_notional`` is determined by the ``fx_fixings`` either initialised or at price
       time and the value of ``notional``. The argument value of ``leg2_notional`` does
       not impact calculations.

    *FXSwaps* are technically complicated instruments. To define a fully **priced** *Instrument*
    they require at least two pricing parameters; ``fx_fixings`` and ``points``. If a
    ``split_notional`` is also given at initialisation it will be assumed to be a split notional
    *FXSwap*. If not, then it will not be assumed to be.

    If ``fx_fixings`` is given then the market pricing parameter ``points`` can be calculated.
    This is an unusual partially *priced* parametrisation, however, and a warning will be emitted.
    As before, if ``split_notional`` is given, or not, at initialisation the *FXSwap* will be
    assumed to be split notional or not.

    If the *FXSwap* is not initialised with any parameters this defines an **unpriced**
    *Instrument* and it will be assumed to be split notional, inline with interbank
    market standards. The mid-market rate of an unpriced FXSwap is the same regardless of whether
    it is split notional or not, albeit split notional FXSwaps result in smaller FX rate
    sensitivity.

    Other combinations of arguments, just providing ``points`` or ``split_notional`` or both of
    those will raise an error. An *FXSwap* cannot be parametrised by these in isolation. This is
    summarised in the below table.

    .. list-table::  Resultant initialisation dependent upon given pricing parameters.
       :widths: 10 10 10 70
       :header-rows: 1

       * - fx_fixings
         - points
         - split_notional
         - Result
       * - X
         - X
         - X
         - A fully *priced* instrument defined with split notionals.
       * - X
         - X
         -
         - A fully *priced* instruments without split notionals.
       * -
         -
         -
         - An *unpriced* instrument with assumed split notionals.
       * - X
         -
         - X
         - A partially priced instrument with split notionals. Warns about unconventionality.
       * - X
         -
         -
         - A partially priced instrument without split notionals. Warns about unconventionality.
       * -
         - X
         - X
         - Raises ValueError. Not allowable partially priced instrument.
       * -
         - X
         -
         - Raises ValueError. Not allowable partially priced instrument.
       * -
         -
         - X
         - Raises ValueError. Not allowable partially priced instrument.

    Examples
    --------
    To value the *FXSwap* we create *Curves* and :class:`~rateslib.fx.FXForwards`
    objects.

    .. ipython:: python

       usd = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.95}, id="usd")
       eur = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.97}, id="eur")
       eurusd = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.971}, id="eurusd")
       fxr = FXRates({"eurusd": 1.10}, settlement=dt(2022, 1, 3))
       fxf = FXForwards(
           fx_rates=fxr,
           fx_curves={"usdusd": usd, "eureur": eur, "eurusd": eurusd},
       )

    Then we define the *FXSwap*. This in an unpriced instrument.

    .. ipython:: python

       fxs = FXSwap(
           effective=dt(2022, 1, 18),
           termination=dt(2022, 4, 19),
           pair="usdeur",
           calendar="nyc",
           notional=1000000,
           curves=["usd", "usd", "eur", "eurusd"],
       )

    Now demonstrate the :meth:`~rateslib.instruments.FXSwap.npv` and
    :meth:`~rateslib.instruments.FXSwap.rate` methods:

    .. ipython:: python

       fxs.npv(curves=[None, usd, None, eurusd], fx=fxf)
       fxs.rate(curves=[None, usd, None, eurusd], fx=fxf)

    In the case of *FXSwaps*, whose mid-market price is the difference between two
    forward FX rates we can also derive this quantity using the independent
    :meth:`FXForwards.swap<rateslib.fx.FXForwards.swap>` method.

    .. ipython:: python

       fxf.swap("usdeur", [dt(2022, 1, 18), dt(2022, 4, 19)])

    The following is an example of a fully priced *FXSwap* with split notionals.

    .. ipython:: python

       fxs = FXSwap(
           effective=dt(2022, 1, 18),
           termination=dt(2022, 4, 19),
           pair="usdeur",
           calendar="nyc",
           notional=1000000,
           curves=["usd", "usd", "eur", "eurusd"],
           fx_fixings=0.90,
           split_notional=1001500,
           points=-49.0
       )
       fxs.npv(curves=[None, usd, None, eurusd], fx=fxf)
       fxs.cashflows(curves=[None, usd, None, eurusd], fx=fxf)
       fxs.cashflows_table(curves=[None, usd, None, eurusd], fx=fxf)

    """

    _unpriced = True
    leg1: FixedLeg
    leg2: FixedLeg

    def _parse_split_flag(
        self,
        fx_fixings: FX_,
        points: DualTypes_,
        split_notional: DualTypes_,
    ) -> None:
        """
        Determine the rules for a priced, unpriced or partially priced derivative and whether
        it is inferred as split notional or not.
        """
        is_none = [isinstance(_, NoInput) for _ in [fx_fixings, points, split_notional]]
        if all(is_none) or not any(is_none):
            self._is_split = True
        elif isinstance(split_notional, NoInput) and not any(
            isinstance(_, NoInput) for _ in [fx_fixings, points]
        ):
            self._is_split = False
        elif not isinstance(fx_fixings, NoInput):
            warnings.warn(
                "Initialising FXSwap with `fx_fixings` but without `points` is unconventional.\n"
                "Pricing can still be performed to determine `points`.",
                UserWarning,
            )
            if not isinstance(split_notional, NoInput):
                self._is_split = True
            else:
                self._is_split = False
        else:
            if not isinstance(points, NoInput):
                raise ValueError("Cannot initialise FXSwap with `points` but without `fx_fixings`.")
            else:
                raise ValueError(
                    "Cannot initialise FXSwap with `split_notional` but without `fx_fixings`",
                )

    def _set_split_notional(self, curve: Curve_ = NoInput(0), at_init: bool = False) -> None:
        """
        Will set the fixed rate, if not zero, for leg1, given the provided split notional or the
        forecast split notional calculated from a curve.

        self._split_notional is used as a temporary storage when mid-market price is determined.

        Parameters
        ----------
        curve: Curve, optional
            A curve used to determine the split notional in the case calculation is needed
        at_init: bool
            A construction flag to indicate if this method is being called during initialisation.
        """
        if not self._is_split:
            self._split_notional = self.kwargs["notional"]
            # fixed rate at zero remains
            return None

        # a split notional is given by a user and then this is set and never updated.
        elif not isinstance(self.kwargs["split_notional"], NoInput):
            if at_init:  # this will be run for one time only at initialisation
                self._split_notional = self.kwargs["split_notional"]
                self._set_leg1_fixed_rate()
            else:
                return None

        # else new pricing parameters will affect and unpriced split notional
        else:
            if at_init:
                self._split_notional = NoInput(0)
            else:
                if isinstance(curve, NoInput):
                    raise ValueError(
                        "A `curve` is required to determine a `split_notional` on an FXSwap if "
                        "the `split_notional` is not provided at initialisation."
                    )
                dt1, dt2 = self.leg1.periods[0].payment, self.leg1.periods[2].payment
                self._split_notional = self.kwargs["notional"] * curve[dt1] / curve[dt2]
                self._set_leg1_fixed_rate()

    def _set_leg1_fixed_rate(self) -> None:
        fixed_rate = (self.leg1.notional - self._split_notional) / (
            -self.leg1.notional * self.leg1._regular_periods[0].dcf
        )
        self.leg1.fixed_rate = fixed_rate * 100

    def __init__(
        self,
        *args: Any,
        pair: str_ = NoInput(0),
        fx_fixings: FX_ = NoInput(0),
        points: DualTypes_ = NoInput(0),
        split_notional: DualTypes_ = NoInput(0),
        **kwargs: Any,
    ) -> None:
        self._parse_split_flag(fx_fixings, points, split_notional)
        currencies = {}
        if isinstance(pair, str):
            # TODO for version 2.0 should look to deprecate 'currency' and 'leg2_currency' as
            #  allowable inputs.
            currencies = {"currency": pair.lower()[0:3], "leg2_currency": pair.lower()[3:6]}

        kwargs_overrides = dict(  # specific args for FXSwap passed to the Base XCS
            fixed=True,
            leg2_fixed=True,
            leg2_mtm=False,
            fixed_rate=0.0,
            frequency="Z",
            leg2_frequency="Z",
            leg2_fixed_rate=NoInput(0),
            fx_fixings=fx_fixings,
        )
        super().__init__(*args, **{**kwargs, **kwargs_overrides, **currencies})

        self.kwargs["split_notional"] = split_notional
        self._set_split_notional(curve=NoInput(0), at_init=True)
        # self._initialise_fx_fixings(fx_fixings)
        self.points = points

    @property
    def points(self) -> DualTypes_:
        return self._points

    @points.setter
    def points(self, value: DualTypes_) -> None:
        self._unpriced = False
        self._points = value
        self._leg2_fixed_rate = NoInput(0)

        # setting points requires leg1.notional leg1.split_notional, fx_fixing and points value

        if not isinstance(value, NoInput):
            # leg2 should have been properly set as part of fx_fixings and set_leg2_notional
            fx_fixing = self.leg2.notional / -self.leg1.notional

            _ = self._split_notional * (fx_fixing + value / 10000) + self.leg2.notional
            fixed_rate = _ / (self.leg2._regular_periods[0].dcf * -self.leg2.notional)

            self.leg2_fixed_rate = fixed_rate * 100

    # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
    # Commercial use of this code, and/or copying and redistribution is prohibited.
    # Contact rateslib at gmail.com if this code is observed outside its intended sphere.

    def _set_pricing_mid(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
    ) -> None:
        # This function ASSUMES that the instrument is unpriced, i.e. all of
        # split_notional, fx_fixing and points have been initialised as None.

        # first we set the split notional which is defined by interest rates on leg1.
        points = self.rate(curves, solver, fx)
        self.points = _dual_float(points)
        self._unpriced = True  # setting pricing mid does not define a priced instrument

    def rate(  # type: ignore[override]
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        fixed_rate: bool = False,
    ) -> DualTypes:
        """
        Return the mid-market pricing parameter of the FXSwapS.

        Parameters
        ----------
        curves : list of Curves
            A list defines the following curves in the order:

            - Forecasting :class:`~rateslib.curves.Curve` for leg1 (if floating).
            - Discounting :class:`~rateslib.curves.Curve` for leg1.
            - Forecasting :class:`~rateslib.curves.Curve` for leg2 (if floating).
            - Discounting :class:`~rateslib.curves.Curve` for leg2.
        solver : Solver, optional
            The numerical :class:`~rateslib.solver.Solver` that
            constructs :class:`~rateslib.curves.Curve` from calibrating instruments.
        fx : FXForwards, optional
            The FX forwards object that is used to determine the initial FX fixing for
            determining ``leg2_notional``, if not specified at initialisation, and for
            determining mark-to-market exchanges on mtm XCSs.
        fixed_rate : bool
            Whether to return the fixed rate for the leg or the FX swap points price.

        Returns
        -------
        float, Dual or Dual2
        """
        curves_, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            fx,
            NoInput(0),
            self.leg1.currency,
        )
        # set the split notional from the curve if not available
        self._set_split_notional(curve=curves_[1])
        # then we will set the fx_fixing and leg2 initial notional.

        # self._set_fx_fixings(fx) # this will be done by super().rate()
        leg2_fixed_rate = super().rate(curves_, solver, fx_, leg=2)

        if fixed_rate:
            return leg2_fixed_rate
        else:
            points: DualTypes = -self.leg2.notional * (
                (1 + leg2_fixed_rate * self.leg2._regular_periods[0].dcf / 100)
                / self._split_notional
                - 1 / self.kwargs["notional"]
            )
            return points * 10000

    def cashflows(
        self,
        curves: Curves_ = NoInput(0),
        solver: Solver_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
    ) -> DataFrame:
        if self._is_unpriced:
            self._set_pricing_mid(curves, solver, fx)
        ret: DataFrame = super().cashflows(curves, solver, fx, base)
        return ret
