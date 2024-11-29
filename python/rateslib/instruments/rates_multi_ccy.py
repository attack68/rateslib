from __future__ import annotations

import warnings
from datetime import datetime

from pandas import DataFrame, DatetimeIndex, MultiIndex, Series

from rateslib import defaults
from rateslib.curves import Curve
from rateslib.default import NoInput
from rateslib.dual import Dual, Dual2, DualTypes
from rateslib.fx import FXForwards, FXRates, forward_fx
from rateslib.instruments.core import (
    BaseMixin,
    Sensitivities,
    _composit_fixings_table,
    _get,
    _get_curves_fx_and_base_maybe_from_solver,
    _update_not_noinput,
)
from rateslib.instruments.rates_derivatives import BaseDerivative
from rateslib.legs import (
    FixedLeg,
    FixedLegMtm,
    FloatLeg,
    FloatLegMtm,
)
from rateslib.periods import (
    Cashflow,
)
from rateslib.solver import Solver

# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# This code cannot be installed or executed on a corporate computer without a paid licence extension
# Contact info at rateslib.com if this code is observed outside its intended sphere of use.


class FXExchange(Sensitivities, BaseMixin):
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

    def __init__(
        self,
        settlement: datetime,
        pair: str,
        fx_rate: float | NoInput = NoInput(0),
        notional: float | NoInput = NoInput(0),
        curves: list | str | Curve | NoInput = NoInput(0),
    ):
        self.curves = curves
        self.settlement = settlement
        self.pair = pair.lower()
        self.leg1 = Cashflow(
            notional=-defaults.notional if notional is NoInput.blank else -notional,
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
    def fx_rate(self):
        return self._fx_rate

    @fx_rate.setter
    def fx_rate(self, value):
        self._fx_rate = value
        self.leg2.notional = 0.0 if value is NoInput.blank else value * -self.leg1.notional
        self.leg2._rate = value

    def _set_pricing_mid(
        self,
        curves: Curve | str | list | NoInput = NoInput(0),
        solver: Solver | NoInput = NoInput(0),
        fx: float | FXRates | FXForwards | NoInput = NoInput(0),
    ):
        if self.fx_rate is NoInput.blank:
            mid_market_rate = self.rate(curves, solver, fx)
            self.fx_rate = float(mid_market_rate)
            self._fx_rate = NoInput(0)

    def npv(
        self,
        curves: Curve | str | list | NoInput = NoInput(0),
        solver: Solver | NoInput = NoInput(0),
        fx: float | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
        local: bool = False,
    ):
        """
        Return the NPV of the *FXExchange* by summing legs.

        For arguments see :meth:`BaseMixin.npv<rateslib.instruments.BaseMixin.npv>`
        """
        self._set_pricing_mid(curves, solver, fx)

        curves, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            fx,
            base,
            self.leg1.currency,
        )

        if fx_ is NoInput.blank:
            raise ValueError(
                "Must have some FX information to price FXExchange, either `fx` or "
                "`solver` containing an FX object.",
            )
        if not isinstance(fx_, (FXRates, FXForwards)):
            # force base_ leg1 currency to be converted consistent.
            leg1_npv = self.leg1.npv(curves[0], curves[1], fx_, base_, local)
            leg2_npv = self.leg2.npv(curves[2], curves[3], 1.0, base_, local)
            warnings.warn(
                "When valuing multi-currency derivatives it not best practice to "
                "supply `fx` as numeric.\nYour input:\n"
                f"`npv(solver={'None' if solver is NoInput.blank else '<Solver>'}, "
                f"fx={fx}, base='{base if base is not NoInput.blank else 'None'}')\n"
                "has been implicitly converted into the following by this operation:\n"
                f"`npv(solver={'None' if solver is NoInput.blank else '<Solver>'}, "
                f"fx=FXRates({{'{self.leg2.currency}{self.leg1.currency}: {fx}}}), "
                f"base='{self.leg2.currency}')\n.",
                UserWarning,
            )
        else:
            leg1_npv = self.leg1.npv(curves[0], curves[1], fx_, base_, local)
            leg2_npv = self.leg2.npv(curves[2], curves[3], fx_, base_, local)

        if local:
            return {
                k: leg1_npv.get(k, 0) + leg2_npv.get(k, 0) for k in set(leg1_npv) | set(leg2_npv)
            }
        else:
            return leg1_npv + leg2_npv

    def cashflows(
        self,
        curves: Curve | str | list | NoInput = NoInput(0),
        solver: Solver | NoInput = NoInput(0),
        fx: float | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
    ):
        """
        Return the cashflows of the *FXExchange* by aggregating legs.

        For arguments see :meth:`BaseMixin.npv<rateslib.instruments.BaseMixin.cashflows>`
        """
        self._set_pricing_mid(curves, solver, fx)
        curves, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            fx,
            base,
            NoInput(0),
        )
        seq = [
            self.leg1.cashflows(curves[0], curves[1], fx_, base_),
            self.leg2.cashflows(curves[2], curves[3], fx_, base_),
        ]
        _ = DataFrame.from_records(seq)
        _.index = MultiIndex.from_tuples([("leg1", 0), ("leg2", 0)])
        return _

    def rate(
        self,
        curves: Curve | str | list | NoInput = NoInput(0),
        solver: Solver | NoInput = NoInput(0),
        fx: float | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
    ):
        """
        Return the mid-market rate of the instrument.

        For arguments see :meth:`BaseMixin.rate<rateslib.instruments.BaseMixin.rate>`
        """
        curves, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            fx,
            base,
            self.leg1.currency,
        )
        if isinstance(fx_, (FXRates, FXForwards)):
            imm_fx = fx_.rate(self.pair)
        else:
            imm_fx = fx_

        if imm_fx is NoInput.blank:
            raise ValueError(
                "`fx` must be supplied to price FXExchange object.\n"
                "Note: it can be attached to and then gotten from a Solver.",
            )
        _ = forward_fx(self.settlement, curves[1], curves[3], imm_fx)
        return _

    def delta(self, *args, **kwargs):
        """
        Calculate the delta of the *Instrument*.

        For arguments see :meth:`Sensitivities.delta()<rateslib.instruments.Sensitivities.delta>`.
        """
        return super().delta(*args, **kwargs)

    def gamma(self, *args, **kwargs):
        """
        Calculate the gamma of the *Instrument*.

        For arguments see :meth:`Sensitivities.gamma()<rateslib.instruments.Sensitivities.gamma>`.
        """
        return super().gamma(*args, **kwargs)

    def analytic_delta(self, *args, **kwargs):
        raise NotImplementedError("`analytic_delta` for FXExchange not defined.")


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

    def __init__(
        self,
        *args,
        fixed: bool | NoInput = NoInput(0),
        payment_lag_exchange: int | NoInput = NoInput(0),
        fixed_rate: float | NoInput = NoInput(0),
        float_spread: float | NoInput = NoInput(0),
        spread_compound_method: str | NoInput = NoInput(0),
        fixings: float | list | Series | NoInput = NoInput(0),
        fixing_method: str | NoInput = NoInput(0),
        method_param: int | NoInput = NoInput(0),
        leg2_fixed: bool | NoInput = NoInput(0),
        leg2_mtm: bool | NoInput = NoInput(0),
        leg2_payment_lag_exchange: int | NoInput = NoInput(1),
        leg2_fixed_rate: float | NoInput = NoInput(0),
        leg2_float_spread: float | NoInput = NoInput(0),
        leg2_fixings: float | list | NoInput = NoInput(0),
        leg2_fixing_method: str | NoInput = NoInput(0),
        leg2_method_param: int | NoInput = NoInput(0),
        leg2_spread_compound_method: str | NoInput = NoInput(0),
        fx_fixings: list | DualTypes | FXRates | FXForwards | NoInput = NoInput(0),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # set defaults for missing values
        default_kwargs = dict(
            fixed=False if fixed is NoInput.blank else fixed,
            leg2_fixed=False if leg2_fixed is NoInput.blank else leg2_fixed,
            leg2_mtm=True if leg2_mtm is NoInput.blank else leg2_mtm,
        )
        self.kwargs = _update_not_noinput(self.kwargs, default_kwargs)

        if self.kwargs["fixed"]:
            self.kwargs.pop("spread_compound_method", None)
            self.kwargs.pop("fixing_method", None)
            self.kwargs.pop("method_param", None)
            self._fixed_rate_mixin = True
            self._fixed_rate = fixed_rate
            leg1_user_kwargs = dict(fixed_rate=fixed_rate)
            Leg1 = FixedLeg
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
            leg2_user_kwargs = dict(leg2_fixed_rate=leg2_fixed_rate)
            Leg2 = FixedLeg if not leg2_mtm else FixedLegMtm
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
            self._is_mtm = True
            leg2_user_kwargs.update(
                dict(
                    leg2_alt_currency=self.kwargs["currency"],
                    leg2_alt_notional=-self.kwargs["notional"],
                    leg2_fx_fixings=fx_fixings,
                ),
            )
        else:
            self._is_mtm = False

        self.kwargs = _update_not_noinput(self.kwargs, {**leg1_user_kwargs, **leg2_user_kwargs})

        self.leg1 = Leg1(**_get(self.kwargs, leg=1, filter=["fixed"]))
        self.leg2 = Leg2(**_get(self.kwargs, leg=2, filter=["leg2_fixed", "leg2_mtm"]))
        self._initialise_fx_fixings(fx_fixings)

    @property
    def fx_fixings(self):
        return self._fx_fixings

    @fx_fixings.setter
    def fx_fixings(self, value):
        self._fx_fixings = value
        self._set_leg2_notional(value)

    def _initialise_fx_fixings(self, fx_fixings):
        """
        Sets the `fx_fixing` for non-mtm XCS instruments, which require only a single
        value.
        """
        if not self._is_mtm:
            self.pair = self.leg1.currency + self.leg2.currency
            # if self.fx_fixing is NoInput.blank this indicates the swap is unfixed and will be set
            # later. If a fixing is given this means the notional is fixed without any
            # further sensitivity, hence the downcast to a float below.
            if isinstance(fx_fixings, FXForwards):
                self.fx_fixings = float(fx_fixings.rate(self.pair, self.leg2.periods[0].payment))
            elif isinstance(fx_fixings, FXRates):
                self.fx_fixings = float(fx_fixings.rate(self.pair))
            elif isinstance(fx_fixings, (float, Dual, Dual2)):
                self.fx_fixings = float(fx_fixings)
            else:
                self._fx_fixings = NoInput(0)
        else:
            self._fx_fixings = fx_fixings

    def _set_fx_fixings(self, fx):
        """
        Checks the `fx_fixings` and sets them according to given object if null.

        Used by ``rate`` and ``npv`` methods when ``fx_fixings`` are not
        initialised but required for pricing and can be inferred from an FX object.
        """
        if not self._is_mtm:  # then we manage the initial FX from the pricing object.
            if self.fx_fixings is NoInput.blank:
                if fx is NoInput.blank:
                    if defaults.no_fx_fixings_for_xcs.lower() == "raise":
                        raise ValueError(
                            "`fx` is required when `fx_fixings` is not pre-set and "
                            "if rateslib option `no_fx_fixings_for_xcs` is set to "
                            "'raise'.",
                        )
                    else:
                        fx_fixing = 1.0
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
                        fx_fixing = fx.rate(self.pair, self.leg2.periods[0].payment)
                    elif isinstance(fx, FXRates):
                        # maybe used in debugging
                        fx_fixing = fx.rate(self.pair)
                    else:
                        # possible float used in debugging also
                        fx_fixing = fx
                self._set_leg2_notional(fx_fixing)
        else:
            self._set_leg2_notional(fx)

    def _set_leg2_notional(self, fx_arg: float | FXForwards):
        """
        Update the notional on leg2 (foreign leg) if the initial fx rate is unfixed.

        ----------
        fx_arg : float or FXForwards
            For non-MTM XCSs this input must be a float.
            The FX rate to use as the initial notional fixing.
            Will only update the leg if ``NonMtmXCS.fx_fixings`` has been initially
            set to `None`.

            For MTM XCSs this input must be ``FXForwards``.
            The FX object from which to determine FX rates used as the initial
            notional fixing, and to determine MTM cashflow exchanges.
        """
        if self._is_mtm:
            self.leg2._set_periods(fx_arg)
            self.leg2_notional = self.leg2.notional
        else:
            self.leg2_notional = self.leg1.notional * -fx_arg
            self.leg2.notional = self.leg2_notional
            if self.kwargs["amortization"] is not NoInput.blank:
                self.leg2_amortization = self.leg1.amortization * -fx_arg
                self.leg2.amortization = self.leg2_amortization

    @property
    def _is_unpriced(self):
        if getattr(self, "_unpriced", None) is True:
            return True
        if self._fixed_rate_mixin and self._leg2_fixed_rate_mixin:
            # Fixed/Fixed where one leg is unpriced.
            if self.fixed_rate is NoInput.blank or self.leg2_fixed_rate is NoInput.blank:  # noqa: SIM103
                return True  # noqa: SIM103
            return False  # noqa: SIM103
        elif self._fixed_rate_mixin and self.fixed_rate is NoInput.blank:
            # Fixed/Float where fixed leg is unpriced
            return True
        elif self._float_spread_mixin and self.float_spread is NoInput.blank:
            # Float leg1 where leg1 is
            pass  # goto 2)
        else:
            return False

        # 2) leg1 is Float
        if self._leg2_fixed_rate_mixin and self.leg2_fixed_rate is NoInput.blank:  # noqa: SIM114, SIM103
            return True  # noqa: SIM114, SIM103
        elif self._leg2_float_spread_mixin and self.leg2_float_spread is NoInput.blank:  # noqa: SIM114, SIM103
            return True  # noqa: SIM114, SIM103
        else:  # noqa: SIM114, SIM103
            return False  # noqa: SIM114, SIM103

    def _set_pricing_mid(
        self,
        curves: Curve | str | list | NoInput = NoInput(0),
        solver: Solver | NoInput = NoInput(0),
        fx: FXForwards | NoInput = NoInput(0),
    ):
        leg: int = 1
        lookup = {
            1: ["_fixed_rate_mixin", "_float_spread_mixin"],
            2: ["_leg2_fixed_rate_mixin", "_leg2_float_spread_mixin"],
        }
        if self._leg2_fixed_rate_mixin and self.leg2_fixed_rate is NoInput.blank:
            # Fixed/Fixed or Float/Fixed
            leg = 2

        rate = self.rate(curves, solver, fx, leg=leg)
        if getattr(self, lookup[leg][0]):
            getattr(self, f"leg{leg}").fixed_rate = float(rate)
        elif getattr(self, lookup[leg][1]):
            getattr(self, f"leg{leg}").float_spread = float(rate)
        else:
            # this line should not be hit: internal code check
            raise AttributeError("BaseXCS leg1 must be defined fixed or float.")  # pragma: no cover

    def npv(
        self,
        curves: Curve | str | list | NoInput = NoInput(0),
        solver: Solver | NoInput = NoInput(0),
        fx: FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
        local: bool = False,
    ):
        """
        Return the NPV of the derivative by summing legs.

        .. warning::

           If ``fx_fixing`` has not been set for the instrument requires
           ``fx`` as an FXForwards object to dynamically determine this.

        See :meth:`BaseDerivative.npv`.
        """
        curves, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            fx,
            base,
            self.leg1.currency,
        )

        if self._is_unpriced:
            self._set_pricing_mid(curves, solver, fx_)

        self._set_fx_fixings(fx_)
        if self._is_mtm:
            self.leg2._do_not_repeat_set_periods = True

        ret = super().npv(curves, solver, fx_, base_, local)
        if self._is_mtm:
            self.leg2._do_not_repeat_set_periods = False  # reset for next calculation
        return ret

    def rate(
        self,
        curves: Curve | str | list | NoInput = NoInput(0),
        solver: Solver | NoInput = NoInput(0),
        fx: FXForwards | NoInput = NoInput(0),
        leg: int = 1,
    ):
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

        Examples
        --------
        """
        curves, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            fx,
            NoInput(0),
            self.leg1.currency,
        )

        if leg == 1:
            tgt_fore_curve, tgt_disc_curve = curves[0], curves[1]
            alt_fore_curve, alt_disc_curve = curves[2], curves[3]
        else:
            tgt_fore_curve, tgt_disc_curve = curves[2], curves[3]
            alt_fore_curve, alt_disc_curve = curves[0], curves[1]

        leg2 = 1 if leg == 2 else 2
        # tgt_str, alt_str = "" if leg == 1 else "leg2_", "" if leg2 == 1 else "leg2_"
        tgt_leg, alt_leg = getattr(self, f"leg{leg}"), getattr(self, f"leg{leg2}")
        base_ = tgt_leg.currency

        _is_float_tgt_leg = "Float" in type(tgt_leg).__name__
        _is_float_alt_leg = "Float" in type(alt_leg).__name__
        if not _is_float_alt_leg and alt_leg.fixed_rate is NoInput.blank:
            raise ValueError(
                "Cannot solve for a `fixed_rate` or `float_spread` where the "
                "`fixed_rate` on the non-solvable leg is NoInput.blank.",
            )

        # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
        # Commercial use of this code, and/or copying and redistribution is prohibited.
        # Contact rateslib at gmail.com if this code is observed outside its intended sphere.

        if not _is_float_tgt_leg:
            tgt_leg_fixed_rate = tgt_leg.fixed_rate
            if tgt_leg_fixed_rate is NoInput.blank:
                # set the target fixed leg to a null fixed rate for calculation
                tgt_leg.fixed_rate = 0.0
            else:
                # set the fixed rate to a float for calculation and no Dual Type crossing PR: XXX
                tgt_leg.fixed_rate = float(tgt_leg_fixed_rate)

        self._set_fx_fixings(fx_)
        if self._is_mtm:
            self.leg2._do_not_repeat_set_periods = True

        tgt_leg_npv = tgt_leg.npv(tgt_fore_curve, tgt_disc_curve, fx_, base_)
        alt_leg_npv = alt_leg.npv(alt_fore_curve, alt_disc_curve, fx_, base_)
        fx_a_delta = 1.0 if not tgt_leg._is_mtm else fx_
        _ = tgt_leg._spread(
            -(tgt_leg_npv + alt_leg_npv),
            tgt_fore_curve,
            tgt_disc_curve,
            fx_a_delta,
        )

        specified_spd = 0.0
        if _is_float_tgt_leg and tgt_leg.float_spread is not NoInput.blank:
            specified_spd = tgt_leg.float_spread
        elif not _is_float_tgt_leg:
            specified_spd = tgt_leg.fixed_rate * 100

        _ += specified_spd

        if self._is_mtm:
            self.leg2._do_not_repeat_set_periods = False  # reset the mtm calc

        return _ if _is_float_tgt_leg else _ * 0.01

    def spread(self, *args, **kwargs):
        """
        Alias for :meth:`~rateslib.instruments.BaseXCS.rate`
        """
        return self.rate(*args, **kwargs)

    def cashflows(
        self,
        curves: Curve | str | list | NoInput = NoInput(0),
        solver: Solver | NoInput = NoInput(0),
        fx: FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
    ):
        curves, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            fx,
            base,
            self.leg1.currency,
        )

        if self._is_unpriced:
            self._set_pricing_mid(curves, solver, fx_)

        self._set_fx_fixings(fx_)
        if self._is_mtm:
            self.leg2._do_not_repeat_set_periods = True

        ret = super().cashflows(curves, solver, fx_, base_)
        if self._is_mtm:
            self.leg2._do_not_repeat_set_periods = False  # reset the mtm calc
        return ret

    def fixings_table(
        self,
        curves: Curve | str | list | NoInput = NoInput(0),
        solver: Solver | NoInput = NoInput(0),
        fx: float | FXRates | FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
        approximate: bool = False,
        right: datetime | NoInput = NoInput(0),
    ):
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
        curves, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            fx,
            base,
            self.leg1.currency,
        )

        try:
            df1 = self.leg1.fixings_table(
                curve=curves[0],
                disc_curve=curves[1],
                fx=fx_,
                base=base_,
                approximate=approximate,
                right=right,
            )
        except AttributeError:
            df1 = DataFrame(
                index=DatetimeIndex([], name="obs_dates", freq=None),
            )

        try:
            df2 = self.leg2.fixings_table(
                curve=curves[2],
                disc_curve=curves[3],
                fx=fx_,
                base=base_,
                approximate=approximate,
                right=right,
            )
        except AttributeError:
            df2 = DataFrame(
                index=DatetimeIndex([], name="obs_dates", freq=None),
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
    fx_fixings : float, FXForwards or None
        The initial FX fixing where leg 1 is considered the domestic currency. For
        example for an ESTR/SOFR XCS in 100mm EUR notional a value of 1.10 for `fx0`
        implies the notional on leg 2 is 110m USD. If `None` determines this
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

    def _parse_split_flag(self, fx_fixings, points, split_notional):
        """
        Determine the rules for a priced, unpriced or partially priced derivative and whether
        it is inferred as split notional or not.
        """
        is_none = [_ is NoInput.blank for _ in [fx_fixings, points, split_notional]]
        if all(is_none) or not any(is_none):
            self._is_split = True
        elif split_notional is NoInput.blank and not any(
            _ is NoInput.blank for _ in [fx_fixings, points]
        ):
            self._is_split = False
        elif fx_fixings is not NoInput.blank:
            warnings.warn(
                "Initialising FXSwap with `fx_fixings` but without `points` is unconventional.\n"
                "Pricing can still be performed to determine `points`.",
                UserWarning,
            )
            if split_notional is not NoInput.blank:
                self._is_split = True
            else:
                self._is_split = False
        else:
            if points is not NoInput.blank:
                raise ValueError("Cannot initialise FXSwap with `points` but without `fx_fixings`.")
            else:
                raise ValueError(
                    "Cannot initialise FXSwap with `split_notional` but without `fx_fixings`",
                )

    def _set_split_notional(self, curve: Curve | NoInput = NoInput(0), at_init: bool = False):
        """
        Will set the fixed rate, if not zero, for leg1, given provided split not or forecast splnot.

        self._split_notional is used as a temporary storage when mid market price is determined.
        """
        if not self._is_split:
            self._split_notional = self.kwargs["notional"]
            # fixed rate at zero remains

        # a split notional is given by a user and then this is set and never updated.
        elif self.kwargs["split_notional"] is not NoInput.blank:
            if at_init:  # this will be run for one time only at initialisation
                self._split_notional = self.kwargs["split_notional"]
                self._set_leg1_fixed_rate()
            else:
                return None

        # else new pricing parameters will affect and unpriced split notional
        else:
            if at_init:
                self._split_notional = None
            else:
                dt1, dt2 = self.leg1.periods[0].payment, self.leg1.periods[2].payment
                self._split_notional = self.kwargs["notional"] * curve[dt1] / curve[dt2]
                self._set_leg1_fixed_rate()

    def _set_leg1_fixed_rate(self):
        fixed_rate = (self.leg1.notional - self._split_notional) / (
            -self.leg1.notional * self.leg1.periods[1].dcf
        )
        self.leg1.fixed_rate = fixed_rate * 100

    def __init__(
        self,
        *args,
        pair: str | NoInput = NoInput(0),
        fx_fixings: float | FXRates | FXForwards | NoInput = NoInput(0),
        points: float | NoInput = NoInput(0),
        split_notional: float | NoInput = NoInput(0),
        **kwargs,
    ):
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
        self._set_split_notional(curve=None, at_init=True)
        # self._initialise_fx_fixings(fx_fixings)
        self.points = points

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, value):
        self._unpriced = False
        self._points = value
        self._leg2_fixed_rate = NoInput(0)

        # setting points requires leg1.notional leg1.split_notional, fx_fixing and points value

        if value is not NoInput.blank:
            # leg2 should have been properly set as part of fx_fixings and set_leg2_notional
            fx_fixing = self.leg2.notional / -self.leg1.notional

            _ = self._split_notional * (fx_fixing + value / 10000) + self.leg2.notional
            fixed_rate = _ / (self.leg2.periods[1].dcf * -self.leg2.notional)

            self.leg2_fixed_rate = fixed_rate * 100

        # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International

    # Commercial use of this code, and/or copying and redistribution is prohibited.
    # Contact rateslib at gmail.com if this code is observed outside its intended sphere.

    def _set_pricing_mid(
        self,
        curves: Curve | str | list | NoInput = NoInput(0),
        solver: Solver | NoInput = NoInput(0),
        fx: FXForwards | NoInput = NoInput(0),
    ):
        # This function ASSUMES that the instrument is unpriced, i.e. all of
        # split_notional, fx_fixing and points have been initialised as None.

        # first we set the split notional which is defined by interest rates on leg1.
        points = self.rate(curves, solver, fx)
        self.points = float(points)
        self._unpriced = True  # setting pricing mid does not define a priced instrument

    def rate(
        self,
        curves: Curve | str | list | NoInput = NoInput(0),
        solver: Solver | NoInput = NoInput(0),
        fx: FXForwards | NoInput = NoInput(0),
        fixed_rate: bool = False,
    ):
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
        curves, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
            self.curves,
            solver,
            curves,
            fx,
            NoInput(0),
            self.leg1.currency,
        )
        # set the split notional from the curve if not available
        self._set_split_notional(curve=curves[1])
        # then we will set the fx_fixing and leg2 initial notional.

        # self._set_fx_fixings(fx) # this will be done by super().rate()
        leg2_fixed_rate = super().rate(curves, solver, fx_, leg=2)

        if fixed_rate:
            return leg2_fixed_rate
        else:
            points = -self.leg2.notional * (
                (1 + leg2_fixed_rate * self.leg2.periods[1].dcf / 100) / self._split_notional
                - 1 / self.kwargs["notional"]
            )
            return points * 10000

    def cashflows(
        self,
        curves: Curve | str | list | NoInput = NoInput(0),
        solver: Solver | NoInput = NoInput(0),
        fx: FXForwards | NoInput = NoInput(0),
        base: str | NoInput = NoInput(0),
    ):
        if self._is_unpriced:
            self._set_pricing_mid(curves, solver, fx)
        ret = super().cashflows(curves, solver, fx, base)
        return ret
