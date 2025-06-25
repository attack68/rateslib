from __future__ import annotations

from abc import ABCMeta
from typing import TYPE_CHECKING

from pandas import Series

from rateslib import defaults
from rateslib.default import NoInput, _drb
from rateslib.dual import Dual, Dual2, Variable
from rateslib.fx import FXForwards
from rateslib.legs.base import BaseLeg, _FixedLegMixin, _FloatLegMixin
from rateslib.periods import Cashflow
from rateslib.periods.utils import _get_fx_fixings_from_non_fx_forwards, _validate_float_args

if TYPE_CHECKING:
    from pandas import DataFrame

    from rateslib.typing import (
        FX_,
        Any,
        Curve,
        Curve_,
        CurveOption_,
        DualTypes,
        DualTypes_,
        FixingsFx_,
        FixingsRates_,
        Schedule,
        datetime_,
        int_,
        str_,
    )


class BaseLegMtm(BaseLeg, metaclass=ABCMeta):
    """
    Abstract base class with common parameters for all ``LegMtm`` subclasses.

    Parameters
    ----------
    args : dict
        Required positional args to :class:`BaseLeg`.
    fx_fixings : float, Dual, Dual2 or list or Series of such
        Define the known FX fixings for each period which affects the mark-the-market
        (MTM) notional exchanges after each period. If not given, or only some
        FX fixings are given, the remaining unknown fixings will be forecast
        by a provided :class:`~rateslib.fx.FXForwards` object later. If a Series must be indexed
        by the date of the notional exchange considering ``payment_lag_exchange``.
    alt_currency : str
        The alternative reference currency against which FX fixings are measured
        for MTM notional exchanges (3-digit code).
    alt_notional : float
        The notional expressed in the alternative currency which will be used to
        determine the notional for this leg using the ``fx_fixings`` as FX rates.
    kwargs : dict
        Required keyword args to :class:`BaseLeg`.

    See Also
    --------
    FixedLegExchangeMtm: Create a fixed leg with notional and Mtm exchanges.
    FloatLegExchangeMtm : Create a floating leg with notional and Mtm exchanges.
    """

    _do_not_repeat_set_periods: bool = False
    _is_mtm: bool = True
    _delay_set_periods: bool = True

    def __init__(
        self,
        *args: Any,
        fx_fixings: NoInput  # type: ignore[type-var]
        | DualTypes
        | list[DualTypes]
        | Series[DualTypes]
        | tuple[DualTypes, Series[DualTypes]] = NoInput(0),
        alt_currency: str | NoInput = NoInput(0),
        alt_notional: DualTypes | NoInput = NoInput(0),
        **kwargs: Any,
    ) -> None:
        if isinstance(alt_currency, NoInput):
            raise ValueError("`alt_currency` and `currency` must be supplied for MtmLeg.")
        self.alt_currency: str = alt_currency.lower()
        self.alt_notional: DualTypes = _drb(defaults.notional, alt_notional)
        if "initial_exchange" not in kwargs:
            kwargs["initial_exchange"] = True
        kwargs["final_exchange"] = True
        super().__init__(*args, **kwargs)
        if self.amortization != 0:
            raise ValueError("`amortization` cannot be supplied to a `FixedLegExchangeMtm` type.")

        # calls the fixings setter, will convert the input types to list
        self.fx_fixings = fx_fixings

    @property
    def notional(self) -> DualTypes:
        return self._notional

    @notional.setter
    def notional(self, value: DualTypes) -> None:
        self._notional = value

    def _get_fx_fixings_from_series(
        self,
        ser: Series[DualTypes],  # type: ignore[type-var]
        ini_period: int = 0,
    ) -> list[DualTypes]:
        last_fixing_date = ser.index[-1]
        fixings_list: list[DualTypes] = []
        for i in range(ini_period, self.schedule.n_periods):
            required_date = self.schedule.calendar.lag(
                self.schedule.aschedule[i], self.payment_lag_exchange, True
            )
            if required_date > last_fixing_date:
                break
            else:
                try:
                    fixings_list.append(ser[required_date])
                except KeyError:
                    raise ValueError(
                        "A Series is provided for FX fixings but the required exchange "
                        f"settlement date, {required_date.strftime('%Y-%d-%m')}, is not "
                        f"available within the Series.",
                    )
        return fixings_list

    @property
    def fx_fixings(self) -> list[DualTypes]:
        """
        list : FX fixing values input by user and attached to the instrument.
        """
        return self._fx_fixings

    @fx_fixings.setter
    def fx_fixings(self, value: FixingsFx_) -> None:  # type: ignore[type-var]
        """
        Parse a 'FixingsFx_' object to convert to a list[DualTypes] attached to _fx_fixings attr.
        """
        if isinstance(value, NoInput):
            self._fx_fixings: list[DualTypes] = []
        elif isinstance(value, list):
            self._fx_fixings = value
        elif isinstance(value, float | Dual | Dual2 | Variable):
            self._fx_fixings = [value]
        elif isinstance(value, Series):
            self._fx_fixings = self._get_fx_fixings_from_series(value)  # type: ignore[arg-type]
        elif isinstance(value, tuple):
            self._fx_fixings = [value[0]]
            self._fx_fixings.extend(self._get_fx_fixings_from_series(value[1], ini_period=1))
        else:
            raise TypeError("`fx_fixings` should be scalar value, list or Series of such.")

        # if self._initialised:
        #     self._set_periods(None)

    # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
    # Commercial use of this code, and/or copying and redistribution is prohibited.
    # Contact rateslib at gmail.com if this code is observed outside its intended sphere.

    def _get_fx_fixings(self, fx: FX_) -> list[DualTypes]:
        """
        Return the calculated FX fixings.

        Initialise with the fx fixings already provided statically.
        Use an FXForwards object to determine the additionally required fixings.
        If FXForwards object not available repeat the final given fixing.
        If no fixings are known default to 1.0.

        Parameters
        ----------
        fx : FXForwards, optional
            The object to derive FX fixings that are not otherwise given in
            ``fx_fixings``.
        """
        n_given, n_req = len(self.fx_fixings), self.schedule.n_periods
        fx_fixings_: list[DualTypes] = self.fx_fixings.copy()

        # Only FXForwards can correctly forecast rates. Other inputs may raise or warn.
        if isinstance(fx, FXForwards):
            for i in range(n_given, n_req):
                fx_fixings_.append(
                    fx.rate(
                        self.alt_currency + self.currency,
                        self.schedule.calendar.lag(
                            self.schedule.aschedule[i],
                            self.payment_lag_exchange,
                            True,
                        ),
                    ),
                )
        elif n_req > 0:  # only check if unknown fixings are required
            fx_fixings_ = _get_fx_fixings_from_non_fx_forwards(n_given, n_req, fx_fixings_)
        return fx_fixings_

    def _set_periods(self) -> None:
        raise NotImplementedError("Mtm Legs do not implement this. Look for _set_periods_mtm().")

    def _set_periods_mtm(self, fx: FX_) -> None:
        fx_fixings_: list[DualTypes] = self._get_fx_fixings(fx)
        self.notional = fx_fixings_[0] * self.alt_notional
        notionals = [self.alt_notional * fx_fixings_[i] for i in range(len(fx_fixings_))]

        # initial exchange
        self.periods = (
            [
                Cashflow(
                    -self.notional,
                    self.schedule.calendar.lag(
                        self.schedule.aschedule[0],
                        self.payment_lag_exchange,
                        True,
                    ),
                    self.currency,
                    "Exchange",
                    fx_fixings_[0],
                ),
            ]
            if self.initial_exchange
            else []
        )

        regular_periods = [
            self._regular_period(
                start=period[defaults.headers["a_acc_start"]],
                end=period[defaults.headers["a_acc_end"]],
                payment=period[defaults.headers["payment"]],
                stub=period[defaults.headers["stub_type"]] == "Stub",
                notional=notionals[i],
                iterator=i,
            )
            for i, period in enumerate(self.schedule.table.to_dict(orient="index").values())
        ]
        mtm_flows = [
            Cashflow(
                -notionals[i + 1] + notionals[i],
                self.schedule.calendar.lag(
                    self.schedule.aschedule[i + 1],
                    self.payment_lag_exchange,
                    True,
                ),
                self.currency,
                "Mtm",
                fx_fixings_[i + 1],
            )
            for i in range(len(fx_fixings_) - 1)
        ]
        interleaved_periods = [
            val for pair in zip(regular_periods, mtm_flows, strict=False) for val in pair
        ]
        interleaved_periods.append(regular_periods[-1])
        self.periods.extend(interleaved_periods)

        # final cashflow
        self.periods.append(
            Cashflow(
                notionals[-1],
                self.schedule.calendar.lag(
                    self.schedule.aschedule[-1],
                    self.payment_lag_exchange,
                    True,
                ),
                self.currency,
                "Exchange",
                fx_fixings_[-1],
            ),
        )

    def npv(
        self,
        curve: Curve,
        disc_curve: Curve_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str | NoInput = NoInput(0),
        local: bool = False,
    ) -> DualTypes | dict[str, DualTypes]:
        if not self._do_not_repeat_set_periods:
            self._set_periods_mtm(fx)
        ret = super().npv(curve, disc_curve, fx, base, local)
        # self._is_set_periods_fx = False
        return ret

    def cashflows(
        self,
        curve: Curve_ = NoInput(0),
        disc_curve: Curve_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str | NoInput = NoInput(0),
    ) -> DataFrame:
        if not self._do_not_repeat_set_periods:
            self._set_periods_mtm(fx)
        ret = super().cashflows(curve, disc_curve, fx, base)
        # self._is_set_periods_fx = False
        return ret

    def analytic_delta(
        self,
        curve: Curve_ = NoInput(0),
        disc_curve: Curve_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str | NoInput = NoInput(0),
    ) -> DualTypes:
        if not self._do_not_repeat_set_periods:
            self._set_periods_mtm(fx)
        ret = super().analytic_delta(curve, disc_curve, fx, base)
        # self._is_set_periods_fx = False
        return ret


class FixedLegMtm(_FixedLegMixin, BaseLegMtm):  # type: ignore[misc]
    """
    Create a leg of :class:`~rateslib.periods.FixedPeriod` s and initial, mtm and
    final :class:`~rateslib.periods.Cashflow` s.

    Parameters
    ----------
    args : dict
        Required positional args to :class:`BaseLeg`.
    fixed_rate : float or None
        The fixed rate applied to determine cashflows. Can be set to `None` and
        designated later, perhaps after a mid-market rate for all periods has been
        calculated.
    fx_fixings : float, Dual, Dual2, list of such
        Specify a known initial FX fixing or a list of such for historical legs.
        Fixings that are not specified will be calculated at pricing time with an
        :class:`~rateslib.fx.FXForwards` object.
    alt_currency : str
        The alternative currency against which mark-to-market fixings and payments
        are made. This is considered as the domestic currency in FX fixings.
    alt_notional : float, optional
        The notional of the alternative currency from which to calculate ``notional``
        under the determined ``fx_fixings``. If `None` sets a
        default for ``alt_notional``.
    kwargs : dict
        Required keyword arguments to :class:`BaseLeg`.

    Notes
    -----

    .. warning::

       ``amortization`` is currently **not implemented** for on ``FloatLegExchangeMtm``.

       ``notional`` is **not** used on an ``FloatLegMtm``. It is determined
       from ``alt_notional`` under given ``fx_fixings``.

       ``currency`` and ``alt_currency`` are required in order to determine FX fixings
       from an :class:`~rateslib.fx.FXForwards` object at pricing time.

    Examples
    --------
    For an example see :ref:`Mtm Legs<mtm-legs>`.
    """

    def __init__(
        self,
        *args: Any,
        fixed_rate: DualTypes | NoInput = NoInput(0),
        **kwargs: Any,
    ) -> None:
        self._fixed_rate = fixed_rate
        super().__init__(
            *args,
            **kwargs,
        )

    # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
    # Commercial use of this code, and/or copying and redistribution is prohibited.
    # Contact rateslib at gmail.com if this code is observed outside its intended sphere.


class FloatLegMtm(_FloatLegMixin, BaseLegMtm):
    """
    Create a leg of :class:`~rateslib.periods.FloatPeriod` s and initial, mtm and
    final :class:`~rateslib.periods.Cashflow` s.

    Parameters
    ----------
    args : dict
        Required positional args to :class:`BaseLeg`.
    float_spread : float or None
        The spread applied to determine cashflows. Can be set to `None` and designated
        later, perhaps after a mid-market spread for all periods has been calculated.
    spread_compound_method : str, optional
        The method to use for adding a spread to compounded rates. Available
        options are `{"none_simple", "isda_compounding", "isda_flat_compounding"}`.
    fixings : float or list, optional
        If a float scalar, will be applied as the determined fixing for the **first**
        whole period of the leg. If a list of *n* items, each successive item will be
        passed to the ``fixing`` argument of the first *n* periods of the leg.
        A list within the list is accepted if it contains a set of RFR fixings that
        will be applied to any individual RFR period.
    fixing_method : str, optional
        The method by which floating rates are determined, set by default. See notes.
    method_param : int, optional
        A parameter that is used for the various ``fixing_method`` s. See notes.
    fx_fixings : float, Dual, Dual2, list of such
        Specify a known initial FX fixing or a list of such for historical legs.
        Fixings that are not specified will be calculated at pricing time with an
        :class:`~rateslib.fx.FXForwards` object.
    alt_currency : str
        The alternative currency against which mark-to-market fixings and payments
        are made. This is considered as the domestic currency in FX fixings.
    alt_notional : float, optional
        The notional of the alternative currency from which to calculate ``notional``
        under the determined ``fx_fixings``. If `None` sets a
        default for ``alt_notional``.
    kwargs : dict
        Required keyword arguments to :class:`BaseLeg`.

    Notes
    -----

    .. warning::

       ``amortization`` is currently **not implemented** for on ``FloatLegExchangeMtm``.

       ``notional`` is **not** used on an ``FloatLegMtm``. It is determined
       from ``alt_notional`` under given ``fx_fixings``.

       ``currency`` and ``alt_currency`` are required in order to determine FX fixings
       from an :class:`~rateslib.fx.FXForwards` object at pricing time.

    Examples
    --------
    For an example see :ref:`Mtm Legs<mtm-legs>`.
    """

    schedule: Schedule

    def __init__(
        self,
        *args: Any,
        float_spread: DualTypes_ = NoInput(0),
        fixings: FixingsRates_ = NoInput(0),  # type: ignore[type-var]
        fixing_method: str_ = NoInput(0),
        method_param: int_ = NoInput(0),
        spread_compound_method: str_ = NoInput(0),
        **kwargs: Any,
    ) -> None:
        self._float_spread = float_spread
        (
            self.fixing_method,
            self.method_param,
            self.spread_compound_method,
        ) = _validate_float_args(fixing_method, method_param, spread_compound_method)

        super().__init__(
            *args,
            **kwargs,
        )

        self._set_fixings(fixings)
        self.fx_fixings = self.fx_fixings  # sets fx_fixings and periods after initialising

    def fixings_table(
        self,
        curve: CurveOption_,
        disc_curve: Curve_ = NoInput(0),
        fx: FX_ = NoInput(0),
        base: str_ = NoInput(0),
        approximate: bool = False,
        right: datetime_ = NoInput(0),
    ) -> DataFrame:
        """
        Return a DataFrame of fixing exposures on a :class:`~rateslib.legs.FloatLegMtm`.

        For arguments see
        :meth:`FloatLeg.fixings_table()<rateslib.legs.FloatLeg.fixings_table>`.
        """
        if not self._do_not_repeat_set_periods:
            self._set_periods_mtm(fx)
        return super()._fixings_table(
            curve=curve, disc_curve=disc_curve, approximate=approximate, right=right
        )
